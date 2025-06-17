import pdfplumber
import re
import nltk
import os
import logging
from nltk.corpus import stopwords
from nltk.tokenize import sent_tokenize
from langchain.text_splitter import RecursiveCharacterTextSplitter
from sentence_transformers import SentenceTransformer
from pinecone import Pinecone, ServerlessSpec
from dotenv import load_dotenv

# Setup logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# Load environment variables
load_dotenv()

nltk.download('punkt')
nltk.download('punkt_tab')
nltk.download('stopwords')

# Step 1: Extract text
def extract_text_from_pdf(pdf_path):
    """Extract text from PDF."""
    if not os.path.exists(pdf_path):
        logger.error(f"{pdf_path} does not exist")
        return ""
    text = ""
    try:
        with pdfplumber.open(pdf_path) as pdf:
            for page in pdf.pages:
                text += page.extract_text() or ""
        if not text.strip():
            logger.warning(f"No text extracted from {pdf_path}")
    except Exception as e:
        logger.error(f"Error extracting {pdf_path}: {e}")
    return text

# Step 2: Preprocess text
stop_words = set(stopwords.words('english'))
def preprocess_text(text):
    """Preprocess text by cleaning and tokenizing."""
    text = re.sub(r'\s+', ' ', text)
    text = re.sub(r'[^\w\s]', '', text)
    text = text.lower().strip()
    sentences = sent_tokenize(text)
    cleaned_sentences = [" ".join(word for word in sent.split() if word not in stop_words) for sent in sentences]
    return cleaned_sentences

# Step 3: Chunk text
splitter = RecursiveCharacterTextSplitter(chunk_size=512, chunk_overlap=50)
def chunk_text(sentences):
    """Chunk text into smaller segments."""
    return splitter.split_text(" ".join(sentences))

# Step 4: Generate embeddings
model = SentenceTransformer('all-MiniLM-L6-v2')
def generate_embeddings(chunks):
    """Generate embeddings for text chunks."""
    return model.encode(chunks, convert_to_tensor=False)

# Step 5: Store in Pinecone
# Initialize Pinecone
api_key = os.getenv('PINECONE_API_KEY')
if not api_key:
    raise ValueError("PINECONE_API_KEY environment variable not set")
pc = Pinecone(api_key=api_key)
index_name = "mock-collection"

# Create or connect to index
if index_name not in pc.list_indexes().names():
    logger.info(f"Creating Pinecone index {index_name}")
    pc.create_index(
        name=index_name,
        dimension=384,  # Dimension of all-MiniLM-L6-v2 embeddings
        metric='cosine',
        spec=ServerlessSpec(cloud='aws', region='us-east-1')  # Adjust region as needed
    )
index = pc.Index(index_name)

# Main pipeline
pdf_files = ["dl.pdf","Ml.pdf"]
for pdf in pdf_files:
    logger.info(f"Processing {pdf}")
    text = extract_text_from_pdf(pdf)
    if not text:
        logger.warning(f"No text extracted from {pdf}")
        continue
    sentences = preprocess_text(text)
    chunks = chunk_text(sentences)
    embeddings = generate_embeddings(chunks)
    # Prepare data for Pinecone
    vectors = []
    existing_ids = set(index.fetch([f"{pdf}_{i}" for i in range(len(chunks))]).vectors.keys())
    for i, (chunk, emb) in enumerate(zip(chunks, embeddings)):
        chunk_id = f"{pdf}_{i}"
        if chunk_id not in existing_ids:
            vectors.append({
                'id': chunk_id,
                'values': emb.tolist(),
                'metadata': {'document': pdf, 'chunk_id': i, 'text': chunk}
            })
    if vectors:
        index.upsert(vectors=vectors)
        logger.info(f"Stored {len(vectors)} chunks for {pdf}")
    else:
        logger.info(f"No new chunks to store for {pdf}")
    logger.info(f"Processed and stored {pdf}")

# Query function
def query_vector_db(query_text, top_k=5):
    """Query the Pinecone vector database."""
    logger.info(f"Querying Pinecone with: {query_text}")
    query_embedding = model.encode([query_text])[0].tolist()
    results = index.query(
        vector=query_embedding,
        top_k=top_k,
        include_metadata=True
    )
    logger.info(f"Retrieved {len(results['matches'])} results, not storing query")
    return results

# Interactive query loop
def main():
    print("Enter your query (or type 'exit' to quit):")
    while True:
        query = input("> ")
        if query.lower() == 'exit':
            break
        if not query.strip():
            print("Please enter a valid query.")
            continue
        results = query_vector_db(query)
        if not results['matches']:
            print("No results found.")
        for i, match in enumerate(results['matches']):
            doc = match['metadata']['text']
            metadata = match['metadata']
            score = match['score']
            print(f"Result {i+1}: {doc} (From: {metadata['document']}, Chunk: {metadata['chunk_id']}, Score: {score})")

if __name__ == "__main__":
    main()