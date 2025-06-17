# PDF Text Extraction and Embedding

## Installation

1. Clone the repository:
   ```
   git clone https://github.com/your-username/pdf-text-extraction.git
   ```
2. Navigate to the project directory:
   ```
   cd pdf-text-extraction
   ```
3. Create a virtual environment and activate it:
   ```
   python -m venv venv
   source venv/bin/activate  # On Windows, use `venv\Scripts\activate`
   ```
4. Install the required dependencies:
   ```
   pip install -r requirements.txt
   ```
5. Set the `PINECONE_API_KEY` environment variable:
   ```
   export PINECONE_API_KEY=your_pinecone_api_key
   ```

## Usage

1. Place your PDF files in the project directory.
2. Run the main script:
   ```
   python app.py
   ```
3. The script will extract text from the PDF files, preprocess the text, generate embeddings, and store the chunks in a Pinecone vector database.
4. To query the vector database, enter your query when prompted:
   ```
   Enter your query (or type 'exit' to quit):
   > your_query_here
   ```
   The script will display the top 5 matching results, including the text, document name, chunk ID, and similarity score.

## API

The main functions in the `app.py` file are:

- `extract_text_from_pdf(pdf_path)`: Extracts text from a PDF file.
- `preprocess_text(text)`: Preprocesses the text by cleaning and tokenizing.
- `chunk_text(sentences)`: Chunks the text into smaller segments.
- `generate_embeddings(chunks)`: Generates embeddings for the text chunks.
- `query_vector_db(query_text, top_k=5)`: Queries the Pinecone vector database and returns the top `top_k` matching results.

## Contributing

1. Fork the repository.
2. Create a new branch for your feature or bug fix.
3. Make your changes and commit them.
4. Push your branch to your forked repository.
5. Submit a pull request to the original repository.

## License

This project is licensed under the [MIT License](LICENSE).

## Testing

To run the tests, execute the following command:

```
python -m unittest discover tests
```

This will run the tests defined in the `tests` directory.
