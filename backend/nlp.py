import nltk
from nltk.tokenize import sent_tokenize

text = "This is a test sentence. This is another one."
sentences = sent_tokenize(text)
print(sentences)