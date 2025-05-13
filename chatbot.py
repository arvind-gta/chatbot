pip install pymupdf langchain nltk faiss-cpu

import os
import argparse
import PyPDF2
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import faiss
import numpy as np

class RAGDatabase:
    def __init__(self, pdf_folder):
        self.pdf_folder = pdf_folder
        self.text_chunks = []
        self.vector_database = None

    def extract_text_from_pdfs(self):
        """Extract text from all PDFs in the specified folder."""
        for filename in os.listdir(self.pdf_folder):
            if filename.endswith('.pdf'):
                with open(os.path.join(self.pdf_folder, filename), 'rb') as f:
                    pdf_reader = PyPDF2.PdfReader(f)
                    text = ""
                    for page in pdf_reader.pages:
                        text += page.extract_text()
                    self.text_chunks.extend(self.semantic_chunking(text))

    def semantic_chunking(self, text, chunk_size=500):
        """Split text into semantic chunks based on chunk_size words."""
        words = text.split()
        chunks = [' '.join(words[i:i+chunk_size]) for i in range(0, len(words), chunk_size)]
        return chunks

    def create_vector_database(self):
        """Create a FAISS vector database with the text chunks."""
        vectorizer = TfidfVectorizer().fit(self.text_chunks)
        vectors = vectorizer.transform(self.text_chunks).toarray().astype('float32')

        self.vector_database = faiss.IndexFlatL2(vectors.shape[1])
        self.vector_database.add(vectors)

    def query_database(self, query, top_k=5):
        """Query the RAG database and return the most relevant chunks."""
        vectorizer = TfidfVectorizer().fit(self.text_chunks)
        query_vector = vectorizer.transform([query]).toarray().astype('float32')

        distances, indices = self.vector_database.search(query_vector, top_k)
        return [self.text_chunks[i] for i in indices[0]]

# Example usage
if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="RAG Database with Semantic Chunking")
    parser.add_argument('--pdf_folder', type=str, default='./pdfs', help='Path to folder containing PDF files')
    args = parser.parse_args()

    pdf_folder = args.pdf_folder
    if not os.path.exists(pdf_folder):
        os.makedirs(pdf_folder)

    db = RAGDatabase(pdf_folder=pdf_folder)
    db.extract_text_from_pdfs()
    db.create_vector_database()

    query = "what are the steps for launch lifeboats using falls method?"
    results = db.query_database(query)
    print("Top results:")
    for res in results:
        print(res)
