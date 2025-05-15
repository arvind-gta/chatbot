import os
import argparse
import PyPDF2
import pdfplumber
import pytesseract
from PIL import Image
from sklearn.feature_extraction.text import TfidfVectorizer
import faiss
import numpy as np
from rich import print

class RAGDatabase:
    def __init__(self, pdf_folder):
        self.pdf_folder = pdf_folder
        self.text_chunks = []
        self.vector_database = None

    def extract_text_from_pdfs(self):
        for filename in os.listdir(self.pdf_folder):
            if filename.endswith('.pdf'):
                pdf_path = os.path.join(self.pdf_folder, filename)
                text = self.extract_text_with_pdfplumber(pdf_path, filename)

                if text is None or not text.strip():
                    print(f"[yellow]No text detected in {filename} using pdfplumber. Trying OCR...[/yellow]")
                    text = self.extract_text_with_ocr(pdf_path, filename)

                if self.text_chunks:
                    print(f"[green]Text extracted from {filename}.[/green]")
                else:
                    print(f"[red]Failed to extract any text from {filename}.[/red]")

    def extract_text_with_pdfplumber(self, pdf_path, filename):
        text = ""
        with pdfplumber.open(pdf_path) as pdf:
            for page_num, page in enumerate(pdf.pages, 1):
                page_text = page.extract_text() or ""
                if page_text.strip():
                    self.text_chunks.append(f"[PDF: {filename} | Page: {page_num}] {page_text}")
                    text += page_text
        return text

    def extract_text_with_ocr(self, pdf_path, filename):
        text = ""
        with pdfplumber.open(pdf_path) as pdf:
            for page_num, page in enumerate(pdf.pages, 1):
                if page.to_image():
                    image = page.to_image().original
                    page_text = pytesseract.image_to_string(image)
                    if page_text.strip():
                        self.text_chunks.append(f"[PDF: {filename} | Page: {page_num}] {page_text}")
                        text += page_text
        return text

    def create_vector_database(self):
        if not self.text_chunks:
            print("[red]Error: No text extracted from PDFs. Ensure your PDFs contain readable text.[/red]")
            return

        vectorizer = TfidfVectorizer().fit(self.text_chunks)
        vectors = vectorizer.transform(self.text_chunks).toarray().astype('float32')

        self.vector_database = faiss.IndexFlatL2(vectors.shape[1])
        self.vector_database.add(vectors)

    def query_database(self, query, top_k=5):
        if not self.text_chunks:
            print("[red]Error: No text database found. Please extract text and create the database first.[/red]")
            return []

        vectorizer = TfidfVectorizer().fit(self.text_chunks)
        query_vector = vectorizer.transform([query]).toarray().astype('float32')

        distances, indices = self.vector_database.search(query_vector, top_k)
        return [self.text_chunks[i] for i in indices[0]]

# Example usage
if __name__ == '__main__':
    import sys
    if any('ipykernel_launcher' in arg for arg in sys.argv):
        pdf_folder = './pdfs'  # Default for Jupyter
    else:
        parser = argparse.ArgumentParser(description="RAG Database with Semantic Chunking")
        parser.add_argument('--pdf_folder', type=str, default='./pdfs', help='Path to folder containing PDF files')
        args = parser.parse_args()
        pdf_folder = args.pdf_folder

    if not os.path.exists(pdf_folder):
        os.makedirs(pdf_folder)

    db = RAGDatabase(pdf_folder=pdf_folder)
    db.extract_text_from_pdfs()
    db.create_vector_database()

    if not db.text_chunks:
        print("[red]No text was extracted. Please check your PDFs.[/red]")
    else:
        query = "what are the steps for launch lifeboats using falls method?"
        results = db.query_database(query)

        print("\n[green]Top results:[/green]")
        for res in results:
            print(f"[blue]- {res}")
