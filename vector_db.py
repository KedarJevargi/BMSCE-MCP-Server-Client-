import os
import chromadb
from chromadb.utils import embedding_functions
from PyPDF2 import PdfReader

# -----------------------------
# PDF Text Extraction
# -----------------------------
def extract_text_from_pdf(pdf_path: str) -> str:
    reader = PdfReader(pdf_path)
    text = ""
    for page in reader.pages:
        text += page.extract_text() or ""  # handle None pages
    return text

# -----------------------------
# Split text into chunks
# -----------------------------
def split_text(text: str, chunk_size: int = 1000, overlap: int = 100) -> list:
    chunks = []
    start = 0
    while start < len(text):
        end = min(start + chunk_size, len(text))
        chunk = text[start:end].strip()
        if chunk:
            chunks.append(chunk)
        start += chunk_size - overlap
    return chunks

# -----------------------------
# Persistent ChromaDB Setup
# -----------------------------
PERSIST_DIR = "chroma_storage"
os.makedirs(PERSIST_DIR, exist_ok=True)

client = chromadb.PersistentClient(path=PERSIST_DIR)

# Embedding function
ollama_ef = embedding_functions.OllamaEmbeddingFunction(model_name="nomic-embed-text:v1.5")

# Create or load collection
collection = client.get_or_create_collection(
    name="docs",
    embedding_function=ollama_ef
)

# -----------------------------
# Add PDF to VectorDB
# -----------------------------
def add_pdf_to_vectordb(pdf_path: str):
    # Extract text
    pdf_text = extract_text_from_pdf(pdf_path)
    
    # Split into chunks
    chunks = split_text(pdf_text)
    
    # Generate unique IDs to avoid collisions
    base_name = os.path.splitext(os.path.basename(pdf_path))[0]
    ids = [f"{base_name}_chunk_{i}" for i in range(len(chunks))]
    
    # Add to collection
    collection.add(
        documents=chunks,
        ids=ids
    )
    print(f"✅ Added {len(chunks)} chunks from {pdf_path} to collection.")

# -----------------------------
# Example Usage
# -----------------------------
if __name__ == "__main__":
    pdf_files = ["sem.pdf"]
    for pdf in pdf_files:
        add_pdf_to_vectordb(pdf)
