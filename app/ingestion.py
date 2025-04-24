from PyPDF2 import PdfReader
from langchain.text_splitter import CharacterTextSplitter
from langchain.embeddings import HuggingFaceEmbeddings
from langchain.vectorstores import FAISS
import os

# Step 1: Extract text from PDF
def extract_text_from_pdf(file_path):
    reader = PdfReader(file_path)
    return "\n".join([page.extract_text() for page in reader.pages if page.extract_text()])

# Step 2: Split text into chunks
def chunk_text(text, chunk_size=1000, chunk_overlap=200):
    text_splitter = CharacterTextSplitter(
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap
    )
    return text_splitter.create_documents([text])

# Step 3: Embed and store in FAISS
def embed_documents(docs, save_path="faiss_store"):
    embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/paraphrase-MiniLM-L6-v2")
    vectorstore = FAISS.from_documents(docs, embeddings)
    vectorstore.save_local(save_path)
    print(f"✅ Saved FAISS index to {save_path}/")

# ✅ Full pipeline
def process_pdf(file_path):
    print("📥 Extracting text...")
    text = extract_text_from_pdf(file_path)

    print("✂️ Chunking...")
    docs = chunk_text(text)

    print(f"🔍 Chunked into {len(docs)} segments.")

    print("🔐 Embedding and storing...")
    embed_documents(docs)

    print("✅ Done!")

# Run this to reprocess your PDF
if __name__ == "__main__":
    process_pdf("YOUR_PDF_FILE.pdf")
