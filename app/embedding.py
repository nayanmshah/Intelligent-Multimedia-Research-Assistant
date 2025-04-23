# app/embedding.py

from langchain.embeddings import HuggingFaceEmbeddings
from langchain.vectorstores import FAISS
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.schema import Document

def create_vectorstore_from_text(text, store_path="faiss_store"):
    splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=50)
    chunks = splitter.create_documents([text])

    if not chunks:
        raise ValueError("No chunks generated from text. Ensure input is valid.")

    embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/paraphrase-MiniLM-L6-v2")
    vectorstore = FAISS.from_documents(chunks, embeddings)

    # Only save if FAISS index is valid
    if vectorstore.index.ntotal > 0:
        vectorstore.save_local(store_path)
        return True
    else:
        raise RuntimeError("FAISS index is empty. Embedding failed.")