import sys
import os
import streamlit as st

# Add the parent directory to the path so `app` is importable
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from app.ingestion import extract_text_from_pdf
from app.embedding import create_vectorstore_from_text
from app.rag_pipeline import answer_question

st.title("🧠 IMRA - Intelligent Multimedia Research Assistant")

uploaded_file = st.file_uploader("Upload a document", type=["pdf"])
if uploaded_file:
    with open("temp.pdf", "wb") as f:
        f.write(uploaded_file.read())
    text = extract_text_from_pdf("temp.pdf")
    create_vectorstore_from_text(text)
    st.success("Document processed and embedded.")

query = st.text_input("Ask a question about your document:")
if query:
    response = answer_question(query)
    st.write("### 📘 Answer:")
    st.write(response)
