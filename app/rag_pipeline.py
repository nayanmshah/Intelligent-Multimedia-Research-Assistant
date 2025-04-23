# app/rag_pipeline.py

import os
import shutil
from app.utils import load_env_variables
from langchain.chains import RetrievalQA
from langchain.vectorstores import FAISS
from langchain.embeddings import HuggingFaceEmbeddings
from langchain_huggingface import HuggingFacePipeline

# Load .env variables
load_env_variables()
os.environ["HUGGINGFACEHUB_API_TOKEN"] = os.getenv("HUGGINGFACEHUB_API_TOKEN")

# Initialize embeddings
embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/paraphrase-MiniLM-L6-v2")
vectorstore_path = "faiss_store"

def answer_question(query):
    index_file = os.path.join(vectorstore_path, "index.faiss")

    if not os.path.exists(index_file):
        return "No documents embedded yet. Please upload and process a PDF first."

    try:
        vectorstore = FAISS.load_local(vectorstore_path, embeddings, allow_dangerous_deserialization=True)
        if vectorstore.index.ntotal == 0:
            raise ValueError("FAISS index is empty")
        retriever = vectorstore.as_retriever()
    except Exception as e:
        shutil.rmtree(vectorstore_path, ignore_errors=True)
        return f"Vector index failed ({type(e).__name__}: {str(e)}). Please re-upload a document."

    # Load a compact, working model
    llm = HuggingFacePipeline.from_model_id(
        model_id="MBZUAI/LaMini-Flan-T5-783M",
        task="text2text-generation",
        pipeline_kwargs={
            "max_new_tokens": 256,
            "temperature": 0.3,
        }
    )

    qa_chain = RetrievalQA.from_chain_type(llm=llm, retriever=retriever)
    return qa_chain.run(query)