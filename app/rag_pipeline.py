import os
import shutil
from app.utils import load_env_variables
from langchain.chains import RetrievalQA
from langchain.llms import HuggingFaceHub
from langchain.vectorstores import FAISS
from langchain.embeddings import HuggingFaceEmbeddings

# Load env vars and token
load_env_variables()
os.environ["HUGGINGFACEHUB_API_TOKEN"] = os.getenv("HUGGINGFACEHUB_API_TOKEN")

embeddings = HuggingFaceEmbeddings()
vectorstore_path = "faiss_store"

def answer_question(query):
    index_file = os.path.join(vectorstore_path, "index.faiss")
    if not os.path.exists(index_file):
        return "No documents embedded yet. Please upload and process a PDF first."

    try:
        vectorstore = FAISS.load_local(vectorstore_path, embeddings)
    except RuntimeError:
        shutil.rmtree(vectorstore_path, ignore_errors=True)
        return "Vector index was corrupted. Please re-upload your document."

    retriever = vectorstore.as_retriever()
    llm = HuggingFaceHub(
        repo_id="google/flan-t5-xl",
        model_kwargs={"task": "text2text-generation"}
    )
    qa_chain = RetrievalQA.from_chain_type(llm=llm, retriever=retriever)
    return qa_chain.run(query)