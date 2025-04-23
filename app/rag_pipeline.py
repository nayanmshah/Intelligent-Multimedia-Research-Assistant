import os
from app.utils import load_env_variables
from langchain.chains import RetrievalQA
from langchain.llms import HuggingFaceHub
from langchain.vectorstores import FAISS
from langchain.embeddings import HuggingFaceEmbeddings
from langchain.schema import Document

# Load env vars and token
load_env_variables()
os.environ["HUGGINGFACEHUB_API_TOKEN"] = os.getenv("HUGGINGFACEHUB_API_TOKEN")

# Initialize components
embeddings = HuggingFaceEmbeddings()
vectorstore_path = "faiss_store"

if os.path.exists(os.path.join(vectorstore_path, "index.faiss")):
    vectorstore = FAISS.load_local(vectorstore_path, embeddings)
else:
    # Initialize an empty FAISS store
    vectorstore = FAISS.from_documents([], embeddings)

retriever = vectorstore.as_retriever()
llm = HuggingFaceHub(repo_id="google/flan-t5-base")
qa_chain = RetrievalQA.from_chain_type(llm=llm, retriever=retriever)

def answer_question(query):
    return qa_chain.run(query)
