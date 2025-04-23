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
    # Initialize FAISS store with a placeholder document to avoid runtime error
    dummy_doc = Document(page_content="This is a placeholder document.", metadata={})
    vectorstore = FAISS.from_documents([dummy_doc], embeddings)

retriever = vectorstore.as_retriever()
llm = HuggingFaceHub(repo_id="google/flan-t5-small")
qa_chain = RetrievalQA.from_chain_type(llm=llm, retriever=retriever)

def answer_question(query):
    return qa_chain.run(query)
