import os
from app.utils import load_env_variables
from langchain.chains import RetrievalQA
from langchain.llms import HuggingFaceHub
from langchain.vectorstores import FAISS
from langchain.embeddings import HuggingFaceEmbeddings

# Load env vars and token
load_env_variables()
os.environ["HUGGINGFACEHUB_API_TOKEN"] = os.getenv("HUGGINGFACEHUB_API_TOKEN")

# Initialize components
embeddings = HuggingFaceEmbeddings()
vectorstore_path = "faiss_store"
vectorstore = None
qa_chain = None

if os.path.exists(os.path.join(vectorstore_path, "index.faiss")):
    vectorstore = FAISS.load_local(vectorstore_path, embeddings)
    retriever = vectorstore.as_retriever()
    llm = HuggingFaceHub(
    repo_id="google/flan-t5-xl",
    model_kwargs={"task": "text2text-generation"}
)
    qa_chain = RetrievalQA.from_chain_type(llm=llm, retriever=retriever)

def answer_question(query):
    if qa_chain is None:
        return "No documents embedded yet. Please upload and process a PDF first."
    return qa_chain.run(query)
