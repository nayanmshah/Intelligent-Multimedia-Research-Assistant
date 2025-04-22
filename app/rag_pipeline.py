from langchain.chains import RetrievalQA
from langchain.llms import HuggingFaceHub
from langchain.vectorstores import FAISS

os.environ["HUGGINGFACEHUB_API_TOKEN"] = os.getenv("HUGGINGFACEHUB_API_TOKEN")

llm = HuggingFaceHub(repo_id="google/flan-t5-base")

vectorstore = FAISS.load_local("faiss_store", HuggingFaceEmbeddings())
retriever = vectorstore.as_retriever()

qa_chain = RetrievalQA.from_chain_type(llm=llm, retriever=retriever)

def answer_question(query):
    return qa_chain.run(query)
