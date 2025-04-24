import os
import shutil
from app.utils import load_env_variables
from langchain.chains import RetrievalQA
from langchain.vectorstores import FAISS
from langchain.embeddings import HuggingFaceEmbeddings
from langchain_huggingface import HuggingFacePipeline

# Load env vars and token
load_env_variables()
os.environ["HUGGINGFACEHUB_API_TOKEN"] = os.getenv("HUGGINGFACEHUB_API_TOKEN")

embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/paraphrase-MiniLM-L6-v2")
vectorstore_path = "faiss_store"

# ✅ Add debug: log top retrieved docs
def log_retrieved_chunks(docs):
    print("\n🔍 Top Retrieved Chunks:")
    for i, d in enumerate(docs):
        print(f"Chunk {i+1} → {d.page_content[:200]}...\n")

def answer_question(query):
    index_file = os.path.join(vectorstore_path, "index.faiss")
    if not os.path.exists(index_file):
        return "No documents embedded yet. Please upload and process a PDF first."

    try:
        vectorstore = FAISS.load_local(vectorstore_path, embeddings, allow_dangerous_deserialization=True)
        if not vectorstore.index.ntotal or vectorstore.index.ntotal == 0:
            raise ValueError("FAISS index is empty")
        retriever = vectorstore.as_retriever()
    except Exception as e:
        shutil.rmtree(vectorstore_path, ignore_errors=True)
        return f"Vector index failed to load properly ({type(e).__name__}: {str(e)}). Please re-upload your document."

    # Retrieve documents
    docs = retriever.get_relevant_documents(query)
    log_retrieved_chunks(docs)

    llm = HuggingFacePipeline.from_model_id(
        model_id="nayanmshah/resume-qa-model",
        task="text2text-generation",
        pipeline_kwargs={
            "max_new_tokens": 100,
            "top_k": 50,
            "temperature": 0.1,
        }
    )
    qa_chain = RetrievalQA.from_chain_type(llm=llm, retriever=retriever)
    return qa_chain.run(query)