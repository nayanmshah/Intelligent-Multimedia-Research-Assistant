"""
# 🧠 Intelligent Multimedia Research Assistant (IMRA)

A smart assistant that ingests documents and returns contextual, AI-generated answers using LangChain + HuggingFace + FAISS + RAG architecture.

## Features
- PDF ingestion
- LLM-based QA (Retrieval Augmented Generation)
- HuggingFace LLM backend
- Optional image generation using diffusion models

## Setup
1. Install dependencies
```bash
pip install -r requirements.txt
```
2. Add Hugging Face token to `.env`
```
HUGGINGFACEHUB_API_TOKEN=your_token_here
```
3. Run the app
```bash
streamlit run ui/app.py
```

## To Do
- Add YouTube transcript parsing
- Add image OCR pipeline
- Improve query refinement
- Deploy on Streamlit Cloud or Hugging Face Spaces
"""