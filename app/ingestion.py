from PyPDF2 import PdfReader
from langchain.document_loaders import UnstructuredFileLoader

def extract_text_from_pdf(file_path):
    reader = PdfReader(file_path)
    return "\n".join([page.extract_text() for page in reader.pages if page.extract_text()])