# -------------------------------------------------------------
# RAG HELPER FUNCTIONS (Render-Optimized)
# -------------------------------------------------------------
# - Loads PDFs
# - Splits them into chunks
# - Provides custom embedding using HuggingFace API
# - Works on Render Free Tier (No GPU / No Torch)
# -------------------------------------------------------------

from langchain_community.document_loaders import PyPDFLoader, DirectoryLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
import os
import requests


# -------------------------------------------------------------
# 1. Load PDF files from a directory
# -------------------------------------------------------------
def load_pdf_file(data_folder: str):
    loader = DirectoryLoader(
        data_folder,
        glob="*.pdf",
        loader_cls=PyPDFLoader
    )
    return loader.load()


# -------------------------------------------------------------
# 2. Split extracted text into chunks
# -------------------------------------------------------------
def text_split(extracted_data):
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=500,
        chunk_overlap=20
    )
    return splitter.split_documents(extracted_data)


# -------------------------------------------------------------
# 3. Custom HuggingFace Embedding Class (API-based)
# -------------------------------------------------------------
# This avoids all torch / heavy model issues. Works via HF API.
# -------------------------------------------------------------
class HFCustomEmbedder:
    def __init__(self):
        self.api_url = (
            "https://api-inference.huggingface.co/models/"
            "sentence-transformers/all-MiniLM-L6-v2"
        )
        token = os.getenv("HF_TOKEN")
        self.headers = {"Authorization": f"Bearer {token}"}

    def embed_query(self, text):
        response = requests.post(
            self.api_url,
            headers=self.headers,
            json={"inputs": text}
        )
        data = response.json()

        # If cold start or error → return fallback vector
        if isinstance(data, dict) and "error" in data:
            return [0.0] * 384

        return data[0]

    def embed_documents(self, texts):
        return [self.embed_query(t) for t in texts]


def get_embeddings():
    return HFCustomEmbedder()

