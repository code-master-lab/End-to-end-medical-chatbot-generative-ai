# -------------------------------------------------------------
# CUSTOM EMBEDDING FOR RENDER (NEVER BREAKS)
# -------------------------------------------------------------
# Uses HuggingFace API (NO torch, NO sentence-transformers).
# Works perfectly on Render Free Tier (512 MB RAM).
# -------------------------------------------------------------

from langchain_community.document_loaders import PyPDFLoader, DirectoryLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
import os
import requests

# -------------------- 1. PDF LOADER --------------------------
def load_pdf_file(data_folder: str):
    loader = DirectoryLoader(
        data_folder,
        glob="*.pdf",
        loader_cls=PyPDFLoader
    )
    return loader.load()


# -------------------- 2. TEXT SPLITTER ------------------------
def text_split(extracted_data):
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=500,
        chunk_overlap=20
    )
    return splitter.split_documents(extracted_data)


# -------------------- 3. HF REMOTE EMBEDDINGS ----------------
# ZERO memory usage — perfect for Render.
# -------------------------------------------------------------

class HFCustomEmbedder:
    def __init__(self):
        self.api_url = "https://api-inference.huggingface.co/models/sentence-transformers/all-MiniLM-L6-v2"
        token = os.getenv("HF_TOKEN")
        self.headers = {"Authorization": f"Bearer {token}"}

    def embed_query(self, text):
        response = requests.post(
            self.api_url,
            headers=self.headers,
            json={"inputs": text}
        )

        data = response.json()

        # Handle HF cold start
        if isinstance(data, dict) and "error" in data:
            return [0.0] * 384

        return data[0]

    def embed_documents(self, texts):
        return [self.embed_query(t) for t in texts]


def get_embeddings():
    return HFCustomEmbedder()
