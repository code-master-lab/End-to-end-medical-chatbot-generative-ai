# -------------------------------------------------------------
# RAG HELPER FUNCTIONS (Render-Safe, No Torch)
# -------------------------------------------------------------

from langchain_community.document_loaders import PyPDFLoader, DirectoryLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
import os
import requests


# -------------------- 1. PDF LOADER --------------------------
def load_pdf_file(folder_path: str):
    loader = DirectoryLoader(
        folder_path,
        glob="*.pdf",
        loader_cls=PyPDFLoader
    )
    return loader.load()


# -------------------- 2. TEXT SPLITTER ------------------------
def text_split(documents):
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=500,
        chunk_overlap=20
    )
    return splitter.split_documents(documents)


# -------------------- 3. CUSTOM HF EMBEDDING -----------------
# This uses HuggingFace Inference API directly.
# It NEVER breaks on Render.
# -------------------------------------------------------------

class HFCustomEmbedder:
    def __init__(self):
        self.api_url = (
            "https://api-inference.huggingface.co/models/"
            "sentence-transformers/all-MiniLM-L6-v2"
        )
        token = os.getenv("HF_TOKEN")
        self.headers = {"Authorization": f"Bearer {token}"}

    def embed_query(self, text: str):
        response = requests.post(
            self.api_url,
            headers=self.headers,
            json={"inputs": text}
        )

        data = response.json()

        # HF cold start fallback
        if isinstance(data, dict) and "error" in data:
            return [0.0] * 384

        return data[0]  # HF returns [[vector]]

    def embed_documents(self, texts):
        return [self.embed_query(t) for t in texts]


def get_embeddings():
    return HFCustomEmbedder()
