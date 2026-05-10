# -------------------------------------------------------------
# RAG HELPER FUNCTIONS (Render-Optimized)
# -------------------------------------------------------------
# - Loads PDFs
# - Splits them into chunks
# - Provides custom embedding using HuggingFace API
# - Works on Render Free Tier (No GPU / No Torch)
# -------------------------------------------------------------

import os

from huggingface_hub import InferenceClient
from langchain_community.document_loaders import DirectoryLoader, PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter


def load_pdf_file(data_folder: str):
    loader = DirectoryLoader(
        data_folder,
        glob="*.pdf",
        loader_cls=PyPDFLoader
    )
    return loader.load()


def text_split(extracted_data):
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=500,
        chunk_overlap=20
    )
    return splitter.split_documents(extracted_data)


class HFCustomEmbedder:
    def __init__(self):
        self.model = "sentence-transformers/all-MiniLM-L6-v2"
        self.client = InferenceClient(
            provider="hf-inference",
            api_key=os.getenv("HF_TOKEN")
        )

    def embed_query(self, text):
        data = self.client.feature_extraction(
            text,
            model=self.model
        )

        if hasattr(data, "tolist"):
            data = data.tolist()

        if isinstance(data, dict) and "error" in data:
            raise RuntimeError(f"HuggingFace embedding failed: {data['error']}")

        # HF may return either [0.1, ...] or [[0.1, ...]] depending on API behavior.
        if isinstance(data, list) and data and isinstance(data[0], list):
            return data[0]

        if isinstance(data, list) and data and isinstance(data[0], (int, float)):
            return data

        raise RuntimeError(f"Unexpected HuggingFace embedding response: {data}")

    def embed_documents(self, texts):
        return [self.embed_query(t) for t in texts]


def get_embeddings():
    return HFCustomEmbedder()
