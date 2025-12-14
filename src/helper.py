
# -------------------------------------------------------------
# helper.py
# -------------------------------------------------------------
# RAG HELPER FUNCTIONS (Render-Optimized)
#
# PURPOSE OF THIS FILE:
# - Load medical PDF documents
# - Split extracted text into manageable chunks
# - Generate embeddings using Hugging Face Inference API
#
# WHY THIS DESIGN:
# - Render Free Tier has no GPU and limited RAM
# - Local embedding models (torch) often fail
# - Hugging Face API avoids heavy local computation
#
# THIS FILE HANDLES:
# - DATA INGESTION
# - TEXT PREPROCESSING
# - EMBEDDING GENERATION
# -------------------------------------------------------------

import os
import requests

from langchain_community.document_loaders import PyPDFLoader, DirectoryLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter


# -------------------------------------------------------------
# 1. Load PDF files from a directory
# -------------------------------------------------------------
# This function:
# - Scans the given folder
# - Finds all PDF files
# - Extracts text page by page
# - Returns LangChain Document objects
#
# NOTE:
# - This is RAW data extraction
# - No chunking
# - No embeddings
# - No vector database interaction

def load_pdf_file(data_folder: str):
    loader = DirectoryLoader(
        data_folder,
        glob="*.pdf",          # Load only PDF files
        loader_cls=PyPDFLoader
    )
    return loader.load()


# -------------------------------------------------------------
# 2. Split extracted text into chunks
# -------------------------------------------------------------
# WHY THIS STEP IS REQUIRED:
# - LLMs have context limits
# - Vector databases work best with small chunks
#
# WHAT THIS DOES:
# - Takes raw documents
# - Splits them into overlapping chunks
# - Preserves meaning across chunk boundaries

def text_split(extracted_data):
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=500,        # Safe size for embeddings + LLM
        chunk_overlap=20       # Prevents context loss
    )
    return splitter.split_documents(extracted_data)


# -------------------------------------------------------------
# 3. Custom Hugging Face Embedding Class (API-based)
# -------------------------------------------------------------
# IMPORTANT:
# - This avoids torch, transformers, and local models
# - Embeddings are generated via Hugging Face API
# - Works reliably on Render Free Tier
#
# LangChain Compatibility:
# - embed_query(text)      → single text
# - embed_documents(texts) → list of texts

class HFCustomEmbedder:
    def __init__(self):
        self.api_url = (
            "https://api-inference.huggingface.co/models/"
            "sentence-transformers/all-MiniLM-L6-v2"
        )

        # Hugging Face API token (must be set in .env)
        token = os.getenv("HF_TOKEN")

        self.headers = {
            "Authorization": f"Bearer {token}"
        }


    # ---------------------------------------------------------
    # Generate embedding for a single query
    # ---------------------------------------------------------
    def embed_query(self, text: str):
        response = requests.post(
            self.api_url,
            headers=self.headers,
            json={"inputs": text}
        )

        data = response.json()

        # Handle cold start or API error
        # Always return a fixed-size vector (384)
        if isinstance(data, dict) and "error" in data:
            return [0.0] * 384

        # Hugging Face returns: [[vector]]
        return data[0]


    # ---------------------------------------------------------
    # Generate embeddings for multiple documents
    # ---------------------------------------------------------
    def embed_documents(self, texts):
        return [self.embed_query(text) for text in texts]


# -------------------------------------------------------------
# 4. Public helper function to get embeddings
# -------------------------------------------------------------
# This function is imported across the project
# It returns a LangChain-compatible embedding object

def get_embeddings():
    return HFCustomEmbedder()
