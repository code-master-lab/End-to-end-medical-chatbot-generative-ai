# -------------------------------------------------------------
# RAG HELPER FUNCTIONS (Render-Optimized)
# -------------------------------------------------------------
# These functions are optimized for low-memory environments
# such as Render Free Tier (512 MB RAM).
#
# IMPORTANT:
# HuggingFaceEmbeddings MUST NOT be loaded at import time.
# Render will crash if heavy ML models load during startup.
# -------------------------------------------------------------

from langchain_community.document_loaders import PyPDFLoader, DirectoryLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
import os



# -------------------------------------------------------------
# 1. Load PDF documents from a directory
# -------------------------------------------------------------
def load_pdf_file(data_folder: str):
    loader = DirectoryLoader(
        data_folder,
        glob="*.pdf",
        loader_cls=PyPDFLoader
    )
    documents = loader.load()
    return documents


# -------------------------------------------------------------
# 2. Split PDF text into chunks
# -------------------------------------------------------------
def text_split(extracted_data):
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=500,
        chunk_overlap=20
    )
    text_chunks = splitter.split_documents(extracted_data)
    return text_chunks


# -------------------------------------------------------------
# 3. Lazy-load HuggingFace embeddings
# -------------------------------------------------------------
# DO NOT load embeddings globally.
# DO NOT load them in global scope.
# DO NOT create them at import time.
#
# Render Free Tier will crash if embeddings load during boot.
#
# This function loads the model ONLY when called inside rag_pipeline().
# -------------------------------------------------------------

import os
from langchain_community.embeddings import HuggingFaceHubEmbeddings

def get_embeddings():
    hf_token = os.getenv("HF_TOKEN")
    
    return HuggingFaceHubEmbeddings(
        repo_id="sentence-transformers/all-MiniLM-L6-v2",
        huggingfacehub_api_token=hf_token
    )

