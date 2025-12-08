# -------------------------------------------------------------
# RAG HELPER FUNCTIONS (LOCAL EMBEDDINGS — OPTION 1)
# -------------------------------------------------------------
# Using HuggingFaceEmbeddings locally AND on Render.
# This ensures the SAME embedding vectors used during:
#   • Index creation (Jupyter)
#   • Retrieval (Render)
# -------------------------------------------------------------

from langchain_community.document_loaders import PyPDFLoader, DirectoryLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.embeddings import HuggingFaceEmbeddings


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


# -------------------- 3. LOCAL EMBEDDINGS (MiniLM) ------------
# SAME MODEL used in Jupyter indexing.
# SAME MODEL used in Render retrieval.
# 100% MATCH = PERFECT RAG ACCURACY.
# --------------------------------------------------------------
def get_embeddings():
    return HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")
