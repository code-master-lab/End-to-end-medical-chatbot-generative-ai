# -------------------------------------------------------------
# RAG HELPER FUNCTIONS
# -------------------------------------------------------------
# These functions handle:
#   • Loading PDF documents
#   • Splitting text into chunks
#   • Creating embeddings
#
# NOTE:
# We use the updated LangChain community loaders and text splitters
# because newer LangChain versions (1.1+) have reorganized imports.
# -------------------------------------------------------------

from langchain_community.document_loaders import PyPDFLoader, DirectoryLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.embeddings import HuggingFaceEmbeddings


# -------------------------------------------------------------
# 1. Load PDF documents from a directory
# -------------------------------------------------------------
# This function reads all PDF files inside the given folder.
# Returns a list of "Document" objects.
#
# Example:
#     extracted_data = load_pdf_file("Data/")
#
# If the PDF file is corrupted or unreadable, this will return []
# which will affect the downstream chunking step.
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
# 2. Split the extracted text into smaller chunks
# -------------------------------------------------------------
# A RAG system performs BEST when text is split into clean,
# overlapping segments. RecursiveCharacterTextSplitter handles:
#   • Beautiful chunk division
#   • Automatic handling of paragraphs
#   • Avoids cutting sentences abruptly
#
# Returns a list of text chunks.
#
# Example:
#     text_chunks = text_split(extracted_data)
#
# If extracted_data is empty (e.g., corrupted PDF),
#     text_chunks will also be empty → []
# -------------------------------------------------------------
def text_split(extracted_data):
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=500,      # each chunk length
        chunk_overlap=20     # overlap improves context retention
    )
    text_chunks = splitter.split_documents(extracted_data)
    return text_chunks


# -------------------------------------------------------------
# 3. Create HuggingFace Embedding model
# -------------------------------------------------------------
# This model converts text chunks into 384-dimensional embeddings.
# We use "all-MiniLM-L6-v2" because:
#   ✔ Fast
#   ✔ Lightweight
#   ✔ Very accurate for semantic search
#   ✔ Works perfectly with Pinecone
#
# Example:
#     embeddings = download_hugging_face_embeddings()
#
# IMPORTANT:
# If you are only *loading* existing Pinecone vectors,
# you still need to initialize this embedding model because
# PineconeVectorStore requires the embedding function.
# -------------------------------------------------------------
def download_hugging_face_embeddings():
    embeddings = HuggingFaceEmbeddings(
        model_name="sentence-transformers/all-MiniLM-L6-v2"
    )
    return embeddings
