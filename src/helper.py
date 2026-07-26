import asyncio
import json
import logging
import os
from pathlib import Path
from typing import Optional

from huggingface_hub import InferenceClient
from langchain_community.document_loaders import DirectoryLoader, PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from pageindex import PageIndexClient

logger = logging.getLogger(__name__)


def load_pdf_file(data_folder: str):
    loader = DirectoryLoader(data_folder, glob="*.pdf", loader_cls=PyPDFLoader)
    return loader.load()


def text_split(extracted_data):
    splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=20)
    return splitter.split_documents(extracted_data)


class HFCustomEmbedder:
    def __init__(self):
        self.model = "sentence-transformers/all-MiniLM-L6-v2"
        self.client = InferenceClient(provider="hf-inference", api_key=os.getenv("HF_TOKEN"))

    def embed_query(self, text):
        data = self.client.feature_extraction(text, model=self.model)
        if hasattr(data, "tolist"):
            data = data.tolist()
        if isinstance(data, dict) and "error" in data:
            raise RuntimeError(f"HuggingFace embedding failed: {data['error']}")
        if isinstance(data, list) and data and isinstance(data[0], list):
            return data[0]
        if isinstance(data, list) and data and isinstance(data[0], (int, float)):
            return data
        raise RuntimeError(f"Unexpected HuggingFace embedding response: {data}")

    def embed_documents(self, texts):
        return [self.embed_query(t) for t in texts]


def get_embeddings():
    return HFCustomEmbedder()


PROJECT_ROOT = Path(__file__).resolve().parent.parent
# The actual committed directory is Data2. This exact capitalization is required on Linux/Render.
DOC_ID_PATH = PROJECT_ROOT / "Data2" / "medical_doc_id.json"
RETRIEVAL_POLL_INTERVAL = int(os.getenv("PAGEINDEX_POLL_INTERVAL", "2"))
RETRIEVAL_MAX_WAIT = int(os.getenv("PAGEINDEX_MAX_WAIT", "90"))
PAGEINDEX_THINKING = os.getenv("PAGEINDEX_THINKING", "true").lower() == "true"


def _load_doc_id() -> str:
    if not DOC_ID_PATH.exists():
        raise FileNotFoundError(f"doc_id file not found at {DOC_ID_PATH}. Run the PageIndex ingestion notebook first.")
    with open(DOC_ID_PATH, "r", encoding="utf-8") as file:
        data = json.load(file)
    return data["doc_id"]


def _extract_relevant_contents(value) -> list:
    if isinstance(value, dict):
        chunks = []
        content = value.get("relevant_content")
        if content:
            chunks.append(content)
        for child in value.values():
            if isinstance(child, (list, dict)):
                chunks.extend(_extract_relevant_contents(child))
        return chunks
    if isinstance(value, list):
        chunks = []
        for item in value:
            chunks.extend(_extract_relevant_contents(item))
        return chunks
    return []


async def fetch_pinecone_context(query: str, retriever) -> str:
    docs = await asyncio.to_thread(retriever.invoke, query)
    return "\n\n".join(doc.page_content for doc in docs)


async def fetch_pageindex_context(query: str, pi_client: PageIndexClient, doc_id: Optional[str] = None) -> str:
    doc_id = doc_id or _load_doc_id()
    logger.info(f"[PageIndex] doc_id={doc_id}")
    ready = await asyncio.to_thread(pi_client.is_retrieval_ready, doc_id)
    logger.info(f"[PageIndex] is_retrieval_ready={ready}")
    if not ready:
        return ""

    submission = await asyncio.to_thread(pi_client.submit_query, doc_id, query, thinking=PAGEINDEX_THINKING)
    retrieval_id = submission["retrieval_id"]
    logger.info(f"[PageIndex] retrieval_id={retrieval_id} thinking={PAGEINDEX_THINKING}")

    waited = 0
    while waited < RETRIEVAL_MAX_WAIT:
        result = await asyncio.to_thread(pi_client.get_retrieval, retrieval_id)
        status = result.get("status")
        logger.info(f"[PageIndex] poll status={status} waited={waited}s")
        if status == "completed":
            nodes = result.get("retrieved_nodes", [])
            chunks = _extract_relevant_contents(nodes)
            logger.info(f"[PageIndex] extracted chunk count={len(chunks)}")
            return "\n\n".join(chunks)
        if status == "failed":
            logger.warning(f"[PageIndex] retrieval failed, full result={result}")
            return ""
        await asyncio.sleep(RETRIEVAL_POLL_INTERVAL)
        waited += RETRIEVAL_POLL_INTERVAL

    logger.warning(f"[PageIndex] retrieval timed out after {RETRIEVAL_MAX_WAIT}s")
    return ""
