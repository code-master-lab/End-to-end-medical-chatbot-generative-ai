# -------------------------------------------------------------
# HELPER.PY — the toolbox
# -------------------------------------------------------------
# This file only DEFINES tools. It never decides when to use them,
# never combines them, never knows about "modes" or the chat flow —
# that decision-making lives entirely in app.py. Think of this file
# as a drawer of separate, ready-to-use tools; app.py is the hand
# that picks which one to use and when.
#
# Two groups of tools live here:
#   1. PDF loading / chunking / embeddings — existed before PageIndex,
#      support the original Pinecone-only engine.
#   2. PageIndex configuration + dual-engine fetchers — everything
#      added later, when the second retrieval engine was introduced.
#      This is where the new part of the project actually begins.
# -------------------------------------------------------------

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
# ↑ reuses the same logging setup app.py configured — that's why messages
#   from this file show up in the same terminal, in the same format



# -------------------------------------------------------------
# PDF LOADING + CHUNKING (original — predates PageIndex entirely)
# -------------------------------------------------------------
# Used once, offline, to prepare the medical PDF before it was uploaded
# to Pinecone. Not called during a live chat request.

def load_pdf_file(data_folder: str):
    loader = DirectoryLoader(data_folder, glob="*.pdf", loader_cls=PyPDFLoader)
    return loader.load()


def text_split(extracted_data):
    splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=20)
    return splitter.split_documents(extracted_data)



# -------------------------------------------------------------
# HUGGINGFACE EMBEDDINGS (original — powers the Pinecone engine only)
# -------------------------------------------------------------
# Turns text into vectors so Pinecone can do similarity search. PageIndex
# never touches this class — PageIndex does its own embedding internally,
# on its own servers.

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
        # HF's API can return either a flat list or a nested one-item list —
        # this handles both shapes so callers always get a flat list back.
        if isinstance(data, list) and data and isinstance(data[0], list):
            return data[0]
        if isinstance(data, list) and data and isinstance(data[0], (int, float)):
            return data
        raise RuntimeError(f"Unexpected HuggingFace embedding response: {data}")

    def embed_documents(self, texts):
        return [self.embed_query(t) for t in texts]


def get_embeddings():
    return HFCustomEmbedder()



# ===============================================================
# PAGEINDEX CONFIGURATION
# ===============================================================
# Everything from this point down is the new part of the project —
# this is where the second retrieval engine begins. app.py's
# rag_pipeline() calls into the functions defined below whenever
# mode is "precise" or "hybrid".
#
# PROJECT_ROOT is computed from this file's own location on disk
# (not from whichever folder you happened to launch python from),
# so DOC_ID_PATH resolves correctly no matter where app.py is run from.

PROJECT_ROOT = Path(__file__).resolve().parent.parent

# The actual committed directory is Data2 (capital D) — confirmed against
# the real filesystem. This exact capitalization matters: Windows ignores
# case, but Linux/Render (where this gets deployed) does not.
DOC_ID_PATH = PROJECT_ROOT / "Data2" / "medical_doc_id.json"

# All three of these are read from .env, with safe defaults if unset —
# see the PAGEINDEX_* keys in your .env file.
RETRIEVAL_POLL_INTERVAL = int(os.getenv("PAGEINDEX_POLL_INTERVAL", "2"))   # seconds between status checks
RETRIEVAL_MAX_WAIT      = int(os.getenv("PAGEINDEX_MAX_WAIT", "90"))       # give up after this long
PAGEINDEX_THINKING      = os.getenv("PAGEINDEX_THINKING", "true").lower() == "true"
# ↑ "thinking" mode makes PageIndex reason more carefully before answering —
#   slower, but far more likely to find the right content (see project history:
#   thinking=False was tried first and returned empty results)


def _load_doc_id() -> str:
    """
    Reads the one piece of information PageIndex needs to know which
    document to search: the doc_id saved by research/Page_indexing.ipynb
    after the PDF was uploaded and processed, one time, months ago.
    This file is never re-uploaded — only this small pointer is read.
    """
    if not DOC_ID_PATH.exists():
        raise FileNotFoundError(f"doc_id file not found at {DOC_ID_PATH}. Run the PageIndex ingestion notebook first.")
    with open(DOC_ID_PATH, "r", encoding="utf-8") as file:
        data = json.load(file)
    return data["doc_id"]


def _extract_relevant_contents(value) -> list:
    """
    PageIndex's retrieved_nodes can nest the actual text at different
    depths depending on the query — sometimes shallow, sometimes buried
    inside child nodes. Rather than assuming one fixed shape, this walks
    the whole structure recursively and pulls out every "relevant_content"
    value it finds, at any depth.
    """
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



# -------------------------------------------------------------
# DUAL-ENGINE CONTEXT FETCHERS
# -------------------------------------------------------------
# These two functions are what app.py's rag_pipeline() actually calls.
# Both are async, both get awaited together via asyncio.gather() when
# mode="hybrid" — that's what lets them run at the same time instead
# of one after another.
#
# Neither function knows about the other, and neither knows what "mode"
# means — that decision belongs entirely to app.py. These just do one
# job each: given a question, return matching text.
# -------------------------------------------------------------

async def fetch_pinecone_context(query: str, retriever) -> str:
    """
    Engine 1 — vector similarity search.

    This is the ORIGINAL retrieval logic — it used to live written
    directly inside rag_pipeline() in app.py (before PageIndex existed,
    there was only one engine, so there was no need for a separate
    function). It was moved here, unchanged, so it could become its own
    named function and run alongside fetch_pageindex_context() below.

    retriever is built once in app.py at startup and passed in here —
    this function never builds its own connection.
    """
    docs = await asyncio.to_thread(retriever.invoke, query)
    # asyncio.to_thread(): retriever.invoke() is a normal blocking call —
    # running it on a background thread stops it from freezing the whole
    # server while it waits for Pinecone's response.
    return "\n\n".join(doc.page_content for doc in docs)


async def fetch_pageindex_context(query: str, pi_client: PageIndexClient, doc_id: Optional[str] = None) -> str:
    """
    Engine 2 — tree-based document reasoning. This entire function is new;
    there was no equivalent before PageIndex was introduced.

    Unlike Pinecone, PageIndex doesn't answer instantly — it works like a
    small job queue: submit a question, then repeatedly check "is it done
    yet?" until it either finishes or times out. The steps below mirror
    that exactly:

      1. Load the doc_id (which document to search)
      2. Confirm PageIndex has this document ready for searching
      3. Submit the question as a retrieval job -> get back a retrieval_id
      4. Poll get_retrieval() every RETRIEVAL_POLL_INTERVAL seconds until
         status is "completed" or "failed", or RETRIEVAL_MAX_WAIT is hit
      5. Pull the actual text out of whatever nodes were found

    pi_client is built once in app.py at startup and passed in here —
    same pattern as fetch_pinecone_context's retriever parameter, so both
    engines are used the same consistent way.

    On any failure, timeout, or "not ready yet" — this returns "" instead
    of raising. That lets rag_pipeline() carry on gracefully (in hybrid
    mode, Pinecone's half of the answer still comes through even if this
    one comes back empty).
    """
    doc_id = doc_id or _load_doc_id()
    logger.info(f"[PageIndex] doc_id={doc_id}")

    # Step 1: is this document actually searchable yet?
    ready = await asyncio.to_thread(pi_client.is_retrieval_ready, doc_id)
    logger.info(f"[PageIndex] is_retrieval_ready={ready}")
    if not ready:
        return ""

    # Step 2: submit the question — this returns almost immediately with a
    # tracking ID; it does NOT wait for the actual reasoning to finish.
    submission = await asyncio.to_thread(pi_client.submit_query, doc_id, query, thinking=PAGEINDEX_THINKING)
    retrieval_id = submission["retrieval_id"]
    logger.info(f"[PageIndex] retrieval_id={retrieval_id} thinking={PAGEINDEX_THINKING}")

    # Step 3: poll until PageIndex is done — this loop is why "precise" and
    # "hybrid" modes can take anywhere from ~20 seconds to RETRIEVAL_MAX_WAIT.
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

        # asyncio.sleep() (not time.sleep()) — pauses only this one function,
        # without freezing the rest of the app while it waits.
        await asyncio.sleep(RETRIEVAL_POLL_INTERVAL)
        waited += RETRIEVAL_POLL_INTERVAL

    # Ran out of patience — degrade gracefully instead of hanging forever.
    logger.warning(f"[PageIndex] retrieval timed out after {RETRIEVAL_MAX_WAIT}s")
    return ""
