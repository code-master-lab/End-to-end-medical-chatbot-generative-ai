import asyncio
import logging
import os

from dotenv import load_dotenv
from fastapi import FastAPI, Form, Request
from fastapi.responses import HTMLResponse, JSONResponse, RedirectResponse
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate
from langchain_groq import ChatGroq
from langchain_pinecone import PineconeVectorStore
from pageindex import PageIndexClient
from pinecone import Pinecone
import uvicorn

from src.helper import get_embeddings, fetch_pinecone_context, fetch_pageindex_context
from src.prompt import SYSTEM_PROMPTS

logging.basicConfig(level=logging.INFO, format="%(asctime)s | %(levelname)s | %(message)s")
logger = logging.getLogger(__name__)
logger.info("Imports loaded and logging configured successfully.")

load_dotenv()

PINECONE_API_KEY = os.getenv("PINECONE_API_KEY")
GROQ_API_KEY = os.getenv("GROQ_API_KEY")
HF_TOKEN = os.getenv("HF_TOKEN")
PAGEINDEX_API_KEY = os.getenv("PAGEINDEX_API_KEY")

required_keys = {
    "PINECONE_API_KEY": PINECONE_API_KEY,
    "GROQ_API_KEY": GROQ_API_KEY,
    "HF_TOKEN": HF_TOKEN,
    "PAGEINDEX_API_KEY": PAGEINDEX_API_KEY,
}
for key_name, key_val in required_keys.items():
    if not key_val:
        raise EnvironmentError(f"Missing required env var: {key_name}")

logger.info("All environment keys loaded successfully.")

PINECONE_INDEX_NAME = "medicalbot"
TOP_K_RESULTS = 3

pc = Pinecone(api_key=PINECONE_API_KEY)
embeddings = get_embeddings()
vectorstore = PineconeVectorStore.from_existing_index(
    index_name=PINECONE_INDEX_NAME,
    embedding=embeddings,
)
retriever = vectorstore.as_retriever(search_kwargs={"k": TOP_K_RESULTS})
logger.info(f"Pinecone retriever ready. Index: {PINECONE_INDEX_NAME}, k={TOP_K_RESULTS}")

pi_client = PageIndexClient(api_key=PAGEINDEX_API_KEY)
logger.info("PageIndex client ready.")

MODEL_NAME = "llama-3.1-8b-instant"
TEMPERATURE = 0.4
llm = ChatGroq(api_key=GROQ_API_KEY, model=MODEL_NAME, temperature=TEMPERATURE)

parser = StrOutputParser()
chains = {}
for mode_name, mode_system_prompt in SYSTEM_PROMPTS.items():
    mode_prompt = ChatPromptTemplate.from_messages([
        ("system", mode_system_prompt),
        ("system", "Relevant medical context:\n{context}"),
        ("human", "{input}"),
    ])
    chains[mode_name] = mode_prompt | llm | parser

logger.info(f"LLM chains ready for modes: {list(chains.keys())}. Model: {MODEL_NAME}, temperature: {TEMPERATURE}")


async def rag_pipeline(query: str, mode: str = "fast") -> str:
    """Retrieve context using fast, precise, or hybrid mode and generate an answer."""
    if not query or not query.strip():
        return "Please enter a valid medical question."

    if mode not in chains:
        logger.warning(f"Unrecognized mode '{mode}' — falling back to 'fast'.")
        mode = "fast"

    logger.info(f"Query received: {query[:80]}... | mode={mode}")
    try:
        if mode == "precise":
            context = await fetch_pageindex_context(query, pi_client)
        elif mode == "hybrid":
            pinecone_context, pageindex_context = await asyncio.gather(
                fetch_pinecone_context(query, retriever),
                fetch_pageindex_context(query, pi_client),
            )
            context = "\n\n".join(c for c in [pinecone_context, pageindex_context] if c)
        else:
            context = await fetch_pinecone_context(query, retriever)

        logger.info(f"Context chars: {len(context)} | mode={mode}")
        answer = await chains[mode].ainvoke({"context": context, "input": query})
        logger.info("Answer generated successfully.")
        return answer
    except Exception as error:
        logger.exception(f"RAG pipeline failed: {error}")
        return "Sorry, something went wrong. Please try again."


app = FastAPI(title="MediBot AI")
app.mount("/static", StaticFiles(directory="frontend_part"), name="static")
templates = Jinja2Templates(directory="frontend_part")


@app.get("/", response_class=HTMLResponse)
def index(request: Request):
    return templates.TemplateResponse(request=request, name="chat.html", context={})


@app.get("/favicon.ico", include_in_schema=False)
def favicon():
    return RedirectResponse(url="/static/favicon.svg")


@app.post("/get")
async def get_bot_response(msg: str = Form(default=""), mode: str = Form(default="fast")):
    query = msg.strip()
    logger.info(f"Chat request received: {query[:80]}... | mode={mode}")
    answer = await rag_pipeline(query, mode=mode)
    return JSONResponse(content={"answer": answer})


if __name__ == "__main__":
    port = int(os.environ.get("PORT", 8080))
    host = os.environ.get("HOST", "127.0.0.1")
    logger.info(f"Starting FastAPI server at http://{host}:{port}")
    uvicorn.run(app, host=host, port=port)
