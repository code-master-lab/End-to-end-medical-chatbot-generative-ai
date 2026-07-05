# ── IMPORTS ───────────────────────────────────────────────────────────────────

# Standard Library
import logging             # track app activity in terminal
import os                  # access environment variables

# Third-Party
from dotenv import load_dotenv          # load API keys from .env file
from fastapi import FastAPI, Form, Request
from fastapi.responses import HTMLResponse, JSONResponse, RedirectResponse
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates
from langchain_core.output_parsers import StrOutputParser   # parse LLM output to string
from langchain_core.prompts import ChatPromptTemplate       # structure the prompt
from langchain_groq import ChatGroq                         # LLM for generating answers
from langchain_pinecone import PineconeVectorStore          # search medical docs
from pinecone import Pinecone           # vector database
import uvicorn

# Local
from src.helper import get_embeddings   # converts text to vectors
from src.prompt import system_prompt    # defines chatbot behavior



# ── LOGGING SETUP ─────────────────────────────────────────────────────────────
# Logging gives live terminal output while app runs
# Level INFO → shows INFO, WARNING, ERROR (hides DEBUG noise)
# Format → timestamp | level | message

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)s | %(message)s"
)

logger = logging.getLogger(__name__)  # tags logs with filename "app"

logger.info("Imports loaded and logging configured successfully.")
# ↑ First message you see in terminal — confirms app is booting




# ── ENV KEYS ──────────────────────────────────────────────────────────────────
# Load .env file once at the top — makes all keys available via os.getenv()
load_dotenv()

# Read all keys upfront — clean and organized in one place
PINECONE_API_KEY = os.getenv("PINECONE_API_KEY")
GROQ_API_KEY     = os.getenv("GROQ_API_KEY")
HF_TOKEN         = os.getenv("HF_TOKEN")

# Validate all keys at startup so configuration errors are clear
# Crashes immediately with a clear message instead of a confusing error later
for key_name, key_val in {
    "PINECONE_API_KEY": PINECONE_API_KEY,
    "GROQ_API_KEY":     GROQ_API_KEY,
    "HF_TOKEN":         HF_TOKEN
}.items():
    if not key_val:
        raise EnvironmentError(f"Missing required env var: {key_name}")

logger.info("All environment keys loaded successfully.")
# ↑ If you see this in terminal → all 3 keys found, app can proceed safely



# ── PINECONE VECTOR STORE ─────────────────────────────────────────────────────
# Connect to existing Pinecone index that holds medical book embeddings
# Wrap it as a LangChain retriever to fetch relevant chunks per user query

PINECONE_INDEX_NAME = "medicalbot"  # named constant — change index name from one place
TOP_K_RESULTS       = 3             # controls how many chunks are retrieved

pc         = Pinecone(api_key=PINECONE_API_KEY)  # open connection to Pinecone cloud
embeddings = get_embeddings()                     # load HuggingFace embedding model locally

vectorstore = PineconeVectorStore.from_existing_index(
    index_name=PINECONE_INDEX_NAME,  # connect to existing index — data already uploaded
    embedding=embeddings             # same model used during upload — must match
)

retriever = vectorstore.as_retriever(search_kwargs={"k": TOP_K_RESULTS})
# as_retriever() → LangChain can now call this automatically in the pipeline
# k=TOP_K_RESULTS → returns top 3 most relevant medical chunks per query

logger.info(f"Pinecone retriever ready. Index: {PINECONE_INDEX_NAME}, k={TOP_K_RESULTS}")
# confirms: Pinecone connected, index name and chunk count visible in terminal



# ── GROQ LLM ──────────────────────────────────────────────────────────────────
# Groq runs LLaMA at high speed via API — fast and low-cost inference
# temperature controls creativity vs precision (0.0 = strict, 1.0 = creative)
# 0.4 is the sweet spot for medical Q&A — focused but naturally worded

MODEL_NAME  = "llama-3.1-8b-instant"  # named constant — swap model from one place
TEMPERATURE = 0.4                      # named constant — tune behavior from one place

llm = ChatGroq(
    api_key=GROQ_API_KEY,   # validated API key from Block 2
    model=MODEL_NAME,        # which LLaMA model Groq should run
    temperature=TEMPERATURE  # how focused vs creative the answers should be
)



# ── PROMPT TEMPLATE ───────────────────────────────────────────────────────────
# Three explicit slots — no hidden injection, everything visible:
#   slot 1 → system_prompt  : defines chatbot role and behavior
#   slot 2 → {context}      : medical chunks retrieved from Pinecone
#   slot 3 → {input}        : user's actual question

prompt = ChatPromptTemplate.from_messages([
    ("system", system_prompt),                        # role and behavior instructions
    ("system", "Relevant medical context:\n{context}"), # Pinecone chunks injected here
    ("human",  "{input}")                             # user's question
])

parser = StrOutputParser()  # extracts plain text string from LLM response object
chain  = prompt | llm | parser
# pipe operator chains: prompt fills slots → llm generates answer → parser returns string

logger.info(f"LLM chain ready. Model: {MODEL_NAME}, temperature: {TEMPERATURE}")
# confirms: model name and temperature visible in terminal on every startup



# ── RAG PIPELINE ──────────────────────────────────────────────────────────────
# RAG = Retrieval Augmented Generation
# Flow: user query → retrieve medical chunks → inject as context → LLM answers
# This function is the core of the entire chatbot

def rag_pipeline(query: str) -> str:
    """
    Run the full RAG pipeline for a given user query.
    Returns the LLM's answer as a plain string.
    """
    # Guard: reject empty or whitespace-only queries before any API call
    if not query or not query.strip():
        return "Please enter a valid medical question."

    logger.info(f"Query received: {query[:80]}...")
    # logs first 80 chars — enough to identify query without terminal noise

    try:
        # Step 1: search Pinecone for top-k relevant medical chunks
        docs = retriever.invoke(query)
        logger.info(f"Retrieved {len(docs)} chunks from Pinecone.")

        # Step 2: join all chunk texts into one context string for the LLM
        context = "\n\n".join([doc.page_content for doc in docs])

        # Step 3: fill prompt slots and send to Groq LLM via chain
        answer = chain.invoke({
            "context": context,  # retrieved medical knowledge
            "input":   query     # user's original question
        })
        logger.info("Answer generated successfully.")
        return answer

    except Exception as e:
        logger.exception(f"RAG pipeline failed: {e}")
        # real error visible in terminal — user sees clean message
        return "Sorry, something went wrong. Please try again."




# ── FASTAPI APP ───────────────────────────────────────────────────────────────
# FastAPI is the web server. It serves frontend_part as templates and static assets.

app = FastAPI(title="MediBot AI")
app.mount("/static", StaticFiles(directory="frontend_part"), name="static")
templates = Jinja2Templates(directory="frontend_part")


# Route 1: serve the chat UI when browser opens the app
@app.get("/", response_class=HTMLResponse)
def index(request: Request):
    return templates.TemplateResponse(
        request=request,
        name="chat.html",
        context={}
    )


@app.get("/favicon.ico", include_in_schema=False)
def favicon():
    return RedirectResponse(url="/static/favicon.svg")


# Route 2: receive user question and return LLM answer as JSON
# Use POST because chat messages are submitted in the request body
@app.post("/get")
def get_bot_response(msg: str = Form(default="")):
    query = msg.strip()
    logger.info(f"Chat request received: {query[:80]}...")
    answer = rag_pipeline(query)
    return JSONResponse(content={"answer": answer})


# ── ENTRY POINT ───────────────────────────────────────────────────────────────
if __name__ == "__main__":
    port = int(os.environ.get("PORT", 8080))
    host = os.environ.get("HOST", "127.0.0.1")
    # reads PORT from environment — deployment platforms set this automatically
    # falls back to 8080 for local development if PORT not set
    # HOST defaults to 127.0.0.1 so the printed URL is directly openable locally
    # set HOST=0.0.0.0 when deploying or exposing the server to other machines
    logger.info(f"Starting FastAPI server at http://{host}:{port}")
    uvicorn.run(app, host=host, port=port)




