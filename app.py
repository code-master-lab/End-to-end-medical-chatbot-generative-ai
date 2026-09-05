# ── IMPORTS ───────────────────────────────────────────────────────────────────

# Standard Library
import asyncio               # lets Pinecone + PageIndex run at the same time (hybrid mode)
import logging                # every status line you see in the terminal comes from this
import os                     # reads values from the .env file

# Third-Party
from dotenv import load_dotenv          # loads .env into the environment
load_dotenv()                           # must run before importing src.helper
from fastapi import FastAPI, Form, Request
from fastapi.responses import HTMLResponse, JSONResponse, RedirectResponse
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates
from langchain_core.output_parsers import StrOutputParser   # turns the LLM's raw reply into plain text
from langchain_core.prompts import ChatPromptTemplate       # fills {context} and {input} into a prompt
from langchain_groq import ChatGroq                         # the LLM that writes the final answer
from langchain_pinecone import PineconeVectorStore          # Engine 1: vector similarity search
from pageindex import PageIndexClient                       # Engine 2: tree-based document reasoning
from pinecone import Pinecone
import uvicorn

# Local — everything this file needs from the rest of the project
from src.helper import get_embeddings, fetch_pinecone_context, fetch_pageindex_context
# ↑ get_embeddings          : builds the embedding model Pinecone needs
# ↑ fetch_pinecone_context  : Engine 1 — fast, vector similarity search
# ↑ fetch_pageindex_context : Engine 2 — slower, reasons across the full document tree
from src.prompt import SYSTEM_PROMPTS
# ↑ one system prompt per mode: "fast", "precise", "hybrid" — see src/prompt.py



# ── LOGGING SETUP ─────────────────────────────────────────────────────────────
# Every line you see in the terminal at runtime exists because of a logger.info()
# or logger.warning() call somewhere in this file or in src/helper.py. Nothing
# appears automatically — if something isn't logged, it won't show up here.

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)s | %(message)s"
)
logger = logging.getLogger(__name__)
logger.info("Imports loaded and logging configured successfully.")
# ↑ first line you should see when the app boots — confirms nothing crashed on import



# ── ENV KEYS ──────────────────────────────────────────────────────────────────
# Loaded once, validated once, at startup — so a missing key fails immediately
# with a clear message instead of crashing confusingly mid-chat later.

PINECONE_API_KEY  = os.getenv("PINECONE_API_KEY")
GROQ_API_KEY      = os.getenv("GROQ_API_KEY")
HF_TOKEN          = os.getenv("HF_TOKEN")
PAGEINDEX_API_KEY = os.getenv("PAGEINDEX_API_KEY")

required_keys = {
    "PINECONE_API_KEY":  PINECONE_API_KEY,
    "GROQ_API_KEY":      GROQ_API_KEY,
    "HF_TOKEN":          HF_TOKEN,
    "PAGEINDEX_API_KEY": PAGEINDEX_API_KEY,
}
for key_name, key_val in required_keys.items():
    if not key_val:
        raise EnvironmentError(f"Missing required env var: {key_name}")

logger.info("All environment keys loaded successfully.")
# ↑ if you see this, all 4 keys were found — safe to continue past this point



# ── PINECONE VECTOR STORE (Engine 1 setup) ────────────────────────────────────
# Connects to the existing Pinecone index holding the medical document's
# embeddings. Built ONCE here at startup, then reused for every single chat
# request — never rebuilt per question.

PINECONE_INDEX_NAME = "medicalbot"
TOP_K_RESULTS       = 3   # how many chunks Pinecone returns per question

pc_client         = Pinecone(api_key=PINECONE_API_KEY)
embeddings = get_embeddings()

vectorstore = PineconeVectorStore.from_existing_index(
    index_name=PINECONE_INDEX_NAME,
    embedding=embeddings,
)
retriever = vectorstore.as_retriever(search_kwargs={"k": TOP_K_RESULTS})

logger.info(f"Pinecone retriever ready. Index: {PINECONE_INDEX_NAME}, k={TOP_K_RESULTS}")



# ── PAGEINDEX CLIENT (Engine 2 setup) ─────────────────────────────────────────
# Mirrors the Pinecone block above exactly, on purpose — same "build once,
# reuse every request" pattern, so both engines are set up the same way.

pi_client = PageIndexClient(api_key=PAGEINDEX_API_KEY)
logger.info("PageIndex client ready.")



# ── GROQ LLM ──────────────────────────────────────────────────────────────────
# The model that actually writes the final answer, regardless of which mode
# supplied the context.

MODEL_NAME  = "openai/gpt-oss-120b"
TEMPERATURE = 0.3  # Optional: lower for medical accuracy

llm = ChatGroq(
    api_key=GROQ_API_KEY,
    model=MODEL_NAME,
    temperature=TEMPERATURE,
)



# ── PROMPT TEMPLATES + CHAINS (one per mode) ──────────────────────────────────
# Why three chains instead of one: each mode hands the LLM a different SHAPE
# of context (short fragments vs. deep reasoned passages vs. both combined),
# so each mode gets its own tailored system prompt from src/prompt.py, telling
# the model what kind of evidence it's actually holding.
#
# All three are built ONCE here at startup — chains["fast"], chains["precise"],
# and chains["hybrid"] are ready and waiting before the first request ever
# arrives. rag_pipeline() below just picks one by name per request.

parser = StrOutputParser()

chains = {}
for mode_name, mode_system_prompt in SYSTEM_PROMPTS.items():
    mode_prompt = ChatPromptTemplate.from_messages([
        ("system", mode_system_prompt),                          # this mode's personality + rules
        ("system", "Relevant medical context:\n{context}"),       # retrieved chunks go here
        ("human", "{input}"),                                     # the user's actual question
    ])
    chains[mode_name] = mode_prompt | llm | parser

logger.info(f"LLM chains ready for modes: {list(chains.keys())}. Model: {MODEL_NAME}, temperature: {TEMPERATURE}")



# ── RAG PIPELINE (fast / precise / hybrid) ────────────────────────────────────
# RAG = Retrieval Augmented Generation.
# Flow: user question → fetch context (one engine, or both) → hand it to the
# matching chain → LLM writes the answer.

async def rag_pipeline(query: str, mode: str = "fast") -> str:
    """
    mode="fast"    (default): Pinecone only. Fast, ~1-2 seconds.
    mode="precise": PageIndex only. Slower (20-90+s) — reasons across the
                    full document tree instead of just matching keywords.
    mode="hybrid":  Pinecone + PageIndex together, run at the same time via
                    asyncio.gather(), then their contexts are merged.
    Each mode answers using its own system prompt (see chains above), matched
    to the kind of context that mode actually produces.
    """
    # Guard: reject empty input before spending any API call on it
    if not query or not query.strip():
        return "Please enter a valid medical question."

    # Guard: an unrecognized mode value should never crash the app —
    # fall back to the safe, fast default instead
    if mode not in chains:
        logger.warning(f"Unrecognized mode '{mode}' — falling back to 'fast'.")
        mode = "fast"

    logger.info(f"Query received: {query[:80]}... | mode={mode}")

    try:
        # Step 1: fetch context — which engine(s) run depends entirely on mode
        if mode == "precise":
            context = await fetch_pageindex_context(query, pi_client)

        elif mode == "hybrid":
            # asyncio.gather() starts both fetches immediately and waits for
            # both — total time is the slower of the two, not the sum of both
            pinecone_context, pageindex_context = await asyncio.gather(
                fetch_pinecone_context(query, retriever),
                fetch_pageindex_context(query, pi_client),
            )
            context = "\n\n".join(c for c in [pinecone_context, pageindex_context] if c)

        else:  # "fast"
            context = await fetch_pinecone_context(query, retriever)

        logger.info(f"Context chars: {len(context)} | mode={mode}")

        # Step 2: hand the context + question to this mode's own chain.
        # .ainvoke() (not .invoke()) keeps this call non-blocking, matching
        # the async function it lives inside.
        answer = await chains[mode].ainvoke({
            "context": context,
            "input":   query,
        })
        logger.info("Answer generated successfully.")
        return answer

    except Exception as error:
        # Real error goes to the terminal via logger.exception(); the user
        # only ever sees a clean, non-technical message.
        logger.exception(f"RAG pipeline failed: {error}")
        return "Sorry, something went wrong. Please try again."



# ── FASTAPI APP ───────────────────────────────────────────────────────────────
# Serves frontend_part/ as both static files (CSS, favicon) and templates
# (chat.html).

app = FastAPI(title="MediBot AI")
app.mount("/static", StaticFiles(directory="frontend_part"), name="static")
templates = Jinja2Templates(directory="frontend_part")


# Route: serves the chat UI when a browser opens the app
@app.get("/", response_class=HTMLResponse)
def index(request: Request):
    return templates.TemplateResponse(request=request, name="chat.html", context={})


# Route: browser's automatic favicon.ico request gets redirected to the real file
@app.get("/favicon.ico", include_in_schema=False)
def favicon():
    return RedirectResponse(url="/static/favicon.svg")


# Route: used by the hosting platform (see render.yaml's healthCheckPath) to
# confirm the app is alive. Deliberately does nothing except respond fast.
@app.get("/health", include_in_schema=False)
def health():
    return {"status": "ok"}


# Route: receives {msg, mode} from chat.html's fetch/AJAX call, runs the full
# RAG pipeline, and returns the answer as JSON.
@app.post("/get")
async def get_bot_response(msg: str = Form(default=""), mode: str = Form(default="fast")):
    query = msg.strip()
    logger.info(f"Chat request received: {query[:80]}... | mode={mode}")
    answer = await rag_pipeline(query, mode=mode)
    return JSONResponse(content={"answer": answer})



# ── ENTRY POINT ───────────────────────────────────────────────────────────────
if __name__ == "__main__":
    port = int(os.environ.get("PORT", 8080))
    host = os.environ.get("HOST", "127.0.0.1")
    # PORT is set automatically by hosting platforms (e.g. Render); 8080 is
    # the local fallback. HOST defaults to 127.0.0.1 so the printed URL is
    # directly clickable when running locally.
    logger.info(f"Starting FastAPI server at http://{host}:{port}")
    uvicorn.run(app, host=host, port=port)
