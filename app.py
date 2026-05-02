# ── IMPORTS ───────────────────────────────────────────────────────────────────

# Standard Library
import os                  # access environment variables
import logging             # NEW: track app activity in terminal

# Third-Party
from flask import Flask, render_template, request, jsonify  # jsonify is NEW
from dotenv import load_dotenv          # load API keys from .env file
from pinecone import Pinecone           # vector database
from langchain_pinecone import PineconeVectorStore          # search medical docs
from langchain_groq import ChatGroq                         # LLM for generating answers
from langchain_core.prompts import ChatPromptTemplate       # structure the prompt
from langchain_core.output_parsers import StrOutputParser   # parse LLM output to string

# Local
from src.helper import get_embeddings   # converts text to vectors
from src.prompt import system_prompt    # defines chatbot behavior


# ── LOGGING SETUP ─────────────────────────────────────────────────────────────
# NEW FEATURE: Logging gives live terminal output while app runs
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

# NEW: Validate all keys at startup — fail loudly if anything is missing
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
TOP_K_RESULTS       = 3             # NEW: named constant — controls how many chunks retrieved

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

# ---------------------- GROQ LLM ----------------------
llm = ChatGroq(
    api_key=GROQ_API_KEY,
    model="llama-3.1-8b-instant",
    temperature=0.4
)

# ---------------------- PROMPT + CHAIN ----------------------
prompt = ChatPromptTemplate.from_messages([
    ("system", system_prompt),
    ("human", "{input}")
])

parser = StrOutputParser()
chain = prompt | llm | parser


# ---------------------- RAG PIPELINE ----------------------
def rag_pipeline(query):
    docs = retriever.invoke(query)
    context = "\n\n".join([d.page_content for d in docs])

    return chain.invoke({
        "context": context,
        "input": query
    })


# ---------------------- FLASK APP ----------------------
app = Flask(__name__)

@app.route("/")
def index():
    return render_template("chat.html")

@app.route("/get", methods=["POST"])
def get_bot_response():
    query = request.form["msg"]
    answer = rag_pipeline(query)
    return answer


if __name__ == "__main__":
    port = int(os.environ.get("PORT", 10000))
    app.run(host="0.0.0.0", port=port)
