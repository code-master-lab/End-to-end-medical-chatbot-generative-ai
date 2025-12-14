# ------------------------------------------------------------
# STEP 45: Import core Flask modules
# ------------------------------------------------------------
# Flask        → backend web server
# render_template → serve HTML frontend
# request      → receive user input from browser

from flask import Flask, render_template, request


# ------------------------------------------------------------
# STEP 46: Load environment variables securely
# ------------------------------------------------------------
# load_dotenv():
# - Loads secrets from .env file into environment memory
# - Prevents hard-coding API keys

from dotenv import load_dotenv
import os


# ------------------------------------------------------------
# STEP 47: Import AI & Vector Database components
# ------------------------------------------------------------
# Pinecone               → infrastructure client
# PineconeVectorStore    → LangChain vector store wrapper
# ChatGroq               → LLM interface (Groq)
# ChatPromptTemplate     → controls LLM behavior
# StrOutputParser        → cleans LLM output

from pinecone import Pinecone
from langchain_pinecone import PineconeVectorStore
from langchain_groq import ChatGroq

from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser


# ------------------------------------------------------------
# STEP 48: Import project helper modules
# ------------------------------------------------------------
# get_embeddings → returns Hugging Face embedding model
# system_prompt → strict system rules for the LLM

from src.helper import get_embeddings
from src.prompt import system_prompt


# ------------------------------------------------------------
# Load environment variables into runtime
# ------------------------------------------------------------
load_dotenv()


# ------------------------------------------------------------
# STEP 49: Read environment keys
# ------------------------------------------------------------
# These keys authenticate external services

PINECONE_API_KEY = os.getenv("PINECONE_API_KEY")
GROQ_API_KEY = os.getenv("GROQ_API_KEY")
HF_TOKEN = os.getenv("HF_TOKEN")


# ------------------------------------------------------------
# STEP 50: Connect to Pinecone (Vector Database)
# ------------------------------------------------------------
# pc → Pinecone client (connection handle)
# index_name → existing knowledge base name

pc = Pinecone(api_key=PINECONE_API_KEY)
index_name = "medicalbot"


# ------------------------------------------------------------
# STEP 51: Load embedding model
# ------------------------------------------------------------
# This model converts text → numerical vectors
# MUST match the model used during ingestion

embeddings = get_embeddings()


# ------------------------------------------------------------
# STEP 52: Load existing Pinecone index
# ------------------------------------------------------------
# No PDFs, no chunking, no re-embedding
# Just reconnect to stored knowledge

vectorstore = PineconeVectorStore.from_existing_index(
    index_name=index_name,
    embedding=embeddings
)


# ------------------------------------------------------------
# STEP 53: Create retriever
# ------------------------------------------------------------
# Retriever fetches top-k relevant chunks for a query

retriever = vectorstore.as_retriever(search_kwargs={"k": 3})


# ------------------------------------------------------------
# STEP 54: Initialize Groq LLM
# ------------------------------------------------------------
# LLM is responsible ONLY for reasoning and answer generation

llm = ChatGroq(
    api_key=GROQ_API_KEY,
    model="llama-3.1-8b-instant",
    temperature=0.4
)


# ------------------------------------------------------------
# STEP 55: Build prompt + chain
# ------------------------------------------------------------
# prompt → behavior rules + user question
# parser → converts LLM output to clean text
# chain  → final reasoning pipeline

prompt = ChatPromptTemplate.from_messages([
    ("system", system_prompt),
    ("human", "{input}")
])

parser = StrOutputParser()
chain = prompt | llm | parser


# ------------------------------------------------------------
# STEP 56: Full RAG pipeline function
# ------------------------------------------------------------
# Flow:
# 1. Retrieve relevant chunks
# 2. Merge them into context
# 3. Send context + question to LLM
# 4. Return final answer

def rag_pipeline(query):
    docs = retriever.invoke(query)
    context = "\n\n".join([d.page_content for d in docs])

    return chain.invoke({
        "context": context,
        "input": query
    })


# ------------------------------------------------------------
# STEP 57: Initialize Flask application
# ------------------------------------------------------------
app = Flask(__name__)


# ------------------------------------------------------------
# STEP 58: Home route (UI)
# ------------------------------------------------------------
# Serves chat.html frontend

@app.route("/")
def index():
    return render_template("chat.html")


# ------------------------------------------------------------
# STEP 59: Chat API endpoint
# ------------------------------------------------------------
# Receives user message
# Sends it through RAG pipeline
# Returns AI-generated answer

@app.route("/get", methods=["POST"])
def get_bot_response():
    query = request.form["msg"]
    answer = rag_pipeline(query)
    return answer


# ------------------------------------------------------------
# STEP 60: Run Flask app
# ------------------------------------------------------------
# host="0.0.0.0" → accessible externally
# port → dynamic for deployment (Render / Docker / Cloud)

if __name__ == "__main__":
    port = int(os.environ.get("PORT", 10000))
    app.run(host="0.0.0.0", port=port)

