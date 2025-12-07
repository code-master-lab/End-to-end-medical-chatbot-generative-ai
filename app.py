# -------------------------------------------------------------
# app.py  --  Flask backend for Medical Chatbot (Render-Optimized)
# -------------------------------------------------------------

from flask import Flask, render_template, request
from dotenv import load_dotenv
import os

# LangChain components
from langchain_groq import ChatGroq
from langchain_pinecone import PineconeVectorStore
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser

# Custom system prompt
from src.prompt import system_prompt

# Lazy embedding loader
from src.helper import get_embeddings

# Pinecone client
from pinecone import Pinecone

# -------------------------------------------------------------
# Load environment variables
# -------------------------------------------------------------
load_dotenv()
PINECONE_API_KEY = os.getenv("PINECONE_API_KEY")
GROQ_API_KEY = os.getenv("GROQ_API_KEY")

# -------------------------------------------------------------
# Initialize Pinecone (v3 client)
# -------------------------------------------------------------
pc = Pinecone(api_key=PINECONE_API_KEY)
index_name = "medicalbot"

# -------------------------------------------------------------
# Load Pinecone index (vectorstore)
# Embeddings will be passed later LAZILY
# -------------------------------------------------------------
embeddings = get_embeddings()   # LAZY MODEL LOADING

docsearch = PineconeVectorStore.from_existing_index(
    index_name=index_name,
    embedding=embeddings
)

retriever = docsearch.as_retriever(search_kwargs={"k": 3})

# -------------------------------------------------------------
# Initialize Groq LLM
# -------------------------------------------------------------
llm = ChatGroq(
    api_key=GROQ_API_KEY,
    model="llama-3.1-8b-instant",
    temperature=0.4,
    max_tokens=500
)

# -------------------------------------------------------------
# Build prompt + RAG chain
# -------------------------------------------------------------
prompt = ChatPromptTemplate.from_messages([
    ("system", system_prompt),
    ("human", "{input}")
])

parser = StrOutputParser()
chain = prompt | llm | parser


def rag_pipeline(query):
    """Full RAG pipeline: retrieve → build context → ask LLM."""
    embeddings = get_embeddings()  # LOAD EMBEDDINGS ONLY NOW
    docs = retriever.get_relevant_documents(query)
    context = "\n\n".join([d.page_content for d in docs])

    return chain.invoke({
        "context": context,
        "input": query
    })


# -------------------------------------------------------------
# Flask app
# -------------------------------------------------------------
app = Flask(__name__)


@app.route("/")
def index():
    return render_template("chat.html")


@app.route("/get", methods=["POST"])
def get_bot_response():
    user_msg = request.form["msg"]
    print("User:", user_msg)

    answer = rag_pipeline(user_msg)
    print("Bot:", answer)

    return answer


# -------------------------------------------------------------
# Run the app locally / Render binding
# -------------------------------------------------------------
if __name__ == "__main__":
    port = int(os.environ.get("PORT", 10000))
    app.run(host="0.0.0.0", port=port)
