from flask import Flask, render_template, request
from dotenv import load_dotenv
import os

from pinecone import Pinecone
from langchain_pinecone import PineconeVectorStore
from langchain_groq import ChatGroq

from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser

from src.helper import get_embeddings
from src.prompt import system_prompt

load_dotenv()

# ---------------------- ENV KEYS ----------------------
PINECONE_API_KEY = os.getenv("PINECONE_API_KEY")
GROQ_API_KEY = os.getenv("GROQ_API_KEY")
HF_TOKEN = os.getenv("HF_TOKEN")

# ---------------------- PINECONE ----------------------
pc = Pinecone(api_key=PINECONE_API_KEY)
index_name = "medicalbot"

embeddings = get_embeddings()

vectorstore = PineconeVectorStore.from_existing_index(
    index_name=index_name,
    embedding=embeddings
)

retriever = vectorstore.as_retriever(search_kwargs={"k": 3})

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
