from flask import Flask, render_template, request
from dotenv import load_dotenv
import os

from langchain_groq import ChatGroq
from langchain_pinecone import PineconeVectorStore
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser

from src.prompt import system_prompt
from src.helper import get_embeddings

from pinecone import Pinecone

load_dotenv()
PINECONE_API_KEY = os.getenv("PINECONE_API_KEY")
GROQ_API_KEY = os.getenv("GROQ_API_KEY")

pc = Pinecone(api_key=PINECONE_API_KEY)
index_name = "medicalbot"

embeddings = get_embeddings()

docsearch = PineconeVectorStore.from_existing_index(
    index_name=index_name,
    embedding=embeddings
)

retriever = docsearch.as_retriever(search_kwargs={"k": 3})

llm = ChatGroq(
    api_key=GROQ_API_KEY,
    model="llama-3.1-8b-instant",
    temperature=0.4
)

prompt = ChatPromptTemplate.from_messages([
    ("system", system_prompt),
    ("human", "{input}")
])

parser = StrOutputParser()
chain = prompt | llm | parser

def rag_pipeline(query):
    embeddings = get_embeddings()
    docs = retriever.get_relevant_documents(query)
    context = "\n\n".join([d.page_content for d in docs])
    
    return chain.invoke({
        "context": context,
        "input": query
    })


app = Flask(__name__)

@app.route("/")
def index():
    return render_template("chat.html")

@app.route("/get", methods=["POST"])
def get_bot_response():
    user_input = request.form["msg"]
    answer = rag_pipeline(user_input)
    return answer


if __name__ == "__main__":
    port = int(os.environ.get("PORT", 10000))
    app.run(host="0.0.0.0", port=port)

