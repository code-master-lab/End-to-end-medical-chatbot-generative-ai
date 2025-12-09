# -------------------------------------------------------------
# prompt.py
# -------------------------------------------------------------
# This file contains the system prompt used by the LLM (Groq)
# in our RAG pipeline.
#
# Purpose:
#   • Provide instructions to the model
#   • Define how answers should be structured
#   • Ensure responses stay grounded in retrieved context
#
# IMPORTANT:
# The RAG workflow injects the retrieved text into {context}.
# The model MUST answer using ONLY that information.
# -------------------------------------------------------------


# System prompt for the LLM
system_prompt = """
You are a helpful medical assistant.
Use the retrieved context to answer clearly and safely.
If the answer is not found, say "I don't know".
Answer in short medically-correct sentences.

Context:
{context}
"""


# NOTE:
# The RAG pipeline will format this prompt as:
#
#   final_prompt = system_prompt.format(context=retrieved_chunks)
#
# Then pass it into the LLM before generating the answer.
#
# Example:
#   answer = llm.invoke(final_prompt)
#
# -------------------------------------------------------------
