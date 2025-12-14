# -------------------------------------------------------------
# prompt.py
# -------------------------------------------------------------
# This file defines the SYSTEM PROMPT for the LLM (Groq)
# used in the RAG (Retrieval-Augmented Generation) pipeline.
#
# ROLE OF THIS FILE:
# - Control how the LLM behaves
# - Restrict answers to retrieved knowledge only
# - Prevent hallucinations
# - Keep medical answers short, safe, and factual
#
# IMPORTANT ARCHITECTURE NOTE:
# - The retriever fetches relevant medical text
# - That text is injected into {context}
# - The LLM MUST answer using ONLY that context
# -------------------------------------------------------------


# -------------------------------------------------------------
# SYSTEM PROMPT (LLM BEHAVIOR DEFINITION)
# -------------------------------------------------------------
# This prompt is sent as a "system" message to the LLM.
# System messages have higher priority than user messages.
#
# This prompt enforces:
# - Medical assistant role
# - Grounded answers (no guessing)
# - Short and medically correct responses
# - Honest fallback when information is missing

system_prompt = """
You are a helpful medical assistant.

Rules you must follow:
- Use ONLY the information provided in the context below.
- Do NOT add information from outside knowledge.
- If the answer is not present in the context, say: "I don't know".
- Answer in short, medically correct sentences.
- Keep the response clear and safe.

Context:
{context}
"""

# -------------------------------------------------------------
# END OF FILE
