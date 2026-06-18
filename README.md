# End-to-end-medical-chatbot-generative-ai


## How to run?
### STEPS:

Clone the repository
```bash
Project repo: https://github.com/
```
### STEP 01- Create a conda environment after opening the repository

```bash
conda create -n medibot python=3.10 -y


``bash
conda activate medibot
```




### STEP 02- install the requirements

```bash

pip install -r requirements.txt

```

```
pip install -e . Very crucial step Add our identity into the environment
Identity means the folder which gives the name to the project
```



                  ┌───────────────────────────────┐
                  │   1. INPUT DOCUMENT           │
                  │  (PDF / Text / Website)       │
                  └───────────────────────────────┘
                                 │
                                 ▼
                 ┌────────────────────────────────┐
                 │   2. CHUNKING                  │
                 │ Document is broken into        │
                 │ small text pieces (“chunks”)   │
                 └────────────────────────────────┘
                                 │
                    Each chunk is processed
                                 │
                                 ▼
        ┌──────────────────────────────────────────────────┐
        │             3. EMBEDDINGS                        │
        │ Chunk → Vector (list of numbers that represent   │
        │ the *meaning* of the chunk)                      │
        └──────────────────────────────────────────────────┘
                                 │
                                 ▼
     ┌──────────────────────────────────────────────────────┐
     │         4. SEMANTIC SEARCH INDEX                     │
     │ Vector DB builds index for fast “meaning search”     │
     │ (HNSW / IVF / ANN algorithms)                        │
     └──────────────────────────────────────────────────────┘
                                 │
                                 ▼
       ┌─────────────────────────────────────────────────┐
       │          5. KNOWLEDGE BASE                      │
       │ All chunks + embeddings + metadata + index      │
       │ = Your AI-searchable knowledge                  │
       └─────────────────────────────────────────────────┘
                                 │
                                 ▼
             ┌────────────────────────────────────┐
             │   6. USER QUESTION                 │
             │  (User asks a query)               │
             └────────────────────────────────────┘
                                 │
                                 ▼
      ┌───────────────────────────────────────────────────────┐
      │ 7. QUESTION EMBEDDING                                 │
      │ Question → Vector (meaning of the question)           │
      └───────────────────────────────────────────────────────┘
                                 │
                                 ▼
     ┌─────────────────────────────────────────────────────────┐
     │     8. SEMANTIC SEARCH (Vector Similarity Search)       │
     │ Database finds chunks whose meaning is closest          │
     │ to the question vector                                  │
     └─────────────────────────────────────────────────────────┘
                                 │
                                 ▼
         ┌────────────────────────────────────────────┐
         │     9. TOP RELEVANT CHUNKS SELECTED       │
         │ These chunks contain the answer context    │
         └────────────────────────────────────────────┘
                                 │
                                 ▼
     ┌───────────────────────────────────────────────────────────┐
     │           10. LLM ANSWERING (RAG)                          │
     │ LLM reads:                                                 │
     │   • User Question                                          │
     │   • Retrieved Relevant Chunks                              │
     │ Then generates a final, accurate answer using context.     │
     └───────────────────────────────────────────────────────────┘

     
```



project/
│
├── app.py # Flask backend (production-ready)
├── requirements.txt # dependencies for hosting
├── .env # API keys (not pushed to GitHub)
│
├── src/
│ ├── helper.py # embeddings, pinecone loader, RAG pipeline
│ ├── prompt.py # system prompt for LLM
│ └── init.py
│
├── frontend_part/
│ ├── chat.html # frontend UI
│ └── style.css # UI styling
│
└── research/
└── trials.ipynb # testing notebook (not used in hosting
```

<hr>

<div style="
  margin-top: 30px;
  padding: 22px;
  background-color: #111827;
  border-left: 4px solid #2563eb;
  border-radius: 8px;
  color: #e5e7eb;
">

  <h3 style="
    margin-top: 0;
    color: #93c5fd;
  ">
    📘 Read Full Project Documentation
  </h3>

  <p style="
    font-size: 14px;
    line-height: 1.6;
  ">
    For a deep engineering-level explanation of architecture, failures,
    fixes, and final RAG system design, read the complete documentation below.
  </p>

  <a href="https://code-master-lab.github.io/documentation-file-end-to-end-medical-chatbot/"
     target="_blank"
     style="
       display: inline-block;
       margin-top: 12px;
       padding: 10px 18px;
       background-color: #2563eb;
       color: #ffffff;
       text-decoration: none;
       border-radius: 6px;
       font-weight: 600;
     ">
     Open Documentation
  </a>

</div>










