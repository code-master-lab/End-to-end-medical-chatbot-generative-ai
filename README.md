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


<pre class="project-structure">
project/
├── app.py
├── requirements.txt
├── .env
├── src/
│   ├── helper.py
│   ├── prompt.py
│   └── __init__.py
├── templates/
│   └── chat.html
├── static/
│   └── style.css
└── research/
    └── trials.ipynb
</pre>

<!-- ✅ OUTSIDE THE PRE -->

<div class="doc-box">
  <h3>📘 Read Full Project Documentation</h3>
  <p>
    For a deep engineering-level explanation of architecture,
    failures, fixes, and final RAG design, visit the documentation below.
  </p>

  <a href="https://code-master-lab.github.io/documentation-file-end-to-end-medical-chatbot/"
     target="_blank" class="doc-button">
    Open Documentation
  </a>
</div>

<!-- ✅ CSS MUST BE INSIDE STYLE TAG -->
<style>
.project-structure {
  background: #0f172a;
  color: #e5e7eb;
  padding: 20px;
  border-radius: 8px;
  overflow-x: auto;
}

.doc-box {
  margin-top: 40px;
  padding: 25px;
  background: #111827;
  border-left: 4px solid #2563eb;
  border-radius: 8px;
}

.doc-button {
  display: inline-block;
  margin-top: 12px;
  padding: 10px 18px;
  background: #2563eb;
  color: white;
  text-decoration: none;
  border-radius: 6px;
  font-weight: 600;
}
</style>





