# How document retrieval works in this project

`src/helper.py` finds information. It does not answer the user directly.
`app.py` gives that found information plus the user question to the Groq LLM,
which writes the final chatbot answer.

## The two search engines

| Mode | Engine | What happens |
| --- | --- | --- |
| `fast` | Pinecone | The question is converted to a vector and Pinecone returns the most similar stored chunks. |
| `precise` | PageIndex | PageIndex searches the uploaded document tree as a job, then returns relevant node text. |
| `hybrid` | Both | Pinecone and PageIndex run at the same time; their text is combined before the LLM answers. |

## Project root

`PROJECT_ROOT = Path(__file__).resolve().parent.parent` means:

```text
project root/                 <- PROJECT_ROOT
  app.py
  Data2/medical_doc_id.json
  src/helper.py               <- __file__
```

Starting at `src/helper.py`, one `parent` goes to `src/`; the next goes to the
project root. This lets the program find `Data2/medical_doc_id.json` even if
you start Python from a different folder.

## What PageIndex sends

Your code does **not** send the PDF on every question. The PDF was uploaded
and indexed earlier. The small JSON file contains its PageIndex `doc_id`.

For each user question, the helper makes these calls:

```python
# 1. Check the already-uploaded document is searchable.
pi_client.is_retrieval_ready(document_id)

# 2. Send document ID + user question + thinking setting.
pi_client.submit_query(document_id, query, thinking=PAGEINDEX_THINKING)

# 3. Repeatedly check the same job using its tracking ID.
pi_client.get_retrieval(retrieval_id)
```

The exact request payload shape is controlled inside the installed PageIndex
Python package. From this project code, we can say precisely that the values
passed are:

* readiness check: `document_id`
* submit query: `document_id`, `query`, and `thinking=True` or `False`
* poll result: `retrieval_id`

The code does not reveal which internal model PageIndex uses. Setting
`PAGEINDEX_THINKING=true` only sends PageIndex the `thinking=True` option.

## PageIndex retrieval job, step by step

```text
user question
    |
    v
submit_query(document_id, question, thinking) -- returns immediately --> retrieval_id
    |
    v
get_retrieval(retrieval_id) --> pending/running? --> wait 2 seconds --> ask again
    |
    +--> failed     --> return empty context
    |
    +--> completed  --> read retrieved_nodes --> extract relevant_content
                                                   |
                                                   v
                                           context text for app.py
```

`status == "completed"` means **PageIndex has finished retrieval**. It does
not mean that the chatbot answer already exists. At that point, the helper
does this:

```python
nodes = result.get("retrieved_nodes", [])
contents = _extract_relevant_contents(nodes)
return "\n\n".join(contents)
```

`retrieved_nodes` can contain dictionaries inside lists inside other
dictionaries. `_extract_relevant_contents` walks through every level and
collects each `relevant_content` text value. That is why it calls itself:
this is recursive extraction, not another database call.

Finally, `app.py` makes the actual answer-generation call:

```python
answer = await chains[mode].ainvoke({
    "context": context,
    "input": query,
})
```

So PageIndex finds context; Groq writes the wording of the answer.

## Pinecone flow and the custom embedder

The `HFCustomEmbedder` uses the Hugging Face model
`sentence-transformers/all-MiniLM-L6-v2`.

```text
text --> Hugging Face embedding model --> vector (a list of numbers)
```

When you initially build a Pinecone index, `embed_documents(texts)` sends each
text chunk through `embed_query` and stores the resulting vector with the
chunk in Pinecone. During a live question, LangChain calls the same embedder
for the question, then Pinecone compares the question vector with stored
vectors and returns the closest chunks.

The visible retrieval call in this project is:

```python
documents = await asyncio.to_thread(retriever.invoke, query)
```

`retriever.invoke` is LangChain's wrapper around the embedding + Pinecone
similarity-search work. `asyncio.to_thread` simply prevents that blocking
network request from freezing the FastAPI server.

## Chunking is not splitting a PDF into smaller PDF files

Current code:

```python
splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=20)
chunks = splitter.split_documents(pdf_pages)
```

It reads PDF text and makes in-memory text chunks of about 500 **characters**
with the last 20 characters repeated in the next chunk. It does not create
new 2,000-page PDF files on disk.

The overlap protects important sentences that happen to cross a chunk
boundary. For example, the end of one chunk may have the beginning of a
medical instruction; the next chunk repeats those final 20 characters so it
has a little context.

For a 200,000-page PDF, do not load all pages and create arbitrary
2,000-page PDFs unless you have a separate storage reason. Instead:

1. Process the source in batches (for example, one file or a limited page
   range at a time) so memory stays bounded.
2. Convert pages to text.
3. Chunk the text, keeping page number, filename, and section as metadata.
4. Embed and upsert each batch into Pinecone, or use PageIndex's separate
   ingestion/indexing workflow.

Chunk size is a text-size choice, not a page-count choice. Choose it by
testing answer quality, retrieval cost, table layout, and the model's context
limit. Medical documents often need special handling for tables, headings, and
references; keeping their page/section metadata makes returned passages
traceable.

## Triple quotes and comments

`"""text"""` is a Python docstring when it is the first statement in a
module, class, or function. `# text` is a comment. Neither changes the
retrieval logic; both are there for humans, IDE help, and documentation.
Python compiles code to bytecode, but docstrings can still be kept at runtime
as the object's `__doc__` value unless Python runs with optimization that
removes them. Comments are not retained in normal bytecode.
