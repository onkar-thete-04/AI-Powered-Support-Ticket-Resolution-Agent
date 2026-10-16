# AI-Powered Support Ticket Resolution Agent

---

## Overview

A Retrieval-Augmented Generation (RAG) pipeline built for SaaS customer support automation.

The system accepts an incoming support ticket as a plain text string, searches a vector database of internal knowledge base articles and past resolved tickets for semantically relevant context, and uses a large language model to produce a structured JSON response containing a category label, priority level, a drafted reply grounded strictly in the retrieved context, and the source documents cited.

The output is designed for direct ingestion by downstream ticketing systems such as Jira or Zendesk.

**Domain:** SaaS customer support — billing, authentication, API usage, account management, integrations.

**Stack:** Python · LangChain · ChromaDB · sentence-transformers · FastAPI · Groq (LLM)

---

## Architecture

```

Incoming ticket (string)
        ↓
Embedding model (all-MiniLM-L6-v2) — query vectorisation
        ↓
ChromaDB similarity search — top 3 relevant chunks retrieved
        ↓
format_context() — chunks formatted with source filenames
        ↓
ChatPromptTemplate — context + ticket slotted into system/human structure
        ↓
Groq LLM (openai/gpt-oss-20b) — grounded response generated
        ↓
StrOutputParser() — raw AIMessage extracted to plain string
        ↓
[Milestone 3] Output parser — validated JSON object
        ↓
[Milestone 4] FastAPI endpoint — 200 response returned to caller
```

**Ingestion flow (run once, offline):**

```
knowledge_base/*.txt
        │
        ▼
[ ingest.py ]
  Load → Chunk (500 chars / 50 overlap) → Embed → Store
        │
        ▼
chroma_db/  (persists to disk)
```

---

## How to Run

### Prerequisites

- Python 3.9 or higher
- A free Groq API key — sign up at https://console.groq.com (no credit card required)

### 1. Clone the repository and set up environment

```bash
git clone https://github.com/<your-username>/support-ticket-agent.git
cd support-ticket-agent

python -m venv venv

# Mac / Linux
source venv/bin/activate

# Windows
venv\Scripts\activate

pip install -r requirements.txt
```

### 2. Configure environment variables

Create a `.env` file in the project root:

```
GROQ_API_KEY=your_groq_api_key_here
```

> ⚠️ Never commit your `.env` file. It is listed in `.gitignore`.

### 3. Run the ingestion script

This reads all documents from `knowledge_base/`, chunks them, embeds them using
`all-MiniLM-L6-v2`, and stores the vectors in a local ChromaDB database.

```bash
python ingest.py
```

Expected output:
```
Created 15 documents in 'knowledge_base/'
Loaded 15 raw documents
Split into 20 chunks
Average chunk size: 358.95 characters
Embedding model loaded
Vector store created at 'chroma_db/'
Total vectors stored: 20
```

> The first run downloads the embedding model (~80MB). Subsequent runs use the local cache.
> You only need to re-run `ingest.py` if the source documents in `knowledge_base/` change.

### 4. Start the API server

> 🚧 Coming in Milestone 4

```bash
# uvicorn api:app --reload
```

### 5. Send a test request

> 🚧 Coming in Milestone 4

```bash
# curl -X POST http://localhost:8000/resolve \
#   -H "Content-Type: application/json" \
#   -d '{"ticket": "I cannot log in. It says my account is locked."}'
```

---



**Milestone 1 — Embeddings and Chunking:**
```
Embeddings convert text into vectors — lists of numbers that encode meaning
so that semantically similar text produces numerically similar vectors.
The same embedding model must be used for both ingestion and querying,
because different models produce incompatible coordinate systems — even if
their vector dimensions match, similarity scores become meaningless.
Retrieval quality is the first line of defence against hallucination:
if the wrong chunks come back from ChromaDB, no amount of prompt engineering
can fix a response built on irrelevant context.
```

**Milestone 2 — Retrieval and Prompt Engineering:**

```
Building the retrieval chain taught me that hallucination prevention 
starts in the prompt, not the model. By splitting the prompt into a 
system message (standing rules and constraints) and a human message 
(the ticket and retrieved context), the LLM is given a clear boundary 
between what it is allowed to do and what the current request is. 
The refusal phrase — a fixed, machine-readable string — means a 
downstream system can detect unanswerable tickets and route them 
to a human agent without any additional classification logic

```

**Milestone 3 — Structured Output:**

<!-- To be added -->

**Milestone 4 — API Integration:**

<!-- To be added -->

---

## Trade-offs

### Chunking strategy

The pipeline uses `RecursiveCharacterTextSplitter` with `chunk_size=500` and
`chunk_overlap=50`. This was chosen because the source documents are short-to-medium
SaaS support articles in plain English, where 500 characters captures roughly one
complete idea without blurring the embedding vector across multiple topics.

**Limitations:**
- A purely character-based split does not respect semantic boundaries — a 501-character
  article splits into two chunks even if the second chunk is only one sentence, which
  can make that sentence's vector less meaningful without its surrounding context.
- Larger documents (e.g. a 3,000-character policy PDF) would need a larger `chunk_size`
  or a semantic chunking strategy (splitting on paragraph breaks or headings first).
- There is no evaluation metric in place to measure retrieval quality — the chunk size
  was chosen by reasoning, not by measuring precision/recall on a held-out query set.

### LLM call latency

> 🚧 To be measured and documented after Milestone 2 is complete.

<!-- Discuss: how long does the Groq API call take? What would this mean for a
     production support system handling 500 tickets per hour?
     What caching strategies could reduce repeated LLM calls for similar tickets? -->

---

## Project Structure

```
support-ticket-agent/
├── knowledge_base/          # 15 source .txt documents (KB articles + resolved tickets)
├── chroma_db/               # ChromaDB vector store (auto-generated by ingest.py)
├── ingest.py                # Milestone 1 — load, chunk, embed, store
├── pipeline.py              # Milestone 2 & 3 — retrieval, prompting, structured output
├── api.py                   # Milestone 4 — FastAPI endpoint
├── .env                     # API keys (not committed)
├── requirements.txt         # Pinned dependencies
└── README.md
```

---

## Milestones

- [x] Milestone 1 — Data Ingestion and Embedding
- [x] Milestone 2 — Retrieval and Prompt Engineering
- [ ] Milestone 3 — Enforcing Structured Output
- [ ] Milestone 4 — API Integration

