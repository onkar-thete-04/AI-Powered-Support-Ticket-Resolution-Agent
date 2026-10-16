import os
import json
import logging
from datetime import datetime
from dotenv import load_dotenv

# ── FASTAPI IMPORTS ────────────────────────────────────────────────────────────
from fastapi import FastAPI, HTTPException
# FastAPI  — the web framework
# HTTPException — used to return structured error responses (e.g. 500, 422)

from pydantic import BaseModel
# BaseModel — used here to define the incoming request shape

# ── PIPELINE IMPORTS ───────────────────────────────────────────────────────────
# Import everything the pipeline built in pipeline.py
# This means the chain, parser, and schema are defined once and reused here
import chromadb
from langchain_chroma import Chroma
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_groq import ChatGroq
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.runnables import RunnablePassthrough
from langchain_core.output_parsers import JsonOutputParser
from pydantic import Field
from typing import List, Literal
from langchain_core.documents import Document

# ── LOAD ENVIRONMENT VARIABLES ────────────────────────────────────────────────
load_dotenv()
GROQ_API_KEY = os.environ.get("GROQ_API_KEY")
if not GROQ_API_KEY:
    raise ValueError("GROQ_API_KEY not found in .env")

# ── LOGGING CONFIGURATION ──────────────────────────────────────────────────────
# Deliverable 5: capture exact prompt sent to LLM and raw response for debugging
os.makedirs("logs", exist_ok=True)   # Create logs/ folder if it doesn't exist

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s — %(levelname)s — %(message)s",
    handlers=[
        logging.FileHandler("logs/pipeline.log"),  # Write to file
        logging.StreamHandler()                    # Also print to terminal
    ]
)
logger = logging.getLogger(__name__)

# ── FASTAPI APP ────────────────────────────────────────────────────────────────
app = FastAPI(
    title="Support Ticket Resolution Agent",
    description="RAG pipeline that categorises tickets and drafts grounded responses",
    version="1.0.0"
)

# ── REQUEST MODEL ──────────────────────────────────────────────────────────────
class TicketRequest(BaseModel):
    ticket: str   # The only required field — the raw support ticket string
    # FastAPI validates this automatically — a request without "ticket" gets 422

# ── PYDANTIC OUTPUT SCHEMA ─────────────────────────────────────────────────────
# Duplicated from pipeline.py so api.py is self-contained
class TicketResolution(BaseModel):
    category: Literal[
        "authentication", "billing", "api",
        "account", "integration", "data", "other"
    ] = Field(description="The category of the support ticket.")
    priority: Literal["low", "medium", "high"] = Field(
        description="Priority level based on ticket urgency."
    )
    response: str = Field(
        description="Drafted response strictly based on retrieved context."
    )
    sources: List[str] = Field(
        description="Source filenames used to construct the response."
    )

print(" FastAPI app initialised")

# ── LOAD ALL RESOURCES AT STARTUP ─────────────────────────────────────────────
# Everything below runs once when uvicorn starts the server
# Subsequent requests reuse these objects from memory — no re-loading

CHROMA_DIR = "chroma_db"
COLLECTION_NAME = "knowledge_base"   # Must match the name used in ingest.py

# ── EMBEDDING MODEL ────────────────────────────────────────────────────────────
logger.info("Loading embedding model...")
embedding_model = HuggingFaceEmbeddings(
    model_name="all-MiniLM-L6-v2",
    model_kwargs={"device": "cpu"},
    encode_kwargs={"normalize_embeddings": True}
)
logger.info("Embedding model loaded")

# ── CHROMADB RETRIEVER ─────────────────────────────────────────────────────────
logger.info("Connecting to ChromaDB...")
chroma_client = chromadb.PersistentClient(path=CHROMA_DIR)
vector_store = Chroma(
    client=chroma_client,
    collection_name=COLLECTION_NAME,
    embedding_function=embedding_model,
)
retriever = vector_store.as_retriever(
    search_type="similarity",
    search_kwargs={"k": 3},
)
logger.info(f"Retriever connected — collection: {COLLECTION_NAME}")

# ── CONTEXT FORMATTER ─────────────────────────────────────────────────────────
def format_context(docs: List[Document]) -> str:
    """Formats retrieved chunks into a single string with source filenames."""
    formatted = []
    for doc in docs:
        source = doc.metadata.get("source", "unknown")
        formatted.append(f"Source: {source}\n{doc.page_content}")
    return "\n\n---\n\n".join(formatted)

# ── PROMPT TEMPLATE ────────────────────────────────────────────────────────────
parser = JsonOutputParser(pydantic_object=TicketResolution)

prompt_template = ChatPromptTemplate.from_messages([
    ("system", """You are a senior support agent for a SaaS company.
Your job is to help human support agents by drafting a response to an incoming support ticket.

STRICT RULES — follow these without exception:
1. Answer ONLY using the information provided in the CONTEXT section below.
2. Do NOT use any knowledge outside the provided context.
3. Do NOT invent, assume, or guess any company policy, pricing, or process.
4. If the context does not contain enough information to answer the ticket,
   set the response field to exactly:
   I don't have enough information in the knowledge base to answer this ticket.
   And set sources to an empty list.
5. Always populate the sources list with only filenames from the CONTEXT section.
6. Keep the response field professional, concise, and actionable.

{format_instructions}"""),

    ("human", """CONTEXT:
{context}

SUPPORT TICKET:
{question}

Respond with a valid JSON object only. No text before or after the JSON.""")
])

# ── LLM ────────────────────────────────────────────────────────────────────────
llm = ChatGroq(
    model="openai/gpt-oss-20b",
    temperature=0,
    api_key=GROQ_API_KEY
)
logger.info(f"LLM initialised: {llm.model_name}")

# ── CHAIN ──────────────────────────────────────────────────────────────────────
chain = (
    {
        "context": retriever | format_context,
        "question": RunnablePassthrough(),
        "format_instructions": lambda _: parser.get_format_instructions(),
    }
    | prompt_template
    | llm
    | parser
)

logger.info(" Pipeline ready — all resources loaded")

# ── ENDPOINT ───────────────────────────────────────────────────────────────────
@app.post("/resolve", response_model=TicketResolution)
# @app.post        — registers this function as a POST endpoint
# "/resolve"       — the URL path
# response_model   — FastAPI validates the return value against TicketResolution
#                    and returns 422 if any field is missing or wrong type

async def resolve_ticket(request: TicketRequest):
    """
    Accepts a support ticket string and returns a structured JSON resolution.
    Logs the incoming ticket and the raw LLM response for every request.
    """
    logger.info(f"Incoming ticket: {request.ticket}")

    try:
        # ── RETRIEVE CONTEXT FOR LOGGING ──────────────────────────────────────
        # Retrieve the chunks separately first so we can log them
        # This gives us full visibility into what context the LLM received
        retrieved_docs = retriever.invoke(request.ticket)
        context_text = format_context(retrieved_docs)

        # ── BUILD AND LOG THE EXACT PROMPT ────────────────────────────────────
        # Deliverable 5: log the exact prompt sent to the LLM
        format_instructions = parser.get_format_instructions()
        filled_prompt = prompt_template.format_messages(
            context=context_text,
            question=request.ticket,
            format_instructions=format_instructions
        )
        # Log each message in the prompt separately for clarity
        for msg in filled_prompt:
            logger.info(
                f"PROMPT [{msg.type.upper()}]:\n{msg.content[:500]}..."
                # Truncate to 500 chars in logs — full prompt can be very long
            )

        # ── CALL THE CHAIN ─────────────────────────────────────────────────────
        raw_response = llm.invoke(filled_prompt)
        # Log the raw response before parsing
        # Deliverable 5: captures exactly what the LLM returned
        logger.info(f"RAW LLM RESPONSE:\n{raw_response.content}")

        # ── PARSE AND VALIDATE ─────────────────────────────────────────────────
        result = parser.invoke(raw_response)
        # parser validates against TicketResolution schema
        # raises OutputParserException if JSON is malformed

        logger.info(
            f"Resolution — category: {result['category']} | "
            f"priority: {result['priority']} | "
            f"sources: {result['sources']}"
        )

        return result
        # FastAPI serialises this dict to JSON automatically
        # response_model=TicketResolution validates it one final time before sending

    except Exception as e:
        # Log the full error before returning a 500
        # This is the debugging hook — check logs/pipeline.log when something breaks
        logger.error(f"Pipeline error: {str(e)}", exc_info=True)
        raise HTTPException(
            status_code=500,
            detail=f"Pipeline error: {str(e)}"
        )

# ── HEALTH CHECK ENDPOINT ──────────────────────────────────────────────────────
@app.get("/health")
async def health_check():
    """Simple health check — confirms the server is running."""
    return {"status": "ok", "model": "openai/gpt-oss-20b"}

# ── SERVER STARTUP ─────────────────────────────────────────────────────────────
if __name__ == "__main__":
    import uvicorn
    uvicorn.run(
        app,
        host="0.0.0.0",   #` Accept requests from any IP (useful for Docker)`
        port=8000,
        log_level="info"
    )