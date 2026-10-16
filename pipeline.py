import os
from dotenv import load_dotenv
# load_dotenv reads the .env file and injects its values into os.environ
# so GROQ_API_KEY becomes available as os.environ["GROQ_API_KEY"]

import json
import chromadb
from langchain_chroma import Chroma
from langchain_huggingface import HuggingFaceEmbeddings
import pydantic

from pydantic import BaseModel, Field
from typing import List, Literal
from langchain_core.output_parsers import JsonOutputParser

# ── THE JSON SCHEMA ────────────────────────────────────────────────────────────
class TicketResolution(BaseModel):
    """
    This class is the contract for every response the pipeline produces.
    Pydantic enforces that all fields are present and correctly typed.
    A downstream system (Jira, Zendesk) can rely on this structure unconditionally.
    """

    category: Literal[
        "authentication",
        "billing",
        "api",
        "account",
        "integration",
        "data",
        "other"
    ] = Field(
        description="The category of the support ticket. Choose the single most relevant category."
    )
    # Literal restricts the LLM to exactly these values
    # Without it, the LLM might output "login" instead of "authentication"
    # breaking any downstream logic that switches on category

    priority: Literal["low", "medium", "high"] = Field(
        description=(
            "Priority level. Use 'high' for account access issues or billing disputes, "
            "'medium' for general how-to questions, 'low' for informational queries."
        )
    )
    # Inline guidance in the Field description helps the LLM
    # make consistent priority decisions without extra prompt text

    response: str = Field(
        description=(
            "The full drafted response to send to the customer. "
            "Must be based strictly on the retrieved context. "
            "If no relevant context was found, use exactly: "
            "'I don't have enough information in the knowledge base to answer this ticket.'"
        )
    )
    # The refusal phrase is now enforced at the schema level too —
    # not just in the system message

    sources: List[str] = Field(
        description=(
            "List of source document filenames used to construct the response. "
            "Use only filenames that appear in the CONTEXT section. "
            "If no relevant context was found, return an empty list."
        )
    )
    # Empty list for out-of-scope tickets is machine-readable
    # A downstream system can check len(sources) == 0 to flag for human review

# ── INITIALISE THE PARSER ──────────────────────────────────────────────────────
parser = JsonOutputParser(pydantic_object=TicketResolution)
# Passing the Pydantic class tells the parser exactly what to validate against
# It also generates format instructions that we inject into the prompt in Cell 2

print(" Output schema defined")
print(f"   Fields: {list(TicketResolution.model_fields.keys())}")


# ── STEP 1: LOAD ENVIRONMENT VARIABLES ────────────────────────────────────────
load_dotenv()
# Always call this before any os.environ.get() — if you call get() first,
# the key won't be found and you'll get a silent None instead of an error

GROQ_API_KEY = os.environ.get("GROQ_API_KEY")
if not GROQ_API_KEY:
    raise ValueError("GROQ_API_KEY not found in .env — add it before running")
# Fail loudly and immediately rather than letting the error surface
# deep inside the LLM call where it's harder to diagnose

# ── STEP 2: RECONNECT TO CHROMADB ─────────────────────────────────────────────
CHROMA_DIR = "chroma_db"
COLLECTION_NAME = "knowledge_base"

embedding_model = HuggingFaceEmbeddings(
    model_name="all-MiniLM-L6-v2",
    model_kwargs={"device": "cpu"},
    encode_kwargs={"normalize_embeddings": True}
    # Must match the model used in ingest.py exactly
    # Different config = different coordinate system = meaningless scores
)

client = chromadb.PersistentClient(path=CHROMA_DIR)
# Opens the existing database — does NOT re-embed anything

vector_store = Chroma(
    client=client,
    collection_name=COLLECTION_NAME,
    embedding_function=embedding_model,
)

# ── STEP 3: WRAP CHROMADB AS A LANGCHAIN RETRIEVER ────────────────────────────
retriever = vector_store.as_retriever(
    search_type="similarity",   # Use cosine similarity to rank chunks
    search_kwargs={"k": 3},     # Return the top 3 most relevant chunks
)
# The retriever is a standard LangChain interface — it accepts a string
# and returns a list of Document objects
# This abstraction means you could swap ChromaDB for any other vector store
# later without changing any downstream code

print(" Retriever connected to ChromaDB collection:", COLLECTION_NAME)

# ── STEP 4: QUICK SANITY CHECK ─────────────────────────────────────────────────
test_docs = retriever.invoke("account is locked")
# invoke() is the standard LangChain call — replaces the older .get_relevant_documents()
print(f" Sanity check: retrieved {len(test_docs)} chunks for 'account is locked'")
print(f"   First chunk source: {test_docs[0].metadata['source']}")

from langchain_core.prompts import ChatPromptTemplate
# ChatPromptTemplate builds a structured prompt with system and human messages
# System message = standing instructions to the LLM (role, rules, constraints)
# Human message = the actual per-request input (context + ticket)

# ── THE PROMPT TEMPLATE ────────────────────────────────────────────────────────
# Two variables will be injected at runtime:
#   {context}  → the 3 retrieved chunks, formatted as text
#   {question} → the raw incoming support ticket string

prompt_template = ChatPromptTemplate.from_messages([

    ("system", """You are a senior support agent for a SaaS company.
Your job is to help human support agents by drafting a response to an incoming support ticket.

STRICT RULES — follow these without exception:
1. Answer ONLY using the information provided in the CONTEXT section below.
2. Do NOT use any knowledge outside the provided context.
3. Do NOT invent, assume, or guess any company policy, pricing, or process.
4. If the context does not contain enough information to answer the ticket,
   respond with exactly this phrase:
   I don't have enough information in the knowledge base to answer this ticket.
5. Always mention which source documents you used to construct your answer.
6. Keep your response professional, concise, and actionable."""),

{format_instructions}"""),
# {format_instructions} is injected by the parser at runtime
# It tells the LLM the exact JSON structure, field names, and allowed values
# Without this, the LLM guesses the output format and gets it wrong ~10% of the time.

    ("human", """CONTEXT:
{context}

SUPPORT TICKET:
{question}

Respond with a valid JSON object only. No text before or after the JSON.""")
# The final line is a hard instruction against conversational wrapping
# "Here is the JSON:" before the object breaks json.loads() silently
])
# Why split into system and human messages?
# Most LLMs are fine-tuned on conversations structured this way —
# system = persistent instructions, human = per-turn input
# This structure gets better instruction-following than a single flat string

print(" Updated prompt template built")
print(f"   Input variables: {prompt_template.input_variables}")



from langchain_groq import ChatGroq
# ChatGroq — LangChain's wrapper around the Groq API
# Behaves identically to ChatOpenAI — swap one import and the rest of the chain
# works unchanged, which is why we structured the retriever and prompt first

from langchain_core.runnables import RunnablePassthrough
# RunnablePassthrough — passes a value through the chain unchanged
# Used here to forward the raw question string alongside the retrieved context

from langchain_core.output_parsers import StrOutputParser
# StrOutputParser — extracts the plain text string from the LLM's response object
# Without this, the chain returns a raw AIMessage object instead of a string

# ── STEP 1: INITIALISE THE LLM ────────────────────────────────────────────────
llm = ChatGroq(
    model="openai/gpt-oss-20b",   
    temperature=0,                   # Zero temperature = deterministic output
    # For a support system, consistency matters more than creativity
    # The same ticket should always get the same response
    api_key=GROQ_API_KEY             # Loaded from .env in Cell 1
)

print(" LLM initialised:", llm.model_name)

# ── STEP 2: BUILD A CONTEXT FORMATTER ─────────────────────────────────────────
def format_context(docs):
    """
    Takes a list of Document objects returned by the retriever
    and formats them into a single string for the {context} slot in the prompt.
    Each chunk is separated by a divider and prefixed with its source filename
    so the LLM knows which document each piece of text came from.
    """
    formatted = []
    for doc in docs:
        source = doc.metadata.get("source", "unknown")  # Get the filename
        formatted.append(f"Source: {source}\n{doc.page_content}")
    return "\n\n---\n\n".join(formatted)
    # The --- divider helps the LLM visually separate chunks
    # and correctly attribute content to its source when citing

# ── STEP 3: WIRE THE FULL CHAIN ───────────────────────────────────────────────
# LangChain uses the | operator (pipe) to connect components left to right
# Each component receives the output of the previous one

chain = (
    {
        "context": retriever | format_context,
        # retriever.invoke(question) → list of Documents → format_context() → string
        # The | pipes the retriever's output directly into format_context

        "question": RunnablePassthrough()
        # The raw ticket string passes through unchanged into the {question} slot

        "format_instructions": lambda _: parser.get_format_instructions(),
    }
    | prompt_template   # Slots {context} and {question} into the prompt
    | llm               # Sends the filled prompt to Groq
    | parser  # Parses and validates the JSON response against TicketResolution
)

print(" Updated Chain assembled: retriever | format_context | prompt | llm | parser")

# ── STEP 4: DEFINE THE HELPER FUNCTION ────────────────────────────────────────
def resolve_ticket(ticket: str) -> str:
    """
    Accepts a raw support ticket string.
    Returns a grounded plain-text response drafted by the LLM.
    """
    print(f"\n Incoming ticket: {ticket}")
    print(" Retrieving context and generating response...")
    response = chain.invoke(ticket)  # One call — the full chain executes end to end
    return response

# ── TEST CASE 1: STANDARD IN-SCOPE TICKET ─────────────────────────────────────
# This maps directly to login_issues.txt and resolved_ticket_003.txt
# Expect: a grounded response citing those sources
response_1 = resolve_ticket("I cannot log into my account, it says it is locked.")
print("\n" + "="*60)
print("RESPONSE 1:")
print("="*60)
print(response_1)

# ── TEST CASE 2: EDGE CASE — SPANS MULTIPLE KB ARTICLES ───────────────────────
# This touches billing_faq.txt, refund_policy.txt, and resolved_ticket_001.txt
# Expect: a grounded response that correctly identifies the 14-day refund window
# and does NOT invent a policy that isn't in the context
response_2 = resolve_ticket("I was charged again after I cancelled. I want my money back.")
print("\n" + "="*60)
print("RESPONSE 2:")
print("="*60)
print(response_2)

# ── TEST CASE 3: OUT-OF-SCOPE TICKET ──────────────────────────────────────────
# This topic does not exist anywhere in the knowledge base
# Expect: the exact refusal phrase from the prompt — nothing invented
response_3 = resolve_ticket("Can you help me set up a custom domain for my dashboard?")
print("\n" + "="*60)
print("RESPONSE 3 (should refuse):")
print("="*60)
print(response_3)

# ── FINAL VALIDATION ──────────────────────────────────────────────────────────
# Prove every response round-trips through json.dumps() and json.loads()
# This is the actual test a downstream system like Jira or Zendesk would run

test_tickets = [
    "I cannot log into my account, it says it is locked.",
    "I was charged again after I cancelled. I want my money back.",
    "Can you help me set up a custom domain for my dashboard?"
]

print("\n" + "="*60)
print("JSON VALIDATION TEST")
print("="*60)

all_passed = True

for ticket in test_tickets:
    result = resolve_ticket(ticket)
    try:
        json_string = json.dumps(result)        # dict → JSON string
        parsed_back = json.loads(json_string)   # JSON string → dict again
        # If this succeeds, the output is genuinely valid JSON
        print(f"\n PASSED — {ticket[:50]}...")
        print(f"   category : {parsed_back['category']}")
        print(f"   priority : {parsed_back['priority']}")
        print(f"   sources  : {parsed_back['sources']}")
    except (json.JSONDecodeError, KeyError) as e:
        print(f"\n FAILED — {ticket[:50]}...")
        print(f"   Error: {e}")
        all_passed = False

print("\n" + "="*60)
print("ALL TESTS PASSED " if all_passed else "SOME TESTS FAILED ")
print("="*60)



