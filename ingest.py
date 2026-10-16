import os  # Standard library — for creating directories and file paths

# ── CONFIGURATION ──────────────────────────────────────────────────────────────
# All source documents will live in this folder.
# Keeping them separate from code makes the ingestion script reusable
# — swap the folder contents and you have a different knowledge base.
KB_DIR = "knowledge_base"
os.makedirs(KB_DIR, exist_ok=True)  # Create folder; do nothing if it already exists


documents = {# ── Knowledge Base Articles ─────────────────────────────────────────────────
    "password_reset.txt": """How to Reset Your Password
If you have forgotten your password, follow these steps:
1. Go to the login page and click Forgot Password.
2. Enter the email address associated with your account.
3. Check your inbox for a reset link. The link expires in 30 minutes.
4. Click the link and enter a new password. Passwords must be at least 8 characters and contain one uppercase letter and one number.
5. If you do not receive the email within 5 minutes, check your spam folder or contact support.
Note: For security reasons, we cannot reset passwords over chat. The self-service flow is the only supported method.""",

    "billing_faq.txt": """Billing and Payment FAQ
Q: When am I charged?
A: Subscriptions are billed on the first day of each billing cycle. Annual plans are charged once per year.
Q: What payment methods do you accept?
A: We accept Visa, Mastercard, American Express, and PayPal. Bank transfers are available for Enterprise plans only.
Q: Can I get a refund?
A: Refunds are available within 14 days of the initial purchase for new subscriptions. Renewals are non-refundable.
Q: How do I update my payment method?
A: Go to Settings > Billing > Payment Methods. Click Update and enter new card details.""",

    "api_rate_limits.txt": """API Rate Limits and Quotas
All API plans are subject to rate limiting to ensure fair usage.
Standard Plan: 100 requests per minute, 10,000 requests per day.
Pro Plan: 500 requests per minute, 100,000 requests per day.
Enterprise Plan: Custom limits — contact your account manager.
If you exceed your rate limit, the API returns a 429 Too Many Requests status code.
Resolution: implement exponential backoff in your client and wait before retrying.
If you consistently hit limits, consider upgrading your plan or requesting a quota increase.""",

    "login_issues.txt": """Troubleshooting Login Problems
Issue: Cannot log in with correct credentials.
Fix: Clear browser cache and cookies, try incognito mode, or use a different browser.
Issue: Account locked after failed attempts.
Fix: Accounts lock for 15 minutes after 5 failed login attempts. Wait 15 minutes, then use the Forgot Password flow.
Issue: SSO not working.
Fix: Confirm with your IT admin that SSO is active. Ensure your email domain is registered under your organization SSO settings. Contact support with your organization ID.""",

    "account_suspension.txt": """Why Was My Account Suspended?
Accounts may be suspended for three reasons:
1. Payment failure: If a charge fails and is not resolved within 7 days, the account suspends automatically.
2. Terms of service violation: Automated abuse, scraping, or reselling access.
3. Security hold: Unusual login activity triggers a temporary hold.
Reactivation:
- Payment issue: Update payment in Settings > Billing. Account reactivates within 1 hour.
- Security hold: Respond to the verification email sent to your registered address.
- ToS violation: Contact support with your account ID. Reviews take 48 hours.""",

    "data_export.txt": """Exporting Your Data
Export all account data from Settings > Data > Export.
Supported formats: CSV and JSON.
Processing time: Under 15 minutes for accounts with fewer than 10,000 records. Larger exports take up to 2 hours.
A secure download link is emailed when the export is ready. Links expire after 24 hours.
Important: Data export is available on Pro and Enterprise plans only. Standard plan users must upgrade to access this feature.""",

    "two_factor_auth.txt": """Setting Up Two-Factor Authentication (2FA)
To enable 2FA:
1. Go to Settings > Security > Two-Factor Authentication.
2. Click Enable 2FA.
3. Scan the QR code with Google Authenticator or Authy.
4. Enter the 6-digit code to confirm.
5. Save your backup codes securely — these are your recovery method if you lose your phone.
To disable 2FA: Settings > Security > Two-Factor Authentication > Disable. Confirmation with your current code is required.""",

    "subscription_upgrade.txt": """Upgrading or Downgrading Your Subscription
To change your plan: Settings > Billing > Current Plan > Change Plan.
Upgrades take effect immediately. You are charged a prorated amount for the rest of the current cycle.
Downgrades take effect at the start of the next billing cycle. You keep current features until then.
Warning: Downgrading may cause data or feature loss if your usage exceeds the lower plan limits. A warning is shown before you confirm.""",

    "integration_setup.txt": """Setting Up Third-Party Integrations
Supported integrations: Slack, Jira, Salesforce, Zapier.
Setup: Settings > Integrations > choose integration > authorize via OAuth.
Troubleshooting:
- Integration stopped working: disconnect and reconnect to refresh the OAuth token.
- Token revoked by the third-party: reauthenticate from their side first.
- Jira specifically: ensure the API token has project-level write access.""",

    "refund_policy.txt": """Refund Policy
New subscriptions: Full refund within 14 days of purchase.
Renewals: Non-refundable.
Annual plans cancelled mid-year: No prorated refund. Access continues to end of billing year.
Add-ons: Refundable within 7 days if unused.
To request a refund: Submit a billing support ticket with your account email and order ID.
Processing time: 5 to 7 business days, returned to the original payment method only.""",

    # ── Past Resolved Tickets ───────────────────────────────────────────────────
    # These teach the system what good resolutions look like in practice.
    "resolved_ticket_001.txt": """Resolved Ticket #1042 — Charged After Cancellation
Customer reported being charged after believing their account was cancelled.
Investigation: The customer had cancelled an add-on, not the main subscription.
Resolution: Agent guided customer to Settings > Billing > Cancel Subscription. Full refund issued — charge was within the 14-day refund window.
Tags: billing, refund, cancellation, subscription""",

    "resolved_ticket_002.txt": """Resolved Ticket #1187 — API Returning 429 Errors
Customer reported API calls failing intermittently with status 429.
Investigation: Customer was on Standard plan making 150 requests per minute, exceeding the 100/min limit.
Resolution: Agent advised upgrading to Pro plan or adding exponential backoff on the client side. Customer implemented a client-side queue. Issue resolved.
Tags: API, rate limit, 429, developer, Standard plan""",

    "resolved_ticket_003.txt": """Resolved Ticket #1203 — Account Locked After Password Attempts
Customer could not log in after multiple failed attempts.
Investigation: Customer was unaware of the 15-minute auto-unlock window after 5 failed attempts.
Resolution: Agent confirmed the lock would expire and directed customer to the Forgot Password flow. Login successful after reset.
Tags: login, locked, password, authentication""",

    "resolved_ticket_004.txt": """Resolved Ticket #1301 — Data Export Never Arrived
Customer initiated a data export 3 hours prior and received no email.
Investigation: Customer was on Standard plan. Data export is a Pro and Enterprise feature only. The export silently failed due to plan restriction.
Resolution: Agent informed customer of the plan limitation and escalated to billing team for plan upgrade discussion.
Tags: export, Standard plan, feature access, billing""",

    "resolved_ticket_005.txt": """Resolved Ticket #1389 — Slack Integration Stopped Posting
Customer reported Slack integration had stopped sending notifications.
Investigation: Slack workspace admin had rotated OAuth tokens, invalidating the existing connection.
Resolution: Agent instructed customer to go to Settings > Integrations > Slack, disconnect, and reconnect via OAuth to generate a fresh token. Integration resumed immediately.
Tags: integration, Slack, OAuth, token, disconnect""",
}



# ── WRITE FILES TO DISK ────────────────────────────────────────────────────────
for filename, content in documents.items():
    filepath = os.path.join(KB_DIR, filename)  # e.g., "knowledge_base/billing_faq.txt"
    with open(filepath, "w", encoding="utf-8") as f:
        f.write(content)  # Write the plain text content to the file

# ── CONFIRM ────────────────────────────────────────────────────────────────────
print(f" Created {len(documents)} documents in '{KB_DIR}/'")
for name in documents.keys():
    print(f"   {name}")

from langchain_core.documents import Document
# Document is LangChain's core data container — page_content + metadata
# Using langchain_core directly avoids the langchain-community deprecation entirely


from langchain_text_splitters import RecursiveCharacterTextSplitter
# RecursiveCharacterTextSplitter - splits the text into smaller chunks. It will split text by trying "\n\n", then "\n", then " ", then ""
# This is useful for long documents that need to be split into smaller chunks for processing.

# Step 1: Load documents from the knowledge base directory

raw_documents = []

for filename in os.listdir(KB_DIR):               # Iterate over all files in the folder
    if not filename.endswith(".txt"):              # Skip anything that isn't a .txt file
        continue
    filepath = os.path.join(KB_DIR, filename)     # Build full path e.g. knowledge_base/billing_faq.txt
    with open(filepath, "r", encoding="utf-8") as f:
        content = f.read()                        # Read the entire file as a string
    raw_documents.append(
        Document(
            page_content=content,                 # The actual text
            metadata={"source": filepath}         # Preserve the filename — needed for citations in Milestone 3
        )
    )

print(f" Loaded {len(raw_documents)} raw documents")

# Step 2: Split the documents into smaller chunks
text_splitter = RecursiveCharacterTextSplitter(
    chunk_size=500,  # The maximum size of each chunk
    chunk_overlap=50,  # The number of characters to overlap between chunks
    length_function=len,  # The function to use to calculate the length of the text
)

chunks = text_splitter.split_documents(raw_documents)  # Split the documents into chunks
# So here every chunk know which source file it came from. This is important because we want to be able to trace back the chunks to the original documents for context and reference.

print(f"Split into {len(chunks)} chunks")  # Print the number of chunks created
print(f"Average chunk size: {sum(len(chunk.page_content) for chunk in chunks) / len(chunks)} characters")  # Print the average chunk size

# ── STEP 3: INSPECT ONE CHUNK ─────────────────────────────────────────────────
# Always inspect before embedding — verify chunks look sensible
print(f"\n Sample chunk from: {chunks[0].metadata['source']}")
print("-" * 50)
print(chunks[0].page_content)
print("-" * 50)


from langchain_huggingface import HuggingFaceEmbeddings
# HuggingFaceEmbeddings - This is the class that will be used to generate embeddings for the text chunks. 
# It uses a model from Hugging Face to convert text into numerical vectors that can be used for similarity search.

from langchain_chroma import Chroma
# Chroma - This is the vector store that will be used to store the embeddings.

print("\n Loading the Embedding model...")

embedding_model = HuggingFaceEmbeddings(
    model_name="sentence-transformers/all-MiniLM-L6-v2",  # The model to use for generating embeddings
    model_kwargs={"device": "cpu"},  # Use CPU for inference
    encode_kwargs={"normalize_embeddings": True}  # Normalize the embeddings to unit length
)

print("Embedding model loaded successfully.")

# Here we are creating a Chroma vector store from the text chunks and the embedding model. The vector store will allow us to perform similarity searches on the text chunks based on their embeddings.
chroma_dir = "chroma_db"  # Directory to store the Chroma database
print(f"\n Embedding {len(chunks)} chunks and storing in '{chroma_dir}/' ...")

vector_store = Chroma.from_documents(
    documents=chunks,  # The text chunks to embed
    embedding=embedding_model,  # The embedding model to use
    persist_directory=chroma_dir,  # The directory to store the vector store
    collection_name="knowledge_base"  # Name of the collection in the vector store
)

print(f" Vector store created and persisted in '{chroma_dir}/' successfully.")
print(f" Total vectors stored: {vector_store._collection.count()}")  # Print the total number of vectors stored in the vector store







