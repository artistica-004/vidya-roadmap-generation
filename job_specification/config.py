# src/config.py
import os
from dotenv import load_dotenv

# Load .env file
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
ENV_PATH = os.path.join(BASE_DIR, ".env")
load_dotenv(ENV_PATH)

# Adzuna config
ADZUNA_APP_ID = os.getenv("ADZUNA_APP_ID")
ADZUNA_APP_KEY = os.getenv("ADZUNA_APP_KEY")
ADZUNA_COUNTRY = os.getenv("ADZUNA_COUNTRY", "in")

# Pinecone config
PINECONE_API_KEY = os.getenv("PINECONE_API_KEY")
PINECONE_INDEX_NAME = os.getenv("PINECONE_INDEX_NAME", "job-embeddings-index")
PINECONE_CLOUD = os.getenv("PINECONE_CLOUD", "aws")
PINECONE_REGION = os.getenv("PINECONE_REGION", "us-east-1")

# Embedding model
EMBEDDING_MODEL_NAME = os.getenv(
    "EMBEDDING_MODEL_NAME",
    "sentence-transformers/all-MiniLM-L6-v2"
)

# OpenAI / LangChain config for role generation
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
OPENAI_MODEL = os.getenv("OPENAI_MODEL", "gpt-4o-mini")

# Simple sanity checks (optional – can comment out in prod)
if not ADZUNA_APP_ID or not ADZUNA_APP_KEY:
    print("[WARN] ADZUNA_APP_ID or ADZUNA_APP_KEY not set in .env")

if not PINECONE_API_KEY:
    print("[WARN] PINECONE_API_KEY not set in .env")

if not OPENAI_API_KEY:
    print("[WARN] OPENAI_API_KEY not set in .env (needed for LangChain role generation)")

# (Existing extra configs – keep as is, even if not used now)

RESUME_INDEX_NAME = os.getenv("RESUME_INDEX_NAME", "resumes-index")
