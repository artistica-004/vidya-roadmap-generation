# src/vector_db.py
from pinecone import Pinecone, ServerlessSpec
from .config import (
    PINECONE_API_KEY,
    PINECONE_INDEX_NAME,
    PINECONE_CLOUD,
    PINECONE_REGION,
)

INDEX_DIMENSION = 1536 # for all-MiniLM-L6-v2


def get_pinecone_index():
    if not PINECONE_API_KEY:
        raise ValueError("PINECONE_API_KEY not set in .env")

    print("[Pinecone] Initializing client...")
    pc = Pinecone(api_key=PINECONE_API_KEY)

    existing = [idx["name"] for idx in pc.list_indexes()]
    if PINECONE_INDEX_NAME not in existing:
        print(f"[Pinecone] Creating index: {PINECONE_INDEX_NAME}")
        pc.create_index(
            name=PINECONE_INDEX_NAME,
            dimension=INDEX_DIMENSION,
            metric="cosine",
            spec=ServerlessSpec(
                cloud=PINECONE_CLOUD,
                region=PINECONE_REGION,
            ),
        )
    else:
        print(f"[Pinecone] Using existing index: {PINECONE_INDEX_NAME}")

    index = pc.Index(PINECONE_INDEX_NAME)
    return index
