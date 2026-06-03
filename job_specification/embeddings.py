# src/embeddings.py
from typing import List
from openai import OpenAI
from .config import OPENAI_API_KEY, EMBEDDING_MODEL_NAME

_client = None

def get_client():
    global _client
    if _client is None:
        if not OPENAI_API_KEY:
            raise ValueError("OPENAI_API_KEY not found in .env")
        _client = OpenAI(api_key=OPENAI_API_KEY)
    return _client


def embed_texts(texts: List[str]):
    """
    Uses OpenAI embeddings - returns 1536-dim vectors
    """
    client = get_client()
    response = client.embeddings.create(
        model=EMBEDDING_MODEL_NAME,  # "text-embedding-3-small" or "text-embedding-ada-002"
        input=texts
    )
    return [x.embedding for x in response.data]
