"""
config.py — Centralized, sanitized environment variable loading for Vidya V3.

WHY THIS FILE EXISTS:
  Hugging Face Spaces stores secrets exactly as typed, including any accidental
  trailing newlines or spaces. When the Pinecone SDK injects a key with a
  trailing '\n' into an HTTP header, urllib3 raises:
      ValueError: Invalid header value b'pcsk_...\n'
  .strip() on every secret read prevents this class of bug permanently.
"""

import os
from dotenv import load_dotenv

load_dotenv()


def _require(name: str) -> str:
    """Read an env var, strip whitespace, and raise clearly if missing."""
    raw = os.environ.get(name, "")
    value = raw.strip()
    if not value:
        raise EnvironmentError(
            f"[CONFIG] ❌ Required environment variable '{name}' is missing or empty.\n"
            f"  → On Hugging Face Spaces: Settings → Secrets → add '{name}' with no trailing spaces.\n"
            f"  → Locally: add '{name}=<value>' to your .env file."
        )
    return value


def _optional(name: str, default: str = "") -> str:
    """Read an optional env var and strip whitespace."""
    return os.environ.get(name, default).strip()


# ── Required keys ────────────────────────────────────────────
PINECONE_API_KEY    = _require("PINECONE_API_KEY")
GEMINI_API_KEY      = _require("GEMINI_API_KEY")

# ── Optional / defaulted keys ────────────────────────────────
# GOOGLE_API_KEY is an alias some libraries use for Gemini.
# We fall back to GEMINI_API_KEY so there is exactly one source of truth.
GOOGLE_API_KEY      = _optional("GOOGLE_API_KEY") or GEMINI_API_KEY

PINECONE_INDEX_NAME = _optional("PINECONE_INDEX_NAME", "live-assistant-index-v2")
EMBEDDING_MODEL     = _optional("EMBEDDING_MODEL",     "gemini-embedding-001")
GOOGLE_MODEL        = _optional("GOOGLE_MODEL",        "gemini-2.5-flash")
OPENAI_API_KEY      = _optional("OPENAI_API_KEY")

# ── Debug flags ───────────────────────────────────────────────
PINECONE_DEBUG          = _optional("PINECONE_DEBUG", "0") == "1"
PINECONE_DEBUG_FETCH_IDS = _optional("PINECONE_DEBUG_FETCH_IDS", "")


# ── Startup validation log ────────────────────────────────────
def log_config_status() -> None:
    """Call once at app startup to confirm all keys loaded cleanly."""
    print("[CONFIG] Environment loaded:")
    print(f"  PINECONE_API_KEY     = {'✓ set' if PINECONE_API_KEY else '✗ MISSING'} "
          f"(repr tail: {repr(PINECONE_API_KEY[-6:]) if PINECONE_API_KEY else 'N/A'})")
    print(f"  GEMINI_API_KEY       = {'✓ set' if GEMINI_API_KEY else '✗ MISSING'}")
    print(f"  GOOGLE_API_KEY       = {'✓ set' if GOOGLE_API_KEY else '✗ MISSING'}")
    print(f"  PINECONE_INDEX_NAME  = {PINECONE_INDEX_NAME}")
    print(f"  GOOGLE_MODEL         = {GOOGLE_MODEL}")
    print(f"  OPENAI_API_KEY       = {'✓ set' if OPENAI_API_KEY else 'not set (Gemini-only mode)'}")
    print(f"  PINECONE_DEBUG       = {PINECONE_DEBUG}")