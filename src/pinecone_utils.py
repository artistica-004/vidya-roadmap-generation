import os
import re
from typing import Optional
from dotenv import load_dotenv
from google import genai
from google.genai import types
from pinecone import Pinecone

# ---------------------------------------------------
# ENV SETUP
# ---------------------------------------------------
load_dotenv()

INDEX_NAME = os.environ.get("PINECONE_INDEX_NAME", "live-assistant-index-v2")
EMBED_MODEL = "gemini-embedding-001"

GEMINI_API_KEY = os.environ["GEMINI_API_KEY"]
PINECONE_API_KEY = os.environ["PINECONE_API_KEY"]
PINECONE_DEBUG = os.environ.get("PINECONE_DEBUG", "0") == "1"
PINECONE_DEBUG_FETCH_IDS = os.environ.get("PINECONE_DEBUG_FETCH_IDS", "")

# ---------------------------------------------------
# CLIENTS (initialize once per container)
# ---------------------------------------------------
genai_client = genai.Client(api_key=GEMINI_API_KEY)
pc = Pinecone(api_key=PINECONE_API_KEY)


# ---------------------------------------------------
# EMBEDDING FUNCTION (3072-DIM)
# ---------------------------------------------------
def get_embedding(text: str, task_type: str = "RETRIEVAL_QUERY") -> list:
    """
    Generate embedding using Gemini (3072-dim).
    3072 embeddings are already normalized.
    """

    response = genai_client.models.embed_content(
        model=EMBED_MODEL,
        contents=text,
        config=types.EmbedContentConfig(
            task_type=task_type,
            output_dimensionality=3072
        )
    )

    return response.embeddings[0].values


# ---------------------------------------------------
# SESSION ID RETRIEVAL
# ---------------------------------------------------
def retrieve_session_id(user_id: str) -> str:
    try:
        index = pc.Index(INDEX_NAME)

        print(f"[SESSION RETRIEVAL] Looking for session ID for user: {user_id}")

        session_vector_id = f"{user_id}_session_metadata"
        fetch_result = index.fetch(ids=[session_vector_id], namespace=user_id)

        if fetch_result and fetch_result.vectors:
            if session_vector_id in fetch_result.vectors:
                metadata = fetch_result.vectors[session_vector_id].metadata or {}
                ai_session_id = metadata.get("ai_session_id")

                if ai_session_id:
                    print(f"[SESSION RETRIEVAL] ✓ Found session ID: {ai_session_id}")
                    return ai_session_id

        print(f"[SESSION RETRIEVAL] ✗ No session metadata found")
        return None

    except Exception as e:
        print(f"[SESSION RETRIEVAL] ✗ Error: {e}")
        return None


def retrieve_icp_type(user_id: str) -> Optional[str]:
    try:
        index = pc.Index(INDEX_NAME)

        print(f"[ICP] Looking for icp_type for user: {user_id}")

        query_embedding = get_embedding(
            "user onboarding profile",
            task_type="RETRIEVAL_QUERY"
        )

        query_response = index.query(
            vector=query_embedding,
            top_k=5,
            namespace=user_id,
            include_metadata=True,
            filter={"doc_type": {"$eq": "onboarding"}}
        )

        matches = query_response.matches if query_response else []

        print(f"[ICP DEBUG] Found {len(matches)} onboarding matches")

        if not matches:
            print("[ICP] No onboarding data found; cannot classify icp_type")
            return None

        def get_metadata(match_obj) -> dict:
            if isinstance(match_obj, dict):
                return match_obj.get("metadata", {}) or {}
            return match_obj.metadata or {}

        def get_match_id(match_obj) -> Optional[str]:
            if isinstance(match_obj, dict):
                return match_obj.get("id")
            return match_obj.id if hasattr(match_obj, "id") else None

        for match in matches:
            metadata = get_metadata(match)

            print(f"[ICP DEBUG] Metadata: {metadata}")

            icp_type = metadata.get("icp_type")

            if icp_type:
                normalized = str(icp_type).strip().lower()
                if normalized in {"high", "high_wage"}:
                    print("[ICP] Found icp_type: high")
                    return "high"
                if normalized in {"low", "low_wage"}:
                    print("[ICP] Found icp_type: low")
                    return "low"
                print(f"[ICP] Unsupported icp_type value: {icp_type}")

        onboarding_texts = []
        for match in matches:
            metadata = get_metadata(match)
            text = metadata.get("text", "")
            if text:
                onboarding_texts.append(text)

        combined_text = " ".join(onboarding_texts).lower().strip()

        if not combined_text:
            print("[ICP] Onboarding data found but text missing; cannot classify")
            return None

        def text_has_phrase(text: str, phrase: str) -> bool:
            if " " in phrase:
                return phrase in text
            return re.search(r"\b" + re.escape(phrase) + r"\b", text) is not None

        def match_any(text: str, phrases: list) -> bool:
            return any(text_has_phrase(text, p) for p in phrases)

        score = 0
        reasons = []

        high_signals = [
            ("employment_role", 2, [
                "software engineer",
                "developer",
                "product manager",
                "analyst",
                "consultant",
                "designer",
                "working professional",
                "service engineer",
                "support engineer"
            ]),
            ("income_career", 2, [
                "salary",
                "promotion",
                "upskill",
                "upskilling",
                "career switch",
                "job switch",
                "switch companies",
                "switching companies",
                "already employed",
                "employed"
            ]),
            ("tooling_access", 1, [
                "laptop",
                "macbook",
                "office",
                "jira",
                "github",
                "slack",
                "aws",
                "azure",
                "gcp"
            ]),
            ("english_comfort", 1, [
                "english preferred",
                "english only",
                "speak english",
                "comfortable in english",
                "confident in english"
            ]),
            ("career_goals", 2, [
                "promotion",
                "faang",
                "switch companies",
                "switching companies",
                "senior engineer",
                "leadership",
                "team lead",
                "lead role",
                "principal",
                "architect"
            ])
        ]

        low_signals = [
            ("entry_level", -2, [
                "student",
                "fresher",
                "12th pass",
                "12th",
                "diploma",
                "iti",
                "college placement",
                "campus",
                "placement",
                "internship",
                "first job"
            ]),
            ("access_constraints", -2, [
                "mobile only",
                "phone only",
                "no laptop",
                "without laptop",
                "hindi preferred",
                "tamil preferred",
                "telugu preferred",
                "regional language",
                "vernacular",
                "low bandwidth",
                "limited internet"
            ]),
            ("economic_constraints", -3, [
                "need job urgently",
                "financial",
                "cheap",
                "free",
                "no budget",
                "price sensitive",
                "afford",
                "low cost"
            ]),
            ("job_goal_entry", -2, [
                "data entry",
                "support role",
                "bpo",
                "basic it job",
                "first job"
            ])
        ]

        if match_any(combined_text, ["no laptop", "without laptop"]):
            high_signals = [
                (name, delta, phrases) if name != "tooling_access" else (name, delta, [p for p in phrases if p not in {"laptop", "macbook"}])
                for name, delta, phrases in high_signals
            ]

        for name, delta, phrases in high_signals:
            if match_any(combined_text, phrases):
                score += delta
                reasons.append(name)

        for name, delta, phrases in low_signals:
            if match_any(combined_text, phrases):
                score += delta
                reasons.append(name)

        icp_type = "high" if score >= 2 else "low"

        print(f"[ICP] Heuristic score: {score}, reasons: {reasons}")
        print(f"[ICP] Heuristic classification: {icp_type}")

        try:
            metadata_update = {
                "icp_type": icp_type,
                "icp_score": score,
                "icp_reasoning": reasons
            }

            for match in matches:
                match_id = get_match_id(match)
                if match_id:
                    index.update(id=match_id, namespace=user_id, set_metadata=metadata_update)
        except Exception as e:
            print(f"[ICP] Failed to persist icp metadata: {e}")

        return icp_type

    except Exception as e:
        print(f"[ICP] Error retrieving icp_type: {e}")
        print("[ICP] icp_type retrieval failed; refusing to default")
        return None

# # ---------------------------------------------------
# # ICP TYPE RETRIEVAL
# # ---------------------------------------------------
# def retrieve_icp_type(user_id: str) -> str:
#     try:
#         index = pc.Index(INDEX_NAME)

#         print(f"[ICP] Looking for icp_type for user: {user_id}")

#         session_vector_id = f"{user_id}_session_metadata"
#         fetch_result = index.fetch(ids=[session_vector_id], namespace=user_id)

#         if fetch_result and fetch_result.vectors:
#             if session_vector_id in fetch_result.vectors:
#                 metadata = fetch_result.vectors[session_vector_id].metadata or {}
#                 icp_type = metadata.get("icp_type")

#                 if icp_type:
#                     print(f"[ICP] Found icp_type: {icp_type}")
#                     return icp_type

#         print("[ICP] icp_type not found, defaulting to high_wage")
#         return "high_wage"

#     except Exception as e:
#         print(f"[ICP] Error retrieving icp_type: {e}")
#         print("[ICP] icp_type not found, defaulting to high_wage")
#         return "high_wage"


# ---------------------------------------------------
# CONTEXT RETRIEVAL
# ---------------------------------------------------
def retrieve_context(user_id: str) -> str:
    index = pc.Index(INDEX_NAME)

    print(f"[PINECONE] Connecting to index: {INDEX_NAME}")

    try:
        stats = index.describe_index_stats()
        print(f"[PINECONE] ✓ Connected! Total vectors: {stats.get('total_vector_count', 0)}")
        if PINECONE_DEBUG:
            print(f"[PINECONE DEBUG] Index name: {INDEX_NAME}")
            print(f"[PINECONE DEBUG] Index stats: {stats}")
            namespaces = stats.get("namespaces", {})
            if isinstance(namespaces, dict):
                print(f"[PINECONE DEBUG] Namespace keys: {list(namespaces.keys())}")
    except Exception as e:
        print(f"[PINECONE] ✗ Connection failed: {e}")
        raise

    context_parts = []

    # ---------------------------------------------------
    # STEP 1: Resume (direct fetch)
    # ---------------------------------------------------
    print(f"[ROADMAP] Retrieving resume for {user_id}...")

    try:
        resume_id = f"{user_id}_resume_summary"
        fetch_result = index.fetch(ids=[resume_id], namespace=user_id)

        if fetch_result and fetch_result.vectors:
            if resume_id in fetch_result.vectors:
                metadata = fetch_result.vectors[resume_id].metadata or {}
                text = metadata.get("formatted_context") or metadata.get("text", "")
                if text:
                    context_parts.append(f"=== USER BACKGROUND ===\n{text}\n")
                    print(f"[ROADMAP] ✓ Found resume ({len(text)} chars)")
        else:
            print("[ROADMAP] ✗ No resume found")

    except Exception as e:
        print(f"[ROADMAP] Resume fetch error: {e}")

    # ---------------------------------------------------
    # STEP 2: Onboarding Data
    # ---------------------------------------------------
    print(f"[ROADMAP] Retrieving onboarding data for {user_id}...")

    try:
        query_embedding = get_embedding(
            "onboarding questions and answers learning goals",
            task_type="RETRIEVAL_QUERY"
        )

        query_response = index.query(
            vector=query_embedding,
            top_k=20,
            namespace=user_id,
            include_metadata=True,
            filter={"doc_type": {"$eq": "onboarding"}}
        )

        matches = query_response.matches if query_response else []

        if not matches and PINECONE_DEBUG:
            print("[PINECONE DEBUG] No onboarding matches with filter; running unfiltered query")
            debug_response = index.query(
                vector=query_embedding,
                top_k=10,
                namespace=user_id,
                include_metadata=True
            )
            debug_matches = debug_response.matches if debug_response else []
            print(f"[PINECONE DEBUG] Unfiltered matches: {len(debug_matches)}")
            for match in debug_matches[:5]:
                print(f"[PINECONE DEBUG] Match metadata: {match.metadata}")

            if PINECONE_DEBUG_FETCH_IDS:
                debug_ids = [s.strip() for s in PINECONE_DEBUG_FETCH_IDS.split(",") if s.strip()]
                if debug_ids:
                    fetch_debug = index.fetch(ids=debug_ids, namespace=user_id)
                    fetched = fetch_debug.vectors if fetch_debug else {}
                    print(f"[PINECONE DEBUG] Fetch ids result keys: {list(fetched.keys()) if fetched else []}")
                    for vector_id, vector in (fetched or {}).items():
                        metadata = vector.metadata if vector else None
                        print(f"[PINECONE DEBUG] {vector_id} metadata: {metadata}")

        if matches:
            sorted_matches = sorted(
                matches,
                key=lambda x: x.metadata.get("question_number", 0)
            )

            onboarding_text = "\n\n".join([
                m.metadata.get("text", "") for m in sorted_matches
            ])

            if onboarding_text.strip():
                context_parts.append(f"=== ONBOARDING INTERVIEW ===\n{onboarding_text}\n")
                print(f"[ROADMAP] ✓ Found {len(sorted_matches)} onboarding Q&As")
        else:
            print("[ROADMAP] ✗ No onboarding data found")

    except Exception as e:
        print(f"[ROADMAP] Onboarding query error: {e}")

    # ---------------------------------------------------
    # STEP 3: Tutor History
    # ---------------------------------------------------
    print(f"[ROADMAP] Retrieving tutor history for {user_id}...")

    try:
        query_embedding = get_embedding(
            "recent learning topics discussions",
            task_type="RETRIEVAL_QUERY"
        )

        query_response = index.query(
            vector=query_embedding,
            top_k=5,
            namespace=user_id,
            include_metadata=True,
            filter={"doc_type": {"$eq": "conversation"}}
        )

        matches = query_response.matches if query_response else []

        if matches:
            sorted_matches = sorted(
                matches,
                key=lambda x: x.metadata.get("timestamp", 0),
                reverse=True
            )

            tutor_text = "\n".join([
                m.metadata.get("text", "") for m in sorted_matches
            ])

            if tutor_text.strip():
                context_parts.append(f"=== RECENT LEARNING DISCUSSIONS ===\n{tutor_text}\n")
                print(f"[ROADMAP] ✓ Found {len(sorted_matches)} tutor conversations")
        else:
            print("[ROADMAP] ℹ No tutor history found")

    except Exception as e:
        print(f"[ROADMAP] Tutor history query error: {e}")

    # ---------------------------------------------------
    # FINAL
    # ---------------------------------------------------
    if not context_parts:
        print("[ROADMAP] ✗ No context found at all!")
        return ""

    combined_context = "\n".join(context_parts)
    print(f"[ROADMAP] ✓ Total context length: {len(combined_context)} chars")

    return combined_context


# =====================================================
# RAW CONTEXT RETRIEVAL
# =====================================================

def retrieve_raw_context(
    user_id: str
) -> str:

    index = pc.Index(INDEX_NAME)

    context_parts = []

    # -------------------------------------------------
    # RESUME CONTEXT
    # -------------------------------------------------

    try:

        resume_id = (
            f"{user_id}_resume_summary"
        )

        fetch_result = index.fetch(
            ids=[resume_id],
            namespace=user_id
        )

        if (
            fetch_result
            and fetch_result.vectors
            and resume_id in fetch_result.vectors
        ):

            metadata = (
                fetch_result
                .vectors[resume_id]
                .metadata
                or {}
            )

            text = (
                metadata.get("formatted_context")
                or metadata.get("text", "")
            )

            if text:

                context_parts.append(
                    f"USER BACKGROUND\n{text}"
                )

    except Exception as exc:

        print(
            f"resume context error "
            f"for user {user_id}: {exc}"
        )

    # -------------------------------------------------
    # ONBOARDING CONTEXT
    # -------------------------------------------------

    try:

        query_embedding = get_embedding(
            "onboarding questions answers learning goals",
            task_type="retrieval_query",
        )

        query_response = index.query(
            vector=query_embedding,
            top_k=20,
            namespace=user_id,
            include_metadata=True,
            filter={
                "doc_type": {
                    "$eq": "onboarding"
                }
            },
        )

        matches = (
            query_response.matches
            if query_response
            else []
        )

        if matches:

            sorted_matches = sorted(
                matches,
                key=lambda x:
                    x.metadata.get(
                        "question_number",
                        0
                    ),
            )

            onboarding_text = "\n\n".join(
                m.metadata.get("text", "")
                for m in sorted_matches
            )

            if onboarding_text.strip():

                context_parts.append(
                    f"ONBOARDING RESPONSES\n{onboarding_text}"
                )

    except Exception as exc:

        print(
            f"onboarding context error "
            f"for user {user_id}: {exc}"
        )

    return "\n\n".join(context_parts)

