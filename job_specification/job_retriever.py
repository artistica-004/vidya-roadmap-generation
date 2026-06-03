# src/job_retriever.py
from typing import List, Dict, Optional
from pinecone import Pinecone
from .config import PINECONE_API_KEY, PINECONE_INDEX_NAME, RESUME_INDEX_NAME
from .embeddings import embed_texts
from .vector_db import get_pinecone_index


# src/job_retriever.py

def get_available_namespaces_for_index(index_name: str) -> List[str]:
    """
    Use Pinecone stats to discover which namespaces exist in a given index.
    Always returns a list of full namespace strings (not characters).
    """
    pc = Pinecone(api_key=PINECONE_API_KEY)
    idx = pc.Index(index_name)
    stats = idx.describe_index_stats()

    raw_ns = stats.get("namespaces") or {}
    namespaces = list(raw_ns.keys())

    # If there are no explicit namespaces, use default "" namespace
    if not namespaces:
        namespaces = [""]

    print(f"[JobRetriever] Namespaces in index '{index_name}': {namespaces}")
    return namespaces



def get_available_namespaces() -> List[str]:
    """
    Convenience wrapper for the job index.
    """
    return get_available_namespaces_for_index(PINECONE_INDEX_NAME)


def search_jobs_for_profile(
    profile_text: str,
    top_k: int = 10,
    namespaces: Optional[List[str]] = None,
) -> List[Dict]:
    """
    OLD MODE (still usable):
    Given a user profile/resume text:
    - embed the text
    - query Pinecone in one or more namespaces
    - return a merged, sorted list of top_k jobs
    """
    index = get_pinecone_index()

    # 1. Embed user profile
    emb = embed_texts([profile_text])[0]

    # 2. Decide which namespaces to use
    if namespaces is None:
        namespaces = get_available_namespaces()
    if not namespaces:
        print("[JobRetriever] No namespaces found in index.")
        return []

    # 3. Query each namespace and collect results
    all_matches: List[Dict] = []
    per_ns_k = max(top_k // len(namespaces), 3)  # small buffer per namespace

    for ns in namespaces:
        print(f"[JobRetriever] Querying namespace='{ns}' for top_k={per_ns_k}")
        res = index.query(
            vector=emb,
            top_k=per_ns_k,
            include_metadata=True,
            namespace=ns,
        )
        matches = res.get("matches", [])
        for m in matches:
            meta = m["metadata"]
            rec = {
                "id": m["id"],
                "score": m["score"],
                "namespace": ns,
                "title": meta.get("title"),
                "company": meta.get("company"),
                "location": meta.get("location"),
                "url": meta.get("url"),
                "category": meta.get("category"),
                "search_query": meta.get("search_query"),
                "source": meta.get("source"),
                "description": meta.get("description", ""),
            }
            all_matches.append(rec)

    # 4. Sort globally by score and keep top_k
    all_matches.sort(key=lambda x: x["score"], reverse=True)
    top_matches = all_matches[:top_k]

    print(f"[JobRetriever] Retrieved {len(top_matches)} matches.")
    return top_matches


def search_jobs_for_resume_vector(
    resume_id: str,
    top_k: int = 10,
    namespaces: Optional[List[str]] = None,
    resume_index_name: Optional[str] = None,
) -> List[Dict]:
    """
    NEW MODE:
    - Fetch resume embedding from a separate Pinecone index (resume_index_name / RESUME_INDEX_NAME)
    - Use that embedding as the query vector
    - Search against the JOB index (PINECONE_INDEX_NAME) across namespaces
    - Return merged top_k job matches
    """
    pc = Pinecone(api_key=PINECONE_API_KEY)

    # 1. Fetch resume vector from resume index (try all namespaces)
    resume_index_name = resume_index_name or RESUME_INDEX_NAME
    print(f"[JobRetriever] Fetching resume vector id='{resume_id}' from index='{resume_index_name}'")
    resume_index = pc.Index(resume_index_name)

    # find which namespace contains this resume id
    stats = resume_index.describe_index_stats()
    resume_namespaces = list(stats.get("namespaces", {}).keys() or [""])
    resume_vector = None

    for rns in resume_namespaces:
        fetched = resume_index.fetch(ids=[resume_id], namespace=rns)
        vectors = fetched.get("vectors", {})
        if resume_id in vectors:
            resume_vector = vectors[resume_id]["values"]
            print(f"[JobRetriever] Found resume id in namespace='{rns}'")
            break

    if resume_vector is None:
        print(
            f"[JobRetriever] Resume id '{resume_id}' not found in ANY namespace of index '{resume_index_name}'."
        )
        return []

    # 2. Prepare job index
    job_index = pc.Index(PINECONE_INDEX_NAME)

    # 3. Decide which namespaces to use in JOB index
    if namespaces is None:
        namespaces = get_available_namespaces_for_index(PINECONE_INDEX_NAME)
    if not namespaces:
        print("[JobRetriever] No namespaces found in job index.")
        return []

    # 4. Query each namespace and collect results
    all_matches: List[Dict] = []
    per_ns_k = max(top_k // len(namespaces), 3)

    for ns in namespaces:
        print(f"[JobRetriever] Querying job index='{PINECONE_INDEX_NAME}', namespace='{ns}'")
        res = job_index.query(
            vector=resume_vector,
            top_k=per_ns_k,
            include_metadata=True,
            namespace=ns,
        )
        matches = res.get("matches", [])
        for m in matches:
            meta = m["metadata"]
            rec = {
                "id": m["id"],
                "score": m["score"],
                "namespace": ns,
                "title": meta.get("title"),
                "company": meta.get("company"),
                "location": meta.get("location"),
                "url": meta.get("url"),
                "category": meta.get("category"),
                "search_query": meta.get("search_query"),
                "source": meta.get("source"),
                "description": meta.get("description", ""),
            }
            all_matches.append(rec)

    all_matches.sort(key=lambda x: x["score"], reverse=True)
    top_matches = all_matches[:top_k]

    print(f"[JobRetriever] Retrieved {len(top_matches)} matches (resume-based).")
    return top_matches



# quick manual test (OPTIONAL)
if __name__ == "__main__":
    # Example: using the text-based mode
    sample_profile = (
        "Experience with Python, SQL, Power BI, dashboards, data analysis, "
        "ETL, and reporting. Looking for analytics or BI roles."
    )
    print("\n[TEST] Text-based search")
    jobs = search_jobs_for_profile(sample_profile, top_k=5)
    for j in jobs:
        print(
            f"{j['score']:.3f} | {j['title']} | {j['company']} | "
            f"{j['location']} | ns={j['namespace']}"
        )
        print(j["url"])
        print("-" * 80)

    # Example: using resume vector mode (only works if resume index + id exist)
    # print("\n[TEST] Resume-vector-based search")
    # jobs2 = search_jobs_for_resume_vector("your_resume_id_here", top_k=5)
    # for j in jobs2:
    #     print(
    #         f"{j['score']:.3f} | {j['title']} | {j['company']} | "
    #         f"{j['location']} | ns={j['namespace']}"
    #     )
    #     print(j["url"])
    #     print("-" * 80)


