# src/pipeline.py
import os
from typing import List, Dict

import pandas as pd

from .adzuna_client import fetch_jobs_from_adzuna
from .vector_db import get_pinecone_index
from .role_generator import get_job_roles
from .embeddings import embed_texts

DATA_DIR = os.path.join(os.path.dirname(os.path.dirname(__file__)), "data")
os.makedirs(DATA_DIR, exist_ok=True)


def normalize_job_record(job: Dict) -> Dict:
    """
    Extracts and normalizes fields from raw Adzuna job JSON.
    """
    return {
        "id": str(job.get("id", "")),
        "title": job.get("title", ""),
        "company": job.get("company", {}).get("display_name", ""),
        "location": job.get("location", {}).get("display_name", ""),
        "description": job.get("description", ""),
        "redirect_url": job.get("redirect_url", ""),
        "created": job.get("created", ""),
        "category": job.get("category", {}).get("label", ""),
        "search_query": job.get("_search_query", ""),  # LLM-generated role keyword
        "source": "adzuna",
    }


def export_jobs_to_csv(jobs: List[Dict], csv_path: str):
    """
    Save normalized job records to a CSV.
    """
    records = [normalize_job_record(j) for j in jobs]
    df = pd.DataFrame(records)
    df.to_csv(csv_path, index=False, encoding="utf-8")
    print(f"[CSV] Saved {len(df)} jobs to {csv_path}")


def upsert_jobs_to_pinecone(jobs: List[Dict]):
    """
    Generate embeddings for jobs and upsert to Pinecone,
    grouping by search_query into separate namespaces.
    """
    index = get_pinecone_index()

    # Group by role keyword (search_query)
    jobs_by_role: Dict[str, List[Dict]] = {}
    for j in jobs:
        rec = normalize_job_record(j)
        role = rec["search_query"] or "unspecified"
        jobs_by_role.setdefault(role, []).append(rec)

    for role, recs in jobs_by_role.items():
        namespace = role.lower().replace(" ", "_").replace("/", "_")
        print(
            f"[Pinecone/Adzuna] Upserting {len(recs)} jobs "
            f"into namespace='{namespace}'"
        )

        texts = []
        ids = []
        metadatas = []

        for rec in recs:
            if not rec["id"]:
                continue
            full_text = f"{rec['title']}\n\n{rec['description']}"
            texts.append(full_text)
            ids.append(rec["id"])
            metadatas.append(
                {
                    "title": rec["title"],
                    "company": rec["company"],
                    "location": rec["location"],
                    "url": rec["redirect_url"],
                    "category": rec["category"],
                    "search_query": rec["search_query"],
                    "source": rec["source"],
                    "description": rec["description"],  # 👈 added for skill extraction later
                }
            )

        if not texts:
            continue

        print(f"[Embeddings/Adzuna] Generating embeddings for {len(texts)} jobs...")
        embeddings = embed_texts(texts)

        vectors = []
        for idx, emb in enumerate(embeddings):
            vectors.append(
                {
                    "id": ids[idx],
                    "values": emb,
                    "metadata": metadatas[idx],
                }
            )

        # 🔹 Upsert into dedicated namespace per role
        index.upsert(vectors=vectors, namespace=namespace)

    print("[Pinecone/Adzuna] All namespaces upsert complete.")


def main():
    # 🔹 High-level description only – LLM decides job titles
    goal_description = (
        "We are building an AI-driven job recommendation system for users in India, "
        "mainly in IT, software engineering, data, analytics, cloud, testing/QA, "
        "and business/tech roles at junior to mid-level."
    )

    # 1) LLM generates relevant job roles (agentic part)
    role_queries = get_job_roles(goal_description=goal_description, n_roles=10)

    page = 1
    results_per_page = 20
    all_jobs: List[Dict] = []

    for q in role_queries:
        print(f"[Pipeline/Adzuna-LLM] Fetching jobs for role keyword: '{q}'")
        jobs = fetch_jobs_from_adzuna(
            query=q,
            page=page,
            results_per_page=results_per_page,
        )
        # Tag with search_query (for namespace + analysis)
        for j in jobs:
            j["_search_query"] = q

        print(f"[Pipeline/Adzuna-LLM] Fetched {len(jobs)} jobs for '{q}'")
        all_jobs.extend(jobs)

    if not all_jobs:
        print("[Pipeline/Adzuna-LLM] No jobs fetched. Exiting.")
        return

    # 2) Export to CSV for inspection / sharing
    csv_path = os.path.join(DATA_DIR, "jobs_adzuna_llm_roles.csv")
    export_jobs_to_csv(all_jobs, csv_path)

    # 3) Embeddings + store in Pinecone with namespaces per job type
    upsert_jobs_to_pinecone(all_jobs)

    print(f"[Pipeline/Adzuna-LLM] Done ✅ Total jobs fetched: {len(all_jobs)}")


if __name__ == "__main__":
    main()

