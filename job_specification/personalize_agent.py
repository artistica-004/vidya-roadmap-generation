#src/personalize_agent.py
from typing import List, Dict
from langchain_openai import ChatOpenAI
from .config import OPENAI_MODEL, OPENAI_API_KEY, RESUME_INDEX_NAME
from .job_retriever import search_jobs_for_resume_vector
from .job_fit import compute_job_fit


def build_enriched_jobs(profile_text: str, raw_jobs: List[Dict]) -> List[Dict]:
    """
    Add job-fit metrics, classification, and course recommendations
    to each retrieved job.
    """
    enriched: List[Dict] = []
    for j in raw_jobs:
        fit_info = compute_job_fit(
            user_profile_text=profile_text,
            job_title=j["title"] or "",
            job_description=j["description"] or "",
            similarity_score=j["score"],
        )
        enriched.append({**j, **fit_info})
    return enriched


def split_by_classification(jobs: List[Dict]):
    eligible = []
    near_eligible = []
    not_recommended = []
    for j in jobs:
        cls = j.get("classification")
        if cls == "eligible":
            eligible.append(j)
        elif cls == "near_eligible":
            near_eligible.append(j)
        else:
            not_recommended.append(j)
    return eligible, near_eligible, not_recommended


def format_jobs_for_prompt(jobs: List[Dict]) -> str:
    lines = []
    for idx, j in enumerate(jobs, start=1):
        line = (
            f"{idx}. {j['title']} at {j['company']} "
            f"({j['location']}) [namespace: {j['namespace']}] "
            f"- job_fit_score: {j['job_fit_score']:.3f}, "
            f"similarity: {j['similarity_score']:.3f}\n"
            f"   URL: {j['url']}\n"
            f"   Job skills: {', '.join(j.get('job_skills', []))}\n"
            f"   User skills: {', '.join(j.get('user_skills', []))}\n"
            f"   Missing skills: {', '.join(j.get('missing_skills', []))}\n"
        )
        lines.append(line)
    return "\n".join(lines)


def format_courses_for_prompt(near_eligible_jobs: List[Dict]) -> str:
    lines = []
    for j in near_eligible_jobs:
        if not j.get("course_suggestions"):
            continue
        lines.append(f"Job: {j['title']} at {j['company']} ({j['location']})")
        for skill, courses in j["course_suggestions"].items():
            for c in courses:
                lines.append(
                    f"  Skill: {skill} -> Course: {c['title']} "
                    f"(id: {c['id']}, provider: {c['provider']})"
                )
    return "\n".join(lines)


def generate_personalised_recommendation(
    profile_text: str,
    eligible_jobs: List[Dict],
    near_eligible_jobs: List[Dict],
) -> str:
    """
    LLM "agent" that takes:
    - user profile/resume summary
    - eligible jobs and near-eligible jobs (with missing skills + course mapping)
    and produces a human-readable personalised recommendation.
    """
    if not OPENAI_API_KEY:
        raise ValueError("OPENAI_API_KEY not set in .env")

    llm = ChatOpenAI(model=OPENAI_MODEL, temperature=0.4)

    # 🔹 Don't show missing skills in Eligible jobs
    sanitized_eligible_jobs: List[Dict] = []
    for j in eligible_jobs:
        j_copy = dict(j)
        j_copy["missing_skills"] = []
        sanitized_eligible_jobs.append(j_copy)

    eligible_block = format_jobs_for_prompt(sanitized_eligible_jobs)
    near_block = format_jobs_for_prompt(near_eligible_jobs)
    courses_block = format_courses_for_prompt(near_eligible_jobs)

    prompt = f"""
You are an AI career assistant helping a user find suitable jobs and upskilling paths.

USER PROFILE:
\"\"\"{profile_text}\"\"\"


ELIGIBLE JOBS (user currently fits these based on embeddings + skill overlap):
{eligible_block if eligible_block else "None"}


NEAR-ELIGIBLE JOBS (good similarity but missing some skills):
{near_block if near_block else "None"}


COURSE SUGGESTIONS (for missing skills in near-eligible jobs):
{courses_block if courses_block else "None"}


Tasks:
1. Summarize the user's current fit: how many eligible and near-eligible jobs did we find?
2. For the Eligible jobs, list 3–5 top roles with short explanations (mention overlapping skills).
3. For the Near-Eligible jobs, explain which key skills are missing and how completing the suggested courses would make the user eligible.
4. Use clear headings like:
   - "Current Eligible Roles"
   - "Near-Eligible Roles & Required Skills"
   - "Recommended Courses to Become Eligible"
5. Keep the explanation concise and practical. Do not mention embeddings or internal scores directly.

Return the explanation as markdown-style text.
"""

    resp = llm.invoke(prompt)
    return resp.content


def main():
    # ✅ This should be the id of the resume vector already stored in your RESUME_INDEX_NAME
    resume_id = "PUT_YOUR_RESUME_VECTOR_ID_HERE"

    # This is a text summary of the resume used for skill extraction & explanation.
    # In your real pipeline, this should come from your resume parser (skills + summary).
    profile_text = (
        "User wants a Data Analyst role. Has 2 years of experience with Excel and Power BI, "
        "basic Python, but no strong SQL background yet. Worked on dashboards and reporting, "
        "but has not done heavy database querying."
    )

    print(
        f"[Agent] Using resume vector id='{resume_id}' "
        f"from resume index='{RESUME_INDEX_NAME}'"
    )

    # 1) Retrieve top jobs from JOB index using the resume embedding (from resume index)
    raw_jobs = search_jobs_for_resume_vector(
        resume_id="640118bf-e925-47d5-94fc-b1d54705f0d1",
        top_k=15,
        namespaces=None,  # or pass specific namespaces if you want
    )
    if not raw_jobs:
        print("[Agent] No jobs retrieved for this resume id.")
        return

    # 2) Compute job-fit metrics and classification
    enriched_jobs = build_enriched_jobs(profile_text, raw_jobs)
    eligible, near_eligible, not_recommended = split_by_classification(enriched_jobs)

    print(f"[Agent] Eligible jobs: {len(eligible)}, Near-eligible: {len(near_eligible)}")

    # 3) Ask LLM to generate personalised explanation with course mapping
    recommendation_text = generate_personalised_recommendation(
        profile_text, eligible, near_eligible
    )

    print("\n================= PERSONALISED JOB RECOMMENDATIONS =================\n")
    print(recommendation_text)


if __name__ == "__main__":
    main()



