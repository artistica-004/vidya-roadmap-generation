# src/job_fit.py
from typing import Dict, List, Set, Literal
from .skills_utils import extract_skills

JobClass = Literal["eligible", "near_eligible", "not_recommended"]


# Very simple course mapping – placeholder for your actual course system
COURSE_RECOMMENDATIONS: Dict[str, List[Dict]] = {
    "sql": [
        {
            "id": "COURSE_SQL_01",
            "title": "SQL for Data Analysis",
            "provider": "Internal / Coursera-like",
        }
    ],
    "power bi": [
        {
            "id": "COURSE_PBI_01",
            "title": "Power BI Dashboards for Beginners",
            "provider": "Internal",
        }
    ],
    "python": [
        {
            "id": "COURSE_PY_01",
            "title": "Python Programming for Data Science",
            "provider": "Internal",
        }
    ],
    "tableau": [
        {
            "id": "COURSE_TAB_01",
            "title": "Data Visualization with Tableau",
            "provider": "Internal",
        }
    ],
}


def compute_job_fit(
    user_profile_text: str,
    job_title: str,
    job_description: str,
    similarity_score: float,
) -> Dict:
    """
    Compute job-fit score and classify job into:
    - eligible
    - near_eligible
    - not_recommended

    Uses:
    - embedding similarity (from Pinecone)
    - skill overlap between user and job description
    """
    # Extract skills
    user_skills: Set[str] = extract_skills(user_profile_text)
    job_skills: Set[str] = extract_skills(job_title + " " + job_description)

    if not job_skills:
        # if no skills found in job text, fallback minimal
        job_skills = set()

    common_skills = user_skills & job_skills
    missing_skills = job_skills - user_skills

    overlap_ratio = len(common_skills) / max(len(job_skills), 1)

    # Simple combined job-fit score: 70% similarity + 30% skill overlap
    job_fit_score = 0.7 * similarity_score + 0.3 * overlap_ratio

    # Classification thresholds – heuristic, adjust later
    if job_fit_score >= 0.5 and len(missing_skills) <= 5:
        classification: JobClass = "eligible"
    elif job_fit_score >= 0.45 and overlap_ratio >= 0.2:
        classification = "near_eligible"
    else:
        classification = "not_recommended"

    # Map missing skills to courses
    course_suggestions: Dict[str, List[Dict]] = {}
    for skill in missing_skills:
        if skill in COURSE_RECOMMENDATIONS:
            course_suggestions[skill] = COURSE_RECOMMENDATIONS[skill]

    return {
        "user_skills": sorted(user_skills),
        "job_skills": sorted(job_skills),
        "common_skills": sorted(common_skills),
        "missing_skills": sorted(missing_skills),
        "overlap_ratio": overlap_ratio,
        "similarity_score": similarity_score,
        "job_fit_score": job_fit_score,
        "classification": classification,
        "course_suggestions": course_suggestions,
    }
