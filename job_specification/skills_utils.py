# src/skills_utils.py
from typing import Set

# Very simple skill vocabulary – extend as needed
SKILL_KEYWORDS = {
    "python",
    "r",
    "sql",
    "power bi",
    "tableau",
    "excel",
    "pandas",
    "numpy",
    "scikit-learn",
    "machine learning",
    "deep learning",
    "statistics",
    "azure",
    "aws",
    "gcp",
    "spark",
    "hadoop",
    "java",
    "c++",
    "javascript",
    "react",
    "node.js",
    "django",
    "flask",
    "docker",
    "kubernetes",
    "git",
    "linux",
    "etl",
    "powerpoint",
    "snowflake",
    "bigquery",
}


def extract_skills(text: str) -> Set[str]:
    """
    Very naive skill extraction:
    lowercases text and checks for presence of known skill keywords.
    """
    text_lower = text.lower()
    found = set()
    for skill in SKILL_KEYWORDS:
        if skill in text_lower:
            found.add(skill)
    return found
