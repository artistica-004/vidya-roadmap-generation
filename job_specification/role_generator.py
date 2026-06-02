# src/role_generator.py
from typing import List
from langchain_openai import ChatOpenAI
from .config import OPENAI_MODEL, OPENAI_API_KEY


def get_job_roles(goal_description: str, n_roles: int = 10) -> List[str]:
    """
    Uses an LLM (via LangChain) to generate a list of job titles
    based on a high-level description of the system.
    """
    if not OPENAI_API_KEY:
        raise ValueError("OPENAI_API_KEY not set in .env")

    llm = ChatOpenAI(model=OPENAI_MODEL, temperature=0.3)

    prompt = f"""
You are an expert job-market analyst for India.

Task:
Given the following high-level description of a job recommendation system, 
generate {n_roles} concrete job titles / role keywords that should be used 
to search job APIs.

The roles should cover a mix of:
- software engineering
- data and analytics
- machine learning / AI
- cloud / DevOps
- testing / QA
- business / product / tech-ops roles

Goal/System Description:
{goal_description}

Output format:
Return ONLY a numbered list of job titles, one per line.
Do not add any explanation, commentary, or extra text.
Example:
1. Data Analyst
2. Software Engineer
3. Machine Learning Engineer
...
"""

    # Call the LLM with a simple string prompt
    resp = llm.invoke(prompt)
    text = resp.content

    roles: List[str] = []

    for line in text.splitlines():
        line = line.strip()
        if not line:
            continue
        # Strip leading numbering like "1.", "1)", "1 -"
        while line and (line[0].isdigit() or line[0] in ".-)"):
            line = line[1:].lstrip()
        if line:
            roles.append(line)

    print("[RoleGenerator] Generated roles:", roles)
    return roles

