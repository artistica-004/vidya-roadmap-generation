"""
Capability Gap Analysis Module

Computes roadmap complexity BEFORE calling the LLM.

Provides a single entry point:

    compute_gap_score(
        current_identity: str,
        target_identity: str,
        years_experience: int,
        weekly_hours_available: int,
        timeline_days: int,
        known_skills: list[str],
    ) -> dict

The returned dict contains:

    gap_score                    float  0.0–1.0
    recommended_milestones       int    2–7

    recommended_modules_per_milestone  int  2–4
    recommended_skill_density    int    3–8
    reasoning                    str    human-readable explanation
"""

import math
import re
from typing import List, Dict, Tuple

# Skill density bounds — single source of truth matching roadmap_agent.py constants
MIN_SKILLS = 3
MAX_SKILLS = 8

# ── Role families ────────────────────────────────────────────────
# Roles within the same family are considered adjacent; identity
# distance between them is capped at 0.35 regardless of title wording.
_ROLE_FAMILIES: Dict[str, List[str]] = {
    "data": [
        "data analyst", "data scientist", "data engineer", "ml engineer",
        "machine learning engineer", "bi analyst", "analytics engineer",
        "data architect", "data strategist", "data operations",
        "analytics manager", "business intelligence",
    ],
    "backend": [
        "backend engineer", "backend developer", "software engineer",
        "software developer", "api engineer", "systems engineer",
        "server engineer", "back end engineer",
    ],
    "frontend": [
        "frontend engineer", "frontend developer", "ui engineer",
        "ui developer", "web developer", "front end engineer",
    ],
    "fullstack": [
        "full stack", "fullstack engineer", "fullstack developer",
        "full stack developer",
    ],
    "devops": [
        "devops engineer", "platform engineer", "sre",
        "site reliability engineer", "infrastructure engineer",
        "cloud engineer", "release engineer",
    ],
    "product": [
        "product analyst", "product manager", "product owner",
        "program manager", "technical program manager",
    ],
    "data_science": [
        "data scientist", "ml engineer", "machine learning engineer",
        "research scientist", "ai engineer", "ai researcher",
    ],
    "design": [
        "ui designer", "ux designer", "product designer",
        "visual designer", "interaction designer",
    ],
    "mobile": [
        "android engineer", "android developer", "ios engineer",
        "ios developer", "mobile engineer", "mobile developer",
    ],
    "security": [
        "security engineer", "cybersecurity analyst",
        "security analyst", "penetration tester",
    ],
}


def _check_role_family(current: str, target: str) -> Tuple[bool, str]:
    """Return (same_family, family_name) for a pair of role titles."""
    cur_lower = current.lower().strip()
    tgt_lower = target.lower().strip()
    for family, keywords in _ROLE_FAMILIES.items():
        cur_match = any(kw in cur_lower for kw in keywords)
        tgt_match = any(kw in tgt_lower for kw in keywords)
        if cur_match and tgt_match:
            return True, family
    return False, ""


# ── Domain distance map ─────────────────────────────────────────
# 0.0 = same domain, 0.5 = completely unrelated
_DOMAIN_DISTANCE: Dict[str, Dict[str, float]] = {
    "software": {
        "software": 0.0,
        "backend": 0.1,
        "frontend": 0.2,
        "fullstack": 0.1,
        "mobile": 0.25,
        "devops": 0.2,
        "data": 0.3,
        "ai": 0.35,
        "embedded": 0.3,
        "security": 0.25,
    },
    "backend": {
        "backend": 0.0,
        "software": 0.1,
        "fullstack": 0.15,
        "devops": 0.15,
        "data": 0.3,
        "ai": 0.35,
        "frontend": 0.4,
        "mobile": 0.4,
    },
    "frontend": {
        "frontend": 0.0,
        "software": 0.2,
        "fullstack": 0.15,
        "mobile": 0.2,
        "backend": 0.4,
    },
    "data": {
        "data": 0.0,
        "ai": 0.15,
        "software": 0.3,
        "backend": 0.3,
        "devops": 0.35,
    },
    "ai": {
        "ai": 0.0,
        "data": 0.15,
        "software": 0.35,
        "backend": 0.35,
        "mlops": 0.15,
    },
    "devops": {
        "devops": 0.0,
        "software": 0.2,
        "backend": 0.15,
        "data": 0.35,
        "ai": 0.35,
        "security": 0.2,
    },
    "mobile": {
        "mobile": 0.0,
        "software": 0.25,
        "frontend": 0.2,
        "backend": 0.4,
    },
    "security": {
        "security": 0.0,
        "software": 0.25,
        "devops": 0.2,
    },
    "student": {
        "student": 0.0,
    },
}


def _extract_domain(identity: str) -> str:
    """Return the most specific domain keyword found in *identity*.

    Uses word-boundary matching to avoid false positives
    (e.g. 'ai' in 'retail').
    """
    identity_lower = identity.lower()
    keywords = sorted(_DOMAIN_DISTANCE.keys(), key=len, reverse=True)
    for kw in keywords:
        if re.search(r'\b' + re.escape(kw) + r'\b', identity_lower):
            return kw
    return "software"


# ── Non-technical / non-professional role keywords ──────────────
# Roles matching these have no professional identity in tech.
_NON_TECHNICAL_KEYWORDS = [
    "student", "college", "undergrad", "graduate",
    "operator", "associate", "retail", "cafe", "clerk",
    "helper", "sales assistant", "attendant", "driver", "cleaner",
    "receptionist", "cashier", "barista", "waiter", "waitress",
    "labour", "warehouse", "factory", "construction",
    # NOT "helpdesk", "technician", "support" — those are entry-level tech
]


def _is_non_technical(identity: str) -> bool:
    """Return True if *identity* is a non-technical / non-professional role."""
    identity_lower = identity.lower()
    for kw in _NON_TECHNICAL_KEYWORDS:
        if re.search(r'\b' + re.escape(kw) + r'\b', identity_lower):
            return True
    return False


def _infer_market_tier(identity: str) -> int:
    """
    Estimate a numeric market-recognised tier (0–6) for *identity*.

    0 = no professional identity (student / non-technical)
    1 = intern / trainee
    2 = junior / fresher
    3 = working professional (engineer / developer)
    4 = senior
    5 = lead / staff / principal / architect
    6 = expert / fellow
    """
    identity_lower = identity.lower()

    # Non-technical roles get tier 0 (same as student)
    if _is_non_technical(identity):
        return 0

    # Entry-level support / technician roles → tier 1
    if re.search(r'\b(helpdesk|desktop support|it support|technician|support engineer|support analyst)\b', identity_lower):
        return 1

    # Check seniority *prefixes* before the role keyword itself so that
    # "Junior Developer" → tier 2, not tier 3 (developer).
    if re.search(r'\b(junior|jr\.?|fresher|entry.level)\b', identity_lower):
        return 2
    if re.search(r'\bsenior\b', identity_lower):
        return 4
    if re.search(r'\b(lead|staff|principal|architect|head.of)\b', identity_lower):
        return 5
    if re.search(r'\b(expert|fellow)\b', identity_lower):
        return 6
    if re.search(r'\b(student|college|undergrad|graduate)\b', identity_lower):
        return 0
    if re.search(r'\b(intern|trainee|apprentice)\b', identity_lower):
        return 1
    if re.search(r'\b(developer|engineer|dev|engineer|mid)\b', identity_lower):
        return 3
    return 3


def _compute_identity_distance(current: str, target: str) -> float:
    """
    Return a float 0.0–1.0 representing the distance between two identities.

    Factors:
      - Domain distance (same field vs cross-field)
      - Seniority gap (larger jump = larger distance)
      - Role-family cap: same-family roles never exceed 0.35
    """
    if not current or not target:
        return 0.6

    same_family, family_name = _check_role_family(current, target)

    cur_domain = _extract_domain(current)
    tgt_domain = _extract_domain(target)
    domain_dist = _DOMAIN_DISTANCE.get(cur_domain, {}).get(tgt_domain, 0.4)

    cur_tier = _infer_market_tier(current)
    tgt_tier = _infer_market_tier(target)
    tier_gap = max(tgt_tier - cur_tier, 0) / 6.0  # normalise to 0–1

    # Blend: 40 % domain, 60 % seniority
    raw = domain_dist * 0.4 + tier_gap * 0.6
    if same_family:
        raw = min(raw, 0.35)

    result = min(raw, 1.0)

    # ── Debug output ─────────────────────────────────────────
    print(f"[ROLE MATCH]")
    print(f"  current_role={current}")
    print(f"  target_role={target}")
    print(f"  same_family={'True' if same_family else 'False'}"
          f"{' (' + family_name + ')' if same_family else ''}")
    print(f"  cur_domain={cur_domain}  tgt_domain={tgt_domain}  domain_dist={domain_dist}")
    print(f"  cur_tier={cur_tier}  tgt_tier={tgt_tier}  tier_gap={tier_gap:.3f}")
    print(f"  identity_distance={result:.2f}"
          f"{'  (capped by same-family rule)' if same_family and raw > 0.35 else ''}")

    return result


def _experience_modifier(years: int) -> float:
    """
    More experience → smaller effective gap.

    0 years  → +0.30
    1–3      → +0.20
    4+       → +0.10
    """
    if years <= 0:
        return 0.30
    if years <= 3:
        return 0.20
    return 0.10


def _time_modifier(hours_per_week: int, timeline_days: int) -> float:
    """
    More available hours → smaller effective gap (you can cover more ground).

    Base 1.0 at 5 h/week / 112 days.
    """
    budget_ratio = (hours_per_week * timeline_days) / (5 * 112)
    if budget_ratio >= 4.0:
        return 0.75
    if budget_ratio >= 2.0:
        return 0.85
    if budget_ratio >= 1.0:
        return 1.0
    return 1.15


def _known_skills_modifier(known_skills: List[str]) -> float:
    """
    Each known skill reduces the effective gap slightly.
    """
    count = len([s for s in known_skills if s.strip()])
    if count == 0:
        return 0.0
    return -min(count * 0.04, 0.25)


# ── Domain-role skill relevance map ─────────────────────────────
# Skills relevant to each target domain for readiness scoring.
_DOMAIN_RELEVANT_SKILLS = {
    "backend": {"java", "python", "spring", "django", "node", "express", "sql", "postgres", "mysql", "mongodb", "redis", "docker", "kubernetes", "rest", "api", "microservices", "aws", "cloud", "linux"},
    "frontend": {"html", "css", "javascript", "typescript", "react", "angular", "vue", "svelte", "webpack", "babel", "responsive", "ui", "ux", "figma"},
    "fullstack": {"java", "python", "javascript", "typescript", "react", "node", "spring", "django", "sql", "mongodb", "html", "css", "rest", "api", "docker", "aws"},
    "data": {"sql", "python", "r", "excel", "tableau", "powerbi", "pandas", "numpy", "scikit", "spark", "hadoop", "statistics", "probability", "linear_regression", "classification", "clustering"},
    "ai": {"python", "tensorflow", "pytorch", "machine_learning", "deep_learning", "nlp", "computer_vision", "transformers", "llm", "rag", "statistics", "probability", "linear_algebra", "calculus"},
    "devops": {"docker", "kubernetes", "jenkins", "gitlab", "terraform", "ansible", "aws", "gcp", "azure", "linux", "bash", "monitoring", "prometheus", "grafana", "cicd"},
    "mobile": {"kotlin", "swift", "android", "ios", "flutter", "react_native", "dart", "xcode"},
    "security": {"network", "firewall", "encryption", "authentication", "penetration", "vulnerability", "siem", "ids", "ips", "compliance", "risk"},
}


def _compute_career_transition_penalty(current_identity: str, target_identity: str) -> float:
    """Penalty when moving from non-technical to technical role.

    0.0 = no penalty (already in tech)
    0.30 = non-technical → technical
    """
    if _is_non_technical(current_identity) and not _is_non_technical(target_identity):
        return 0.30
    return 0.0


def _compute_foundation_penalty(years_experience: int, known_skills: list) -> float:
    """Penalty for beginners with low experience and few relevant skills.

    0.0 = experienced professional
    0.25 = beginner (0-1yr) with no foundation skills
    """
    if years_experience > 1:
        return 0.0
    foundation_count = sum(1 for s in known_skills if s.lower().strip() in {"python", "sql", "html", "css", "javascript", "linux", "git"})
    if foundation_count < 1:
        return 0.25
    if foundation_count < 2:
        return 0.15
    return 0.05


def _compute_domain_switch_penalty(current_identity: str, target_identity: str) -> float:
    """Penalty when switching to an unrelated domain.

    0.0 = same domain family
    0.20 = different domain
    """
    same_family, _ = _check_role_family(current_identity, target_identity)
    cur_domain = _extract_domain(current_identity)
    tgt_domain = _extract_domain(target_identity)
    if same_family or cur_domain == tgt_domain:
        return 0.0
    return 0.20


def _compute_skill_readiness(known_skills: list, target_identity: str) -> float:
    """Score 0.0–1.0 how well known skills support the target role.

    1.0 = every skill is relevant to the target domain
    0.0 = no relevant skills
    """
    if not known_skills:
        return 0.0
    tgt_domain = _extract_domain(target_identity)
    relevant = _DOMAIN_RELEVANT_SKILLS.get(tgt_domain, set())
    if not relevant:
        return 0.3  # unknown domain — moderate readiness
    total = len(known_skills)
    relevant_count = sum(1 for s in known_skills if _skill_matches_domain(s, relevant))
    return min(relevant_count / max(total, 1), 1.0)


def _skill_matches_domain(skill: str, relevant_set: set) -> bool:
    """Check if a skill string matches any keyword in the relevant set.

    Uses word-boundary matching for short keywords (≤3 chars) to avoid
    false positives like 'r' matching 'customer service'.
    """
    skill_lower = skill.lower().strip()
    for kw in relevant_set:
        if len(kw) <= 3:
            if re.search(r'\b' + re.escape(kw) + r'\b', skill_lower):
                return True
        else:
            if kw in skill_lower or skill_lower in kw:
                return True
    return False


def _recommend_milestones(gap_score: float) -> tuple:
    """Return (max_milestones, rationale). Count is a bound, not a target."""
    if gap_score <= 0.25:
        return 2, f"gap_score={gap_score:.2f} (≤0.25): small gap, at most 2 milestones"
    if gap_score <= 0.45:
        return 3, f"gap_score={gap_score:.2f} (≤0.45): moderate gap, at most 3 milestones"
    if gap_score <= 0.65:
        return 4, f"gap_score={gap_score:.2f} (≤0.65): notable gap, at most 4 milestones"
    if gap_score <= 0.80:
        return 5, f"gap_score={gap_score:.2f} (≤0.80): large gap, at most 5 milestones"
    if gap_score <= 0.90:
        return 6, f"gap_score={gap_score:.2f} (≤0.90): very large gap, at most 6 milestones"
    return 7, f"gap_score={gap_score:.2f} (>0.90): extreme gap, at most 7 milestones"


def _recommend_modules(gap_score: float) -> tuple:
    """Return (max_modules, rationale). Count is a bound, not a target."""
    if gap_score <= 0.25:
        return 2, f"gap_score={gap_score:.2f}: small gap, at most 2 modules per milestone"
    if gap_score <= 0.50:
        return 2, f"gap_score={gap_score:.2f} (≤0.50): moderate gap, at most 2 modules"
    if gap_score <= 0.70:
        return 3, f"gap_score={gap_score:.2f} (≤0.70): notable gap, at most 3 modules"
    return 4, f"gap_score={gap_score:.2f} (>0.70): wide gap, at most 4 modules"


def _recommend_skill_density(gap_score: float, weekly_hours: int) -> tuple:
    """Return (max_skills, rationale). Count is a bound, not a target.

    Aligned with MIN_SKILLS=3 / MAX_SKILLS=8 from roadmap_agent.py constants.
    """
    base = MIN_SKILLS  # start from shared constant
    parts = [f"gap_score={gap_score:.2f}"]
    if gap_score > 0.50:
        base += 1
        parts.append("gap>0.50 → +1")
    if gap_score > 0.75:
        base += 1
        parts.append("gap>0.75 → +1")
    if weekly_hours >= 15:
        base += 1
        parts.append(f"hours={weekly_hours}≥15 → +1")
    if weekly_hours >= 25:
        base += 1
        parts.append(f"hours={weekly_hours}≥25 → +1")
    max_skills = min(base, MAX_SKILLS)
    parts.append(f"max={max_skills}")
    return max_skills, "; ".join(parts)


def compute_gap_score(
    current_identity: str = "",
    target_identity: str = "",
    years_experience: int = 0,
    weekly_hours_available: int = 5,
    timeline_days: int = 112,
    known_skills: List[str] = None,
) -> Dict:
    """
    Compute roadmap complexity before calling the LLM.

    Parameters
    ----------
    current_identity : str
        Learner's current professional identity (e.g. "Student", "Backend Engineer").
    target_identity : str
        Desired professional identity after the roadmap.
    years_experience : int
        Years of professional experience.
    weekly_hours_available : int
        Hours per week the learner can dedicate.
    timeline_days : int
        Target duration in days.
    known_skills : list of str, optional
        Skills the learner already possesses.

    Returns
    -------
    dict with keys:
        gap_score                     float   0.0–1.0
        recommended_milestones        int     2–7
        recommended_modules_per_milestone  int  2–4
        recommended_skill_density     int     3–8
        skill_readiness_score         float   0.0–1.0
        beginner_penalties            dict    breakdown of added penalties
        reasoning                     str
    """
    if known_skills is None:
        known_skills = []

    # ── Component scores ───────────────────────────────────────
    id_dist = _compute_identity_distance(current_identity, target_identity)
    same_family, family_name = _check_role_family(current_identity, target_identity)
    exp_mod = _experience_modifier(years_experience)
    time_mod = _time_modifier(weekly_hours_available, timeline_days)
    skill_mod = _known_skills_modifier(known_skills)

    # ── Beginner / transition penalties (Phase 3.3) ────────────
    career_penalty = _compute_career_transition_penalty(current_identity, target_identity)
    foundation_penalty = _compute_foundation_penalty(years_experience, known_skills)
    domain_penalty = _compute_domain_switch_penalty(current_identity, target_identity)

    # ── Skill readiness score ──────────────────────────────────
    skill_readiness = _compute_skill_readiness(known_skills, target_identity)
    readiness_penalty = (1.0 - skill_readiness) * 0.20  # 0.00–0.20

    # ── Zero-experience penalty ────────────────────────────────
    zero_exp_penalty = 0.15 if years_experience == 0 else 0.0

    # ── Composite gap score (0.0 – 1.0) ────────────────────────
    base_penalties = career_penalty + foundation_penalty + domain_penalty + readiness_penalty + zero_exp_penalty
    raw = id_dist + exp_mod + base_penalties
    raw = raw * time_mod
    raw = raw + skill_mod
    gap_score = max(0.0, min(round(raw, 3), 1.0))

    # ── Formula debug output ──────────────────────────────────
    raw_before_time = id_dist + exp_mod + base_penalties
    raw_after_time = raw_before_time * time_mod
    raw_after_skill = raw_after_time + skill_mod
    print(f"")
    print(f"[GAP COMPUTATION]")
    print(f"  current_role={current_identity}")
    print(f"  target_role={target_identity}")
    if same_family:
        print(f"  role_family={family_name}  (same-family cap=0.35)")
    else:
        print(f"  role_family=different  (no cap)")
    print(f"  identity_distance_source = domain_dist blended with tier_gap")
    print(f"")
    print(f"  identity_distance     (id_dist)             = {id_dist:.3f}")
    print(f"  experience_modifier   (exp_mod)             = {exp_mod:.3f}")
    print(f"  career_penalty                              = {career_penalty:.3f}")
    print(f"  foundation_penalty                          = {foundation_penalty:.3f}")
    print(f"  domain_penalty                              = {domain_penalty:.3f}")
    print(f"  readiness_penalty                           = {readiness_penalty:.3f}")
    print(f"  zero_exp_penalty                            = {zero_exp_penalty:.3f}")
    print(f"  -------------------------------------------------")
    print(f"  id_dist + exp_mod + penalties               = {raw_before_time:.3f}")
    print(f"  * time_mod           (time_mod)             = {time_mod:.3f}")
    print(f"  -------------------------------------------------")
    print(f"  (id_dist + exp_mod + penalties) * time_mod  = {raw_after_time:.3f}")
    print(f"  + skill_mod          (skill_mod)            = {skill_mod:.3f}")
    print(f"  -------------------------------------------------")
    print(f"  gap_score                                    = {gap_score:.3f}")
    print(f"")

    # ── Beginner audit output ──────────────────────────────────
    print("========== BEGINNER AUDIT ==========")
    print(f"  current_role              = {current_identity}")
    print(f"  target_role               = {target_identity}")
    print(f"  years_experience          = {years_experience}")
    print(f"  known_skills              = {known_skills}")
    print(f"  gap_score                 = {gap_score:.3f}")
    print(f"  skill_readiness_score     = {skill_readiness:.3f}")
    is_non_tech = _is_non_technical(current_identity)
    is_beginner = years_experience <= 1
    print(f"  is_non_technical          = {is_non_tech}")
    print(f"  is_beginner               = {is_beginner}")
    print(f"  career_family_change       = {not same_family}")
    print(f"  career_reset_penalty       = {career_penalty:.3f}")
    print(f"  foundation_penalty_added   = {foundation_penalty:.3f}")
    print(f"  domain_switch_penalty      = {domain_penalty:.3f}")
    print(f"  zero_experience_penalty    = {zero_exp_penalty:.3f}")
    print("===================================")

    # ── Recommendations ────────────────────────────────────────
    recommended_milestones, ms_rationale = _recommend_milestones(gap_score)
    recommended_modules, mod_rationale = _recommend_modules(gap_score)
    recommended_density, skill_rationale = _recommend_skill_density(gap_score, weekly_hours_available)

    # Low-ICP override: beginners get at least 3 milestones
    if is_beginner or is_non_tech:
        if recommended_milestones < 3:
            recommended_milestones = 3
            ms_rationale += "; low-ICP beginner override → min 3"

    # ── Reasoning ──────────────────────────────────────────────
    parts = [
        f"identity_distance={id_dist:.2f}",
        f"experience_mod={exp_mod:.2f}",
        f"career_penalty={career_penalty:.2f}",
        f"foundation_penalty={foundation_penalty:.2f}",
        f"domain_penalty={domain_penalty:.2f}",
        f"readiness_penalty={readiness_penalty:.2f}",
        f"zero_exp_penalty={zero_exp_penalty:.2f}",
        f"skill_mod={skill_mod:.2f}",
        f"gap_score={gap_score:.2f}",
    ]
    reasoning = (
        f"Gap analysis: {'; '.join(parts)}. "
        f"Milestones: {ms_rationale}. "
        f"Modules: {mod_rationale}. "
        f"Skills: {skill_rationale}."
    )

    return {
        "gap_score": gap_score,
        "recommended_milestones": recommended_milestones,
        "recommended_milestones_rationale": ms_rationale,
        "recommended_modules_per_milestone": recommended_modules,
        "recommended_modules_per_milestone_rationale": mod_rationale,
        "recommended_skill_density": recommended_density,
        "recommended_skill_density_rationale": skill_rationale,
        "skill_readiness_score": round(skill_readiness, 3),
        "beginner_penalties": {
            "career_transition": round(career_penalty, 3),
            "foundation": round(foundation_penalty, 3),
            "domain_switch": round(domain_penalty, 3),
            "readiness": round(readiness_penalty, 3),
            "zero_experience": round(zero_exp_penalty, 3),
        },
        "reasoning": reasoning,
    }


# ── Role-specific domain maps ─────────────────────────────────
# Each role family maps to appropriate breadth domains.
# Analyst roles get business/analytics domains (NOT engineering).
_ROLE_DOMAIN_MAP: Dict[str, List[str]] = {
    "analyst": [
        "business_analytics", "data_visualization", "sql_analytics",
        "statistical_analysis", "reporting",
    ],
    "data": [
        "data_modeling", "statistics", "ml_pipelines", "deployment",
    ],
    "backend": [
        "distributed_systems", "api_design", "architecture",
    ],
    "fullstack": [
        "distributed_systems", "api_design", "architecture",
    ],
    "frontend": [
        "ui_architecture", "performance", "testing",
    ],
    "ai": [
        "ml_pipelines", "experimentation", "deployment",
    ],
    "devops": [
        "infrastructure", "ci_cd", "monitoring", "cloud_architecture",
    ],
    "mobile": [
        "mobile_architecture", "app_performance", "cross_platform",
    ],
    "security": [
        "security_architecture", "threat_modeling", "compliance",
    ],
    "product": [
        "product_strategy", "user_research", "metrics_analytics",
    ],
    "design": [
        "design_systems", "user_research", "prototyping",
    ],
}


def _detect_role_category(target_role: str) -> str:
    """Classify target role into a breadth domain category.

    Returns one of the keys in _ROLE_DOMAIN_MAP, or None.
    """
    tgt = target_role.lower()

    # Analyst category is a subset of data roles that should NOT
    # get engineering domains (deployment, ml_pipelines, etc.)
    _analyst_keywords = {"analyst", "analytics", "bi ", "business intelligence",
                         "reporting", "dashboard"}

    for kw in _analyst_keywords:
        if kw in tgt:
            return "analyst"

    role_categories = {
        "data": {"data", "ml", "machine learning", "data science"},
        "backend": {"backend", "back end", "server", "api"},
        "fullstack": {"fullstack", "full stack"},
        "frontend": {"frontend", "front end", "ui", "web developer"},
        "ai": {"ai", "artificial intelligence", "deep learning", "llm"},
        "devops": {"devops", "sre", "platform", "infrastructure", "cloud"},
        "mobile": {"mobile", "android", "ios", "flutter"},
        "security": {"security", "cyber"},
        "product": {"product manager", "product analyst", "program manager"},
        "design": {"designer", "ux", "ui designer"},
    }

    for category, keywords in role_categories.items():
        for kw in keywords:
            if kw in tgt:
                return category

    return None


def compute_capability_breadth(
    gap_score: float = 0.5,
    current_role: str = "",
    target_role: str = "",
    years_experience: int = 0,
    known_skills: list = None,
) -> dict:
    """Compute required capability breadth for a role transition.

    Returns:
      breadth_score: float (0.0–1.0) how broad the roadmap must be
      required_domains: list of domain labels the roadmap must cover
    """
    if known_skills is None:
        known_skills = []

    current_role = (current_role or "").lower()
    target_role = (target_role or "").lower()

    # Determine breadth based on transition type
    same_family = False
    for family_name, members in _ROLE_FAMILIES.items():
        c_match = any(m in current_role for m in members)
        t_match = any(m in target_role for m in members)
        if c_match and t_match:
            same_family = True
            break

    required_domains = []

    # Role-specific breadth domains (role-category-aware)
    category = _detect_role_category(target_role)
    if category and category in _ROLE_DOMAIN_MAP:
        required_domains.extend(_ROLE_DOMAIN_MAP[category])

    # Leadership / seniority domains
    if "senior" in target_role or "lead" in target_role or "principal" in target_role or "architect" in target_role:
        required_domains.extend(["system_design", "leadership"])
    if "lead" in target_role or "manager" in target_role:
        required_domains.append("team_management")

    # Cross-family transitions need more breadth
    if not same_family:
        required_domains.append("foundations")

    # Gap amplifies breadth with role-appropriate domains
    # (analyst roles get business_analytics_depth, NOT engineering domains)
    if gap_score > 0.65:
        if category == "analyst":
            required_domains.append("advanced_analytics")
        else:
            required_domains.append("scalability")
    if gap_score > 0.80:
        if category == "analyst":
            required_domains.append("data_strategy")
        else:
            required_domains.append("advanced_architecture")

    # Deduplicate while preserving order
    seen = set()
    required_domains = [d for d in required_domains if not (d in seen or seen.add(d))]

    breadth_score = min(0.3 + len(required_domains) * 0.1, 1.0)
    return {
        "breadth_score": round(breadth_score, 2),
        "required_domains": required_domains,
    }
