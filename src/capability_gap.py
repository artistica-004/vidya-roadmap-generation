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
    """Return the most specific domain keyword found in *identity*."""
    identity_lower = identity.lower()
    keywords = sorted(_DOMAIN_DISTANCE.keys(), key=len, reverse=True)
    for kw in keywords:
        if kw in identity_lower:
            return kw
    return "software"


def _infer_market_tier(identity: str) -> int:
    """
    Estimate a numeric market-recognised tier (0–6) for *identity*.

    0 = no professional identity (student)
    1 = intern / trainee
    2 = junior / fresher
    3 = working professional (engineer / developer)
    4 = senior
    5 = lead / staff / principal / architect
    6 = expert / fellow
    """
    identity_lower = identity.lower()

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


def _recommend_milestones(gap_score: float) -> int:
    if gap_score <= 0.25:
        return 2
    if gap_score <= 0.45:
        return 3
    if gap_score <= 0.65:
        return 4
    if gap_score <= 0.80:
        return 5
    if gap_score <= 0.90:
        return 6
    return 7


def _recommend_modules(gap_score: float) -> int:
    """Wider gaps justify more modules per milestone."""
    if gap_score <= 0.25:
        return 2
    if gap_score <= 0.50:
        return 2
    if gap_score <= 0.70:
        return 3
    return 4


def _recommend_skill_density(gap_score: float, weekly_hours: int) -> int:
    """Larger gaps + more time → higher skill density."""
    base = 3
    if gap_score > 0.50:
        base += 1
    if gap_score > 0.75:
        base += 1
    if weekly_hours >= 15:
        base += 1
    if weekly_hours >= 25:
        base += 1
    return min(base, 8)


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

    # ── Composite gap score (0.0 – 1.0) ────────────────────────
    raw = id_dist + exp_mod
    raw = raw * time_mod
    raw = raw + skill_mod
    gap_score = max(0.0, min(round(raw, 3), 1.0))

    # ── Formula debug output ──────────────────────────────────
    raw_before_time = id_dist + exp_mod
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
    print(f"  identity_distance   (id_dist)               = {id_dist:.3f}")
    print(f"  experience_modifier (exp_mod)               = {exp_mod:.3f}")
    print(f"  -------------------------------------------------")
    print(f"  id_dist + exp_mod                           = {raw_before_time:.3f}")
    print(f"  * time_mod           (time_mod)             = {time_mod:.3f}")
    print(f"  -------------------------------------------------")
    print(f"  (id_dist + exp_mod) * time_mod              = {raw_after_time:.3f}")
    print(f"  + skill_mod          (skill_mod)            = {skill_mod:.3f}")
    print(f"  -------------------------------------------------")
    print(f"  gap_score                                    = {gap_score:.3f}")
    print(f"")

    # ── Recommendations ────────────────────────────────────────
    recommended_milestones = _recommend_milestones(gap_score)
    recommended_modules = _recommend_modules(gap_score)
    recommended_density = _recommend_skill_density(gap_score, weekly_hours_available)

    # ── Reasoning ──────────────────────────────────────────────
    parts = [
        f"identity_distance={id_dist:.2f}",
        f"experience_mod={exp_mod:.2f}",
        f"time_mod={time_mod:.2f}",
        f"skill_mod={skill_mod:.2f}",
        f"gap_score={gap_score:.2f}",
    ]
    reasoning = (
        f"Gap analysis: {'; '.join(parts)}. "
        f"Recommend {recommended_milestones} milestone(s), "
        f"{recommended_modules} module(s) per milestone, "
        f"{recommended_density} skill(s) per module."
    )

    return {
        "gap_score": gap_score,
        "recommended_milestones": recommended_milestones,
        "recommended_modules_per_milestone": recommended_modules,
        "recommended_skill_density": recommended_density,
        "reasoning": reasoning,
    }
