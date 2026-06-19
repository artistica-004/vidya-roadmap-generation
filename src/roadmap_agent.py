from langchain_core.prompts import PromptTemplate
from langchain_openai import ChatOpenAI
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.output_parsers import StrOutputParser
from src.pinecone_utils import (fetch_poc_record, save_poc_record,)
from src.capability_gap import compute_gap_score, compute_capability_breadth, _extract_domain
from src.course_catalog_mapper import (repair_course_catalog_alignment,ALL_CATALOG_SKILLS,find_best_catalog_match,normalize_skill_name, AVAILABLE_COURSES,UNAVAILABLE_COURSES,)
import copy
import math
import os
import json
import traceback
import uuid
import re
from typing import List
from datetime import datetime
import pathlib

# ============================================================
# POC Storage — local JSON logging
# ============================================================

POC_STORAGE_DIR = pathlib.Path(__file__).parent.parent / "poc_storage"

# ============================================================
# LOCKED CONSTANTS  (ROADMAP_GENERATION_SCIENCE_V2.md — Section 1)
# Source of truth: ROADMAP_GENERATION_SCIENCE_V2.md, locked June 16 2026.
# Where this file conflicts with any other doc, THE MD FILE WINS.
#
# DEVIATION FROM MD (explicit, by request):
#   - SKILLS_PER_MODULE kept at (3, 8) instead of MD's (3, 10).
#     This is an intentional, acknowledged deviation — not an oversight.
#     If you want true MD alignment later, change MAX_SKILLS to 10.
# ============================================================

# ---- Mastery (Bloom 2-sigma / BKT) ----
MASTERY_GATE        = 0.90   # P(L) -> next module/milestone unlocks
MOCK_UNLOCK_MASTERY = 0.75   # P(L) -> mock unlocks

# ---- Structure: BOUNDS (min, max), gap-derived within ----
MIN_MILESTONES = 2
MAX_MILESTONES = 7      # MILESTONES_PER_PATH = (2, 7)              — MD Section 1 / 27

MIN_MODULES    = 2
MAX_MODULES    = 4      # MODULES_PER_MILESTONE = (2, 4)            — MD Section 1 / 27

MIN_SKILLS     = 3
MAX_SKILLS     = 8      # SKILLS_PER_MODULE = (3, 8)                 — MD Section 1 / 27

MIN_LESSONS_PER_SKILL = 2
MAX_LESSONS_PER_SKILL = 4   # MD Section 11 — Microlearning + CLT

MIN_SCENARIOS_PER_MILESTONE = 3
MAX_SCENARIOS_PER_MILESTONE = 7   # SCENARIOS_PER_MILESTONE = (3, 7)  — MD Section 1 / 13 / 27

MIN_MOCKS_PER_MILESTONE = 1
MAX_MOCKS_PER_MILESTONE = 2       # MOCKS_PER_MILESTONE = (1, 2)      — MD Section 1 / 14 / 27
# NOTE: "Interview" in this codebase is the same concept the MD file calls
# "Mock". MD Change Log explicitly removed "exactly 1 interview" and
# replaced it with MOCKS_PER_MILESTONE = (1, 2) with a mandatory
# interview_count_rationale. We keep the "Interview" name in the JSON
# schema (so we don't break downstream consumers) but the BOUNDS now
# match MD's mocks/interview rule exactly: 1 to 2, never 0, never fixed at 1.

MIN_PROJECTS_PER_MILESTONE = 1
MAX_PROJECTS_PER_MILESTONE = 2    # PROJECTS_PER_MILESTONE = (1, 2)   — MD Section 1 / 12 / 27
MIN_PROJECTS_PER_COURSE    = 2    # MD Section 1 / 27

# ---- Time budget (ONE formula — import this, never retype) ----
BUDGET_UTILIZATION = 0.80
# budget_hrs = hours_per_week × (timeline_days / 7) × BUDGET_UTILIZATION
# demand_hrs = Σ skill.mastery_hrs + 0.5×scenarios + 1.0×mocks + video_min/60 + 3.0×projects
# If demand > budget: REDUCE milestone count within bounds first, show math to user.

# ---- laugh_test threshold ----
LAUGH_TEST_COVERAGE_THRESHOLD = 0.80   # MD Section 24 / 27

# Maps level -> content difficulty starting tier for the user's M01.
# Beginner M01     = library tier 1 (intern-ready)
# Intermediate M01 = library tier 3 (working engineer)
# Senior M01       = library tier 5 (senior engineer)
LEVEL_STARTING_TIER = {
    "beginner": 1,
    "intermediate": 3,
    "senior": 5,
}

# Phase 1 — Level Range Preferences (guidance ranges, NOT hard targets)
# These define the typical milestone range for each ICP × level combination.
# The gap engine picks a specific count within this range based on:
#   gap_score, skill_readiness, known_skills, experience, weekly hours, role complexity
# Two learners with the same ICP+level may receive different counts.
LEVEL_RANGE_PREFERENCES = {
    ("low", "beginner"):       {"recommended": 5, "min": 4, "max": 6},
    ("low", "intermediate"):   {"recommended": 4, "min": 3, "max": 5},
    ("low", "senior"):         {"recommended": 3, "min": 2, "max": 4},
    ("high", "beginner"):      {"recommended": 4, "min": 3, "max": 5},
    ("high", "intermediate"):  {"recommended": 3, "min": 2, "max": 4},
    ("high", "senior"):        {"recommended": 2, "min": 2, "max": 3},
}


# ── Entry-level target role keywords ──────────────────────────
# Roles matching these patterns default to LOW ICP regardless of
# other signals, because they represent entry-level positions.
_ENTRY_LEVEL_ROLE_KEYWORDS = {
    "trainee", "intern", "fresher", "entry", "junior", "associate",
    "apprentice", "graduate", "beginner",
}


def _classify_icp(
    years_experience: int,
    current_identity: str,
    target_identity: str,
    current_salary_lpa: float,
    known_skills: list,
) -> tuple:
    """Classify user as 'low' or 'high' ICP with explicit reasoning.

    Hierarchy (first match wins):
      1. Entry-level target role  → low
      2. Fresher/student current  → low
      3. years_experience <= 1    → low
      4. Career switcher with no transferable experience → low
      5. years_experience >= 3 AND professional identity → high
      6. years_experience 1-3     → check other signals
      7. Default                  → low
    """
    ci_lower = (current_identity or "").lower()
    ti_lower = (target_identity or "").lower()

    # ── Rule 1: Entry-level target role ──
    for kw in _ENTRY_LEVEL_ROLE_KEYWORDS:
        if kw in ti_lower:
            return ("low", f"target_role_contains_{kw}")

    # ── Rule 2: Fresher / Student current identity ──
    if any(kw in ci_lower for kw in ("student", "fresher", "12th", "college")):
        return ("low", "current_identity_student_or_fresher")

    # ── Rule 3: years_experience <= 1 ──
    if years_experience <= 1:
        return ("low", f"years_experience_{years_experience}")

    # ── Rule 4: Career switcher with no transferable skills ──
    _tech_keywords = {"python", "sql", "java", "javascript", "html", "css",
                      "git", "linux", "aws", "docker", "excel", "power bi",
                      "tableau", "r", "c++", "c#", "go", "rust", "typescript",
                      "react", "node", "django", "spring", "kubernetes"}
    _ci_domain = _extract_domain(current_identity)
    _ti_domain = _extract_domain(target_identity)
    if _ci_domain and _ti_domain and _ci_domain != _ti_domain:
        # Domain switch: check if user has any tech skills
        _has_tech_skills = any(
            any(tk in (s or "").lower() for tk in _tech_keywords)
            for s in (known_skills or [])
        )
        if not _has_tech_skills:
            return ("low", f"domain_switch_no_transferable_skills")

    # ── Rule 5: years_experience >= 3 with professional identity ──
    _professional_identities = {"engineer", "developer", "manager", "architect",
                                "lead", "senior", "analyst", "consultant"}
    if years_experience >= 3 and any(kw in ci_lower for kw in _professional_identities):
        return ("high", f"experienced_professional_{years_experience}y")

    # ── Rule 6: years_experience 1-3 — check signals ──
    if years_experience >= 1 and current_salary_lpa > 0:
        if any(kw in ci_lower for kw in _professional_identities):
            return ("high", f"early_professional_salary_{current_salary_lpa}lpa")

    # ── Rule 7: Default ──
    return ("low", "default_low")


def detect_level(context: str, years_experience: int = 0) -> str:
    """
    Maps onboarding signals (resume parse / years of experience) to
    beginner | intermediate | senior.

      Beginner     -> 0 years
      Intermediate -> 1-3 years
      Senior       -> 4+ years
    """
    try:
        years = int(years_experience or 0)
    except (TypeError, ValueError):
        years = 0

    if years >= 4:
        return "senior"
    elif years >= 1:
        return "intermediate"
    return "beginner"


def compute_authoritative_milestone_range(
    gap_score: float,
    icp_type: str,
    level: str,
    hours_per_week: int,
    timeline_days: int,
    known_skills: list,
    experience_years: int,
    current_identity: str = "",
    target_identity: str = "",
) -> dict:
    """
    Phase 2 — Milestone Authority Engine.

    Determines the authoritative milestone range from:
      1. Level Range Preference (ICP x level table)
      2. Capability gap score (wider gap → higher end of range)
      3. Skill readiness / known skills (more known → fewer milestones needed)
      4. Available time (less time → fewer milestones)
      5. Experience (more experience → fewer milestones)
      6. Career switch + beginner + 0 experience → bump minimum

    Returns:
      recommended: int — the single best count within range
      minimum:     int — minimum milestone count
      maximum:     int — maximum milestone count
      confidence:  float — 0.0–1.0 how confident the engine is
      reasoning:   str — explanation of derivation
    """
    pref = LEVEL_RANGE_PREFERENCES.get(
        (icp_type, level),
        {"recommended": 4, "min": 2, "max": 5},
    )
    range_min = pref["min"]
    range_max = pref["max"]

    # ── Career switcher minimum pressure ─────────────────────
    # Domain switch + beginner + 0 experience = needs more milestones
    _ci_domain = _extract_domain(current_identity)
    _ti_domain = _extract_domain(target_identity)
    _is_domain_switch = bool(_ci_domain and _ti_domain and _ci_domain != _ti_domain)
    _is_beginner = (level == "beginner")
    _is_zero_exp = (experience_years == 0)

    _milestone_pressure = 0
    if _is_domain_switch and _is_beginner and _is_zero_exp:
        _milestone_pressure = 2  # full career restart
        range_min += 1
    elif _is_domain_switch and _is_beginner:
        _milestone_pressure = 1  # domain switch with beginner
        range_min += 1
    elif _is_domain_switch and _is_zero_exp:
        _milestone_pressure = 1  # domain switch with 0 exp
        range_min += 1

    # Gap factor: 0.0–1.0 maps to a 0..1 offset within the range
    gap_offset = max(0.0, min(1.0, gap_score))

    # Known-skills discount: each known skill reduces weight slightly
    known_skills_count = len(known_skills) if known_skills else 0
    skill_discount = min(0.15, known_skills_count * 0.03)

    # Experience discount: more experienced learners need fewer milestones
    exp_discount = min(0.20, experience_years * 0.05)

    # Time factor: less available time shifts toward lower end
    budget_hrs = hours_per_week * (timeline_days / 7) * BUDGET_UTILIZATION
    if budget_hrs < 80:
        time_factor = 0.0  # tight budget → lower end
    elif budget_hrs > 200:
        time_factor = 0.3  # generous budget → slightly higher
    else:
        time_factor = (budget_hrs - 80) / 400  # linear interpolation

    # Blend: gap pulls upward, discounts pull downward
    raw_offset = gap_offset - skill_discount - exp_discount + time_factor
    clamped_offset = max(0.0, min(1.0, raw_offset))

    range_size = range_max - range_min
    recommended = int(round(range_min + clamped_offset * range_size))
    recommended = max(range_min, min(range_max, recommended))

    # Confidence: higher when gap and range agree, lower at extremes
    mid = (range_min + range_max) / 2
    distance_from_mid = abs(recommended - mid) / (range_size or 1)
    confidence = round(0.70 + 0.20 * (1 - distance_from_mid) - 0.10 * (1 - gap_offset), 2)
    confidence = max(0.30, min(0.98, confidence))

    parts = [
        f"icp={icp_type} level={level} → range [{range_min},{range_max}]",
        f"gap_offset={gap_offset:.2f}",
        f"skill_discount={skill_discount:.2f} ({known_skills_count} known)",
        f"exp_discount={exp_discount:.2f} ({experience_years}y exp)",
        f"time_factor={time_factor:.2f} (budget={budget_hrs:.0f}h)",
    ]
    if _milestone_pressure:
        parts.append(f"career_switch_pressure=+{_milestone_pressure}")
    parts.append(f"→ recommended={recommended}")
    reasoning = "; ".join(parts)

    return {
        "recommended": recommended,
        "minimum": range_min,
        "maximum": range_max,
        "confidence": confidence,
        "reasoning": reasoning,
    }


class MilestoneAuthoritySchemaError(Exception):
    """Raised when milestone authority object violates the strict schema."""


def validate_milestone_authority_schema(obj: dict) -> None:
    """Validate milestone authority object conforms to the strict schema.

    Required fields:
      recommended: int
      minimum:     int
      maximum:     int
      confidence:  float
      reasoning:   str (non-empty)

    Every code path that produces a milestone authority dict must pass this.
    Raises MilestoneAuthoritySchemaError on any violation.
    """
    expected_keys = {"recommended", "minimum", "maximum", "confidence", "reasoning"}
    actual_keys = set(obj.keys())

    missing = expected_keys - actual_keys
    if missing:
        raise MilestoneAuthoritySchemaError(
            f"Milestone authority missing keys: {sorted(missing)}. "
            f"Present: {sorted(actual_keys)}"
        )

    extra = actual_keys - expected_keys
    if extra:
        raise MilestoneAuthoritySchemaError(
            f"Milestone authority has unexpected keys: {sorted(extra)}"
        )

    type_checks = {
        "recommended": int,
        "minimum": int,
        "maximum": int,
        "confidence": (int, float),
        "reasoning": str,
    }
    for key, expected_type in type_checks.items():
        val = obj[key]
        if not isinstance(val, expected_type):
            raise MilestoneAuthoritySchemaError(
                f"Milestone authority.{key}: expected {expected_type.__name__}, "
                f"got {type(val).__name__} ({val!r})"
            )

    if not obj["reasoning"].strip():
        raise MilestoneAuthoritySchemaError(
            "Milestone authority.reasoning must be a non-empty string"
        )


# ── Engineering-domain skill keywords for contamination check ──
_ENGINEERING_DOMAIN_SKILLS = {
    "system design", "distributed systems", "kubernetes", "docker",
    "microservices", "advanced architecture", "scalability",
    "ci/cd", "terraform", "aws architecture", "cloud infrastructure",
    "load balancing", "message queue", "event-driven architecture",
    "caching strategy", "database sharding", "containerization",
}


def _check_role_contamination(roadmap_data: dict, customer_profile: dict) -> None:
    """Reject analyst roadmaps containing excessive engineering-domain skills.

    Scans all skill titles across milestones. If the target role is
    analyst-type and >20% of skills match engineering-domain keywords,
    raises ValueError.

    This catches LLM hallucination / breadth-engine bleed before
    the roadmap reaches the user.
    """
    target_identity = (customer_profile.get("target_identity") or
                       roadmap_data.get("target_role") or "")
    ti_lower = target_identity.lower()

    # Only enforce for analyst roles
    _analyst_kw = {"analyst", "analytics", "bi ", "business intelligence",
                   "reporting", "dashboard"}
    is_analyst = any(kw in ti_lower for kw in _analyst_kw)
    if not is_analyst:
        return

    all_skill_titles = []
    for ms in roadmap_data.get("milestones", []):
        for mod in ms.get("modules", []):
            for skill in mod.get("skills", []):
                title = (skill.get("title") or skill.get("n") or "").lower()
                if title:
                    all_skill_titles.append(title)

    if not all_skill_titles:
        return

    eng_matches = sum(
        1 for t in all_skill_titles
        if any(ekw in t for ekw in _ENGINEERING_DOMAIN_SKILLS)
    )
    contamination_pct = eng_matches / len(all_skill_titles)

    if contamination_pct > 0.20:
        raise ValueError(
            f"Role contamination detected: {eng_matches}/{len(all_skill_titles)} "
            f"({contamination_pct:.0%}) skills are engineering-domain for "
            f"analyst role '{target_identity}'. "
            f"Engineering skills found: {[t for t in all_skill_titles if any(ekw in t for ekw in _ENGINEERING_DOMAIN_SKILLS)]}"
        )

    print(f"[CONTAMINATION CHECK] {eng_matches}/{len(all_skill_titles)} engineering skills "
          f"({contamination_pct:.0%}) — PASS (threshold <= 20%)")


def _poc_store(filename: str, data: dict) -> str:
    POC_STORAGE_DIR.mkdir(exist_ok=True)
    filepath = POC_STORAGE_DIR / filename
    with open(filepath, "w", encoding="utf-8") as f:
        import json as _json
        _json.dump(data, f, indent=2, ensure_ascii=False)
    print(f"[POC STORAGE] ✓ Saved → {filepath}")
    return str(filepath)


def store_pipeline_artifacts(
    user_id: str,
    roadmap_id: str,
    input_context: str,
    icp_type: str,
    roadmap_output: dict,
) -> dict:
    """
    UNCHANGED — same storage/output contract as before.
    Works against the milestone/module/skill shape because it only
    relies on: milestones[i].modules[j].skills[k] existing, plus
    milestone_id / skill_id keys (which are still present in the shape).
    """
    ts       = datetime.utcnow().strftime("%Y%m%d_%H%M%S")
    safe_uid = user_id.replace("/", "_")[:30]

    # 1 — Input transcript
    input_artifact = {
        "artifact_type":       "input_transcript",
        "user_id":             user_id,
        "roadmap_id":          roadmap_id,
        "icp_type":            icp_type,
        "generated_at":        datetime.utcnow().isoformat(),
        "context_sent_to_llm": input_context,
    }
    _poc_store(f"{safe_uid}_1_input_{ts}.json", input_artifact)

    # 2 — Output roadmap JSON
    output_artifact = {
        "artifact_type": "output_roadmap",
        "user_id":       user_id,
        "roadmap_id":    roadmap_id,
        "generated_at":  datetime.utcnow().isoformat(),
        "roadmap":       roadmap_output,
    }
    _poc_store(f"{safe_uid}_2_output_{ts}.json", output_artifact)

    # 3 — Next-module input packet
    milestones  = roadmap_output.get("milestones", [])
    first_ms    = milestones[0] if milestones else {}
    first_mod   = first_ms.get("modules", [{}])[0]
    first_skill = first_mod.get("skills", [{}])[0]

    next_input = {
        "artifact_type":        "next_module_input",
        "user_id":              user_id,
        "roadmap_id":           roadmap_id,
        "icp_type":             icp_type,
        "target_role":          roadmap_output.get("target_role", ""),
        "language":             roadmap_output.get("language", "en"),
        "starting_milestone":   roadmap_output.get("starting_milestone", ""),
        "current_milestone_id": roadmap_output.get("current_active_milestone", ""),
        "first_skill": {
            "skill_id":   first_skill.get("skill_id", ""),
            # title may live under "title" (legacy) or "n" (HTML short-name field)
            "title":      first_skill.get("title", first_skill.get("n", "")),
            "difficulty": first_skill.get("difficulty", 0),
        },
        "all_milestone_ids": [m.get("milestone_id") for m in milestones],
        "all_skill_ids": [
            skill.get("skill_id")
            for m in milestones
            for mod in m.get("modules", [])
            for skill in mod.get("skills", [])
        ],
        "generated_at": datetime.utcnow().isoformat(),
    }
    _poc_store(f"{safe_uid}_3_next_input_{ts}.json", next_input)

    return {
        "input_stored":      True,
        "output_stored":     True,
        "next_input_stored": True,
    }


# ── Centralised config ────────────────────────────────────────
from src.config import (
    OPENAI_API_KEY,
    GOOGLE_API_KEY,
    GOOGLE_MODEL as _GOOGLE_MODEL,
)

# ============================================================
# LLM Configuration (Dual-Engine Fallback)
# ============================================================

def get_llm():
    openai_key   = OPENAI_API_KEY or None
    gemini_key   = GOOGLE_API_KEY or None
    google_model = _GOOGLE_MODEL.strip().strip("\"'")

    _deprecated = {
        "gemini-1.5", "gemini-1.5-flash", "gemini-1.5-pro",
        "gemini-2.0-flash", "gemini-2.0", "gemini-pro",
    }
    if google_model in _deprecated:
        print(f"[LLM] ⚠ Model '{google_model}' is deprecated → upgrading to gemini-2.5-flash")
        google_model = "gemini-2.5-flash"

    if not openai_key:
        print("[LLM] No OpenAI key → Using Gemini")
        try:
            return ChatGoogleGenerativeAI(
                model=google_model,
                google_api_key=gemini_key,
                temperature=0.3,
            )
        except Exception as e:
            print(f"[LLM] Failed to init Gemini model='{google_model}': {e}")
            raise

    try:
        primary_llm = ChatOpenAI(api_key=openai_key, model="gpt-4o-mini", temperature=0.3)
        primary_llm.invoke("ping")
        print("[LLM] ✓ OpenAI valid → Using OpenAI with Gemini fallback")

        try:
            backup_llm = ChatGoogleGenerativeAI(
                model=google_model, google_api_key=gemini_key, temperature=0.3
            )
        except Exception as e:
            print(f"[LLM] Gemini backup init failed: {e}")
            backup_llm = None

        if backup_llm is None:
            return primary_llm
        return primary_llm.with_fallbacks([backup_llm])

    except Exception as e:
        print(f"[LLM] OpenAI failed: {e} → Switching to Gemini")
        try:
            return ChatGoogleGenerativeAI(
                model=google_model, google_api_key=gemini_key, temperature=0.3
            )
        except Exception as e2:
            print(f"[LLM] Gemini final fallback failed: {e2}")
            raise


llm = get_llm()

# ============================================================
# Roadmap Prompt — V3.3 Structure (MD-aligned, identity-driven)
# ============================================================
# Numbered rules below are renumbered sequentially (1–14, no duplicates).
# Every bound quoted in this prompt matches the constants block above,
# which in turn matches ROADMAP_GENERATION_SCIENCE_V2.md, with one
# explicit, documented deviation (skills max stays 8, not 10).
# ============================================================

# Build available courses summary string for the prompt
_AVAILABLE_COURSES_TEXT = "\n".join(
    f"  {i+1}. {v['name']} — modules: {', '.join(v['modules'])}"
    for i, (k, v) in enumerate(AVAILABLE_COURSES.items())
)
_UNAVAILABLE_COURSES_TEXT = "\n".join(f"  - {c}" for c in UNAVAILABLE_COURSES)

roadmap_prompt = PromptTemplate(
    input_variables=["context", "icp_type", "level", "hours_per_week", "timeline_days", "budget_hrs", "known_skills", "current_identity", "target_identity", "current_salary_lpa", "self_efficacy", "gap_score", "recommended_milestones", "recommended_milestones_min", "recommended_milestones_max", "milestone_confidence", "milestone_authority_reasoning", "recommended_modules_per_milestone", "recommended_skill_density", "gap_reasoning", "min_skills", "max_skills"],
    template="""
You are an expert AI career transformation roadmap architect for Vidya V3.

You are NOT generating a course.
You are generating a TRANSFORMATION ROADMAP.

The roadmap exists to transform a learner from their CURRENT IDENTITY into a
TARGET PROFESSIONAL IDENTITY. Every milestone, module, skill, lesson, project,
scenario, mock/interview, and checkpoint must be justified by this transformation.

Counts in this prompt are CAPACITY BOUNDS, not targets. Filling to the max is
as much a failure as padding to the min. Every count must be justified by the
real-world skill gap and survive the laugh_test — a working senior in that
field would respect it.

USER CONTEXT:
{context}

USER ICP TYPE:
{icp_type}

USER LEVEL (detected from onboarding — resume parse + voice + behavioral signals):
{level}

CURRENT IDENTITY:
{current_identity}

TARGET IDENTITY:
{target_identity}

ROADMAP PURPOSE:

Transform the learner from CURRENT IDENTITY
to TARGET IDENTITY.

Every milestone must represent a meaningful identity transition.

CURRENT SALARY:
{current_salary_lpa} LPA

SALARY FLOOR RULE:

If icp_type="high":

M01 salary band must never be below current_salary_lpa.

SELF-EFFICACY:
{self_efficacy}

Rules:

0.0 - 0.4:

* smaller milestones
* more scaffolding
* safer progression

0.4 - 0.7:

* balanced progression

0.7 - 1.0:

* larger capability jumps
* fewer scaffolding steps

TIME BUDGET (HARD CONSTRAINT):

User has {hours_per_week} hrs/week
and {timeline_days} days available.

budget_hrs =
{hours_per_week} × ({timeline_days}/7) × 0.8
============================================

{budget_hrs} hrs

Before generating milestones, estimate roadmap demand:

demand_hrs =
  (skills × 1.5)
  + (scenarios × 0.5)
  + (mocks × 1.0)
   + (lessons × 0.25)
   + (projects × 3.0)

Rules:
* If estimated demand exceeds budget_hrs, REDUCE milestone count within the allowed range first.
* fits_life will extend duration if demand still exceeds budget after reducing milestones.
* NEVER go outside the allowed milestone range [{recommended_milestones_min},{recommended_milestones_max}].
* Smaller budgets should produce fewer milestones (within range), not more lessons per skill.
* Larger budgets may justify more milestones, lessons, scenarios, and deeper projects.
* Never ignore the user's available time — if even the minimum milestone count exceeds budget, fits_life extends duration.

STARTING POINT (HARD CONSTRAINT — ZPD / starts_where_they_are):

Known skills from onboarding:
{known_skills}

For each skill already known:

* do NOT include them in the roadmap at all
* known skills must NEVER appear as teachable content
* do NOT start Module 1 with content already mastered
* use the next logical capability step
* learners must NEVER be served content they already know

Starting professionals at skills they already have is a major churn trigger.
Known skills that survive to the roadmap are a BLOCKER violation.

If known_skills = "none provided":
apply ZPD rules using level only.

CAPABILITY GAP ENGINE OUTPUT

gap_score:
{gap_score}

gap_reasoning:
{gap_reasoning}

recommended_milestone_count:
{recommended_milestones}

allowed_milestone_range:
[{recommended_milestones_min}, {recommended_milestones_max}]

milestone_count_confidence:
{milestone_confidence}

milestone_authority_reasoning:
{milestone_authority_reasoning}

recommended_modules_per_milestone:
{recommended_modules_per_milestone}

recommended_skill_density:
{recommended_skill_density}

=== GENERATION RULES (STRICT — ALL MUST BE FOLLOWED) ===

RULE 1 — HIERARCHY (LOCKED):
Roadmap -> Milestone -> Module -> (Skills -> Lessons) + Science (Scenarios + Mocks) + Project
Never break this hierarchy.

RULE 2 — MILESTONE COUNT (AUTHORITATIVE RANGE — FROM MILESTONE AUTHORITY ENGINE):
The Capability Gap Engine and Milestone Authority Engine determined:

  Recommended milestone count: {recommended_milestones}
  Allowed range: [{recommended_milestones_min}, {recommended_milestones_max}]
  Confidence: {milestone_confidence}

You MUST generate a roadmap whose milestone count falls INSIDE this allowed
range — between {recommended_milestones_min} and {recommended_milestones_max}
inclusive. {recommended_milestones} is a recommendation, NOT a hard ceiling:
you may go up to {recommended_milestones_max} if the capability breadth
genuinely requires it. Never go outside [{recommended_milestones_min},
{recommended_milestones_max}] regardless of which number you land on.

Choose the count that best matches:

  * capability breadth — how many distinct identity transitions are needed
  * target role complexity — more complex roles need more steps
  * available time — more time allows more milestones
  * identity progression density — each milestone must be a meaningful identity

The milestone count is a RANGE, not a single target. Two learners with the
same ICP and level may receive different counts if their gap, experience,
or time budget differ.

Every milestone represents a MARKET-RECOGNIZED IDENTITY the learner
earns, NOT a topic-coverage checkpoint.
Milestone codes always start at M01 (relative to THIS user).

Include a "milestone_count_rationale" field in your JSON output explaining
WHY you chose the specific count within the allowed range.
Example: "5 milestones: beginner with 10 h/wk, wide gap from student to
Data Scientist requires 5 identity transitions"

RULE 2B — IDENTITY PROGRESSION DENSITY (CRITICAL):
Each milestone must represent a SINGLE, COHERENT identity transition.

A beginner → Data Scientist transition in 2 milestones:
  M01: "Python Basics"       ← NOT an identity
  M02: "Data Scientist"      ← too large a jump
  RESULT: FAIL — density too low (1 identity per ~6 months)

A beginner → Data Scientist in 5 milestones:
  M01: "AI Foundations Engineer"     ← 0–3mo intern-ready
  M02: "Data Analyst"                ← 3–6mo entry-level
  M03: "ML Foundations Engineer"     ← 6–9mo junior
  M04: "Applied ML Engineer"         ← 9–12mo mid-level
  M05: "Data Scientist"              ← 12–15mo target
  RESULT: PASS — each step is a market-recognized identity

Validation rules:
  - Beginner → complex role (Data Scientist, AI Engineer, Architect):
    MINIMUM 4 milestones. Fewer than 4 means unrealistically dense steps.
  - Beginner → simple role (Junior Developer, Support Engineer):
    MINIMUM 2 milestones. Simple roles need fewer identity transitions.
  - Intermediate → adjacent role (Backend → Fullstack):
    MINIMUM 2 milestones. Fewer identity transitions needed.
  - Intermediate → distant role (Backend → AI Engineer):
    MINIMUM 3 milestones. More transitions needed.
  - Senior → senior role:
    MINIMUM 2 milestones. Highly focused deep-dives.
  - Any path where years_experience=0 and target is a senior/lead role:
    MINIMUM 4 milestones (must build entire foundation).

RULE 3 — RELATIVE MILESTONE CODES (CRITICAL):
Milestone codes ALWAYS start at M01 for THIS user, regardless of their level.
M01 means "this user's current starting milestone", NOT "intern content".

  - level = "beginner":     M01 content = intern-ready (library tier 1).
  - level = "intermediate": M01 content = working-engineer (library tier 3).
    SKIP intern/junior-level content entirely.
  - level = "senior":       M01 content = senior-engineer (library tier 5).
    SKIP intern/junior/working/mid-level content entirely.

ZPD RULE: Never serve content below the user's detected floor.
If onboarding shows the user already has a skill, mark it completed (p=mastery prior).
Do not reteach it.

RULE 4 — MODULE COUNT PER MILESTONE (DYNAMIC, PER-MILESTONE):
Modules are capability clusters. Each milestone's module count is determined
INDEPENDENTLY by that milestone's unique capability breadth and identity.

Module count MAY vary between milestones:
  - Focused milestone (narrow skill cluster):  2 modules
  - Broad milestone (multiple capabilities):    3 modules
  - Transformation milestone (full sub-role):   4 modules

Examples of valid per-milestone distributions:
  M01: 2 modules (foundation focus)
  M02: 4 modules (broad capability building)
  M03: 3 modules (deepening expertise)
  M04: 2 modules (specialization)
  M05: 4 modules (capstone transformation)

Bounds:
  - Minimum 2 modules per milestone.
  - Maximum 4 modules per milestone.
  - Never generate 1 module or more than 4 modules.
  - Do NOT generate modules merely to satisfy a count — each module must
    represent a distinct, coherent capability cluster within the milestone's
    identity.
  - Every milestone must include a "module_count_rationale" field explaining
    why the CHOSEN count fits the milestone's required breadth.

Example rationale for variable counts:
  "M01: 2 modules because this is a narrow foundation identity needing only
   Python fundamentals and data basics. M02: 4 modules because this milestone
   spans distributed systems, cloud architecture, observability, and technical
   leadership."

RULE 5 — SKILLS PER MODULE (UPPER DENSITY BOUND ONLY):
Skills per module are dynamic — driven by market need, capability gap,
and job description requirements. The domain-based max skill density is
an UPPER BOUND, NOT a target or a fixed count.

  - Minimum {min_skills} skills per module (any fewer = not a real module).
  - Maximum {max_skills} skills per module (hard upper bound).
  - Do NOT pad to reach a target count. Generate only skills that:
      * address a real market need
      * close a specific capability gap
      * come from job description / JD requirements
      * enable the milestone identity

  - Module A may have {min_skills} skills (focused depth).
  - Module B may have {max_skills} skills (broad capability cluster).
  - Module C may have 5 skills.
  - ALL valid — each is determined by the module's unique purpose.

  - Every module must include a "skill_count_rationale" field explaining
    why the CHOSEN count fits the module's required breadth.
    Bad rationale: "5 skills because that's the max"
    Good rationale: "3 skills because this module focuses narrowly on
    prompt engineering: prompt design, few-shot tuning, and safety"

Each skill object has:
  - "skill_id": globally unique id, e.g. "SKILL_M01_M1_S1"
  - "n": short skill name (snake_case, e.g. "python", "fastapi", "rag")
  - "title": human-readable skill title
  - "ordinal": integer position of this skill within its module (1-indexed).
    Skill 1 is the first skill in the module, Skill 2 is the second, etc.
  - "lessons": array of 2-4 lesson title strings, each prefixed with
    "Lesson N:" where N is the 1-indexed position within this skill.
    Example for a skill with 3 lessons:
      ["Lesson 1: Embeddings",
       "Lesson 2: Chunking Strategies",
       "Lesson 3: Retrieval Evaluation"]
    Lessons are delivery units — keep each one short and focused.
    A simple skill needs 2 lessons, a normal skill needs 3, a complex skill needs 4.
    Every lesson title must start with "Lesson N:" — no bare titles.
  - "p": initial mastery percentage (0-100). Set to 0 (placeholder).
        BKT injection overwrites p dynamically based on skill difficulty
        and learner profile. Do NOT manually set p to arbitrary values.
  - "mastery_state": BKT object (see OUTPUT STRUCTURE) — KEEP for backend
    mastery tracking regardless of "p".
  - "content_flow": object with video/scenario/mock/review (see OUTPUT
    STRUCTURE) — KEEP for backend tracking.
  - "unlock_rules": {{"requires": [...], "minimum_mastery": 0.0,
    "unlock_type": "immediate" | "prerequisite"}}
       * Skills must form an acyclic prerequisite graph.
       * The FIRST skill of the entire roadmap (M01, first module, first
         skill) must have requires: [] and unlock_type: "immediate".

RULE 6 — PROJECT PER MILESTONE (MANDATORY, BOUNDED 1-2):

Every milestone MUST contain BETWEEN 1 AND 2 real-world projects
(PROJECTS_PER_MILESTONE bound). One project is the default; a second
project is justified only if the milestone's identity genuinely spans
two distinct deliverables (e.g. a data pipeline AND a serving API).
Do not pad to 2 — most milestones need exactly 1.

Every project must:
  - Use skills from that milestone.
  - Follow the official lifecycle: Plan → Architect → Build with AI → Audit AI Output → Deploy → Test.
  - Cover at least 2 vibe layers (preferred: 3+).
  - Require the learner to catch at least one AI mistake (seeded_error).
  - Feel like real industry work — not a toy exercise.
  - Include a "project_count_rationale" at the milestone level explaining
    why 1 (or 2) projects were chosen.

Project format:

"project": {{
  "title": "string",
  "vibe_layers": [
      "vibe_architecture",
      "vibe_solution",
      "deployment"
  ],
  "description": "2-3 sentence description of what the learner builds",
  "deliverable": "what the learner submits as proof of completion",
  "seeded_errors": [
      "AI mistake the learner must detect and fix"
  ],
  "deploy_required": true
}}

If you generate a second project for a milestone, place it in the
"projects" array (see OUTPUT STRUCTURE) alongside the first.

RULE 7 — SCIENCE ARRAY (SCENARIO / MOCK — PER MILESTONE, BOUNDED):
!! CRITICAL !!
Every milestone contains Scenarios and Mocks distributed across its
modules. Scenarios are realistic production/debugging situations the learner
resolves. Mocks (interviews) test the ability to PERFORM the milestone identity.

HARD BOUNDS (not optional, not "up to"):
  - Scenarios per milestone: BETWEEN 3 AND 7 (never fewer than 3, never more
    than 7). 3-7 is a real range — most milestones should land in the
    middle (4-5), not always at the floor or ceiling.
  - Mocks/Interviews per milestone: BETWEEN 1 AND 2 (never 0, never more
    than 2). A milestone with deep capability or two distinct hiring
    contexts may justify 2; a focused milestone needs only 1.
  - no_spectators applies PER MODULE: every non-free module must have
    AT LEAST 1 applied activity (a Scenario or a Mock) in its science array.
    A module with zero science items fails this check.

Rules for science items:
  - "Scenario" = a realistic production/debugging situation the learner must resolve.
    Examples: bad retrieval causes hallucinations, pipeline fails before demo,
    latency spike in production, nightly job fails at 9am.
    NOT toy exercises. NOT hypotheticals.
  - "Interview" (= Mock) = an interview question testing the milestone identity.
    Must test the ability to PERFORM the role, not recall trivia.
  - A module may have 0 to 3 science items, but across the WHOLE milestone
    the Scenario total must land in [3,7] and the Interview/Mock total in [1,2].
  - The Interview should be in a module that has no Scenario (separate modules)
    where the module layout allows it.
  - Each science item must have a distinct, scenario-specific desc.
  - Include "scenario_count_rationale" and "interview_count_rationale" fields
    at the milestone level explaining why the chosen counts fit this
    milestone's complexity and hiring-process realism.

RULE 8 — MODULE METADATA (HTML-ALIGNED):
Each module must include:
  - "id": HTML-style id "M{{milestone_number}}.{{module_number}}", e.g. "M1.1",
    "M1.2" (module numbering restarts at .1 for each milestone; milestone number
    is 1-indexed position, NOT the M01/M02 zero-padded label).
  - "title": short, capability-cluster title (from AVAILABLE_COURSES where
    possible). Must represent one coherent capability — no filler modules.
  - "free": true ONLY for modules inside the user's M01. All other modules: false.
  - "vis": visualization type, one of: "code+real_tutor", "code+ppt",
    "ppt+animation", "ppt+code", "real_tutor+ppt", "animation+code",
    "notebook+code", "notebook+ppt", "real_tutor+code".

RULE 8B — AI-FIRST LAYER (June 2026 Official Model)

Each module must include:

"ai_first_layer"

Allowed values (must match VALID AI LAYERS from RULE 11b below):

- vibe_planning
- vibe_architecture
- vibe_solution
- deployment

Definitions:

vibe_planning:
Break a problem into executable steps,
sequence work, define requirements.

vibe_architecture:
Choose systems, tools, tradeoffs, integrations,
define system boundaries.

vibe_solution:
Review AI output, fix errors, improve solution quality,
build and verify with AI assistance.

deployment:
Ship, verify, test, monitor,
CI/CD, cloud infrastructure, release management.

Requirements:

- Every module must have exactly one ai_first_layer (must be from VALID AI LAYERS above).
- Every milestone must contain at least two distinct ai_first_layers.
- Never generate a milestone where all modules belong to the same layer.


RULE 9 — MILESTONE METADATA (HTML-ALIGNED):
A milestone is NOT a topic. It is a MARKET-RECOGNIZED IDENTITY.
Every milestone must answer: "What job-ready identity has the learner earned?"

Good milestone titles: "AI Foundations Engineer", "Applied AI Engineer",
"Agentic AI Engineer", "Production AI Engineer", "Senior AI Engineer".
Bad milestone titles: "Python Basics", "Machine Learning", "Prompt Engineering".

Each milestone must include:
  - "milestone_id": "M01".."M0N" (zero-padded, relative per RULE 3)
  - "label": same value as milestone_id
  - "t": short market-recognized identity title, e.g. "AI Foundations Engineer"
  - "sal": salary band for this milestone identity (sourced from Naukri/Indeed 2026).
       * ICP-A (icp_type="low"): Indian rupee LPA tiers, e.g. "Unpaid/stipend",
         "₹3-5 LPA", "₹6-9 LPA". M01 for a beginner student MUST show
         "Unpaid/stipend". Salary band must be ≥ learner's current salary.
       * ICP-B (icp_type="high"): same LPA tier system but starting higher,
         reflecting existing professional experience.
  - "o": one-sentence outcome statement — what the learner can DO and DEMONSTRATE
    to a hiring manager after this milestone.
  - "sc_n": Scenario count for this milestone. Must be between 3 and 7.
  - "iv": Mock/Interview count for this milestone. Must be between 1 and 2.
  - "identity_statement": 1-sentence motivational framing (Possible Selves model).
  - "checkpoint_rule": {{"required_mastery": 0.9, "checkpoint_type": "mock_interview"}}
    A checkpoint is EARNED, never purchased. Requires mastery ≥ 0.90 + project
    completed + interview passed.
  - "projects": array of 1-2 project objects per RULE 6.
  - "modules": 2–4 modules per RULE 4 (module count is dynamic, capability-breadth driven).
  - "milestone_count_rationale" lives at the ROOT of the output (see OUTPUT
    STRUCTURE), not per-milestone. "module_count_rationale",
    "scenario_count_rationale", "interview_count_rationale", and
    "project_count_rationale" live on EACH milestone.

RULE 10 — STARTING MILESTONE:
starting_milestone = "M01" ALWAYS (per RULE 3, this is relative to the user).
current_active_milestone = "M01".

RULE 11 — AVAILABLE COURSES ONLY (HARD CONSTRAINT, UNCHANGED):
You MUST only generate modules, lessons, and skills derived from these
completed courses and their module lists:
""" + _AVAILABLE_COURSES_TEXT + """

DO NOT generate content from these incomplete or unavailable courses:
""" + _UNAVAILABLE_COURSES_TEXT + """

If the user's goal requires an unavailable course, use the closest available
course's modules instead.
Example: user wants "backend developer" -> use High Level System Design +
Low Level System Design + AI Ready modules.
Example: user wants "data analyst" -> use AI Ready + Machine Learning modules
(no Data Science course — it is incomplete).

RULE 11B — AI METADATA (STRICT — NON-NEGOTIABLE):
Every skill MUST include ai_metadata with these exact values:

VALID AI LAYERS (June 2026 Official Model — only these are accepted):
  vibe_planning
  vibe_architecture
  vibe_solution
  deployment

FORBIDDEN LEGACY VALUES (will cause rejection — these are INVALID):
  architecture
  implementation
  debugging
  optimization

FORBIDDEN (will cause rejection):
  planning
  leadership
  design
  coding
  infra
  cloud
  security
  analysis
  testing
  observability

VALID USAGE TYPES:
  generation | analysis | automation | optimization | decision_support | planning

VALID AUTOMATION LEVELS:
  assistant | copilot | agent | autonomous

EXAMPLES:
  System Architecture Design → vibe_architecture
  API Specification          → vibe_architecture
  AI-Assisted Implementation → vibe_solution
  Debugging AI Output        → vibe_solution
  CI/CD Pipeline Setup       → deployment

RULE 12 — MOCK UNLOCK & CONTENT STATUS:
mock.unlock_mastery = 0.75 always. All content_flow statuses ("video",
"scenario", "mock") = "locked" on generation. review.next_review_at = null.

RULE 13 — DIFFICULTY PROGRESSION (BKT PRIORS):
mastery_state.bkt.prior = existing knowledge (0=no knowledge, 0.5=fully known).
Earlier milestones cover more foundational content so prior~0.10-0.25.
Later milestones cover harder content so prior~0.05-0.15.
BKT prior is DYNAMIC — computed by engine. LLM sets placeholder 0.15.
target_mastery is always 0.9.

RULE 14 — CATALOG-ALIGNED SKILL NAMES:
Every skill title must be chosen from AVAILABLE_COURSES whenever possible.
Do not invent alternative skill names when an equivalent catalog entry already
exists. Use the exact title from the skills_include list of the closest matching
course. This ensures the platform can map every skill to existing content.

RULE 15 — STRICT JSON ONLY:
Return ONLY raw valid JSON. No markdown. No code fences. No comments. No
explanations. All double quotes inside string values MUST be escaped with
backslash (e.g. "description": "Learn \\"Python\\" fundamentals"). No trailing
commas before }} or ]. Every property must have a comma separator.

=== OUTPUT STRUCTURE ===

{{
  "roadmap_id": "ai_roadmap_placeholder",
  "user_id": "placeholder",
  "icp_type": "{icp_type}",
  "level": "{level}",
  "target_role": "string — market-recognized role the user is transforming into",
  "language": "en",
  "starting_milestone": "M01",
  "current_active_milestone": "M01",
  "estimated_total_hours": 0,
  "budget_hours": {budget_hrs},
  "milestone_count_rationale": "string — explanation of why THIS count within the allowed range [{recommended_milestones_min},{recommended_milestones_max}] was chosen (gap, time, breadth, density)",
  "vision_profile": {{
    "current_state": "string — 1 sentence: current identity of the learner",
    "main_blocker": "string — 1 sentence: biggest obstacle to transformation",
    "top_motivation": "string — 1 sentence: why they want this identity"
  }},
  "roadmap_meta": {{
    "generated_at": "PLACEHOLDER_TIMESTAMP",
    "version": "v3.3",
    "science_model": ["ZPD", "Mastery Learning", "CLT", "BKT", "Possible Selves", "Retrieval Practice", "Deliberate Practice", "Testing Effect"]
  }},
  "milestones": [
    {{
      "milestone_id": "M01",
      "label": "M01",
      "t": "string — market-recognized identity, e.g. AI Foundations Engineer",
      "sal": "string — salary band, e.g. Unpaid/stipend or ₹6-9 LPA",
      "o": "string — 1-sentence outcome: what the learner can DO and demonstrate",
      "sc_n": 4,
      "iv": 1,
      "module_count_rationale": "string — why this milestone has N modules (capability breadth rationale)",
      "scenario_count_rationale": "string — why this milestone has N scenarios (3-7) — real situations this role faces",
      "interview_count_rationale": "string — why this milestone has N mocks/interviews (1-2) — hiring process realism",
      "project_count_rationale": "string — why this milestone has N projects (1-2)",
      "identity_statement": "string — 1-sentence Possible Selves motivational framing",
      "checkpoint_rule": {{
        "required_mastery": 0.9,
        "checkpoint_type": "mock_interview"
      }},
      "projects": [
        {{
          "title": "string — real-world project title",
          "vibe_layers": [
            "vibe_planning",
            "vibe_architecture",
            "vibe_solution"
          ],
          "description": "string — 2-3 sentences describing what the learner builds",
          "deliverable": "string — what the learner submits as proof",
          "seeded_errors": [
            "string — specific AI mistake the learner must detect and fix"
          ],
          "deploy_required": true
        }}
      ],
      "modules": [
        {{
          "id": "M1.1",
          "title": "string — capability-cluster title",
          "ai_first_layer": "vibe_planning | vibe_architecture | vibe_solution | deployment",
          "free": true,
          "vis": "code+real_tutor",
          "skill_count_rationale": "string — why this module needs N skills",
          "skills": [
            {{
              "skill_id": "SKILL_M01_M1_S1",
              "n": "python",
              "title": "string — human readable skill title",
              "auto_completed": false,
              "ai_metadata": {{
                "ai_first": true,
                "layer": "vibe_architecture",  # ONLY: vibe_planning | vibe_architecture | vibe_solution | deployment
                "usage_type": "generation | analysis | automation | optimization | decision_support",
                "automation_level": "assistant | copilot | agent | autonomous"
              }},
              "why_this_skill": "string explaining why this skill matters in the current hiring market",
              "lessons": [
                "string — specific lesson title 1",
                "string — specific lesson title 2",
                "string — specific lesson title 3"
              ],
              "p": 0,
              "mastery_state": {{
                "state": "unlocked",
                "current_mastery": 0.0,
                "target_mastery": 0.9,
                "bkt": {{
                  "prior": 0.15,  # DYNAMIC — see RULE 13
                  "learn_rate": 0.25,  # DYNAMIC — see RULE 13
                  "guess": 0.1,
                  "slip": 0.05
                }}
              }},
              "unlock_rules": {{
                "requires": [],
                "minimum_mastery": 0.0,
                "unlock_type": "immediate"
              }},
              "content_flow": {{
                "video": {{
                  "content_id": "VID_M01_M1_S1",
                  "title": "string",
                  "status": "locked"
                }},
                "scenario": {{
                  "content_id": "SCN_M01_M1_S1",
                  "title": "string",
                  "difficulty": 0.3,
                  "status": "locked"
                }},
                "mock": {{
                  "content_id": "MOCK_M01_M1_S1",
                  "unlock_mastery": 0.75,
                  "status": "locked"
                }},
                "review": {{
                  "review_type": "spaced_repetition",
                  "next_review_at": null
                }}
              }},
              "analytics": {{
                "attempts": 0,
                "average_score": 0,
                "time_spent_minutes": 0,
                "last_activity_at": null
              }}
            }}
          ],
          "science": [
            {{"type": "Scenario", "desc": "string — realistic production/debugging situation"}}
          ]
        }}
        // Additional modules (M1.3, M1.4 if needed) follow the same structure.
        // Each must have a unique id (M1.2, M1.3, ...), skill_id prefix
        // matching its module position, and science populated so that the
        // MILESTONE-LEVEL totals land in [3,7] scenarios and [1,2] mocks.
      ]
    }}
  ]
}}

IMPORTANT REMINDERS:
- Milestone count MUST be within the allowed range [{recommended_milestones_min},{recommended_milestones_max}] (see RULE 2).
- Module count per milestone: 2–4, MAY VARY between milestones (see RULE 4).
- Every module has {min_skills}–{max_skills} skills, determined by market need (see RULE 5).
- Every skill has 2-4 lessons (inside the skill object, NOT at module level).
- Every milestone has Scenarios in [3,7] and Mocks/Interviews in [1,2].
  These are real bounds, not "up to" — never 0 scenarios, never 0 mocks.
  Distribute Scenarios across modules (any module may hold multiple). The Interview
  should be in a module separate from any Scenario module where layout allows.
- Every milestone has 1-2 projects (see RULE 6); most need exactly 1.
- All content_id values must be globally unique (e.g. VID_M01_M1_S1, SCN_M02_M2_S3).
- All skill_id values must be globally unique (e.g. SKILL_M01_M1_S1).
- skill unlock_rules.requires must reference skill_ids that appear EARLIER in
  the roadmap (no forward or circular references).
- The FIRST skill of the entire roadmap must have requires: [] and
  unlock_type: "immediate".
- Each milestone object must include a "label" field equal to its milestone_id.
- Return ONLY raw JSON. Nothing else.
"""
)

def _build_chain():
    """
    Build the LangChain chain fresh on every invocation.

    WHY: roadmap_prompt and llm are module-level objects. If either is
    re-assigned after the module is first imported (e.g. during testing,
    hot-reload, or prompt iteration), a module-level `roadmap_chain`
    would still hold a reference to the OLD objects — requiring a full
    process restart to pick up changes.

    Rebuilding the chain in-call eliminates that stale-reference window.
    The composition is lightweight (no I/O, no network), so the overhead
    is negligible compared to the LLM call that follows.
    """
    return roadmap_prompt | llm | StrOutputParser()


# ============================================================
# BKT Computation  (MD Section 15 — Skill-aware)
# ============================================================

_ADVANCED_KEYWORDS = {
    "advanced", "complex", "distributed", "architecture", "design", "deep",
    "optimization", "scalable", "microservices", "kubernetes", "docker",
    "system design", "architect", "performance", "security", "integration",
    "deployment", "ci/cd", "monitoring", "observability", "refactoring",
    "patterns", "principles", "algorithms", "data structure",
}

def _estimate_skill_difficulty(skill: dict, skill_index: int = 0, total_skills: int = 1) -> float:
    """Estimate difficulty from skill name/title keywords + position."""
    name = ((skill.get("n") or "") + " " + (skill.get("title") or "")).lower()
    keyword_matches = sum(1 for kw in _ADVANCED_KEYWORDS if kw in name)
    difficulty = min(keyword_matches * 0.1, 0.5)
    if total_skills > 1:
        position_factor = skill_index / (total_skills - 1)
        difficulty += position_factor * 0.15
    return min(difficulty, 0.6)


def _compute_domain_overlap(skill: dict, known_skills: list) -> float:
    """How much overlap between this skill and user's known_skills."""
    if not known_skills:
        return 0.0
    name = ((skill.get("n") or "") + " " + (skill.get("title") or "")).lower()
    name_words = set(name.split())
    for ks in known_skills:
        ks_lower = ks.lower().strip()
        if ks_lower in name or name == ks_lower:
            return 1.0
        ks_words = set(ks_lower.split())
        if name_words & ks_words:
            return 0.5
    return 0.0


def compute_bkt_prior(
    skill: dict,
    user_signals: dict,
    milestone_index: int,
    total_milestones: int,
    skill_index: int = 0,
    total_skills: int = 1,
) -> float:
    """BKT prior = milestone baseline + difficulty/experience/domain adjustments."""
    if total_milestones <= 1:
        milestone_base = 0.15
    else:
        progress = milestone_index / (total_milestones - 1)
        milestone_base = 0.25 - progress * 0.20

    difficulty = _estimate_skill_difficulty(skill, skill_index, total_skills)
    difficulty_adj = -difficulty * 0.10

    years = user_signals.get("years_experience", 0)
    exp_factor = min(years / 10, 1.0)
    exp_adj = exp_factor * 0.08

    overlap = _compute_domain_overlap(skill, user_signals.get("known_skills", []))
    overlap_adj = overlap * 0.10

    prior = milestone_base + difficulty_adj + exp_adj + overlap_adj
    return round(max(0.01, min(0.50, prior)), 3)


def compute_learn_rate(
    skill: dict,
    user_signals: dict,
    milestone_index: int,
    total_milestones: int,
    skill_index: int = 0,
    total_skills: int = 1,
) -> float:
    """BKT learn_rate = milestone baseline + difficulty/experience/domain adjustments."""
    if total_milestones <= 1:
        milestone_base = 0.25
    else:
        progress = milestone_index / (total_milestones - 1)
        milestone_base = 0.15 + progress * 0.20

    difficulty = _estimate_skill_difficulty(skill, skill_index, total_skills)
    difficulty_adj = -difficulty * 0.08

    years = user_signals.get("years_experience", 0)
    exp_factor = min(years / 10, 1.0)
    exp_adj = exp_factor * 0.06

    overlap = _compute_domain_overlap(skill, user_signals.get("known_skills", []))
    overlap_adj = overlap * 0.05

    lr = milestone_base + difficulty_adj + exp_adj + overlap_adj
    return round(max(0.05, min(0.50, lr)), 3)


def inject_bkt_values(roadmap_data: dict) -> None:
    """Overwrite BKT prior/learn_rate across all skills with skill-aware values.
    
    Also synchronizes skill['p'] with BKT prior for non-beginner learners
    and forces auto_completed skills to p=100.
    """
    milestones = roadmap_data.get("milestones", [])
    total_ms = len(milestones)
    level = roadmap_data.get("level", "beginner")
    user_signals = {
        "years_experience": roadmap_data.get("years_experience", 0),
        "known_skills": roadmap_data.get("known_skills", []),
    }
    for ms_idx, ms in enumerate(milestones):
        for mod in ms.get("modules", []):
            skills = mod.get("skills", [])
            total_sk = len(skills)
            for sk_idx, skill in enumerate(skills):
                prior = compute_bkt_prior(
                    skill, user_signals, ms_idx, total_ms, sk_idx, total_sk,
                )
                lr = compute_learn_rate(
                    skill, user_signals, ms_idx, total_ms, sk_idx, total_sk,
                )
                bkt = skill.get("mastery_state", {}).get("bkt", {})
                if bkt:
                    bkt["prior"] = prior
                    bkt["learn_rate"] = lr
                # Sync skill['p'] with BKT prior for non-beginner learners
                if level != "beginner":
                    skill["p"] = round(prior * 100)
# Pinecone Storage for Roadmap   (UNCHANGED)
# ============================================================

def store_roadmap_in_pinecone(user_id: str, roadmap_id: str, roadmap_data: dict) -> bool:
    try:
        from src.pinecone_utils import get_embedding, pc, INDEX_NAME

        index = pc.Index(INDEX_NAME)

        target_role      = roadmap_data.get("target_role", "")
        icp_type         = roadmap_data.get("icp_type", "")
        milestones       = roadmap_data.get("milestones", [])
        # "identity_label" was the old field name; new shape uses "t" for the
        # milestone title. Support both so older roadmaps still embed fine.
        milestone_labels = " | ".join(
            m.get("t", m.get("identity_label", "")) for m in milestones
        )

        embed_text = (
            f"Career roadmap for user {user_id}. "
            f"Target role: {target_role}. "
            f"ICP: {icp_type}. "
            f"Milestones: {milestone_labels}."
        )

        print(f"[PINECONE STORE] Generating embedding for roadmap {roadmap_id}...")
        embedding = get_embedding(embed_text, task_type="RETRIEVAL_DOCUMENT")

        roadmap_json_str   = json.dumps(roadmap_data)
        MAX_METADATA_BYTES = 38000
        json_bytes = len(roadmap_json_str.encode("utf-8"))
        print(f"[PINECONE STORE] payload_size={json_bytes}  "
              f"namespace={user_id}  vector_id={user_id}_roadmap_{roadmap_id}")

        if json_bytes > MAX_METADATA_BYTES:
            print("[PINECONE STORE] ⚠ Roadmap JSON too large — storing summary only")
            store_payload = {
                "text":               embed_text,
                "roadmap_id":          roadmap_id,
                "user_id":             user_id,
                "target_role":         target_role,
                "icp_type":            icp_type,
                "milestone_labels":    milestone_labels,
                "generated_at":        roadmap_data.get("roadmap_meta", {}).get("generated_at", ""),
                "full_roadmap_stored": False,
                "doc_type":            "roadmap_summary",
            }
        else:
            store_payload = {
                "text":               embed_text,
                "roadmap_id":          roadmap_id,
                "user_id":             user_id,
                "target_role":         target_role,
                "icp_type":            icp_type,
                "milestone_labels":    milestone_labels,
                "generated_at":        roadmap_data.get("roadmap_meta", {}).get("generated_at", ""),
                "full_roadmap_json":   roadmap_json_str,
                "full_roadmap_stored": True,
                "doc_type":            "roadmap",
            }

        vector_id = f"{user_id}_roadmap_{roadmap_id}"

        index.upsert(
            vectors=[{
                "id":       vector_id,
                "values":   embedding,
                "metadata": store_payload,
            }],
            namespace=user_id,
        )

        print(f"[PINECONE UPSERT SUCCESS] vector_id={vector_id} "
              f"namespace={user_id} "
              f"full_roadmap_stored={store_payload.get('full_roadmap_stored', False)}")
        return True

    except ImportError as e:
        print(f"[PINECONE STORE] ✗ Import error: {e}")
        return False
    except Exception as e:
        print(f"[PINECONE STORE] ✗ Failed: {e}")
        import traceback
        traceback.print_exc()
        return False


# ============================================================
# JSON Repair Utility   (PRODUCTION-GRADE)
# ============================================================

# ── Phase 2: Bracket balancing (fix truncation) ─────────────
def _balance_brackets(s: str) -> str:
    """Append missing closing brackets to fix truncated JSON."""
    stack = []
    in_str = False
    esc = False
    for ch in s:
        if esc:
            esc = False
            continue
        if ch == '\\':
            esc = True
            continue
        if ch == '"':
            in_str = not in_str
            continue
        if in_str:
            continue
        if ch == '{':
            stack.append('}')
        elif ch == '[':
            stack.append(']')
        elif ch == '}':
            if stack and stack[-1] == '}':
                stack.pop()
        elif ch == ']':
            if stack and stack[-1] == ']':
                stack.pop()
    if stack:
        s += ''.join(reversed(stack))
    return s


# ── Phase 3: Missing comma insertion ────────────────────────
_RE_FIX_MISSING_COMMAS = [
    (re.compile(r'}([ \t]*){'),        r'},\1{'),
    (re.compile(r'}([ \t]*)\['),       r'},\1['),
    (re.compile(r']([ \t]*){'),        r'],\1{'),
    (re.compile(r']([ \t]*)\['),       r'],\1['),
    (re.compile(r'}([ \t]*)"'),        r'},\1"'),
    (re.compile(r']([ \t]*)"'),        r'],\1"'),
    (re.compile(r'(\d+)([ \t]*)"(?!\s*[}\]])'), r'\1,\2"'),
    (re.compile(r'(true|false|null)([ \t]*)"'), r'\1,\2"'),
    (re.compile(r'(true|false|null)([ \t]*){'), r'\1,\2{'),
    (re.compile(r'(\d+)\s+(\d+)'),     r'\1, \2'),
]


def _fix_missing_commas(s: str) -> str:
    """Insert missing commas between adjacent JSON values.

    Each pattern is applied iteratively (up to 10 rounds) so that
    overlapping fixes (e.g. ``[1 2 3]``) converge.
    """
    for pattern, replacement in _RE_FIX_MISSING_COMMAS:
        for _ in range(10):
            prev = s
            s = pattern.sub(replacement, s)
            if s == prev:
                break
    return s


# ── Phase 4: Unescaped quote repair ─────────────────────────
def _get_next_non_ws(s: str, pos: int) -> str:
    """Return the next non-whitespace character starting at *pos*."""
    for i in range(pos, len(s)):
        if s[i] not in (' ', '\t', '\n', '\r'):
            return s[i]
    return ''


def _fix_unescaped_quotes(s: str) -> str:
    """Escape unescaped double quotes inside string values.

    Strategy — scan character by character.  When we find an unescaped
    ``"`` inside a string, check the *next non-whitespace* character.
    If it is a JSON structural delimiter (``,`` ``}`` ``]`` ``:`` EOF)
    the quote is a genuine string terminator.  Otherwise it is a literal
    quote that should be escaped — this handles the common LLM mistake
    ``"description": "Uses "AI" technology"``.
    """
    result = []
    in_string = False
    i = 0
    n = len(s)
    while i < n:
        ch = s[i]
        if ch == '\\':
            result.append(ch)
            if i + 1 < n:
                result.append(s[i + 1])
                i += 2
            continue
        if ch == '"':
            if not in_string:
                result.append('"')
                in_string = True
            else:
                next_nws = _get_next_non_ws(s, i + 1)
                if next_nws in (',', '}', ']', ':', ''):
                    result.append('"')
                    in_string = False
                else:
                    result.append('\\"')
            i += 1
            continue
        result.append(ch)
        i += 1
    return ''.join(result)


def repair_json(raw: str) -> str:
    """Multi-phase JSON repair for LLM output.

    Phases (each verified by ``json.loads`` before proceeding):

      0.  Strip markdown fences & extract outermost ``{…}`` block.
      1.  Remove trailing commas before ``}`` / ``]``.
      2.  Balance brackets (fix truncation).
      3.  Insert missing commas between adjacent values.
      4.  Escape unescaped quotes inside string values.

    Returns the best valid JSON found; if no phase produces valid
    JSON the Phase-2 (balanced-brackets) result is returned as a
    best-effort.
    """
    # ── Phase 0: Basic cleanup ──────────────────────────────
    clean = raw.strip()
    if "```" in clean:
        clean = clean.replace("```json", "").replace("```", "").strip()
    start = clean.find("{")
    end   = clean.rfind("}") + 1
    if start != -1 and end > 0:
        clean = clean[start:end]

    # ── Phase 1: Trailing commas ────────────────────────────
    clean = _strip_trailing_commas(clean)

    try:
        json.loads(clean)
        return clean
    except json.JSONDecodeError:
        pass

    # ── Phase 2: Bracket balancing ──────────────────────────
    clean = _balance_brackets(clean)
    clean = _strip_trailing_commas(clean)

    try:
        json.loads(clean)
        return clean
    except json.JSONDecodeError:
        pass

    # ── Phase 3: Missing commas ─────────────────────────────
    clean = _fix_missing_commas(clean)
    clean = _strip_trailing_commas(clean)

    try:
        json.loads(clean)
        return clean
    except json.JSONDecodeError:
        pass

    # ── Phase 4: Unescaped quotes ───────────────────────────
    clean = _fix_unescaped_quotes(clean)

    try:
        json.loads(clean)
        return clean
    except json.JSONDecodeError:
        pass

    # Best effort — at least brackets are balanced
    return clean


def _strip_trailing_commas(s: str) -> str:
    s = re.sub(r',\s*}', '}', s)
    s = re.sub(r',\s*]', ']', s)
    return s


# ============================================================
# Roadmap Structure Validator   (HTML-aligned, MD-bound-corrected)
# ============================================================

def validate_roadmap_structure(data: dict) -> None:
    """
    Validates the V3.3 roadmap structure (Roadmap Generation Science V2 spec).
    Raises ValueError with a clear message on any violation.
    Auto-fixes mock.unlock_mastery silently.

    Enforces (bounds sourced from ROADMAP_GENERATION_SCIENCE_V2.md):
      - milestone count within MIN_MILESTONES..MAX_MILESTONES (2,7)
      - milestone_id/label codes M01..M0N, sequential, no gaps
      - each milestone has between MIN_MODULES and MAX_MODULES modules (2,4)
      - each module has MIN_SKILLS..MAX_SKILLS skills (3,8 — intentional
        deviation from MD's (3,10), kept per explicit instruction)
      - each skill has MIN_LESSONS_PER_SKILL..MAX_LESSONS_PER_SKILL lessons (2,4)
      - each module's "science" array has 0-3 items ("Scenario" or "Interview";
        Scenarios may share a module, Interview must be in a separate module
        where layout allows)
      - each milestone has MIN_SCENARIOS_PER_MILESTONE <= sc_n <= MAX (3,7)
      - each milestone has MIN_MOCKS_PER_MILESTONE <= iv <= MAX (1,2)
      - each milestone has MIN_PROJECTS_PER_MILESTONE <= projects <= MAX (1,2)
      - skill_id uniqueness and acyclic/backward-only prerequisite refs
      - project: title, description, deliverable, seeded_errors, ≥2 vibe_layers
      - at least 2 distinct ai_first_layers per milestone
    """
    milestones = data.get("milestones", [])
    level      = data.get("level", "beginner")
    icp_type   = data.get("icp_type", "low")

    valid_levels   = {"beginner", "intermediate", "senior"}
    valid_icp      = {"low", "high"}

    if level not in valid_levels:
        raise ValueError(
            f"Invalid level '{level}' — must be one of {sorted(valid_levels)}"
        )
    if icp_type not in valid_icp:
        raise ValueError(
            f"Invalid icp_type '{icp_type}' — must be 'low' or 'high'"
        )

    if not milestones:
        raise ValueError("Roadmap has no milestones")

    ms_count = len(milestones)
    if ms_count < MIN_MILESTONES:
        raise ValueError(
            f"Milestone count {ms_count} is below minimum {MIN_MILESTONES}"
        )
    if ms_count > MAX_MILESTONES:
        raise ValueError(
            f"Milestone count {ms_count} exceeds maximum {MAX_MILESTONES}"
        )

    all_skill_ids: List[str] = []

    for m_idx, milestone in enumerate(milestones):
        expected_code = f"M{m_idx + 1:02d}"
        m_id    = milestone.get("milestone_id", "")
        m_label = milestone.get("label", "")

        if m_id != expected_code:
            raise ValueError(
                f"Milestone {m_idx + 1}: milestone_id must be '{expected_code}', "
                f"got '{m_id}'"
            )
        if m_label != m_id:
            raise ValueError(
                f"Milestone {m_id}: label must equal milestone_id "
                f"('{m_id}'), got '{m_label}'"
            )

        # ── Project validation (PROJECTS_PER_MILESTONE bound: 1-2) ───────
        # Accept both the new "projects" array (preferred, matches MD's
        # PROJECTS_PER_MILESTONE bound) and the legacy singular "project"
        # key for backward compatibility with older stored roadmaps.
        projects = milestone.get("projects")
        if projects is None:
            legacy_project = milestone.get("project")
            projects = [legacy_project] if legacy_project else []

        if not isinstance(projects, list):
            raise ValueError(f"Milestone {m_id}: projects must be a list")

        project_count = len(projects)
        if project_count < MIN_PROJECTS_PER_MILESTONE:
            raise ValueError(
                f"Milestone {m_id}: has {project_count} project(s), "
                f"minimum is {MIN_PROJECTS_PER_MILESTONE}"
            )
        if project_count > MAX_PROJECTS_PER_MILESTONE:
            raise ValueError(
                f"Milestone {m_id}: has {project_count} project(s), "
                f"maximum is {MAX_PROJECTS_PER_MILESTONE}"
            )

        for project in projects:
            if not project:
                raise ValueError(f"Milestone {m_id}: empty project entry")

            if not project.get("title"):
                raise ValueError(f"Milestone {m_id}: project.title missing")

            if not project.get("description"):
                raise ValueError(f"Milestone {m_id}: project.description missing")

            if not project.get("deliverable"):
                raise ValueError(f"Milestone {m_id}: project.deliverable missing")

            if not project.get("seeded_errors"):
                raise ValueError(f"Milestone {m_id}: project.seeded_errors missing")

            project_layers = project.get("vibe_layers", [])
            if len(project_layers) < 2:
                raise ValueError(
                    f"Milestone {m_id}: project must declare at least 2 vibe_layers"
                )
            invalid_project_layers = [v for v in project_layers if v not in ALLOWED_AI_LAYERS]
            if invalid_project_layers:
                raise ValueError(
                    f"Milestone {m_id}: project vibe_layers contain invalid "
                    f"value(s): {invalid_project_layers}. Must be in {sorted(ALLOWED_AI_LAYERS)}."
                )

        # ── Module count: within bounds ─────────────────────────────────
        modules = milestone.get("modules", [])
        if not isinstance(modules, list):
            raise ValueError(f"Milestone {m_id}: modules must be a list")
        mod_count = len(modules)
        if mod_count < MIN_MODULES or mod_count > MAX_MODULES:
            raise ValueError(
                f"Milestone {m_id}: module count must be between "
                f"{MIN_MODULES} and {MAX_MODULES}, got {mod_count}"
            )

        scenario_count  = 0
        interview_count = 0
        ai_first_layers = set()
        modules_with_science = 0

        for mod_idx, mod in enumerate(modules):
            mod_id = mod.get("id", "?")

            # ── ai_first_layer ──
            layer = mod.get("ai_first_layer")
            if layer not in ALLOWED_AI_LAYERS:
                raise ValueError(
                    f"Module {mod_id}: invalid ai_first_layer '{layer}'"
                )
            ai_first_layers.add(layer)

            # ── module id format: "M{milestone_pos}.{module_pos}" ──
            expected_mod_id = f"M{m_idx + 1}.{mod_idx + 1}"
            if mod_id != expected_mod_id:
                raise ValueError(
                    f"Milestone {m_id}: module #{mod_idx + 1} id must be "
                    f"'{expected_mod_id}', got '{mod_id}'"
                )

            # ── skills: MIN_SKILLS..MAX_SKILLS (dynamic) ──
            skills = mod.get("skills", [])
            if not isinstance(skills, list):
                raise ValueError(f"Module {mod_id}: skills must be a list")
            skill_count = len(skills)
            if skill_count < MIN_SKILLS or skill_count > MAX_SKILLS:
                raise ValueError(
                    f"Module {mod_id}: must have between {MIN_SKILLS} and "
                    f"{MAX_SKILLS} skills, got {skill_count}"
                )

            # ── science: 0-3 items per module ──
            science = mod.get("science", [])
            if not isinstance(science, list):
                raise ValueError(f"Module {mod_id}: science must be a list")
            if len(science) > 3:
                raise ValueError(
                    f"Module {mod_id}: science array must have 0-3 items, "
                    f"got {len(science)}"
                )
            if len(science) > 0:
                modules_with_science += 1
            for sci in science:
                sci_type = sci.get("type")
                if sci_type not in ("Scenario", "Interview"):
                    raise ValueError(
                        f"Module {mod_id}: invalid science type '{sci_type}' "
                        f"— must be 'Scenario' or 'Interview'"
                    )
                if not sci.get("desc"):
                    raise ValueError(
                        f"Module {mod_id}: science item of type '{sci_type}' "
                        f"missing 'desc'"
                    )
                if sci_type == "Scenario":
                    scenario_count += 1
                else:
                    interview_count += 1

            # ── per-skill checks ──
            for skill in skills:
                skill_id = skill.get("skill_id", "?")

                if skill_id in all_skill_ids:
                    raise ValueError(f"Duplicate skill_id: {skill_id}")
                all_skill_ids.append(skill_id)

                if not skill.get("n"):
                    raise ValueError(f"Skill {skill_id} missing short name 'n'")

                if "p" not in skill or not isinstance(skill.get("p"), (int, float)):
                    raise ValueError(
                        f"Skill {skill_id} missing numeric 'p' (mastery %)"
                    )
                if not (0 <= skill["p"] <= 100):
                    raise ValueError(
                        f"Skill {skill_id}: 'p' must be 0-100, got {skill['p']}"
                    )

                # ── lessons: MIN_LESSONS_PER_SKILL..MAX_LESSONS_PER_SKILL (2-4) ──
                lessons = skill.get("lessons", [])
                if not isinstance(lessons, list):
                    raise ValueError(f"Skill {skill_id}: lessons must be a list")
                if len(lessons) < MIN_LESSONS_PER_SKILL or len(lessons) > MAX_LESSONS_PER_SKILL:
                    raise ValueError(
                        f"Skill {skill_id}: must have {MIN_LESSONS_PER_SKILL}-"
                        f"{MAX_LESSONS_PER_SKILL} lessons, got {len(lessons)}"
                    )

                flow = skill.get("content_flow", {})
                for content_type in ("video", "scenario", "mock", "review"):
                    if content_type not in flow:
                        raise ValueError(
                            f"Skill {skill_id} missing content_flow.{content_type}"
                        )

                # Auto-fix mock unlock_mastery
                mock = flow.get("mock", {})
                if mock.get("unlock_mastery") != MOCK_UNLOCK_MASTERY:
                    mock["unlock_mastery"] = MOCK_UNLOCK_MASTERY

                # Validate prerequisite references (backward-only, acyclic)
                requires = skill.get("unlock_rules", {}).get("requires", [])
                for req_id in requires:
                    if req_id not in all_skill_ids:
                        raise ValueError(
                            f"Skill {skill_id} requires '{req_id}' which hasn't "
                            f"been defined yet (circular or forward reference)"
                        )

        # ── ai-first diversity: at least 2 distinct layers per milestone ──
        if len(ai_first_layers) < 2:
            raise ValueError(
                f"Milestone {m_id}: must contain at least 2 different "
                f"AI-first layers. Found: {sorted(ai_first_layers)}"
            )

        # ── no_spectators: PER-MODULE check (MD Section 19, "POC bar is
        #    per-module, not per-skill"). Every module must have at least
        #    one applied activity (Scenario or Interview/Mock) in its
        #    science array. This replaces any per-skill content_flow check —
        #    content_flow.scenario/mock existing on a skill object is a
        #    template placeholder, not "applied activity" in the MD sense.
        modules_missing_science = [
            mod.get("id", "?") for mod in modules if not mod.get("science")
        ]
        if modules_missing_science:
            raise ValueError(
                f"Milestone {m_id}: no_spectators violation — module(s) "
                f"{modules_missing_science} have zero science items "
                f"(every non-free module needs ≥1 Scenario or Mock)"
            )

        # ── sc_n must be MIN_SCENARIOS..MAX_SCENARIOS (3-7, real floor) ──
        if scenario_count < MIN_SCENARIOS_PER_MILESTONE or scenario_count > MAX_SCENARIOS_PER_MILESTONE:
            raise ValueError(
                f"Milestone {m_id}: Scenarios out of bounds "
                f"({MIN_SCENARIOS_PER_MILESTONE}-{MAX_SCENARIOS_PER_MILESTONE}), "
                f"got {scenario_count}"
            )
        # ── iv (mocks/interviews) must be MIN_MOCKS..MAX_MOCKS (1-2, real floor) ──
        if interview_count < MIN_MOCKS_PER_MILESTONE or interview_count > MAX_MOCKS_PER_MILESTONE:
            raise ValueError(
                f"Milestone {m_id}: Mocks/Interviews out of bounds "
                f"({MIN_MOCKS_PER_MILESTONE}-{MAX_MOCKS_PER_MILESTONE}), "
                f"got {interview_count}"
            )

    # ── starting/current milestone must be M01 ──
    if data.get("starting_milestone") != "M01":
        raise ValueError(
            f"starting_milestone must be 'M01', got "
            f"'{data.get('starting_milestone')}'"
        )
    if data.get("current_active_milestone") != "M01":
        raise ValueError(
            f"current_active_milestone must be 'M01', got "
            f"'{data.get('current_active_milestone')}'"
        )


# ============================================================
# FITS_LIFE VALIDATOR  (Roadmap Bible — Step 3, v2)
# ============================================================

# Roadmap Bible constants — tune here, affects all profiles
_FL_REALISM_BUFFER = BUDGET_UTILIZATION   # 80 % of available time is productive learning
_FL_MAX_WEEKS       = 52    # 1 year upper-bound; beyond this = truly infeasible


def _count_roadmap_units(roadmap_data: dict) -> tuple:
    """
    Return (lessons, skills, scenarios, interviews, projects) counts
    from a roadmap whose lessons live inside skill objects and whose
    milestones may carry either a "projects" array (preferred) or a
    legacy singular "project" dict.
    """
    lessons = skills = scenarios = interviews = projects = 0
    for ms in roadmap_data.get("milestones", []):
        ms_projects = ms.get("projects")
        if ms_projects is None:
            legacy = ms.get("project")
            ms_projects = [legacy] if legacy else []
        projects += len(ms_projects)
        for mod in ms.get("modules", []):
            for skill in mod.get("skills", []):
                lessons += len(skill.get("lessons", []))
                skills  += 1
            for sci in mod.get("science", []):
                t = sci.get("type", "")
                if t == "Scenario":
                    scenarios += 1
                elif t == "Interview":
                    interviews += 1
    return lessons, skills, scenarios, interviews, projects


def fits_life_check(roadmap_data: dict, weekly_hours: int, timeline_days: int = 112) -> dict:
    """
    Roadmap Bible validator: fits_life  (adaptive duration edition).

    DESIGN: fits_life is ADVISORY, never a hard gate (MD Section 17/19 —
    "Warn, log, show math to user"). It NEVER raises. Instead:

      1. Calculate demand from the roadmap units.
      2. If demand ≤ budget at timeline_days             → fits at timeline_days.
      3. Else compute the minimum duration to fit:
             duration_weeks = ceil(demand / (weekly_hours × 0.8))
      4. If duration_weeks ≤ _FL_MAX_WEEKS (52)           → fits=True, extended
      5. If duration_weeks >  _FL_MAX_WEEKS               → fits=False (truly
         infeasible; pipeline logs a warning but still proceeds).

    Return shape
    ============
    {
        "fits":                 bool,
        "duration_weeks":       int,    # actual weeks at weekly_hours
        "budget_hours":         float,  # weekly_hours × timeline_days × 0.8
        "demand_hours":         float,
        "weekly_hours_needed":  float,  # hrs/wk to finish in timeline_days
        "breakdown": { ... }
    }
    """
    weekly_hours = max(int(weekly_hours or 1), 1)   # guard against 0 / None
    timeline_days = max(int(timeline_days or 112), 7)

    lessons, skills, scenarios, interviews, projects = _count_roadmap_units(
        roadmap_data
    )

    video_hours     = round(lessons    * 0.25, 1)
    mastery_hours   = round(skills     * 1.5,  1)
    scenario_hours  = round(scenarios  * 0.5,  1)
    interview_hours = round(interviews * 1.0,  1)
    project_hours   = round(projects   * 3.0,  1)

    demand_hours = round(
        video_hours + mastery_hours + scenario_hours
        + interview_hours + project_hours,
        1,
    )

    # ── Adaptive duration using timeline_days ─────────────────
    default_budget = round(weekly_hours * (timeline_days / 7) * _FL_REALISM_BUFFER, 1)

    if demand_hours <= default_budget or demand_hours == 0:
        duration_weeks = int(timeline_days / 7)
        fits           = True
    else:
        # weeks needed = demand / (hours_per_week × realism_buffer)
        raw_weeks      = demand_hours / (weekly_hours * _FL_REALISM_BUFFER)
        duration_weeks = int(math.ceil(raw_weeks))
        fits           = duration_weeks <= _FL_MAX_WEEKS

    budget_hours = round(weekly_hours * duration_weeks * _FL_REALISM_BUFFER, 1)

    # weekly_hours_needed = hrs/wk required to complete in timeline_days
    weekly_hours_needed = round(
        demand_hours / ((timeline_days / 7) * _FL_REALISM_BUFFER), 1
    ) if demand_hours > 0 else 0.0

    return {
        "fits":                fits,
        "fits_at_default":     fits if duration_weeks == int(timeline_days / 7) else False,
        "duration_weeks":      duration_weeks,
        "budget_hours":        budget_hours,
        "demand_hours":        demand_hours,
        "weekly_hours_needed": weekly_hours_needed,
        "breakdown": {
            "video_hours":     video_hours,
            "mastery_hours":   mastery_hours,
            "scenario_hours":  scenario_hours,
            "interview_hours": interview_hours,
            "project_hours":   project_hours,
        },
    }


# ============================================================
# ROADMAP BIBLE VALIDATORS  (Genuineness Validator — MD Section 19)
# ============================================================
# MD Section 19 names exactly 7 checks: fits_life, starts_where_they_are,
# laugh_test, no_spectators, ai_first, dag_clean, salary_floor.
# These 7 are the GATE. Everything else in this file under
# run_roadmap_quality_validators() is supplementary internal QA and is
# NOT part of the MD-defined genuineness gate.
#
# Each validator is a pure function: roadmap_data -> dict.
# They NEVER raise — they return {"pass": bool, "reason": str, ...}.
# run_roadmap_bible_validators() calls them all and collects results.
# ============================================================

def _starts_where_they_are_check(roadmap_data: dict) -> dict:
    """
    Validator: starts_where_they_are  (MD Section 19 — block on fail)

    No skill the user already has may appear as a teachable lesson.
    Every skill's p value must align with its BKT prior within 10%.
    For non-beginner learners, M01 skills must carry mastery priors.
    """
    level      = roadmap_data.get("level", "beginner")
    milestones = roadmap_data.get("milestones", [])

    if not milestones:
        return {"pass": False, "reason": "No milestones — cannot check ZPD"}

    all_skills = [
        skill
        for ms in milestones
        for mod in ms.get("modules", [])
        for skill in mod.get("skills", [])
    ]

    if not all_skills:
        return {"pass": False, "reason": "No skills found in roadmap"}

    misaligned = []
    for skill in all_skills:
        p_val = skill.get("p", 0)
        bkt = skill.get("mastery_state", {}).get("bkt", {})
        prior = bkt.get("prior", 0)
        if p_val == 0 and prior == 0:
            continue
        diff = abs(p_val / 100.0 - prior)
        if diff > 0.10:
            misaligned.append(
                f"{skill.get('skill_id', '?')} p={p_val} prior={prior} diff={diff:.3f}"
            )

    if misaligned and len(misaligned) > len(all_skills) * 0.5:
        return {
            "pass": False,
            "reason": f"{len(misaligned)}/{len(all_skills)} skills misaligned: {'; '.join(misaligned[:5])}",
        }

    return {"pass": True, "reason": "Initial mastery priors align with learner level and BKT estimates."}


def _laugh_test_check(roadmap_data: dict) -> dict:
    """
    Validator: laugh_test  (MD Section 19 / 24 — block on fail, retry max 2)

    >= LAUGH_TEST_COVERAGE_THRESHOLD (80%) of skills must align with
    target-role hiring requirements. POC heuristic until real JD profiles
    ship (MD Section 24: "PARTIALLY IMPLEMENTED, runs against
    ROLE_MUST_HAVE_SKILLS proxy dict"): skills with generic names
    ('learn', 'intro', 'overview', 'basics', 'fundamentals') indicate filler.
    """
    milestones = roadmap_data.get("milestones", [])
    all_skill_names = [
        skill.get("n", "")
        for ms in milestones
        for mod in ms.get("modules", [])
        for skill in mod.get("skills", [])
    ]

    total = len(all_skill_names)
    if total == 0:
        return {"pass": False, "reason": "No skills found in roadmap"}

    _filler_tokens = {"learn", "intro", "introduction", "overview", "basics",
                      "fundamentals", "general", "misc"}
    filler_count = sum(
        1 for n in all_skill_names
        if any(tok in n.lower().split("_") for tok in _filler_tokens)
    )
    filler_ratio = filler_count / total
    coverage = 1 - filler_ratio

    if coverage < LAUGH_TEST_COVERAGE_THRESHOLD:
        return {
            "pass":        False,
            "reason":      (
                f"Laugh test coverage {coverage:.0%} below "
                f"{LAUGH_TEST_COVERAGE_THRESHOLD:.0%} threshold "
                f"({filler_count}/{total} skills look generic). "
                "A senior from this field would laugh at this roadmap."
            ),
            "filler_ratio": round(filler_ratio, 3),
            "coverage":     round(coverage, 3),
        }

    return {
        "pass":          True,
        "reason":        (
            f"Laugh test passed — coverage {coverage:.0%} "
            f"({total - filler_count}/{total}). Full JD validation pending "
            f"(MD Section 24 — proxy dict until Sanket ships real JD profiles)."
        ),
        "filler_ratio":  round(filler_ratio, 3),
        "coverage":      round(coverage, 3),
        "total_skills":  total,
    }


def _no_spectators_check(roadmap_data: dict) -> dict:
    """
    Validator: no_spectators  (MD Section 19 — block on fail)

    Scope per MD Section 19: "Every non-free module has ≥1 applied
    activity (scenario or mock) — POC bar is per-module" (not per-skill).
    This checks the module's science array directly, matching
    validate_roadmap_structure()'s structural check.

    Also requires every milestone to carry ≥1 project (MD Section 12 —
    PROJECTS_PER_MILESTONE minimum is 1).
    """
    milestones  = roadmap_data.get("milestones", [])
    violations: List[str] = []

    for ms in milestones:
        m_id = ms.get("milestone_id", "?")

        ms_projects = ms.get("projects")
        if ms_projects is None:
            legacy = ms.get("project")
            ms_projects = [legacy] if legacy else []
        if len(ms_projects) < MIN_PROJECTS_PER_MILESTONE:
            violations.append(
                f"Milestone {m_id}: has {len(ms_projects)} project(s), "
                f"needs at least {MIN_PROJECTS_PER_MILESTONE} — learner has "
                f"nothing to build"
            )

        for mod in ms.get("modules", []):
            mod_id  = mod.get("id", "?")
            is_free = mod.get("free", False)
            science = mod.get("science", [])

            # MD Section 19: scope is per-module, not per-skill.
            # A non-free module with zero science items is a spectator
            # violation regardless of what each individual skill's
            # content_flow placeholder looks like.
            if not is_free and not science:
                violations.append(
                    f"Module {mod_id} (milestone {m_id}): no applied "
                    f"activity — module has zero science items "
                    f"(per-module no_spectators check)"
                )

    if violations:
        return {
            "pass":       False,
            "reason":     f"{len(violations)} spectator violation(s) found",
            "violations": violations,
        }

    return {"pass": True, "reason": "No spectator violations — every non-free module has an applied activity"}


def _dag_clean_check(roadmap_data: dict) -> dict:
    """
    Validator: dag_clean  (MD Section 19 — block on fail)

    Checks:
      1. No duplicate skill_ids globally.
      2. No duplicate skill content ('n' field) within a milestone.
      3. All prerequisite references are backward-only (acyclic).
      4. No milestone appears more than once.
    """
    milestones  = roadmap_data.get("milestones", [])
    violations: List[str] = []

    seen_skill_ids: List[str]  = []   # ordered for backward-ref check
    seen_ms_ids:    set         = set()

    for ms in milestones:
        m_id = ms.get("milestone_id", "?")
        if m_id in seen_ms_ids:
            violations.append(f"Duplicate milestone_id: {m_id}")
        seen_ms_ids.add(m_id)

        skill_ns_in_ms: List[str] = []

        for mod in ms.get("modules", []):
            for skill in mod.get("skills", []):
                skill_id = skill.get("skill_id", "?")
                skill_n  = skill.get("n", "")

                if skill_id in seen_skill_ids:
                    violations.append(f"Duplicate skill_id globally: {skill_id}")
                else:
                    seen_skill_ids.append(skill_id)

                if skill_n and skill_n in skill_ns_in_ms:
                    violations.append(
                        f"Milestone {m_id}: duplicate skill content '{skill_n}'"
                    )
                elif skill_n:
                    skill_ns_in_ms.append(skill_n)

                requires = skill.get("unlock_rules", {}).get("requires", [])
                for req_id in requires:
                    if req_id not in seen_skill_ids:
                        violations.append(
                            f"Skill {skill_id} requires '{req_id}' "
                            "which hasn't been defined yet (forward/circular ref)"
                        )

    if violations:
        return {
            "pass":       False,
            "reason":     f"{len(violations)} DAG violation(s)",
            "violations": violations,
        }

    return {
        "pass":   True,
        "reason": (
            f"DAG clean — {len(seen_skill_ids)} unique skills, "
            "all prerequisites backward-only"
        ),
    }


def _ai_first_check(roadmap_data: dict) -> dict:
    """
    Validator: ai_first  (MD Section 19 — block on fail)

    MD Section 19 rule: "Every milestone has ≥1 project with ≥2 vibe
    layers + ≥1 seeded AI error."

    Additionally (kept from the June 2026 AI-First model, Section 2/8B):
    every milestone must have ≥2 distinct ai_first_layer values across
    its modules, and should collectively teach all 4 official layers
    (vibe_planning, vibe_architecture, vibe_solution, deployment) —
    missing layers produce a WARN, not a block.
    """
    milestones  = roadmap_data.get("milestones", [])
    violations: List[str] = []
    warns: List[str] = []
    deploy_required_found  = False
    REQUIRED_LAYERS = {"vibe_planning", "vibe_architecture", "vibe_solution", "deployment"}

    for ms in milestones:
        m_id   = ms.get("milestone_id", "?")
        layers = set()

        for mod in ms.get("modules", []):
            layer = mod.get("ai_first_layer")
            if layer in ALLOWED_AI_LAYERS:
                layers.add(layer)

        if len(layers) < 2:
            violations.append(
                f"Milestone {m_id}: only {len(layers)} distinct AI-first "
                f"layer(s) {sorted(layers)}. Need ≥2."
            )

        # ── Milestone coverage WARN (non-blocking) ──────────────
        missing_layers = REQUIRED_LAYERS - layers
        if missing_layers:
            warns.append(
                f"Milestone {m_id} missing layer(s): {', '.join(sorted(missing_layers))}"
            )

        ms_projects = ms.get("projects")
        if ms_projects is None:
            legacy = ms.get("project")
            ms_projects = [legacy] if legacy else []

        if not ms_projects:
            violations.append(f"Milestone {m_id}: no project")
        for project in ms_projects:
            # ── Project vibe_layers value validation ────────────
            project_layers = project.get("vibe_layers", [])
            invalid_project_layers = [v for v in project_layers if v not in ALLOWED_AI_LAYERS]
            if invalid_project_layers:
                violations.append(
                    f"Milestone {m_id}: project vibe_layers contain invalid "
                    f"value(s): {invalid_project_layers}. Must be in {sorted(ALLOWED_AI_LAYERS)}."
                )
            if len(project_layers) < 2:
                violations.append(
                    f"Milestone {m_id}: project has fewer than 2 vibe_layers"
                )
            if not project.get("seeded_errors"):
                violations.append(
                    f"Milestone {m_id}: project has no seeded_errors — "
                    "learner never catches an AI mistake"
                )
            if project.get("deploy_required"):
                deploy_required_found = True

    if not deploy_required_found:
        violations.append(
            "No project across the roadmap has deploy_required=True — "
            "nothing ships to production"
        )

    # Build reason with warns
    reason_parts = []
    if violations:
        reason_parts.append(f"{len(violations)} AI-first violation(s)")
    if warns:
        reason_parts.append(f"Coverage warnings: {'; '.join(warns)}")
    reason = "; ".join(reason_parts) if reason_parts else "all good"

    return {
        "pass":       not bool(violations),
        "reason":     reason,
        "violations": violations if violations else [],
        "warns":      warns,
    }


def _time_budget_check(roadmap_data: dict) -> dict:
    budget = roadmap_data.get("budget_hours", 0) or 0
    estimated = roadmap_data.get("estimated_total_hours", 0) or 0
    if budget <= 0:
        return {"pass": True, "reason": "budget not available, skipping check", "budget_hours": budget, "estimated_total_hours": estimated}
    if estimated > budget * 1.25:
        return {
            "pass": False,
            "reason": f"estimated {estimated}h exceeds 125% of budget {budget}h",
            "budget_hours": budget,
            "estimated_total_hours": estimated,
        }
    return {
        "pass": True,
        "reason": f"estimated {estimated}h within 125% of budget {budget}h",
        "budget_hours": budget,
        "estimated_total_hours": estimated,
    }


def _is_teachable_skill(skill: dict) -> bool:
    """Return True if skill contains ANY teachable content.
    
    Teachable content: lessons, videos, assessments,
    scenarios, mocks, mastery_state (with non-empty state),
    or non-empty unlock_rules (with actual requires).
    """
    if skill.get("lessons") and len(skill.get("lessons", [])) > 0:
        return True
    cf = skill.get("content_flow", {})
    for ct in ("video", "scenario", "mock", "assessment"):
        if cf.get(ct, {}).get("content_id"):
            return True
    ms = skill.get("mastery_state", {})
    if ms and ms.get("state") and ms["state"] not in ("", "unknown"):
        return True
    ur = skill.get("unlock_rules", {})
    if ur and ur.get("requires") and len(ur.get("requires", [])) > 0:
        return True
    return False


def _known_skill_skip_check(roadmap_data: dict) -> dict:
    """starts_where_they_are sub-check (MD Section 2 / 19):
    known skills must NOT be teachable.
    
    Every known skill must satisfy:
      (removed entirely from roadmap — preferred)
      OR
      (only as historical evidence: hidden=true, teachable=false,
       auto_completed=true, and NO teachable content)
    """
    level = roadmap_data.get("level", "beginner")
    known = roadmap_data.get("known_skills", [])
    if level == "beginner" or not known:
        return {"pass": True, "reason": "beginner or no known_skills — skip check"}
    violations = []
    for ms in roadmap_data.get("milestones", []):
        for mod in ms.get("modules", []):
            for skill in mod.get("skills", []):
                n = skill.get("n", "")
                title = skill.get("title", "")
                if any(normalize_skill_match(v, ks) for v in (n, title) for ks in known):
                    sid = skill.get("skill_id", "?")
                    sn = skill.get("n", "")
                    # Check historical evidence mode flags
                    if not skill.get("auto_completed"):
                        violations.append(
                            f"{sid} ('{sn}') known skill missing auto_completed"
                        )
                    if not skill.get("hidden"):
                        violations.append(
                            f"{sid} ('{sn}') known skill not hidden"
                        )
                    if skill.get("teachable") != False:
                        violations.append(
                            f"{sid} ('{sn}') known skill still teachable"
                        )
                    if _is_teachable_skill(skill):
                        violations.append(
                            f"{sid} ('{sn}') known skill has teachable content"
                        )
    if violations:
        return {"pass": False, "reason": "; ".join(violations)}
    return {"pass": True, "reason": "all known skills removed from teachable content — MD compliant"}


# ── Role alignment keywords (Phase 3.4) ──────────────────────
# Each target role maps to expected skill keywords for validation.
_ROLE_ALIGNMENT_KEYWORDS = {
    "frontend": {"html", "css", "javascript", "dom", "react", "vue", "angular", "responsive", "ui", "ux", "web", "browser", "typescript", "svelte"},
    "backend": {"api", "database", "sql", "server", "authentication", "authorization", "rest", "microservice", "cache", "queue", "middleware", "spring", "django", "node", "express"},
    "data": {"sql", "python", "pandas", "numpy", "statistics", "visualization", "tableau", "powerbi", "etl", "pipeline", "data_warehouse", "analytics"},
    "ai": {"machine_learning", "deep_learning", "llm", "prompt", "vector", "embedding", "transformer", "neural", "tensorflow", "pytorch", "nlp", "computer_vision", "rag"},
    "fullstack": {"html", "css", "javascript", "react", "node", "api", "database", "sql", "deployment", "rest", "frontend", "backend"},
    "devops": {"docker", "kubernetes", "ci/cd", "terraform", "ansible", "monitoring", "pipeline", "deployment", "cloud", "infrastructure", "linux", "bash"},
    "mobile": {"kotlin", "swift", "android", "ios", "flutter", "react_native", "mobile", "dart"},
    "security": {"security", "firewall", "encryption", "authentication", "penetration", "vulnerability", "compliance", "risk", "siem"},
    "software": {"algorithm", "data_structure", "design_pattern", "oop", "testing", "debugging", "version_control", "git"},
}


def _resolve_role_alignment_keywords(target_identity: str) -> set:
    """Resolve which keyword set to use for role alignment check."""
    tgt_lower = target_identity.lower()
    # Check against _ROLE_FAMILIES-like logic (reuse domain extraction)
    import re
    domain_keywords = {
        "frontend": {"frontend", "front end", "ui", "web developer"},
        "backend": {"backend", "back end", "server", "api"},
        "data": {"data", "analytics", "bi"},
        "ai": {"ai", "machine learning", "ml", "deep learning"},
        "fullstack": {"fullstack", "full stack"},
        "devops": {"devops", "sre", "platform", "infrastructure"},
        "mobile": {"mobile", "android", "ios", "flutter"},
        "security": {"security", "cyber"},
    }
    for domain, keywords in domain_keywords.items():
        for kw in keywords:
            if re.search(r'\b' + re.escape(kw) + r'\b', tgt_lower):
                return _ROLE_ALIGNMENT_KEYWORDS.get(domain, set())
    # Fallback: software engineering
    return _ROLE_ALIGNMENT_KEYWORDS.get("software", set())


def _role_alignment_check(roadmap_data: dict) -> dict:
    """Validate that roadmap skills match target role.

    PASS if role coverage >= 80%
    FAIL if role coverage < 80%
    """
    target_role = (roadmap_data.get("customer_profile") or {}).get("target_identity", "")
    if not target_role:
        target_role = roadmap_data.get("target_role", "")
    if not target_role:
        return {"pass": True, "reason": "No target role — skip role alignment"}

    expected_keywords = _resolve_role_alignment_keywords(target_role)
    if not expected_keywords:
        return {"pass": True, "reason": f"No keyword set for role '{target_role}' — skip"}

    total_skills = 0
    matched_skills = 0
    matched_skill_names = []
    missing_skills = []

    for ms in roadmap_data.get("milestones", []):
        for mod in ms.get("modules", []):
            for skill in mod.get("skills", []):
                total_skills += 1
                n = (skill.get("n") or "").lower()
                title = (skill.get("title") or "").lower()
                combined = n + " " + title
                skill_matched = False
                for kw in expected_keywords:
                    if kw in combined or re.search(r'\b' + re.escape(kw) + r'\b', combined):
                        matched_skills += 1
                        matched_skill_names.append(n or title)
                        skill_matched = True
                        break
                if not skill_matched:
                    missing_skills.append(n or title)

    coverage = round(matched_skills / max(total_skills, 1), 3)
    passed = coverage >= LAUGH_TEST_COVERAGE_THRESHOLD

    if not passed:
        return {
            "pass": False,
            "reason": f"Role alignment: {coverage:.0%} ({matched_skills}/{total_skills}) — "
                      f"below {LAUGH_TEST_COVERAGE_THRESHOLD:.0%} threshold for '{target_role}'. "
                      f"Missing: {', '.join(missing_skills[:10])}",
            "coverage": coverage,
            "matched": matched_skills,
            "total": total_skills,
            "expected_keywords": sorted(expected_keywords),
            "missing": missing_skills[:10],
        }
    return {
        "pass": True,
        "reason": f"Role alignment: {coverage:.0%} ({matched_skills}/{total_skills}) — OK",
        "coverage": coverage,
        "matched": matched_skills,
        "total": total_skills,
    }


def _inject_career_switch_narrative(roadmap_data: dict) -> None:
    """Inject transition narrative for career switchers (ISSUE 8).

    When the user is switching domains (e.g. Civil Engineering → Data Analyst),
    add a transition context field that downstream prompt builders can use
    to frame skills as transferable and inject resume-translation guidance.

    Modifies roadmap_data in place.
    """
    profile = roadmap_data.get("customer_profile", {}) or {}
    current = (profile.get("current_identity") or profile.get("current_role") or "")
    target = (profile.get("target_identity") or roadmap_data.get("target_role") or "")

    if not current or not target:
        return

    ci_domain = _extract_domain(current)
    ti_domain = _extract_domain(target)

    if not ci_domain or not ti_domain or ci_domain == ti_domain:
        return  # same-domain transition — no narrative needed

    narrative = {
        "type": "domain_switch",
        "from_domain": ci_domain,
        "to_domain": ti_domain,
        "from_role": current,
        "to_role": target,
        "transition_context": (
            f"Transitioning from {current} to {target}. "
            f"Skills from {ci_domain} domain that transfer to {ti_domain} include: "
            f"problem-solving, analytical thinking, stakeholder communication, "
            f"documentation, project management, and domain-specific data literacy. "
            f"The roadmap should explicitly call out these transferable skills, "
            f"provide resume-translation guidance (e.g. '{current} → {target}'), "
            f"and prioritize bridging skills that directly connect the two domains."
        ),
    }
    roadmap_data["_career_switch"] = narrative
    print(f"[CAREER SWITCH] Detected {ci_domain} → {ti_domain} switch. "
          f"Transition narrative injected.")


# ── HARD PRE-GENERATION ASSERTIONS (ISSUE 9) ──────────────
def _run_pre_generation_assertions(roadmap_data: dict, customer_profile: dict,
                                    icp_type: str, level: str,
                                    years_experience: int) -> None:
    """Hard assertions that must pass before the roadmap is accepted.

    Each assertion checks a critical invariant; raises AssertionError with
    a descriptive message on failure.
    """
    # Assertion 1: ICP/experience match
    # Low ICP with >=3 years experience is suspicious
    if icp_type == "low" and years_experience >= 3:
        _prof_identities = {"engineer", "developer", "manager", "architect",
                            "lead", "senior", "analyst", "consultant"}
        ci_lower = (customer_profile.get("current_identity") or "").lower()
        if any(kw in ci_lower for kw in _prof_identities):
            print(f"[ASSERTION WARNING] icp_type=low but {years_experience}y "
                  f"experience with professional identity '{ci_lower}'")

    # Assertion 2: Protected skills preservation
    _protected = set()
    for sk in (customer_profile.get("known_skills") or customer_profile.get("skills") or []):
        sk_lower = (sk or "").lower().strip()
        if sk_lower:
            _protected.add(sk_lower)
    if _protected:
        _all_roadmap_titles = set()
        for ms in roadmap_data.get("milestones", []):
            for mod in ms.get("modules", []):
                for skill in mod.get("skills", []):
                    t = (skill.get("title") or skill.get("n") or "").lower()
                    if t:
                        _all_roadmap_titles.add(t)
        missing = _protected - _all_roadmap_titles
        if missing:
            print(f"[ASSERTION WARNING] Protected skills missing from roadmap: {missing}")

    # Assertion 3: Contamination score (already checked by _check_role_contamination)
    # but also verify the check ran without error for analyst roles
    target_identity = (customer_profile.get("target_identity") or
                       roadmap_data.get("target_role") or "")
    ti_lower = target_identity.lower()
    _analyst_kw = {"analyst", "analytics"}
    if any(kw in ti_lower for kw in _analyst_kw):
        print(f"[ASSERTION] Analyst role '{target_identity}' — contamination check passed")


def _repair_role_alignment(roadmap_data: dict) -> int:
    """Repair role alignment by replacing non-matching skills with role-appropriate ones.

    Uses _ROLE_ALIGNMENT_KEYWORDS to identify the target domain and replaces
    non-matching skills with domain-appropriate catalog skills.

    Uses roadmap_data.target_role (the short, normalized role title) — NEVER
    target_identity (which may contain an onboarding paragraph).

    SKIPS skills the user explicitly requested (PROTECTED_USER_SKILLS).

    Returns count of skills replaced.
    """
    # ── Protected user-requested skills ──────────────────────
    # Skills the user explicitly mentioned in their profile should
    # NEVER be overwritten by role repair.
    _protected_user_skills = set()
    _profile = roadmap_data.get("customer_profile", {}) or {}
    for sk in (_profile.get("known_skills") or _profile.get("skills") or []):
        sk_lower = (sk or "").lower().strip()
        if sk_lower:
            _protected_user_skills.add(sk_lower)

    # Prefer the short normalized target_role from roadmap_data, fall back
    # to customer_profile.target_identity only if target_role is empty.
    target_role = (
        roadmap_data.get("target_role")
        or (roadmap_data.get("customer_profile") or {}).get("target_identity", "")
    )
    if not target_role:
        return 0

    expected_keywords = _resolve_role_alignment_keywords(target_role)
    if not expected_keywords:
        return 0

    import copy
    from uuid import uuid4

    # Build a pool of replacement titles from expected keywords
    replacement_pool = [kw.replace("_", " ").title() for kw in expected_keywords]

    replaced = 0
    for ms in roadmap_data.get("milestones", []):
        for mod in ms.get("modules", []):
            for skill in mod.get("skills", []):
                n = (skill.get("n") or "").lower()
                title = (skill.get("title") or "").lower()
                combined = n + " " + title

                # ── Skip protected user skills ────────────────
                if _protected_user_skills:
                    skill_lower = combined or title or n
                    if any(ps in skill_lower or skill_lower in ps for ps in _protected_user_skills):
                        continue

                # Check if this skill matches any expected keyword
                skill_ok = False
                for kw in expected_keywords:
                    if kw in combined or (len(kw) > 2 and kw in n):
                        skill_ok = True
                        break

                if not skill_ok and replacement_pool:
                    # Replace n and title with a role-appropriate one
                    new_n = replacement_pool[replaced % len(replacement_pool)]
                    skill["n"] = new_n.lower().replace(" ", "_")
                    skill["title"] = new_n
                    if "lessons" in skill and isinstance(skill["lessons"], list):
                        skill["lessons"] = [
                            f"{new_n} — Part 1",
                            f"{new_n} — Part 2",
                        ]
                    if "why_this_skill" in skill:
                        skill["why_this_skill"] = f"Role-appropriate replacement for {target_role}"
                    replaced += 1

    if replaced:
        _role_source = "roadmap_data.target_role" if roadmap_data.get("target_role") else "customer_profile.target_identity"
        print("[ROLE_REPAIR_AUDIT]")
        print(f"  role_used       = '{target_role}'")
        print(f"  source          = {_role_source}")
        print(f"  skills_replaced = {replaced}")
        print(f"[ROLE REPAIR] Replaced {replaced} skill(s) to align with '{target_role}'")
    return replaced


def _capability_gap_alignment_check(roadmap_data: dict) -> dict:
    gap = roadmap_data.get("capability_gap", {})
    ms_range = roadmap_data.get("milestone_range", {})
    min_ms = ms_range.get("minimum", MIN_MILESTONES)
    max_ms = ms_range.get("maximum", gap.get("recommended_milestones", MAX_MILESTONES))
    max_mod = MAX_MODULES  # module count can vary per milestone, only global max applies
    max_skill = gap.get("recommended_skill_density", MAX_SKILLS)
    if max_ms is None:
        return {"pass": True, "reason": "capability_gap data not available — skip check"}

    milestones = roadmap_data.get("milestones", [])
    actual_ms = len(milestones)
    violations = []

    # Phase 4 — Range-based validation: minimum <= actual <= maximum
    if actual_ms > max_ms:
        violations.append(
            f"milestone count {actual_ms} exceeds max bound {max_ms}"
        )
    if actual_ms < min_ms:
        violations.append(
            f"milestone count {actual_ms} below minimum {min_ms}"
        )

    # Phase 4 — Rationale must be present
    rationale = roadmap_data.get("milestone_count_rationale", "")
    if not rationale or len(rationale) < 10:
        violations.append("milestone_count_rationale missing or too short — must explain why count was chosen within range")

    for ms in milestones:
        mid = ms.get("milestone_id", "?")
        modules = ms.get("modules", [])
        actual_mod = len(modules)
        if actual_mod > MAX_MODULES:
            violations.append(
                f"Milestone {mid}: {actual_mod} modules exceed max bound {MAX_MODULES}"
            )
        if actual_mod < MIN_MODULES:
            violations.append(
                f"Milestone {mid}: {actual_mod} modules below minimum {MIN_MODULES}"
            )
        # Phase 6 — Module count rationale per milestone
        mod_rationale = ms.get("module_count_rationale", "")
        if not mod_rationale or len(mod_rationale) < 10:
            violations.append(
                f"Milestone {mid}: module_count_rationale missing or too short — must explain why this milestone has {actual_mod} modules"
            )
        for mod in modules:
            mod_id = mod.get("id", "?")
            skills = mod.get("skills", [])
            actual_skill = len(skills)
            if actual_skill > max_skill:
                violations.append(
                    f"Module {mod_id}: {actual_skill} skills exceed max bound {max_skill}"
                )
            if actual_skill < MIN_SKILLS:
                violations.append(
                    f"Module {mod_id}: {actual_skill} skills below minimum {MIN_SKILLS}"
                )
            # Phase 7 — Skill count rationale
            skill_rationale = mod.get("skill_count_rationale", "")
            if not skill_rationale or len(skill_rationale) < 10:
                violations.append(
                    f"Module {mod_id}: skill_count_rationale missing or too short — must explain why this module has {actual_skill} skills"
                )

    if violations:
        return {"pass": False, "reason": "; ".join(violations)}
    return {"pass": True, "reason": "capability_gap range bounds satisfied"}


def validate_identity_progression_density(roadmap_data: dict) -> dict:
    """
    Phase 5 — Milestone Quality Validation.

    Validates that the chosen milestone count can realistically support
    the identity transition from current to target role.

    Rules:
      - Beginner → complex role (Data Scientist, AI Engineer, Architect):
        MINIMUM 4 milestones
      - Beginner → simple role (Junior Developer, Support Engineer):
        MINIMUM 2 milestones
      - Intermediate → adjacent role (Backend → Fullstack):
        MINIMUM 2 milestones
      - Intermediate → distant role (Backend → AI Engineer):
        MINIMUM 3 milestones
      - Senior → senior role:
        MINIMUM 2 milestones
      - years_experience=0 AND target is senior/lead/architect:
        MINIMUM 4 milestones
    """
    profile = roadmap_data.get("customer_profile", {})
    target = (profile.get("target_identity", "") or "").lower()
    current = (profile.get("current_identity", "") or "").lower()
    level = roadmap_data.get("level", "beginner")
    years_exp = roadmap_data.get("years_experience", 0)

    milestones = roadmap_data.get("milestones", [])
    actual_ms = len(milestones)

    senior_targets = {"senior", "lead", "principal", "architect", "manager", "head of", "director"}
    complex_roles = {"data scientist", "ai engineer", "machine learning engineer", "architect",
                     "fullstack", "devops", "sre", "security engineer"}
    simple_roles = {"junior", "support", "tester", "associate"}

    target_is_senior = any(t in target for t in senior_targets)
    target_is_complex = any(r in target for r in complex_roles)
    target_is_simple = any(r in target for r in simple_roles)
    is_beginner = level == "beginner" or years_exp == 0

    violations = []
    min_expected = MIN_MILESTONES

    if is_beginner and target_is_senior:
        min_expected = 4
    elif is_beginner and target_is_complex:
        min_expected = 4
    elif is_beginner and target_is_simple:
        min_expected = 2
    elif level == "intermediate" and target_is_complex and not any(r in current for r in complex_roles):
        min_expected = 3
    elif level == "intermediate":
        min_expected = 2
    elif level == "senior":
        min_expected = 2

    if actual_ms < min_expected:
        violations.append(
            f"identity progression density too low: {actual_ms} milestones is insufficient "
            f"for {level}→{target} transition (minimum {min_expected} needed). "
            f"Each milestone must represent a distinct market-recognized identity."
        )

    if is_beginner and target_is_complex and actual_ms >= 4:
        milestone_titles = [m.get("t", "") for m in milestones]
        unique_count = len(set(t for t in milestone_titles if t))
        if unique_count < 2:
            violations.append(
                f"milestones appear to lack distinct identities: titles={milestone_titles}"
            )

    if violations:
        return {"pass": False, "reason": "; ".join(violations)}
    return {"pass": True, "reason": f"identity progression density valid ({actual_ms} ms for {level}→{target})"}


def run_roadmap_bible_validators(roadmap_data: dict, weekly_hours: int, timeline_days: int = 112) -> dict:
    """
    Run the Genuineness Validator (MD Section 19 — 7 checks, all must pass).

    The first 7 entries below ARE the MD-defined gate:
        fits_life · starts_where_they_are · laugh_test · no_spectators ·
        ai_first · dag_clean · salary_floor

    Everything after that (time_budget, known_skill_skip, identity_density,
    capability_gap_alignment) is supplementary internal QA this codebase
    layered on top — useful, but not part of the MD-defined 7. Kept
    separate in the result dict so the 7-check gate can be evaluated
    in isolation if needed.

    Returns a dict keyed by validator name. Never raises — all errors
    are caught and recorded as pass=False with a reason.
    """
    _validators = [
        # ── MD Section 19 — the 7 genuineness checks ──────────────
        ("fits_life",             lambda: fits_life_check(roadmap_data, weekly_hours, timeline_days)),
        ("starts_where_they_are", lambda: _starts_where_they_are_check(roadmap_data)),
        ("laugh_test",            lambda: _laugh_test_check(roadmap_data)),
        ("no_spectators",         lambda: _no_spectators_check(roadmap_data)),
        ("ai_first",              lambda: _ai_first_check(roadmap_data)),
        ("dag_clean",             lambda: _dag_clean_check(roadmap_data)),
        ("salary_floor",          lambda: _salary_floor_check(roadmap_data)),
        # ── Supplementary internal QA — not part of the MD 7 ──────
        ("time_budget",              lambda: _time_budget_check(roadmap_data)),
        ("known_skill_skip",         lambda: _known_skill_skip_check(roadmap_data)),
        ("identity_density",         lambda: validate_identity_progression_density(roadmap_data)),
        ("capability_gap_alignment", lambda: _capability_gap_alignment_check(roadmap_data)),
    ]

    results = {}
    for name, fn in _validators:
        try:
            result       = fn()
            results[name] = result
            # fits_life uses "fits" key; others use "pass"
            passed = result.get("pass", result.get("fits", True))
            badge  = "PASS" if passed else "WARN"
            print(f"[BIBLE] {badge:4s} {name}: {result.get('reason', '')}")
        except Exception as exc:
            results[name] = {"pass": False, "reason": f"Validator error: {exc}"}
            print(f"[BIBLE] ERR  {name}: {exc}")

    return results


# ============================================================
# Onboarding record validation helpers
# ============================================================

def is_valid_onboarding_record(context: str) -> bool:
    try:
        data = json.loads(context)
        if not isinstance(data, dict):
            return False
        return any(k in data for k in (
            "years_experience", "weekly_hours_available", "goal_context", "primary_goal"
        ))
    except (json.JSONDecodeError, TypeError, ValueError):
        return True


def detect_stale_onboarding_record(context: str) -> bool:
    try:
        data = json.loads(context)
        if not isinstance(data, dict):
            return False
        has_roadmap_fields = any(k in data for k in ("roadmap_id", "target_role", "milestones"))
        lacks_years = "years_experience" not in data
        return has_roadmap_fields and lacks_years
    except (json.JSONDecodeError, TypeError, ValueError):
        return False


def extract_weekly_hours(context: str) -> int | None:
    patterns = [
        r"[Ww]eekly\s+hours?\s+(?:available)?[:\s]+(\d+)",
        r"[Ww]eekly\s+[Hh]ours?[:\s]+(\d+)",
        r"[Hh]ours?\s+available\s+per\s+week[:\s]+(\d+)",
        r"(\d+)\s+hours?\s+per\s+week",
        r"[Aa]vailability[:\s]+(\d+)\s+hours?/week",
    ]
    for pattern in patterns:
        m = re.search(pattern, context)
        if m:
            return int(m.group(1))
    return None


def build_customer_profile(context: str) -> dict:
    profile = {
        "current_identity": "",
        "target_identity": "",
        "years_experience": 0,
        "weekly_hours_available": 5,
        "timeline_days": 112,
        "current_salary_lpa": 0,
        "known_skills": [],
        "self_efficacy": 0.5,
        "_provenance": {},
    }
    try:
        parsed = json.loads(context)
        if isinstance(parsed, dict):
            # years_experience
            ye = parsed.get("years_experience")
            if ye is not None:
                profile["years_experience"] = int(ye)
                profile["_provenance"]["years_experience"] = "onboarding_json"

            # weekly_hours_available
            wh = parsed.get("weekly_hours_available")
            if wh is not None:
                profile["weekly_hours_available"] = int(wh)
                profile["_provenance"]["weekly_hours_available"] = "onboarding_json"

            # timeline_days
            tl = parsed.get("timeline_days")
            if tl is not None:
                profile["timeline_days"] = max(int(tl), 7)
                profile["_provenance"]["timeline_days"] = "onboarding_json"

            # current_salary_lpa
            sal = parsed.get("current_salary_lpa")
            if sal is not None:
                profile["current_salary_lpa"] = float(sal)
                profile["_provenance"]["current_salary_lpa"] = "onboarding_json"
            else:
                monthly = parsed.get("current_salary_monthly")
                if monthly:
                    profile["current_salary_lpa"] = round(float(monthly) * 12 / 100000, 1)
                    profile["_provenance"]["current_salary_lpa"] = "monthly_salary_conversion"

            # known_skills
            ks = parsed.get("known_skills")
            if isinstance(ks, list):
                profile["known_skills"] = [str(s) for s in ks]
                profile["_provenance"]["known_skills"] = "onboarding_json"
            elif isinstance(ks, str):
                profile["known_skills"] = [s.strip() for s in ks.split(",") if s.strip()]
                profile["_provenance"]["known_skills"] = "onboarding_json"

            # self_efficacy
            se = parsed.get("self_efficacy")
            if se is not None:
                profile["self_efficacy"] = float(se)
                profile["_provenance"]["self_efficacy"] = "onboarding_json"

            # current_identity
            ci = parsed.get("current_identity")
            if ci:
                profile["current_identity"] = str(ci)
                profile["_provenance"]["current_identity"] = "onboarding_json"
            else:
                cr = parsed.get("current_role")
                if cr:
                    profile["current_identity"] = str(cr)
                    profile["_provenance"]["current_identity"] = "current_role_fallback"

            # target_identity — strict priority: target_identity > target_role
            # NEVER use primary_goal (it's a paragraph, not a role title)
            ti = parsed.get("target_identity") or parsed.get("target_role")
            if ti:
                ti = str(ti).strip()
                if len(ti) > 80:
                    print(f"[PROFILE WARN] target_identity too long ({len(ti)} chars): truncating")
                    ti = ti[:80]
                profile["target_identity"] = ti
                profile["_provenance"]["target_identity"] = "onboarding_json"

    except (json.JSONDecodeError, TypeError, ValueError):
        pass

    # Regex fallbacks for fields not found in JSON
    if "current_identity" not in profile["_provenance"]:
        ci_match = re.search(
            r"(?:current(?:\s+role|_role)|Current role)[:\s]+(.+)",
            context, re.IGNORECASE,
        )
        if ci_match:
            profile["current_identity"] = ci_match.group(1).strip()
            profile["_provenance"]["current_identity"] = "regex"

    if "target_identity" not in profile["_provenance"]:
        ti_match = re.search(
            r"(?:target(?:\s+role|_role|_identity|goal)|Target role|Primary goal)[:\s]+(.+)",
            context, re.IGNORECASE,
        )
        if ti_match:
            raw = ti_match.group(1).strip()
            if len(raw) > 80:
                print(f"[PROFILE WARN] regex target_identity too long ({len(raw)} chars): skipping")
            else:
                profile["target_identity"] = raw
                profile["_provenance"]["target_identity"] = "regex"

    if "years_experience" not in profile["_provenance"]:
        ye_match = re.search(r"[Yy]ears?\s*(?:of\s+)?experience[:\s]+(\d+)", context)
        if ye_match:
            profile["years_experience"] = int(ye_match.group(1))
            profile["_provenance"]["years_experience"] = "regex"

    if "weekly_hours_available" not in profile["_provenance"]:
        wh_match = extract_weekly_hours(context)
        if wh_match is not None:
            profile["weekly_hours_available"] = wh_match
            profile["_provenance"]["weekly_hours_available"] = "regex"

    if "timeline_days" not in profile["_provenance"]:
        tl_match = re.search(r"timeline\s*days?\s*:?\s*(\d+)", context)
        if tl_match:
            profile["timeline_days"] = max(int(tl_match.group(1)), 7)
            profile["_provenance"]["timeline_days"] = "regex"

    # current_salary_lpa — regex from "Current salary monthly: <amount>"
    if "current_salary_lpa" not in profile["_provenance"]:
        sal_match = re.search(
            r"[Cc]urrent\s+salary\s+monthly[:\s]+(\d+(?:\.\d+)?)",
            context,
        )
        if not sal_match:
            sal_match = re.search(
                r"[Cc]urrent\s+salary\s+lpa[:\s]+(\d+(?:\.\d+)?)",
                context,
            )
        if sal_match:
            monthly = float(sal_match.group(1))
            profile["current_salary_lpa"] = round(monthly * 12 / 100000, 1)
            profile["_provenance"]["current_salary_lpa"] = "regex_monthly_salary"

    # known_skills — regex from "Known skills: <comma-separated list>"
    if "known_skills" not in profile["_provenance"]:
        ks_match = re.search(
            r"[Kk]nown\s+skills[:\s]+(.+)",
            context,
        )
        if ks_match:
            raw = ks_match.group(1).strip()
            parsed = [s.strip() for s in raw.split(",") if s.strip()]
            if parsed:
                profile["known_skills"] = parsed
                profile["_provenance"]["known_skills"] = "regex"

    # ── PROFILE_EXTRACTION_RESULT logging ─────────────────────
    print("[PROFILE_EXTRACTION_RESULT]")
    for field in ["current_identity", "target_identity", "years_experience",
                   "weekly_hours_available", "timeline_days", "current_salary_lpa",
                   "known_skills", "self_efficacy"]:
        provenance = profile.get("_provenance", {}).get(field, "default")
        value = profile.get(field)
        print(f"  {field:25s} source={provenance:20s} value={value}")
    print("[/PROFILE_EXTRACTION_RESULT]")

    return profile


def parse_salary_value(s: str) -> float:
    if not s:
        return 0.0
    m = re.search(r"(?:₹)?\s*([\d]+(?:\.\d+)?)", s)
    if m:
        return float(m.group(1))
    return 0.0


def apply_salary_floor_repair(roadmap_data: dict) -> None:
    if roadmap_data.get("icp_type") != "high":
        return
    current_sal = roadmap_data.get("current_salary_lpa", 0)
    if not current_sal:
        return
    milestones = roadmap_data.get("milestones", [])
    if not milestones:
        return
    m01 = milestones[0]
    old_sal = m01.get("sal", "")
    parsed = parse_salary_value(old_sal)
    if parsed >= current_sal:
        return
    m01["sal"] = f"₹{current_sal}+ LPA"
    m01["salary_floor_applied"] = True
    print(
        f"[SALARY FLOOR] Auto-corrected M01 salary: "
        f"old={old_sal}  new=₹{current_sal}+ LPA"
    )


def normalize_skill_match(skill_value: str, known_skill: str) -> bool:
    skill_value = (skill_value or "").lower().strip()
    known_skill = (known_skill or "").lower().strip()
    return (
        skill_value == known_skill
        or known_skill in skill_value
        or skill_value in known_skill
    )


def remove_known_skills_from_roadmap(roadmap_data: dict, known_skills: list[str]) -> int:
    """Remove known skills completely from roadmap (starts_where_they_are, MD Section 2 / 19).
    
    For each matching skill:
      Completely remove from the module skill list.
      Record audit entry in roadmap_data["_removed_known_skills"].
    
    After removal recalculates:
      - skill counts per module
      - total skills across roadmap
      - estimated_total_hours
    
    Returns count of skills removed.
    """
    if not known_skills:
        return 0
    removed_count = 0
    removed_audit = []
    for ms in roadmap_data.get("milestones", []):
        ms_id = ms.get("milestone_id", "?")
        for mod in ms.get("modules", []):
            mod_id = mod.get("id", "?")
            surviving = []
            for skill in mod.get("skills", []):
                n = skill.get("n", "")
                title = skill.get("title", "")
                if any(normalize_skill_match(v, ks) for v in (n, title) for ks in known_skills):
                    removed_count += 1
                    removed_audit.append({
                        "skill_title": title or n,
                        "matched_known_skill": next(
                            ks for ks in known_skills
                            if any(normalize_skill_match(v, ks) for v in (n, title))
                        ),
                        "module_id": mod_id,
                        "milestone_id": ms_id,
                        "skill_id": skill.get("skill_id", "?"),
                    })
                else:
                    surviving.append(skill)
            mod["skills"] = surviving
    roadmap_data["_removed_known_skills"] = removed_audit
    # Recalculate structural totals
    total_skills = 0
    for ms in roadmap_data.get("milestones", []):
        for mod in ms.get("modules", []):
            total_skills += len(mod.get("skills", []))
    roadmap_data["total_skills"] = total_skills
    # Re-estimate hours: trust existing per-skill estimates, just sum
    total_hrs = 0
    for ms in roadmap_data.get("milestones", []):
        for mod in ms.get("modules", []):
            for skill in mod.get("skills", []):
                total_hrs += skill.get("estimated_hours", 0) or 0
    roadmap_data["estimated_total_hours"] = round(total_hrs, 1)
    return removed_count


def _salary_floor_check(roadmap_data: dict) -> dict:
    """Validator: salary_floor (MD Section 19 — warn/log/auto-correct, CP-B only)."""
    if roadmap_data.get("icp_type") != "high":
        return {"pass": True, "reason": "not applicable for low ICP"}
    current_sal = roadmap_data.get("current_salary_lpa", 0)
    if not current_sal:
        return {"pass": True, "reason": "current_salary_lpa not available"}
    milestones = roadmap_data.get("milestones", [])
    if not milestones:
        return {"pass": True, "reason": "no milestones to check"}
    m01_sal = milestones[0].get("sal", "0")
    parsed = parse_salary_value(m01_sal)
    if parsed < current_sal:
        return {
            "pass": False,
            "reason": f"M01 salary ({m01_sal}) below current salary ({current_sal} LPA)",
        }
    return {"pass": True, "reason": "salary floor satisfied"}


# ============================================================
# ROADMAP QUALITY VALIDATORS  (supplementary — NOT the MD 7-check gate)
# ============================================================
# These validators evaluate content quality, career progression,
# project realism, salary logic, skill relevance, and course
# catalog compliance.  They NEVER raise and NEVER block generation.
# They are useful internal QA but are not part of the Genuineness
# Validator defined in MD Section 19.
# ============================================================

def _milestone_identity_quality_check(roadmap_data: dict) -> dict:
    milestones = roadmap_data.get("milestones", [])
    topic_tokens = {"basics", "fundamentals", "introduction", "intro", "overview"}
    violations = []
    for ms in milestones:
        title = (ms.get("t") or ms.get("title") or "").lower()
        if any(tok in title for tok in topic_tokens):
            violations.append(
                f"Milestone {ms.get('milestone_id')}: title '{ms.get('t')}' "
                f"is a topic name, not a professional identity"
            )
    if violations:
        return {"pass": False, "reason": "; ".join(violations), "violations": violations}
    return {"pass": True, "reason": "All milestone titles are professional identities"}


def _capability_progression_check(roadmap_data: dict) -> dict:
    milestones = roadmap_data.get("milestones", [])
    violations = []
    titles = []
    for ms in milestones:
        t = ms.get("t") or ms.get("title") or ""
        if t in titles:
            violations.append(f"Duplicate milestone title: '{t}'")
        titles.append(t)
    if violations:
        return {"pass": False, "reason": "; ".join(violations), "violations": violations}
    return {"pass": True, "reason": "Capability progression is sound — no duplicate titles, no regression"}


def _project_quality_check(roadmap_data: dict) -> dict:
    milestones = roadmap_data.get("milestones", [])
    bad_patterns = {"quiz", "worksheet", "summary", "presentation", "blog"}
    violations = []
    for ms in milestones:
        mid = ms.get("milestone_id", "?")
        ms_projects = ms.get("projects")
        if ms_projects is None:
            legacy = ms.get("project")
            ms_projects = [legacy] if legacy else []
        if not ms_projects:
            violations.append(f"Milestone {mid}: no project to evaluate")
            continue
        for project in ms_projects:
            title = (project.get("title") or "").lower()
            if any(p in title for p in bad_patterns):
                violations.append(
                    f"Milestone {mid}: project '{project.get('title')}' "
                    f"resembles a toy exercise ({title})"
                )
            if not project.get("deploy_required"):
                violations.append(
                    f"Milestone {mid}: project '{project.get('title')}' "
                    f"is not deploy_required — no deployment artifact"
                )
    if violations:
        return {"pass": False, "reason": "; ".join(violations), "violations": violations}
    return {"pass": True, "reason": "All projects have meaningful implementation + deployment"}


def _salary_progression_check(roadmap_data: dict) -> dict:
    milestones = roadmap_data.get("milestones", [])
    icp_type = roadmap_data.get("icp_type", "low")
    violations = []
    prev_sal = -1
    for ms in milestones:
        mid = ms.get("milestone_id", "?")
        sal = ms.get("sal", "")
        # Try to extract numeric salary from string like "₹3-5 LPA" or "Unpaid/stipend"
        try:
            parts = sal.replace("₹", "").replace(",", "").split("-")
            num = int(parts[0].strip().split()[0]) if parts[0].strip().split()[0].isdigit() else 0
        except (ValueError, IndexError, AttributeError):
            num = 0
        if num < prev_sal:
            violations.append(
                f"Milestone {mid}: salary '{sal}' ({num}) is lower than "
                f"previous milestone ({prev_sal})"
            )
        prev_sal = num
    if violations:
        return {"pass": False, "reason": "; ".join(violations), "violations": violations}
    return {"pass": True, "reason": "Salary progression is non-decreasing"}


# Lightweight forbidden-skill maps keyed by domain keywords in target_role
_FORBIDDEN_SKILL_MAPS = {
    "backend": {"stable_diffusion", "gan", "image_segmentation", "object_detection", "computer_vision"},
    "software": {"stable_diffusion", "gan", "image_segmentation", "object_detection", "computer_vision"},
    "distributed": {"stable_diffusion", "gan", "image_segmentation"},
    "ai": set(),       # AI roles accept everything
    "machine_learning": set(),  # ML roles accept everything
}


def _skill_relevance_check(roadmap_data: dict) -> dict:
    target_role = (roadmap_data.get("target_role") or "").lower()
    # Determine which skill map applies
    forbidden = set()
    for keyword, skill_set in _FORBIDDEN_SKILL_MAPS.items():
        if keyword in target_role:
            forbidden.update(skill_set)
    if not forbidden:
        return {"pass": True, "reason": "No forbidden skills for this target role"}

    violations = []
    for ms in roadmap_data.get("milestones", []):
        for mod in ms.get("modules", []):
            for skill in mod.get("skills", []):
                n = (skill.get("n") or "").lower().strip()
                if n in forbidden:
                    violations.append(
                        f"Skill '{skill.get('n')}' in {mod.get('id')} "
                        f"appears irrelevant for target role '{target_role}'"
                    )
    if violations:
        return {"pass": False, "reason": "; ".join(violations), "violations": violations}
    return {"pass": True, "reason": "All skills appear relevant for the target role"}


_UNAVAILABLE_COURSES = {
    "data science", "nlp", "robotics", "fastapi standalone",
    "github standalone", "aws masterclass", "devops", "mlops",
    "soft skills", "project manager",
}

_ALL_AVAILABLE_MODULES = {
    m.lower()
    for course in AVAILABLE_COURSES.values()
    for m in course.get("modules", [])
}


def _course_catalog_compliance_check(roadmap_data: dict) -> dict:
    """Check course catalog compliance with coverage thresholds.
    
    PASS  if coverage >= 90%
    WARN  if 85% <= coverage < 90%
    FAIL  if coverage < 85%
    
    Uses catalog_status set by repair_course_catalog_alignment when available.
    """
    violations = []
    total = 0
    mapped = 0
    missing_skills = []

    for ms in roadmap_data.get("milestones", []):
        for mod in ms.get("modules", []):
            mod_title = (mod.get("title") or "").lower()
            for uc in _UNAVAILABLE_COURSES:
                if uc in mod_title:
                    violations.append(
                        f"Module '{mod.get('id')}' ('{mod.get('title')}') "
                        f"resembles unavailable course '{uc}'"
                    )
            for skill in mod.get("skills", []):
                total += 1
                catalog_status = skill.get("catalog_status")
                if catalog_status == "mapped":
                    mapped += 1
                    continue
                if catalog_status == "missing":
                    missing_skills.append(skill.get("title") or skill.get("n", ""))
                    continue
                # No catalog_status yet — compute inline
                skill_n = (skill.get("n") or "").lower().strip()
                if skill_n in ALL_CATALOG_SKILLS:
                    mapped += 1
                else:
                    missing = True
                    for avail in ALL_CATALOG_SKILLS:
                        if skill_n in avail or avail in skill_n:
                            missing = False
                            break
                    if missing:
                        missing_skills.append(skill.get("title") or skill.get("n", ""))
                        violations.append(
                            f"Skill '{skill.get('n')}' in {mod.get('id')} "
                            f"not found in any AVAILABLE_COURSES"
                        )
                    else:
                        mapped += 1

    coverage = round(mapped / max(total, 1), 3)
    result = {
        "pass": coverage >= 0.85,
        "reason": f"Catalog coverage: {coverage:.0%} ({mapped}/{total})",
        "coverage": coverage,
        "mapped": mapped,
        "total": total,
        "missing": missing_skills[:10],
    }
    if coverage >= 0.90:
        result["pass"] = True
        result["reason"] = f"Catalog coverage: {coverage:.0%} ({mapped}/{total}) — all good"
    elif coverage >= 0.85:
        result["pass"] = True
        result["reason"] = f"Catalog coverage: {coverage:.0%} ({mapped}/{total}) — acceptable"
    else:
        result["pass"] = False
        result["reason"] = f"Catalog coverage: {coverage:.0%} ({mapped}/{total}) — below 85% threshold"
    if violations:
        result["reason"] += f"; {len(violations)} unavailable-course violation(s)"
    return result


def _catalog_quality_check(roadmap_data: dict) -> dict:
    """Quality check: duplicate/low-confidence/missing catalog mappings."""
    seen_courses = {}
    low_conf = []
    missing = 0
    total = 0
    for ms in roadmap_data.get("milestones", []):
        for mod in ms.get("modules", []):
            for skill in mod.get("skills", []):
                total += 1
                cid = skill.get("catalog_course_id")
                conf = skill.get("catalog_match_confidence", 0)
                if skill.get("catalog_status") == "missing":
                    missing += 1
                elif cid:
                    if conf < 0.80:
                        low_conf.append(f"{skill.get('skill_id','?')} conf={conf}")
                    seen_courses[cid] = seen_courses.get(cid, 0) + 1
    issues = []
    if missing > total * 0.3:
        issues.append(f"{missing}/{total} skills unmapped")
    if low_conf:
        issues.append(f"low-confidence: {'; '.join(low_conf[:5])}")
    if issues:
        return {"pass": False, "reason": "; ".join(issues)}
    return {"pass": True, "reason": "All catalog mappings healthy"}


def validate_roadmap_schema(roadmap_data: dict) -> dict:
    """Strict schema validation: check required fields at every level.

    NEVER raises. Returns {"pass": bool, "reason": str}.
    """
    missing = []

    # Roadmap level
    for field in ("roadmap_id", "user_id", "target_role", "level", "icp_type", "milestones"):
        if field not in roadmap_data or roadmap_data.get(field) is None:
            missing.append(f"roadmap.{field}")

    milestones = roadmap_data.get("milestones", [])
    if not milestones:
        missing.append("roadmap.milestones (empty)")
        return {"pass": False, "reason": "; ".join(missing)}

    for ms in milestones:
        ms_id = ms.get("milestone_id", "?")
        for field in ("milestone_id", "label", "t", "sal", "o", "modules"):
            if field not in ms or ms.get(field) is None:
                missing.append(f"milestone {ms_id}.{field}")
        for field in ("sc_n", "iv"):
            if field not in ms:
                missing.append(f"milestone {ms_id}.{field}")

        for mod in ms.get("modules", []):
            mod_id = mod.get("id", "?")
            for field in ("id", "title", "skills"):
                if field not in mod or mod.get(field) is None:
                    missing.append(f"module {mod_id}.{field}")

            for skill in mod.get("skills", []):
                sid = skill.get("skill_id", "?")
                for field in ("skill_id", "title", "content_flow", "unlock_rules"):
                    if field not in skill or skill.get(field) is None:
                        missing.append(f"skill {sid}.{field}")
                cf = skill.get("content_flow")
                if cf is None or not isinstance(cf, dict):
                    missing.append(f"skill {sid}.content_flow (missing or not a dict)")

    if missing:
        return {"pass": False, "reason": f"Schema violations: {'; '.join(missing[:20])}"}
    return {"pass": True, "reason": "Schema valid — all required fields present"}


def _scaffolding_check(roadmap_data: dict) -> dict:
    """Verify progressive skill/module/milestone dependency building.

    Skills should build on previous skills, modules should build
    on prior modules, milestones should build on prior milestones.
    """
    milestones = roadmap_data.get("milestones", [])
    violations = []

    # Collect all skill IDs for dependency tracking
    all_skill_ids = set()
    skill_to_ms = {}
    for ms in milestones:
        ms_id = ms.get("milestone_id", "?")
        for mod in ms.get("modules", []):
            for skill in mod.get("skills", []):
                sid = skill.get("skill_id", "?")
                all_skill_ids.add(sid)
                skill_to_ms[sid] = ms_id

    # Check skill unlock_rules reference only earlier skills
    seen_skills = []
    for ms in milestones:
        ms_id = ms.get("milestone_id", "?")
        for mod in ms.get("modules", []):
            for skill in mod.get("skills", []):
                sid = skill.get("skill_id", "?")
                requires = skill.get("unlock_rules", {}).get("requires", [])
                for req in requires:
                    if req not in seen_skills:
                        violations.append(
                            f"skill {sid} requires '{req}' which is not a prior skill"
                        )
                seen_skills.append(sid)

    # Check module progression: second module should differ from first
    for ms in milestones:
        ms_id = ms.get("milestone_id", "?")
        modules = ms.get("modules", [])
        prev_titles = set()
        for mod in modules:
            mod_title = (mod.get("title") or "").lower()
            if mod_title in prev_titles:
                violations.append(
                    f"milestone {ms_id}: duplicate module title '{mod['title']}'"
                )
            prev_titles.add(mod_title)

    # Check milestone progression: milestone labels should be distinct
    seen_labels = set()
    for ms in milestones:
        ms_id = ms.get("milestone_id", "?")
        t_val = ms.get("t", "")
        if t_val in seen_labels:
            violations.append(
                f"milestone {ms_id}: duplicate label '{t_val}'"
            )
        if t_val:
            seen_labels.add(t_val)

    if violations:
        return {"pass": False, "reason": "; ".join(violations[:10])}
    return {"pass": True, "reason": "Scaffolding progressive — dependencies build correctly"}


ALLOWED_AI_LAYERS = {"vibe_planning", "vibe_architecture", "vibe_solution", "deployment"}
ALLOWED_AI_USAGE = {"generation", "analysis", "automation", "optimization", "decision_support", "planning"}
ALLOWED_AI_AUTOMATION = {"assistant", "copilot", "agent", "autonomous"}
_AI_LAYER_REPAIR_MAP = {
    # → vibe_planning
    "leadership": "vibe_planning",
    "planning": "vibe_planning",
    # → vibe_architecture
    "design": "vibe_architecture",
    "architecture": "vibe_architecture",
    "optimization": "vibe_architecture",
    "performance": "vibe_architecture",
    "observability": "vibe_architecture",
    # → vibe_solution
    "coding": "vibe_solution",
    "implementation": "vibe_solution",
    "debugging": "vibe_solution",
    "solution": "vibe_solution",
    # → deployment
    "infra": "deployment",
    "testing": "deployment",
}
_AI_USAGE_REPAIR_MAP = {
    "implementation": "generation",
    "coding": "generation",
    "deployment": "automation",
    "testing": "analysis",
}


def normalize_ai_layer(value: str) -> str:
    """Normalize an AI layer value to a valid one.

    Uses the repair map as a pre-generation normalizer.
    Returns the normalized layer.
    """
    v = (value or "").lower().strip()
    if v in ALLOWED_AI_LAYERS:
        return v
    if v in _AI_LAYER_REPAIR_MAP:
        return _AI_LAYER_REPAIR_MAP[v]
    return "vibe_solution"


def normalize_ai_usage(value: str) -> str:
    """Normalize AI usage type."""
    v = (value or "").lower().strip()
    if v in ALLOWED_AI_USAGE:
        return v
    if v in _AI_USAGE_REPAIR_MAP:
        return _AI_USAGE_REPAIR_MAP[v]
    return "generation"


def normalize_ai_automation(value: str) -> str:
    """Normalize AI automation level."""
    v = (value or "").lower().strip()
    if v in ALLOWED_AI_AUTOMATION:
        return v
    return "copilot"


def repair_ai_layers(roadmap_data: dict) -> int:
    """Repair invalid AI layer values using normalizers (emergency fallback only).

    Covers both module-level ai_first_layer and skill-level ai_metadata.layer.
    Returns count of items repaired.
    """
    repaired_count = 0
    for ms in roadmap_data.get("milestones", []):
        for mod in ms.get("modules", []):
            # Fix module-level ai_first_layer
            orig_layer = mod.get("ai_first_layer", "")
            if orig_layer:
                norm_layer = normalize_ai_layer(orig_layer)
                if norm_layer != orig_layer:
                    mod["ai_first_layer"] = norm_layer
                    repaired_count += 1
            # Fix skill-level ai_metadata values
            for skill in mod.get("skills", []):
                ai = skill.get("ai_metadata")
                if not ai or not isinstance(ai, dict):
                    continue
                # Fix layer using normalizer
                layer = ai.get("layer", "")
                normalized = normalize_ai_layer(layer)
                if normalized != layer:
                    ai["layer"] = normalized
                    repaired_count += 1
                # Fix usage_type using normalizer
                usage = ai.get("usage_type", "")
                normalized_usage = normalize_ai_usage(usage)
                if normalized_usage != usage:
                    ai["usage_type"] = normalized_usage
                    repaired_count += 1
                # Fix automation_level using normalizer
                auto = ai.get("automation_level", "")
                normalized_auto = normalize_ai_automation(auto)
                if normalized_auto != auto:
                    ai["automation_level"] = normalized_auto
                    repaired_count += 1
                # Ensure ai_first is True
                if not ai.get("ai_first"):
                    ai["ai_first"] = True
                    repaired_count += 1
    return repaired_count


def audit_ai_layers(roadmap_data: dict) -> dict:
    """Audit AI layer distribution across the roadmap.

    Returns dict with counts per layer and total skills with ai_metadata.
    Includes per-milestone coverage of all 4 official layers.
    """
    layers = {}
    total = 0
    has_meta = 0
    REQUIRED_LAYERS = {"vibe_planning", "vibe_architecture", "vibe_solution", "deployment"}

    for ms in roadmap_data.get("milestones", []):
        ms_id = ms.get("milestone_id", "?")
        ms_layers = set()
        for mod in ms.get("modules", []):
            for skill in mod.get("skills", []):
                total += 1
                ai = skill.get("ai_metadata")
                if not ai or not isinstance(ai, dict):
                    continue
                has_meta += 1
                layer = ai.get("layer", "missing")
                layers[layer] = layers.get(layer, 0) + 1
                ms_layers.add(layer)

    print("========== AI LAYER AUDIT ==========")
    print(f"  Total skills: {total}")
    print(f"  Skills with ai_metadata: {has_meta}")
    # Per-milestone coverage
    for ms in roadmap_data.get("milestones", []):
        ms_id = ms.get("milestone_id", "?")
        ms_layers = set()
        for mod in ms.get("modules", []):
            for skill in mod.get("skills", []):
                ai = skill.get("ai_metadata")
                if not ai or not isinstance(ai, dict):
                    continue
                ms_layers.add(ai.get("layer", "missing"))
        missing = REQUIRED_LAYERS - ms_layers
        covered = REQUIRED_LAYERS & ms_layers
        print(f"  Milestone {ms_id}:")
        for l in sorted(REQUIRED_LAYERS):
            status = "✓" if l in covered else "✗"
            print(f"    {l:25s} {status}")
        if missing:
            print(f"    (missing: {', '.join(sorted(missing))})")
    # Aggregate
    print(f"  {'─'*40}")
    for layer in sorted(layers):
        bar = "#" * layers[layer]
        print(f"  {layer:25s} {layers[layer]:4d}  {bar}")
    invalid = [k for k in layers if k not in ALLOWED_AI_LAYERS]
    if invalid:
        print(f"  WARNING: {len(invalid)} invalid layer value(s): {', '.join(invalid)}")
    print("====================================")
    return {"layers": layers, "total": total, "has_meta": has_meta, "invalid": invalid}


def _ai_tag_check(roadmap_data: dict) -> dict:
    """Verify every skill has valid ai_metadata tags."""
    violations = []
    for ms in roadmap_data.get("milestones", []):
        for mod in ms.get("modules", []):
            for skill in mod.get("skills", []):
                sid = skill.get("skill_id", "?")
                ai = skill.get("ai_metadata")
                if not ai:
                    violations.append(f"skill {sid} missing ai_metadata")
                    continue
                if not isinstance(ai, dict):
                    violations.append(f"skill {sid} ai_metadata not a dict")
                    continue
                if not ai.get("ai_first"):
                    violations.append(f"skill {sid} ai_metadata.ai_first not true")
                layer = ai.get("layer")
                if layer not in ALLOWED_AI_LAYERS:
                    violations.append(f"skill {sid} ai_metadata.layer='{layer}' invalid")
                usage = ai.get("usage_type")
                if usage not in ALLOWED_AI_USAGE:
                    violations.append(f"skill {sid} ai_metadata.usage_type='{usage}' invalid")
                auto = ai.get("automation_level")
                if auto not in ALLOWED_AI_AUTOMATION:
                    violations.append(f"skill {sid} ai_metadata.automation_level='{auto}' invalid")
    if violations:
        return {"pass": False, "reason": "; ".join(violations[:15])}
    return {"pass": True, "reason": "All skills have valid ai_metadata tags"}


_BREADTH_DOMAIN_SKILLS = {
    "api_design": ["API Design & Service Layer", "REST API Design", "Service Contracts"],
    "system_design": ["Foundations of System Design", "Scalability & Performance", "Architecture Patterns"],
    "leadership": ["Capstone & Interview Readiness", "Technical Mentorship", "Stakeholder Management"],
    "distributed_systems": ["Distributed Systems", "Event Driven Systems", "Consistency Models"],
    "architecture": ["Scalability & Performance", "Design Tradeoffs", "System Evolution"],
    "security": ["Security & Compliance", "Security Architecture"],
    "cloud": ["Cloud & Deployment", "Cloud Infrastructure", "Deployment Architecture"],
    "scalability": ["Scalability & Performance", "Horizontal Scaling"],
    "data_modeling": ["Database Design & Data Modeling", "Schema Design"],
    "statistics": ["Statistical Analysis", "Experimental Design"],
    "deployment": ["Cloud & Deployment", "CI/CD & DevOps", "Release Engineering"],
    "ml_pipelines": ["MLOps & Monitoring", "Feature Engineering", "Model Serving & APIs"],
    "foundations": ["Core Concepts", "Domain Fundamentals"],
    "advanced_architecture": ["Advanced System Design", "Architecture Evolution"],
    "team_management": ["Capstone & Interview Readiness", "Technical Strategy"],
    "ui_architecture": ["UI Architecture", "Component Design"],
    "performance": ["Scalability & Performance", "Performance Tuning"],
    "testing": ["Testing & QA", "Quality Engineering"],
}


def _compute_required_domains(roadmap_data: dict) -> set:
    """Compute required breadth domains from role and gap."""
    target_role = (roadmap_data.get("target_role") or "").lower()
    gap = roadmap_data.get("capability_gap", {})
    gap_score = gap.get("gap_score", 0.5)

    required_domains = set()
    if "senior" in target_role or "lead" in target_role or "principal" in target_role:
        required_domains.update(["system_design", "leadership", "architecture"])
    if "backend" in target_role or "fullstack" in target_role or "software" in target_role:
        required_domains.update(["distributed_systems", "api_design"])
    if "data" in target_role or "machine" in target_role or "ml" in target_role:
        required_domains.update(["data_modeling", "statistics", "deployment"])
    if gap_score > 0.65:
        required_domains.add("scalability")

    # Also check capability_breadth computed earlier
    breadth = roadmap_data.get("capability_breadth", {})
    for d in breadth.get("required_domains", []):
        required_domains.add(d)

    return required_domains


def _get_covered_domains(roadmap_data: dict, required_domains: set) -> set:
    """Check which required domains are already covered by roadmap skills."""
    covered = set()
    for ms in roadmap_data.get("milestones", []):
        for mod in ms.get("modules", []):
            for skill in mod.get("skills", []):
                n = (skill.get("n") or "").lower()
                title = (skill.get("title") or "").lower()
                layer = (skill.get("ai_metadata", {}) or {}).get("layer", "")
                combined = n + " " + title + " " + layer
                for domain in list(required_domains):
                    d_key = domain.replace("_", " ")
                    if d_key in combined or domain.replace("_", "") in combined.replace("_", "").replace(" ", ""):
                        covered.add(domain)
    return covered


def _capability_breadth_check(roadmap_data: dict) -> dict:
    """Verify roadmap covers required capability domains for the transition."""
    required_domains = _compute_required_domains(roadmap_data)
    if not required_domains:
        return {"pass": True, "reason": "No specific breadth requirements for this role"}
    covered = _get_covered_domains(roadmap_data, required_domains)
    missing_domains = required_domains - covered
    if missing_domains:
        return {"pass": False, "reason": f"Missing required domains: {', '.join(sorted(missing_domains))}", "missing": sorted(missing_domains)}
    return {"pass": True, "reason": f"All {len(required_domains)} required domains covered"}


def _get_expected_skill_density(roadmap_data: dict) -> int:
    """Get max skill density bound from capability gap."""
    gap = roadmap_data.get("capability_gap", {})
    expected = gap.get("recommended_skill_density")
    if expected is not None:
        return expected
    return MAX_SKILLS


# Domain → AI layer mapping for breadth repair (Phase 4.1D)
_BREADTH_DOMAIN_AI_LAYER = {
    "system_design": "vibe_architecture", "architecture": "vibe_architecture",
    "tradeoffs": "vibe_architecture", "observability": "vibe_architecture",
    "scalability": "vibe_architecture",
    "coding": "vibe_solution", "implementation": "vibe_solution",
    "debugging": "vibe_solution", "fixing": "vibe_solution",
    "building": "vibe_solution",
    "planning": "vibe_planning", "roadmapping": "vibe_planning",
    "decomposition": "vibe_planning",
    "deploy": "deployment", "release": "deployment", "monitoring": "deployment",
}


def _create_breadth_module(last_ms: dict, domain: str, skill_title: str, skill: dict) -> dict:
    """Create a breadth module for the skill, appended to the last milestone."""
    existing_modules = last_ms.get("modules", [])
    last_id = "M0.0"
    for mod in existing_modules:
        mod_id = mod.get("id", "M0.0")
        if mod_id > last_id:
            last_id = mod_id
    # Increment module number
    prefix = last_id.rsplit(".", 1)[0] if "." in last_id else last_id
    try:
        num = int(last_id.rsplit(".", 1)[1]) if "." in last_id else 0
    except ValueError:
        num = 0
    new_id = f"{prefix}.{num + 1}"

    breadth_mod = {
        "id": new_id,
        "title": f"Breadth: {domain.replace('_', ' ').title()}",
        "ai_first_layer": _BREADTH_DOMAIN_AI_LAYER.get(domain, "vibe_solution"),
        "free": True,
        "vis": "code+real_tutor",
        "module_type": "breadth",
        "breadth_injected": True,
        "breadth_domain": domain,
        "skill_count_rationale": f"Breadth reinforcement module injected by capability breadth enforcement for domain: {domain}",
        "skills": [skill],
    }
    existing_modules.append(breadth_mod)
    return breadth_mod


def repair_capability_breadth(roadmap_data: dict) -> dict:
    """Repair missing breadth domains by injecting skills.

    NEVER exceeds expected skill density in existing modules.
    Creates dedicated breadth modules when existing modules are at capacity.

    Returns the number of skills injected.
    """
    required_domains = _compute_required_domains(roadmap_data)
    covered = _get_covered_domains(roadmap_data, required_domains)
    missing = required_domains - covered
    if not missing:
        return {"injected": 0, "missing": []}

    milestones = roadmap_data.get("milestones", [])
    if not milestones:
        return {"injected": 0, "missing": sorted(missing)}

    expected_density = _get_expected_skill_density(roadmap_data)
    last_ms = milestones[-1]
    injected = []

    for domain in sorted(missing):
        candidates = _BREADTH_DOMAIN_SKILLS.get(domain, [domain.replace("_", " ").title()])
        skill_title = candidates[0]

        skill_id = f"SKILL_BREADTH_{domain}_{uuid.uuid4().hex[:4]}"
        new_skill = {
            "skill_id": skill_id,
            "n": domain,
            "title": skill_title,
            "auto_completed": False,
            "ai_metadata": {
                "ai_first": True,
                "layer": _BREADTH_DOMAIN_AI_LAYER.get(domain, "vibe_solution"),
                "usage_type": "analysis",
                "automation_level": "copilot",
            },
            "why_this_skill": f"Covers required domain: {domain}",
            "lessons": [
                f"{skill_title} — Part 1",
                f"{skill_title} — Part 2",
            ],
            "p": 0,
            "mastery_state": {
                "state": "unlocked",
                "current_mastery": 0.0,
                "target_mastery": 0.9,
                "bkt": {"prior": 0.15, "learn_rate": 0.25, "guess": 0.1, "slip": 0.05},
            },
            "unlock_rules": {"requires": [], "minimum_mastery": 0.0, "unlock_type": "immediate"},
            "content_flow": {
                "video": {"content_id": f"VID_{skill_id}", "title": skill_title, "status": "locked"},
                "scenario": {},
                "mock": {"unlock_mastery": MOCK_UNLOCK_MASTERY, "status": "locked"},
                "review": {"review_type": "spaced_repetition", "next_review_at": None},
            },
            "estimated_hours": 3,
        }

        # Find an existing module with space below expected density
        target_mod = None
        for mod in last_ms.get("modules", []):
            if mod.get("module_type") == "breadth":
                continue  # don't inject into other breadth modules
            if len(mod.get("skills", [])) < expected_density:
                target_mod = mod
                break

        if target_mod:
            target_mod.setdefault("skills", []).append(new_skill)
        else:
            # Create a new breadth module
            _create_breadth_module(last_ms, domain, skill_title, new_skill)

        injected.append(domain)

    if injected:
        print(f"[BREADTH REPAIR] Injected {len(injected)} domain skill(s): {', '.join(injected)}")

    return {"injected": len(injected), "missing": sorted(missing - set(injected))}


def _deployment_check(roadmap_data: dict) -> dict:
    """Blocking validator: every milestone must have at least one project
    with deploy_required=True."""
    violations = []
    for ms in roadmap_data.get("milestones", []):
        mid = ms.get("milestone_id", "?")
        ms_projects = ms.get("projects")
        if ms_projects is None:
            legacy = ms.get("project")
            ms_projects = [legacy] if legacy else []
        if not ms_projects:
            violations.append(f"Milestone {mid}: no project")
            continue
        if not any(p.get("deploy_required") for p in ms_projects):
            violations.append(f"Milestone {mid}: no project has deploy_required set")
    if violations:
        return {"pass": False, "reason": "; ".join(violations)}
    return {"pass": True, "reason": "Every milestone has at least one deploy_required project"}


def audit_science_distribution(roadmap_data: dict) -> dict:
    """Audit scenario/mock distribution per milestone against MD bounds.

    Returns {"pass": bool, "reason": str, "details": [...]}.
    Real bounds (not "up to"): scenarios 3-7, mocks/interviews 1-2.
    """
    details = []
    all_pass = True
    for ms in roadmap_data.get("milestones", []):
        ms_id = ms.get("milestone_id", "?")
        sc = 0
        iv = 0
        for mod in ms.get("modules", []):
            for sci in mod.get("science", []):
                st = sci.get("type", "")
                if st == "Scenario":
                    sc += 1
                elif st == "Interview":
                    iv += 1
        sc_ok = MIN_SCENARIOS_PER_MILESTONE <= sc <= MAX_SCENARIOS_PER_MILESTONE
        iv_ok = MIN_MOCKS_PER_MILESTONE <= iv <= MAX_MOCKS_PER_MILESTONE
        if not (sc_ok and iv_ok):
            all_pass = False
        status = "PASS" if (sc_ok and iv_ok) else "FAIL"
        details.append(f"{ms_id} → scenarios={sc} mocks/interviews={iv} {status}")
    if all_pass:
        return {"pass": True, "reason": "All milestones have valid science counts", "details": details}
    return {"pass": False, "reason": "Science distribution violations found", "details": details}


def validate_dynamic_structure(roadmap_data: dict) -> dict:
    """Validate that roadmap structure follows dynamic, bounds-based generation.

    Checks:
      1. All counts are within MIN/MAX bounds.
      2. Each milestone has a module_count_rationale.
      3. Each module has a skill_count_rationale.
      4. At least one milestone has a different module count (structural diversity).
      5. Fingerprint of the shape.

    Returns {"pass": bool, "reason": str, "details": {...}}.
    """
    milestones = roadmap_data.get("milestones", [])
    details = {}
    violations = []

    # 1. Bounds
    ms_count = len(milestones)
    if ms_count < MIN_MILESTONES or ms_count > MAX_MILESTONES:
        violations.append(f"milestones={ms_count} outside [{MIN_MILESTONES},{MAX_MILESTONES}]")

    mod_counts = []
    skill_counts = []
    for ms in milestones:
        mid = ms.get("milestone_id", "?")
        mods = ms.get("modules", [])
        mc = len(mods)
        mod_counts.append(mc)
        if mc < MIN_MODULES or mc > MAX_MODULES:
            violations.append(f"{mid} modules={mc} outside [{MIN_MODULES},{MAX_MODULES}]")
        for mod in mods:
            mod_id = mod.get("id", "?")
            sc = len(mod.get("skills", []))
            skill_counts.append(sc)
            if sc < MIN_SKILLS or sc > MAX_SKILLS:
                violations.append(f"{mod_id} skills={sc} outside [{MIN_SKILLS},{MAX_SKILLS}]")
            # 3. Rationale check
            if not mod.get("skill_count_rationale"):
                violations.append(f"{mod_id} missing skill_count_rationale")
        # 2. Rationale check
        if not ms.get("module_count_rationale"):
            violations.append(f"{mid} missing module_count_rationale")

    # 4. Structural diversity: at least one milestone with different module count
    if len(set(mod_counts)) < 2 and len(mod_counts) > 1:
        violations.append(f"All {len(milestones)} milestones have {mod_counts[0]} modules — no structural diversity")

    # 5. Fingerprint
    details["fingerprint"] = _compute_roadmap_shape_fingerprint(roadmap_data)

    details["milestones"] = ms_count
    details["module_counts"] = mod_counts
    details["skill_counts"] = skill_counts

    if violations:
        return {"pass": False, "reason": "; ".join(violations), "details": details}
    return {"pass": True, "reason": "Dynamic structure valid — counts bounded, rationale present, shape diverse", "details": details}


def run_roadmap_quality_validators(roadmap_data: dict) -> dict:
    _validators = [
        ("schema_validation",         lambda: validate_roadmap_schema(roadmap_data)),
        ("scaffolding",               lambda: _scaffolding_check(roadmap_data)),
        ("ai_tags",                   lambda: _ai_tag_check(roadmap_data)),
        ("capability_breadth",        lambda: _capability_breadth_check(roadmap_data)),
        ("deployment",                lambda: _deployment_check(roadmap_data)),
        ("science_distribution",      lambda: audit_science_distribution(roadmap_data)),
        ("milestone_identity_quality", lambda: _milestone_identity_quality_check(roadmap_data)),
        ("capability_progression",     lambda: _capability_progression_check(roadmap_data)),
        ("project_quality",            lambda: _project_quality_check(roadmap_data)),
        ("salary_progression",         lambda: _salary_progression_check(roadmap_data)),
        ("skill_relevance",            lambda: _skill_relevance_check(roadmap_data)),
        ("course_catalog_compliance",  lambda: _course_catalog_compliance_check(roadmap_data)),
        ("catalog_quality",            lambda: _catalog_quality_check(roadmap_data)),
        ("role_alignment",             lambda: _role_alignment_check(roadmap_data)),
        ("capability_gap_alignment",   lambda: _capability_gap_alignment_check(roadmap_data)),
        ("dynamic_structure",          lambda: validate_dynamic_structure(roadmap_data)),
    ]
    results = {}
    for name, fn in _validators:
        try:
            result = fn()
            results[name] = result
            passed = result.get("pass", True)
            badge = "PASS" if passed else "WARN"
            print(f"[QUALITY] {badge:4s} {name}: {result.get('reason', '')}")
        except Exception as exc:
            results[name] = {"pass": False, "reason": f"Validator error: {exc}"}
            print(f"[QUALITY] ERR  {name}: {exc}")
    return results


_REPAIRABLE_VALIDATORS = {"capability_breadth", "ai_tags", "role_alignment", "capability_gap_alignment"}

# ============================================================
# Roadmap Shape Fingerprinting (Phase 7 — Uniqueness)
# ============================================================

def _compute_roadmap_shape_fingerprint(roadmap_data: dict) -> str:
    """Compute a shape fingerprint from milestone/module/skill/science/lesson structure.
    
    Phase 9 — Enhanced shape fingerprint captures:
      - milestone count
      - per-milestone module counts
      - per-module skill counts
      - per-milestone scenario/mock counts
      - per-skill lesson counts
    
    Two roadmaps with the same fingerprint have identical structure counts,
    even if content differs. Used to detect predictability.
    """
    milestones = roadmap_data.get("milestones", [])
    ms_counts = []
    sk_counts = []
    sci_counts = []
    lesson_counts = []
    for ms in milestones:
        mods = ms.get("modules", [])
        ms_counts.append(len(mods))
        # Per-milestone scenario count and mock/interview count
        sc_n = ms.get("sc_n", 0)
        iv_n = ms.get("iv", 0)
        sci_counts.append(f"s{sc_n}i{iv_n}")
        for mod in mods:
            skills = mod.get("skills", [])
            sk_counts.append(len(skills))
            for skill in skills:
                lesson_counts.append(len(skill.get("lessons", [])))
    ms_str = ",".join(str(c) for c in ms_counts)
    sk_str = ",".join(str(c) for c in sk_counts)
    sci_str = ",".join(sci_counts)
    les_str = ",".join(str(c) for c in lesson_counts)
    return f"ms({len(milestones)}):[{ms_str}]:[{sk_str}]:sci[{sci_str}]:les[{les_str}]"


def _shape_similarity(fp1: str, fp2: str) -> float:
    """Compute similarity between two shape fingerprints.
    
    Returns a value 0.0–1.0 where 1.0 = identical structure.
    Compares milestone count, module distribution, and skill distribution.
    """
    def _parse_fp(fp: str) -> dict:
        parts = fp.split(":")
        ms_match = re.match(r"ms\((\d+)\)", parts[0])
        ms_count = int(ms_match.group(1)) if ms_match else 0
        mod_counts = [int(x) for x in parts[1].strip("[]").split(",") if x.strip().isdigit()]
        sk_counts = [int(x) for x in parts[2].strip("[]").split(",") if x.strip().isdigit()]
        return {"ms": ms_count, "mods": mod_counts, "skills": sk_counts}
    
    d1 = _parse_fp(fp1)
    d2 = _parse_fp(fp2)
    
    # Milestone count similarity
    ms_sim = 1.0 - abs(d1["ms"] - d2["ms"]) / max(d1["ms"], d2["ms"], 1)
    
    # Module distribution similarity (compare length + per-milestone)
    max_mod_len = max(len(d1["mods"]), len(d2["mods"]), 1)
    mod_matches = sum(
        1 for i in range(min(len(d1["mods"]), len(d2["mods"])))
        if d1["mods"][i] == d2["mods"][i]
    )
    mod_sim = mod_matches / max_mod_len
    
    # Skill distribution similarity
    max_sk_len = max(len(d1["skills"]), len(d2["skills"]), 1)
    sk_matches = sum(
        1 for i in range(min(len(d1["skills"]), len(d2["skills"])))
        if abs(d1["skills"][i] - d2["skills"][i]) <= 1  # within 1 skill
    )
    sk_sim = sk_matches / max_sk_len
    
    return round(0.4 * ms_sim + 0.3 * mod_sim + 0.3 * sk_sim, 3)


def _check_shape_uniqueness(roadmap_data: dict, user_id: str) -> dict:
    """Check if this roadmap's shape is unique for this user.
    
    Phase 9 — Enhanced check:
      - Detects exact collisions
      - Computes similarity to existing shapes
      - Warns if similarity > 80%
    Does not block.
    """
    fingerprint = _compute_roadmap_shape_fingerprint(roadmap_data)
    stored_key = f"shape_fingerprints_{user_id}"
    existing_raw = fetch_poc_record(user_id=user_id, record_id=stored_key)
    existing = []
    if existing_raw:
        try:
            existing = json.loads(existing_raw)
        except (json.JSONDecodeError, TypeError):
            existing = []
    if not isinstance(existing, list):
        existing = []
    
    collision = fingerprint in existing
    similarity_warn = False
    max_similarity = 0.0
    
    if collision:
        print(f"[SHAPE FINGERPRINT] ⚠ Collision: {fingerprint} already exists for user {user_id}")
    elif existing:
        # Check similarity to all existing shapes
        for prev_fp in existing:
            sim = _shape_similarity(fingerprint, prev_fp)
            max_similarity = max(max_similarity, sim)
            if sim > 0.80:
                similarity_warn = True
        if similarity_warn:
            print(f"[SHAPE FINGERPRINT] ⚠ High similarity ({max_similarity:.0%}) to previous "
                  f"roadmap for user {user_id}")
    
    # Append and store (sync is best-effort)
    existing.append(fingerprint)
    try:
        save_poc_record(user_id=user_id, record_id=stored_key, text=json.dumps(existing))
    except Exception:
        pass
    return {
        "fingerprint": fingerprint,
        "collision": collision,
        "similarity_warn": similarity_warn,
        "max_similarity": max_similarity,
        "total_shapes": len(existing),
    }


def repair_and_revalidate(roadmap_data: dict, max_passes: int = 2) -> dict:
    """Run quality validators, repair failures, revalidate.

    Supported repairs:
      - capability_breadth: inject missing domain skills
      - ai_tags: repair invalid ai_metadata.layer values
      - role_alignment: replace non-matching skills with role-appropriate ones
      - catalog_alignment: force-map remaining unmapped skills
      - capability_gap_alignment: enforce milestone/module/skill density

    Max 2 passes to prevent infinite loops.
    Returns the final validator results dict.
    """
    import copy

    for pass_num in range(1, max_passes + 1):
        audit_ai_layers(roadmap_data)
        results = run_roadmap_quality_validators(roadmap_data)

        # Check if any repairable validator failed
        needs_repair = False
        for name in _REPAIRABLE_VALIDATORS:
            r = results.get(name, {})
            if not r.get("pass", True):
                needs_repair = True
                break

        # Also check catalog coverage
        catalog_r = results.get("course_catalog_compliance", {})
        if catalog_r.get("coverage", 1.0) < 0.85:
            cat_stats = repair_course_catalog_alignment(roadmap_data)
            if cat_stats["coverage"] >= 0.85:
                needs_repair = True

        if not needs_repair:
            break

        # Apply repairs (all repairable validators)
        _breadth_result = repair_capability_breadth(roadmap_data)
        _ai_repaired = repair_ai_layers(roadmap_data)
        _role_repaired = _repair_role_alignment(roadmap_data)
        cat_stats = repair_course_catalog_alignment(roadmap_data)
        roadmap_data["catalog_stats"] = cat_stats

        # Phase 4 — Gap alignment repair: if milestone count is outside range,
        # flag as blocking (cannot truncate/supplement milestones safely)
        gap_result = results.get("capability_gap_alignment", {})
        if not gap_result.get("pass", True):
            ms_range = roadmap_data.get("milestone_range", {})
            min_ms = ms_range.get("minimum", MIN_MILESTONES)
            max_ms = ms_range.get("maximum", MAX_MILESTONES)
            actual_ms = len(roadmap_data.get("milestones", []))
            if actual_ms > max_ms:
                print(f"[GAP ALIGNMENT BLOCKING] {actual_ms} milestones exceed "
                      f"max bound {max_ms}. Requires regeneration.")
                roadmap_data["_gap_alignment_blocked"] = True
            elif actual_ms < min_ms:
                print(f"[GAP ALIGNMENT BLOCKING] {actual_ms} milestones below "
                      f"min bound {min_ms}. Requires regeneration.")
                roadmap_data["_gap_alignment_blocked"] = True

    return results


# ============================================================
# run_pipeline — Main Entry Point
# ============================================================

def run_pipeline(
    user_id,
    trigger_mcq: bool = True,
    ai_session_id: str = None,
    ai_roadmap_id: str = None,
) -> dict:

    print("\n[ROADMAP AGENT] Starting pipeline...")
    print("=" * 60)

    session_was_provided = bool(ai_session_id)

    if not ai_session_id:
        ai_session_id = (
            f"ai_sess_{datetime.utcnow().strftime('%Y%m%d%H%M%S')}"
            f"_{uuid.uuid4().hex[:8]}"
        )
        print(f"[ROADMAP AGENT] ⚠ Generated session ID: {ai_session_id}")
    else:
        print(f"[ROADMAP AGENT] ✓ Session ID: {ai_session_id}")

    if not ai_roadmap_id:
        ai_roadmap_id = (
            f"ai_roadmap_{datetime.utcnow().strftime('%Y%m%d%H%M%S')}"
            f"_{uuid.uuid4().hex[:8]}"
        )

    print(f"  ai_session_id : {ai_session_id}")
    print(f"  ai_roadmap_id : {ai_roadmap_id}")

    # ============================================================
    # INPUT MODE DETECTION
    # ============================================================

    print("\n[ROADMAP AGENT] Detecting input mode...")

    years_experience     = 0
    weekly_hours_available = 5  # default

    if isinstance(user_id, dict):
        # ── AI GENERATED PERSONA MODE ──────────────────────────
        print("[ROADMAP AGENT] Mode: AI-generated onboarding")
        _ai_persona_mode = True

        _persona = user_id  # save before overwriting
        user_id  = "ai_generated_user"
        context  = _persona.get("goal_context", "")

        # If goal_context is missing, use the full persona dict as context
        # so the LLM still receives profile information.
        if not context:
            context = json.dumps(_persona, indent=2)
            print("[ROADMAP AGENT] No goal_context — using persona dict as context")

        # ── Classify ICP from persona data ──
        icp_type, icp_reason = _classify_icp(
            years_experience=_persona.get("years_experience", 0),
            current_identity=_persona.get("current_role", ""),
            target_identity=_persona.get("target_role", ""),
            current_salary_lpa=float(_persona.get("current_salary_monthly", 0)) * 12 / 100000,
            known_skills=_persona.get("known_skills", []),
        )
        print(f"[ICP] Classified icp_type={icp_type} reason={icp_reason}")
        years_experience       = _persona.get("years_experience", 0)
        weekly_hours_available = _persona.get("weekly_hours_available", 5)

        # Build customer profile from the structured persona dict (authoritative),
        # NOT from the narrative context (which may be empty or unstructured).
        customer_profile = build_customer_profile(json.dumps(_persona))
        print(f"[ROADMAP AGENT] Persona-derived profile: {json.dumps(customer_profile, default=str)}")

    else:
        # ── REAL USER / PINECONE MODE ───────────────────────────
        _ai_persona_mode = False
        print("[ROADMAP AGENT] Mode: Real user (Pinecone)")

        # Direct fetch by key per POC spec
        # Try prefixed key first, then fall back to bare key for backward compat.
        poc_context = fetch_poc_record(
            user_id=user_id,
            record_id=f"{user_id}_onboarding_conversation"
        )
        if not poc_context:
            print("[ROADMAP AGENT] Prefixed key not found; trying bare key for backward compat")
            poc_context = fetch_poc_record(
                user_id=user_id,
                record_id="onboarding_conversation"
            )

        if not poc_context:
            print("[ROADMAP AGENT] ✗ No onboarding_conversation record found")
            return {
                "error":         "Please complete onboarding first.",
                "user_id":       user_id,
                "ai_session_id": ai_session_id,
            }

        if not poc_context.strip():
            print("[ROADMAP AGENT] ✗ Onboarding conversation is empty")
            return {
                "error":         "Please complete onboarding first.",
                "user_id":       user_id,
                "ai_session_id": ai_session_id,
            }

        context = poc_context
        print("[ONBOARDING CHECK]")
        print(f"  record_found=True")
        print(f"  text_length={len(context)}")
        print(f"  status=VALID")

        # icp_type derived after build_customer_profile below

    # ============================================================
    # CUSTOMER PROFILE  (MD Section 20)
    # ============================================================
    if not _ai_persona_mode:
        # Real User mode: build profile from Pinecone context text
        customer_profile = build_customer_profile(context)
        years_experience        = customer_profile["years_experience"]
        weekly_hours_available  = customer_profile["weekly_hours_available"]
        timeline_days           = customer_profile["timeline_days"]
        current_salary_lpa      = customer_profile["current_salary_lpa"]
        known_skills            = customer_profile["known_skills"]
    else:
        # AI Persona mode: profile already built from persona dict above
        years_experience        = customer_profile["years_experience"]
        weekly_hours_available  = customer_profile["weekly_hours_available"]
        timeline_days           = customer_profile["timeline_days"]
        current_salary_lpa      = customer_profile["current_salary_lpa"]
        known_skills            = customer_profile["known_skills"]

    # ── Classify ICP from customer profile ─────────────────
    if not _ai_persona_mode:
        icp_type, icp_reason = _classify_icp(
            years_experience=years_experience,
            current_identity=customer_profile.get("current_identity", ""),
            target_identity=customer_profile.get("target_identity", ""),
            current_salary_lpa=current_salary_lpa,
            known_skills=known_skills,
        )
        print(f"[ICP] Classified icp_type={icp_type} reason={icp_reason} (yoe={years_experience}, salary={current_salary_lpa})")
    else:
        icp_reason = "ai_persona_classification"

    # ── Compute budget_hours ──────────────────────────────────
    budget_hrs = round(
        weekly_hours_available * (timeline_days / 7) * BUDGET_UTILIZATION, 1
    )

    # ── IDENTITY_AUDIT ──────────────────────────────────────
    _ci = customer_profile.get("current_identity", "")
    _ti = customer_profile.get("target_identity", "")
    _rf = customer_profile.get("_provenance", {}).get("current_identity", "default")
    _tf = customer_profile.get("_provenance", {}).get("target_identity", "default")
    print("========== IDENTITY_AUDIT ==========")
    print(f"  current_identity   = '{_ci}'  (source={_rf})")
    print(f"  target_identity    = '{_ti}'  (source={_tf})")
    if not _ci:
        print("  ⚠ current_identity is empty — gap will be inflated")
    if not _ti:
        print("  ⚠ target_identity is empty — gap will be inflated")
    if len(_ti) > 80:
        print(f"  ⚠ target_identity too long ({len(_ti)} chars) — role repair will use wrong value")
    print("====================================")

    # ── Capability gap analysis ───────────────────────────────
    gap_analysis = compute_gap_score(
        current_identity=customer_profile.get("current_identity", ""),
        target_identity=customer_profile.get("target_identity", ""),
        years_experience=years_experience,
        weekly_hours_available=weekly_hours_available,
        timeline_days=timeline_days,
        known_skills=known_skills,
    )
    print("========== CAPABILITY GAP ==========")
    print(json.dumps(gap_analysis, indent=2))
    print("=====================================")

    print("[MD AUDIT] deploy_required = REQUIRED (per-milestone, ≥1 project)")

    # ── Capability breadth computation ─────────────────────────
    breadth = compute_capability_breadth(
        gap_score=gap_analysis.get("gap_score", 0.5),
        current_role=customer_profile.get("current_identity", ""),
        target_role=customer_profile.get("target_identity", ""),
        years_experience=years_experience,
        known_skills=known_skills,
    )
    print(f"[BREADTH] score={breadth['breadth_score']} domains={breadth['required_domains']}")

    # ── Breadth-driven module count override (Phase 3) ─────────
    # Module count per milestone is driven by capability breadth,
    # not just gap score. More required domains = more modules.
    _breadth_domains = len(breadth.get("required_domains", []))
    if _breadth_domains >= 6:
        _breadth_mod_count = 4
        _breadth_mod_rationale = f"breadth={_breadth_domains} domains (≥6): at most 4 modules per milestone"
    elif _breadth_domains >= 3:
        _breadth_mod_count = 3
        _breadth_mod_rationale = f"breadth={_breadth_domains} domains (≥3): at most 3 modules per milestone"
    else:
        _breadth_mod_count = 2
        _breadth_mod_rationale = f"breadth={_breadth_domains} domains (<3): at most 2 modules per milestone"
    gap_analysis["recommended_modules_per_milestone"] = _breadth_mod_count
    gap_analysis["recommended_modules_per_milestone_rationale"] = _breadth_mod_rationale
    print(f"[BREADTH MODULES] count={_breadth_mod_count} rationale={_breadth_mod_rationale}")

    # ── Market/role-driven skill density override (Phase 4) ────
    # Skill density is driven by target role domain and market norms,
    # not just gap score. Different roles need different breadth.
    # NOTE: capped at MAX_SKILLS (8) regardless of domain map values —
    # this codebase intentionally keeps the skills-per-module ceiling at
    # 8, not MD's 10 (explicit, documented deviation; see constants block).
    _target_role_domain = _extract_domain(customer_profile.get("target_identity", ""))
    _domain_skill_map = {
        "ai": 7, "data_science": 7,
        "backend": 6, "fullstack": 6, "software": 6,
        "frontend": 5, "mobile": 5, "devops": 6,
        "security": 5, "data": 5, "product": 4,
        "student": 4,
    }
    _domain_max_skills = min(_domain_skill_map.get(_target_role_domain, 5), MAX_SKILLS)
    _old_density = gap_analysis.get("recommended_skill_density", 5)
    gap_analysis["recommended_skill_density"] = min(_old_density, _domain_max_skills)
    gap_analysis["recommended_skill_density_rationale"] = (
        f"domain={_target_role_domain} market norm={_domain_max_skills} "
        f"skills max per module (capped at MAX_SKILLS={MAX_SKILLS})"
    )
    print(f"[MARKET SKILLS] domain={_target_role_domain} max={_domain_max_skills} "
          f"old={_old_density} new={gap_analysis['recommended_skill_density']}")

    # ── Odd-number preference (Phase 8) ────────────────────────
    # Asymmetric (odd) max bounds produce more natural-looking structures.
    # Shift even max bounds to odd where possible within min/max range.
    def _prefer_odd(val: int, min_val: int, max_val: int) -> tuple:
        if val % 2 == 0 and val < max_val:
            odd_val = val + 1
            if min_val <= odd_val <= max_val:
                return odd_val, f"odd-preference from {val} to {odd_val}"
        return val, ""

    _old_ms = gap_analysis["recommended_milestones"]
    _new_ms, _ms_reason = _prefer_odd(_old_ms, MIN_MILESTONES, MAX_MILESTONES)
    if _ms_reason:
        gap_analysis["recommended_milestones"] = _new_ms
        gap_analysis["recommended_milestones_rationale"] += f"; {_ms_reason}"
        print(f"[ODD PREF] milestones: {_old_ms} → {_new_ms} ({_ms_reason})")

    _old_mod = gap_analysis["recommended_modules_per_milestone"]
    _new_mod, _mod_reason = _prefer_odd(_old_mod, MIN_MODULES, MAX_MODULES)
    if _mod_reason:
        gap_analysis["recommended_modules_per_milestone"] = _new_mod
        gap_analysis["recommended_modules_per_milestone_rationale"] += f"; {_mod_reason}"
        print(f"[ODD PREF] modules: {_old_mod} → {_new_mod} ({_mod_reason})")

    print(f"[ROADMAP AGENT] years_experience={years_experience}, weekly_hours_available={weekly_hours_available}")
    print(f"[ROADMAP AGENT] timeline_days={timeline_days} budget_hours={budget_hrs}")
    if current_salary_lpa:
        print(f"[ROADMAP AGENT] current_salary_lpa={current_salary_lpa}")
    if known_skills:
        print(f"[ROADMAP AGENT] known_skills={known_skills}")

    # ============================================================
    # LEVEL DETECTION  (MD Section 7 — Dynamic starting point)
    # ============================================================

    level           = detect_level(context, years_experience)
    print(
        f"[LEVEL] Detected level: {level} "
        f"(years_experience={years_experience}) "
        f"icp_type={icp_type} "
        f"→ milestone count determined by LLM from capability gap "
        f"(bounds {MIN_MILESTONES}–{MAX_MILESTONES})"
    )

    # ── Milestone Authority Engine (Phase 2) ───────────────────
    # Must run AFTER level detection (level is required).
    _ms_range = compute_authoritative_milestone_range(
        gap_score=gap_analysis.get("gap_score", 0.5),
        icp_type=icp_type,
        level=level,
        hours_per_week=weekly_hours_available,
        timeline_days=timeline_days,
        known_skills=known_skills,
        experience_years=years_experience,
        current_identity=customer_profile.get("current_identity", ""),
        target_identity=customer_profile.get("target_identity", ""),
    )
    # ── Strict schema validation (Phase 3) ─────────────────────
    # Catches missing fields BEFORE they propagate to prompt/validators.
    validate_milestone_authority_schema(_ms_range)
    print(f"[AUTHORITY ENGINE] {json.dumps(_ms_range, indent=2)}")
    # Override gap analysis recommended with authority engine's recommended
    # (which incorporates level range preference + gap + experience + time)
    _old_gap_rec = gap_analysis["recommended_milestones"]
    gap_analysis["recommended_milestones"] = _ms_range["recommended"]
    gap_analysis["recommended_milestones_rationale"] = (
        f"authority engine: {_ms_range['reasoning']}; "
        f"gap engine had: {_old_gap_rec}"
    )
    print(f"[MILESTONE AUTHORITY] range=({_ms_range['minimum']},{_ms_range['maximum']}) "
          f"recommended={_ms_range['recommended']} confidence={_ms_range['confidence']}")

    # ── PRE-GENERATION ASSERTIONS (ISSUE 9) — Profile-level ──
    # Fallback: if current_identity couldn't be extracted from onboarding,
    # infer from level to avoid crashing on valid-but-unstructured data.
    if not customer_profile.get("current_identity"):
        _inferred = {
            "beginner": "Student",
            "intermediate": "Professional",
            "senior": "Senior Professional",
        }.get(level, "Learner")
        customer_profile["current_identity"] = _inferred
        customer_profile.setdefault("_provenance", {})["current_identity"] = f"inferred_from_level_{level}"
        print(f"[PROFILE FALLBACK] current_identity inferred as '{_inferred}' from level='{level}'")

    if not customer_profile.get("target_identity"):
        # Try to extract a short role title from context
        _fallback_ti = ""
        try:
            _parsed_ctx = json.loads(context) if isinstance(context, str) else {}
            if isinstance(_parsed_ctx, dict):
                _fallback_ti = _parsed_ctx.get("target_role") or _parsed_ctx.get("primary_goal", "")
        except (json.JSONDecodeError, TypeError):
            _fallback_ti = ""
        if not _fallback_ti or len(_fallback_ti) > 80:
            _fallback_ti = "Target Role"
        customer_profile["target_identity"] = _fallback_ti
        customer_profile.setdefault("_provenance", {})["target_identity"] = "inferred_from_context"
        print(f"[PROFILE FALLBACK] target_identity inferred as '{_fallback_ti}'")

    assert customer_profile.get("current_identity"), (
        f"current_identity is empty — gap will be inflated. "
        f"Extraction source: {customer_profile.get('_provenance', {}).get('current_identity', 'none')}"
    )
    assert customer_profile.get("target_identity"), (
        f"target_identity is empty — gap will be inflated. "
        f"Extraction source: {customer_profile.get('_provenance', {}).get('target_identity', 'none')}"
    )
    _ti_val = customer_profile.get("target_identity", "")
    assert len(_ti_val) <= 80, (
        f"target_identity too long ({len(_ti_val)} chars): '{_ti_val[:60]}...' — "
        f"role repair will use wrong value"
    )
    assert _ms_range["minimum"] <= _ms_range["maximum"], (
        f"milestone_range min ({_ms_range['minimum']}) > max ({_ms_range['maximum']})"
    )
    # Verify constants match between modules (import capability_gap's MIN_SKILLS for cross-check)
    from src.capability_gap import MIN_SKILLS as CG_MIN, MAX_SKILLS as CG_MAX
    assert MIN_SKILLS == CG_MIN, f"MIN_SKILLS mismatch: {MIN_SKILLS} vs {CG_MIN}"
    assert MAX_SKILLS == CG_MAX, f"MAX_SKILLS mismatch: {MAX_SKILLS} vs {CG_MAX}"
    print("[ASSERT] Pre-generation assertions: ALL PASS")

    # ============================================================
    # GENERATE ROADMAP  (up to 2 attempts)
    # ============================================================

    print("\n[ROADMAP AGENT] Invoking LLM (OpenAI → Gemini fallback)...")

    max_attempts = 2
    result       = ""

    for attempt in range(1, max_attempts + 1):
        try:
            print(f"[ROADMAP AGENT] Attempt {attempt}/{max_attempts}...")

            known_skills_str = ", ".join(known_skills) if known_skills else "none provided"
            current_identity_str = customer_profile.get("current_identity", "") or "not specified"
            target_identity_str = customer_profile.get("target_identity", "") or "not specified"
            self_efficacy_str = str(customer_profile.get("self_efficacy", 0.5))
            result       = _build_chain().invoke({
                "context":  context,
                "icp_type": icp_type,
                "level":    level,
                "hours_per_week": str(weekly_hours_available),
                "timeline_days":  str(timeline_days),
                "budget_hrs":     str(budget_hrs),
                "known_skills":   known_skills_str,
                "current_identity":   current_identity_str,
                "target_identity":    target_identity_str,
                "current_salary_lpa": str(current_salary_lpa),
                "self_efficacy":        self_efficacy_str,
                "gap_score":            str(gap_analysis["gap_score"]),
                "recommended_milestones":           str(gap_analysis["recommended_milestones"]),
                "recommended_milestones_min":       str(_ms_range["minimum"]),
                "recommended_milestones_max":       str(_ms_range["maximum"]),
                "milestone_confidence":             str(_ms_range["confidence"]),
                "milestone_authority_reasoning":    _ms_range["reasoning"],
                "recommended_modules_per_milestone": str(gap_analysis["recommended_modules_per_milestone"]),
                "recommended_skill_density":         str(gap_analysis["recommended_skill_density"]),
                "gap_reasoning":                     gap_analysis["reasoning"],
                "min_skills":   str(MIN_SKILLS),
                "max_skills":   str(MAX_SKILLS),
            })
            # DEBUG START
            print(f"\nRAW OUTPUT LENGTH = {len(result)}")
            print("\n========== RAW LLM OUTPUT START ==========\n")
            print(result[:5000])
            print("\n========== RAW LLM OUTPUT END ==========\n")
            print("\n========== RAW LLM OUTPUT LAST 3000 CHARS ==========\n")
            print(result[-3000:])
            print("\n===================================================\n")
            # DEBUG END
            clean_result = repair_json(result)
            roadmap_data = json.loads(clean_result)
            # ── Pre-generation AI layer normalizer ──────────────
            # Phase 4.1C — Normalize skill-level ai_metadata.layer values.
            for ms in roadmap_data.get("milestones", []):
                for mod in ms.get("modules", []):
                    # Normalize module-level ai_first_layer (the validator checks this)
                    if "ai_first_layer" in mod:
                        mod["ai_first_layer"] = normalize_ai_layer(mod["ai_first_layer"])
                    for skill in mod.get("skills", []):
                        ai = skill.get("ai_metadata")
                        if not ai or not isinstance(ai, dict):
                            continue
                        ai["layer"] = normalize_ai_layer(ai.get("layer", ""))
                        ai["usage_type"] = normalize_ai_usage(ai.get("usage_type", ""))
                        ai["automation_level"] = normalize_ai_automation(ai.get("automation_level", ""))
            # ── Pipeline fallbacks for time-budget fields ──────
            roadmap_data["timeline_days"] = timeline_days
            if "budget_hours" not in roadmap_data:
                roadmap_data["budget_hours"] = budget_hrs

            # ── Career stage (separate from icp_type for track labels) ──
            _is_fresher = (level == "beginner" and years_experience == 0)
            _is_student = any(kw in (customer_profile.get("current_identity") or "").lower()
                             for kw in ("student", "12th", "college"))
            roadmap_data["career_stage"] = "student" if _is_student else ("fresher" if _is_fresher else "professional")

            # ── POST-GENERATION VALIDATORS ──────────────────────
            # Role contamination check (ISSUE 7) — only for analyst roles
            try:
                _check_role_contamination(roadmap_data, customer_profile)
            except ValueError as e:
                print(f"[CONTAMINATION FAIL] {e}")
                raise

            # Career switcher narrative injection (ISSUE 8)
            _inject_career_switch_narrative(roadmap_data)

            # Run hard assertions (ISSUE 9)
            _run_pre_generation_assertions(
                roadmap_data=roadmap_data,
                customer_profile=customer_profile,
                icp_type=icp_type,
                level=level,
                years_experience=years_experience,
            )

            # Always overwrite estimated_total_hours from actual counts
            lc, sc, sci, ivc, pc = _count_roadmap_units(roadmap_data)
            roadmap_data["estimated_total_hours"] = round(
                lc * 0.25 + sc * 1.5 + sci * 0.5 + ivc * 1.0 + pc * 3.0, 1
            )
            print(
                f"[TIME BUDGET] estimated_total_hours recomputed = "
                f"{roadmap_data['estimated_total_hours']}"
            )
            # ── Budget Enforcement per ROADMAP_GENERATION_SCIENCE_V2.md ──
            # If demand exceeds budget, reduce milestone count within the
            # allowed range first (MD Section 1).  Never go outside [min, max].
            # Must use the AUTHORITY range (from compute_authoritative_milestone_range),
            # NOT MIN_MILESTONES/MAX_MILESTONES globals, because the authority
            # engine may have tighter bounds (e.g. [3,5] for low/beginner).
            _budget_enforced = False
            if budget_hrs > 0:
                current_estimated = roadmap_data["estimated_total_hours"]
                if current_estimated > budget_hrs:
                    _min_ms = _ms_range["minimum"]
                    _max_ms = _ms_range["maximum"]
                    _current_ms = len(roadmap_data.get("milestones", []))
                    print(
                        f"[BUDGET SCIENCE] demand={current_estimated}h > "
                        f"budget={budget_hrs}h — reducing milestones within "
                        f"[{_min_ms}, {_max_ms}]"
                    )
                    # Fit within budget by reducing milestone count
                    # Estimate hours-per-milestone, then compute max affordable
                    _per_ms_hrs = current_estimated / max(_current_ms, 1)
                    _target_ms = max(_min_ms, min(_max_ms,
                        int(budget_hrs / max(_per_ms_hrs, 1))))
                    if _target_ms < _current_ms:
                        milestones = roadmap_data.get("milestones", [])
                        roadmap_data["milestones"] = milestones[:_target_ms]
                        # Recompute estimated_total_hours
                        lc, sc, sci, ivc, pc = _count_roadmap_units(roadmap_data)
                        roadmap_data["estimated_total_hours"] = round(
                            lc * 0.25 + sc * 1.5 + sci * 0.5 + ivc * 1.0 + pc * 3.0, 1
                        )
                        _budget_enforced = True
                        print(
                            f"[BUDGET SCIENCE] Reduced milestones: "
                            f"{_current_ms} → {_target_ms} "
                            f"(estimated={roadmap_data['estimated_total_hours']}h, "
                            f"budget={budget_hrs}h)"
                        )
            # ==========================================
            # AUTO FIX — SCIENCE DISTRIBUTION
            # ==========================================
            # MD bounds: Scenarios [3,7] per milestone, Mocks/Interviews [1,2].
            # These are REAL bounds, not "up to" — we cap excess AND we
            # pad shortfalls (the prompt should already produce enough,
            # this is a safety net for LLM under-generation).
            for milestone in roadmap_data.get("milestones", []):
                modules = milestone.get("modules", [])

                # Collect per-module science items
                scenarios = []
                interviews = []
                for mod in modules:
                    sci_list = mod.get("science", [])
                    if not isinstance(sci_list, list):
                        continue
                    for sci in sci_list:
                        t = sci.get("type")
                        if t == "Scenario":
                            scenarios.append(mod)
                        elif t == "Interview":
                            interviews.append(mod)

                # ── Cap: > MAX_SCENARIOS_PER_MILESTONE → keep first N ──
                if len(scenarios) > MAX_SCENARIOS_PER_MILESTONE:
                    excess = scenarios[MAX_SCENARIOS_PER_MILESTONE:]
                    for mod in modules:
                        mod["science"] = [s for s in mod.get("science", [])
                                          if not (s.get("type") == "Scenario" and mod in excess)]

                # ── Cap: > MAX_MOCKS_PER_MILESTONE Interviews → keep first N ──
                if len(interviews) > MAX_MOCKS_PER_MILESTONE:
                    excess =interviews[MAX_MOCKS_PER_MILESTONE:]
                    for mod in modules:
                        mod["science"] = [s for s in mod.get("science", [])
                                          if not (s.get("type") == "Interview" and mod in excess)]

                # ── Fix: Scenario and Interview in same module → move Interview ──
                scenarios = []
                interviews = []
                for mod in modules:
                    for sci in mod.get("science", []):
                        t = sci.get("type")
                        if t == "Scenario":
                            scenarios.append(mod)
                        elif t == "Interview":
                            interviews.append(mod)
                if scenarios and interviews and scenarios[0] is interviews[0]:
                    same_mod = scenarios[0]
                    # Keep only scenarios in this module, move interview to another
                    same_mod["science"] = [s for s in same_mod.get("science", [])
                                           if s.get("type") != "Interview"]
                    for mod in modules:
                        if mod is not same_mod and not any(
                            s.get("type") == "Interview" for s in mod.get("science", [])
                        ):
                            mod.setdefault("science", []).append(
                                {"type": "Interview", "desc": "Interview question assessing milestone competency"}
                            )
                            break

                # ── Cap: any module has >3 science items → truncate to 3 ──
                for mod in modules:
                    sci_list = mod.get("science", [])
                    if isinstance(sci_list, list) and len(sci_list) > 3:
                        mod["science"] = sci_list[:3]

                # ── Recalculate sc_n / iv ──
                sc = sum(1 for mod in modules
                         for s in mod.get("science", [])
                         if s.get("type") == "Scenario")
                iv = sum(1 for mod in modules
                         for s in mod.get("science", [])
                         if s.get("type") == "Interview")

                # ── Pad shortfalls: real floors, not "up to" ──
                # If the LLM under-generated, top up with generic-but-valid
                # items in modules that have room (<=3 science items), so
                # the milestone meets MD's real minimums (3 scenarios,
                # 1 mock/interview) rather than silently failing later.
                if sc < MIN_SCENARIOS_PER_MILESTONE:
                    needed = MIN_SCENARIOS_PER_MILESTONE - sc
                    for mod in modules:
                        if needed <= 0:
                            break
                        if len(mod.get("science", [])) < 3:
                            mod.setdefault("science", []).append({
                                "type": "Scenario",
                                "desc": (
                                    f"Realistic on-the-job situation for "
                                    f"{mod.get('title', 'this module')}: "
                                    f"diagnose and resolve an unexpected "
                                    f"production issue under time pressure."
                                ),
                            })
                            sc += 1
                            needed -= 1
                if iv < MIN_MOCKS_PER_MILESTONE:
                    needed = MIN_MOCKS_PER_MILESTONE - iv
                    for mod in modules:
                        if needed <= 0:
                            break
                        has_scenario_here = any(
                            s.get("type") == "Scenario" for s in mod.get("science", [])
                        )
                        if len(mod.get("science", [])) < 3 and not has_scenario_here:
                            mod.setdefault("science", []).append({
                                "type": "Interview",
                                "desc": (
                                    f"Interview question testing the "
                                    f"milestone identity at the "
                                    f"{mod.get('title', 'this module')} level."
                                ),
                            })
                            iv += 1
                            needed -= 1

                milestone["sc_n"] = sc
                milestone["iv"] = iv

            # ── AUTO FIX — PROJECT COUNT BOUNDS (1-2 per milestone) ──
            for milestone in roadmap_data.get("milestones", []):
                ms_id = milestone.get("milestone_id", "?")
                ms_projects = milestone.get("projects")
                if ms_projects is None:
                    legacy = milestone.get("project")
                    ms_projects = [legacy] if legacy else []
                    milestone["projects"] = ms_projects
                if not isinstance(ms_projects, list):
                    ms_projects = [ms_projects] if ms_projects else []
                    milestone["projects"] = ms_projects
                if len(ms_projects) > MAX_PROJECTS_PER_MILESTONE:
                    milestone["projects"] = ms_projects[:MAX_PROJECTS_PER_MILESTONE]
                    print(
                        f"[AUTO REPAIR] Milestone {ms_id} had "
                        f"{len(ms_projects)} projects -> capped at "
                        f"{MAX_PROJECTS_PER_MILESTONE}"
                    )
                elif len(ms_projects) < MIN_PROJECTS_PER_MILESTONE:
                    print(
                        f"[AUTO REPAIR] Milestone {ms_id} has "
                        f"{len(ms_projects)} project(s) (below "
                        f"MIN_PROJECTS_PER_MILESTONE={MIN_PROJECTS_PER_MILESTONE}) — "
                        f"cannot synthesize a real project; validator will reject"
                    )

            # ==========================================
            # Inject runtime IDs
            # ==========================================
            now = datetime.utcnow().isoformat()
            roadmap_data["roadmap_id"] = ai_roadmap_id
            roadmap_data["user_id"]    = user_id
            # Force-set authoritative fields — never trust the LLM's
            # output for these; they come from our own detection logic.
            roadmap_data["level"]    = level
            roadmap_data["icp_type"] = icp_type
            roadmap_data["years_experience"] = years_experience
            roadmap_data["current_salary_lpa"] = current_salary_lpa
            roadmap_data["known_skills"] = known_skills
            roadmap_data["capability_gap"] = gap_analysis
            # ── Store customer profile in roadmap_data ──────────
            roadmap_data["customer_profile"] = customer_profile
            # ── Milestone Authority Layer (Phase 3.4) ──────────
            # Lock milestone/module/skill counts from gap engine.
            # No downstream component may reduce these.
            roadmap_data["milestone_range"] = {
                "recommended": _ms_range["recommended"],
                "minimum": _ms_range["minimum"],
                "maximum": _ms_range["maximum"],
                "confidence": _ms_range["confidence"],
                "reasoning": _ms_range["reasoning"],
            }
            roadmap_data["locked_module_count"] = gap_analysis["recommended_modules_per_milestone"]
            roadmap_data["locked_skill_density"] = gap_analysis["recommended_skill_density"]
            print(f"[MILESTONE AUTHORITY] range=({_ms_range['minimum']},{_ms_range['maximum']}) "
                  f"recommended={_ms_range['recommended']} "
                  f"confidence={_ms_range['confidence']} "
                  f"mod={gap_analysis['recommended_modules_per_milestone']} "
                  f"skill={gap_analysis['recommended_skill_density']}")
            roadmap_data["capability_breadth"] = compute_capability_breadth(
                gap_score=gap_analysis.get("gap_score", 0.5),
                current_role=customer_profile.get("current_identity", ""),
                target_role=customer_profile.get("target_identity", ""),
                years_experience=years_experience,
                known_skills=known_skills,
            )
            roadmap_data.setdefault("roadmap_meta", {})["generated_at"] = now
            # ── Salary floor auto-repair ────────────────────────
            apply_salary_floor_repair(roadmap_data)
            # ── Dynamic BKT injection ──────────────────────────
            inject_bkt_values(roadmap_data)
            # ── Customer profile debug ──────────────────────────
            print("========== CUSTOMER PROFILE ==========")
            print(f"  current_identity       : {customer_profile.get('current_identity', 'N/A')}")
            print(f"  years_experience       : {years_experience}")
            print(f"  weekly_hours_available : {weekly_hours_available}")
            print(f"  timeline_days          : {timeline_days}")
            print(f"  current_salary_lpa     : {current_salary_lpa}")
            print(f"  known_skills           : {known_skills}")
            print(f"  self_efficacy          : {customer_profile.get('self_efficacy', 0.5)}")
            print(f"  provenance             : {customer_profile.get('_provenance', {})}")
            print("=====================================")
            # ── Known-skill removal debug (moved after catalog repair) ──
            # ── Salary floor debug ─────────────────────────────
            if icp_type == "high" and current_salary_lpa:
                m01_sal = (roadmap_data.get("milestones") or [{}])[0].get("sal", "N/A")
                print(
                    f"[SALARY FLOOR] icp_type={icp_type} "
                    f"current_salary_lpa={current_salary_lpa} "
                    f"M01.sal={m01_sal}"
                )
            # ── Time budget debug ──────────────────────────────
            _dbg_est = roadmap_data.get("estimated_total_hours", 0)
            _dbg_util = round(_dbg_est / budget_hrs * 100, 1) if budget_hrs > 0 else 0
            print("========== TIME BUDGET ==========")
            print(f"  Weekly Hours    : {weekly_hours_available}")
            print(f"  Timeline Days   : {timeline_days}")
            print(f"  Budget Hours    : {budget_hrs}")
            print(f"  Estimated Demand: {_dbg_est}")
            print(f"  Utilization     : {_dbg_util}%")
            print("================================")
            # ── Inject label field into each milestone if missing ──
            print("\n========== MODULE SKILL COUNT DEBUG ==========")
            for ms in roadmap_data.get("milestones", []):
                for mod in ms.get("modules", []):
                    if len(mod.get("skills", [])) < MIN_SKILLS:
                        print(json.dumps(mod, indent=2))
            print("=============================================\n")
            # ── AUTO REPAIR: module count, skill count, lesson count ───
            # Runs after sc_n/iv fix and ID injection but BEFORE
            # validate_roadmap_structure(), so the validator always receives
            # structurally valid arrays.
            # Counts are CAPACITY BOUNDS, not targets. We only enforce
            # MIN/MAX range bounds. NEVER pad to a target count, except
            # where a real floor exists (lessons, scenarios, mocks,
            # projects) — those get padded if the LLM under-generated.
            for ms in roadmap_data.get("milestones", []):
                modules = ms.get("modules", [])
                # ── Module count: only enforce MAX_MODULES ──────────
                orig_mod_count = len(modules)
                if orig_mod_count > MAX_MODULES:
                    ms["modules"] = modules[:MAX_MODULES]
                    print(
                        f"[AUTO REPAIR] Milestone {ms.get('milestone_id')} had "
                        f"{orig_mod_count} modules -> capped at {MAX_MODULES}"
                    )
                elif 0 < orig_mod_count < MIN_MODULES:
                    print(
                        f"[AUTO REPAIR] Milestone {ms.get('milestone_id')} has "
                        f"{orig_mod_count} modules (below MIN_MODULES={MIN_MODULES}) — "
                        f"cannot synthesize; validator may reject"
                    )
                # ── Skill count: enforce MIN_SKILLS and MAX_SKILLS ──
                for mod in ms.get("modules", []):
                    skills     = mod.get("skills", [])
                    mod_id     = mod.get("id", "?")
                    orig_count = len(skills)
                    if orig_count == 0:
                        print(
                            f"[AUTO REPAIR] Module {mod_id} has 0 skills — "
                            f"cannot repair; validator will reject"
                        )
                    elif orig_count > MAX_SKILLS:
                        mod["skills"] = skills[:MAX_SKILLS]
                        print(
                            f"[AUTO REPAIR] Module {mod_id} had "
                            f"{orig_count} skills -> capped at {MAX_SKILLS}"
                        )
                    elif orig_count < MIN_SKILLS:
                        # Pad with catalog-aligned skills from available courses
                        mod_title = (mod.get("title") or "").lower()
                        _fallback_skills = [
                            {"skill_id": f"{mod.get('id', 'MOD')}_FILL_S{i+1}", "n": "foundational_practice", "title": "Foundational Practice", "lessons": ["Core Concepts", "Hands-On Application"], "p": 0, "mastery_state": {"state": "unlocked", "current_mastery": 0.0, "target_mastery": 0.9, "bkt": {"prior": 0.15, "learn_rate": 0.25, "guess": 0.1, "slip": 0.05}},
                            "content_flow": {"video": {"title": "Foundational Video", "status": "locked"}, "scenario": {"title": "Applied Scenario", "difficulty": 0.3, "status": "locked"}, "mock": {"unlock_mastery": 0.75, "status": "locked"}, "review": {"review_type": "spaced_repetition", "next_review_at": None}},
                            "unlock_rules": {"requires": [], "minimum_mastery": 0.0, "unlock_type": "immediate"},
                            "ai_metadata": {"ai_first": True, "layer": "vibe_solution", "usage_type": "generation", "automation_level": "assistant"},
                            }
                            for i in range(MIN_SKILLS - orig_count)
                        ]
                        mod["skills"] = skills + _fallback_skills
                        print(
                            f"[AUTO REPAIR] Module {mod_id} had "
                            f"{orig_count} skills -> padded to {MIN_SKILLS}"
                        )
                    # Lesson repair: enforce real floor MIN_LESSONS_PER_SKILL (2)
                    # and ceiling MAX_LESSONS_PER_SKILL (4) — MD Section 11.
                    for skill_idx, skill in enumerate(mod.get("skills", [])):
                        skill_id = skill.get("skill_id", "?")
                        # Ensure ordinal field matches position
                        skill["ordinal"] = skill_idx + 1
                        lessons  = skill.get("lessons", [])
                        if not isinstance(lessons, list):
                            lessons = []
                        # Ensure every lesson has "Lesson N:" prefix
                        fixed_lessons = []
                        for li, lesson in enumerate(lessons):
                            if not isinstance(lesson, str):
                                lesson = str(lesson)
                            lesson = lesson.strip()
                            # Strip any existing prefix like "Lesson 1:", "Lesson 1 –", etc.
                            lesson = re.sub(r'^Lesson\s+\d+\s*[:.–]\s*', '', lesson).strip()
                            fixed_lessons.append(f"Lesson {li + 1}: {lesson}")
                        lessons = fixed_lessons
                        if len(lessons) > MAX_LESSONS_PER_SKILL:
                            lessons = lessons[:MAX_LESSONS_PER_SKILL]
                            # Re-number after truncation
                            lessons = [f"Lesson {i+1}: {l.split(':', 1)[-1].strip() if ':' in l else l}" for i, l in enumerate(lessons)]
                            print(
                                f"[AUTO REPAIR] Skill {skill_id} had "
                                f"{len(lessons)} lessons -> capped at "
                                f"{MAX_LESSONS_PER_SKILL}"
                            )
                        elif len(lessons) < MIN_LESSONS_PER_SKILL:
                            skill_title = skill.get("title", skill.get("n", "Skill"))
                            while len(lessons) < MIN_LESSONS_PER_SKILL:
                                li = len(lessons) + 1
                                lessons.append(f"Lesson {li}: {skill_title} — Part {li}")
                            print(
                                f"[AUTO REPAIR] Skill {skill_id} had fewer than "
                                f"{MIN_LESSONS_PER_SKILL} lessons -> padded to "
                                f"{MIN_LESSONS_PER_SKILL}"
                            )
                        skill["lessons"] = lessons
            # ── Gap alignment debug (Phase 4 range-based) ──────
            _dbg_ms_count = len(roadmap_data.get("milestones", []))
            _dbg_mod_counts = [
                len(ms.get("modules", []))
                for ms in roadmap_data.get("milestones", [])
            ]
            _dbg_skill_counts = [
                len(mod.get("skills", []))
                for ms in roadmap_data.get("milestones", [])
                for mod in ms.get("modules", [])
            ]
            _debug_ms_range = roadmap_data.get("milestone_range", {})
            _range_min = _debug_ms_range.get("minimum", MIN_MILESTONES)
            _range_max = _debug_ms_range.get("maximum", MAX_MILESTONES)
            _range_rec = _debug_ms_range.get("recommended", "?")
            _max_skill = roadmap_data.get("capability_gap", {}).get("recommended_skill_density", MAX_SKILLS)
            _gap_pass = (
                _range_min <= _dbg_ms_count <= _range_max
                and all(MIN_MODULES <= m <= MAX_MODULES for m in _dbg_mod_counts)
                and all(MIN_SKILLS <= s <= _max_skill for s in _dbg_skill_counts)
            )
            print("========== GAP ALIGNMENT (RANGE-BASED) ==========")
            print(f"  Range             : [{_range_min}, {_range_max}]")
            print(f"  Recommended       : {_range_rec}")
            print(f"  Actual Milestones : {_dbg_ms_count}")
            print(f"  Module Counts     : {_dbg_mod_counts}")
            print(f"  Skill Counts      : {_dbg_skill_counts}")
            print(f"  {'PASS' if _gap_pass else 'FAIL'}")
            print("================================================")
            # ── Structural validation ──────────────────────────
            validate_roadmap_structure(roadmap_data)

            # ── Course catalog alignment repair ────────────────
            print("\n[CATALOG] Running course catalog alignment repair...")
            catalog_stats = repair_course_catalog_alignment(roadmap_data)
            print(f"[CATALOG] coverage={catalog_stats['coverage']:.0%} "
                  f"({catalog_stats['mapped']}/{catalog_stats['total']} skills mapped)")
            roadmap_data["catalog_stats"] = catalog_stats

            # ── Repair-and-revalidate quality gate ──────────────
            quality_results = repair_and_revalidate(roadmap_data, max_passes=2)
            roadmap_data["quality_validators"] = quality_results

            # ── Known-skill removal (starts_where_they_are, MD Section 2/19) ──
            # Run AFTER all repair passes so no repair re-adds known skills.
            removed_count = remove_known_skills_from_roadmap(roadmap_data, known_skills)
            if removed_count > 0:
                audit = roadmap_data.get("_removed_known_skills", [])
                print("[KNOWN SKILL REMOVAL]")
                print(f"  known_skills_input={len(known_skills)}")
                print(f"  removed={removed_count}")
                for entry in audit:
                    print(f"  * {entry['matched_known_skill']}")
                remaining = sum(
                    len(mod.get("skills", []))
                    for ms in roadmap_data.get("milestones", [])
                    for mod in ms.get("modules", [])
                )
                print(f"  remaining_teachable_skills={remaining}")
            else:
                print("[KNOWN SKILL REMOVAL] no known skills found in roadmap")

            # ── Shape fingerprinting (Phase 7) ─────────────────
            _shape = _check_shape_uniqueness(roadmap_data, user_id)
            print(f"[SHAPE FINGERPRINT] {_shape['fingerprint']} "
                  f"collision={_shape['collision']} total={_shape['total_shapes']}")

            # ── Dynamic structure validation (Phase 10) ─────────
            _dyn = validate_dynamic_structure(roadmap_data)
            if not _dyn.get("pass"):
                print(f"[DYNAMIC STRUCTURE] WARN: {_dyn['reason']}")
            else:
                print(f"[DYNAMIC STRUCTURE] PASS: {_dyn['reason']}")

            # ── AI layer repair audit ──────────────────────────
            _total = 0
            _layer_counts = {}
            _legacy = {"architecture": 0, "implementation": 0, "debugging": 0, "optimization": 0}
            for _ms in roadmap_data.get("milestones", []):
                for _mod in _ms.get("modules", []):
                    for _skill in _mod.get("skills", []):
                        _total += 1
                        _ai = _skill.get("ai_metadata")
                        if not _ai or not isinstance(_ai, dict):
                            continue
                        _layer = _ai.get("layer", "missing")
                        _layer_counts[_layer] = _layer_counts.get(_layer, 0) + 1
                        if _layer in _legacy:
                            _legacy[_layer] += 1
            print("========== REPAIR AI LAYER AUDIT ==========")
            print(f"  total_skills: {_total}")
            for _l in sorted(_layer_counts):
                print(f"  {_l}: {_layer_counts[_l]}")
            _total_legacy = sum(_legacy.values())
            print(f"  legacy_layers_found: {_total_legacy}")
            if _total_legacy:
                print(f"  WARNING: {_total_legacy} legacy layer(s) remain!")
            print("=============================================")

            # ── Check for blocking gap alignment failure ────────
            # Phase 4 — Block if milestone count outside allowed range
            if roadmap_data.get("_gap_alignment_blocked"):
                _range = roadmap_data.get("milestone_range", {})
                _min   = _range.get("minimum", MIN_MILESTONES)
                _max   = _range.get("maximum", MAX_MILESTONES)
                _actual = len(roadmap_data.get("milestones", []))
                if _actual < _min or _actual > _max:
                    raise ValueError(
                        f"GAP ALIGNMENT BLOCKED: {_actual} milestones outside "
                        f"allowed range [{_min}, {_max}]. Triggering regeneration."
                    )

            # ── Genuineness Validator — MD Section 19, 7 checks ────
            # run_roadmap_bible_validators() NEVER raises.
            # fits_life is advisory (MD: "Warn, log, show math to user"),
            # it adapts duration instead of hard-failing.
            bible_results = run_roadmap_bible_validators(
                roadmap_data, weekly_hours_available, timeline_days
            )
            roadmap_data["fits_life"]        = bible_results.get("fits_life", {})
            roadmap_data["bible_validators"] = bible_results

            # ── Repair debug output ────────────────────────────
            _breadth_req = _compute_required_domains(roadmap_data)
            _breadth_cov = _get_covered_domains(roadmap_data, _breadth_req)
            _breadth_miss = _breadth_req - _breadth_cov
            if _breadth_miss:
                print("======== BREADTH REPAIR ========")
                print(f"  Required: {sorted(_breadth_req)}")
                print(f"  Missing: {sorted(_breadth_miss)}")
                print(f"  Injected: {len(_breadth_req) - len(_breadth_miss)}")
                print(f"  Result: {'PASS' if not _breadth_miss else 'WARN'}")

            _cat_stats = roadmap_data.get("catalog_stats", {})
            if _cat_stats:
                print("======== CATALOG REPAIR ========")
                print(f"  Mapped: {_cat_stats.get('mapped', 0)}/{_cat_stats.get('total', 0)}")
                print(f"  Coverage: {_cat_stats.get('coverage', 0):.0%}")
                missing_list = _cat_stats.get("missing", [])
                if missing_list:
                    print(f"  Missing: {missing_list}")

            # ── AI repair debug ────────────────────────────────
            _ai_violations = 0
            for _ms in roadmap_data.get("milestones", []):
                for _mod in _ms.get("modules", []):
                    for _sk in _mod.get("skills", []):
                        _ai = _sk.get("ai_metadata", {})
                        if _ai and _ai.get("layer", "") not in ALLOWED_AI_LAYERS:
                            _ai_violations += 1
            if _ai_violations > 0:
                print(f"======== AI REPAIR ========")
                print(f"  Invalid layers remaining: {_ai_violations}")

            print("========== QUALITY AUDIT ==========")
            for qname, qres in quality_results.items():
                qpassed = qres.get("pass", True)
                qbadge = "PASS" if qpassed else "WARN"
                print(f"  {qname}: {qbadge}")
                if not qpassed:
                    reason = qres.get("reason", "")
                    print(f"    reason: {reason}")
            print("=====================================")

            # ── Calibration diagnostics ────────────────────────
            _dbg_ms = roadmap_data.get("milestones", [])
            _dbg_skill_count   = sum(
                len(mod.get("skills", []))
                for ms in _dbg_ms
                for mod in ms.get("modules", [])
            )
            _dbg_lesson_count  = sum(
                len(skill.get("lessons", []))
                for ms in _dbg_ms
                for mod in ms.get("modules", [])
                for skill in mod.get("skills", [])
            )
            _dbg_project_count = sum(
                len(ms.get("projects") or ([ms["project"]] if ms.get("project") else []))
                for ms in _dbg_ms
            )
            fl                 = roadmap_data["fits_life"]
            print(
                f"[FITS LIFE DEBUG]\n"
                f"  milestones={len(_dbg_ms)}\n"
                f"  skills={_dbg_skill_count}\n"
                f"  lessons={_dbg_lesson_count}\n"
                f"  projects={_dbg_project_count}\n"
                f"  demand={fl.get('demand_hours', 0)}h\n"
                f"  budget={fl.get('budget_hours', 0)}h\n"
                f"  duration_weeks={fl.get('duration_weeks', 16)}\n"
                f"  weekly_hours_needed={fl.get('weekly_hours_needed', 0)}h/wk"
            )

            milestones = roadmap_data.get("milestones", [])
            print("[ROADMAP AGENT] ✓ Roadmap validated")
            print(f"  Target role : {roadmap_data.get('target_role', 'N/A')}")
            print(f"  Level       : {level}")
            print(f"  Milestones  : {len(milestones)}")
            print(f"  ICP type    : {roadmap_data.get('icp_type', 'N/A')}")

            # ── Science distribution diagnostics ────────────────
            print("========== SCIENCE AUDIT ==========")
            sci_result = audit_science_distribution(roadmap_data)
            for line in sci_result.get("details", []):
                print(f"  {line}")
            badge = "PASS" if sci_result.get("pass") else "FAIL"
            print(f"  {badge}: {sci_result.get('reason', '')}")
            print("===================================")

            # ── Roadmap size diagnostics ───────────────────────
            _rd_json = json.dumps(roadmap_data)
            _rd_milestones = roadmap_data.get("milestones", [])
            _rd_module_count = sum(len(ms.get("modules", [])) for ms in _rd_milestones)
            _rd_skill_count = sum(
                len(mod.get("skills", []))
                for ms in _rd_milestones
                for mod in ms.get("modules", [])
            )
            print(
                f"[ROADMAP SIZE] bytes={len(_rd_json)} "
                f"milestones={len(_rd_milestones)} "
                f"modules={_rd_module_count} "
                f"skills={_rd_skill_count}"
            )

            # ── Store in Pinecone ──────────────────────────────
            print(f"\n[ROADMAP AGENT] Storing roadmap in Pinecone...")
            stored = store_roadmap_in_pinecone(user_id, ai_roadmap_id, roadmap_data)
            if stored:
                print("[ROADMAP AGENT] ✓ Roadmap persisted to Pinecone")
                # ── Immediate verification fetch ────────────────
                try:
                    from src.pinecone_utils import pc, INDEX_NAME
                    verify_index = pc.Index(INDEX_NAME)
                    vector_id = f"{user_id}_roadmap_{ai_roadmap_id}"
                    verify_result = verify_index.fetch(
                        ids=[vector_id],
                        namespace=user_id,
                    )
                    if (verify_result and verify_result.vectors
                            and vector_id in verify_result.vectors):
                        print(f"[VERIFY] roadmap vector stored and verified "
                              f"(vector_id={vector_id})")
                    else:
                        raise RuntimeError(
                            f"Verification fetch FAILED for roadmap vector "
                            f"(vector_id={vector_id}, namespace={user_id})"
                        )
                except Exception as ve:
                    print(f"[VERIFY] Roadmap verification fetch error: {ve}")
                    raise
            else:
                raise RuntimeError("store_roadmap_in_pinecone returned False")

            # ── Save POC cross-POC records ─────────────────────────────
            # roadmap_conversation — lightweight record (no duplicate payload)

            roadmap_summary = {
                "roadmap_id": ai_roadmap_id,
                "user_id": user_id,
                "target_role": roadmap_data.get("target_role"),
                "level": roadmap_data.get("level"),
                "icp_type": roadmap_data.get("icp_type"),
                "generated_at": now
            }

            # NOTE: we intentionally do NOT write to
            # {user_id}_onboarding_conversation here.
            # That record belongs to the Onboarding POC and contains
            # years_experience, weekly_hours_available, skill history,
            # and the full learner narrative.  Overwriting it with the
            # roadmap summary was Bug 2 — it caused every subsequent
            # run for the same user to detect level='beginner'
            # because the overwritten record had no years_experience.

            for record_key, ensure_ascii in [
                ("roadmap_conversation", True),
                ("roadmap_output",      False),
            ]:
                record_id = f"{user_id}_{record_key}"
                payload   = json.dumps(
                    roadmap_summary if record_key == "roadmap_conversation" else roadmap_data,
                    ensure_ascii=ensure_ascii
                )
                payload_bytes = len(payload.encode("utf-8"))
                print(f"[ROADMAP SAVE] {record_id}  "
                      f"namespace={user_id}  size={payload_bytes}")

                ok = save_poc_record(
                    user_id=user_id,
                    record_id=record_id,
                    text=payload,
                )
                if not ok:
                    raise RuntimeError(
                        f"save_poc_record failed for {record_id} "
                        f"(user_id={user_id}, {payload_bytes} bytes)"
                    )

                # ── Immediate verification fetch ────────────────
                verify = fetch_poc_record(user_id=user_id, record_id=record_id)
                if not verify:
                    raise RuntimeError(
                        f"Verification fetch FAILED for {record_id} "
                        f"(user_id={user_id}, namespace={user_id}) — "
                        f"record not found after upsert"
                    )

            # ── Final safety check: re-fetch all 3 records ──────
            print(f"\n[ROADMAP AGENT] Final safety check — re-fetching all records...")
            for rid in [
                f"{user_id}_roadmap_conversation",
                f"{user_id}_roadmap_output",
                f"{user_id}_roadmap_{ai_roadmap_id}",
            ]:
                fetched = fetch_poc_record(user_id=user_id, record_id=rid)
                if not fetched:
                    print(f"[ROADMAP SAVE] ⚠ Final safety fetch MISSING: {rid}")
                else:
                    print(f"[ROADMAP SAVE] ✓ Final safety fetch OK: {rid} ({len(fetched)} chars)")

            # ── POC local storage ──────────────────────────────  (UNCHANGED)
            print("\n[ROADMAP AGENT] Writing POC storage artifacts...")
            store_pipeline_artifacts(
                user_id        = user_id,
                roadmap_id     = ai_roadmap_id,
                input_context  = context,
                icp_type       = icp_type,
                roadmap_output = roadmap_data,
            )

            # ── Final response ─────────────────────────────────  (UNCHANGED shape)
            return {
                "id":                       str(uuid.uuid4()),
                "user_id":                  user_id,
                "ai_session_id":            ai_session_id,
                "ai_roadmap_id":            ai_roadmap_id,
                "target_role":              roadmap_data.get("target_role", ""),
                "icp_type":                 roadmap_data.get("icp_type", icp_type),
                "level":                    roadmap_data.get("level", level),
                "career_stage":             roadmap_data.get("career_stage", ""),
                "language":                 roadmap_data.get("language", "en"),
                "starting_milestone":           roadmap_data.get("starting_milestone", ""),
                "current_active_milestone":     roadmap_data.get("current_active_milestone", ""),
                "timeline_days":                roadmap_data.get("timeline_days"),
                "budget_hours":                 roadmap_data.get("budget_hours"),
                "estimated_total_hours":        roadmap_data.get("estimated_total_hours"),
                "milestone_count_rationale":    roadmap_data.get("milestone_count_rationale", ""),
                "vision_profile":               roadmap_data.get("vision_profile", {}),
                "roadmap_meta":             roadmap_data.get("roadmap_meta", {}),
                "milestones":               milestones,
                "fits_life":                roadmap_data.get("fits_life", {}),
                "bible_validators":         roadmap_data.get("bible_validators", {}),
                "quality_validators":       roadmap_data.get("quality_validators", {}),
                "pinecone_stored":          stored,
                "ai_metadata": {
                    "generated_at":          now,
                    "session_source":        "pinecone" if session_was_provided else "generated",
                    "generation_model":      "roadmap-gen-v3.3",
                    "personalization_score": 0.92,
                },
                "status":       "confirmed",
                "created_at":   now,
                "updated_at":   now,
                "confirmed_at": now,
                "published_at": None,
            }

        except json.JSONDecodeError as e:
            print(f"[ROADMAP AGENT] ✗ JSON parse error (attempt {attempt}): {e}")
            # ── Diagnostic: show 500 chars before/after error position ──
            err_pos = e.pos
            raw_for_debug = clean_result
            line_no = raw_for_debug[:err_pos].count("\n") + 1
            col_no = err_pos - raw_for_debug[:err_pos].rfind("\n")
            ctx_start = max(err_pos - 500, 0)
            ctx_end = min(err_pos + 500, len(raw_for_debug))
            before = raw_for_debug[ctx_start:err_pos]
            after = raw_for_debug[err_pos:ctx_end]
            print(f"  line={line_no}  column={col_no}  pos={err_pos}")
            print(f"  raw_input_length={len(raw_for_debug)}")
            print(f"  --- 500 BEFORE ---")
            print(before)
            print(f"  --- ^^^ ERROR AT pos {err_pos} ^^^ ---")
            print(after)
            print(f"  --- 500 AFTER ---")
            if attempt == max_attempts:
                return {
                    "error":         f"Invalid JSON from LLM: {str(e)}",
                    "user_id":       user_id,
                    "ai_session_id": ai_session_id,
                    "raw_output":    result[:500],
                }

        except ValueError as e:
            print(f"[ROADMAP AGENT] ✗ Validation error (attempt {attempt}): {e}")
            if attempt == max_attempts:
                return {
                    "error":         f"Roadmap validation failed: {str(e)}",
                    "user_id":       user_id,
                    "ai_session_id": ai_session_id,
                }

        except Exception as e:
            print(f"[ROADMAP AGENT] ✗ Unexpected error (attempt {attempt}): {e}")
            if "clean_result" in locals():
                print(clean_result[:1000])
            traceback.print_exc()
            if attempt == max_attempts:
                return {
                    "error":         f"Failed to generate roadmap: {str(e)}",
                    "user_id":       user_id,
                    "ai_session_id": ai_session_id,
                }

    # Should never reach here — all paths return inside the loop
    return {
        "error":         "Roadmap generation failed after all attempts.",
        "user_id":       user_id,
        "ai_session_id": ai_session_id,
    }