import streamlit as st
import json
import sys
import os
import traceback
import re

# =====================================================
# PATH SETUP
# =====================================================
CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, CURRENT_DIR)

from src.config import log_config_status, GEMINI_API_KEY, GOOGLE_API_KEY
log_config_status()

from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.prompts import PromptTemplate
from langchain_core.output_parsers import StrOutputParser
from src.roadmap_agent import run_pipeline
from src.pinecone_utils import retrieve_raw_context

# =====================================================
# SPEC-LOCKED MILESTONE SALARY DATA (M01–M07 keyed)
# =====================================================

ICP_A_MARKET = {
    "M01": {"range": "Unpaid / Stipend",  "tier": "intern"},
    "M02": {"range": "₹3–5 LPA",          "tier": "fresher"},
    "M03": {"range": "₹6–10 LPA",         "tier": "junior"},
    "M04": {"range": "₹10–16 LPA",        "tier": "mid"},
    "M05": {"range": "₹16–24 LPA",        "tier": "senior"},
    "M06": {"range": "₹24–35 LPA",        "tier": "staff"},
    "M07": {"range": "₹35L+ LPA",         "tier": "principal"},
}

ICP_B_MARKET = {
    "M01": {"range": "₹12–20K/mo",  "tier": "fresher"},
    "M02": {"range": "₹20–35K/mo",  "tier": "junior"},
    "M03": {"range": "₹35–55K/mo",  "tier": "mid"},
    "M04": {"range": "₹55–80K/mo",  "tier": "senior"},
    "M05": {"range": "₹80K–1.2L/mo","tier": "staff"},
    "M06": {"range": "₹1.2–1.8L/mo","tier": "lead"},
    "M07": {"range": "₹1.8L+/mo",   "tier": "principal"},
}

ROLE_SALARY_OVERLAY = {
    "software engineer":     (("4–8 LPA"), ("6–12 LPA"), ("10–20 LPA"), ("18–30 LPA"), ("28–45 LPA"), ("42–60 LPA"), ("60L+ LPA")),
    "data scientist":        (("6–10 LPA"), ("10–18 LPA"), ("18–35 LPA"), ("30–50 LPA"), ("45–70 LPA"), ("65–90 LPA"), ("90L+ LPA")),
    "ai/ml engineer":        (("8–14 LPA"), ("14–25 LPA"), ("25–50 LPA"), ("45–70 LPA"), ("65–95 LPA"), ("90L+ LPA"), ("1Cr+ LPA")),
    "machine learning":      (("8–14 LPA"), ("14–25 LPA"), ("25–50 LPA"), ("45–70 LPA"), ("65–95 LPA"), ("90L+ LPA"), ("1Cr+ LPA")),
    "backend developer":     (("5–9 LPA"),  ("8–15 LPA"),  ("15–30 LPA"), ("28–45 LPA"), ("42–60 LPA"), ("58–80 LPA"), ("80L+ LPA")),
    "frontend developer":    (("4–8 LPA"),  ("7–13 LPA"),  ("12–22 LPA"), ("20–35 LPA"), ("32–50 LPA"), ("48–70 LPA"), ("70L+ LPA")),
    "full stack developer":  (("5–10 LPA"), ("9–18 LPA"),  ("18–35 LPA"), ("32–50 LPA"), ("48–70 LPA"), ("68–90 LPA"), ("90L+ LPA")),
    "devops engineer":       (("6–10 LPA"), ("10–20 LPA"), ("20–40 LPA"), ("38–55 LPA"), ("52–75 LPA"), ("72–95 LPA"), ("95L+ LPA")),
    "cloud engineer":        (("6–10 LPA"), ("12–22 LPA"), ("22–45 LPA"), ("42–60 LPA"), ("58–80 LPA"), ("78–1Cr LPA"), ("1Cr+ LPA")),
    "data analyst":          (("4–8 LPA"),  ("7–14 LPA"),  ("14–25 LPA"), ("22–38 LPA"), ("36–55 LPA"), ("52–75 LPA"), ("75L+ LPA")),
    "product manager":       (("8–14 LPA"), ("14–25 LPA"), ("25–50 LPA"), ("45–70 LPA"), ("65–90 LPA"), ("85L+ LPA"), ("1Cr+ LPA")),
    "android developer":     (("4–9 LPA"),  ("8–16 LPA"),  ("16–30 LPA"), ("28–45 LPA"), ("42–60 LPA"), ("58–80 LPA"), ("80L+ LPA")),
    "ios developer":         (("5–10 LPA"), ("9–18 LPA"),  ("18–35 LPA"), ("32–50 LPA"), ("48–70 LPA"), ("68–90 LPA"), ("90L+ LPA")),
    "cybersecurity":         (("6–10 LPA"), ("10–20 LPA"), ("20–40 LPA"), ("38–55 LPA"), ("52–75 LPA"), ("72–95 LPA"), ("95L+ LPA")),
    "ui/ux designer":        (("4–8 LPA"),  ("7–14 LPA"),  ("14–28 LPA"), ("26–42 LPA"), ("40–60 LPA"), ("58–80 LPA"), ("80L+ LPA")),
    "business analyst":      (("5–9 LPA"),  ("8–16 LPA"),  ("16–30 LPA"), ("28–45 LPA"), ("42–60 LPA"), ("58–80 LPA"), ("80L+ LPA")),
}

ROLE_SALARY_OVERLAY_MONTHLY = {
    "data analyst":          ("15–22K/mo",  "22–35K/mo",  "35–60K/mo",  "58–80K/mo",  "78K–1L/mo",  "1–1.4L/mo",  "1.4L+/mo"),
    "data entry":            ("10–15K/mo",  "14–22K/mo",  "22–35K/mo",  "33–50K/mo",  "48–65K/mo",  "62–85K/mo",  "85K+/mo"),
    "software developer":    ("18–28K/mo",  "28–45K/mo",  "45–80K/mo",  "78K–1.2L/mo","1.2–1.6L/mo","1.6–2L/mo",  "2L+/mo"),
    "web developer":         ("15–25K/mo",  "25–40K/mo",  "40–70K/mo",  "68K–1L/mo",  "98K–1.4L/mo","1.4–1.8L/mo","1.8L+/mo"),
    "python developer":      ("18–28K/mo",  "28–45K/mo",  "45–80K/mo",  "78K–1.2L/mo","1.2–1.6L/mo","1.6–2L/mo",  "2L+/mo"),
    "it support":            ("12–18K/mo",  "18–28K/mo",  "28–45K/mo",  "43–62K/mo",  "60–80K/mo",  "78K–1L/mo",  "1L+/mo"),
    "network engineer":      ("15–22K/mo",  "22–35K/mo",  "35–58K/mo",  "56–78K/mo",  "75K–1L/mo",  "98K–1.3L/mo","1.3L+/mo"),
    "ui ux":                 ("15–22K/mo",  "22–35K/mo",  "35–60K/mo",  "58–82K/mo",  "80K–1.1L/mo","1.1–1.5L/mo","1.5L+/mo"),
    "graphic designer":      ("12–18K/mo",  "18–28K/mo",  "28–45K/mo",  "43–62K/mo",  "60–82K/mo",  "80K–1L/mo",  "1L+/mo"),
    "digital marketing":     ("12–20K/mo",  "20–32K/mo",  "32–55K/mo",  "52–75K/mo",  "72K–1L/mo",  "98K–1.3L/mo","1.3L+/mo"),
    "content writer":        ("10–16K/mo",  "16–25K/mo",  "25–42K/mo",  "40–60K/mo",  "58–80K/mo",  "78K–1L/mo",  "1L+/mo"),
    "social media":          ("12–18K/mo",  "18–28K/mo",  "28–45K/mo",  "43–62K/mo",  "60–82K/mo",  "80K–1L/mo",  "1L+/mo"),
    "excel analyst":         ("12–18K/mo",  "18–28K/mo",  "28–45K/mo",  "43–62K/mo",  "60–80K/mo",  "78K–1L/mo",  "1L+/mo"),
    "back office":           ("10–16K/mo",  "16–25K/mo",  "25–40K/mo",  "38–55K/mo",  "52–70K/mo",  "68–88K/mo",  "88K+/mo"),
    "customer support":      ("12–18K/mo",  "18–28K/mo",  "28–45K/mo",  "43–60K/mo",  "58–78K/mo",  "75K–1L/mo",  "1L+/mo"),
    "accounts assistant":    ("12–18K/mo",  "18–28K/mo",  "28–45K/mo",  "43–62K/mo",  "60–80K/mo",  "78K–1L/mo",  "1L+/mo"),
    "hr assistant":          ("12–18K/mo",  "18–28K/mo",  "28–45K/mo",  "43–62K/mo",  "60–80K/mo",  "78K–1L/mo",  "1L+/mo"),
    "sales executive":       ("12–20K/mo",  "20–35K/mo",  "35–60K/mo",  "58–82K/mo",  "80K–1.1L/mo","1.1–1.5L/mo","1.5L+/mo"),
}

TIER_ORDER_M = ["M01", "M02", "M03", "M04", "M05", "M06", "M07"]


def get_milestone_label(ms: dict, icp_val: str, seq: int) -> str:
    """Always return M01-M07 style label."""
    label = ms.get("label", ms.get("milestone_label", "")).strip()
    if label and label.upper().startswith("M") and label[1:].isdigit():
        return label.upper()
    return f"M{seq:02d}"


def get_salary_for_milestone(ms: dict, ms_label: str, icp_val: str, target_role: str = "") -> dict:
    """
    Priority: 1) ms["sal"] field (LLM-generated), 2) role overlay, 3) market dict fallback.
    """
    sal_from_ms = ms.get("sal", "").strip()
    if sal_from_ms:
        return {
            "display": sal_from_ms if sal_from_ms.startswith("₹") else f"₹{sal_from_ms}",
            "source":  "Naukri / Indeed 2026",
            "tier":    ms.get("tier", ""),
            "matched": True,
        }

    is_low   = icp_val in ("low", "low_wage")
    try:
        tier_idx = TIER_ORDER_M.index(ms_label)
    except ValueError:
        tier_idx = 0

    if target_role:
        role_key   = target_role.lower().strip()
        overlay_db = ROLE_SALARY_OVERLAY_MONTHLY if is_low else ROLE_SALARY_OVERLAY
        for key in overlay_db:
            if key in role_key or any(part in role_key for part in key.split()):
                tup = overlay_db[key]
                role_range = tup[tier_idx] if tier_idx < len(tup) else tup[-1]
                return {
                    "display": f"₹{role_range}",
                    "source":  "Naukri / Indeed 2026",
                    "tier":    "",
                    "matched": True,
                }

    base = (ICP_B_MARKET if is_low else ICP_A_MARKET).get(
        ms_label, (ICP_B_MARKET if is_low else ICP_A_MARKET)["M01"]
    )
    return {
        "display": base["range"],
        "source":  "Naukri / Indeed 2026",
        "tier":    base["tier"],
        "matched": False,
    }


def skill_state_color(state: str) -> tuple:
    return {
        "locked":      ("#1a1020", "#3d1f5c", "🔒 Locked"),
        "unlocked":    ("#0d1f2d", "#1a4a6b", "🔓 Available"),
        "in_progress": ("#1a1f0d", "#3d5c1f", "⚡ In Progress"),
        "mock_ready":  ("#1f1a0d", "#5c4a1f", "📝 Mock Ready"),
        "mastered":    ("#0d2d1a", "#1f6b4a", "✅ Mastered"),
        "review_due":  ("#2d1a0d", "#6b3a1f", "🔁 Review Due"),
    }.get(state, ("#1a1a26", "#2e2e45", "⭕ Unknown"))


def milestone_color(seq: int) -> tuple:
    palettes = [
        ("#7c3aed", "#a78bfa"),
        ("#2563eb", "#60a5fa"),
        ("#0891b2", "#22d3ee"),
        ("#059669", "#34d399"),
        ("#d97706", "#fbbf24"),
        ("#dc2626", "#f87171"),
        ("#db2777", "#f472b6"),
    ]
    return palettes[(seq - 1) % len(palettes)]


def milestone_emoji(seq: int) -> str:
    return ["🌱", "📘", "⚙️", "🚀", "🏆", "⭐", "🎓"][(seq - 1) % 7]


# =====================================================
# HELPER: Lesson Extraction
# =====================================================

def extract_module_lessons(module):
    """
    Return all lesson strings inside a module regardless of
    where lessons are stored.

    Search order:
      A) module["lessons"]
      B) skill["lessons"] for every skill
      C) skill["content_flow"]["video"]["lessons"] if present
    """
    result = []
    # A — module-level lessons
    raw = module.get("lessons")
    if isinstance(raw, list):
        result.extend(raw)
    # B — skill-level lessons
    for skill in module.get("skills", []):
        raw = skill.get("lessons")
        if isinstance(raw, list):
            result.extend(raw)
    # C — content_flow video lessons
    for skill in module.get("skills", []):
        flow = skill.get("content_flow", {})
        video = flow.get("video", {})
        raw = video.get("lessons")
        if isinstance(raw, list):
            result.extend(raw)
    return result


# =====================================================
# PAGE CONFIG
# =====================================================
st.set_page_config(
    page_title="Vidya V3 — Roadmap Lab",
    page_icon="🎯",
    layout="wide",
    initial_sidebar_state="expanded"
)

# =====================================================
# CSS
# =====================================================
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700;800&display=swap');

*, *::before, *::after { box-sizing: border-box; }

html, body, [data-testid="stAppViewContainer"] {
    background: #08080f !important;
    font-family: 'Inter', sans-serif;
}
[data-testid="stSidebar"] {
    background: #0d0d18 !important;
    border-right: 1px solid #1e1e30;
}

.hero {
    background: linear-gradient(135deg, #0f0c29, #302b63, #24243e);
    border-radius: 20px;
    padding: 36px 40px;
    margin-bottom: 32px;
    position: relative;
    overflow: hidden;
}
.hero::after {
    content: '';
    position: absolute;
    top: -60px; right: -60px;
    width: 240px; height: 240px;
    background: radial-gradient(circle, rgba(167,139,250,0.18) 0%, transparent 70%);
    border-radius: 50%;
}
.hero h1 { font-size: 32px; font-weight: 800; color: #fff; margin: 0 0 8px 0; }
.hero p  { font-size: 15px; color: #a0a0c8; margin: 0; }

.ms-card {
    border-radius: 18px;
    padding: 22px 26px 20px 26px;
    margin-bottom: 22px;
    position: relative;
    overflow: hidden;
    transition: transform .15s;
}
.ms-card:hover { transform: translateY(-2px); }

.ms-label-badge {
    display: inline-flex;
    align-items: center; justify-content: center;
    min-width: 48px; height: 48px;
    border-radius: 12px;
    font-weight: 800; font-size: 15px;
    color: #fff;
    flex-shrink: 0;
    margin-right: 14px;
    padding: 0 10px;
    letter-spacing: 0.04em;
}

.salary-block {
    background: #0a0a16;
    border-radius: 12px;
    padding: 12px 16px;
    margin-top: 14px;
    margin-left: 62px;
    border: 1px solid #1e1e32;
}
.salary-source {
    font-size: 10px;
    font-weight: 600;
    text-transform: uppercase;
    letter-spacing: 0.09em;
    color: #4a4a6a;
    margin-bottom: 6px;
}
.salary-main {
    font-size: 20px;
    font-weight: 800;
    letter-spacing: 0.01em;
    margin-bottom: 2px;
}

.state-pill {
    display: inline-block;
    padding: 3px 10px;
    border-radius: 20px;
    font-size: 11px;
    font-weight: 700;
    letter-spacing: 0.04em;
}

.skill-card {
    border-radius: 14px;
    padding: 18px 20px;
    margin-bottom: 12px;
    border: 1px solid;
    transition: transform .12s;
}
.skill-card:hover { transform: translateY(-1px); }

.skill-title {
    font-size: 15px;
    font-weight: 700;
    color: #e0deff;
    margin-bottom: 4px;
}
.skill-desc {
    font-size: 13px;
    color: #7070a0;
    line-height: 1.5;
    margin-bottom: 12px;
}

.content-tags {
    display: flex;
    flex-wrap: wrap;
    gap: 8px;
    margin-top: 10px;
}
.ctag {
    display: inline-flex;
    align-items: center;
    gap: 4px;
    padding: 5px 12px;
    border-radius: 20px;
    font-size: 12px;
    font-weight: 600;
    background: #1a1a2e;
    color: #a0a0cc;
    border: 1px solid #2a2a42;
}

.stat-grid {
    display: grid;
    grid-template-columns: repeat(5, 1fr);
    gap: 12px;
    margin: 16px 0;
}
.stat-box {
    background: #0f0f1c;
    border: 1px solid #1e1e32;
    border-radius: 12px;
    padding: 14px;
    text-align: center;
}
.stat-box .num { font-size: 26px; font-weight: 800; }
.stat-box .lbl { font-size: 11px; color: #6b7280; margin-top: 3px; }

.icp-badge {
    display: inline-block;
    padding: 4px 12px;
    border-radius: 20px;
    font-size: 11px;
    font-weight: 700;
    text-transform: uppercase;
    letter-spacing: 0.07em;
}

.level-badge {
    display: inline-block;
    padding: 4px 12px;
    border-radius: 20px;
    font-size: 11px;
    font-weight: 700;
    text-transform: uppercase;
    letter-spacing: 0.07em;
    margin-left: 8px;
}

.v-card {
    background: #0f0f1c;
    border: 1px solid #1e1e32;
    border-radius: 14px;
    padding: 16px;
}
.v-label {
    font-size: 10px; font-weight: 700; text-transform: uppercase;
    letter-spacing: 0.09em; color: #6b7280; margin-bottom: 6px;
}
.v-text { font-size: 14px; color: #d0cff0; line-height: 1.55; }

.mastery-bar-wrap {
    background: #0f0f1c;
    border-radius: 8px;
    height: 6px;
    margin: 10px 0 6px 0;
    overflow: hidden;
}
.mastery-bar-fill {
    height: 100%;
    border-radius: 8px;
    background: linear-gradient(90deg, #7c3aed, #a78bfa);
}

.sec-title {
    font-size: 20px; font-weight: 800; color: #f0edff;
    margin: 28px 0 14px 0;
    padding-left: 12px;
    border-left: 4px solid #7c3aed;
}

[data-testid="stTabs"] button { color: #a0a0c0 !important; }
[data-testid="stTabs"] button[aria-selected="true"] {
    color: #a78bfa !important;
    border-bottom-color: #a78bfa !important;
}

.module-header {
    font-size: 16px;
    font-weight: 700;
    color: #c0beff;
    margin-bottom: 4px;
}
.module-desc {
    font-size: 13px;
    color: #5a5a7a;
    margin-bottom: 16px;
}

.skill-status-row {
    display: flex;
    align-items: center;
    justify-content: space-between;
    flex-wrap: wrap;
    gap: 8px;
    margin-bottom: 8px;
}

.lesson-list {
    list-style: none;
    padding: 0;
    margin: 0 0 14px 0;
}
.lesson-item {
    display: flex;
    align-items: center;
    gap: 8px;
    padding: 6px 10px;
    border-radius: 8px;
    font-size: 13px;
    color: #c0c0e0;
    background: #0c0c1a;
    margin-bottom: 6px;
    border: 1px solid #1a1a2e;
}

.science-badge {
    display: inline-flex;
    align-items: center;
    gap: 6px;
    padding: 6px 14px;
    border-radius: 20px;
    font-size: 12px;
    font-weight: 700;
    margin: 4px 4px 4px 0;
}
.science-scenario {
    background: #1a1030;
    color: #c084fc;
    border: 1px solid #6d28d9;
}
.science-interview {
    background: #0d2010;
    color: #4ade80;
    border: 1px solid #166534;
}

.mod-chip {
    display: inline-flex;
    align-items: center;
    gap: 4px;
    padding: 3px 10px;
    border-radius: 20px;
    font-size: 11px;
    font-weight: 600;
    background: #12122a;
    color: #8888cc;
    border: 1px solid #22224a;
    margin-right: 6px;
}
.mod-chip-free {
    background: #0d2010;
    color: #4ade80;
    border: 1px solid #166534;
}
</style>
""", unsafe_allow_html=True)

# =====================================================
# SESSION STATE
# =====================================================
for key, default in [
    ("generated_onboarding", ""),
    ("generated_onboarding_json", {}),
    ("generated_context", ""),
    ("roadmap_data", None),
]:
    if key not in st.session_state:
        st.session_state[key] = default

# =====================================================
# LLM
# =====================================================
gemini_key = GEMINI_API_KEY or GOOGLE_API_KEY
if not gemini_key:
    st.error("❌ Missing GEMINI_API_KEY or GOOGLE_API_KEY in .env")
    st.stop()

llm = ChatGoogleGenerativeAI(
    model="gemini-2.5-flash",
    google_api_key=gemini_key,
    temperature=0.7,
    convert_system_message_to_human=True
)

# =====================================================
# PROMPTS
# =====================================================
persona_prompt = PromptTemplate(
    input_variables=["weekly_hours", "goal"],
    template="""
You are generating realistic onboarding summaries for Vidya V3.

Generate a believable learner onboarding summary based ONLY on the goal and weekly hours provided.

Infer from the goal whether this person is:
- A working professional (if goal mentions promotion, switch, upskilling, years of experience)
- A student or fresher (if goal mentions placement, first job, college, internship)

The onboarding summary MUST include:
- current background (inferred from goal)
- current struggles
- learning goals
- skill gaps
- confidence level
- career target
- urgency
- available weekly time

Tone: emotionally believable, realistic, India-specific, simple language (no jargon).

WEEKLY HOURS: {weekly_hours}
GOAL: {goal}

OUTPUT: Return ONLY the onboarding summary text. No headings, no JSON, no markdown.
"""
)

onboarding_json_prompt = PromptTemplate(
    input_variables=["weekly_hours", "goal", "persona_summary"],
    template="""
You are a data extraction engine for Vidya V3.

Given the learner persona summary below, extract and return a structured onboarding JSON object.

PERSONA SUMMARY:
{persona_summary}

ADDITIONAL INPUTS:
WEEKLY HOURS: {weekly_hours}
GOAL: {goal}

REQUIRED JSON FIELDS (spec-aligned — Vidya V3 onboarding signals only):
{{
  "full_name": "realistic name",
  "age": integer,
  "location": "Indian city",
  "education": "highest qualification",
  "current_role": "current job title or 'student'",
  "years_experience": integer,
  "current_salary_monthly": integer (0 if student / no job),
  "target_role": "specific role they want e.g. Data Scientist at a startup",
  "primary_goal": "one clear sentence what they want to achieve",
  "skill_gaps": ["most critical gap", "second gap", "third gap"],
  "known_skills": ["skill they already have 1", "skill 2"],
  "language_preference": "en | hi | ta",
  "weekly_hours_available": integer,
  "confidence_level": "low | medium | high",
  "urgency": "low | medium | high",
  "icp_type": "high if working professional with salary > 0, else low",
  "level": "beginner | intermediate | senior"
}}

RULES:
- Infer icp_type from the persona: if the person has a job and salary > 0, set icp_type = "high". Otherwise set icp_type = "low".
- Infer level: 0 years experience = beginner, 1-3 years = intermediate, 4+ years = senior.
- Return ONLY valid raw JSON. No markdown, no code fences, no explanation.
- All values must be realistic and India-specific.
"""
)

persona_chain         = persona_prompt | llm | StrOutputParser()
onboarding_json_chain = onboarding_json_prompt | llm | StrOutputParser()

# =====================================================
# SIDEBAR
# =====================================================
# st.sidebar.markdown("## ⚙️ Test Settings")
# weekly_hours = st.sidebar.slider("Weekly Hours", 2, 20, 5)
# st.sidebar.markdown("---")
# st.sidebar.caption("Vidya V3 · Roadmap Lab · Gemini + Pinecone + BKT")

# =====================================================
# HERO
# =====================================================
st.markdown("""
<div class="hero">
  <h1>🎯 Vidya V3 — Roadmap Lab</h1>
  <p>AI-powered personalized career roadmaps · Internal testing dashboard</p>
</div>
""", unsafe_allow_html=True)

# =====================================================
# MODE
# =====================================================
mode = st.radio(
    "Test mode", ["✨ AI-Generated Persona", "👤 Real User ID"],
    horizontal=True, label_visibility="collapsed"
)
mode = "Real" if "Real" in mode else "AI"

user_id = None
goal    = ""

# =====================================================
# REAL USER
# =====================================================
if mode == "Real":
    st.subheader("👤 Test with a Real User ID")
    user_id = st.text_input("User ID", placeholder="e.g. user_001")
    if user_id:
        if st.button("🔍 Fetch Pinecone Data"):
            with st.spinner("Fetching..."):
                raw = retrieve_raw_context(user_id)
            if raw:
                st.success("✅ Found")
                st.text_area("Raw context", value=raw, height=180)
            else:
                st.error("❌ No data found — is onboarding complete?")

# =====================================================
# AI PERSONA
# =====================================================
else:
    st.subheader("🤖 Generate a Test Persona")
    goal = st.text_input(
        "Learner's goal",
        placeholder="e.g. Become a backend developer in 6 months"
    )

    if st.button("✨ Generate Persona", type="primary"):
        if not goal.strip():
            st.error("Please enter a goal first.")
            st.stop()
        with st.spinner("Building learner profile..."):
            summary = persona_chain.invoke({
                "weekly_hours": weekly_hours, "goal": goal
            })
            st.session_state.generated_onboarding = summary

            raw_json = onboarding_json_chain.invoke({
                "weekly_hours": weekly_hours, "goal": goal,
                "persona_summary": summary
            })
            clean = re.sub(r"```json|```", "", raw_json).strip()
            try:
                st.session_state.generated_onboarding_json = json.loads(clean)
            except json.JSONDecodeError:
                st.session_state.generated_onboarding_json = {}
                st.warning("⚠ JSON parse failed — text summary only")
        st.success("✅ Persona ready!")

# =====================================================
# SHOW PERSONA
# =====================================================
if mode == "AI" and st.session_state.generated_onboarding:
    st.markdown('<div class="sec-title">📋 Learner Profile</div>', unsafe_allow_html=True)
    tab1, tab2 = st.tabs(["📖 Summary", "🗂 Structured Data"])

    with tab1:
        st.info(st.session_state.generated_onboarding)

    with tab2:
        oj = st.session_state.generated_onboarding_json
        if oj:
            col_a, col_b = st.columns(2)
            labels = {
                "full_name":              "👤 Name",
                "age":                    "🎂 Age",
                "location":               "📍 Location",
                "education":              "🎓 Education",
                "current_role":           "💼 Role",
                "years_experience":       "📅 Experience",
                "current_salary_monthly": "💰 Monthly Salary",
                "target_role":            "🎯 Target Role",
                "primary_goal":           "🚀 Goal",
                "weekly_hours_available": "⏰ Hours/wk",
                "language_preference":    "🗣 Language",
                "confidence_level":       "💪 Confidence",
                "urgency":                "⚡ Urgency",
                "icp_type":               "🏷 ICP",
                "level":                  "📊 Level",
            }
            SPEC_FIELDS = {
                "full_name", "age", "location", "education",
                "current_role", "years_experience", "current_salary_monthly",
                "target_role", "primary_goal", "language_preference",
                "weekly_hours_available", "confidence_level", "urgency",
                "icp_type", "level",
            }
            simple = [(k, v) for k, v in oj.items() if k in SPEC_FIELDS]
            for i, (k, v) in enumerate(simple):
                col  = col_a if i % 2 == 0 else col_b
                lbl  = labels.get(k, k.replace("_", " ").title())
                disp = f"₹{v:,}" if "salary" in k and isinstance(v, (int, float)) else str(v)
                col.metric(lbl, disp)

            st.markdown("---")
            cc, cd = st.columns(2)
            with cc:
                st.markdown("**🚧 Skill Gaps**")
                for g in oj.get("skill_gaps", []):
                    st.write(f"• {g}")
            with cd:
                st.markdown("**✅ Known Skills**")
                for s in oj.get("known_skills", []):
                    st.write(f"• {s}")
            with st.expander("🔧 Raw JSON"):
                st.json(oj)
        else:
            st.warning("Structured data not available.")

# =====================================================
# GENERATE ROADMAP
# =====================================================
st.markdown("---")
if st.button("🚀 Generate Personalized Roadmap", type="primary", use_container_width=True):
    try:
        if mode == "AI":
            if not st.session_state.generated_onboarding:
                st.error("Generate a persona first.")
                st.stop()
            oj       = st.session_state.generated_onboarding_json
            js       = json.dumps(oj, indent=2) if oj else "Not available"
            icp_type = oj.get("icp_type", "low")
            level    = oj.get("level", "beginner")
            context  = f"""USER PROFILE
ICP TYPE: {icp_type}
LEVEL: {level}
WEEKLY HOURS: {weekly_hours}

ONBOARDING SUMMARY:
{st.session_state.generated_onboarding}

STRUCTURED DATA:
{js}
"""
            roadmap_input = {
                "goal":                   goal or oj.get("primary_goal", ""),
                "current_role":           oj.get("current_role", "student"),
                "years_experience":       oj.get("years_experience", 0),
                "current_salary_annual":  oj.get("current_salary_monthly", 0) * 12,
                "current_salary_monthly": oj.get("current_salary_monthly", 0),
                "language_preference":    oj.get("language_preference", "en"),
                "weekly_hours_available": weekly_hours,
                "urgency":                oj.get("urgency", "high"),
                "self_efficacy":          oj.get("confidence_level", "low"),
                "level":                  level,
                "goal_context":           context,
            }
        else:
            if not user_id:
                st.error("Enter a User ID.")
                st.stop()
            with st.spinner("Fetching Pinecone context..."):
                raw = retrieve_raw_context(user_id)
            if not raw:
                st.error("❌ No data for this user.")
                st.stop()
            context       = raw
            roadmap_input = user_id

        st.session_state.generated_context = context
        with st.expander("🧠 Context sent to AI"):
            st.text_area("Context", value=context, height=200)

        with st.spinner("🤖 Generating roadmap — 30–60 sec..."):
            roadmap_data = run_pipeline(roadmap_input)
            st.session_state.roadmap_data = roadmap_data

        if "error" in roadmap_data:
            st.error(f"❌ {roadmap_data['error']}")
            st.stop()
        st.success("✅ Roadmap generated!")

    except Exception as e:
        st.error(f"❌ {e}")
        st.code(traceback.format_exc(), language="python")

# =====================================================
# RENDER ROADMAP
# =====================================================
if st.session_state.roadmap_data:
    rd = st.session_state.roadmap_data
    if "error" in rd:
        st.stop()

    milestones    = rd.get("milestones", [])
    icp_val       = rd.get("icp_type", "low")
    level_val     = rd.get("level", st.session_state.generated_onboarding_json.get("level", "beginner"))
    is_low        = icp_val in ("low", "low_wage")
    total_ms      = len(milestones)
    total_modules = sum(len(m.get("modules", [])) for m in milestones)
    total_skills  = sum(
        len(mod.get("skills", []))
        for m in milestones
        for mod in m.get("modules", [])
    )
    # Debug: inspect first module structure
    if milestones:
        mod = milestones[0]["modules"][0]
        print("LESSON DEBUG")
        print(json.dumps(mod, indent=2)[:5000])
        if mod.get("skills"):
            print("FIRST SKILL:")
            print(json.dumps(mod["skills"][0], indent=2)[:3000])
    total_lessons = sum(
        len(extract_module_lessons(mod))
        for m in milestones
        for mod in m.get("modules", [])
    )
    print(f"[LESSON AUDIT] Total lessons: {total_lessons}")

    target_role = (
        rd.get("target_role", "")
        or st.session_state.generated_onboarding_json.get("target_role", "")
    ).lower().strip()

    # ── ICP + Level badges ──
    st.markdown("---")
    icp_style = (
        "background:#1a1030;color:#a78bfa;border:1px solid #5a3aaa"
        if not is_low else
        "background:#0d2010;color:#4ade80;border:1px solid #2d6a4f"
    )
    icp_label = (
        "🏢 Professional Track"
        if not is_low else
        "🎓 Fresher / Student Track"
    )

    level_style_map = {
        "beginner":     "background:#0d1a2e;color:#60a5fa;border:1px solid #1e4a7a",
        "intermediate": "background:#1a1a0d;color:#fbbf24;border:1px solid #6a5a1e",
        "senior":       "background:#2d0d0d;color:#f87171;border:1px solid #7a1e1e",
    }
    level_style = level_style_map.get(level_val, level_style_map["beginner"])

    # ── v2.2 CHANGE 1: ICP-aware level labels ──
    # ICP-A (student): Beginner / Intermediate / Advanced
    # ICP-B (professional): Career Switcher / Intermediate / Senior
    level_label_map = {
        "a": {   # ICP-A: Student / Fresher (is_low=True)
            "beginner":     "🌱 Beginner",
            "intermediate": "⚡ Intermediate",
            "senior":       "🔥 Advanced",         # internship done ≠ industry "Senior"
        },
        "b": {   # ICP-B: Working Professional (is_low=False)
            "beginner":     "🌱 Career Switcher",  # switching domain, not starting life
            "intermediate": "⚡ Intermediate",
            "senior":       "🔥 Senior",
        },
    }
    icp_key    = "a" if is_low else "b"
    level_label = level_label_map[icp_key].get(level_val, "🌱 Beginner")

    # ── v2.2 CHANGE 2: ICP-aware ZPD context banner copy ──
    LEVEL_BANNER_TEXT = {
        "a": {   # ICP-A: Student / Fresher
            "beginner":     "No prior coding experience — onboarding confirms this from your diagnostic responses. You start from the very beginning.",
            "intermediate": "You've written code before — college projects, self-study, or tutorials. Foundations are skipped. You start where it gets real.",
            "senior":       "You've shipped something real — an internship, a live project, a hackathon win. You start at the level that matches what you've already proved.",
        },
        "b": {   # ICP-B: Working Professional
            "beginner":     "You're switching domains — your experience is real but in a different field. The roadmap starts at domain foundations, not career foundations.",
            "intermediate": "You have domain experience — 1 to 3 years doing this professionally. Foundations are auto-completed. You start where your ZPD actually is.",
            "senior":       "You're already senior. The roadmap skips everything you've earned. You start at the level that leads to principal, lead, or CTO track.",
        },
    }
    level_banner = LEVEL_BANNER_TEXT[icp_key].get(level_val, "")

    st.markdown(
        f'<span class="icp-badge" style="{icp_style}">{icp_label}</span>'
        f'<span class="level-badge" style="{level_style}">{level_label}</span>',
        unsafe_allow_html=True
    )

    # Render the ZPD context banner under the badges
    if level_banner:
        st.markdown(
            f'<div style="background:#0f0f1c;border:1px solid #1e1e32;border-radius:12px;'
            f'padding:12px 16px;margin:10px 0 4px 0;font-size:13px;color:#a0a0c8;'
            f'line-height:1.6;">{level_banner}</div>',
            unsafe_allow_html=True
        )

    # ── Stats strip (5 boxes) ──
    stat_colors = ["#a78bfa", "#60a5fa", "#34d399", "#fbbf24", "#f472b6"]
    st.markdown(f"""
    <div class="stat-grid">
      <div class="stat-box">
        <div class="num" style="color:{stat_colors[0]}">{total_ms}</div>
        <div class="lbl">Milestones in your path</div>
      </div>
      <div class="stat-box">
        <div class="num" style="color:{stat_colors[1]}">{total_modules}</div>
        <div class="lbl">Learning modules</div>
      </div>
      <div class="stat-box">
        <div class="num" style="color:{stat_colors[2]}">{total_skills}</div>
        <div class="lbl">Skills to master</div>
      </div>
      <div class="stat-box">
        <div class="num" style="color:{stat_colors[3]}">{total_lessons}</div>
        <div class="lbl">Total lessons</div>
      </div>
      <div class="stat-box">
        <div class="num" style="color:{stat_colors[4]}">{rd.get('language','en').upper()}</div>
        <div class="lbl">Language</div>
      </div>
    </div>
    """, unsafe_allow_html=True)

    # ── Vision profile ──
    oj_data         = st.session_state.generated_onboarding_json
    v_where         = rd.get("current_role") or oj_data.get("current_role") or "—"
    skill_gaps_list = oj_data.get("skill_gaps", [])
    v_blocker       = skill_gaps_list[0] if skill_gaps_list else rd.get("main_skill_gap", "—")
    v_efficacy      = rd.get("self_efficacy") or oj_data.get("confidence_level") or ""
    v_urgency       = rd.get("urgency") or oj_data.get("urgency") or ""
    drive_parts     = []
    if v_efficacy:
        drive_parts.append(f"Confidence: {v_efficacy.title()}")
    if v_urgency:
        drive_parts.append(f"Urgency: {v_urgency.title()}")
    v_drive = "  ·  ".join(drive_parts) if drive_parts else "—"

    if any(x not in ("—", "") for x in [v_where, v_blocker, v_drive]):
        st.markdown('<div class="sec-title">🔭 Your Vision</div>', unsafe_allow_html=True)
        vc    = st.columns(3)
        items = [
            ("📍 Where you are",   v_where),
            ("🚧 Biggest blocker", v_blocker),
            ("🔥 Your drive",      v_drive),
        ]
        accent = ["#a78bfa", "#f87171", "#34d399"]
        for i, (col, (lbl, txt)) in enumerate(zip(vc, items)):
            col.markdown(
                f'<div class="v-card" style="border-color:{accent[i]}33">'
                f'<div class="v-label" style="color:{accent[i]}">{lbl}</div>'
                f'<div class="v-text">{txt}</div></div>',
                unsafe_allow_html=True
            )

    dest = rd.get("target_role", "")
    if dest:
        st.markdown(f"<br>🎯 **Your goal:** {dest}", unsafe_allow_html=True)

    # ── Milestones ──
    if milestones:
        # ── v2.2 CHANGE 3: Consultation video banner before milestones ──
        st.markdown(f"""
<div style="background:linear-gradient(135deg,#1E3FA8 0%,#2D5BE3 60%,#3B6FFF 100%);
     border-radius:16px;padding:18px 22px;margin-bottom:20px;
     display:flex;align-items:center;gap:14px;
     box-shadow:0 4px 20px rgba(45,91,227,0.25);">
  <div style="width:44px;height:44px;border-radius:50%;
       background:rgba(255,255,255,0.15);display:flex;
       align-items:center;justify-content:center;
       font-size:18px;color:#fff;flex-shrink:0;
       border:1.5px solid rgba(255,255,255,0.25);">▶</div>
  <div style="flex:1;">
    <div style="font-size:14px;font-weight:700;color:#fff;margin-bottom:4px;">
      Your Career Consultation is ready
    </div>
    <div style="font-size:12px;color:rgba(255,255,255,0.75);line-height:1.5;">
      A personalised video built from your onboarding — your roadmap, your language, your next step.
    </div>
  </div>
  <div style="padding:9px 20px;border-radius:10px;background:#fff;color:#2D5BE3;
       font-size:12px;font-weight:700;white-space:nowrap;cursor:pointer;">
    Watch now ▶
  </div>
</div>
""", unsafe_allow_html=True)

        st.markdown(
            '<div class="sec-title">🏆 Your Learning Journey</div>',
            unsafe_allow_html=True
        )

        active_id = rd.get("current_active_milestone", "")

        for idx, ms in enumerate(milestones):
            seq       = ms.get("sequence_order", idx + 1)
            m_id      = get_milestone_label(ms, icp_val, seq)
            m_label   = ms.get("t", ms.get("identity_label", m_id))
            m_stmt    = ms.get("o", ms.get("identity_statement", ""))
            modules   = ms.get("modules", [])
            raw_ms_id = ms.get("milestone_id", f"M{seq:02d}")
            is_active = (raw_ms_id == active_id)

            c1, c2      = milestone_color(seq)
            salary_data = get_salary_for_milestone(ms, m_id, icp_val, target_role)
            emoji       = milestone_emoji(seq)
            border_col  = c2 if not is_active else "#34d399"
            glow        = f"box-shadow:0 0 24px {border_col}28;" if is_active else ""

            sc_n = ms.get("sc_n", 0)
            iv   = ms.get("iv", 0)
            scenario_chip = (
                f'<span class="mod-chip">🧩 {sc_n} Scenarios</span>' if sc_n else ""
            )
            interview_chip = (
                f'<span class="mod-chip">🎤 {iv} Interviews</span>' if iv else ""
            )

            active_badge = (
                "<div style='font-size:11px;background:#0d2010;color:#4ade80;"
                "border:1px solid #2d6a4f;border-radius:20px;padding:3px 10px;"
                "align-self:flex-start;white-space:nowrap;'>🟢 Active Now</div>"
                if is_active else ""
            )

            st.markdown(f"""
<div class="ms-card" style="background:linear-gradient(135deg,#0f0f1c,#13131f);
     border:1px solid {border_col}44;{glow}">
  <div style="position:absolute;top:0;left:0;right:0;height:4px;
       background:linear-gradient(90deg,{c1},{c2});border-radius:18px 18px 0 0;"></div>

  <div style="display:flex;align-items:flex-start;gap:14px;">
    <div class="ms-label-badge" style="background:linear-gradient(135deg,{c1},{c2})">{m_id}</div>
    <div style="flex:1;min-width:0;">
      <div style="font-size:19px;font-weight:800;color:#f0edff;line-height:1.3;">
        {emoji} {m_label}
      </div>
      <div style="font-size:13px;color:#8888aa;margin-top:3px;font-style:italic;">
        {m_stmt}
      </div>
      <div style="margin-top:8px;">{scenario_chip}{interview_chip}</div>
    </div>
    {active_badge}
  </div>

  <div class="salary-block">
    <div class="salary-source">💰 Earning potential at this level &nbsp;·&nbsp; {salary_data["source"]}</div>
    <div class="salary-main" style="color:{c2}">{salary_data["display"]}</div>
  </div>
</div>
""", unsafe_allow_html=True)

            # ── Modules ──
            for mod in modules:
                mod_id    = mod.get("id", mod.get("module_id", f"MOD{seq}"))
                mod_title = mod.get("title", f"Module {mod.get('sequence_order', '?')}")
                mod_desc  = mod.get("description", "")
                mod_seq   = mod.get("sequence_order", "?")
                is_free   = mod.get("free", False)
                vis_type  = mod.get("vis", "")
                skills    = mod.get("skills", [])
                lessons   = extract_module_lessons(mod)
                science   = mod.get("science", [])

                free_tag = " 🔓 Free" if is_free else " 🔒"
                expander_label = (
                    f"📦 {mod_id}. {mod_title}{free_tag}"
                    f"  ·  {len(skills)} skills  ·  {len(lessons)} lessons"
                )

                with st.expander(expander_label):

                    chips_html = ""
                    if vis_type:
                        chips_html += f'<span class="mod-chip">🖥 {vis_type}</span>'
                    if is_free:
                        chips_html += '<span class="mod-chip mod-chip-free">🔓 Free</span>'
                    if chips_html:
                        st.markdown(
                            f'<div style="margin-bottom:12px;">{chips_html}</div>',
                            unsafe_allow_html=True
                        )

                    if mod_desc:
                        st.markdown(
                            f'<div class="module-desc">{mod_desc}</div>',
                            unsafe_allow_html=True
                        )

                    # ── Lessons list ──
                    if lessons:
                        st.markdown(
                            '<div style="font-size:12px;font-weight:700;color:#6b7280;'
                            'text-transform:uppercase;letter-spacing:0.08em;margin-bottom:8px;">'
                            '🎬 Lessons</div>',
                            unsafe_allow_html=True
                        )
                        lessons_html = "".join(
                            f'<div class="lesson-item">▶ {lesson}</div>'
                            for lesson in lessons
                        )
                        st.markdown(lessons_html, unsafe_allow_html=True)

                    # ── Science: Scenarios / Interviews ──
                    if science:
                        science_html = ""
                        for sci in science:
                            sci_type = sci.get("type", "")
                            sci_desc = sci.get("desc", sci.get("description", ""))
                            if sci_type == "Scenario":
                                science_html += (
                                    f'<span class="science-badge science-scenario">'
                                    f'🧩 Scenario: {sci_desc}</span>'
                                )
                            elif sci_type == "Interview":
                                science_html += (
                                    f'<span class="science-badge science-interview">'
                                    f'🎤 Interview: {sci_desc}</span>'
                                )
                        if science_html:
                            st.markdown(
                                f'<div style="margin:10px 0 14px 0;">{science_html}</div>',
                                unsafe_allow_html=True
                            )

                    # ── Skills ──
                    if skills:
                        st.markdown(
                            '<div style="font-size:12px;font-weight:700;color:#6b7280;'
                            'text-transform:uppercase;letter-spacing:0.08em;'
                            'margin:14px 0 10px 0;">⚡ Skills</div>',
                            unsafe_allow_html=True
                        )

                    for skill in skills:
                        s_title = skill.get("title", skill.get("n", ""))
                        s_desc  = skill.get("description", "")

                        mastery_state = skill.get("mastery_state", {})
                        p_val         = skill.get("p", None)

                        if p_val is not None:
                            curr_mastery   = p_val / 100.0
                            target_mastery = 0.9
                            s_state        = (
                                "mastered"    if p_val >= 90  else
                                "mock_ready"  if p_val >= 70  else
                                "in_progress" if p_val >= 30  else
                                "unlocked"    if p_val > 0    else
                                "locked"
                            )
                        else:
                            curr_mastery   = mastery_state.get("current_mastery", 0.0)
                            target_mastery = mastery_state.get("target_mastery", 0.9)
                            s_state        = mastery_state.get("state", "locked")

                        bg, border, state_lbl = skill_state_color(s_state)

                        content_flow  = skill.get("content_flow", {})
                        video    = content_flow.get("video", {})
                        scenario = content_flow.get("scenario", {})
                        mock     = content_flow.get("mock", {})

                        content_tags = ""
                        if video:
                            content_tags += '<span class="ctag">🎬 Video</span>'
                        if scenario:
                            content_tags += '<span class="ctag">🧩 Scenario</span>'
                        if mock:
                            content_tags += '<span class="ctag">📝 Mock Test</span>'

                        mastery_pct = int(curr_mastery * 100)
                        bar_width   = max(mastery_pct, 2)

                        st.markdown(f"""
<div class="skill-card" style="background:{bg};border-color:{border};">
  <div class="skill-status-row">
    <div class="skill-title">{s_title}</div>
    <span class="state-pill" style="background:{bg};border:1px solid {border};color:{border};">
      {state_lbl}
    </span>
  </div>
  <div class="skill-desc">{s_desc}</div>
  <div class="mastery-bar-wrap">
    <div class="mastery-bar-fill" style="width:{bar_width}%;"></div>
  </div>
  <div style="font-size:11px;color:#5a5a7a;margin-bottom:10px;">
    Progress: {mastery_pct}% &nbsp;·&nbsp; Target: {int(target_mastery * 100)}%
  </div>
  <div class="content-tags">{content_tags}</div>
</div>
""", unsafe_allow_html=True)

    # ── Pinecone status (UNCHANGED) ──
    stored = rd.get("pinecone_stored", None)
    if stored is True:
        st.success("✅ Roadmap saved to Pinecone")
    elif stored is False:
        st.warning("⚠ Generated but NOT saved to Pinecone — check logs")

    # ── Download (UNCHANGED) ──
    st.markdown("---")
    with st.expander("🧾 Raw JSON (developers)"):
        st.json(rd)

    st.download_button(
        label="⬇ Download Roadmap JSON",
        data=json.dumps(rd, indent=2),
        file_name="vidya_roadmap_v3.json",
        mime="application/json",
        key="download_roadmap_json"
    )
    
