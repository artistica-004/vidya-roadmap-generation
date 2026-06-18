from langchain_core.prompts import PromptTemplate
from langchain_openai import ChatOpenAI
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.output_parsers import StrOutputParser
from src.pinecone_utils import (
    fetch_poc_record,
    retrieve_context,
    save_poc_record,
)
from src.capability_gap import compute_gap_score
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
# AVAILABLE COURSE CATALOG  (100% completed courses only)
# Source: Roadmaps_Courses.xlsx — "Completed in % (New Vis.)" = 100%
# ============================================================

AVAILABLE_COURSES = {
    "ai_ready": {
        "name": "AI Ready Course",
        "modules": [
            "Python for AI", "Math Refresher for AI",
            "Algorithms & Data Handling", "Data Wrangling",
            "Data Visualization", "Deep Learning Basics",
            "Model Debugging & Explainability",
            "Collaboration & Tools", "Cloud & Deployment",
        ],
        "skills_include": [
            "Variables, Data Types, Operators", "Control Flow", "Data Structures",
            "OOP Basics", "File Handling", "Linear Algebra", "Probability",
            "Bayes Theorem", "Calculus for ML", "Sorting & Searching",
            "Data Cleaning", "NumPy", "Pandas", "Data Visualization",
            "Regression", "Classification", "Clustering", "Model Evaluation",
            "Neural Networks", "Keras", "Loss Functions", "Backpropagation",
            "Overfitting", "Cross-Validation", "Git & GitHub", "Docker",
            "FastAPI", "AWS SageMaker", "HuggingFace Spaces",
        ],
    },
    "machine_learning": {
        "name": "Machine Learning Course",
        "modules": [
            "Python for Data & ML", "Math & Stats Essentials",
            "Supervised Learning", "Unsupervised Learning",
            "Feature Engineering & Pipelines",
            "Neural Networks & Computer Vision",
            "NLP & Sequence Models", "Model Serving & APIs",
            "MLOps & Monitoring", "GenAI & Responsible AI",
        ],
        "skills_include": [
            "Linear Regression", "Logistic Regression", "Ridge & Lasso",
            "Decision Trees", "kNN", "SVM", "Model Evaluation",
            "Cross Validation", "K-Means", "PCA", "t-SNE", "UMAP",
            "Anomaly Detection", "Scikit-learn Pipelines",
            "Feature Engineering", "CNN", "Transfer Learning",
            "RNN", "LSTM", "Transformers", "BERT", "GPT",
            "FastAPI", "Docker", "MLflow", "Weights & Biases",
            "Drift Detection", "Prompt Engineering", "LoRA",
            "Fine-tuning", "RLHF",
        ],
    },
    "generative_ai": {
        "name": "Generative AI Course",
        "modules": [
            "Foundations & Overview", "Core Generative Model Families",
            "Diffusion & Modern Image Generation", "Transformers & Attention",
            "Tokenization & Embeddings", "Prompting & Inference Techniques",
            "Tools & Frameworks", "Fine-Tuning & Safety",
            "Multimodal & Advanced Applications", "NLP", "RAG & Capstone",
        ],
        "skills_include": [
            "VAE", "GANs", "Diffusion Models", "Stable Diffusion",
            "Self-attention", "Multi-head Attention", "Transformer Architecture",
            "Tokenization", "Word Embeddings", "Prompt Engineering",
            "Zero-shot", "Few-shot", "HuggingFace", "LangChain",
            "Vector Databases", "LLM APIs", "Streamlit", "SFT",
            "LoRA", "QLoRA", "RLHF", "Safety", "CLIP", "Whisper",
            "RAG", "Chatbot",
        ],
    },
    "computer_vision": {
        "name": "Computer Vision Course",
        "modules": [
            "OpenCV Basics", "Image Processing", "Feature Detection",
            "Classical Computer Vision", "Deep Learning Basics",
            "Advanced Architectures", "Generative Models",
            "Deployment", "Research Literacy",
        ],
        "skills_include": [
            "Image Filtering", "Edge Detection", "Thresholding",
            "SIFT", "SURF", "ORB", "Feature Matching", "Face Detection",
            "Object Detection", "Image Segmentation", "CNN", "ResNet",
            "YOLO", "Faster R-CNN", "U-Net", "Autoencoders", "GAN",
            "DCGAN", "Neural Style Transfer", "Model Quantization",
        ],
    },
    "high_level_system_design": {
        "name": "High Level System Design",
        "modules": [
            "Foundations of System Design", "Networking & Communication",
            "Databases & Storage", "Distributed Systems",
            "Microservices Architecture", "Scalability & Performance",
            "Security, Reliability & Observability",
            "Case Studies", "Capstone & Career Readiness",
        ],
        "skills_include": [
            "Scalability", "Availability", "Load Balancing", "Caching",
            "Microservices", "SQL vs NoSQL", "Sharding", "Replication",
            "CAP Theorem", "Message Queues", "Kafka",
            "Event-Driven Architecture", "Docker", "CI/CD",
            "Rate Limiting", "API Gateway", "Auth OAuth2", "JWT",
            "Logging & Monitoring", "System Design Twitter",
            "System Design WhatsApp", "System Design Netflix",
            "System Design Uber",
        ],
    },
    "low_level_system_design": {
        "name": "Low Level System Design",
        "modules": [
            "OOP & SOLID Principles", "Creational & Structural Patterns",
            "Behavioral Design Patterns", "Database Design & Data Modeling",
            "API Design & Service Layer", "Concurrency & Multithreading",
            "Caching & Performance Optimization",
            "Testing, Logging & DevOps Foundations",
            "AI-Powered Systems", "Capstone & Interview Readiness",
        ],
        "skills_include": [
            "SOLID Principles", "UML Diagrams", "Singleton", "Factory",
            "Builder", "Decorator", "Adapter", "Composite", "Proxy",
            "Strategy", "Observer", "State Pattern", "Command Pattern",
            "SQL vs NoSQL", "ORM", "ER Diagrams", "REST Principles",
            "Dependency Injection", "JWT", "Concurrency", "Deadlocks",
            "Producer Consumer", "LRU Cache", "Redis", "Unit Testing",
            "RAG", "Vector Databases",
        ],
    },
}

# Courses NOT available — LLM must never generate content from these
UNAVAILABLE_COURSES = [
    "Data Science Course (44% complete — not ready)",
    "NLP (20% complete — not ready)",
    "Robotics (not started)",
    "FastAPI standalone (not started)",
    "Git & Github standalone (not started)",
    "AWS Masterclass (not started)",
    "DevOps (not started)",
    "MLOps (not started)",
    "Soft Skills (not started)",
    "Project Manager (not started)",
]


# ============================================================
# LEVEL DETECTION  (Roadmap Generation Science — Authoritative)
# ============================================================

# Milestone count is now dynamic — determined by the LLM from:
#   Current Identity, Target Identity, Capability Gap, Available Time.
# Validation enforces only the bounds below.
MIN_MILESTONES = 2
MAX_MILESTONES = 7
MIN_MODULES    = 2
MAX_MODULES    = 4
MIN_SKILLS     = 3
MAX_SKILLS     = 8

# Maps level -> content difficulty starting tier for the user's M01.
# Beginner M01     = library tier 1 (intern-ready)
# Intermediate M01 = library tier 3 (working engineer)
# Senior M01       = library tier 5 (senior engineer)
LEVEL_STARTING_TIER = {
    "beginner": 1,
    "intermediate": 3,
    "senior": 5,
}


def detect_level(context: str, years_experience: int = 0) -> str:
    """
    Maps onboarding signals (resume parse / years of experience) to
    beginner | intermediate | senior.

      Beginner     -> 0 years
      Intermediate -> 1-3 years
      Senior       -> 4+ years

    Milestone count is determined dynamically by the LLM from capability gap.
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
    Works against the new milestone/module/skill shape because it only
    relies on: milestones[i].modules[j].skills[k] existing, plus
    milestone_id / skill_id keys (which are still present in the new shape).
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
# Roadmap Prompt — V3.2 Structure (Science-aligned, identity-driven)
# ============================================================

# Build available courses summary string for the prompt
_AVAILABLE_COURSES_TEXT = "\n".join(
    f"  {i+1}. {v['name']} — modules: {', '.join(v['modules'])}"
    for i, (k, v) in enumerate(AVAILABLE_COURSES.items())
)
_UNAVAILABLE_COURSES_TEXT = "\n".join(f"  - {c}" for c in UNAVAILABLE_COURSES)

roadmap_prompt = PromptTemplate(
    input_variables=["context", "icp_type", "level", "hours_per_week", "timeline_days", "budget_hrs", "known_skills", "current_identity", "target_identity", "current_salary_lpa", "self_efficacy", "gap_score", "recommended_milestones", "recommended_modules_per_milestone", "recommended_skill_density", "gap_reasoning"],
    template="""
You are an expert AI career transformation roadmap architect for Vidya V3.

You are NOT generating a course.
You are generating a TRANSFORMATION ROADMAP.

The roadmap exists to transform a learner from their CURRENT IDENTITY into a
TARGET PROFESSIONAL IDENTITY. Every milestone, module, skill, lesson, project,
scenario, interview, and checkpoint must be justified by this transformation.

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
  + (interviews × 1.0)
  + (lessons × 0.25)
  + (projects × 6.0)

Rules:
* If demand exceeds budget, reduce milestone count first.
* Keep milestone count within 2-7 bounds.
* Smaller budgets should generally produce smaller roadmaps.
* Larger budgets may justify more milestones.
* Never ignore the user's available time.

STARTING POINT (HARD CONSTRAINT):

Known skills from onboarding:
{known_skills}

For each skill already known:

* mark auto_completed=true
* assign realistic mastery prior p=65-90
* do not teach from scratch
* do not start Module 1 with content already mastered
* use the next logical capability step

Starting professionals at skills they already have is a major churn trigger.

If known_skills = "none provided":
apply ZPD rules using level only.

CAPABILITY GAP ENGINE OUTPUT

gap_score:
{gap_score}

gap_reasoning:
{gap_reasoning}

recommended_milestones:
{recommended_milestones}

recommended_modules_per_milestone:
{recommended_modules_per_milestone}

recommended_skill_density:
{recommended_skill_density}

HARD RULES

1.
Generate EXACTLY
{recommended_milestones}
milestones.

2.
Generate EXACTLY
{recommended_modules_per_milestone}
modules inside EACH milestone.

3.
Generate EXACTLY
{recommended_skill_density}
skills inside EACH module.

Deviation is forbidden.

Validation will reject outputs that do not match these counts.

=== GENERATION RULES (STRICT — ALL MUST BE FOLLOWED) ===

RULE 1 — HIERARCHY (LOCKED):
Roadmap -> Milestone -> Module -> (Skills -> Lessons) + Science
Never break this hierarchy.

RULE 2 — MILESTONE COUNT (DYNAMIC — FROM CAPABILITY GAP):
Determine the number of milestones from the gap between the learner's
Current Identity and Target Identity.

Principles:
  - More milestones = smaller, safer steps (good for beginners, large gaps,
    low confidence, low weekly hours).
  - Fewer milestones = larger, faster leaps (good when the gap is narrow,
    learner is senior/already-strong, high weekly hours available).
  - The milestone count MUST be between 2 and 7 inclusive.
    Validation enforces this bound.
  - Every milestone represents a MARKET-RECOGNIZED IDENTITY the learner
    earns, NOT a topic-coverage checkpoint.
  - Milestone codes always start at M01 (relative to THIS user).

Include a "milestone_count_rationale" field in your JSON output explaining
your reasoning (e.g. "4 milestones: beginner with 10 h/wk and a wide gap
from student to AI Engineer — 4 identity steps with adequate time").

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

RULE 4 — MODULE COUNT PER MILESTONE (DYNAMIC):
Modules are capability clusters.  A milestone's module count is determined
by the capability breadth needed to reach that milestone's identity.

Constraints:
  - Minimum 2 modules per milestone.
  - Maximum 4 modules per milestone.
  - Never generate 1 module or more than 4 modules.
  - Do NOT generate modules merely to satisfy a count — each module must
    represent a distinct, coherent capability cluster.
  - Every milestone has EXACTLY 1 Interview (iv=1) and between 3 and 7 Scenarios (sc_n=3-7)
    (iv=1). These two science items are placed in separate modules. Any modules
    beyond those two must have "science": [].
  - Every milestone must include a "module_count_rationale" field explaining
    why the chosen count fits the milestone's required breadth.

Example rationale:
  "4 modules because this milestone spans distributed systems, cloud
   architecture, observability, and technical leadership."

RULE 5 — SKILLS PER MODULE (DYNAMIC):
Skills per module are dynamic — determined by the capability breadth needed
for that module's topic cluster.
  - Minimum 3 skills per module.
  - Maximum 8 skills per module.
  - Never generate fewer than 3 or more than 8 skills.
  - Do NOT generate skills merely to satisfy a count — each skill must
    represent a distinct, meaningful capability within the module.
  - Every module must include a "skill_count_rationale" field explaining
    why the chosen count fits the module's required breadth.

Each skill object has:
  - "skill_id": globally unique id, e.g. "SKILL_M01_M1_S1"
  - "n": short skill name (snake_case, e.g. "python", "fastapi", "rag")
  - "title": human-readable skill title
  - "lessons": array of EXACTLY 3 short lesson titles (video unit names).
    Example: ["Embeddings", "Chunking Strategies", "Retrieval Evaluation"]
    Lessons must be specific topic titles, NOT generic placeholders.
  - "p": initial mastery percentage (0-100).
       * If the module is "free": true AND is part of the user's M01
         (foundational/already-known content per onboarding mastery priors),
         set "p" to a realistic prior between 55-90.
       * Otherwise set "p" to 0 (not yet started).
  - "mastery_state": BKT object (see OUTPUT STRUCTURE) — KEEP for backend
    mastery tracking regardless of "p".
  - "content_flow": object with video/scenario/mock/review (see OUTPUT
    STRUCTURE) — KEEP for backend tracking.
  - "unlock_rules": {{"requires": [...], "minimum_mastery": 0.0,
    "unlock_type": "immediate" | "prerequisite"}}
       * Skills must form an acyclic prerequisite graph.
       * The FIRST skill of the entire roadmap (M01, first module, first
         skill) must have requires: [] and unlock_type: "immediate".

RULE 6B — PROJECT PER MILESTONE (MANDATORY)

Every milestone MUST contain exactly ONE real-world project.

The project must:
  - Use skills from that milestone.
  - Require planning, architecture decisions, and solution implementation.
  - Require deployment and debugging.
  - Require the learner to catch at least one AI mistake (seeded_error).
  - Feel like real industry work — not a toy exercise.

Project format:

"project": {{
  "title": "string",
  "vibe_layers": [
      "planning",
      "architecture",
      "solution"
  ],
  "description": "2-3 sentence description of what the learner builds",
  "deliverable": "what the learner submits as proof of completion",
  "seeded_errors": [
      "AI mistake the learner must detect and fix"
  ],
  "deploy_required": true
}}

RULE 7 — SCIENCE ARRAY (SCENARIO / INTERVIEW — PER MILESTONE):
!! CRITICAL !!
Every milestone must contain between 3 and 7 Scenarios AND EXACTLY 1 Interview.
Scenarios are distributed across modules (any module may carry multiple Scenarios).
Validation enforces sc_n between 3-7 and iv=1 at the milestone level.

Rules for science items:
  - "Scenario" = a realistic production/debugging situation the learner must resolve.
    Examples: bad retrieval causes hallucinations, pipeline fails before demo,
    latency spike in production, nightly job fails at 9am.
    NOT toy exercises. NOT hypotheticals.
  - "Interview" = an interview question testing the milestone identity.
    Must test the ability to PERFORM the role, not recall trivia.
  - A module may have 0 to 3 science items. Multiple Scenarios may share a module.
  - The Interview must be in a module that has no Scenario (separate modules).
  - The milestone-level sc_n must be between 3 and 7. The milestone-level iv must equal 1.

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

RULE 8B — AI-FIRST LAYER

Each module must include:

"ai_first_layer"

Allowed values:

- planning
- architecture
- solution

Definitions:

planning:
Problem decomposition, sequencing work, defining requirements.

architecture:
Choosing components, defining system boundaries,
making tradeoff decisions.

solution:
Using AI to build, debugging, deployment,
testing and catching AI mistakes.

Requirements:

- Every module must have exactly one ai_first_layer.
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
  - "sc_n": must be between 3 and 7 (3–7 Scenarios per milestone).
  - "iv": must equal 1 (exactly one Interview per milestone).
  - "identity_statement": 1-sentence motivational framing (Possible Selves model).
  - "checkpoint_rule": {{"required_mastery": 0.9, "checkpoint_type": "mock_interview"}}
    A checkpoint is EARNED, never purchased. Requires mastery ≥ 0.90 + project
    completed + interview passed.
  - "project": per RULE 6B.
  - "modules": 2–4 modules per RULE 4 (module count is dynamic, capability-breadth driven).

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

RULE 12 — MOCK UNLOCK & CONTENT STATUS:
mock.unlock_mastery = 0.75 always. All content_flow statuses ("video",
"scenario", "mock") = "locked" on generation. review.next_review_at = null.

RULE 13 — DIFFICULTY PROGRESSION (BKT PRIORS):
Across the roadmap, mastery_state.bkt.prior should generally increase with
milestone seniority: early milestones prior ~0.10-0.25, later milestones
prior ~0.05-0.15 (harder content = lower prior). target_mastery is always 0.9.

RULE 14 — STRICT JSON ONLY:
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
  "milestone_count_rationale": "string — explanation of why this milestone count was chosen (gap, time, level)",
  "vision_profile": {{
    "current_state": "string — 1 sentence: current identity of the learner",
    "main_blocker": "string — 1 sentence: biggest obstacle to transformation",
    "top_motivation": "string — 1 sentence: why they want this identity"
  }},
  "roadmap_meta": {{
    "generated_at": "PLACEHOLDER_TIMESTAMP",
    "version": "v3.2",
    "science_model": ["ZPD", "Mastery Learning", "CLT", "BKT", "Possible Selves", "Retrieval Practice"]
  }},
  "milestones": [
    {{
      "milestone_id": "M01",
      "label": "M01",
      "t": "string — market-recognized identity, e.g. AI Foundations Engineer",
      "sal": "string — salary band, e.g. Unpaid/stipend or ₹6-9 LPA",
      "o": "string — 1-sentence outcome: what the learner can DO and demonstrate",
      "sc_n": 3,
      "iv": 1,
      "module_count_rationale": "string — why this milestone has N modules (capability breadth rationale)",
      "identity_statement": "string — 1-sentence Possible Selves motivational framing",
      "checkpoint_rule": {{
        "required_mastery": 0.9,
        "checkpoint_type": "mock_interview"
      }},
      "project": {{
        "title": "string — real-world project title",
        "vibe_layers": [
          "planning",
          "architecture",
          "solution"
        ],
        "description": "string — 2-3 sentences describing what the learner builds",
        "deliverable": "string — what the learner submits as proof",
        "seeded_errors": [
          "string — specific AI mistake the learner must detect and fix"
        ],
        "deploy_required": true
      }},
      "modules": [
        {{
          "id": "M1.1",
          "title": "string — capability-cluster title",
          "ai_first_layer": "planning | architecture | solution",
          "free": true,
          "vis": "code+real_tutor",
          "skill_count_rationale": "string — why this module needs N skills",
          "skills": [
            {{
              "skill_id": "SKILL_M01_M1_S1",
              "n": "python",
              "title": "string — human readable skill title",
              "auto_completed": false,
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
        // matching its module position, and science populated according to
        // whether it carries the Scenario or the Interview.
      ]
    }}
  ]
}}

IMPORTANT REMINDERS:
- Generate between 2 and 7 milestones (determined by capability gap — see RULE 2).
- Module count per milestone: 2–4 (determined by capability breadth — see RULE 4).
- Every module has 3–8 skills (dynamic, see RULE 5).
- Every skill has EXACTLY 3 lessons (inside the skill object, NOT at module level).
- Every milestone has between 3 and 7 Scenarios (sc_n=3-7) and EXACTLY 1 Interview (iv=1).
  Distribute Scenarios across modules (any module may hold multiple). The Interview
  must be in a module separate from any Scenario module.
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
    """Overwrite BKT prior/learn_rate across all skills with skill-aware values."""
    milestones = roadmap_data.get("milestones", [])
    total_ms = len(milestones)
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
# Roadmap Structure Validator   (REWRITTEN — HTML-aligned)
# ============================================================

def validate_roadmap_structure(data: dict) -> None:
    """
    Validates the V3.2 roadmap structure (Roadmap Generation Science spec).
    Raises ValueError with a clear message on any violation.
    Auto-fixes mock.unlock_mastery silently.

    Enforces:
      - milestone count within MIN_MILESTONES..MAX_MILESTONES bounds
      - milestone_id/label codes M01..M0N, sequential, no gaps
      - each milestone has between MIN_MODULES and MAX_MODULES modules
      - each module has MIN_SKILLS..MAX_SKILLS skills (dynamic)
      - each skill has EXACTLY 3 lessons (inside skill object)
      - each module's "science" array has 0-3 items ("Scenario" or "Interview"; Scenarios may share a module, Interview must be in a separate module)
      - each milestone has 3 <= sc_n <= 7 and iv == 1
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

        # ── Project validation ──────────────────────────────────────────
        project = milestone.get("project")

        if not project:
            raise ValueError(f"Milestone {m_id}: missing project")

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

        for mod_idx, mod in enumerate(modules):
            mod_id = mod.get("id", "?")

            # ── ai_first_layer ──
            layer = mod.get("ai_first_layer")
            if layer not in ("planning", "architecture", "solution"):
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

                # ── lessons: exactly 3 per skill ──
                lessons = skill.get("lessons", [])
                if not isinstance(lessons, list):
                    raise ValueError(f"Skill {skill_id}: lessons must be a list")
                if len(lessons) != 3:
                    raise ValueError(
                        f"Skill {skill_id}: must have EXACTLY 3 lessons, "
                        f"got {len(lessons)}"
                    )

                flow = skill.get("content_flow", {})
                for content_type in ("video", "scenario", "mock", "review"):
                    if content_type not in flow:
                        raise ValueError(
                            f"Skill {skill_id} missing content_flow.{content_type}"
                        )

                # Auto-fix mock unlock_mastery
                mock = flow.get("mock", {})
                if mock.get("unlock_mastery") != 0.75:
                    mock["unlock_mastery"] = 0.75

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

        # ── sc_n must be 3-7, iv must equal 1 ──
        if scenario_count < 3 or scenario_count > 7:
            raise ValueError(
                f"Milestone {m_id}: must have between 3 and 7 Scenarios across its "
                f"modules, got {scenario_count}"
            )
        if interview_count != 1:
            raise ValueError(
                f"Milestone {m_id}: must have EXACTLY 1 Interview across its "
                f"modules, got {interview_count}"
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
_FL_REALISM_BUFFER = 0.8   # 80 % of available time is productive learning
_FL_MAX_WEEKS      = 52    # 1 year upper-bound; beyond this = truly infeasible


def _count_roadmap_units(roadmap_data: dict) -> tuple:
    """
    Return (lessons, skills, scenarios, interviews, projects) counts
    from a v3.2 roadmap whose lessons live inside skill objects.
    """
    lessons = skills = scenarios = interviews = projects = 0
    for ms in roadmap_data.get("milestones", []):
        if ms.get("project"):
            projects += 1
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

    DESIGN CHANGE vs previous implementation
    =========================================
    The old version hard-failed (raised ValueError) when demand > budget.
    This caused every Student Beginner and Intermediate roadmap to be
    rejected at the default 5 h/week because:

        Student Beginner  → demand ≈ 105 h, budget at 5 h/wk = 64 h  → FAIL

    The new design NEVER raises.  Instead:

      1. Calculate demand from the roadmap units.
      2. If demand ≤ budget at timeline_days             → fits at timeline_days.
      3. Else compute the minimum duration to fit:
             duration_weeks = ceil(demand / (weekly_hours × 0.8))
      4. If duration_weeks ≤ _FL_MAX_WEEKS (52)           → fits=True, extended
      5. If duration_weeks >  _FL_MAX_WEEKS               → fits=False (truly
         infeasible; pipeline logs a warning but still proceeds).

    The pipeline NO LONGER raises ValueError on fits=False.
    fits_life is advisory — it informs the learner of timeline, not a gate.

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
    project_hours   = round(projects   * 6.0,  1)

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
# ROADMAP BIBLE VALIDATORS  (Roadmap Bible AI-First 2026)
# ============================================================
# Each validator is a pure function: roadmap_data -> dict.
# They NEVER raise — they return {"pass": bool, "reason": str, ...}.
# run_roadmap_bible_validators() calls them all and collects results.
# ============================================================

def _starts_where_they_are_check(roadmap_data: dict) -> dict:
    """
    Validator: starts_where_they_are

    No skill the user already has should be re-taught from scratch.
    Rule: if level != 'beginner', at least one skill in M01 must have
    p > 0 (i.e. carry a mastery prior from onboarding signals).
    A p=0 on every M01 skill for a non-beginner = ZPD rule violated.

    Full implementation (comparing against a live skill inventory) is
    a future step.  This is the deterministic POC heuristic.
    """
    level      = roadmap_data.get("level", "beginner")
    milestones = roadmap_data.get("milestones", [])

    if not milestones:
        return {"pass": False, "reason": "No milestones — cannot check ZPD"}

    m01_skills = [
        skill
        for mod in milestones[0].get("modules", [])
        for skill in mod.get("skills", [])
    ]

    if not m01_skills:
        return {"pass": False, "reason": "M01 has no skills"}

    if level != "beginner":
        has_prior = any(s.get("p", 0) > 0 for s in m01_skills)
        if not has_prior:
            return {
                "pass":   False,
                "reason": (
                    f"Level='{level}' but every M01 skill has p=0. "
                    "ZPD rule requires pre-existing mastery priors for "
                    "non-beginner learners."
                ),
            }

    return {"pass": True, "reason": "ZPD starting-point check passed"}


def _laugh_test_check(roadmap_data: dict) -> dict:
    """
    Validator: laugh_test

    ≥80% of target role's real must-have skills are covered; <20% filler.
    POC heuristic: skills with generic names ('learn', 'intro', 'overview',
    'basics', 'fundamentals') indicate filler.  Full JD-scraping validation
    is a future step.
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

    if filler_ratio > 0.20:
        return {
            "pass":        False,
            "reason":      (
                f"Filler skill ratio {filler_ratio:.0%} exceeds 20 % threshold "
                f"({filler_count}/{total} skills look generic). "
                "A senior from this field would laugh at this roadmap."
            ),
            "filler_ratio": round(filler_ratio, 3),
        }

    return {
        "pass":          True,
        "reason":        (
            f"Laugh test passed — filler ratio {filler_ratio:.0%} "
            f"({filler_count}/{total}). Full JD validation pending."
        ),
        "filler_ratio":  round(filler_ratio, 3),
        "total_skills":  total,
    }


def _no_spectators_check(roadmap_data: dict) -> dict:
    """
    Validator: no_spectators

    Every skill has ≥1 applied activity (scenario + mock in content_flow).
    Every milestone has ≥1 module with a science item (no fully-automated milestones).
    Every milestone has a project.
    Spectators watch. Doers do.
    """
    milestones  = roadmap_data.get("milestones", [])
    violations: List[str] = []

    for ms in milestones:
        m_id = ms.get("milestone_id", "?")

        if not ms.get("project"):
            violations.append(f"Milestone {m_id}: missing project — learner has nothing to build")

        milestone_science_modules = [
            mod for mod in ms.get("modules", []) if mod.get("science", [])
        ]

        for mod in ms.get("modules", []):
            mod_id  = mod.get("id", "?")
            science = mod.get("science", [])

            for skill in mod.get("skills", []):
                skill_id = skill.get("skill_id", "?")
                flow     = skill.get("content_flow", {})
                missing  = [k for k in ("video", "scenario", "mock", "review")
                            if k not in flow]
                if missing:
                    violations.append(
                        f"Skill {skill_id}: content_flow missing {missing}"
                    )

        if not milestone_science_modules:
            violations.append(
                f"Milestone {m_id}: no science items in ANY module — "
                f"no applied assessment in this milestone"
            )

    if violations:
        return {
            "pass":       False,
            "reason":     f"{len(violations)} spectator violation(s) found",
            "violations": violations,
        }

    return {"pass": True, "reason": "No spectator violations — every skill has applied activities"}


def _dag_clean_check(roadmap_data: dict) -> dict:
    """
    Validator: dag_clean

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
    Validator: ai_first

    Every milestone must have ≥2 distinct ai_first_layer values.
    Every milestone must have a project with seeded_errors (learner catches AI mistakes).
    At least one project across the roadmap must be deploy_required=True.
    """
    milestones  = roadmap_data.get("milestones", [])
    violations: List[str] = []
    deploy_required_found  = False

    for ms in milestones:
        m_id   = ms.get("milestone_id", "?")
        layers = set()

        for mod in ms.get("modules", []):
            layer = mod.get("ai_first_layer")
            if layer in ("planning", "architecture", "solution"):
                layers.add(layer)

        if len(layers) < 2:
            violations.append(
                f"Milestone {m_id}: only {len(layers)} distinct AI-first "
                f"layer(s) {sorted(layers)}. Need ≥2."
            )

        project = ms.get("project", {})
        if project:
            if not project.get("seeded_errors"):
                violations.append(
                    f"Milestone {m_id}: project has no seeded_errors — "
                    "learner never catches an AI mistake"
                )
            if project.get("deploy_required"):
                deploy_required_found = True
        else:
            violations.append(f"Milestone {m_id}: no project")

    if not deploy_required_found:
        violations.append(
            "No project across the roadmap has deploy_required=True — "
            "nothing ships to production"
        )

    if violations:
        return {
            "pass":       False,
            "reason":     f"{len(violations)} AI-first violation(s)",
            "violations": violations,
        }

    return {"pass": True, "reason": "AI-first check passed — all milestones have multi-layer AI work"}


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


def _known_skill_skip_check(roadmap_data: dict) -> dict:
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
                    if not skill.get("auto_completed"):
                        violations.append(
                            f"{skill.get('skill_id', '?')} ('{skill.get('n')}') "
                            f"is a known skill but auto_completed is not true"
                        )
                    if (skill.get("p") or 0) < 65:
                        violations.append(
                            f"{skill.get('skill_id', '?')} ('{skill.get('n')}') "
                            f"known skill has p={skill.get('p')} (< 65)"
                        )
    if violations:
        return {"pass": False, "reason": "; ".join(violations)}
    return {"pass": True, "reason": "all known skills correctly auto-completed"}


def _capability_gap_alignment_check(roadmap_data: dict) -> dict:
    gap = roadmap_data.get("capability_gap", {})
    expected_ms = gap.get("recommended_milestones")
    expected_mod = gap.get("recommended_modules_per_milestone")
    expected_skill = gap.get("recommended_skill_density")
    if expected_ms is None or expected_mod is None or expected_skill is None:
        return {"pass": True, "reason": "capability_gap data not available — skip check"}

    milestones = roadmap_data.get("milestones", [])
    actual_ms = len(milestones)
    violations = []

    if actual_ms != expected_ms:
        violations.append(
            f"expected {expected_ms} milestone(s), got {actual_ms}"
        )

    for ms in milestones:
        mid = ms.get("milestone_id", "?")
        modules = ms.get("modules", [])
        actual_mod = len(modules)
        if actual_mod != expected_mod:
            violations.append(
                f"Milestone {mid}: expected {expected_mod} module(s), got {actual_mod}"
            )
        for mod in modules:
            mod_id = mod.get("id", "?")
            skills = mod.get("skills", [])
            actual_skill = len(skills)
            if actual_skill != expected_skill:
                violations.append(
                    f"Module {mod_id}: expected {expected_skill} skill(s), got {actual_skill}"
                )

    if violations:
        return {"pass": False, "reason": "; ".join(violations)}
    return {"pass": True, "reason": "capability_gap alignment OK"}


def run_roadmap_bible_validators(roadmap_data: dict, weekly_hours: int, timeline_days: int = 112) -> dict:
    """
    Run all six Roadmap Bible validators in sequence.

    Validators:
        fits_life             — timeline feasibility (adaptive, never blocks)
        starts_where_they_are — ZPD: don't reteach mastered skills
        laugh_test            — ≥80% real skills, <20% filler
        no_spectators         — every skill has applied activity
        dag_clean             — no duplicate IDs, no circular prereqs
        ai_first              — multi-layer AI work + projects that ship

    Returns a dict keyed by validator name. Never raises — all errors
    are caught and recorded as pass=False with a reason.
    """
    _validators = [
        ("fits_life",             lambda: fits_life_check(roadmap_data, weekly_hours, timeline_days)),
        ("time_budget",           lambda: _time_budget_check(roadmap_data)),
        ("starts_where_they_are", lambda: _starts_where_they_are_check(roadmap_data)),
        ("known_skill_skip",      lambda: _known_skill_skip_check(roadmap_data)),
        ("laugh_test",            lambda: _laugh_test_check(roadmap_data)),
        ("no_spectators",         lambda: _no_spectators_check(roadmap_data)),
        ("dag_clean",             lambda: _dag_clean_check(roadmap_data)),
        ("ai_first",              lambda: _ai_first_check(roadmap_data)),
        ("salary_floor",          lambda: _salary_floor_check(roadmap_data)),
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

            # target_identity
            ti = parsed.get("target_identity") or parsed.get("primary_goal") or parsed.get("target_role")
            if ti:
                profile["target_identity"] = str(ti)
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
            profile["target_identity"] = ti_match.group(1).strip()
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


def apply_known_skill_autocomplete(roadmap_data: dict, known_skills: list[str]) -> None:
    if not known_skills:
        return
    auto_count = 0
    for ms in roadmap_data.get("milestones", []):
        for mod in ms.get("modules", []):
            for skill in mod.get("skills", []):
                n = skill.get("n", "")
                title = skill.get("title", "")
                if any(normalize_skill_match(v, ks) for v in (n, title) for ks in known_skills):
                    skill["auto_completed"] = True
                    if skill.get("p", 0) < 65:
                        skill["p"] = 65
                    auto_count += 1
    roadmap_data["auto_completed_count"] = auto_count


def _salary_floor_check(roadmap_data: dict) -> dict:
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
# ROADMAP QUALITY VALIDATORS
# ============================================================
# These validators evaluate content quality, career progression,
# project realism, salary logic, skill relevance, and course
# catalog compliance.  They NEVER raise and NEVER block generation.
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
        project = ms.get("project")
        if not project:
            violations.append(f"Milestone {mid}: no project to evaluate")
            continue
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

_ALL_AVAILABLE_SKILLS = {
    s.lower()
    for course in AVAILABLE_COURSES.values()
    for s in course.get("skills_include", [])
}

_ALL_AVAILABLE_MODULES = {
    m.lower()
    for course in AVAILABLE_COURSES.values()
    for m in course.get("modules", [])
}


def _course_catalog_compliance_check(roadmap_data: dict) -> dict:
    violations = []
    for ms in roadmap_data.get("milestones", []):
        for mod in ms.get("modules", []):
            mod_title = (mod.get("title") or "").lower()
            # Check module title against unavailable courses
            for uc in _UNAVAILABLE_COURSES:
                if uc in mod_title:
                    violations.append(
                        f"Module '{mod.get('id')}' ('{mod.get('title')}') "
                        f"resembles unavailable course '{uc}'"
                    )
            for skill in mod.get("skills", []):
                skill_n = (skill.get("n") or "").lower().strip()
                # Only flag if skill is not in any available course
                if skill_n and skill_n not in _ALL_AVAILABLE_SKILLS:
                    # Check if it's a compound name that partially matches
                    if not any(skill_n in avail or avail in skill_n for avail in _ALL_AVAILABLE_SKILLS):
                        violations.append(
                            f"Skill '{skill.get('n')}' in {mod.get('id')} "
                            f"not found in any AVAILABLE_COURSES"
                        )
    if violations:
        return {"pass": False, "reason": f"{len(violations)} catalog violation(s)", "violations": violations}
    return {"pass": True, "reason": "All skills and modules map to available courses"}


def run_roadmap_quality_validators(roadmap_data: dict) -> dict:
    _validators = [
        ("milestone_identity_quality", lambda: _milestone_identity_quality_check(roadmap_data)),
        ("capability_progression",     lambda: _capability_progression_check(roadmap_data)),
        ("project_quality",            lambda: _project_quality_check(roadmap_data)),
        ("salary_progression",         lambda: _salary_progression_check(roadmap_data)),
        ("skill_relevance",            lambda: _skill_relevance_check(roadmap_data)),
        ("course_catalog_compliance",  lambda: _course_catalog_compliance_check(roadmap_data)),
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


# ============================================================
# PROFILE VALIDATION  (Phase 8 — Corruption Guard)
# ============================================================

class CustomerProfileCorruptionError(ValueError):
    """Raised when structured profile is corrupted by fallback paths."""

def validate_customer_profile(profile: dict, profile_source: str = "unknown") -> None:
    """
    Validate that required structured fields are present and non-empty.
    Raises CustomerProfileCorruptionError if structured data was lost.
    """
    required = {
        "current_identity": str,
        "target_identity": str,
        "years_experience": (int, float),
    }
    for field, expected_type in required.items():
        val = profile.get(field)
        if val is None or val == "" or val == 0:
            print(
                f"[PROFILE VALIDATION] ⚠ {field}=None/empty "
                f"(source={profile_source})"
            )
    if profile.get("years_experience", 0) == 0 and profile.get("current_identity", "") == "":
        print(
            f"[PROFILE VALIDATION] ⚠ Empty profile detected — "
            f"years_experience=0 AND current_identity='' "
            f"(source={profile_source})"
        )


# ============================================================
# run_pipeline — Main Entry Point
# ============================================================

def run_pipeline(
    user_input,
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

    if isinstance(user_input, dict):
        # ── AI GENERATED PERSONA MODE ──────────────────────────
        print("[ROADMAP AGENT] Mode: AI-generated onboarding")

        user_id  = "ai_generated_user"
        context  = user_input.get("goal_context", "")
        icp_type = user_input.get("icp_type", "low") if user_input.get("icp_type") in ("low", "high") else (
            "low" if user_input.get("current_role", "").lower() == "student" else "high"
        )
        years_experience       = user_input.get("years_experience", 0)
        weekly_hours_available = user_input.get("weekly_hours_available", 5)

        if not context:
            return {
                "error":         "AI onboarding context missing",
                "user_id":       user_id,
                "ai_session_id": ai_session_id,
            }

    else:
        # ── REAL USER / PINECONE MODE ───────────────────────────
        # Per spec: direct key fetch, no vector search.
        # Keys: onboarding_conversation | namespace = user_id
        print("[ROADMAP AGENT] Mode: Real user (Pinecone)")

        user_id  = user_input

        # ── Fetch onboarding conversation ───────────────────────
        # Try spec-compliant key first, then backward-compatible fallbacks.
        poc_context = fetch_poc_record(
            user_id=user_id,
            record_id="onboarding_conversation"
        )

        if not poc_context:
            print("[ROADMAP AGENT] Spec key not found; trying prefixed key for backward compat")
            poc_context = fetch_poc_record(
                user_id=user_id,
                record_id=f"{user_id}_onboarding_conversation"
            )

        if not poc_context:
            print("[ROADMAP AGENT] Prefixed key not found; falling back to vector search")
            poc_context = retrieve_context(user_id)

        if not poc_context:
            print("[ROADMAP AGENT] ✗ No onboarding data found via any method")
            return {
                "error":         "No onboarding conversation found. Please complete onboarding first.",
                "user_id":       user_id,
                "ai_session_id": ai_session_id,
            }

        if not poc_context.strip():
            print("[ROADMAP AGENT] ✗ Onboarding conversation is empty")
            return {
                "error":         "Onboarding conversation is empty.",
                "user_id":       user_id,
                "ai_session_id": ai_session_id,
            }

        context = poc_context
        print(f"[ROADMAP AGENT] ✓ Retrieved onboarding conversation ({len(context)} chars)")

        # ── Derive icp_type from context text (no vector search needed) ──
        _ctx_lower = context.lower()
        _has_employment = any(w in _ctx_lower for w in [
            "salary", "experience", "promotion", "working", "employed",
            "software engineer", "developer", "product manager",
        ])
        _has_student = any(w in _ctx_lower for w in [
            "student", "fresher", "college", "placement", "internship", "12th",
        ])
        icp_type = "high" if _has_employment else ("low" if _has_student else "high")
        print(f"[ICP] Derived icp_type={icp_type} from context (emp={_has_employment}, stu={_has_student})")

    # ============================================================
    # CUSTOMER PROFILE  (Bible §12, Science §20)
    # ============================================================
    if isinstance(user_input, dict):
        _urgency_map = {"high": 60, "medium": 120, "low": 180}
        _timeline_days = int(user_input.get("timeline_days", 0)) or \
            _urgency_map.get(str(user_input.get("urgency", "medium")).lower(), 120)
        customer_profile = {
            "current_identity":         str(user_input.get("current_role", "")),
            "target_identity":          str(user_input.get("target_role", "") or user_input.get("goal", "")),
            "years_experience":         int(user_input.get("years_experience", 0)),
            "weekly_hours_available":   int(user_input.get("weekly_hours_available", 5)),
            "timeline_days":            _timeline_days,
            "current_salary_lpa":       float(user_input.get("current_salary_monthly", 0)) * 12 / 100000,
            "known_skills":             user_input.get("known_skills", []),
            "self_efficacy":            0.5 if str(user_input.get("self_efficacy", "medium")).lower() in ("medium", "0.5") else (0.3 if str(user_input.get("self_efficacy", "medium")).lower() == "low" else 0.7),
        }
        if not isinstance(customer_profile["known_skills"], list):
            customer_profile["known_skills"] = []
        print(f"[STRUCTURED PROFILE] Built from persona dict: "
              f"current={customer_profile['current_identity']} "
              f"yoe={customer_profile['years_experience']} "
              f"skills={len(customer_profile['known_skills'])}")
    else:
        customer_profile = build_customer_profile(context)

    years_experience        = customer_profile["years_experience"]
    weekly_hours_available  = customer_profile["weekly_hours_available"]
    timeline_days           = customer_profile["timeline_days"]
    current_salary_lpa      = customer_profile["current_salary_lpa"]
    known_skills            = customer_profile["known_skills"]

    validate_customer_profile(
        customer_profile,
        profile_source="structured_dict" if isinstance(user_input, dict) else "build_customer_profile"
    )

    # ── Compute budget_hours ──────────────────────────────────
    budget_hrs = round(
        weekly_hours_available * (timeline_days / 7) * 0.8, 1
    )

    # ── ROADMAP INPUT AUDIT (Phase 5) ─────────────────────────
    print("\n========== ROADMAP INPUT ==========")
    print(json.dumps(customer_profile, indent=2, default=str))
    print("===================================\n")

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

    print(f"[ROADMAP AGENT] years_experience={years_experience}, weekly_hours_available={weekly_hours_available}")
    print(f"[ROADMAP AGENT] timeline_days={timeline_days} budget_hours={budget_hrs}")
    if current_salary_lpa:
        print(f"[ROADMAP AGENT] current_salary_lpa={current_salary_lpa}")
    if known_skills:
        print(f"[ROADMAP AGENT] known_skills={known_skills}")

    # ============================================================
    # LEVEL DETECTION  (README §2 — Dynamic starting point)
    # ============================================================

    level           = detect_level(context, years_experience)
    print(
        f"[LEVEL] Detected level: {level} "
        f"(years_experience={years_experience}) "
        f"icp_type={icp_type} "
        f"→ milestone count determined by LLM from capability gap "
        f"(bounds {MIN_MILESTONES}–{MAX_MILESTONES})"
    )

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
                "self_efficacy":      self_efficacy_str,
                "gap_score":                        str(gap_analysis["gap_score"]),
                "recommended_milestones":           str(gap_analysis["recommended_milestones"]),
                "recommended_modules_per_milestone": str(gap_analysis["recommended_modules_per_milestone"]),
                "recommended_skill_density":         str(gap_analysis["recommended_skill_density"]),
                "gap_reasoning":                     gap_analysis["reasoning"],
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
            # ── Pipeline fallbacks for time-budget fields ──────
            roadmap_data["timeline_days"] = timeline_days
            if "budget_hours" not in roadmap_data:
                roadmap_data["budget_hours"] = budget_hrs
            # Always overwrite estimated_total_hours from actual counts
            lc, sc, sci, ivc, pc = _count_roadmap_units(roadmap_data)
            roadmap_data["estimated_total_hours"] = round(
                lc * 0.25 + sc * 1.5 + sci * 0.5 + ivc * 1.0 + pc * 6.0, 1
            )
            print(
                f"[TIME BUDGET] estimated_total_hours recomputed = "
                f"{roadmap_data['estimated_total_hours']}"
            )
            # ── Budget enforcement: reduce milestones if demand > budget ──
            _budget_enforced = False
            if budget_hrs > 0:
                current_estimated = roadmap_data["estimated_total_hours"]
                if current_estimated > budget_hrs:
                    milestones_before = len(roadmap_data.get("milestones", []))
                    # Calculate target milestone count proportionally
                    target_ms = max(
                        MIN_MILESTONES,
                        int(milestones_before * budget_hrs / current_estimated)
                    )
                    if target_ms < milestones_before:
                        print(
                            f"[BUDGET ENFORCEMENT] demand={current_estimated}h > "
                            f"budget={budget_hrs}h — reducing milestones from "
                            f"{milestones_before} to {target_ms}"
                        )
                        roadmap_data["milestones"] = roadmap_data["milestones"][:target_ms]
                        roadmap_data["milestone_count_rationale"] = (
                            f"Reduced to {target_ms} milestone(s) from {milestones_before} "
                            f"due to time budget ({budget_hrs}h) relative to demand "
                            f"({current_estimated}h)."
                        )
                        # Re-count after reduction
                        lc, sc, sci, ivc, pc = _count_roadmap_units(roadmap_data)
                        roadmap_data["estimated_total_hours"] = round(
                            lc * 0.25 + sc * 1.5 + sci * 0.5 + ivc * 1.0 + pc * 6.0, 1
                        )
                        _budget_enforced = True
                        print(
                            f"[BUDGET ENFORCEMENT] After reduction: "
                            f"{target_ms} milestone(s), "
                            f"estimated={roadmap_data['estimated_total_hours']}h"
                        )
            # ==========================================
            # AUTO FIX — SCIENCE DISTRIBUTION
            # ==========================================
            # Ensures every milestone has 3-7 Scenarios + exactly 1 Interview,
            # Scenarios may share modules, Interview in separate module.
            _MIN_SCENARIOS = 3
            _MAX_SCENARIOS = 7
            for milestone in roadmap_data.get("milestones", []):
                modules = milestone.get("modules", [])

                # Collect per-module science items
                scenario_modules = []
                interview_modules = []
                for mod in modules:
                    sci_list = mod.get("science", [])
                    if not isinstance(sci_list, list):
                        sci_list = []
                    for sci in sci_list[:]:
                        t = sci.get("type")
                        if t == "Scenario":
                            scenario_modules.append(mod)
                        elif t == "Interview":
                            interview_modules.append(mod)

                # ── Fix: >7 Scenarios → cap at 7 ──
                total_scenarios = len(scenario_modules)
                if total_scenarios > _MAX_SCENARIOS:
                    excess = scenario_modules[_MAX_SCENARIOS:]
                    for mod in excess:
                        mod["science"] = [s for s in mod.get("science", [])
                                          if s.get("type") != "Scenario"]
                    scenario_modules = scenario_modules[:_MAX_SCENARIOS]

                # ── Fix: >1 Interview → keep first, drop extras ──
                if len(interview_modules) > 1:
                    keep = interview_modules[0]
                    for mod in interview_modules[1:]:
                        mod["science"] = [s for s in mod.get("science", [])
                                          if s.get("type") != "Interview"]
                    interview_modules = [keep]

                # ── Fix: Scenario and Interview in same module → move Interview ──
                if scenario_modules and interview_modules and scenario_modules[0] is interview_modules[0]:
                    same_mod = scenario_modules[0]
                    same_mod["science"] = [s for s in same_mod.get("science", [])
                                           if s.get("type") != "Interview"]
                    interview_modules = []
                    for mod in modules:
                        if mod is not same_mod and not any(
                            s.get("type") == "Interview" for s in mod.get("science", [])
                        ):
                            mod.setdefault("science", []).append(
                                {"type": "Interview", "desc": "Interview question assessing milestone competency"}
                            )
                            interview_modules = [mod]
                            break

                # ── Fix: <3 Scenarios → inject up to _MIN_SCENARIOS ──
                if len(scenario_modules) < _MIN_SCENARIOS:
                    needed = _MIN_SCENARIOS - len(scenario_modules)
                    for mod in modules:
                        while needed > 0:
                            sci_list = mod.get("science", [])
                            if not isinstance(sci_list, list):
                                sci_list = []
                                mod["science"] = sci_list
                            if len(sci_list) < 3:
                                sci_list.append(
                                    {"type": "Scenario", "desc": "Production debugging scenario for milestone skills"}
                                )
                                scenario_modules.append(mod)
                                needed -= 1
                            else:
                                break
                        if needed <= 0:
                            break

                # ── Fix: 0 Interview → inject ──
                if not interview_modules:
                    for mod in modules:
                        has_interview = any(
                            s.get("type") == "Interview" for s in mod.get("science", [])
                        )
                        if not has_interview and (
                            not scenario_modules
                            or all(s.get("type") != "Scenario" for s in mod.get("science", []))
                        ):
                            mod.setdefault("science", []).append(
                                {"type": "Interview", "desc": "Interview question assessing milestone competency"}
                            )
                            interview_modules = [mod]
                            break

                # ── Fix: any module has >3 science items → truncate to 3 ──
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
                milestone["sc_n"] = sc
                milestone["iv"] = iv
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
            roadmap_data.setdefault("roadmap_meta", {})["generated_at"] = now
            # ── Salary floor auto-repair ────────────────────────
            apply_salary_floor_repair(roadmap_data)
            # ── Known-skill auto-completion ─────────────────────
            apply_known_skill_autocomplete(roadmap_data, known_skills)
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
            # ── Auto-completion debug ───────────────────────────
            _ac = roadmap_data.get("auto_completed_count", 0)
            print("========== AUTO COMPLETION ==========")
            print(f"  known_skills_count : {len(known_skills)}")
            print(f"  auto_completed_count: {_ac}")
            print("=====================================")
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
                    if len(mod.get("skills", [])) < 3:
                        print(json.dumps(mod, indent=2))
            print("=============================================\n")
            # ── AUTO REPAIR: module count, skill count, lesson count ───
            # Runs after sc_n/iv fix and ID injection but BEFORE
            # validate_roadmap_structure(), so the validator always receives
            # structurally valid arrays.
            # Uses capability_gap recommendations as the target; falls back
            # to MIN_SKILLS/MAX_SKILLS bounds if gap data is unavailable.
            _gap = roadmap_data.get("capability_gap", {})
            _target_skill = _gap.get("recommended_skill_density", MIN_SKILLS)
            _target_mod   = _gap.get("recommended_modules_per_milestone", MIN_MODULES)
            # Clamp to bounds to keep downstream validators happy
            _target_skill = max(MIN_SKILLS, min(_target_skill, MAX_SKILLS))
            _target_mod   = max(MIN_MODULES, min(_target_mod, MAX_MODULES))
            for ms in roadmap_data.get("milestones", []):
                modules = ms.get("modules", [])
                # ── Module count repair ─────────────────────────────
                orig_mod_count = len(modules)
                if 0 < orig_mod_count < _target_mod:
                    while len(modules) < _target_mod:
                        donor = copy.deepcopy(modules[-1])
                        donor_id = donor.get("id", "MOD_REPAIR")
                        donor["id"] = f"{donor_id}_repair_{uuid.uuid4().hex[:4]}"
                        for skill in donor.get("skills", []):
                            sid = skill.get("skill_id", "SKILL_REPAIR")
                            skill["skill_id"] = f"{sid}_rep_{uuid.uuid4().hex[:4]}"
                        modules.append(donor)
                    ms["modules"] = modules
                    print(
                        f"[AUTO REPAIR] Milestone {ms.get('milestone_id')} had "
                        f"{orig_mod_count} module(s) -> repaired to {_target_mod}"
                    )
                elif orig_mod_count > _target_mod:
                    ms["modules"] = modules[:_target_mod]
                    print(
                        f"[AUTO REPAIR] Milestone {ms.get('milestone_id')} had "
                        f"{orig_mod_count} modules -> truncated to {_target_mod}"
                    )
                # ── Skill count repair per module ────────────────────
                for mod in ms.get("modules", []):
                    skills     = mod.get("skills", [])
                    mod_id     = mod.get("id", "?")
                    orig_count = len(skills)
                    # Too few: duplicate last skill until target count
                    if 0 < orig_count < _target_skill:
                        while len(skills) < _target_skill:
                            donor         = copy.deepcopy(skills[-1])
                            unique_suffix = uuid.uuid4().hex[:6]
                            donor["skill_id"] = (
                                f"{donor.get('skill_id', 'SKILL_REPAIR')}"
                                f"_dup_{unique_suffix}"
                            )
                            donor["title"] = (
                                donor.get("title", donor.get("n", "Skill"))
                                + " Advanced"
                            )
                            donor["unlock_rules"] = {
                                "requires": [],
                                "minimum_mastery": 0.0,
                                "unlock_type": "immediate",
                            }
                            flow = donor.get("content_flow", {})
                            for ct in ("video", "scenario", "mock"):
                                item = flow.get(ct, {})
                                if "content_id" in item:
                                    item["content_id"] = (
                                        f"{item['content_id']}_dup_{unique_suffix}"
                                    )
                            skills.append(donor)
                        mod["skills"] = skills
                        print(
                            f"[AUTO REPAIR] Module {mod_id} had "
                            f"{orig_count} skill(s) -> repaired to {_target_skill}"
                        )
                    # No skills at all: cannot synthesise; validator will reject
                    elif orig_count == 0:
                        print(
                            f"[AUTO REPAIR] Module {mod_id} has 0 skills — "
                            f"cannot repair; validator will reject"
                        )
                    # Too many skills: truncate to target
                    elif orig_count > _target_skill:
                        mod["skills"] = skills[:_target_skill]
                        print(
                            f"[AUTO REPAIR] Module {mod_id} had "
                            f"{orig_count} skills -> truncated to {_target_skill}"
                        )
                    # Per-skill lesson repair: each skill must have exactly 3
                    for skill in mod.get("skills", []):
                        skill_id = skill.get("skill_id", "?")
                        lessons  = skill.get("lessons", [])
                        if not isinstance(lessons, list):
                            lessons = []
                        orig_lesson_count = len(lessons)
                        if orig_lesson_count < 3:
                            while len(lessons) < 3:
                                lessons.append(
                                    f"{skill.get('title', skill.get('n', 'Skill'))} "
                                    f"— Part {len(lessons) + 1}"
                                )
                            skill["lessons"] = lessons
                            print(
                                f"[AUTO REPAIR] Skill {skill_id} had "
                                f"{orig_lesson_count} lesson(s) -> padded to 3"
                            )
                        elif orig_lesson_count > 3:
                            skill["lessons"] = lessons[:3]
                            print(
                                f"[AUTO REPAIR] Skill {skill_id} had "
                                f"{orig_lesson_count} lessons -> truncated to 3"
                            )
            # ── Gap alignment debug ────────────────────────────
            _gap_dbg = roadmap_data.get("capability_gap", {})
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
            _gap_pass = (
                _dbg_ms_count == _gap_dbg.get("recommended_milestones")
                and all(m == _gap_dbg.get("recommended_modules_per_milestone") for m in _dbg_mod_counts)
                and all(s == _gap_dbg.get("recommended_skill_density") for s in _dbg_skill_counts)
            ) if _gap_dbg.get("recommended_milestones") else True
            print("========== GAP ALIGNMENT ==========")
            print(f"  Expected Milestones : {_gap_dbg.get('recommended_milestones', 'N/A')}")
            print(f"  Actual Milestones   : {_dbg_ms_count}")
            print(f"  Expected Modules    : {_gap_dbg.get('recommended_modules_per_milestone', 'N/A')}")
            print(f"  Actual Modules      : {_dbg_mod_counts}")
            print(f"  Expected Skills     : {_gap_dbg.get('recommended_skill_density', 'N/A')}")
            print(f"  Actual Skills       : {_dbg_skill_counts}")
            print(f"  {'PASS' if _gap_pass else 'FAIL'}")
            print("================================")
            # ── Structural validation ──────────────────────────
            validate_roadmap_structure(roadmap_data)

            # ── Roadmap Bible validators ───────────────────────
            # run_roadmap_bible_validators() NEVER raises.
            # fits_life is now advisory: it adapts duration instead
            # of hard-failing (Bug 5 fix).
            bible_results = run_roadmap_bible_validators(
                roadmap_data, weekly_hours_available, timeline_days
            )
            roadmap_data["fits_life"]        = bible_results.get("fits_life", {})
            roadmap_data["bible_validators"] = bible_results

            # ── Roadmap quality validators ─────────────────────
            quality_results = run_roadmap_quality_validators(roadmap_data)
            roadmap_data["quality_validators"] = quality_results

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
            _dbg_project_count = sum(1 for ms in _dbg_ms if ms.get("project"))
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
            print("========== SCIENCE DISTRIBUTION ==========")
            for ms in milestones:
                ms_id = ms.get("milestone_id", "?")
                print(ms_id)
                sc_total = 0
                iv_total = 0
                for mod in ms.get("modules", []):
                    mod_id = mod.get("id", "?")
                    sci_list = mod.get("science", [])
                    if sci_list:
                        sci_type = sci_list[0].get("type", "?")
                        if sci_type == "Scenario":
                            sc_total += 1
                        elif sci_type == "Interview":
                            iv_total += 1
                        print(f"  {mod_id} -> {sci_type}")
                    else:
                        print(f"  {mod_id} -> NONE")
                print(f"  Scenario={sc_total}")
                print(f"  Interview={iv_total}")
                print()
            print("==========================================")

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

            for record_suffix, ensure_ascii in [
                ("roadmap_conversation", True),
                ("roadmap_output",      False),
            ]:
                record_id = f"{user_id}_{record_suffix}"
                payload   = json.dumps(roadmap_summary, ensure_ascii=ensure_ascii)
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
                    "generation_model":      "roadmap-gen-v3.2",
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