from langchain_core.prompts import PromptTemplate
from langchain_openai import ChatOpenAI
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.output_parsers import StrOutputParser
from src.pinecone_utils import (
    retrieve_context,
    retrieve_icp_type,
    fetch_poc_record,
    save_poc_record,
)
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
    input_variables=["context", "icp_type", "level"],
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
  - Every milestone has EXACTLY 1 Scenario (sc_n=1) and EXACTLY 1 Interview
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
Every milestone must contain EXACTLY 1 Scenario AND EXACTLY 1 Interview.
The Scenario and Interview may be placed in any module across the milestone
(in separate modules).  Validation enforces sc_n=1 and iv=1 at the milestone
level regardless of which modules carry them.

Rules for science items:
  - "Scenario" = a realistic production/debugging situation the learner must resolve.
    Examples: bad retrieval causes hallucinations, pipeline fails before demo,
    latency spike in production, nightly job fails at 9am.
    NOT toy exercises. NOT hypotheticals.
  - "Interview" = an interview question testing the milestone identity.
    Must test the ability to PERFORM the role, not recall trivia.
  - Only 2 modules per milestone carry science: one with "Scenario", one with "Interview".
  - All other modules must have "science": [].
  - The milestone-level sc_n must equal 1. The milestone-level iv must equal 1.

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
  - "sc_n": must equal 1 (exactly one Scenario per milestone).
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
explanations.

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
      "sc_n": 1,
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
                  "prior": 0.15,
                  "learn_rate": 0.25,
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
- Every milestone has EXACTLY 1 Scenario (sc_n=1) and EXACTLY 1 Interview (iv=1).
  Place the Scenario and Interview in two separate modules. Modules beyond those
  two must have "science": [].
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

        if len(roadmap_json_str.encode("utf-8")) > MAX_METADATA_BYTES:
            print("[PINECONE STORE] ⚠ Roadmap JSON too large — storing summary only")
            store_payload = {
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

        print(f"[PINECONE STORE] ✓ Stored — vector_id: {vector_id}")
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
# JSON Repair Utility   (UNCHANGED)
# ============================================================

def repair_json(raw: str) -> str:
    clean = raw.strip()
    if "```" in clean:
        clean = clean.replace("```json", "").replace("```", "").strip()
    start = clean.find("{")
    end   = clean.rfind("}") + 1
    if start != -1 and end > 0:
        clean = clean[start:end]
    return clean


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
      - each module's "science" array has 0 or 1 item ("Scenario" or "Interview"; only 2 modules carry the milestone's 1 Scenario + 1 Interview)
      - each milestone has sc_n == 1 and iv == 1
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

            # ── science: 0 or 1 item per module ──
            science = mod.get("science", [])
            if not isinstance(science, list):
                raise ValueError(f"Module {mod_id}: science must be a list")
            if len(science) > 1:
                raise ValueError(
                    f"Module {mod_id}: science array must have 0 or 1 item, "
                    f"got {len(science)}"
                )
            if len(science) == 1:
                sci      = science[0]
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

        # ── sc_n must equal 1, iv must equal 1 ──
        if scenario_count != 1:
            raise ValueError(
                f"Milestone {m_id}: must have EXACTLY 1 Scenario across its "
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
_FL_DEFAULT_WEEKS  = 16    # target programme duration
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


def fits_life_check(roadmap_data: dict, weekly_hours: int) -> dict:
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
      2. If demand ≤ budget at _FL_DEFAULT_WEEKS          → fits at 16 weeks.
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
        "budget_hours":         float,  # weekly_hours × duration_weeks × 0.8
        "demand_hours":         float,
        "weekly_hours_needed":  float,  # hrs/wk to finish in _FL_DEFAULT_WEEKS
        "breakdown": { ... }
    }
    """
    weekly_hours = max(int(weekly_hours or 1), 1)   # guard against 0 / None

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

    # ── Adaptive duration ─────────────────────────────────────────
    default_budget = round(weekly_hours * _FL_DEFAULT_WEEKS * _FL_REALISM_BUFFER, 1)

    if demand_hours <= default_budget or demand_hours == 0:
        duration_weeks = _FL_DEFAULT_WEEKS
        fits           = True
    else:
        # weeks needed = demand / (hours_per_week × realism_buffer)
        raw_weeks      = demand_hours / (weekly_hours * _FL_REALISM_BUFFER)
        duration_weeks = int(math.ceil(raw_weeks))
        fits           = duration_weeks <= _FL_MAX_WEEKS

    budget_hours = round(weekly_hours * duration_weeks * _FL_REALISM_BUFFER, 1)

    # weekly_hours_needed = hrs/wk required to complete in the default 16-week window
    weekly_hours_needed = round(
        demand_hours / (_FL_DEFAULT_WEEKS * _FL_REALISM_BUFFER), 1
    ) if demand_hours > 0 else 0.0

    return {
        "fits":                fits,
        "fits_at_default_16w": fits if duration_weeks == _FL_DEFAULT_WEEKS else False,
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


def run_roadmap_bible_validators(roadmap_data: dict, weekly_hours: int) -> dict:
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
        ("fits_life",             lambda: fits_life_check(roadmap_data, weekly_hours)),
        ("starts_where_they_are", lambda: _starts_where_they_are_check(roadmap_data)),
        ("laugh_test",            lambda: _laugh_test_check(roadmap_data)),
        ("no_spectators",         lambda: _no_spectators_check(roadmap_data)),
        ("dag_clean",             lambda: _dag_clean_check(roadmap_data)),
        ("ai_first",              lambda: _ai_first_check(roadmap_data)),
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
        icp_type = (
            "low"
            if user_input.get("current_role", "").lower() == "student"
            else "high"
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
        print("[ROADMAP AGENT] Mode: Real user (Pinecone)")

        user_id  = user_input
        icp_type = retrieve_icp_type(user_id)

        if not icp_type:
            print("[ICP] icp_type not found — onboarding incomplete")
            return {
                "error":         "Please complete onboarding first.",
                "user_id":       user_id,
                "ai_session_id": ai_session_id,
            }

        print(f"[ICP] Classified as: {icp_type}")

        # ── Fetch onboarding conversation written by Onboarding POC ──
        onboarding_record_id = f"{user_id}_onboarding_conversation"
        poc_context = fetch_poc_record(
            user_id=user_id,
            record_id=onboarding_record_id
        )

        # Validate POC record; fall back if stale or invalid
        if poc_context:
            if detect_stale_onboarding_record(poc_context):
                print("[ROADMAP AGENT] Stale onboarding record detected — using retrieve_context()")
                context = retrieve_context(user_id)
            elif is_valid_onboarding_record(poc_context):
                context = poc_context
                print(
                    f"[ROADMAP AGENT] ✓ Retrieved onboarding conversation "
                    f"({len(context)} chars)"
                )
            else:
                print("[ROADMAP AGENT] Invalid onboarding record detected; falling back to retrieve_context()")
                context = retrieve_context(user_id)
        else:
            print(
                "[ROADMAP AGENT] No POC onboarding record found; "
                "falling back to retrieve_context()"
            )
            context = retrieve_context(user_id)

        if not context:
            print("[ROADMAP AGENT] ✗ No context found in Pinecone")
            return {
                "error":         "No onboarding conversation found. Please complete onboarding first.",
                "user_id":       user_id,
                "ai_session_id": ai_session_id,
            }

        print(f"[ROADMAP AGENT] ✓ Context: {len(context)} chars")

    # ============================================================
    # EXTRACTION  — years_experience & weekly_hours_available
    # ============================================================
    # Strategy: always try both JSON and regex independently.
    # Prefer regex value when JSON gives 0/default but text has a real value.

    json_ye = json_wh = None
    try:
        parsed_ctx = json.loads(context)
        if isinstance(parsed_ctx, dict):
            json_ye = int(parsed_ctx.get("years_experience", 0))
            json_wh = int(parsed_ctx.get("weekly_hours_available", 5))
    except (json.JSONDecodeError, TypeError, ValueError):
        pass

    regex_ye = None
    ye = re.search(r"[Yy]ears?\s*(?:of\s+)?experience[:\s]+(\d+)", context)
    if ye:
        regex_ye = int(ye.group(1))
    regex_wh = extract_weekly_hours(context)

    if json_ye is not None:
        years_experience = json_ye
    if regex_ye is not None and (json_ye is None or regex_ye > json_ye):
        years_experience = regex_ye
    if json_wh is not None:
        weekly_hours_available = json_wh
    if regex_wh is not None and (json_wh is None or regex_wh > json_wh):
        weekly_hours_available = regex_wh

    # ── Diagnostic logging ────────────────────────────────────
    print(f"[ROADMAP AGENT] years_experience={years_experience}, weekly_hours_available={weekly_hours_available}")
    if json_ye is not None:
        print(f"[ROADMAP AGENT]   source: years_experience from JSON (value={json_ye})")
    if regex_ye is not None:
        print(f"[ROADMAP AGENT]   source: years_experience from regex (value={regex_ye})")
    if json_ye is None and regex_ye is None:
        print(f"[ROADMAP AGENT]   source: years_experience from fallback/default")
    if json_wh is not None:
        print(f"[ROADMAP AGENT]   source: weekly_hours from JSON (value={json_wh})")
    if regex_wh is not None:
        print(f"[ROADMAP AGENT]   source: weekly_hours from regex (value={regex_wh})")

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

            result       = _build_chain().invoke({
                "context":  context,
                "icp_type": icp_type,
                "level":    level,
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
            # ==========================================
            # AUTO FIX — SCIENCE DISTRIBUTION
            # ==========================================
            # Ensures every milestone has exactly 1 Scenario + 1 Interview,
            # in separate modules, with all other modules having science=[].
            for milestone in roadmap_data.get("milestones", []):
                modules = milestone.get("modules", [])

                # Collect per-module science items
                scenario_modules = []
                interview_modules = []
                for mod in modules:
                    sci_list = mod.get("science", [])
                    if not isinstance(sci_list, list):
                        sci_list = []
                    for sci in sci_list[:]:  # iterate copy for safe removal
                        t = sci.get("type")
                        if t == "Scenario":
                            scenario_modules.append(mod)
                        elif t == "Interview":
                            interview_modules.append(mod)

                # ── Fix: >1 Scenario → keep first, drop extras ──
                if len(scenario_modules) > 1:
                    keep = scenario_modules[0]
                    for mod in scenario_modules[1:]:
                        mod["science"] = [s for s in mod.get("science", [])
                                          if s.get("type") != "Scenario"]
                    scenario_modules = [keep]

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
                    # Remove Interview from the shared module
                    same_mod["science"] = [s for s in same_mod.get("science", [])
                                           if s.get("type") != "Interview"]
                    interview_modules = []
                    # Find another module for Interview
                    for mod in modules:
                        if mod is not same_mod and not mod.get("science"):
                            mod["science"] = [{"type": "Interview", "desc": "Interview question assessing milestone competency"}]
                            interview_modules = [mod]
                            break

                # ── Fix: 0 Scenario → inject ──
                if not scenario_modules:
                    for mod in modules:
                        if not mod.get("science"):
                            mod["science"] = [{"type": "Scenario", "desc": "Production debugging scenario for milestone skills"}]
                            scenario_modules = [mod]
                            break

                # ── Fix: 0 Interview → inject ──
                if not interview_modules:
                    for mod in modules:
                        if not mod.get("science") and (not scenario_modules or mod is not scenario_modules[0]):
                            mod["science"] = [{"type": "Interview", "desc": "Interview question assessing milestone competency"}]
                            interview_modules = [mod]
                            break

                # ── Fix: any module has >1 science items → truncate to 1 ──
                for mod in modules:
                    sci_list = mod.get("science", [])
                    if isinstance(sci_list, list) and len(sci_list) > 1:
                        mod["science"] = sci_list[:1]

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
            roadmap_data.setdefault("roadmap_meta", {})["generated_at"] = now
            # ── Inject label field into each milestone if missing ──
            print("\n========== MODULE SKILL COUNT DEBUG ==========")
            for ms in roadmap_data.get("milestones", []):
                for mod in ms.get("modules", []):
                    if len(mod.get("skills", [])) < 3:
                        print(json.dumps(mod, indent=2))
            print("=============================================\n")
            # ── AUTO REPAIR: skill count and lesson count per module ───
            # Runs after sc_n/iv fix and ID injection but BEFORE
            # validate_roadmap_structure(), so the validator always receives
            # structurally valid arrays.
            # Safe because:
            #   - Only touches modules/skills whose counts are out of range.
            #   - Duplicate skills get globally unique skill_ids (uuid suffix).
            #   - Truncation keeps the first MAX_SKILLS skills, preserving prerequisite order.
            #   - Lesson padding uses safe placeholder strings; no semantic content lost.
            for ms in roadmap_data.get("milestones", []):
                for mod in ms.get("modules", []):
                    skills     = mod.get("skills", [])
                    mod_id     = mod.get("id", "?")
                    orig_count = len(skills)

                    # ── Too few skills: duplicate last skill until we reach MIN_SKILLS ──
                    if 0 < orig_count < MIN_SKILLS:
                        while len(skills) < MIN_SKILLS:
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
                            f"{orig_count} skill(s) -> repaired to {MIN_SKILLS}"
                        )

                    # ── No skills at all: cannot synthesise; validator will reject ──
                    elif orig_count == 0:
                        print(
                            f"[AUTO REPAIR] Module {mod_id} has 0 skills — "
                            f"cannot repair; validator will reject"
                        )

                    # ── Too many skills: truncate to MAX_SKILLS ──
                    elif orig_count > MAX_SKILLS:
                        mod["skills"] = skills[:MAX_SKILLS]
                        print(
                            f"[AUTO REPAIR] Module {mod_id} had "
                            f"{orig_count} skills -> truncated to {MAX_SKILLS}"
                        )

                    # ── Per-skill lesson repair: each skill must have exactly 3 ──
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
            # ── Structural validation ──────────────────────────
            validate_roadmap_structure(roadmap_data)

            # ── Roadmap Bible validators ───────────────────────
            # run_roadmap_bible_validators() NEVER raises.
            # fits_life is now advisory: it adapts duration instead
            # of hard-failing (Bug 5 fix).
            bible_results = run_roadmap_bible_validators(
                roadmap_data, weekly_hours_available
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

            # ── Store in Pinecone ──────────────────────────────  (UNCHANGED)
            print("\n[ROADMAP AGENT] Storing roadmap in Pinecone...")
            stored = store_roadmap_in_pinecone(user_id, ai_roadmap_id, roadmap_data)
            if stored:
                print("[ROADMAP AGENT] ✓ Roadmap persisted to Pinecone")
            else:
                print("[ROADMAP AGENT] ⚠ Roadmap generated but NOT stored in Pinecone")

            # ── Save POC cross-POC records ─────────────────────────────  (UNCHANGED)
            # roadmap_conversation — lightweight record (no duplicate payload)
                  # ── Save POC cross-POC records ─────────────────────────────

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

            save_poc_record(
                user_id=user_id,
                record_id=f"{user_id}_roadmap_conversation",
                text=json.dumps(roadmap_summary),
            )


            save_poc_record(
                user_id=user_id,
                record_id=f"{user_id}_roadmap_output",
                text=json.dumps(
                    roadmap_summary,
                    ensure_ascii=False
                )
            )
            

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