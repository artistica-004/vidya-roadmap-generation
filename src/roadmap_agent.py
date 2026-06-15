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
import os
import json
import uuid
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
# LEVEL DETECTION  (README §2 — Dynamic starting point)
# ============================================================

LEVEL_MILESTONE_COUNT = {
    "beginner": 7,
    "intermediate": 5,
    "senior": 3,
}

# Maps level -> the "absolute library tier" that the user's relative M01 maps to.
# Beginner M01 = library tier 1 (Intern-ready)
# Intermediate M01 = library tier 3 (Working engineer)
# Senior M01 = library tier 5 (Senior engineer)
LEVEL_STARTING_TIER = {
    "beginner": 1,
    "intermediate": 3,
    "senior": 5,
}


def detect_level(context: str, years_experience: int = 0) -> str:
    """
    Maps onboarding signals (resume parse / years of experience) to
    beginner | intermediate | senior, per README §2 starting-point table.

      Beginner     -> 0 years   -> Library M01 -> 7 milestones shown
      Intermediate -> 1-3 years -> Library M03 -> 5 milestones shown
      Senior       -> 4+ years  -> Library M05 -> 3 milestones shown
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
# Roadmap Prompt — V3.1 Structure (HTML-aligned, level-driven)
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
You are an expert AI career roadmap architect for Vidya V3.

Generate a deeply personalized career roadmap for this specific user, matching the
Vidya V3 frontend manifest structure (vidya_v3_roadmap_full.html) exactly.

USER CONTEXT:
{context}

USER ICP TYPE:
{icp_type}

USER LEVEL (detected from onboarding — resume parse + voice + behavioral signals):
{level}

=== GENERATION RULES (STRICT — ALL MUST BE FOLLOWED) ===

RULE 1 — HIERARCHY (LOCKED):
Roadmap -> Milestone -> Module -> (Lessons + Skills + Science)
Never break this hierarchy.

RULE 2 — MILESTONE COUNT (LEVEL-DRIVEN, HTML SPEC — README §2):
Milestone count and codes depend ONLY on the USER LEVEL above:

  level = "beginner"     -> generate EXACTLY 7 milestones, codes M01-M07
  level = "intermediate" -> generate EXACTLY 5 milestones, codes M01-M05
  level = "senior"       -> generate EXACTLY 3 milestones, codes M01-M03

This is non-negotiable. Do not generate fewer or more milestones than the count
for the given level.

RULE 3 — RELATIVE MILESTONE CODES (CRITICAL — README §2):
Milestone codes ALWAYS start at M01 for THIS user, regardless of their level.
M01 does NOT mean "intern-level content" for every user — it means
"this user's current starting milestone".

  - If level = "beginner": M01 content difficulty = intern-ready (library tier 1).
  - If level = "intermediate": M01 content difficulty = working-engineer
    (library tier 3) — i.e. SKIP intern/junior-level content entirely.
  - If level = "senior": M01 content difficulty = senior-engineer
    (library tier 5) — i.e. SKIP intern/junior/working/mid-level content entirely.

Each subsequent milestone (M02, M03, ...) increases in seniority/difficulty by
exactly one library tier from the previous milestone. Never serve content below
the user's detected ZPD floor (mastery < 0.30 means too easy -> skip it).

RULE 4 — MODULE COUNT PER MILESTONE (VARIABLE, HTML-ALIGNED):
Each milestone must have 1 to 3 modules (inclusive). Earlier/foundational
milestones typically have more modules (2-3); senior/leadership milestones
typically have fewer (1-2). Choose a count appropriate to the milestone's
real-world scope — do not pad with filler modules.

RULE 5 — LESSONS PER MODULE (VARIABLE, HTML-ALIGNED):
Each module must have 4 to 8 lesson titles in its "lessons" array. Lessons are
short topic titles (e.g. "Pydantic models", "OOP patterns") representing
individual videos.

RULE 6 — SKILLS PER MODULE (VARIABLE, HTML-ALIGNED):
Each module must have 3 to 4 skill objects in its "skills" array. Each skill
object has:
  - "skill_id": globally unique id, e.g. "SKILL_M01_M1_S1"
  - "n": short skill name (snake_case, e.g. "python", "fastapi", "rag")
  - "title": human-readable skill title
  - "p": initial mastery percentage (0-100).
       * If the module is "free": true AND is part of the user's M01
         (i.e. foundational/already-known content per onboarding mastery
         priors), set "p" to a realistic prior between 55-90.
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

RULE 7 — SCIENCE ARRAY (SCENARIO / INTERVIEW — HTML-ALIGNED):
Each module has a "science" array with 0 or 1 item (never more than one).
  - If present, item shape: {{"type": "Scenario", "desc": "..."}}
    OR {{"type": "Interview", "desc": "..."}}
  - "Scenario" = a realistic debugging/decision scenario tied to the module's
    skills (retrieval-practice style, embedded mid-video per README §5).
  - "Interview" = a realistic interview-style question testing the module's
    skills.
  - Not every module needs a science entry — leave "science": [] if it does
    not naturally fit.

RULE 8 — MODULE METADATA (HTML-ALIGNED):
Each module must include:
  - "id": HTML-style id "M{{milestone_number}}.{{module_number}}", e.g. "M1.1",
    "M1.2", "M3.1" (module numbering restarts at .1 for each milestone, and
    the milestone number in the module id matches the milestone's position,
    1-indexed, NOT the M01/M02 zero-padded label).
  - "title": short module title (from AVAILABLE_COURSES modules list where
    possible).
  - "free": true ONLY for modules inside the user's M01 (first 1-2 modules of
    M01 only, matching README §7 "Free vs locked" rule). All other modules:
    false.
  - "vis": visualization type, one of: "code+real_tutor", "code+ppt",
    "ppt+animation", "ppt+code", "real_tutor+ppt", "animation+code",
    "notebook+code", "notebook+ppt", "real_tutor+code".

RULE 9 — MILESTONE METADATA (HTML-ALIGNED):
Each milestone must include:
  - "milestone_id": "M01".."M07" (zero-padded, relative per RULE 3)
  - "label": same value as milestone_id
  - "t": short title for this career level, e.g. "Working AI engineer"
  - "sal": salary/earning string for this milestone.
       * ICP-A (student/fresher, icp_type="low"): use Indian rupee LPA tiers,
         e.g. "Unpaid/stipend", "₹3-5 LPA", "₹6-9 LPA", etc. M01 for a true
         beginner student MUST show "Unpaid/stipend" (never a salary).
       * ICP-B (working professional, icp_type="high"): same LPA-tier system
         but starting points are HIGHER (their M01 != student's M01 in
         salary), reflecting existing work experience.
  - "o": one-sentence outcome statement describing what the user can DO after
    this milestone (e.g. "You build REST APIs with FastAPI, handle auth, and
    write unit tests confidently.")
  - "sc_n": total count of "Scenario" items across this milestone's modules'
    science arrays.
  - "iv": total count of "Interview" items across this milestone's modules'
    science arrays.
  - "identity_statement": 1-sentence motivational framing of this milestone
    (Possible Selves science model).
  - "checkpoint_rule": {{"required_mastery": 0.9, "checkpoint_type":
    "mock_interview"}}
  - "modules": array per RULE 4-8.

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
  "target_role": "string — what the user wants to become",
  "language": "en",
  "starting_milestone": "M01",
  "current_active_milestone": "M01",
  "vision_profile": {{
    "current_state": "string — 1 sentence: where the user is right now",
    "main_blocker": "string — 1 sentence: their biggest obstacle",
    "top_motivation": "string — 1 sentence: why they want this"
  }},
  "roadmap_meta": {{
    "generated_at": "PLACEHOLDER_TIMESTAMP",
    "version": "v3.1",
    "science_model": ["ZPD", "Mastery Learning", "CLT", "BKT", "Possible Selves", "Retrieval Practice"]
  }},
  "milestones": [
    {{
      "milestone_id": "M01",
      "label": "M01",
      "t": "string — career-level title, e.g. Intern-ready",
      "sal": "string — salary/earning tier, e.g. Unpaid/stipend or ₹6-9 LPA",
      "o": "string — 1-sentence outcome statement",
      "sc_n": 0,
      "iv": 0,
      "identity_statement": "string — 1-sentence motivational statement",
      "checkpoint_rule": {{
        "required_mastery": 0.9,
        "checkpoint_type": "mock_interview"
      }},
      "modules": [
        {{
          "id": "M1.1",
          "title": "string — module title",
          "free": true,
          "vis": "code+real_tutor",
          "lessons": ["string", "string", "string", "string"],
          "skills": [
            {{
              "skill_id": "SKILL_M01_M1_S1",
              "n": "python",
              "title": "string — human readable skill title",
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
            {{"type": "Scenario", "desc": "string — realistic debugging/decision scenario"}}
          ]
        }}
      ]
    }}
  ]
}}

IMPORTANT REMINDERS:
- All content_id values must be globally unique (e.g. VID_M01_M1_S1, SCN_M02_M1_S2).
- All skill_id values must be globally unique (e.g. SKILL_M01_M1_S1).
- skill unlock_rules.requires must reference skill_ids that appear EARLIER in
  the roadmap (no forward or circular references).
- The FIRST skill of the entire roadmap must have requires: [] and
  unlock_type: "immediate".
- Milestone count and codes MUST match RULE 2 exactly for the given level.
- Each milestone's "sc_n" and "iv" must equal the actual counts of
  "Scenario"/"Interview" entries summed across that milestone's modules'
  "science" arrays.
- "science" arrays have 0 or 1 items only — never more than one per module.
- Each milestone object must include a "label" field equal to its
  milestone_id (e.g. "label": "M01").
- Return ONLY raw JSON. Nothing else.
"""
)

roadmap_chain = roadmap_prompt | llm | StrOutputParser()

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
    Validates the V3.1 (HTML-aligned) roadmap structure.
    Raises ValueError with a clear message on any violation.
    Auto-fixes mock.unlock_mastery silently.

    Enforces (per README §2 + frontend manifest):
      - milestone count = 7 / 5 / 3 depending on data["level"]
        (beginner / intermediate / senior)
      - milestone_id/label codes are M01..M0N, sequential, no gaps
      - each milestone has 1-3 modules
      - each module has 3-4 skills and 4-8 lessons
      - each module's "science" array has at most 1 item, each of type
        "Scenario" or "Interview"
      - each milestone's sc_n/iv match the actual science-array counts
      - skill_id uniqueness and acyclic/backward-only prerequisite refs
    """
    milestones = data.get("milestones", [])
    level      = data.get("level", "beginner")

    if level not in LEVEL_MILESTONE_COUNT:
        raise ValueError(
            f"Invalid level '{level}' — must be one of "
            f"{list(LEVEL_MILESTONE_COUNT.keys())}"
        )

    if not milestones:
        raise ValueError("Roadmap has no milestones")

    expected_ms = LEVEL_MILESTONE_COUNT[level]
    if len(milestones) != expected_ms:
        raise ValueError(
            f"Wrong milestone count: got {len(milestones)}, "
            f"expected {expected_ms} for level='{level}'"
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

        modules = milestone.get("modules", [])
        if not isinstance(modules, list):
            raise ValueError(f"Milestone {m_id}: modules must be a list")
        if not (1 <= len(modules) <= 3):
            raise ValueError(
                f"Milestone {m_id}: must have 1-3 modules, got {len(modules)}"
            )

        scenario_count = 0
        interview_count = 0

        for mod_idx, mod in enumerate(modules):
            mod_id = mod.get("id", "?")

            # ── module id format check: "M{milestone_pos}.{module_pos}" ──
            expected_mod_id = f"M{m_idx + 1}.{mod_idx + 1}"
            if mod_id != expected_mod_id:
                raise ValueError(
                    f"Milestone {m_id}: module #{mod_idx + 1} id must be "
                    f"'{expected_mod_id}', got '{mod_id}'"
                )

            # ── lessons ──
            lessons = mod.get("lessons", [])
            if not isinstance(lessons, list):
                raise ValueError(f"Module {mod_id}: lessons must be a list")
            if not (4 <= len(lessons) <= 8):
                raise ValueError(
                    f"Module {mod_id}: must have 4-8 lessons, got {len(lessons)}"
                )

            # ── skills ──
            skills = mod.get("skills", [])
            if not isinstance(skills, list):
                raise ValueError(f"Module {mod_id}: skills must be a list")
            if not (3 <= len(skills) <= 4):
                raise ValueError(
                    f"Module {mod_id}: must have 3-4 skills, got {len(skills)}"
                )

            # ── science (scenario/interview) ──
            science = mod.get("science", [])
            if not isinstance(science, list):
                raise ValueError(f"Module {mod_id}: science must be a list")
            if len(science) > 1:
                raise ValueError(
                    f"Module {mod_id}: science array must have at most 1 item, "
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
                    raise ValueError(f"Skill {skill_id} missing numeric 'p' (mastery %)")
                if not (0 <= skill["p"] <= 100):
                    raise ValueError(
                        f"Skill {skill_id}: 'p' must be between 0-100, got {skill['p']}"
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

                # Validate prerequisite references (must be backward-only, acyclic)
                requires = skill.get("unlock_rules", {}).get("requires", [])
                for req_id in requires:
                    if req_id not in all_skill_ids:
                        raise ValueError(
                            f"Skill {skill_id} requires '{req_id}' which hasn't "
                            f"been defined yet (circular or forward reference)"
                        )

        # ── milestone-level sc_n / iv must match actual science counts ──
        declared_sc_n = milestone.get("sc_n", 0)
        declared_iv   = milestone.get("iv", 0)
        if declared_sc_n != scenario_count:
            raise ValueError(
                f"Milestone {m_id}: sc_n={declared_sc_n} does not match actual "
                f"Scenario count={scenario_count} across its modules"
            )
        if declared_iv != interview_count:
            raise ValueError(
                f"Milestone {m_id}: iv={declared_iv} does not match actual "
                f"Interview count={interview_count} across its modules"
            )

    # ── starting/current milestone must be M01 (RULE 10) ──
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

    years_experience = 0

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
        years_experience = user_input.get("years_experience", 0)

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
        context = fetch_poc_record(
            user_id=user_id,
            record_id=onboarding_record_id
        )

        # Fallback: try legacy vector-based context if POC record missing
        if not context:
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

        print(
            f"[ROADMAP AGENT] ✓ Retrieved onboarding conversation "
            f"({len(context)} chars)"
        )

        print(f"[ROADMAP AGENT] ✓ Context: {len(context)} chars")

        # Best-effort: try to pull years_experience out of structured
        # onboarding JSON if it was embedded in the context as JSON.
        try:
            parsed_ctx = json.loads(context)
            if isinstance(parsed_ctx, dict):
                years_experience = parsed_ctx.get("years_experience", 0)
        except (json.JSONDecodeError, TypeError):
            pass

    # ============================================================
    # LEVEL DETECTION  (README §2 — Dynamic starting point)
    # ============================================================

    level = detect_level(context, years_experience)
    print(f"[LEVEL] Detected level: {level} "
          f"(years_experience={years_experience}) "
          f"→ {LEVEL_MILESTONE_COUNT[level]} milestones "
          f"(M01-M{LEVEL_MILESTONE_COUNT[level]:02d})")

    # ============================================================
    # GENERATE ROADMAP  (up to 2 attempts)
    # ============================================================

    print("\n[ROADMAP AGENT] Invoking LLM (OpenAI → Gemini fallback)...")

    max_attempts = 2
    result       = ""

    for attempt in range(1, max_attempts + 1):
        try:
            print(f"[ROADMAP AGENT] Attempt {attempt}/{max_attempts}...")

            result       = roadmap_chain.invoke({
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
            # AUTO FIX sc_n / iv COUNTS
            # ==========================================
            for milestone in roadmap_data.get("milestones", []):
                scenario_count = 0
                interview_count = 0
                for mod in milestone.get("modules", []):
                    for sci in mod.get("science", []):
                        if sci.get("type") == "Scenario":
                            scenario_count += 1
                        elif sci.get("type") == "Interview":
                            interview_count += 1
                milestone["sc_n"] = scenario_count
                milestone["iv"] = interview_count
            # ==========================================
            # Inject runtime IDs
            # ==========================================
            now = datetime.utcnow().isoformat()
            roadmap_data["roadmap_id"]                              = ai_roadmap_id
            roadmap_data["user_id"]                                 = user_id
            roadmap_data.setdefault("level", level)
            roadmap_data.setdefault("roadmap_meta", {})["generated_at"] = now
            # ── Inject label field into each milestone if missing ──
            print("\n========== MODULE SKILL COUNT DEBUG ==========")
            for ms in roadmap_data.get("milestones", []):
                for mod in ms.get("modules", []):
                    if len(mod.get("skills", [])) < 3:
                        print(json.dumps(mod, indent=2))
            print("=============================================\n")
            # ── Validate structure ─────────────────────────────
            validate_roadmap_structure(roadmap_data)

            milestones = roadmap_data.get("milestones", [])
            print("[ROADMAP AGENT] ✓ Roadmap validated")
            print(f"  Target role : {roadmap_data.get('target_role', 'N/A')}")
            print(f"  Level       : {level}")
            print(f"  Milestones  : {len(milestones)}")
            print(f"  ICP type    : {roadmap_data.get('icp_type', 'N/A')}")

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

            save_poc_record(
    user_id=user_id,
    record_id=f"{user_id}_onboarding_conversation",
    text=json.dumps(roadmap_summary)
)

            save_poc_record(
            user_id=user_id,
            record_id=f"{user_id}_roadmap_conversation",
            text=json.dumps(roadmap_summary)
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
                "starting_milestone":       roadmap_data.get("starting_milestone", ""),
                "current_active_milestone": roadmap_data.get("current_active_milestone", ""),
                "vision_profile":           roadmap_data.get("vision_profile", {}),
                "roadmap_meta":             roadmap_data.get("roadmap_meta", {}),
                "milestones":               milestones,
                "pinecone_stored":          stored,
                "ai_metadata": {
                    "generated_at":          now,
                    "session_source":        "pinecone" if session_was_provided else "generated",
                    "generation_model":      "roadmap-gen-v3.1",
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
            import traceback
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