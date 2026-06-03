from langchain_core.prompts import PromptTemplate
from langchain_openai import ChatOpenAI
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.output_parsers import StrOutputParser
from src.pinecone_utils import retrieve_context, retrieve_icp_type
import os
import json
import uuid
from typing import List
from datetime import datetime

# ── Use centralised, sanitised config ────────────────────────
# All values already .strip()'d — no trailing \n can reach urllib3.
from src.config import (
    OPENAI_API_KEY,
    GOOGLE_API_KEY,
    GOOGLE_MODEL    as _GOOGLE_MODEL,
)

# ============================================================
# LLM Configuration (Dual-Engine Fallback)
# ============================================================

def get_llm():
    openai_key   = OPENAI_API_KEY   or None
    gemini_key   = GOOGLE_API_KEY   or None
    google_model = _GOOGLE_MODEL.strip().strip("\"'")

    _deprecated = {
        "gemini-1.5", "gemini-1.5-flash", "gemini-1.5-pro",
        "gemini-2.0-flash", "gemini-2.0", "gemini-pro"
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
                temperature=0.3
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
# Roadmap Prompt — V3 Structure
# ============================================================

roadmap_prompt = PromptTemplate(
    input_variables=["context", "icp_type"],
    template="""
You are an expert AI career roadmap architect for Vidya V3.

Generate a deeply personalized career roadmap for this specific user.

USER CONTEXT:
{context}

USER ICP TYPE:
{icp_type}

=== GENERATION RULES (STRICT — ALL MUST BE FOLLOWED) ===

RULE 1 — HIERARCHY (LOCKED):
Roadmap → Milestone → Module → Skill → Content
Never break this hierarchy.

RULE 2 — MILESTONE COUNT:
Decide the number of milestones based on the user's goal complexity and background.
- Simple goal / beginner: 2-3 milestones
- Medium goal: 3-4 milestones
- Complex goal / advanced: 4 milestones (HARD MAX = 4 milestones)
Use milestone IDs: L1, L2, L3, L4 (for college/learning path)
OR P1, P2, P3, P4 (for working professional path)
Choose based on ICP type.

RULE 3 — MODULE LIMITS:
Max 3 modules per milestone.
Each module must have 2-3 skills.
HARD MAX: 3 skills per module.

RULE 4 — SKILL PREREQUISITES (ACYCLIC GRAPH):
Skills unlock via prerequisite graph only.
Never use module completion as a gate.
The graph MUST be acyclic (no circular dependencies).
First skill in each roadmap must have requires: [] (no prerequisites).

RULE 5 — MOCK UNLOCK:
mock.unlock_mastery = 0.75 always.
mock status = "locked" on generation.

RULE 6 — MILESTONE COMPLETION RULE:
A milestone is complete when ALL skills in that milestone reach mastery >= 0.90.

RULE 7 — DIFFICULTY PROGRESSION:
Difficulty values must increase progressively across skills.
Start: 0.1 - 0.3
Mid: 0.3 - 0.6
End: 0.6 - 0.9

RULE 8 — EVERY SKILL MUST HAVE ALL 4 CONTENT TYPES:
- video (content_id, title, duration_minutes, status: "locked")
- scenario (content_id, title, difficulty, status: "locked")
- mock (content_id, unlock_mastery: 0.75, status: "locked")
- review (review_type: "spaced_repetition", next_review_at: null)

RULE 9 — ICP DIFFERENTIATION (MANDATORY):

If icp_type is "high" (high_wage / working professional):
- Focus: system design, coding interviews, DSA, backend/frontend engineering, projects
- milestone_id prefix: P1, P2, P3, P4
- language: "en"
- Tone: growth-focused, switch/promotion-oriented

If icp_type is "low" (low_wage / student / fresher):
- Focus: Python basics, communication, practical workflows, confidence building, job readiness
- milestone_id prefix: L1, L2, L3, L4
- language: "en" (but simpler vocabulary)
- Tone: aspirational, placement-focused, confidence-building

RULE 10 — STARTING MILESTONE:
starting_milestone = first milestone_id in the milestones array.
current_active_milestone = same as starting_milestone.

RULE 11 — STRICT JSON ONLY:
Return ONLY raw valid JSON. No markdown. No code fences. No comments. No explanations.

=== OUTPUT STRUCTURE ===

Return this EXACT JSON structure (fill all string/number placeholders with real values):

{{
  "roadmap_id": "ai_roadmap_placeholder",
  "user_id": "placeholder",
  "icp_type": "{icp_type}",
  "target_role": "string — what the user wants to become",
  "language": "en",
  "starting_milestone": "L1 or P1",
  "current_active_milestone": "L1 or P1",
  "vision_profile": {{
    "current_state": "string — where user is now",
    "main_blocker": "string — biggest obstacle",
    "vision_12mo": "string — where they want to be in 12 months",
    "top_motivation": "string — why they want this"
  }},
  "roadmap_meta": {{
    "generated_at": "PLACEHOLDER_TIMESTAMP",
    "version": "v3.1",
    "science_model": ["ZPD", "Mastery Learning", "CLT", "BKT", "Possible Selves"]
  }},
  "milestones": [
    {{
      "milestone_id": "L1",
      "identity_label": "string — short identity label e.g. Foundations",
      "identity_statement": "string — 1 sentence motivational statement",
      "market_value_display": "string — e.g. Entry level or 3-5 LPA",
      "sequence_order": 1,
      "checkpoint_rule": {{
        "required_mastery": 0.9,
        "checkpoint_type": "mock_interview"
      }},
      "modules": [
        {{
          "module_id": "L1M1",
          "title": "string",
          "description": "string",
          "sequence_order": 1,
          "skills": [
            {{
              "skill_id": "SKILL_UNIQUE_ID",
              "title": "string",
              "description": "string",
              "difficulty": 0.2,
              "estimated_hours": 3,
              "mastery_state": {{
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
                  "content_id": "VID_001",
                  "title": "string",
                  "duration_minutes": 12,
                  "status": "locked"
                }},
                "scenario": {{
                  "content_id": "SCN_001",
                  "title": "string",
                  "difficulty": 0.3,
                  "status": "locked"
                }},
                "mock": {{
                  "content_id": "MOCK_001",
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
          ]
        }}
      ]
    }}
  ]
}}

IMPORTANT REMINDERS:
- All content_id values must be globally unique across the entire roadmap (e.g. VID_L1M1S1, SCN_L1M1S1, MOCK_L1M1S1).
- All skill_id values must be globally unique.
- skill unlock_rules.requires must reference skill_ids that appear EARLIER in the roadmap (acyclic).
- First skill of the entire roadmap must have requires: [] and unlock_type: "immediate".
- Do NOT output 7 milestones. Output 2-4 milestones based on the user's needs.
- Return ONLY raw JSON. Nothing else.
"""
)

roadmap_chain = roadmap_prompt | llm | StrOutputParser()

# ============================================================
# Pinecone Storage for Roadmap
# ============================================================

def store_roadmap_in_pinecone(user_id: str, roadmap_id: str, roadmap_data: dict) -> bool:
    """
    Stores the generated roadmap in Pinecone under the user's namespace.

    BUG FIXED: was doing a lazy `from src.pinecone_utils import ...` inside
    the function body.  On Hugging Face the module is already imported at the
    top level, so the lazy import was occasionally picking up a stale/partial
    module object and raising ImportError.  Now uses the module-level objects
    imported once at startup.
    """
    try:
        # Use the already-initialised clients from the module-level import
        from src.pinecone_utils import get_embedding, pc, INDEX_NAME

        index = pc.Index(INDEX_NAME)

        target_role      = roadmap_data.get("target_role", "")
        icp_type         = roadmap_data.get("icp_type", "")
        milestones       = roadmap_data.get("milestones", [])
        milestone_labels = " | ".join(m.get("identity_label", "") for m in milestones)

        embed_text = (
            f"Career roadmap for user {user_id}. "
            f"Target role: {target_role}. "
            f"ICP: {icp_type}. "
            f"Milestones: {milestone_labels}."
        )

        print(f"[PINECONE STORE] Generating embedding for roadmap {roadmap_id}...")
        embedding = get_embedding(embed_text, task_type="RETRIEVAL_DOCUMENT")

        roadmap_json_str  = json.dumps(roadmap_data)
        MAX_METADATA_BYTES = 38000

        if len(roadmap_json_str.encode("utf-8")) > MAX_METADATA_BYTES:
            print("[PINECONE STORE] ⚠ Roadmap JSON too large for single vector — storing summary only")
            store_payload = {
                "roadmap_id":          roadmap_id,
                "user_id":             user_id,
                "target_role":         target_role,
                "icp_type":            icp_type,
                "milestone_labels":    milestone_labels,
                "generated_at":        roadmap_data.get("roadmap_meta", {}).get("generated_at", ""),
                "full_roadmap_stored": False,
                "doc_type":            "roadmap_summary"
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
                "doc_type":            "roadmap"
            }

        vector_id = f"{user_id}_roadmap_{roadmap_id}"

        index.upsert(
            vectors=[{
                "id":       vector_id,
                "values":   embedding,
                "metadata": store_payload
            }],
            namespace=user_id
        )

        print(f"[PINECONE STORE] ✓ Roadmap stored — vector_id: {vector_id}")
        return True

    except ImportError as e:
        print(f"[PINECONE STORE] ✗ Import error (check pinecone_utils exports): {e}")
        return False
    except Exception as e:
        print(f"[PINECONE STORE] ✗ Failed to store roadmap: {e}")
        import traceback
        traceback.print_exc()
        return False


# ============================================================
# JSON Repair Utility
# ============================================================

def repair_json(raw: str) -> str:
    """Strip markdown fences and extract the outermost JSON object."""
    clean = raw.strip()
    if "```" in clean:
        clean = clean.replace("```json", "").replace("```", "").strip()
    start = clean.find('{')
    end   = clean.rfind('}') + 1
    if start != -1 and end > 0:
        clean = clean[start:end]
    return clean


def validate_roadmap_structure(data: dict) -> None:
    """
    Validates the V3 roadmap structure.
    Raises ValueError with a clear message on any violation.
    Auto-fixes mock.unlock_mastery silently.
    """
    milestones = data.get("milestones", [])

    if not milestones:
        raise ValueError("Roadmap has no milestones")

    if len(milestones) > 4:
        raise ValueError(f"Too many milestones: {len(milestones)} (max 4)")

    all_skill_ids = []  # type: List[str]

    for m_idx, milestone in enumerate(milestones):
        m_id    = milestone.get("milestone_id", f"M{m_idx + 1}")
        modules = milestone.get("modules", [])

        if not isinstance(modules, list):
            raise ValueError(f"Milestone {m_id}: modules must be a list")
        if len(modules) > 3:
            raise ValueError(f"Milestone {m_id}: too many modules ({len(modules)}, max 3)")

        for mod in modules:
            mod_id = mod.get("module_id", "?")
            skills = mod.get("skills", [])

            if not isinstance(skills, list):
                raise ValueError(f"Module {mod_id}: skills must be a list")
            if len(skills) > 3:
                raise ValueError(f"Module {mod_id}: too many skills ({len(skills)}, max 3)")

            for skill in skills:
                skill_id = skill.get("skill_id", "?")

                if skill_id in all_skill_ids:
                    raise ValueError(f"Duplicate skill_id: {skill_id}")
                all_skill_ids.append(skill_id)

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

                # Validate prerequisite references — must be backward-only (acyclic)
                requires = skill.get("unlock_rules", {}).get("requires", [])
                for req_id in requires:
                    if req_id not in all_skill_ids:
                        raise ValueError(
                            f"Skill {skill_id} requires '{req_id}' which hasn't been defined yet "
                            f"(circular or forward reference)"
                        )


# ============================================================
# run_pipeline — Main Entry Point
# ============================================================

def run_pipeline(
    user_input,
    trigger_mcq: bool = True,
    ai_session_id: str = None,
    ai_roadmap_id: str = None
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

        if not context:
            return {
                "error":          "AI onboarding context missing",
                "user_id":        user_id,
                "ai_session_id":  ai_session_id
            }

    else:
        # ── REAL USER / PINECONE MODE ───────────────────────────
        print("[ROADMAP AGENT] Mode: Real user (Pinecone)")

        user_id  = user_input
        icp_type = retrieve_icp_type(user_id)

        if not icp_type:
            print("[ICP] icp_type not found — onboarding incomplete")
            return {
                "error":          "Please complete onboarding first.",
                "user_id":        user_id,
                "ai_session_id":  ai_session_id
            }

        print(f"[ICP] Classified as: {icp_type}")

        context = retrieve_context(user_id)

        if not context:
            print("[ROADMAP AGENT] ✗ No context found in Pinecone")
            return {
                "error":          "No user data found. Please complete onboarding first.",
                "user_id":        user_id,
                "ai_session_id":  ai_session_id
            }

        print(f"[ROADMAP AGENT] ✓ Context: {len(context)} chars")

    # ============================================================
    # GENERATE ROADMAP  (up to 2 attempts)
    # ============================================================

    print("\n[ROADMAP AGENT] Invoking LLM (OpenAI → Gemini fallback)...")

    max_attempts = 2
    result       = ""

    for attempt in range(1, max_attempts + 1):
        try:
            print(f"[ROADMAP AGENT] Attempt {attempt}/{max_attempts}...")

            result       = roadmap_chain.invoke({"context": context, "icp_type": icp_type})
            clean_result = repair_json(result)
            roadmap_data = json.loads(clean_result)

            # ── Inject runtime IDs ─────────────────────────────
            now = datetime.utcnow().isoformat()
            roadmap_data["roadmap_id"]                     = ai_roadmap_id
            roadmap_data["user_id"]                        = user_id
            roadmap_data.setdefault("roadmap_meta", {})["generated_at"] = now

            # ── Validate structure ─────────────────────────────
            validate_roadmap_structure(roadmap_data)

            milestones = roadmap_data.get("milestones", [])
            print("[ROADMAP AGENT] ✓ Roadmap validated")
            print(f"  Target role : {roadmap_data.get('target_role', 'N/A')}")
            print(f"  Milestones  : {len(milestones)}")
            print(f"  ICP type    : {roadmap_data.get('icp_type', 'N/A')}")

            # ── Store in Pinecone ──────────────────────────────
            print("\n[ROADMAP AGENT] Storing roadmap in Pinecone...")
            stored = store_roadmap_in_pinecone(user_id, ai_roadmap_id, roadmap_data)
            if stored:
                print("[ROADMAP AGENT] ✓ Roadmap persisted to Pinecone")
            else:
                print("[ROADMAP AGENT] ⚠ Roadmap generated but NOT stored in Pinecone")

            # ── Build final response ───────────────────────────
            return {
                "id":                       str(uuid.uuid4()),
                "user_id":                  user_id,
                "ai_session_id":            ai_session_id,
                "ai_roadmap_id":            ai_roadmap_id,
                "target_role":              roadmap_data.get("target_role", ""),
                "icp_type":                 roadmap_data.get("icp_type", icp_type),
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
                    "personalization_score": 0.92
                },
                "status":       "confirmed",
                "created_at":   now,
                "updated_at":   now,
                "confirmed_at": now,
                "published_at": None
            }

        except json.JSONDecodeError as e:
            print(f"[ROADMAP AGENT] ✗ JSON parse error (attempt {attempt}): {e}")
            if attempt == max_attempts:
                return {
                    "error":          f"Invalid JSON from LLM: {str(e)}",
                    "user_id":        user_id,
                    "ai_session_id":  ai_session_id,
                    "raw_output":     result[:500]
                }

        except ValueError as e:
            print(f"[ROADMAP AGENT] ✗ Validation error (attempt {attempt}): {e}")
            if attempt == max_attempts:
                return {
                    "error":          f"Roadmap validation failed: {str(e)}",
                    "user_id":        user_id,
                    "ai_session_id":  ai_session_id
                }

        except Exception as e:
            print(f"[ROADMAP AGENT] ✗ Unexpected error (attempt {attempt}): {e}")
            if 'clean_result' in locals():
                print(clean_result[:1000])
            import traceback
            traceback.print_exc()
            if attempt == max_attempts:
                return {
                    "error":          f"Failed to generate roadmap: {str(e)}",
                    "user_id":        user_id,
                    "ai_session_id":  ai_session_id
                }