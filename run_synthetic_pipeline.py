"""
Run Vidya V3 synthetic onboarding personas through the real roadmap pipeline.

Flow:
1. Ensure synthetic onboarding files exist.
2. Store each onboarding profile in Pinecone under its user_id namespace.
3. Call src.roadmap_agent.run_pipeline(user_id).
4. Validate and save generated roadmap JSON.
5. Write success, error, and metrics logs.
"""

from __future__ import annotations

import argparse
import json
import sys
import time
import traceback
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


BASE_DIR = Path(__file__).resolve().parent
sys.path.insert(0, str(BASE_DIR))

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")
if hasattr(sys.stderr, "reconfigure"):
    sys.stderr.reconfigure(encoding="utf-8")

from generate_dummy_data import DEFAULT_OUTPUT_DIR, write_personas  # noqa: E402
from src.pinecone_utils import INDEX_NAME, get_embedding, pc  # noqa: E402
from src.roadmap_agent import run_pipeline  # noqa: E402


DATA_DIR = BASE_DIR / "synthetic_test_data"
ONBOARDING_DIR = DATA_DIR / "onboarding_inputs"
ROADMAP_DIR = DATA_DIR / "generated_roadmaps"
LOG_DIR = DATA_DIR / "logs"

SUCCESS_LOG = LOG_DIR / "success.log"
ERROR_LOG = LOG_DIR / "errors.log"
METRICS_LOG = LOG_DIR / "pipeline_metrics.log"


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def append_jsonl(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as handle:
        handle.write(json.dumps(payload, ensure_ascii=False) + "\n")


def load_personas(onboarding_dir: Path) -> list[dict[str, Any]]:
    files = sorted(onboarding_dir.glob("user_*.json"))
    personas: list[dict[str, Any]] = []
    for file_path in files:
        with file_path.open("r", encoding="utf-8") as handle:
            persona = json.load(handle)
        persona.setdefault("user_id", file_path.stem)
        personas.append(persona)
    return personas


def format_onboarding_context(persona: dict[str, Any]) -> str:
    return "\n".join(
        [
            f"User ID: {persona['user_id']}",
            f"Full name: {persona['full_name']}",
            f"Age: {persona['age']}",
            f"Location: {persona['location']}",
            f"Education: {persona['education']}",
            f"Current role: {persona['current_role']}",
            f"Years experience: {persona['years_experience']}",
            f"Current salary monthly: {persona['current_salary_monthly']}",
            f"Target role: {persona['target_role']}",
            f"Target salary monthly: {persona['target_salary_monthly']}",
            f"Primary goal: {persona['primary_goal']}",
            f"Skill gaps: {', '.join(persona['skill_gaps'])}",
            f"Known skills: {', '.join(persona['known_skills'])}",
            f"Learning style: {persona['learning_style']}",
            f"Weekly hours available: {persona['weekly_hours_available']}",
            f"Device access: {persona['device_access']}",
            f"Language preference: {persona['language_preference']}",
            f"Confidence level: {persona['confidence_level']}",
            f"Urgency: {persona['urgency']}",
            f"Motivation: {persona['motivation']}",
            f"Biggest fear: {persona['biggest_fear']}",
            f"ICP type: {persona['icp_type']}",
        ]
    )


def store_onboarding_in_pinecone(persona: dict[str, Any]) -> str:
    user_id = persona["user_id"]
    context = format_onboarding_context(persona)
    vector_id = f"{user_id}_synthetic_onboarding"

    embedding = get_embedding(context, task_type="RETRIEVAL_DOCUMENT")
    index = pc.Index(INDEX_NAME)
    index.upsert(
        vectors=[
            {
                "id": vector_id,
                "values": embedding,
                "metadata": {
                    "doc_type": "onboarding",
                    "source": "synthetic_vidya_v3_batch",
                    "user_id": user_id,
                    "question_number": 1,
                    "text": context,
                    "icp_type": persona["icp_type"],
                    "target_role": persona["target_role"],
                    "current_role": persona["current_role"],
                    "confidence_level": persona["confidence_level"],
                    "urgency": persona["urgency"],
                    "generated_at": utc_now(),
                },
            }
        ],
        namespace=user_id,
    )
    return vector_id


def validate_roadmap_payload(roadmap: dict[str, Any]) -> None:
    if "error" in roadmap:
        raise ValueError(roadmap["error"])

    milestones = roadmap.get("milestones")
    if not isinstance(milestones, list) or not milestones:
        raise ValueError("Roadmap must include a non-empty milestones array")
    if len(milestones) > 4:
        raise ValueError(f"Roadmap has too many milestones: {len(milestones)}")

    seen_skills: set[str] = set()
    for milestone in milestones:
        modules = milestone.get("modules")
        if not isinstance(modules, list) or not modules:
            raise ValueError(f"Milestone {milestone.get('milestone_id')} has no modules")
        if len(modules) > 3:
            raise ValueError(f"Milestone {milestone.get('milestone_id')} has too many modules")

        for module in modules:
            skills = module.get("skills")
            if not isinstance(skills, list) or not skills:
                raise ValueError(f"Module {module.get('module_id')} has no skills")
            if len(skills) > 3:
                raise ValueError(f"Module {module.get('module_id')} has too many skills")

            for skill in skills:
                skill_id = skill.get("skill_id")
                if not skill_id:
                    raise ValueError("Skill missing skill_id")
                if skill_id in seen_skills:
                    raise ValueError(f"Duplicate skill_id: {skill_id}")

                content_flow = skill.get("content_flow", {})
                for content_type in ("video", "scenario", "mock", "review"):
                    if content_type not in content_flow:
                        raise ValueError(f"Skill {skill_id} missing {content_type}")

                mock = content_flow.get("mock", {})
                if mock.get("unlock_mastery") != 0.75:
                    raise ValueError(f"Skill {skill_id} has invalid mock.unlock_mastery")

                requires = skill.get("unlock_rules", {}).get("requires", [])
                for required_skill_id in requires:
                    if required_skill_id not in seen_skills:
                        raise ValueError(f"Skill {skill_id} requires forward or missing skill {required_skill_id}")

                seen_skills.add(skill_id)


def save_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")


def run_one_persona(persona: dict[str, Any], retries: int, trigger_mcq: bool) -> dict[str, Any]:
    user_id = persona["user_id"]
    last_error = ""

    for attempt in range(1, retries + 1):
        started = time.perf_counter()
        try:
            vector_id = store_onboarding_in_pinecone(persona)
            roadmap = run_pipeline(user_id, trigger_mcq=trigger_mcq)
            validate_roadmap_payload(roadmap)

            duration = time.perf_counter() - started
            output_path = ROADMAP_DIR / f"{user_id}_roadmap.json"
            save_json(output_path, roadmap)

            metric = {
                "timestamp": utc_now(),
                "user_id": user_id,
                "icp_type": persona["icp_type"],
                "attempt": attempt,
                "status": "success",
                "duration_seconds": round(duration, 3),
                "milestone_count": len(roadmap.get("milestones", [])),
                "target_role": roadmap.get("target_role"),
                "pinecone_onboarding_vector_id": vector_id,
                "pinecone_roadmap_stored": roadmap.get("pinecone_stored"),
                "output_path": str(output_path),
            }
            append_jsonl(SUCCESS_LOG, metric)
            append_jsonl(METRICS_LOG, metric)
            return metric

        except Exception as exc:
            duration = time.perf_counter() - started
            last_error = str(exc)
            error_payload = {
                "timestamp": utc_now(),
                "user_id": user_id,
                "icp_type": persona.get("icp_type"),
                "attempt": attempt,
                "status": "failed",
                "duration_seconds": round(duration, 3),
                "error": last_error,
                "traceback": traceback.format_exc(),
            }
            append_jsonl(ERROR_LOG, error_payload)
            append_jsonl(METRICS_LOG, error_payload)

    return {
        "timestamp": utc_now(),
        "user_id": user_id,
        "icp_type": persona.get("icp_type"),
        "status": "failed",
        "error": last_error,
    }


def ensure_inputs(onboarding_dir: Path, force_regenerate: bool) -> None:
    existing = sorted(onboarding_dir.glob("user_*.json"))
    if force_regenerate or len(existing) < 20:
        write_personas(onboarding_dir)


def main() -> None:
    parser = argparse.ArgumentParser(description="Run Vidya V3 synthetic roadmap pipeline.")
    parser.add_argument("--onboarding-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--retries", type=int, default=3)
    parser.add_argument("--limit", type=int, default=20)
    parser.add_argument("--force-regenerate", action="store_true")
    parser.add_argument("--trigger-mcq", action="store_true", help="Trigger downstream MCQ generation if configured.")
    args = parser.parse_args()

    ROADMAP_DIR.mkdir(parents=True, exist_ok=True)
    LOG_DIR.mkdir(parents=True, exist_ok=True)
    ensure_inputs(args.onboarding_dir, args.force_regenerate)

    personas = load_personas(args.onboarding_dir)[: args.limit]
    if not personas:
        raise RuntimeError(f"No onboarding inputs found in {args.onboarding_dir}")

    batch_started = time.perf_counter()
    results = []
    for persona in personas:
        user_id = persona["user_id"]
        print(f"[SYNTHETIC PIPELINE] Processing {user_id} ({persona['icp_type']})")
        results.append(run_one_persona(persona, retries=args.retries, trigger_mcq=args.trigger_mcq))

    success_count = sum(1 for result in results if result["status"] == "success")
    failed_count = len(results) - success_count
    summary = {
        "timestamp": utc_now(),
        "status": "batch_complete",
        "total": len(results),
        "success": success_count,
        "failed": failed_count,
        "duration_seconds": round(time.perf_counter() - batch_started, 3),
        "onboarding_dir": str(args.onboarding_dir),
        "roadmap_dir": str(ROADMAP_DIR),
        "log_dir": str(LOG_DIR),
    }
    append_jsonl(METRICS_LOG, summary)
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
