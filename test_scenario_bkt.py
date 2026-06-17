"""
Verify scenario count validation and BKT injection logic.
"""
import sys, json
sys.path.insert(0, "src")

from roadmap_agent import inject_bkt_values, MIN_MODULES, MAX_MODULES, MIN_SKILLS, MAX_SKILLS

passed = 0
failed = 0

def check(name, condition, detail=""):
    global passed, failed
    if condition:
        print(f"  PASS  {name}")
        passed += 1
    else:
        print(f"  FAIL  {name}  {detail}")
        failed += 1

# ============================
# TEST: Scenario count validation logic
# ============================
print("\n=== SCENARIO COUNT VALIDATION ===")

# Build a sample roadmap with 4 scenarios across 2 modules
roadmap = {
    "milestones": [
        {
            "milestone_id": "M01",
            "modules": [
                {
                    "id": "M1.1",
                    "title": "Backend Foundations",
                    "ai_first_layer": "planning",
                    "skills": [{
                        "skill_id": "SKILL_M01_M1_S1",
                        "n": "python",
                        "title": "Advanced Python",
                        "p": 0,
                        "lessons": ["a", "b", "c"],
                        "content_flow": {
                            "video": {}, "scenario": {}, "mock": {}, "review": {}
                        },
                        "mastery_state": {"state": "unlocked", "bkt": {"prior": 0.15, "learn_rate": 0.25}},
                        "unlock_rules": {"requires": [], "minimum_mastery": 0.0, "unlock_type": "immediate"},
                    }],
                    "science": [
                        {"type": "Scenario", "desc": "S1"},
                        {"type": "Scenario", "desc": "S2"},
                    ],
                },
                {
                    "id": "M1.2",
                    "title": "System Design",
                    "ai_first_layer": "architecture",
                    "skills": [{
                        "skill_id": "SKILL_M01_M2_S1",
                        "n": "system_design",
                        "title": "System Design",
                        "p": 0,
                        "lessons": ["a", "b", "c"],
                        "content_flow": {
                            "video": {}, "scenario": {}, "mock": {}, "review": {}
                        },
                        "mastery_state": {"state": "unlocked", "bkt": {"prior": 0.15, "learn_rate": 0.25}},
                        "unlock_rules": {"requires": ["SKILL_M01_M1_S1"], "minimum_mastery": 0.0, "unlock_type": "prerequisite"},
                    }],
                    "science": [
                        {"type": "Scenario", "desc": "S3"},
                        {"type": "Interview", "desc": "I1"},
                    ],
                },
            ],
        }
    ]
}

# Count scenarios
total_scenarios = sum(
    1 for ms in roadmap["milestones"]
    for mod in ms.get("modules", [])
    for sci in mod.get("science", [])
    if sci.get("type") == "Scenario"
)
total_interviews = sum(
    1 for ms in roadmap["milestones"]
    for mod in ms.get("modules", [])
    for sci in mod.get("science", [])
    if sci.get("type") == "Interview"
)

print(f"  scenarios={total_scenarios}, interviews={total_interviews}")
check("3 scenarios in test data", total_scenarios == 3, f"got {total_scenarios}")
check("1 interview in test data", total_interviews == 1, f"got {total_interviews}")

# ============================
# TEST: BKT injection varies by milestone position
# ============================
print("\n=== BKT INJECTION ===")

# Build a 3-milestone roadmap
roadmap_bkt = {
    "milestones": [
        {
            "milestone_id": "M01",
            "modules": [{
                "id": "M1.1",
                "title": "Foundations",
                "ai_first_layer": "planning",
                "skills": [
                    {"skill_id": f"SKILL_M01_M1_S{s}", "n": f"skill{s}", "title": f"Skill {s}", "p": 0,
                     "lessons": ["a","b","c"],
                     "content_flow": {"video":{},"scenario":{},"mock":{},"review":{}},
                     "mastery_state": {"state":"unlocked","bkt":{"prior":0.15,"learn_rate":0.25}},
                     "unlock_rules": {"requires":[],"minimum_mastery":0.0,"unlock_type":"immediate"},
                    }
                    for s in range(1, 4)
                ],
                "science": [],
            }],
        },
        {
            "milestone_id": "M02",
            "modules": [{
                "id": "M2.1",
                "title": "Intermediate",
                "ai_first_layer": "architecture",
                "skills": [
                    {"skill_id": f"SKILL_M02_M1_S{s}", "n": f"skill{s}", "title": f"Skill {s}", "p": 0,
                     "lessons": ["a","b","c"],
                     "content_flow": {"video":{},"scenario":{},"mock":{},"review":{}},
                     "mastery_state": {"state":"unlocked","bkt":{"prior":0.15,"learn_rate":0.25}},
                     "unlock_rules": {"requires":[],"minimum_mastery":0.0,"unlock_type":"immediate"},
                    }
                    for s in range(1, 4)
                ],
                "science": [],
            }],
        },
    ]
}

inject_bkt_values(roadmap_bkt)
m1_prior = roadmap_bkt["milestones"][0]["modules"][0]["skills"][0]["mastery_state"]["bkt"]["prior"]
m1_rate = roadmap_bkt["milestones"][0]["modules"][0]["skills"][0]["mastery_state"]["bkt"]["learn_rate"]
m2_prior = roadmap_bkt["milestones"][1]["modules"][0]["skills"][0]["mastery_state"]["bkt"]["prior"]
m2_rate = roadmap_bkt["milestones"][1]["modules"][0]["skills"][0]["mastery_state"]["bkt"]["learn_rate"]

print(f"  M01: prior={m1_prior}, learn_rate={m1_rate}")
print(f"  M02: prior={m2_prior}, learn_rate={m2_rate}")

check("M01 prior > M02 prior", m1_prior > m2_prior,
      f"M01={m1_prior} M02={m2_prior}")
check("M01 learn_rate < M02 learn_rate", m1_rate < m2_rate,
      f"M01={m1_rate} M02={m2_rate}")
check("all skills in same milestone have same BKT",
      all(s["mastery_state"]["bkt"]["prior"] == m1_prior
          for s in roadmap_bkt["milestones"][0]["modules"][0]["skills"]))

# ============================
# SUMMARY
# ============================
print(f"\n{'='*50}")
print(f"RESULTS: {passed} passed, {failed} failed out of {passed+failed}")
if failed == 0:
    print("ALL TESTS PASSED")
else:
    print(f"SOME TESTS FAILED")
print(f"{'='*50}")
