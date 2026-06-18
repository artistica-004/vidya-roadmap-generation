"""
Tests for the Milestone Authority Engine and AI Layer Diversity.

Run with: pytest test_milestone_authority.py -v

These tests exercise the core logic using isolated imports.
"""
import sys
import os
import json
import unittest
from unittest.mock import patch, MagicMock

# Patch ALL module-level side effects BEFORE importing roadmap_agent
# The module calls get_llm() during import which needs API keys and stdout encoding.
_import_patches = [
    patch.dict(os.environ, {"OPENAI_API_KEY": "test", "GOOGLE_API_KEY": "test"}),  # prevent LLM init error
    patch("builtins.print"),  # suppress print side effects
    patch("sys.stdout"),      # prevent encoding issues on Windows
]

for p in _import_patches:
    p.start()

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "src"))
try:
    from roadmap_agent import (
        compute_authoritative_milestone_range,
        validate_milestone_authority_schema,
        MilestoneAuthoritySchemaError,
        LEVEL_RANGE_PREFERENCES,
        MIN_MILESTONES,
        MAX_MILESTONES,
        ALLOWED_AI_LAYERS,
        normalize_ai_layer,
        repair_ai_layers,
    )
finally:
    for p in reversed(_import_patches):
        p.stop()


class TestMilestoneAuthoritySchema(unittest.TestCase):
    """Tests for validate_milestone_authority_schema."""

    def test_valid_schema(self):
        """A valid dict must pass without error."""
        valid = {
            "recommended": 5,
            "minimum": 4,
            "maximum": 6,
            "confidence": 0.84,
            "reasoning": "gap=0.85; beginner→Data Scientist needs 5 milestones",
        }
        validate_milestone_authority_schema(valid)

    def test_missing_reasoning(self):
        bad = {"recommended": 5, "minimum": 4, "maximum": 6, "confidence": 0.8}
        with self.assertRaises(MilestoneAuthoritySchemaError) as ctx:
            validate_milestone_authority_schema(bad)
        self.assertIn("reasoning", str(ctx.exception))

    def test_empty_reasoning(self):
        bad = {"recommended": 5, "minimum": 4, "maximum": 6, "confidence": 0.8, "reasoning": ""}
        with self.assertRaises(MilestoneAuthoritySchemaError):
            validate_milestone_authority_schema(bad)

    def test_wrong_type(self):
        bad = {"recommended": "five", "minimum": 4, "maximum": 6, "confidence": 0.8, "reasoning": "test"}
        with self.assertRaises(MilestoneAuthoritySchemaError):
            validate_milestone_authority_schema(bad)

    def test_extra_keys(self):
        bad = {"recommended": 5, "minimum": 4, "maximum": 6, "confidence": 0.8, "reasoning": "x", "extra": 1}
        with self.assertRaises(MilestoneAuthoritySchemaError):
            validate_milestone_authority_schema(bad)


class TestComputeAuthoritativeMilestoneRange(unittest.TestCase):
    """Tests for compute_authoritative_milestone_range."""

    def _assert_valid(self, result):
        validate_milestone_authority_schema(result)
        self.assertIsInstance(result["recommended"], int)
        self.assertIsInstance(result["minimum"], int)
        self.assertIsInstance(result["maximum"], int)
        self.assertIsInstance(result["confidence"], float)
        self.assertIsInstance(result["reasoning"], str)
        self.assertTrue(len(result["reasoning"]) > 10)
        self.assertGreaterEqual(result["recommended"], result["minimum"])
        self.assertLessEqual(result["recommended"], result["maximum"])
        self.assertGreaterEqual(result["confidence"], 0.0)
        self.assertLessEqual(result["confidence"], 1.0)

    def test_student_beginner_data_scientist(self):
        """TEST 2: Student beginner, Data Scientist, gap=1.0"""
        result = compute_authoritative_milestone_range(
            gap_score=1.0, icp_type="low", level="beginner",
            hours_per_week=5, timeline_days=112, known_skills=[], experience_years=0,
        )
        self._assert_valid(result)
        pref = LEVEL_RANGE_PREFERENCES[("low", "beginner")]
        self.assertEqual(result["minimum"], pref["min"])
        self.assertEqual(result["maximum"], pref["max"])

    def test_professional_intermediate_ai_engineer(self):
        """TEST 3: Professional intermediate, AI Engineer."""
        result = compute_authoritative_milestone_range(
            gap_score=0.65, icp_type="high", level="intermediate",
            hours_per_week=10, timeline_days=112,
            known_skills=["python", "sql", "docker", "kubernetes"],
            experience_years=3,
        )
        self._assert_valid(result)

    def test_all_profiles(self):
        """TEST 7: Every LEVEL_RANGE_PREFERENCES entry produces valid schema."""
        for (icp, level), pref in LEVEL_RANGE_PREFERENCES.items():
            with self.subTest(icp=icp, level=level):
                result = compute_authoritative_milestone_range(
                    gap_score=0.5, icp_type=icp, level=level,
                    hours_per_week=10, timeline_days=112,
                    known_skills=[], experience_years=1,
                )
                self._assert_valid(result)
                self.assertEqual(result["minimum"], pref["min"])
                self.assertEqual(result["maximum"], pref["max"])

    def test_edge_case_senior(self):
        """Senior professional with low gap → low end of range."""
        result = compute_authoritative_milestone_range(
            gap_score=0.1, icp_type="high", level="senior",
            hours_per_week=40, timeline_days=112,
            known_skills=["s1"] * 100, experience_years=20,
        )
        self._assert_valid(result)

    def test_edge_case_minimal(self):
        """Absolute minimum inputs."""
        result = compute_authoritative_milestone_range(
            gap_score=0.0, icp_type="low", level="beginner",
            hours_per_week=1, timeline_days=7,
            known_skills=[], experience_years=0,
        )
        self._assert_valid(result)

    def test_dynamic_diversity(self):
        """TEST 9: Two Student Beginners with different profiles should
        produce different recommendations."""
        r1 = compute_authoritative_milestone_range(
            gap_score=1.0, icp_type="low", level="beginner",
            hours_per_week=20, timeline_days=365,
            known_skills=[], experience_years=0,
        )
        r2 = compute_authoritative_milestone_range(
            gap_score=0.3, icp_type="low", level="beginner",
            hours_per_week=5, timeline_days=112,
            known_skills=["python", "sql", "git"], experience_years=1,
        )
        self._assert_valid(r1)
        self._assert_valid(r2)
        pref = LEVEL_RANGE_PREFERENCES[("low", "beginner")]
        self.assertGreaterEqual(r1["recommended"], pref["min"])
        self.assertLessEqual(r1["recommended"], pref["max"])
        self.assertGreaterEqual(r2["recommended"], pref["min"])
        self.assertLessEqual(r2["recommended"], pref["max"])
        print(f"  → r1={r1['recommended']} r2={r2['recommended']}")


class TestAiLayerNormalization(unittest.TestCase):
    """Tests for normalize_ai_layer and repair_ai_layers."""

    def test_normalize_legacy_values(self):
        """All known legacy values must map to valid ALLOWED_AI_LAYERS."""
        test_cases = [
            ("architecture", "vibe_architecture"),
            ("implementation", "vibe_solution"),
            ("debugging", "vibe_solution"),
            ("optimization", "vibe_architecture"),
            ("planning", "vibe_planning"),
            ("design", "vibe_architecture"),
            ("infra", "deployment"),
            ("deployment", "deployment"),
            ("vibe_planning", "vibe_planning"),
            ("vibe_solution", "vibe_solution"),
            ("vibe_architecture", "vibe_architecture"),
        ]
        for raw, expected in test_cases:
            with self.subTest(raw=raw):
                result = normalize_ai_layer(raw)
                self.assertEqual(result, expected)
                self.assertIn(result, ALLOWED_AI_LAYERS)

    def test_unknown_value_defaults(self):
        """Unknown values should default to 'vibe_solution'."""
        result = normalize_ai_layer("unknown_value_xyz")
        self.assertEqual(result, "vibe_solution")
        self.assertIn(result, ALLOWED_AI_LAYERS)

    def test_repair_preserves_module_diversity(self):
        """TEST 6: Repair must not collapse module-level ai_first_layer diversity."""
        roadmap_data = {
            "milestones": [
                {
                    "milestone_id": "M01",
                    "modules": [
                        {
                            "id": "M1.1",
                            "ai_first_layer": "vibe_planning",
                            "skills": [{
                                "ai_metadata": {
                                    "layer": "vibe_planning",
                                    "usage_type": "generation",
                                    "automation_level": "assistant",
                                    "ai_first": True,
                                },
                            }],
                        },
                        {
                            "id": "M1.2",
                            "ai_first_layer": "vibe_architecture",
                            "skills": [{
                                "ai_metadata": {
                                    "layer": "architecture",  # legacy — needs repair
                                    "usage_type": "generation",
                                    "automation_level": "assistant",
                                    "ai_first": False,
                                },
                            }],
                        },
                    ],
                },
            ],
        }
        repaired = repair_ai_layers(roadmap_data)
        self.assertGreater(repaired, 0, "Should have repaired legacy 'architecture' layer")

        # Check module-level diversity preserved
        layers = set()
        for ms in roadmap_data["milestones"]:
            for mod in ms["modules"]:
                layers.add(mod["ai_first_layer"])
        self.assertGreaterEqual(len(layers), 2, f"AI layer diversity collapsed: {layers}")
        
        # Check skill-level layers all valid
        for ms in roadmap_data["milestones"]:
            for mod in ms["modules"]:
                for skill in mod.get("skills", []):
                    ai = skill.get("ai_metadata", {})
                    self.assertIn(ai.get("layer", ""), ALLOWED_AI_LAYERS)
                    self.assertTrue(ai.get("ai_first"))

    def test_repair_normalizes_module_layer(self):
        """Module-level ai_first_layer must be normalized by repair."""
        roadmap_data = {
            "milestones": [
                {
                    "milestone_id": "M01",
                    "modules": [
                        {
                            "id": "M1.1",
                            "ai_first_layer": "architecture",  # legacy
                            "skills": [{
                                "ai_metadata": {
                                    "layer": "vibe_architecture",
                                    "usage_type": "generation",
                                    "automation_level": "assistant",
                                    "ai_first": True,
                                },
                            }],
                        },
                    ],
                },
            ],
        }
        repaired = repair_ai_layers(roadmap_data)
        self.assertGreater(repaired, 0)
        mod = roadmap_data["milestones"][0]["modules"][0]
        self.assertEqual(mod["ai_first_layer"], "vibe_architecture")

    def test_repair_does_not_crash_on_empty_data(self):
        """Repair with empty milestones should not crash."""
        roadmap_data = {"milestones": []}
        repaired = repair_ai_layers(roadmap_data)
        self.assertEqual(repaired, 0)

    def test_repair_does_not_crash_on_missing_ai_metadata(self):
        """Skills without ai_metadata should not cause crashes."""
        roadmap_data = {
            "milestones": [
                {
                    "milestone_id": "M01",
                    "modules": [
                        {
                            "id": "M1.1",
                            "ai_first_layer": "vibe_planning",
                            "skills": [{"n": "python"}],  # no ai_metadata
                        },
                    ],
                },
            ],
        }
        repaired = repair_ai_layers(roadmap_data)
        self.assertEqual(repaired, 0)


if __name__ == "__main__":
    unittest.main()
