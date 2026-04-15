from __future__ import annotations

import sys
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from latentdriver_waymax_experiments.candidate_diversity import (  # noqa: E402
    probe_candidate_diversity,
    probe_candidate_diversity_suite,
)


class CandidateDiversityTests(unittest.TestCase):
    def test_latentdriver_probe_reports_explicit_mode_candidates(self) -> None:
        payload = probe_candidate_diversity("latentdriver_t2_j3")
        verdict = payload["verdict"]
        self.assertTrue(verdict["supports_exposed_candidate_diversity"])
        self.assertEqual(verdict["candidate_interface"], "explicit_mode_distribution")
        self.assertEqual(verdict["exposed_candidate_count"], 6)
        self.assertEqual(verdict["refinement_stage_count"], 3)
        evidence_kinds = {item["kind"] for item in payload["source_evidence"]}
        self.assertIn("candidate_queries", evidence_kinds)
        self.assertIn("native_selector", evidence_kinds)
        self.assertTrue(verdict["reranking_ready"])

    def test_plant_probe_reports_single_plan_output(self) -> None:
        payload = probe_candidate_diversity("plant")
        verdict = payload["verdict"]
        self.assertFalse(verdict["supports_exposed_candidate_diversity"])
        self.assertEqual(verdict["candidate_interface"], "single_plan_regression")
        self.assertEqual(verdict["exposed_candidate_count"], 1)
        evidence_kinds = {item["kind"] for item in payload["source_evidence"]}
        self.assertIn("policy_binding", evidence_kinds)
        self.assertIn("single_plan_shape", evidence_kinds)
        self.assertFalse(verdict["reranking_ready"])

    def test_suite_probe_builds_latentdriver_vs_plant_comparison(self) -> None:
        payload = probe_candidate_diversity_suite(["latentdriver_t2_j3", "plant"])
        self.assertEqual([item["model"] for item in payload["models"]], ["latentdriver_t2_j3", "plant"])
        self.assertEqual(len(payload["comparisons"]), 1)
        comparison = payload["comparisons"][0]
        self.assertEqual(comparison["left_model"], "latentdriver_t2_j3")
        self.assertEqual(comparison["right_model"], "plant")
        self.assertTrue(comparison["left_reranking_ready"])
        self.assertFalse(comparison["right_reranking_ready"])


if __name__ == "__main__":
    unittest.main()
