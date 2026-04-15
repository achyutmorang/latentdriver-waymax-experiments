from __future__ import annotations

import json
import sys
import tempfile
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from latentdriver_waymax_experiments.metadata_join import (  # noqa: E402
    check_reasoning_causal_join,
    load_scenario_subset,
)


class MetadataJoinTests(unittest.TestCase):
    def test_check_reasoning_causal_join_with_subset_file(self) -> None:
        with tempfile.TemporaryDirectory() as td:
            root = Path(td)
            reasoning = root / "reasoning.jsonl"
            causal = root / "causal.jsonl"
            subset = root / "subset.txt"
            output = root / "out"

            reasoning.write_text(
                "\n".join(
                    [
                        json.dumps({"sid": "100", "ego": "1", "rel_id": ["2", "3"], "qa": "a"}),
                        json.dumps({"sid": "100", "rel_id": ["4"], "qa": "b"}),
                        json.dumps({"sid": "200", "ego": "8", "rel_id": ["9"]}),
                        json.dumps({"sid": "300", "ego": "10", "rel_id": ["11"]}),
                    ]
                )
                + "\n",
                encoding="utf-8",
            )
            causal.write_text(
                "\n".join(
                    [
                        json.dumps({"scenario_id": "100", "causal_agent_ids": ["3", "7"]}),
                        json.dumps({"scenario_id": "100", "causal_agent_ids": ["2", "3"]}),
                        json.dumps({"scenario_id": "200", "causal_agent_ids": ["20"]}),
                        json.dumps({"scenario_id": "400", "causal_agent_ids": ["99"]}),
                    ]
                )
                + "\n",
                encoding="utf-8",
            )
            subset.write_text("100\n200\n300\n", encoding="utf-8")

            payload = check_reasoning_causal_join(
                reasoning_path=reasoning,
                causal_path=causal,
                scenario_ids=load_scenario_subset(scenario_ids_file=subset),
                causal_policy="union",
                output_dir=output,
            )

            self.assertEqual(payload["subset"]["requested_scenario_count"], 3)
            self.assertEqual(payload["join"]["joined_scenario_count"], 2)
            self.assertAlmostEqual(payload["join"]["intersection_rate_vs_reasoning_subset"], 2 / 3)
            self.assertAlmostEqual(payload["join"]["agent_overlap_rate"], 1 / 2)
            self.assertEqual(payload["join"]["joined_scenarios_with_agent_overlap"], 1)
            self.assertTrue((output / "summary.json").is_file())
            joined_lines = (output / "joined_metadata.jsonl").read_text(encoding="utf-8").splitlines()
            self.assertEqual(len(joined_lines), 2)
            joined_rows = [json.loads(line) for line in joined_lines]
            scenario_100 = next(row for row in joined_rows if row["scenario_id"] == "100")
            self.assertEqual(scenario_100["reasoning_record_count"], 2)
            self.assertEqual(scenario_100["agent_overlap_ids"], ["2", "3"])
            self.assertEqual(scenario_100["causal_agent_ids"], ["2", "3", "7"])

    def test_majority_policy_requires_more_than_half_labelers(self) -> None:
        with tempfile.TemporaryDirectory() as td:
            root = Path(td)
            reasoning = root / "reasoning.json"
            causal = root / "causal.json"

            reasoning.write_text(
                json.dumps(
                    [
                        {"sid": "500", "ego": "1", "rel_id": ["10", "20", "30"]},
                    ]
                ),
                encoding="utf-8",
            )
            causal.write_text(
                json.dumps(
                    [
                        {
                            "scenario_id": "500",
                            "annotations": [
                                {"causal_agent_ids": ["10", "20"]},
                                {"causal_agent_ids": ["20", "30"]},
                                {"causal_agent_ids": ["20", "40"]},
                            ],
                        }
                    ]
                ),
                encoding="utf-8",
            )

            payload = check_reasoning_causal_join(
                reasoning_path=reasoning,
                causal_path=causal,
                causal_policy="majority",
            )

            self.assertEqual(payload["join"]["joined_scenario_count"], 1)
            self.assertEqual(payload["join"]["joined_scenarios_with_agent_overlap"], 1)
            self.assertEqual(payload["join"]["overlapping_agent_ids_sample"], ["20"])

    def test_load_scenario_subset_from_preprocess_map_dir(self) -> None:
        with tempfile.TemporaryDirectory() as td:
            root = Path(td)
            map_dir = root / "map"
            map_dir.mkdir(parents=True, exist_ok=True)
            for scenario_id in ("003", "001", "002"):
                (map_dir / f"{scenario_id}.npy").write_bytes(b"x")
            subset = load_scenario_subset(preprocess_map_dir=map_dir, limit=2)
            self.assertEqual(subset, ["001", "002"])


if __name__ == "__main__":
    unittest.main()
