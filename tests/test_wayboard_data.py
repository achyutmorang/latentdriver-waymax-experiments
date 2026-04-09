from __future__ import annotations

import json
import sys
import tempfile
import unittest
from pathlib import Path

from bokeh.document import Document

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT / "src"))

from latentdriver_waymax_experiments.wayboard.app import WaymaxBoard
from latentdriver_waymax_experiments.wayboard.data import discover_runs, discover_suites


class WaymaxBoardDataTests(unittest.TestCase):
    def test_discover_runs_loads_manifest_metrics_and_media(self) -> None:
        with tempfile.TemporaryDirectory() as td:
            root = Path(td)
            run_dir = root / "20260410T000000Z_smoke_reactive_latentdriver_seed0"
            vis_dir = run_dir / "vis" / "turning_left"
            vis_dir.mkdir(parents=True, exist_ok=True)
            metrics_path = run_dir / "metrics.json"
            stdout_path = run_dir / "stdout.log"
            stderr_path = run_dir / "stderr.log"
            metrics_path.write_text(
                json.dumps(
                    {
                        "average": {
                            "number of episodes": 1,
                            "metric/AR[75:95]": 0.41,
                            "metric/collision_rate": 0.0,
                            "metric/offroad_rate": 0.1,
                            "metric/progress_rate": 0.8,
                        },
                        "average_over_class": {"metric/AR[75:95]": 0.33},
                    }
                ),
                encoding="utf-8",
            )
            stdout_path.write_text("stdout", encoding="utf-8")
            stderr_path.write_text("stderr", encoding="utf-8")
            (vis_dir / "demo.mp4").write_bytes(b"fake")
            manifest = {
                "run_id": run_dir.name,
                "model": "latentdriver_t2_j3",
                "tier": "smoke_reactive",
                "seed": 0,
                "vis": "video",
                "metrics_path": str(metrics_path),
                "stdout_path": str(stdout_path),
                "stderr_path": str(stderr_path),
            }
            (run_dir / "run_manifest.json").write_text(json.dumps(manifest), encoding="utf-8")

            records = discover_runs(root)
            self.assertEqual(len(records), 1)
            record = records[0]
            self.assertEqual(record.summary["ar_75_95"], 0.41)
            self.assertEqual(record.summary["mar_75_95"], 0.33)
            self.assertEqual(len(record.media_artifacts), 1)
            self.assertEqual(record.media_artifacts[0].media_type, "video")
            self.assertEqual(
                record.media_artifacts[0].artifact_url("/artifacts"),
                f"/artifacts/{run_dir.name}/vis/turning_left/demo.mp4",
            )

    def test_discover_suites_loads_dry_run_bundle(self) -> None:
        with tempfile.TemporaryDirectory() as td:
            root = Path(td)
            run_dir = root / "20260410T000000Z_suite_smoke_reactive"
            run_dir.mkdir(parents=True, exist_ok=True)
            payload = {
                "tier": "smoke_reactive",
                "seed": 0,
                "models": ["latentdriver_t2_j3", "plant"],
                "runs": [{"dry_run": True}, {"dry_run": True}],
            }
            (run_dir / "suite_summary.json").write_text(json.dumps(payload), encoding="utf-8")
            suites = discover_suites(root)
            self.assertEqual(len(suites), 1)
            self.assertEqual(suites[0].models, ("latentdriver_t2_j3", "plant"))

    def test_waymax_board_builds_document_from_results_root(self) -> None:
        with tempfile.TemporaryDirectory() as td:
            root = Path(td)
            run_dir = root / "20260410T000000Z_smoke_reactive_latentdriver_seed0"
            run_dir.mkdir(parents=True, exist_ok=True)
            metrics_path = run_dir / "metrics.json"
            metrics_path.write_text(
                json.dumps(
                    {
                        "average": {
                            "number of episodes": 1,
                            "metric/AR[75:95]": 0.41,
                            "metric/collision_rate": 0.0,
                            "metric/offroad_rate": 0.1,
                            "metric/progress_rate": 0.8,
                        },
                        "average_over_class": {"metric/AR[75:95]": 0.33},
                    }
                ),
                encoding="utf-8",
            )
            manifest = {
                "run_id": run_dir.name,
                "model": "latentdriver_t2_j3",
                "tier": "smoke_reactive",
                "seed": 0,
                "vis": False,
                "metrics_path": str(metrics_path),
            }
            (run_dir / "run_manifest.json").write_text(json.dumps(manifest), encoding="utf-8")

            doc = Document()
            board = WaymaxBoard(results_dir=root)
            board._build_document(doc)
            self.assertEqual(doc.title, "WaymaxBoard")
            self.assertTrue(doc.roots)


if __name__ == "__main__":
    unittest.main()
