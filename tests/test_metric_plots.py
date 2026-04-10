from __future__ import annotations

import importlib.util
import json
import sys
import tempfile
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from latentdriver_waymax_experiments.metric_plots import generate_metric_comparison, latest_metric_rows, write_metric_csv, write_metric_json
from latentdriver_waymax_experiments.wayboard.data import discover_runs


class MetricPlotTests(unittest.TestCase):
    def _write_run(self, root: Path, *, run_id: str, model: str, ar: float, collision: float = 0.0) -> None:
        run_dir = root / run_id
        run_dir.mkdir(parents=True)
        metrics_path = run_dir / "metrics.json"
        metrics_path.write_text(
            json.dumps(
                {
                    "average": {
                        "number of episodes": 1,
                        "metric/AR[75:95]": ar,
                        "metric/collision_rate": collision,
                        "metric/offroad_rate": 0.0,
                        "metric/progress_rate": 1.0,
                        "reward/reward_mean": -10.0,
                    },
                    "average_over_class": {"metric/AR[75:95]": ar},
                    "per_class": {},
                }
            ),
            encoding="utf-8",
        )
        (run_dir / "run_manifest.json").write_text(
            json.dumps(
                {
                    "run_id": run_id,
                    "run_dir": str(run_dir),
                    "model": model,
                    "tier": "smoke_reactive",
                    "seed": 0,
                    "vis": False,
                    "metrics_path": str(metrics_path),
                }
            ),
            encoding="utf-8",
        )

    def test_latest_metric_rows_selects_latest_per_model(self) -> None:
        with tempfile.TemporaryDirectory() as td:
            root = Path(td)
            self._write_run(root, run_id="20260101_smoke_reactive_latentdriver", model="latentdriver_t2_j3", ar=0.2)
            self._write_run(root, run_id="20260102_smoke_reactive_latentdriver", model="latentdriver_t2_j3", ar=0.9)
            self._write_run(root, run_id="20260102_smoke_reactive_plant", model="plant", ar=0.4)

            rows = latest_metric_rows(records=discover_runs(root), tier="smoke_reactive", seed=0, models=["latentdriver_t2_j3", "plant"])

            self.assertEqual([row.model for row in rows], ["latentdriver_t2_j3", "plant"])
            self.assertEqual(rows[0].values["ar_75_95"], 0.9)
            self.assertEqual(rows[1].values["ar_75_95"], 0.4)

    def test_generate_metric_comparison_dry_run_reports_missing_models(self) -> None:
        with tempfile.TemporaryDirectory() as td:
            root = Path(td)
            self._write_run(root, run_id="20260101_smoke_reactive_latentdriver", model="latentdriver_t2_j3", ar=1.0)

            payload = generate_metric_comparison(root=root, tier="smoke_reactive", seed=0, models=["latentdriver_t2_j3", "plant"], dry_run=True)

            self.assertTrue(payload["ready"])
            self.assertEqual(payload["missing_models"], ["plant"])
            self.assertIn("model_metrics.png", payload["outputs"]["plot"])

    def test_write_metric_csv_and_json(self) -> None:
        with tempfile.TemporaryDirectory() as td:
            root = Path(td)
            self._write_run(root, run_id="20260101_smoke_reactive_latentdriver", model="latentdriver_t2_j3", ar=1.0)
            rows = latest_metric_rows(records=discover_runs(root), tier="smoke_reactive", seed=0, models=["latentdriver_t2_j3"])
            csv_path = root / "out" / "metrics.csv"
            json_path = root / "out" / "metrics.json"

            write_metric_csv(csv_path, rows, ["ar_75_95", "reward_mean"])
            write_metric_json(json_path, rows, ["ar_75_95", "reward_mean"])

            self.assertIn("latentdriver_t2_j3", csv_path.read_text(encoding="utf-8"))
            payload = json.loads(json_path.read_text(encoding="utf-8"))
            self.assertEqual(payload["rows"][0]["values"]["ar_75_95"], 1.0)

    @unittest.skipIf(importlib.util.find_spec("matplotlib") is None, "matplotlib is not installed")
    def test_generate_metric_comparison_writes_plot_when_matplotlib_is_available(self) -> None:
        with tempfile.TemporaryDirectory() as td:
            root = Path(td)
            self._write_run(root, run_id="20260101_smoke_reactive_latentdriver", model="latentdriver_t2_j3", ar=1.0)

            payload = generate_metric_comparison(root=root, tier="smoke_reactive", seed=0, models=["latentdriver_t2_j3"])

            plot_path = Path(payload["outputs"]["plot"])
            self.assertTrue(plot_path.exists())
            self.assertGreater(plot_path.stat().st_size, 0)


if __name__ == "__main__":
    unittest.main()
