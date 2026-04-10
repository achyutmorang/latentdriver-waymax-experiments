from __future__ import annotations

import json
import sys
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from latentdriver_waymax_experiments.visual_compare import (
    build_ffmpeg_hstack_command,
    compare_latest_visualizations,
    find_latest_visualization,
)
from latentdriver_waymax_experiments.wayboard.data import discover_runs


class VisualCompareTests(unittest.TestCase):
    def _write_run(self, root: Path, *, run_id: str, model: str, tier: str, seed: int) -> Path:
        run_dir = root / run_id
        media_path = run_dir / "vis" / "straight_" / f"{model}.mp4"
        media_path.parent.mkdir(parents=True, exist_ok=True)
        media_path.write_bytes(b"fake-video")
        metrics_path = run_dir / "metrics.json"
        metrics_path.write_text(json.dumps({"average": {}, "average_over_class": {}, "per_class": {}}), encoding="utf-8")
        (run_dir / "run_manifest.json").write_text(
            json.dumps(
                {
                    "run_id": run_id,
                    "run_dir": str(run_dir),
                    "model": model,
                    "tier": tier,
                    "seed": seed,
                    "vis": "video",
                    "metrics_path": str(metrics_path),
                    "media_files": [str(media_path)],
                }
            ),
            encoding="utf-8",
        )
        return media_path

    def test_find_latest_visualization_uses_manifest_media_files(self) -> None:
        with tempfile.TemporaryDirectory() as td:
            root = Path(td)
            media_path = self._write_run(root, run_id="20260101_smoke_reactive_latentdriver", model="latentdriver_t2_j3", tier="smoke_reactive", seed=0)
            selected = find_latest_visualization(records=discover_runs(root), model="latentdriver_t2_j3", tier="smoke_reactive", seed=0)
            self.assertEqual(selected.media_path, media_path)

    def test_find_latest_visualization_falls_back_when_manifest_path_is_not_local(self) -> None:
        with tempfile.TemporaryDirectory() as td:
            root = Path(td)
            self._write_run(root, run_id="20260101_smoke_reactive_latentdriver", model="latentdriver_t2_j3", tier="smoke_reactive", seed=0)
            manifest_path = root / "20260101_smoke_reactive_latentdriver" / "run_manifest.json"
            manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
            manifest["media_files"] = ["/content/drive/not-visible-locally/demo.mp4"]
            manifest_path.write_text(json.dumps(manifest), encoding="utf-8")

            selected = find_latest_visualization(records=discover_runs(root), model="latentdriver_t2_j3", tier="smoke_reactive", seed=0)

            self.assertTrue(selected.media_path.exists())
            self.assertEqual(selected.media_path.name, "latentdriver_t2_j3.mp4")

    def test_build_ffmpeg_hstack_command_scales_and_stacks(self) -> None:
        cmd = build_ffmpeg_hstack_command(left=Path("left.mp4"), right=Path("right.mp4"), output=Path("out.mp4"), height=540)
        joined = " ".join(cmd)
        self.assertIn("hstack=inputs=2", joined)
        self.assertIn("scale=-2:540", joined)
        self.assertEqual(cmd[-1], "out.mp4")

    def test_compare_latest_visualizations_dry_run(self) -> None:
        with tempfile.TemporaryDirectory() as td:
            root = Path(td)
            self._write_run(root, run_id="20260102_smoke_reactive_latentdriver", model="latentdriver_t2_j3", tier="smoke_reactive", seed=0)
            self._write_run(root, run_id="20260102_smoke_reactive_plant", model="plant", tier="smoke_reactive", seed=0)
            with patch("latentdriver_waymax_experiments.visual_compare.shutil.which", return_value="/usr/bin/ffmpeg"):
                payload = compare_latest_visualizations(
                    root=root,
                    left_model="latentdriver_t2_j3",
                    right_model="plant",
                    tier="smoke_reactive",
                    seed=0,
                    dry_run=True,
                )
            self.assertEqual(payload["left"]["model"], "latentdriver_t2_j3")
            self.assertEqual(payload["right"]["model"], "plant")
            self.assertTrue(payload["comparison"]["ready"])


if __name__ == "__main__":
    unittest.main()
