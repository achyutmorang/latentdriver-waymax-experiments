from __future__ import annotations

import sys
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from latentdriver_waymax_experiments.colab import _bind_symlink, bind_drive_layout


class ColabBindingTests(unittest.TestCase):
    def test_bind_symlink_replaces_placeholder_directory(self) -> None:
        with tempfile.TemporaryDirectory() as td:
            root = Path(td)
            target = root / "target"
            source = root / "source"
            target.mkdir()
            (target / ".gitkeep").write_text("", encoding="utf-8")
            source.mkdir()
            _bind_symlink(target, source)
            self.assertTrue(target.is_symlink())
            self.assertEqual(target.resolve(), source.resolve())

    def test_bind_symlink_refuses_non_empty_directory(self) -> None:
        with tempfile.TemporaryDirectory() as td:
            root = Path(td)
            target = root / "target"
            source = root / "source"
            target.mkdir()
            (target / "data.txt").write_text("x", encoding="utf-8")
            (source / "data.txt").parent.mkdir(parents=True, exist_ok=True)
            (source / "data.txt").write_text("y", encoding="utf-8")
            with self.assertRaises(RuntimeError):
                _bind_symlink(target, source)

    def test_bind_symlink_migrates_existing_directory_contents(self) -> None:
        with tempfile.TemporaryDirectory() as td:
            root = Path(td)
            target = root / "target"
            source = root / "source"
            target.mkdir()
            (target / "data.txt").write_text("x", encoding="utf-8")
            source.mkdir()
            _bind_symlink(target, source)
            self.assertTrue(target.is_symlink())
            self.assertEqual((source / "data.txt").read_text(encoding="utf-8"), "x")

    def test_bind_drive_layout_binds_debug_runs_to_drive(self) -> None:
        with tempfile.TemporaryDirectory() as td:
            root = Path(td)
            repo_root = root / "repo"

            def fake_resolve(path: str) -> Path:
                return repo_root / path

            with patch("latentdriver_waymax_experiments.colab.resolve_repo_relative", side_effect=fake_resolve):
                binding = bind_drive_layout(str(root / "waymax_research"))

            debug_target = repo_root / "results" / "debug_runs"
            self.assertTrue(debug_target.is_symlink())
            self.assertEqual(debug_target.resolve(), Path(binding["debug_runs"]).resolve())
            self.assertEqual(Path(binding["debug_runs"]), root / "waymax_research" / "latentdriver_waymax_experiments" / "debug_runs")


if __name__ == "__main__":
    unittest.main()
