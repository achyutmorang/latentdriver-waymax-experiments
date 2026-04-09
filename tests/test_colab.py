from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

from latentdriver_waymax_experiments.colab import _bind_symlink


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
            source.mkdir()
            with self.assertRaises(RuntimeError):
                _bind_symlink(target, source)


if __name__ == "__main__":
    unittest.main()
