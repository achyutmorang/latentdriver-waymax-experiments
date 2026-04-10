from __future__ import annotations

import sys
import tempfile
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from latentdriver_waymax_experiments.upstream import (
    PYTHON312_SITE_CUSTOMIZE_BLOCK,
    ensure_python312_compat_sitecustomize,
)


class UpstreamCompatTests(unittest.TestCase):
    def test_ensure_python312_compat_sitecustomize_creates_file(self) -> None:
        with tempfile.TemporaryDirectory() as td:
            upstream_dir = Path(td)
            sitecustomize_path = ensure_python312_compat_sitecustomize(upstream_dir)
            self.assertEqual(sitecustomize_path, upstream_dir / "sitecustomize.py")
            self.assertIn(PYTHON312_SITE_CUSTOMIZE_BLOCK, sitecustomize_path.read_text(encoding="utf-8"))

    def test_ensure_python312_compat_sitecustomize_is_idempotent(self) -> None:
        with tempfile.TemporaryDirectory() as td:
            upstream_dir = Path(td)
            first = ensure_python312_compat_sitecustomize(upstream_dir)
            first_text = first.read_text(encoding="utf-8")
            second = ensure_python312_compat_sitecustomize(upstream_dir)
            self.assertEqual(first_text, second.read_text(encoding="utf-8"))


if __name__ == "__main__":
    unittest.main()
