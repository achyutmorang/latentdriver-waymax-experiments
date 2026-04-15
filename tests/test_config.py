from __future__ import annotations

import sys
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from latentdriver_waymax_experiments.config import load_config


class ConfigTests(unittest.TestCase):
    def test_load_config(self) -> None:
        cfg = load_config()
        self.assertEqual(cfg["project"]["name"], "latentdriver-waymax-experiments")
        self.assertIn("latentdriver_t2_j3", cfg["checkpoints"])
        self.assertIn("full_reactive", cfg["evaluation"]["tiers"])
        self.assertIn("interactive_pilot", cfg["validation"])
        self.assertIn("interactive_pilot_reactive", cfg["evaluation"]["tiers"])


if __name__ == "__main__":
    unittest.main()
