from __future__ import annotations

import json
import sys
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from latentdriver_waymax_experiments.modulation.config import collect_modulation_environment, load_modulation_config_from_env
from latentdriver_waymax_experiments.modulation.features import extract_risk_features
from latentdriver_waymax_experiments.modulation.heuristic import HeuristicActionModulator
from latentdriver_waymax_experiments.modulation.runtime import ActionModulationRuntime, build_action_modulation_runtime_from_env


class ModulationTests(unittest.TestCase):
    def _fake_state(self) -> object:
        scenario_ids = np.array([[101, 202]], dtype=np.int64)
        x = np.array([[[[0.0], [2.0], [30.0]], [[0.0], [50.0], [70.0]]]], dtype=np.float32)
        y = np.zeros_like(x)
        vel_x = np.array([[[[1.0], [0.0], [0.0]], [[1.0], [0.0], [0.0]]]], dtype=np.float32)
        vel_y = np.zeros_like(x)
        length = np.full_like(x, 4.0)
        width = np.full_like(x, 2.0)
        valid = np.ones_like(x, dtype=bool)
        is_sdc = np.array([[[True, False, False], [True, False, False]]], dtype=bool)
        return SimpleNamespace(
            _scenario_id=scenario_ids,
            current_sim_trajectory=SimpleNamespace(
                x=x,
                y=y,
                vel_x=vel_x,
                vel_y=vel_y,
                length=length,
                width=width,
                valid=valid,
            ),
            object_metadata=SimpleNamespace(is_sdc=is_sdc),
        )

    def test_load_modulation_config_from_env_parses_and_collects_values(self) -> None:
        env = {
            "LATENTDRIVER_ACTION_MODULATION": "heuristic",
            "LATENTDRIVER_ACTION_MODULATION_MIN_SCALE": "0.4",
            "LATENTDRIVER_ACTION_MODULATION_TRACE_PATH": "/tmp/trace.jsonl",
        }
        config = load_modulation_config_from_env(env)
        self.assertEqual(config.mode, "heuristic")
        self.assertAlmostEqual(config.min_scale, 0.4)
        self.assertEqual(config.trace_path, "/tmp/trace.jsonl")
        self.assertEqual(
            collect_modulation_environment(env),
            {
                "LATENTDRIVER_ACTION_MODULATION": "heuristic",
                "LATENTDRIVER_ACTION_MODULATION_MIN_SCALE": "0.4",
                "LATENTDRIVER_ACTION_MODULATION_TRACE_PATH": "/tmp/trace.jsonl",
            },
        )

    def test_extract_risk_features_uses_interaction_radius(self) -> None:
        action = np.array([[4.0, 0.0, 0.0], [4.0, 0.0, 0.0]], dtype=np.float32)
        features = extract_risk_features(
            self._fake_state(),
            action,
            batch_dims=(1, 2),
            interaction_radius_meters=15.0,
        )
        np.testing.assert_allclose(features.min_distance_meters, np.array([2.0, 50.0], dtype=np.float32))
        np.testing.assert_allclose(features.min_ttc_seconds, np.array([2.0, 50.0], dtype=np.float32))
        np.testing.assert_allclose(features.interaction_density, np.array([1.0, 0.0], dtype=np.float32))
        np.testing.assert_allclose(features.ego_speed_mps, np.array([1.0, 1.0], dtype=np.float32))
        np.testing.assert_array_equal(features.scenario_ids, np.array([101, 202], dtype=np.int64))

    def test_heuristic_modulator_scales_only_risky_envs(self) -> None:
        action = np.array([[4.0, 0.0, 0.0], [4.0, 0.0, 0.0]], dtype=np.float32)
        config = load_modulation_config_from_env({"LATENTDRIVER_ACTION_MODULATION": "heuristic"})
        features = extract_risk_features(
            self._fake_state(),
            action,
            batch_dims=(1, 2),
            interaction_radius_meters=config.interaction_radius_meters,
        )
        modulator = HeuristicActionModulator(config)
        modulated, decision = modulator.modulate(action, features)
        self.assertLess(float(decision.scale[0]), 1.0)
        self.assertAlmostEqual(float(decision.scale[1]), 1.0, places=5)
        self.assertLess(float(modulated[0, 0]), float(action[0, 0]))
        self.assertAlmostEqual(float(modulated[1, 0]), float(action[1, 0]), places=5)

    def test_runtime_trace_records_batch_step_payload(self) -> None:
        action = np.array([[4.0, 0.0, 0.0], [4.0, 0.0, 0.0]], dtype=np.float32)
        with tempfile.TemporaryDirectory() as td:
            trace_path = f"{td}/trace.jsonl"
            config = load_modulation_config_from_env(
                {
                    "LATENTDRIVER_ACTION_MODULATION": "heuristic",
                    "LATENTDRIVER_ACTION_MODULATION_TRACE_PATH": trace_path,
                }
            )
            runtime = ActionModulationRuntime(config=config, batch_dims=(1, 2))
            modulated = runtime.apply(action=action, current_state=self._fake_state(), batch_index=3, step_index=7)
            self.assertEqual(modulated.shape, action.shape)
            summary = runtime.summary()
            self.assertEqual(summary["steps"], 1)
            self.assertGreater(summary["intervened_env_steps"], 0)
            with open(trace_path, encoding="utf-8") as handle:
                rows = [json.loads(line) for line in handle]
            self.assertEqual(len(rows), 1)
            self.assertEqual(rows[0]["batch_index"], 3)
            self.assertEqual(rows[0]["step_index"], 7)
            self.assertEqual(rows[0]["scenario_ids"], [101, 202])
            self.assertEqual(len(rows[0]["scale"]), 2)

    def test_runtime_factory_returns_none_when_disabled(self) -> None:
        with patch.dict("os.environ", {}, clear=True):
            runtime = build_action_modulation_runtime_from_env((1, 1))
        self.assertIsNone(runtime)


if __name__ == "__main__":
    unittest.main()
