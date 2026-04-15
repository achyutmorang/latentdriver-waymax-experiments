from __future__ import annotations

import tempfile
import unittest
import sys
from pathlib import Path
from unittest.mock import patch

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from latentdriver_waymax_experiments.preprocess_archive import (
    archive_status,
    create_archive,
    create_shard_archives,
    extract_archive,
    extract_shard_archives,
)


class PreprocessArchiveTests(unittest.TestCase):
    def _write_preprocess_triple(self, preprocessed: Path, mode: str, scenario_id: str) -> None:
        source_preprocess = preprocessed / mode / "val_preprocessed_path"
        source_intention = preprocessed / mode / "val_intention_label"
        (source_preprocess / "map").mkdir(parents=True, exist_ok=True)
        (source_preprocess / "route").mkdir(parents=True, exist_ok=True)
        source_intention.mkdir(parents=True, exist_ok=True)
        (source_preprocess / "map" / f"{scenario_id}.npy").write_bytes(f"map-{scenario_id}".encode("utf-8"))
        (source_preprocess / "route" / f"{scenario_id}.npy").write_bytes(f"route-{scenario_id}".encode("utf-8"))
        (source_intention / f"{scenario_id}.txt").write_text(f"intention-{scenario_id}", encoding="utf-8")

    def test_create_and_extract_full_archive(self) -> None:
        with tempfile.TemporaryDirectory() as td:
            root = Path(td)
            preprocessed = root / "preprocessed"
            self._write_preprocess_triple(preprocessed, "full", "1")
            archive = root / "full_preprocess_cache.tar"
            target = root / "local"

            with patch("latentdriver_waymax_experiments.preprocess_archive.preprocessed_root", return_value=preprocessed):
                created = create_archive(mode="full", archive_path=archive)
                status = archive_status(mode="full", archive_path=archive, target_root=target)
                extracted = extract_archive(mode="full", archive_path=archive, target_root=target)

            self.assertTrue(archive.is_file())
            self.assertGreater(created["archive_size_bytes"], 0)
            self.assertTrue(status["archive_exists"])
            self.assertEqual((target / "full" / "val_preprocessed_path" / "map" / "1.npy").read_bytes(), b"map-1")
            self.assertEqual((target / "full" / "val_preprocessed_path" / "route" / "1.npy").read_bytes(), b"route-1")
            self.assertEqual((target / "full" / "val_intention_label" / "1.txt").read_text(encoding="utf-8"), "intention-1")
            self.assertEqual(extracted["target_root"], str(target))

    def test_create_and_extract_interactive_pilot_archive(self) -> None:
        with tempfile.TemporaryDirectory() as td:
            root = Path(td)
            preprocessed = root / "preprocessed"
            self._write_preprocess_triple(preprocessed, "interactive_pilot", "pilot-1")
            archive = root / "interactive_pilot_preprocess_cache.tar"
            target = root / "local"

            with patch("latentdriver_waymax_experiments.preprocess_archive.preprocessed_root", return_value=preprocessed):
                created = create_archive(mode="interactive_pilot", archive_path=archive)
                status = archive_status(mode="interactive_pilot", archive_path=archive, target_root=target)
                extracted = extract_archive(mode="interactive_pilot", archive_path=archive, target_root=target)

            self.assertTrue(archive.is_file())
            self.assertGreater(created["archive_size_bytes"], 0)
            self.assertTrue(status["archive_exists"])
            self.assertEqual(
                (target / "interactive_pilot" / "val_preprocessed_path" / "map" / "pilot-1.npy").read_bytes(),
                b"map-pilot-1",
            )
            self.assertEqual(
                (target / "interactive_pilot" / "val_intention_label" / "pilot-1.txt").read_text(encoding="utf-8"),
                "intention-pilot-1",
            )
            self.assertEqual(extracted["target_root"], str(target))

    def test_create_and_extract_shard_archives_skips_completed_parts(self) -> None:
        with tempfile.TemporaryDirectory() as td:
            root = Path(td)
            preprocessed = root / "preprocessed"
            for scenario_id in ("001", "002", "003", "004", "005"):
                self._write_preprocess_triple(preprocessed, "full", scenario_id)
            archive_dir = root / "shard_archives"
            target = root / "local"

            with patch("latentdriver_waymax_experiments.preprocess_archive.preprocessed_root", return_value=preprocessed):
                created = create_shard_archives(mode="full", archive_dir=archive_dir, shards=3)
                recreated = create_shard_archives(mode="full", archive_dir=archive_dir, shards=3)
                status = archive_status(mode="full", archive_dir=archive_dir, target_root=target)
                extracted = extract_shard_archives(mode="full", archive_dir=archive_dir, target_root=target)

            archives = sorted(archive_dir.glob("shard-*.tar"))
            self.assertEqual(len(archives), 3)
            self.assertEqual(created["scenario_count"], 5)
            self.assertEqual(created["file_count"], 15)
            self.assertEqual(recreated["shards_skipped"], 3)
            self.assertEqual(extracted["shards_total"], 3)
            self.assertGreater(status["shard_archive_count"], 0)
            self.assertEqual(
                (target / "full" / "val_preprocessed_path" / "map" / "005.npy").read_bytes(),
                b"map-005",
            )
            self.assertEqual(
                (target / "full" / "val_intention_label" / "003.txt").read_text(encoding="utf-8"),
                "intention-003",
            )


if __name__ == "__main__":
    unittest.main()
