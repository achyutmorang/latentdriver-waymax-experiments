from __future__ import annotations

import tempfile
import unittest
import sys
from pathlib import Path
from unittest.mock import patch

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from latentdriver_waymax_experiments.preprocess_archive import archive_status, create_archive, extract_archive


class PreprocessArchiveTests(unittest.TestCase):
    def test_create_and_extract_full_archive(self) -> None:
        with tempfile.TemporaryDirectory() as td:
            root = Path(td)
            preprocessed = root / "preprocessed"
            source_preprocess = preprocessed / "full" / "val_preprocessed_path"
            source_intention = preprocessed / "full" / "val_intention_label"
            (source_preprocess / "map").mkdir(parents=True)
            (source_preprocess / "route").mkdir(parents=True)
            source_intention.mkdir(parents=True)
            (source_preprocess / "map" / "1.npy").write_bytes(b"map")
            (source_preprocess / "route" / "1.npy").write_bytes(b"route")
            (source_intention / "1.txt").write_text("straight_", encoding="utf-8")
            archive = root / "full_preprocess_cache.tar"
            target = root / "local"

            with patch("latentdriver_waymax_experiments.preprocess_archive.preprocessed_root", return_value=preprocessed):
                created = create_archive(mode="full", archive_path=archive)
                status = archive_status(mode="full", archive_path=archive, target_root=target)
                extracted = extract_archive(mode="full", archive_path=archive, target_root=target)

            self.assertTrue(archive.is_file())
            self.assertGreater(created["archive_size_bytes"], 0)
            self.assertTrue(status["archive_exists"])
            self.assertEqual((target / "full" / "val_preprocessed_path" / "map" / "1.npy").read_bytes(), b"map")
            self.assertEqual((target / "full" / "val_preprocessed_path" / "route" / "1.npy").read_bytes(), b"route")
            self.assertEqual((target / "full" / "val_intention_label" / "1.txt").read_text(encoding="utf-8"), "straight_")
            self.assertEqual(extracted["target_root"], str(target))


if __name__ == "__main__":
    unittest.main()
