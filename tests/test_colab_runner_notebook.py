from __future__ import annotations

import json
import subprocess
import tempfile
import unittest
from pathlib import Path


class ColabRunnerNotebookTests(unittest.TestCase):
    def test_runner_notebook_allows_only_two_platform_python_cells(self) -> None:
        repo_root = Path(__file__).resolve().parents[1]
        notebook_path = repo_root / "notebooks" / "latentdriver_colab_runner.ipynb"
        notebook = json.loads(notebook_path.read_text(encoding="utf-8"))

        python_cell_indices = []
        for index, cell in enumerate(notebook["cells"]):
            if cell["cell_type"] != "code":
                continue
            source = "".join(cell["source"])
            if source.startswith("%%bash\n"):
                with tempfile.NamedTemporaryFile("w", suffix=".sh", delete=False) as handle:
                    handle.write(source.split("\n", 1)[1])
                    script_path = Path(handle.name)
                try:
                    proc = subprocess.run(["bash", "-n", str(script_path)], text=True, capture_output=True, check=False)
                finally:
                    script_path.unlink(missing_ok=True)
                self.assertEqual(proc.returncode, 0, f"code cell {index} is not valid bash: {proc.stderr}")
                continue
            compile(source, f"<notebook cell {index}>", "exec")
            python_cell_indices.append(index)

        python_sources = ["".join(notebook["cells"][index]["source"]) for index in python_cell_indices]
        self.assertEqual(len(python_sources), 2, "only Drive mount and GCS auth may use notebook-native Python")
        self.assertIn("drive.mount(DRIVE_MOUNTPOINT)", python_sources[0])
        self.assertIn("auth.authenticate_user()", python_sources[1])


if __name__ == "__main__":
    unittest.main()
