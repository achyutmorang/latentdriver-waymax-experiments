from __future__ import annotations

import json
import subprocess
import tempfile
import unittest
from pathlib import Path


class ColabRunnerNotebookTests(unittest.TestCase):
    def test_runner_notebook_code_cells_are_shell_only(self) -> None:
        repo_root = Path(__file__).resolve().parents[1]
        notebook_path = repo_root / "notebooks" / "latentdriver_colab_runner.ipynb"
        notebook = json.loads(notebook_path.read_text(encoding="utf-8"))

        for index, cell in enumerate(notebook["cells"]):
            if cell["cell_type"] != "code":
                continue
            source = "".join(cell["source"])
            self.assertTrue(source.startswith("%%bash\n"), f"code cell {index} must be a %%bash terminal cell")
            with tempfile.NamedTemporaryFile("w", suffix=".sh", delete=False) as handle:
                handle.write(source.split("\n", 1)[1])
                script_path = Path(handle.name)
            try:
                proc = subprocess.run(["bash", "-n", str(script_path)], text=True, capture_output=True, check=False)
            finally:
                script_path.unlink(missing_ok=True)
            self.assertEqual(proc.returncode, 0, f"code cell {index} is not valid bash: {proc.stderr}")


if __name__ == "__main__":
    unittest.main()
