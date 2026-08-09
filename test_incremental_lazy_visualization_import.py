"""Regression tests for the incremental CLI's lazy visualization imports."""

from __future__ import annotations

import os
import subprocess
import sys
import unittest
from pathlib import Path


class IncrementalLazyVisualizationImportTests(unittest.TestCase):
    def test_import_does_not_load_visualization_modules(self) -> None:
        repository_root = Path(__file__).resolve().parent
        script = """
import sys
import incremental_clustering

heavy_modules = {
    "cluster_visualization",
    "cluster_plotting",
    "umap_projection",
}
loaded = sorted(heavy_modules.intersection(sys.modules))
if loaded:
    raise SystemExit("unexpected visualization imports: " + ", ".join(loaded))
"""
        environment = os.environ.copy()
        environment["PYTHONDONTWRITEBYTECODE"] = "1"
        completed = subprocess.run(
            [sys.executable, "-c", script],
            cwd=repository_root,
            env=environment,
            capture_output=True,
            text=True,
            check=False,
        )
        self.assertEqual(
            completed.returncode,
            0,
            msg=completed.stderr or completed.stdout,
        )


if __name__ == "__main__":
    unittest.main()
