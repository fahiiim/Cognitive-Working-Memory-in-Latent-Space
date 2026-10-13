"""Rebuild every paper figure and table from the saved run outputs (no training).

Executes the analysis notebooks in order; each reads results/runs/*/ and writes PDFs to
paper/figures and LaTeX tables to paper/tables.

    .venv\\Scripts\\python.exe scripts\\make_figures.py            # notebooks 02-05 and 07-09
    .venv\\Scripts\\python.exe scripts\\make_figures.py 09         # only notebook 09
"""

from __future__ import annotations

import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
ANALYSIS = ["02", "03", "04", "05", "07", "08", "09"]

if __name__ == "__main__":
    which = sys.argv[1:] or ANALYSIS
    sys.exit(subprocess.call([sys.executable, str(ROOT / "scripts" / "build_notebooks.py"), "--run", *which], cwd=ROOT))
