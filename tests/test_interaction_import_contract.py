"""Regression test for the interaction/pairwise import contract."""
from pathlib import Path
import subprocess
import sys


def test_complete_launcher_imports_and_builds_cli():
    root = Path(__file__).resolve().parents[1]
    launcher = root / "generate_nuplan_predicates_modular_v9_5_31.py"
    completed = subprocess.run(
        [sys.executable, str(launcher), "--help"],
        cwd=str(root),
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
        check=False,
    )
    assert completed.returncode == 0, completed.stderr
    assert "--categories" in completed.stdout
    assert "--num-workers" in completed.stdout
