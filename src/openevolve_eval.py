"""Evaluator file OpenEvolve loads by path; it delegates to the harness evaluator."""
import sys
from pathlib import Path

ROOT = str(Path(__file__).resolve().parent.parent)
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

from src.openevolve_arm import evaluate  # noqa: E402,F401
