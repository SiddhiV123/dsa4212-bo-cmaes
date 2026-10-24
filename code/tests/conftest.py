"""Make code/ importable so tests run from the repo root or from code/."""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))