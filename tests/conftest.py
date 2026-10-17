"""Pytest configuration: puts the project's src/ folder on sys.path so tests can import pipeline modules directly."""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
