"""Lets the scripts in this folder import Experiment 3's modules (data, lda, metrics, ...)."""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "exp3"))
