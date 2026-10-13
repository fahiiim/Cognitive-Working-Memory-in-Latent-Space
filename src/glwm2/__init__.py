"""GLWM-2: learned termination detection for recurrent latent reasoners."""

from pathlib import Path

__version__ = "0.1.0"

PROJECT_ROOT = Path(__file__).resolve().parents[2]
DATA_DIR = PROJECT_ROOT / "data"
RESULTS_DIR = PROJECT_ROOT / "results"
CKPT_DIR = PROJECT_ROOT / "checkpoints"
PAPER_DIR = PROJECT_ROOT / "paper"
