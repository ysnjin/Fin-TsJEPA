from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]

CACHE_DIR = REPO_ROOT / "data" / "cache_sp1500"
PANEL_PATH = REPO_ROOT / "panels" / "panel_full845.npz"
CKPT_DIR = REPO_ROOT / "checkpoints"
RESULTS_DIR = REPO_ROOT / "results"

EVAL_START, EVAL_END = "1990-01-01", "2025-12-31"
SEEDS = [7, 42, 99]
