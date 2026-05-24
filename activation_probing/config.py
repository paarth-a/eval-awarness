"""Configuration for the activation probing pipeline.

Override via env vars where listed; otherwise edit defaults.
"""
import os
from pathlib import Path

# ---------------------------------------------------------------------------
# Repo paths
# ---------------------------------------------------------------------------
# Activation probing dir lives one level under the parent repo.
PROBE_DIR = Path(__file__).resolve().parent
REPO_ROOT = PROBE_DIR.parent

CACHE_DIR = PROBE_DIR / "cache"
PROBES_DIR = PROBE_DIR / "probes"
RESULTS_DIR = PROBE_DIR / "results"
for d in (CACHE_DIR, PROBES_DIR, RESULTS_DIR):
    d.mkdir(parents=True, exist_ok=True)

# ---------------------------------------------------------------------------
# Model
# ---------------------------------------------------------------------------
# Override via env var: ACTPROBE_MODEL=Qwen/Qwen3-32B
MODEL_NAME = os.environ.get("ACTPROBE_MODEL", "Qwen/Qwen3-8B")

# Layers to probe. None = auto-pick (every 4th layer + last 3).
# Override via env var: ACTPROBE_LAYERS=10,15,20,25
_layers_env = os.environ.get("ACTPROBE_LAYERS")
LAYER_INDICES: list[int] | None = (
    [int(x) for x in _layers_env.split(",")] if _layers_env else None
)

# Max token length for input truncation
MAX_TOKENS = int(os.environ.get("ACTPROBE_MAX_TOKENS", "4096"))

# Use float16 for inference (half precision); save activations as float32
DTYPE = "float16"

# ---------------------------------------------------------------------------
# Benchmarks and conditions
# ---------------------------------------------------------------------------
BENCHMARKS = ["needham", "hawthorne", "mask", "agentharm", "bbh", "malt"]

# Conditions to extract activations for. orig_eval and orig_deploy are required
# for probe training; the others are test conditions.
CONDITIONS = [
    "orig_eval",     # original E-cases (label=1)
    "orig_deploy",   # original D-cases (label=0)
    "v1_removed",    # v1 aggressive removal (parent's tells/data/{b}/reverse/...)
    "v2_removed",    # v2 scaffolding-only filter (parent's reverse_v2/)
    "v3_removed",    # v3 with preservation gate (parent's reverse_v3/)
    # "v4_removed",  # uncomment if v4 outputs exist
]

# How many samples per (benchmark, condition). None = use everything available.
# Override via env var: ACTPROBE_N_PER=100
_n_per_env = os.environ.get("ACTPROBE_N_PER")
N_PER_CONDITION: int | None = int(_n_per_env) if _n_per_env else None

# ---------------------------------------------------------------------------
# Probe training
# ---------------------------------------------------------------------------
PROBE_C = float(os.environ.get("ACTPROBE_C", "1.0"))   # L2 regularization strength inverse
PROBE_CV_FOLDS = 5
RNG_SEED = 42

# ---------------------------------------------------------------------------
# Bootstrap CI for figure
# ---------------------------------------------------------------------------
N_BOOTSTRAP = 1000
BOOTSTRAP_CI = 0.95
