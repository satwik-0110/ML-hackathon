"""Central configuration: paths and fixed constants used across the project."""
import os
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent

# Raw data folder containing `train/` and `student_test/` (one sub-folder per trip).
DATA_DIR = Path(os.environ.get("PATHSNAP_DATA_DIR", ROOT.parent / "data" / "Student folder"))

ARTIFACT_DIR = ROOT / "artifacts"
RESULTS_DIR = ROOT / "results"
CACHE_DIR = ROOT / "cache"          # derived datasets (not committed)

SEED = 42

# --- candidate generation ---
CAND_RADIUS_M = 50.0      # search radius around each GPS point
CAND_MAX_K = 10           # keep at most K nearest directed arcs
CAND_FALLBACK_M = 200.0   # if nothing within the radius, widen the search to this

# --- label derivation ---
LABEL_MAX_DIST_M = 50.0   # GPS points further than this from the aligned route arc get no label

# --- training ---
TRAIN_STRIDE = 5          # use every 5th GPS point per training trip (1 Hz points are near-duplicates)
CV_FOLDS = 5
EXCLUDED_TRAIN_TRIPS = {"00000033", "00000054"}  # reference route ~130 m away from GPS (see EDA)
