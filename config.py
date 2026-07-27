"""
Central configuration + shared constants.
"""


# ---- Storage locations ------------------------------------------------------
DB_PATH = "violations.db"
SNAPSHOT_DIR = "snapshots"
UPLOAD_DIR = "uploads"
FEEDS_CONFIG_PATH = "feeds.json"   # feeds added from the dashboard get persisted here

# ---- Upload / detection tuning ---------------------------------------------
ALLOWED_VIDEO_EXT = {".mp4", ".avi", ".mov", ".mkv", ".wmv"}
MIN_THRESH = 0.3

# Per-track: once a violating track ID is logged, it won't be logged again
# until it has been GONE (unseen) for this many seconds. Prevents duplicate
# rows for the same person while they stay in frame.
TRACK_RELOG_AFTER = 60.0

# Tracker config shipped with ultralytics ("bytetrack.yaml" or "botsort.yaml")
TRACKER_CFG = "bytetrack.yaml"

# Default inference device. main.py overrides this at startup after probing
# for CUDA: "0" (first GPU) or "cpu". Requires a CUDA build of torch for GPU.
DEVICE = "cpu"

# ---- Label matching keywords ------------------------------------------------
# keywords (normalized) that identify each item
HELMET_KEYS = ("helmet", "hardhat")
VEST_KEYS = ("vest",)
# tokens that indicate the NEGATIVE / missing case
NEG_KEYS = ("no", "without", "missing")

# Colors for non-violation bounding boxes (BGR, cycled by class index)
BBOX_COLORS = [(164, 120, 87), (68, 148, 228), (93, 97, 209), (178, 182, 133),
               (88, 159, 106), (96, 202, 231), (159, 124, 168), (169, 162, 241),
               (98, 118, 150), (172, 176, 184)]
