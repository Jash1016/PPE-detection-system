"""
Central configuration + shared constants.
"""


# ---- Models -----------------------------------------------------------------
# Each feed picks the net that suits its camera. The two do NOT share a label
# vocabulary, so every entry declares how its class names are to be read:
#   "negative" - the model names the violation itself ("No Helmets"), matched
#                by classify_label()
#   "positive" - the model names only the gear that IS present (head/helmet/
#                vest). A bare head is an ABSENCE of a box, so violations are
#                worked out geometrically -- see detection.derive_violations()
# "path" is relative to the folder holding these scripts.
MODELS = {
    "close": {
        "label": "Close range",
        "hint": "Gates and doorways, where people fill much of the frame.",
        "path": "best.pt",
        "labels": "positive",
    },
    "distance": {
        "label": "Distance",
        "hint": "Yards and wide shots, where people are small in frame.",
        "path": "my_model.pt",
        "labels": "negative",
    },
}

# Used by any feed that doesn't name a model -- including every feed saved to
# feeds.json before per-feed selection existed.
DEFAULT_MODEL = "distance"

# Default model filename. If --model isn't passed, main.py looks for this file
# next to the scripts, then falls back to any *.pt it finds there. Derived from
# the registry so the default lives in exactly one place.
MODEL_PATH = MODELS[DEFAULT_MODEL]["path"]

# ---- Authentication ---------------------------------------------------------
# Flask signs session cookies with a secret key. It's generated once and kept
# in this file so logins survive a server restart.
SECRET_KEY_FILE = ".secret_key"
SESSION_HOURS = 12               # how long a login stays valid

# On first run (empty users table) this account is created so you can get in.
# CHANGE THE PASSWORD from the Users panel immediately.
DEFAULT_ADMIN_USER = "admin"
DEFAULT_ADMIN_PASS = "admin"

# ---- Storage locations ------------------------------------------------------
DB_PATH = "violations.db"
SNAPSHOT_DIR = "snapshots"
UPLOAD_DIR = "uploads"
FEEDS_CONFIG_PATH = "feeds.json"   # feeds added from the dashboard get persisted here

# ---- Upload / detection tuning ---------------------------------------------
ALLOWED_VIDEO_EXT = {".mp4", ".avi", ".mov", ".mkv", ".wmv"}
# Display floor: boxes below this are dropped by NMS and never drawn. Low, so
# a marginal detection still shows up on screen for a human to judge.
MIN_THRESH = 0.30
# Logging floor: a violation is only written to the DB at this confidence or
# above. Higher than MIN_THRESH because a false row in the violation log is
# worse than a box that appears on screen but never becomes a record.
LOG_THRESH = 0.55

# Per-track: once a violating track ID is logged, it won't be logged again
# until it has been GONE (unseen) for this many seconds. Prevents duplicate
# rows for the same person while they stay in frame.
TRACK_RELOG_AFTER = 60.0

# Per-feed backstop for TRACK_RELOG_AFTER: after logging a violation type on a
# feed, ignore further rows of that type on that feed for this many seconds.
# TRACK_RELOG_AFTER only works while IDs are stable -- a flickering track picks
# up a new ID and looks like a brand new person, so it slips straight past it.
# 8s is long enough to swallow that churn, short enough that a genuinely new
# violator walking into frame is still recorded promptly.
FEED_RELOG_COOLDOWN = 8.0

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
# the body part a positive-label model detects, which gear is matched against
HEAD_KEYS = ("head", "person")

# ---- Positive-label geometry (close-range model) ----------------------------
# Gear counts as WORN when at least this much of the gear box sits inside the
# body region being tested. Deliberately not IoU: a vest box and the torso
# region it is tested against differ a lot in size, and IoU reads that size
# mismatch as "no overlap" even when the vest is plainly on the person.
GEAR_OVERLAP_MIN = 0.30
# best.pt has no person or torso class, so the torso is estimated from the head
# box in multiples of its width/height. ~3 heads wide and ~3 heads tall below
# the chin approximates shoulders-to-hips on an upright adult.
TORSO_WIDTH_HEADS = 3.0
TORSO_HEIGHT_HEADS = 3.0

# Colors for non-violation bounding boxes (BGR, cycled by class index)
BBOX_COLORS = [(164, 120, 87), (68, 148, 228), (93, 97, 209), (178, 182, 133),
               (88, 159, 106), (96, 202, 231), (159, 124, 168), (169, 162, 241),
               (98, 118, 150), (172, 176, 184)]
