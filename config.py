# =========================================================
# GOOGLE API
# =========================================================

API_KEY = "AIzaSyDOUCm3MkUnahNvQK6hPoHyfHjQWXN7uME"

# =========================================================
# DEFAULT MAP
# =========================================================

DEFAULT_CENTER = [
    50.104493,
    14.394761,
]

DEFAULT_ZOOM = 16

# =========================================================
# ROUTE SETTINGS
# =========================================================

STEP_M = 5
NETWORK_TYPE = "walk"

# =========================================================
# STREET VIEW
# =========================================================

IMAGE_SIZE = "1280x720"

FOV = 120
PITCH = -5

RADIUS = 8

MAX_DISTANCE_TO_PANO = 8

HEADING_MODE = "route"

# =========================================================
# ORB MATCHING
# =========================================================

ORB_FEATURES = 3000
ORB_DISTANCE_THRESHOLD = 55

# =========================================================
# CNN MATCHING
# =========================================================

CNN_IMAGE_SIZE = 224

# =========================================================
# HYBRID MATCHING
# =========================================================

ORB_WEIGHT = 0.4
CNN_WEIGHT = 0.6

# =========================================================
# VIDEO / REPLAY
# =========================================================

VIDEO_FRAME_EXTRACTION_FPS = 1

REPLAY_SEARCH_WINDOW = 5

# =========================================================
# VPR BACKEND (embedded in this V3 folder)
# =========================================================

import os

PROJECT_DIR = os.path.dirname(os.path.abspath(__file__))
VPR_BACKEND_DIR = os.path.join(PROJECT_DIR, "vpr_backend")
MIXVPR_CHECKPOINT = os.environ.get(
    "MIXVPR_CHECKPOINT",
    os.path.join(PROJECT_DIR, "weights", "resnet50_MixVPR_4096.ckpt"),
)
VPR_CACHE_DIR = os.path.join(PROJECT_DIR, "cache", "vpr")
VPR_OUTPUT_DIR = os.path.join(PROJECT_DIR, "data", "results", "vpr")
VPR_BATCH_SIZE = 8
VPR_FILTER_ENABLED = True
VPR_INITIAL_SEARCH_SIZE = 5
VPR_WINDOW_FORWARD = 5
VPR_MAX_WINDOW_DISTANCE_M = 30.0
