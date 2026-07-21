# =========================================================
# PROJECT
# =========================================================

import os

PROJECT_DIR = os.path.dirname(os.path.abspath(__file__))

# =========================================================
# GOOGLE API
# =========================================================

API_KEY = "YOUR_API_KEY_HERE"

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
ROUTE_CACHE_DIR = os.path.join(PROJECT_DIR, "cache", "routes")
ROUTE_DOWNLOAD_TIMEOUT_S = 60
ROUTE_MAX_RADIUS_M = 2500
ROUTE_USE_FALLBACK_DIRECT = True

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
# HISTORICAL STREET VIEW
# =========================================================

HISTORICAL_IMAGE_SIZE = "640x640"
HISTORICAL_FOV = 100
HISTORICAL_PITCH = 0
HISTORICAL_BEFORE_DATE = "2015-01"
HISTORICAL_AFTER_DATE = None
HISTORICAL_CANDIDATE_LIMIT = 3
HISTORICAL_DOWNLOAD_SLEEP_S = 0.2

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

# =========================================================
# V4 VISUAL PATH GUIDANCE
# =========================================================
# ROS convention: positive angular_z turns left, negative turns right.
GUIDANCE_LATERAL_GAIN = 0.75
GUIDANCE_HEADING_GAIN = 0.35
GUIDANCE_MAX_ANGULAR_Z = 0.60
GUIDANCE_NOMINAL_LINEAR_X = 0.25
GUIDANCE_DEADBAND = 0.035
GUIDANCE_MIN_CONFIDENCE = 0.18
GUIDANCE_PROCESSING_FPS = 10
GUIDANCE_REFERENCE_ALIGNMENT_GAIN = 0.75
GUIDANCE_ROUTE_TURN_GAIN = 0.45
GUIDANCE_LOCAL_STABILITY_GAIN = 0.20
GUIDANCE_REFERENCE_INTERVAL_S = 2.0
GUIDANCE_ROUTE_LOOKAHEAD_IMAGES = 3
