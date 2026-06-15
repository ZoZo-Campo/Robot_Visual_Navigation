#!/usr/bin/env bash
set -euo pipefail

PROJECT_DIR="$(cd "$(dirname "$0")/.." && pwd)"
BACKEND_DIR="${DEV_MATCHING_ROOT:-$(cd "$PROJECT_DIR/../Dev_Matching_V2" && pwd)}"

python -m pip install --upgrade pip
python -m pip install -r "$PROJECT_DIR/requirements.txt"
python -m pip install -e "$BACKEND_DIR"

test -f "$BACKEND_DIR/weights/resnet50_MixVPR_4096.ckpt" || {
  echo "Missing MixVPR checkpoint in $BACKEND_DIR/weights"
  exit 1
}

echo "Robot Visual Navigation V3 is ready."
