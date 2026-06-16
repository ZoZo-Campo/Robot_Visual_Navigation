#!/usr/bin/env bash
set -euo pipefail

PROJECT_DIR="$(cd "$(dirname "$0")/.." && pwd)"
CHECKPOINT="$PROJECT_DIR/weights/resnet50_MixVPR_4096.ckpt"

python -m pip install --upgrade pip
python -m pip install -r "$PROJECT_DIR/requirements.txt"

mkdir -p "$PROJECT_DIR/weights"

if [ ! -f "$CHECKPOINT" ]; then
  echo "Missing local checkpoint: $CHECKPOINT"
  echo "The delivered Robot_Visual_Navigation_V3 folder must include weights/resnet50_MixVPR_4096.ckpt"
  exit 1
fi

echo "Robot Visual Navigation V3 is ready."
