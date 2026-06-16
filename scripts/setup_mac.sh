#!/usr/bin/env bash
set -euo pipefail

PROJECT_DIR="$(cd "$(dirname "$0")/.." && pwd)"
CHECKPOINT="$PROJECT_DIR/weights/resnet50_MixVPR_4096.ckpt"
CHECKPOINT_URL="https://drive.usercontent.google.com/download?id=1vuz3PvnR7vxnDDLQrdHJaOA04SQrtk5L&export=download&confirm=t"

python -m pip install --upgrade pip
python -m pip install -r "$PROJECT_DIR/requirements.txt"

mkdir -p "$PROJECT_DIR/weights"

if [ ! -f "$CHECKPOINT" ]; then
  echo "Downloading MixVPR checkpoint..."
  curl -L "$CHECKPOINT_URL" -o "$CHECKPOINT"
fi

echo "Robot Visual Navigation V3 is ready."
