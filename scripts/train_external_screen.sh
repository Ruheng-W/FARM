#!/usr/bin/env bash
# Train FARM on the full 10-cohort dataset (no holdout) and score the
# 144-drug external-screen test set (Maier et al. 2018 derived subset).
#
# Usage:
#   scripts/train_external_screen.sh /path/to/FARM_dataset_v1 /path/to/output

set -euo pipefail

DATA_ROOT="${1:?usage: $0 <data-root> <out-dir>}"
OUT_DIR="${2:?usage: $0 <data-root> <out-dir>}"

mkdir -p "$OUT_DIR"

python -m farm.train \
    --mode full \
    --data-root "$DATA_ROOT" \
    --out-dir "$OUT_DIR"

# After training, score the external screen using the trained checkpoint.
python -m farm.predict \
    --checkpoint "$OUT_DIR/checkpoint.h5" \
    --external-screen "$DATA_ROOT/external_screen" \
    --out "$OUT_DIR/external_screen_predictions.csv"

echo "External-screen predictions written to $OUT_DIR/external_screen_predictions.csv"
