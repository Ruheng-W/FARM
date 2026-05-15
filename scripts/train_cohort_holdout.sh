#!/usr/bin/env bash
# Train FARM under the 10-fold cohort-holdout protocol.
# For each cohort in turn, train on the remaining nine and score the held-out one.
#
# Usage:
#   scripts/train_cohort_holdout.sh /path/to/FARM_dataset_v1 /path/to/output

set -euo pipefail

DATA_ROOT="${1:?usage: $0 <data-root> <out-dir>}"
OUT_DIR="${2:?usage: $0 <data-root> <out-dir>}"

mkdir -p "$OUT_DIR"

COHORTS=(antibiogram ARIsolateBank AstraZeneca CF Chile German PATRIC Rabin Shelburne TIDB)

for c in "${COHORTS[@]}"; do
    echo ">>> cohort-holdout fold: test = $c"
    python -m farm.train \
        --mode cohort-holdout \
        --data-root "$DATA_ROOT" \
        --test-cohort "$c" \
        --out-dir "$OUT_DIR/$c"
done

echo "All 10 folds finished. Per-isolate predictions are in $OUT_DIR/<cohort>/predictions.csv."
