#!/usr/bin/env bash
# Leave-one-drug-out: for each of the 36 training drugs, train on the
# other 35 and score the held-out drug.
#
# Usage:
#   scripts/leave_one_drug_out.sh /path/to/FARM_dataset_v1 /path/to/output

set -euo pipefail

DATA_ROOT="${1:?usage: $0 <data-root> <out-dir>}"
OUT_DIR="${2:?usage: $0 <data-root> <out-dir>}"

mkdir -p "$OUT_DIR"

DRUGS_FILE="$DATA_ROOT/reference/drug_smiles_36training.tsv"

# Skip the header line; column 1 is the drug name.
tail -n +2 "$DRUGS_FILE" | cut -f1 | while read -r drug; do
    safe="${drug// /_}"
    echo ">>> LODO fold: test drug = $drug"
    python -m farm.train \
        --mode lodo \
        --data-root "$DATA_ROOT" \
        --test-drug "$drug" \
        --out-dir "$OUT_DIR/$safe"
done

echo "All LODO folds finished. Predictions per drug are in $OUT_DIR/<drug>/predictions.csv."
