"""Training entry-point for FARM.

This is a thin wrapper that dispatches to the cohort-holdout, leave-one-drug-out,
or full-training mode used in the paper. The actual training loop lives in
``train_cohort_holdout.py``; here we expose a uniform CLI.

Usage
-----
    # Cohort-holdout: train on 9 cohorts, score the 10th
    python -m farm.train --mode cohort-holdout \
        --data-root /path/to/FARM_dataset_v1 \
        --test-cohort CF \
        --out-dir runs/cohort_holdout_CF

    # Leave-one-drug-out
    python -m farm.train --mode lodo \
        --data-root /path/to/FARM_dataset_v1 \
        --test-drug amikacin \
        --out-dir runs/lodo_amikacin

    # Train on the full 10-cohort dataset (no holdout); used for the
    # external-screen evaluation in Figure 6.
    python -m farm.train --mode full \
        --data-root /path/to/FARM_dataset_v1 \
        --out-dir runs/full
"""
import argparse
import json
import os
import sys


def _parse_args(argv):
    p = argparse.ArgumentParser(description='Train a FARM model.')
    p.add_argument('--mode', required=True,
                   choices=['cohort-holdout', 'lodo', 'full'],
                   help='Which evaluation protocol to use.')
    p.add_argument('--data-root', required=True,
                   help='Root directory of the unpacked FARM_dataset_v1 '
                        '(must contain ten_cohorts/ and reference/).')
    p.add_argument('--test-cohort', default=None,
                   help='For --mode cohort-holdout: which cohort to hold out.')
    p.add_argument('--test-drug', default=None,
                   help='For --mode lodo: which drug to hold out.')
    p.add_argument('--out-dir', required=True,
                   help='Output directory (will be created).')
    p.add_argument('--epochs', type=int, default=50)
    p.add_argument('--batch-size', type=int, default=512)
    p.add_argument('--lr', type=float, default=5e-5,
                   help='Adam learning rate. Default 5e-5 matches the paper.')
    p.add_argument('--alpha', type=float, default=0.6,
                   help='Drug Graphformer atom-attention weight (α).')
    p.add_argument('--beta', type=float, default=0.2,
                   help='Drug Graphformer bond-attention weight (β).')
    p.add_argument('--delta', type=float, default=0.3,
                   help='Drug Graphformer position-attention weight (δ).')
    return p.parse_args(argv)


def main(argv=None):
    args = _parse_args(argv if argv is not None else sys.argv[1:])
    os.makedirs(args.out_dir, exist_ok=True)
    with open(os.path.join(args.out_dir, 'config.json'), 'w') as f:
        json.dump(vars(args), f, indent=2)

    # The heavy lifting is in train_cohort_holdout.py — import and call
    # the right entry point. We import lazily so that --help does not pull
    # in TensorFlow.
    from . import train_cohort_holdout as legacy

    if args.mode == 'cohort-holdout':
        if not args.test_cohort:
            raise SystemExit('--test-cohort is required for cohort-holdout mode')
        legacy.run_cohort_holdout(args)
    elif args.mode == 'lodo':
        if not args.test_drug:
            raise SystemExit('--test-drug is required for lodo mode')
        legacy.run_lodo(args)
    elif args.mode == 'full':
        legacy.run_full(args)


if __name__ == '__main__':
    main()
