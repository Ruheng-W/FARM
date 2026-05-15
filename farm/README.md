# `farm/` — core model code

| File | Purpose |
|---|---|
| `model.py` | Drug Graphformer encoder + drug–gene cross-attention decoder + classification head. The mixing weights α/β/δ in the Graphformer attention are hard-coded to the paper-reported configuration (0.6 / 0.2 / 0.2 in this file; full sweep available in `Supplementary Table 8`). |
| `smile_rel_dist_interpreter.py` | Parses canonical SMILES into atom set + bond types + all-pairs shortest-path distance matrix (via RDKit + Dijkstra). |
| `process_data.py` | Joins drug graph inputs with isolate-level KO-cluster gene features (presence + mutation). Produces the tensors consumed by `model.py`. |
| `train_cohort_holdout.py` | Cohort-level holdout training entry: hold out one cohort, train on the remaining 9, save predictions on the held-out cohort. Iterates over all 10 cohorts. |

The original 1,445-line model file (`drug_transformer3_w060020020.py`) was
copied as `model.py` with minimal changes. A planned refactor will pull
α / β / δ and other hyperparameters into a `config.yaml` so the file no
longer needs to be cloned for hyperparameter sweeps.
