# `farm/` — core model code

| File | Purpose |
|---|---|
| `model.py` | Drug Graphformer encoder + drug–gene cross-attention decoder + classification head. The mixing weights α/β/δ in the Graphformer attention are hard-coded to the paper-reported configuration (α = 0.6, β = 0.2, δ = 0.2 in this file). |
| `smile_rel_dist_interpreter.py` | Parses canonical SMILES into an atom set + bond types + all-pairs shortest-path distance matrix (RDKit + Dijkstra). |
| `process_data.py` | Joins drug-graph inputs with isolate-level KO-cluster gene features (presence + mutation). Produces the tensors consumed by `model.py`. |
| `data.py` | High-level loaders for the harmonised cohort phenotype.txt and VAMP.txt files plus the `(drug, isolate)` example builder used by `train.py`. |
| `train.py` | Command-line training entry; dispatches to cohort-holdout / leave-one-drug-out / full-data modes. |
| `train_cohort_holdout.py` | Inner training loop shared by the three training modes. |
| `predict.py` | Single-pair (`predict_one`) and batched (`predict_batch`) inference helpers. |
| `eval.py` | Accuracy, balanced accuracy, AUROC, AUPRC, F1, MCC, sensitivity, specificity, and isolate-level bootstrap 95 % CIs. |
