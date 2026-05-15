# Notebooks

Hands-on demonstrations of the model.

| Notebook | Purpose |
|---|---|
| `01_quick_start.ipynb` | Score one (drug, isolate) pair with a trained checkpoint. |
| `02_attention_visualization.ipynb` | Atom-level saliency and gene-level cross-attention plots. |
| `03_external_screen_eval.ipynb` | Evaluate the 144-drug external screen end-to-end. |

Each notebook has a configuration cell at the top with a `CHECKPOINT_PATH`
placeholder — point it at your trained FARM checkpoint (or train one with
`python -m farm.train ...`). The full reference data is on Zenodo:
<https://doi.org/10.5281/zenodo.20217971>.
