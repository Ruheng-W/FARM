# Notebooks

Reviewer-facing demos to make the work accessible without retraining.

| Notebook | Purpose | Runtime |
|---|---|---|
| `01_quick_start.ipynb` | Load the tiny example dataset → predict resistance for one (drug, isolate) pair → visualise atom and gene attention. | < 1 min on CPU |
| `02_attention_visualization.ipynb` | Reproduce Figure 3 (atom-level saliency) and Figure 4 (gene-level cross-attention) on the trained model. | 2–5 min on GPU |
| `03_external_screen_eval.ipynb` | Reproduce Figure 6 (out-of-distribution generalisation to the 144-drug Maier et al. screen). | ~10 min on GPU |

> ⚠️ **Note**: notebooks `02` and `03` require the pretrained FARM checkpoint,
> which is not redistributed in this repository. For one-off predictions
> use the FARM web server: <https://cdc.biohpc.swmed.edu/farm/intro>.
> For full retraining + evaluation use the data on Zenodo and the
> training scripts in `farm/`.
