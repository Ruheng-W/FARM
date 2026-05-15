# FARM — Drug-conditioned cross-attention for antimicrobial-resistance prediction

[![Dataset](https://img.shields.io/badge/data-Zenodo-orange)](https://doi.org/10.5281/zenodo.XXXXXXX)
[![Web server](https://img.shields.io/badge/web%20server-cdc.biohpc.swmed.edu%2Ffarm-green)](https://cdc.biohpc.swmed.edu/farm/intro)
[![License](https://img.shields.io/badge/license-MIT-lightgrey)](LICENSE)

FARM (**F**unctional **A**ntimicrobial **R**esistance **M**odel) is a deep-learning
framework that predicts whether a given antibiotic will be effective against
a given bacterial isolate. The model represents each antibiotic as a
molecular graph (via a Drug Graphformer self-attention encoder) and each
isolate by gene-presence / mutation features over 1,962 KEGG-ortholog
clusters, and couples them through a multi-head **drug–gene cross-attention**
module. The same forward pass produces both a resistance prediction and two
interpretable readouts: atom-level saliency (which substructures the drug
relies on) and gene-level importance (which bacterial determinants the
model attends to).

We evaluate FARM on 10 harmonized cohorts spanning **7,216 isolates, 36
antibiotics, 11 species, and 63,870 isolate–drug observations**, and on an
independent in-vitro screen of 144 antibiotics against 39 sequenced strains.

> 📄 **No-install demo:** the FARM web server lets you upload a genome /
> drug pair and get a prediction directly in the browser:
> <https://cdc.biohpc.swmed.edu/farm/intro>

---

## Highlights

- **Cross-pathogen, cross-drug**: one model replaces the conventional
  "one model per drug per species" pattern; predictions transfer across
  cohorts, drug classes and bacterial species.
- **Strong performance under matched evaluation**: mean cross-cohort
  AUROC 0.88 / AUPRC 0.87, +0.12 AUROC and +0.15 AUPRC margin over VAMPr
  under the matched antibiogram-excluded protocol.
- **Generalises to unseen drugs and species**: leave-one-drug-out and an
  independent in-vitro screen of 144 antibiotics against 39 strains both
  retain predictive signal.
- **Interpretable by design**: atom-level saliency aligns with PDB-derived
  ligand-binding atoms; cross-attention recovers canonical resistance
  genes (β-lactamases, aminoglycoside-modifying enzymes, ribosomal
  protection proteins, etc.) and KEGG resistance pathways.

---

## Quick start

```bash
# 1. Clone
git clone https://github.com/Ruheng-W/FARM.git
cd FARM

# 2. Install the environment
conda env create -f environment.yml
conda activate farm

# 3. Try the web server (no install needed)
#    https://cdc.biohpc.swmed.edu/farm/intro

# 4. Or run the bundled example notebook
jupyter notebook notebooks/01_quick_start.ipynb
```

`notebooks/01_quick_start.ipynb` walks through:
1. Loading the small example data shipped with the repo
2. Predicting resistance for an antibiotic–isolate pair
3. Visualising atom-level saliency and gene-level attention

---

## Installation

**Hardware.** The repo runs on CPU for inference; training the full FARM
model requires a CUDA-capable GPU (we used NVIDIA A100 80 GB). Inference
on a single drug–isolate pair completes in ≪1 s on CPU.

**Software.** Python ≥ 3.9, TensorFlow 2.15 + Keras 2.15, RDKit, scikit-learn.
The exact pinned environment is in `environment.yml`. A pip fallback is
provided as `requirements.txt`.

```bash
conda env create -f environment.yml
conda activate farm
```

---

## Data

The processed AMR cohorts, the external-screen panel, train/test splits and
all reference files are hosted on **Zenodo** (DOI:
`10.5281/zenodo.XXXXXXX`, ~XX GB):

- 10 harmonised cohort genotype–phenotype datasets (7,216 isolates, 36 drugs)
- 144-drug external-screen panel (Maier et al. 2018, filtered to antibiotics)
- KEGG ortholog reference (1,962 KO-clusters, AMR-related)
- Drug SMILES (36 training drugs + 144 external drugs)
- All train/test splits used for cohort-holdout, leave-one-drug-out, and OOD
  evaluations

See `data/README.md` for download instructions and the Zenodo record's
`DATA_README.md` for the full file manifest.

**For private cohorts (Chile, TIDB):** raw data are subject to local IRB
restrictions and are not redistributed. Aggregated / de-identified summaries
are included where permitted; full access is available on reasonable request
to the corresponding authors.

---

## Reproducing the paper

| Output | Command |
|---|---|
| Figure 2B (cross-cohort accuracy) | `bash scripts/train_cohort_holdout.sh` |
| Figure 2C (per-drug accuracy)     | `bash scripts/train_cohort_holdout.sh && python farm/eval.py --by-drug` |
| Figure 3 (atom-level saliency)    | `notebooks/02_attention_visualization.ipynb` |
| Figure 4 (gene rankings)          | `notebooks/02_attention_visualization.ipynb` |
| Figure 5 (LODO)                   | `bash scripts/leave_one_drug_out.sh` |
| Figure 6 (OOD screen)             | `notebooks/03_external_screen_eval.ipynb` |
| Supplementary tables              | `python supplementary/build_all_supp_tables.py` |

Each script takes 1–18 h on a single A100 depending on the workload. The
pretrained main FARM checkpoint is not redistributed here — use the
**FARM web server** for one-off inference, or retrain with the supplied
splits + data for full reproducibility.

---

## Repository layout

```
FARM/
├── README.md
├── LICENSE                          MIT
├── CITATION.cff
├── environment.yml
├── requirements.txt
│
├── farm/                            FARM model & training code
│   ├── model.py                       Drug Graphformer + cross-attention
│   ├── data.py                        Dataset / dataloader
│   ├── train.py                       Training loop
│   ├── predict.py                     Inference on a drug–isolate pair
│   └── eval.py                        Metrics + bootstrap CIs
│
├── baselines/                       Reference baselines used in the paper
│   ├── mlp.py / cnn.py / transformer.py
│   └── classical.py                 Random Forest, Elastic Net
│
├── scripts/                         Shell wrappers for common runs
│   ├── train_cohort_holdout.sh
│   ├── leave_one_drug_out.sh
│   └── reproduce_figure_2.sh
│
├── notebooks/                       Reviewer-facing demos
│   ├── 01_quick_start.ipynb
│   ├── 02_attention_visualization.ipynb
│   └── 03_external_screen_eval.ipynb
│
├── data/
│   ├── README.md                    Points to Zenodo
│   └── small_example/               <50 MB toy subset for the quick start
│
└── supplementary/                   Supplementary Information for the paper
    ├── Supplementary_Information.pdf
    └── Supplementary_Tables.xlsx
```

---

## Train on your own data

The model expects three inputs per drug–isolate pair:
1. Drug **canonical SMILES** (parsed by RDKit into an atom set, bond types,
   and an all-pairs shortest-path distance matrix).
2. Isolate **gene-presence vector** over the 1,962 KO-cluster reference
   (0 / 1 per KO-cluster).
3. Isolate **gene-mutation vector** over the same reference (0 / 1 per
   KO-cluster).

`farm/predict.py` exposes a single-pair API:

```python
from farm import predict
result = predict.predict_one(
    smiles='CC1(C(N2C(S1)C(C2=O)NC(=O)C(C3=CC=CC=C3)N)C(=O)O)C',
    gene_presence='path/to/genome_KOcluster_presence.npy',
    gene_mutation='path/to/genome_KOcluster_mutation.npy',
)
print(result['probability_resistant'])
```

For batch / training pipelines see `farm/train.py` and the demo notebooks.

---

## Citation

If you use FARM in your work please cite:

> Wang R., Wanyan T., Gan S., Kim J., Zarek C. M., Koh A. Y., Liu F., Cong Q.,
> Liu D., Greenberg D., Xiao G., Xie Y., Zhan X. (2026). *Interpretable
> Antimicrobial Resistance Prediction via Cross-Attention Between Antibiotic
> Substructures and Genomic Features.* Manuscript in preparation.

BibTeX (will be updated once the paper is accepted):

```bibtex
@unpublished{farm2026,
  title  = {Interpretable Antimicrobial Resistance Prediction via Cross-Attention Between Antibiotic Substructures and Genomic Features},
  author = {Wang, Ruheng and Wanyan, Tingyi and Gan, Shuheng and Kim, Jiwoong and Zarek, Christina M. and Koh, Andrew Y. and Liu, Fangyu and Cong, Qian and Liu, Dajiang and Greenberg, David and Xiao, Guanghua and Xie, Yang and Zhan, Xiaowei},
  year   = {2026},
  note   = {Manuscript in preparation}
}
```

---

## License

- **Code**: MIT (see `LICENSE`)
- **Data on Zenodo**: CC BY 4.0

---

## Contact

- Web server: <https://cdc.biohpc.swmed.edu/farm/intro>
- Corresponding authors:
  - Yang Xie, [Yang.Xie@UTSouthwestern.edu](mailto:Yang.Xie@UTSouthwestern.edu)
  - Xiaowei Zhan, [Xiaowei.Zhan@UTSouthwestern.edu](mailto:Xiaowei.Zhan@UTSouthwestern.edu)

Please open an issue on this repository for code or reproducibility
questions; for clinical / dataset questions please email the corresponding
authors directly.
