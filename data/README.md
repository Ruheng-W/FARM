# Data

The processed AMR cohorts, the external screen, KO-cluster features,
drug SMILES and train/test splits are hosted on Zenodo:

> **Zenodo record**: <https://doi.org/10.5281/zenodo.XXXXXXX>
>
> Contains:
> - 10 harmonised cohort genotype–phenotype datasets (7,216 isolates × 36 drugs)
> - 144-drug external-screen panel (Maier et al. 2018 *Nature*, filtered to antibiotics)
> - 1,962 KO-cluster reference + drug SMILES + train/test splits
> - License: CC BY 4.0

Total size: approximately **XX GB**. The Zenodo record has its own
`DATA_README.md` that documents every file in detail.

## Quick start example

The `small_example/` directory ships a tiny < 5 MB subset of the harmonised
data so the quick-start notebook can run without downloading the full
dataset. The example is **not representative** of the full benchmark — it
exists only for code verification.

## Private cohorts

The Chile and TIDB cohorts contain isolates from clinical collections at
the contributing institutions and are subject to local IRB restrictions.
De-identified aggregate statistics are included in `Supplementary
Table 2` / `STable3` (Excel sheet). Per-isolate genotype + phenotype data
are available on reasonable request to the corresponding authors.

## Public data sources used in the harmonised cohort

| Cohort | Source | Reference |
|---|---|---|
| Antibiogram | NCBI Antibiogram–SRA (re-curated) | Kim et al. 2020 (VAMPr) |
| ARIsolateBank | FDA–CDC AR Isolate Bank | Lutgring et al. 2018 |
| AstraZeneca | AstraZeneca *P. aeruginosa* resistome panel | Kos et al. 2015 |
| CF | Cystic-fibrosis *P. aeruginosa* cohort | Monogue et al. 2023 |
| German | German ML diagnostics cohort | Khaledi et al. 2020 |
| PATRIC | PATRIC / BV-BRC subset | Antonopoulos et al. 2019 |
| Rabin | Rabin Medical Center Gram-negative set | Koch et al. 2025 |
| Shelburne | Bloodstream-infection cohort | Shelburne et al. 2017 |
| Chile | private |  |
| TIDB | private |  |
