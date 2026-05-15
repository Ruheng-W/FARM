# Data

The processed AMR cohorts, the external screen, KO-cluster features,
drug SMILES and train/test splits are hosted on Zenodo:

> **Zenodo record**: <https://doi.org/10.5281/zenodo.20217971>
>
> Contains:
> - 10 harmonised cohort genotype–phenotype datasets (7,216 isolates × 36 drugs)
> - 144-drug external-screen panel (Maier et al. 2018 *Nature*, filtered to antibiotics)
> - 1,962 KO-cluster reference + drug SMILES + train/test splits
> - License: CC BY 4.0

The Zenodo record has its own `DATA_README.md` documenting every file
in detail. Download the `FARM_dataset_v1.zip` archive, extract it, and
point the notebooks at the resulting `FARM_dataset_v1/` directory.

## Public data sources used in the harmonised cohort

| Cohort | Source | Reference |
|---|---|---|
| Antibiogram | NCBI Antibiogram–SRA (re-curated) | Kim et al. 2020 (VAMPr) |
| ARIsolateBank | FDA–CDC AR Isolate Bank | Lutgring et al. 2018 |
| AstraZeneca | AstraZeneca *P. aeruginosa* resistome panel | Kos et al. 2015 |
| CF | Cystic-fibrosis *P. aeruginosa* cohort | Monogue et al. 2023 |
| Chile | Clinical *P. aeruginosa* dataset, Chile | released here |
| German | German ML diagnostics cohort | Khaledi et al. 2020 |
| PATRIC | PATRIC / BV-BRC subset | Antonopoulos et al. 2019 |
| Rabin | Rabin Medical Center Gram-negative set | Koch et al. 2025 |
| Shelburne | Bloodstream-infection cohort | Shelburne et al. 2017 |
| TIDB | TIDB institutional dataset | released here |

The 144-drug external-screen subset is derived from Maier *et al.* 2018,
*Nature* 555(7698): 623–628 — DOI: <https://doi.org/10.1038/nature25979>.
