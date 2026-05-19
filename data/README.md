# Data

The processed AMR cohorts, the external screen, KO-cluster features,
drug SMILES and train/test splits are hosted on Zenodo:

> **Zenodo record**: <https://doi.org/10.5281/zenodo.20217971>
>
> Contains:
> - 6 publicly redistributable cohort genotype–phenotype datasets
>   (6,446 isolates × 34 drugs, 57,338 observations)
> - 144-drug external-screen panel (Maier et al. 2018 *Nature*, filtered to antibiotics)
> - 1,962 KO-cluster reference + drug SMILES + train/test splits
> - License: GNU General Public License v3.0 (GPL-3.0)

The Zenodo record has its own `DATA_README.md` documenting every file
in detail. Download the `FARM_dataset_v1.zip` archive, extract it, and
point the notebooks at the resulting `FARM_dataset_v1/` directory.

## Public data sources in this release (6 cohorts)

| Cohort | Source | Reference |
|---|---|---|
| Antibiogram | NCBI Antibiogram–SRA (re-curated) | Kim et al. 2020 (VAMPr) |
| ARIsolateBank | FDA–CDC AR Isolate Bank | Lutgring et al. 2018 |
| AstraZeneca | AstraZeneca *P. aeruginosa* resistome panel | Kos et al. 2015 |
| German | German ML diagnostics cohort | Khaledi et al. 2020 |
| PATRIC | PATRIC / BV-BRC subset | Antonopoulos et al. 2019 |
| Shelburne | Bloodstream-infection cohort | Shelburne et al. 2017 |

## Restricted-access clinical cohorts (4 cohorts)

The manuscript also reports results on four clinical cohorts whose
underlying genotype–phenotype data cannot be redistributed in this
release due to data-use restrictions imposed by the contributing
institutions. Per-cohort dimensions are documented in the Zenodo
record's `COHORT_MANIFEST.md`. The underlying data are available from
the corresponding authors upon reasonable request, subject to the
originating institutions' data-sharing policies.

| Cohort | Origin | Reference |
|---|---|---|
| CF | Cystic-fibrosis *P. aeruginosa* cohort | Monogue et al. 2023 |
| Chile | Clinical *P. aeruginosa* dataset | (this study; restricted) |
| Rabin | Rabin Medical Center Gram-negative cohort | Koch et al. 2025 |
| TIDB | TIDB institutional dataset | (this study; restricted) |

Three drugs (**ofloxacin**, **ceftazidime-avibactam**, **piperacillin**)
appear only in the restricted cohorts and are therefore absent from the
public bundle.

## External screen

The 144-drug external-screen subset is derived from Maier *et al.* 2018,
*Nature* 555(7698): 623–628 — DOI: <https://doi.org/10.1038/nature25979>.
