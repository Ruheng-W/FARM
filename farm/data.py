"""Data loading utilities for FARM.

Provides helpers to:
  - parse the harmonised cohort phenotype.txt files
  - parse the per-cohort VAMP.txt gene-presence + mutation features
  - build (drug_SMILES_graph, gene_presence, gene_mutation, label) tuples
    in the format the Drug Graphformer + cross-attention model expects.

The full reference dataset (10 cohorts, 144-drug external screen, splits)
is on Zenodo: https://doi.org/10.5281/zenodo.20217971.
"""
import os
import numpy as np
import pandas as pd

from .smile_rel_dist_interpreter import smiles_to_graph_inputs


COHORTS = [
    'antibiogram', 'ARIsolateBank', 'AstraZeneca', 'CF', 'Chile',
    'German', 'PATRIC', 'Rabin', 'Shelburne', 'TIDB',
]


def load_phenotype(path):
    """Load a single cohort's phenotype.txt.

    Parameters
    ----------
    path : str
        Path to ``{cohort}_phenotype.txt`` (tab-separated, header row).

    Returns
    -------
    pandas.DataFrame
        Columns: ``sample_id``, ``bacteria``, ``antibiotics``, ``phenotype``.
        ``phenotype`` is the raw "Resistant" / "Susceptible" string;
        intermediate calls and any extra columns are dropped.
    """
    rows = []
    with open(path) as f:
        for line in f:
            parts = line.rstrip('\n').split('\t')
            if len(parts) >= 4 and parts[0] != 'sample_id':
                rows.append(parts[:4])
    df = pd.DataFrame(rows, columns=['sample_id', 'bacteria',
                                     'antibiotics', 'phenotype'])
    df = df[df['phenotype'].str.lower().isin(['resistant', 'susceptible'])]
    df['label'] = (df['phenotype'].str.lower() == 'resistant').astype(int)
    return df.reset_index(drop=True)


def load_vamp_features(path):
    """Load VAMPr gene-presence + mutation features.

    Returns a dict ``{sample_id: (presence_vec, mutation_vec)}`` where each
    vector is a NumPy array indexed by KO-cluster ID in the order they
    appear in the file.
    """
    df = pd.read_csv(path, sep='\t')
    needed = {'sample_id', 'KO_id', 'presence', 'mutation'}
    missing = needed - set(df.columns)
    if missing:
        raise ValueError(f'{path}: missing columns {sorted(missing)}')
    out = {}
    for sid, g in df.groupby('sample_id'):
        out[sid] = (g['presence'].to_numpy(dtype=np.uint8),
                    g['mutation'].to_numpy(dtype=np.uint8))
    return out


def load_drug_smiles(path):
    """Load the drug → SMILES mapping (``reference/drug_smiles_36training.tsv``).
    Returns ``{drug_name_lowercase: SMILES}``."""
    df = pd.read_csv(path, sep='\t')
    return {row['drug'].lower(): row['SMILES'] for _, row in df.iterrows()}


def build_examples(phenotype_df, vamp_features, drug_to_smiles):
    """Materialise (graph_inputs, gene_presence, gene_mutation, label) tuples.

    Each row of ``phenotype_df`` becomes one example. Rows whose drug is
    not in ``drug_to_smiles`` or whose sample is not in ``vamp_features``
    are silently dropped.

    Returns
    -------
    list of tuples (drug_graph_inputs, gene_presence_vec, gene_mutation_vec, label)
        ``drug_graph_inputs`` is whatever ``smiles_to_graph_inputs`` returns
        (atom one-hot, bond-type one-hot, shortest-path matrix, …).
    """
    out = []
    for _, row in phenotype_df.iterrows():
        drug = row['antibiotics'].lower()
        sid = row['sample_id']
        if drug not in drug_to_smiles:
            continue
        if sid not in vamp_features:
            continue
        smiles = drug_to_smiles[drug]
        try:
            drug_inputs = smiles_to_graph_inputs(smiles)
        except Exception:
            continue
        ge, im = vamp_features[sid]
        out.append((drug_inputs, ge, im, row['label']))
    return out


def iter_cohorts(data_root):
    """Yield ``(cohort, phenotype_path, vamp_path)`` tuples.

    ``data_root`` is the directory containing the Zenodo ``ten_cohorts/``
    subfolder, e.g. ``/path/to/FARM_dataset_v1/ten_cohorts``.
    """
    for c in COHORTS:
        pheno = os.path.join(data_root, f'{c}_phenotype.txt')
        vamp = os.path.join(data_root, f'{c}_VAMP.txt')
        if os.path.exists(pheno) and os.path.exists(vamp):
            yield c, pheno, vamp
