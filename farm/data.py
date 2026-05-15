"""Data loading utilities for FARM.

Direct readers for the file formats shipped in the Zenodo dataset
(`FARM_dataset_v1.zip` — DOI 10.5281/zenodo.20217971):

  - ``ten_cohorts/{cohort}_phenotype.txt``  tab-separated, 4 columns with header
        sample_id  bacteria  antibiotics  phenotype

  - ``ten_cohorts/{cohort}_VAMP.txt``       tab-separated, 2 columns, NO header
        sample_id  KO_cluster_id_or_variant
    where the second column is either
        "K00116.16"                              KO-cluster present (no variant)
        "K01207.764|174|I|N"                     KO-cluster present AND mutated
                                                  (cluster | aa-pos | wild | mut)

Helpers below convert this representation into the dense binary tensors the
Drug Graphformer + cross-attention model expects.
"""
import os
import numpy as np
import pandas as pd


def _smiles_to_graph_inputs(smiles):
    """Convert a canonical SMILES string into the dict of tensors the Drug
    Graphformer expects (atom one-hot, edge-type one-hot, shortest-path
    distance matrix).

    Implemented as a lazy wrapper around the lower-level helpers in
    ``smile_rel_dist_interpreter`` so importing :mod:`farm.data` does not
    require RDKit when only the phenotype/VAMP loaders are used.
    """
    from .smile_rel_dist_interpreter import (
        generate_interpret_smile, generate_rel_dist_matrix, get_drug_edge_type,
    )
    atom_one_hot = generate_interpret_smile(smiles)
    shortest_path = generate_rel_dist_matrix(smiles)
    bond_type_one_hot = get_drug_edge_type(smiles)
    return {
        'atom_one_hot': np.asarray(atom_one_hot),
        'bond_type_one_hot': np.asarray(bond_type_one_hot),
        'shortest_path': np.asarray(shortest_path),
    }


COHORTS = [
    'antibiogram', 'ARIsolateBank', 'AstraZeneca', 'CF', 'Chile',
    'German', 'PATRIC', 'Rabin', 'Shelburne', 'TIDB',
]


# ---------- phenotype.txt ----------

def load_phenotype(path):
    """Load a single cohort's ``{cohort}_phenotype.txt``.

    Returns a DataFrame with columns ``sample_id``, ``bacteria``,
    ``antibiotics``, ``phenotype``, ``label`` (1 = resistant, 0 = susceptible).
    Rows whose phenotype is not "Resistant" / "Susceptible" (intermediate
    calls) are dropped.
    """
    df = pd.read_csv(path, sep='\t', dtype=str)
    expected = {'sample_id', 'bacteria', 'antibiotics', 'phenotype'}
    missing = expected - set(df.columns)
    if missing:
        raise ValueError(f'{path}: missing columns {sorted(missing)}')
    df = df[df['phenotype'].str.lower().isin(['resistant', 'susceptible'])].copy()
    df['label'] = (df['phenotype'].str.lower() == 'resistant').astype(int)
    return df.reset_index(drop=True)


# ---------- VAMP.txt ----------

def _split_ko_variant(token):
    """Return (KO_cluster_id, is_mutated) from a VAMP token.

    Examples
    --------
    "K00116.16"               -> ("K00116.16", False)
    "K01207.764|174|I|N"      -> ("K01207.764", True)
    """
    if '|' in token:
        return token.split('|', 1)[0], True
    return token, False


def load_vamp_features(path):
    """Parse a cohort's VAMP.txt into per-isolate present / mutated KO sets.

    Returns
    -------
    dict
        ``{sample_id: {"present": set_of_KOcluster_ids,
                       "mutated": set_of_KOcluster_ids}}``
    """
    out = {}
    with open(path) as f:
        for line in f:
            parts = line.rstrip('\n').split('\t')
            if len(parts) < 2:
                continue
            sid, token = parts[0], parts[1]
            ko, mutated = _split_ko_variant(token)
            entry = out.setdefault(sid, {'present': set(), 'mutated': set()})
            entry['present'].add(ko)
            if mutated:
                entry['mutated'].add(ko)
    return out


# ---------- KO-cluster vocabulary ----------

def build_ko_cluster_index(vamp_dicts):
    """Build a sorted, deterministic KO-cluster vocabulary.

    Parameters
    ----------
    vamp_dicts : iterable of the dicts returned by :func:`load_vamp_features`

    Returns
    -------
    dict ``{KO_cluster_id: int_index}`` sorted by KO_cluster_id.
    """
    all_kos = set()
    for d in vamp_dicts:
        for v in d.values():
            all_kos.update(v['present'])
    return {ko: i for i, ko in enumerate(sorted(all_kos))}


def vamp_to_vectors(vamp_dict, ko_index):
    """Project a per-isolate VAMP dict into dense uint8 vectors.

    Parameters
    ----------
    vamp_dict : dict — value returned by :func:`load_vamp_features` for one
        isolate; must have keys ``present`` and ``mutated``.
    ko_index : dict — mapping KO_cluster_id → vocabulary index (see
        :func:`build_ko_cluster_index`).

    Returns
    -------
    (presence, mutation) : two uint8 arrays of length ``len(ko_index)``.
    """
    n = len(ko_index)
    presence = np.zeros(n, dtype=np.uint8)
    mutation = np.zeros(n, dtype=np.uint8)
    for ko in vamp_dict['present']:
        i = ko_index.get(ko)
        if i is not None:
            presence[i] = 1
    for ko in vamp_dict['mutated']:
        i = ko_index.get(ko)
        if i is not None:
            mutation[i] = 1
    return presence, mutation


# ---------- drug SMILES ----------

def load_drug_smiles(path):
    """Load the drug → SMILES mapping (``reference/drug_smiles_36training.tsv``).

    Returns ``{drug_name_lowercase: SMILES}``.
    """
    df = pd.read_csv(path, sep='\t')
    smiles_col = 'SMILES' if 'SMILES' in df.columns else 'drug_smiles'
    return {row['drug'].lower(): row[smiles_col] for _, row in df.iterrows()}


# ---------- example builder ----------

def build_examples(phenotype_df, vamp_dict, drug_to_smiles, ko_index):
    """Materialise ``(drug_graph_inputs, presence, mutation, label)`` tuples.

    Rows whose drug is unknown or whose isolate has no VAMP entry are
    silently dropped.

    Parameters
    ----------
    phenotype_df : DataFrame returned by :func:`load_phenotype`.
    vamp_dict    : dict returned by :func:`load_vamp_features`
                   (covering the same cohort as ``phenotype_df``).
    drug_to_smiles : dict — see :func:`load_drug_smiles`.
    ko_index     : dict — see :func:`build_ko_cluster_index`.

    Returns
    -------
    list of (drug_graph_inputs, presence_vec, mutation_vec, label)
    """
    out = []
    for _, row in phenotype_df.iterrows():
        drug = row['antibiotics'].lower()
        sid = row['sample_id']
        if drug not in drug_to_smiles or sid not in vamp_dict:
            continue
        try:
            drug_inputs = _smiles_to_graph_inputs(drug_to_smiles[drug])
        except Exception:
            continue
        pres, mut = vamp_to_vectors(vamp_dict[sid], ko_index)
        out.append((drug_inputs, pres, mut, int(row['label'])))
    return out


# ---------- cohort iterator ----------

def iter_cohorts(data_root):
    """Yield ``(cohort_name, phenotype_path, vamp_path)`` for each cohort
    present under ``{data_root}/ten_cohorts/``.

    ``data_root`` is the unpacked Zenodo record, e.g.
    ``/path/to/FARM_dataset_v1``.
    """
    base = os.path.join(data_root, 'ten_cohorts')
    for c in COHORTS:
        pheno = os.path.join(base, f'{c}_phenotype.txt')
        vamp  = os.path.join(base, f'{c}_VAMP.txt')
        if os.path.exists(pheno) and os.path.exists(vamp):
            yield c, pheno, vamp


def load_all_cohorts(data_root):
    """Convenience: load every cohort's phenotype + VAMP at once.

    Returns
    -------
    (combined_phenotype_df, combined_vamp_dict, ko_index)
        ``combined_phenotype_df`` has an additional ``cohort`` column.
        ``ko_index`` is the global vocabulary across all 10 cohorts.
    """
    frames = []
    vamps = {}
    for cohort, pheno_path, vamp_path in iter_cohorts(data_root):
        df = load_phenotype(pheno_path)
        df['cohort'] = cohort
        frames.append(df)
        vamps[cohort] = load_vamp_features(vamp_path)

    combined_pheno = pd.concat(frames, ignore_index=True) if frames else \
                     pd.DataFrame(columns=['sample_id','bacteria','antibiotics',
                                            'phenotype','label','cohort'])
    ko_index = build_ko_cluster_index(vamps.values())

    # Flatten per-cohort vamp dicts into a single keyed-by-(cohort, sample_id) dict
    combined_vamp = {(cohort, sid): v
                     for cohort, vd in vamps.items()
                     for sid, v in vd.items()}
    return combined_pheno, combined_vamp, ko_index
