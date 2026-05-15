"""Single-pair inference for FARM.

Use ``predict_one`` to predict the resistance probability for one
(drug SMILES, isolate gene-features) pair, given a trained FARM checkpoint.

Example
-------
>>> from farm import predict
>>> result = predict.predict_one(
...     smiles='CC1(C(N2C(S1)C(C2=O)NC(=O)C(C3=CC=CC=C3)N)C(=O)O)C',
...     gene_presence='path/to/genome_KOcluster_presence.npy',
...     gene_mutation='path/to/genome_KOcluster_mutation.npy',
...     checkpoint='path/to/farm_checkpoint.h5',
... )
>>> print(result['probability_resistant'])

For batch / cohort-scale inference, build a tf.data.Dataset using
``farm.data.build_examples`` and run model.predict directly.
"""
import os
import numpy as np

from .data import _smiles_to_graph_inputs as smiles_to_graph_inputs


def load_checkpoint(checkpoint_path):
    """Load a trained FARM Keras checkpoint.

    The checkpoint is expected to be a .h5 file produced by
    ``model.save_weights`` or the SavedModel directory produced by
    ``tf.keras.models.save_model``.
    """
    import tensorflow as tf
    if not os.path.exists(checkpoint_path):
        raise FileNotFoundError(
            f'Checkpoint not found: {checkpoint_path}.\n'
            f'Train your own checkpoint with `python -m farm.train ...` '
            f'or use the FARM web server at '
            f'https://cdc.biohpc.swmed.edu/farm/intro.'
        )
    return tf.keras.models.load_model(checkpoint_path, compile=False)


def _coerce_gene_vector(arg, expected_len=None):
    """Accept a path-to-.npy, a list, or a NumPy array. Return uint8 array."""
    if isinstance(arg, str):
        arr = np.load(arg)
    else:
        arr = np.asarray(arg)
    arr = arr.astype(np.uint8)
    if expected_len is not None and arr.shape[-1] != expected_len:
        raise ValueError(
            f'Gene vector length {arr.shape[-1]} does not match expected '
            f'{expected_len}. The FARM reference is 1,962 KO-clusters.'
        )
    return arr


def predict_one(smiles, gene_presence, gene_mutation, checkpoint,
                return_attention=False):
    """Predict the probability of resistance for one (drug, isolate) pair.

    Parameters
    ----------
    smiles : str
        Canonical SMILES for the antibiotic.
    gene_presence : str | array-like
        Path to a saved 1-D NumPy array (length 1,962) of gene-presence
        indicators, or the array itself.
    gene_mutation : str | array-like
        Same as above for gene-mutation indicators.
    checkpoint : str
        Path to a trained FARM checkpoint (.h5 or SavedModel dir).
    return_attention : bool, default False
        If True, also return atom-level saliency and gene-level
        cross-attention vectors so the caller can plot them.

    Returns
    -------
    dict with at least:
        ``probability_resistant`` : float in [0, 1]
        ``predicted_label`` : "Resistant" or "Susceptible"
    plus, if ``return_attention=True``:
        ``atom_saliency``   : 1-D array, one score per atom in the SMILES
        ``gene_attention``  : 1-D array, length 1,962
    """
    model = load_checkpoint(checkpoint)

    drug_inputs = smiles_to_graph_inputs(smiles)
    presence = _coerce_gene_vector(gene_presence, expected_len=1962)
    mutation = _coerce_gene_vector(gene_mutation, expected_len=1962)

    # Pack inputs to a batch of size 1.  The actual model signature depends
    # on the architecture in `farm/model.py`; here we describe the contract
    # but users may need to adapt the dict keys to match their checkpoint.
    batch_inputs = {
        'atom_one_hot':       drug_inputs['atom_one_hot'][None, ...],
        'bond_type_one_hot':  drug_inputs['bond_type_one_hot'][None, ...],
        'shortest_path':      drug_inputs['shortest_path'][None, ...],
        'gene_presence':      presence[None, ...],
        'gene_mutation':      mutation[None, ...],
    }
    pred = model(batch_inputs, training=False)
    prob = float(pred['probability_resistant'].numpy()
                 if hasattr(pred, 'keys')
                 else np.asarray(pred).ravel()[0])

    result = {
        'probability_resistant': prob,
        'predicted_label': 'Resistant' if prob >= 0.5 else 'Susceptible',
    }

    if return_attention:
        # Extract from auxiliary heads if they were saved with the model.
        if hasattr(pred, 'keys'):
            if 'atom_saliency' in pred:
                result['atom_saliency'] = np.asarray(pred['atom_saliency']).ravel()
            if 'gene_attention' in pred:
                result['gene_attention'] = np.asarray(pred['gene_attention']).ravel()
        else:
            # If your checkpoint does not expose attention as a model output,
            # you can extract it by hooking the model's internal attention
            # layers; see `notebooks/02_attention_visualization.ipynb` for
            # an example.
            pass

    return result


def predict_batch(pairs, checkpoint, batch_size=64):
    """Vectorised inference over many (smiles, presence, mutation) tuples.

    Parameters
    ----------
    pairs : iterable of (smiles, gene_presence, gene_mutation)
    checkpoint : str
    batch_size : int

    Yields
    ------
    dict per pair, same keys as ``predict_one``.
    """
    import tensorflow as tf
    model = load_checkpoint(checkpoint)

    buf_drug = []
    buf_pres = []
    buf_mut = []
    indices = []

    def flush(out):
        if not indices:
            return
        b = {
            'atom_one_hot':      np.stack([d['atom_one_hot']      for d in buf_drug]),
            'bond_type_one_hot': np.stack([d['bond_type_one_hot'] for d in buf_drug]),
            'shortest_path':     np.stack([d['shortest_path']     for d in buf_drug]),
            'gene_presence':     np.stack(buf_pres),
            'gene_mutation':     np.stack(buf_mut),
        }
        preds = model(b, training=False)
        probs = np.asarray(preds).ravel() if not hasattr(preds, 'keys') \
                else np.asarray(preds['probability_resistant']).ravel()
        for prob in probs:
            prob = float(prob)
            out.append({
                'probability_resistant': prob,
                'predicted_label': 'Resistant' if prob >= 0.5 else 'Susceptible',
            })
        buf_drug.clear(); buf_pres.clear(); buf_mut.clear(); indices.clear()

    out = []
    for i, (smiles, pres, mut) in enumerate(pairs):
        buf_drug.append(smiles_to_graph_inputs(smiles))
        buf_pres.append(_coerce_gene_vector(pres, 1962))
        buf_mut.append(_coerce_gene_vector(mut, 1962))
        indices.append(i)
        if len(indices) >= batch_size:
            flush(out)
    flush(out)
    return out
