"""Sequence-Transformer baseline (Supplementary Table 8).

Architecture
------------
    SMILES branch (sequence transformer):
        Dense(64) projection → positional encoding
        → 2× [MultiHeadAttention(heads=4, key_dim=64) + Dense(64)-ReLU + LN]
        → GlobalMaxPool1D
    Gene branch:
        Flatten → Dense(64) → ReLU
    Concat → Dense(128) → ReLU → Dense(1) → sigmoid

Inputs
------
    atom_one_hot : (max_atoms=114, atom_vocab=16) one-hot sequence
    gene_features : (3287,) flattened gene-presence + mutation

Optimiser: Adam, learning rate 1e-5, BCE loss.
"""
import numpy as np
import tensorflow as tf
from tensorflow.keras import layers, Model


def _positional_encoding(seq_len: int, d_model: int) -> tf.Tensor:
    pos = np.arange(seq_len)[:, None]
    i = np.arange(d_model)[None, :]
    angle = pos / np.power(10000.0, (2 * (i // 2)) / np.float32(d_model))
    pe = np.where(i % 2 == 0, np.sin(angle), np.cos(angle))
    return tf.constant(pe[None, ...], dtype=tf.float32)


def _transformer_block(x, num_heads: int, key_dim: int, ff_dim: int):
    attn = layers.MultiHeadAttention(num_heads=num_heads, key_dim=key_dim)(x, x)
    x = layers.LayerNormalization()(x + attn)
    ff = layers.Dense(ff_dim, activation='relu')(x)
    return layers.LayerNormalization()(x + ff)


def build_model(max_atoms: int = 114, atom_vocab: int = 16,
                gene_dim: int = 3287, d_model: int = 64,
                num_heads: int = 4, n_blocks: int = 2,
                lr: float = 1e-5) -> Model:
    smiles_in = layers.Input(shape=(max_atoms, atom_vocab),
                             name='atom_one_hot')
    gene_in   = layers.Input(shape=(gene_dim,), name='gene_features')

    s = layers.Dense(d_model)(smiles_in)
    s = s + _positional_encoding(max_atoms, d_model)
    for _ in range(n_blocks):
        s = _transformer_block(s, num_heads=num_heads,
                               key_dim=d_model, ff_dim=d_model)
    s = layers.GlobalMaxPool1D()(s)

    g = layers.Dense(d_model, activation='relu')(gene_in)

    x = layers.Concatenate()([s, g])
    x = layers.Dense(128, activation='relu')(x)
    out = layers.Dense(1, activation='sigmoid',
                       name='probability_resistant')(x)

    model = Model([smiles_in, gene_in], out, name='Transformer_baseline')
    model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=lr),
        loss='binary_crossentropy',
        metrics=['accuracy', tf.keras.metrics.AUC(name='auroc')],
    )
    return model
