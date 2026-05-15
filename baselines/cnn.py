"""1-D CNN baseline (Supplementary Table 8).

Architecture
------------
    SMILES branch:
        Conv1D(filters=128, kernel=5, padding='same')
        → GlobalMaxPool1D
    Gene branch:
        Flatten → Dense(256) → ReLU
    Concat both branches → Dense(256) → ReLU → Dense(1) → sigmoid

Inputs
------
    smiles_one_hot : (max_atoms=114, atom_vocab=16) one-hot sequence
    gene_features  : (3287,) one-hot flattened gene-presence + mutation

Optimiser: Adam, learning rate 1e-5, BCE loss.
"""
import tensorflow as tf
from tensorflow.keras import layers, Model


def build_model(max_atoms: int = 114, atom_vocab: int = 16,
                gene_dim: int = 3287, lr: float = 1e-5) -> Model:
    smiles_in = layers.Input(shape=(max_atoms, atom_vocab), name='atom_one_hot')
    gene_in   = layers.Input(shape=(gene_dim,),             name='gene_features')

    s = layers.Conv1D(filters=128, kernel_size=5, padding='same',
                      activation='relu')(smiles_in)
    s = layers.GlobalMaxPool1D()(s)

    g = layers.Dense(256, activation='relu')(gene_in)

    x = layers.Concatenate()([s, g])
    x = layers.Dense(256, activation='relu')(x)
    out = layers.Dense(1, activation='sigmoid',
                       name='probability_resistant')(x)

    model = Model([smiles_in, gene_in], out, name='CNN_baseline')
    model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=lr),
        loss='binary_crossentropy',
        metrics=['accuracy', tf.keras.metrics.AUC(name='auroc')],
    )
    return model
