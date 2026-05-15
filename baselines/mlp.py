"""MLP baseline (Supplementary Table 8).

Architecture
------------
    Dense(1024) → ReLU → Dropout(0.5)
    → Dense(512)  → ReLU → Dropout(0.5)
    → Dense(256)  → ReLU → Dropout(0.5)
    → Dense(1)    → sigmoid

Inputs
------
    Concatenation of:
      - atom one-hot vector, flattened (max 114 atoms × 16-symbol vocab)
      - gene-presence one-hots flattened
      - gene-mutation one-hots flattened
    Total input dim ≈ 3,287 + 114 × 16 = 5,111 features per (drug, isolate) pair.

Optimiser: Adam, learning rate 1e-5, BCE loss.
"""
import tensorflow as tf
from tensorflow.keras import layers, Model


def build_model(input_dim: int, dropout: float = 0.5, lr: float = 1e-5) -> Model:
    inp = layers.Input(shape=(input_dim,), name='flat_features')
    x = layers.Dense(1024, activation='relu')(inp)
    x = layers.Dropout(dropout)(x)
    x = layers.Dense(512, activation='relu')(x)
    x = layers.Dropout(dropout)(x)
    x = layers.Dense(256, activation='relu')(x)
    x = layers.Dropout(dropout)(x)
    out = layers.Dense(1, activation='sigmoid', name='probability_resistant')(x)
    model = Model(inp, out, name='MLP_baseline')
    model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=lr),
        loss='binary_crossentropy',
        metrics=['accuracy', tf.keras.metrics.AUC(name='auroc')],
    )
    return model
