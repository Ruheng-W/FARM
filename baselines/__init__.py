"""Reference baseline models compared against FARM in the paper.

Each module exposes a ``build_model(config)`` function returning a
compiled tf.keras.Model (deep-learning baselines) or a scikit-learn
estimator (classical baselines).

Architectures and training settings are tabulated in Supplementary Table 8.
"""
