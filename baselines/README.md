# `baselines/` — reference models

The paper compares FARM against several deep-learning and classical-ML
baselines, trained on the same harmonised AMR cohort data.

| Module | Model | Build function |
|---|---|---|
| `mlp.py` | Multi-layer perceptron, 1024→512→256→1 with Dropout 0.5 | `mlp.build_model(input_dim, dropout, lr)` |
| `cnn.py` | 1-D CNN on the atom one-hot SMILES sequence + dense gene branch | `cnn.build_model(max_atoms, atom_vocab, gene_dim, lr)` |
| `transformer.py` | 2-block sequence Transformer on the SMILES + dense gene branch | `transformer.build_model(max_atoms, atom_vocab, gene_dim, d_model, num_heads, n_blocks, lr)` |
| `classical.py` | Random Forest + Elastic-Net logistic regression (scikit-learn) | `classical.build_random_forest(...)`, `classical.build_elastic_net(...)` |

All deep-learning baselines compile with Adam + BCE loss; learning rates
follow the values reported in the paper.

The three rule/database baselines (**VAMPr**, **ResFinder**, **RGI/CARD**)
are run from their upstream repositories:

- VAMPr: <https://github.com/jaehyunkim-rutgers/VAMPr>
- ResFinder: <https://bitbucket.org/genomicepidemiology/resfinder/>
- RGI (CARD): <https://github.com/arpcard/rgi>
