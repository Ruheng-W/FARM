"""Classical-ML baselines (Supplementary Table 8).

- ``build_random_forest``: scikit-learn RandomForestClassifier
                            (n_estimators=200, default depth).
- ``build_elastic_net``  : scikit-learn LogisticRegression with
                            penalty='elasticnet', C=1.0, l1_ratio=0.5,
                            solver='saga'.

Both consume flat feature vectors:
    [atom_one_hot.flatten() ⊕ gene_presence ⊕ gene_mutation]
i.e. the same input layout as the MLP baseline.
"""
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression


def build_random_forest(n_estimators: int = 200, n_jobs: int = -1,
                        random_state: int = 0) -> RandomForestClassifier:
    return RandomForestClassifier(
        n_estimators=n_estimators,
        n_jobs=n_jobs,
        random_state=random_state,
    )


def build_elastic_net(C: float = 1.0, l1_ratio: float = 0.5,
                      max_iter: int = 1000, n_jobs: int = -1,
                      random_state: int = 0) -> LogisticRegression:
    return LogisticRegression(
        penalty='elasticnet',
        solver='saga',
        C=C,
        l1_ratio=l1_ratio,
        max_iter=max_iter,
        n_jobs=n_jobs,
        random_state=random_state,
    )
