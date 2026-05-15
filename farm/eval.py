"""Evaluation utilities for FARM and baselines.

Computes the metrics reported in the paper (accuracy, balanced accuracy,
AUROC, AUPRC, F1, MCC, sensitivity, specificity) with isolate-level
bootstrap 95 % confidence intervals.

Usage
-----
    from farm.eval import score, bootstrap_ci

    summary = score(y_true, y_prob, threshold=0.5)
    ci_low, ci_high = bootstrap_ci(y_true, y_prob, metric='AUROC',
                                   n_resamples=1000, seed=0)
"""
import numpy as np
from sklearn.metrics import (
    accuracy_score, balanced_accuracy_score, roc_auc_score,
    average_precision_score, f1_score, matthews_corrcoef,
    recall_score, confusion_matrix,
)


def _binary_predictions(y_prob, threshold):
    return (np.asarray(y_prob) >= threshold).astype(int)


def score(y_true, y_prob, threshold: float = 0.5) -> dict:
    """Compute all reported metrics for one prediction set.

    Metrics whose value is mathematically undefined for the supplied data
    (e.g. AUROC on a single-class label set; sensitivity when there are
    no positives) are returned as ``float('nan')``.
    """
    y_true = np.asarray(y_true).astype(int)
    y_prob = np.asarray(y_prob, dtype=float)
    y_pred = _binary_predictions(y_prob, threshold)

    out = {'resistance_rate': float(y_true.mean()) if len(y_true) else float('nan')}
    out['accuracy'] = accuracy_score(y_true, y_pred)
    out['balanced_accuracy'] = (balanced_accuracy_score(y_true, y_pred)
                                if len(set(y_true)) >= 2 else float('nan'))
    out['AUROC'] = (roc_auc_score(y_true, y_prob)
                    if len(set(y_true)) >= 2 else float('nan'))
    out['AUPRC'] = (average_precision_score(y_true, y_prob)
                    if len(set(y_true)) >= 2 else float('nan'))
    out['F1']  = f1_score(y_true, y_pred, zero_division=0)
    out['MCC'] = matthews_corrcoef(y_true, y_pred) if len(set(y_pred)) >= 2 else 0.0

    # Sensitivity / specificity from the confusion matrix
    tn, fp, fn, tp = (confusion_matrix(y_true, y_pred, labels=[0, 1]).ravel()
                      if len(y_true) else (0, 0, 0, 0))
    out['sensitivity'] = (tp / (tp + fn)) if (tp + fn) > 0 else float('nan')
    out['specificity'] = (tn / (tn + fp)) if (tn + fp) > 0 else float('nan')
    return out


def bootstrap_ci(y_true, y_prob, metric: str = 'AUROC',
                 n_resamples: int = 1000, alpha: float = 0.05,
                 seed: int = 0):
    """Isolate-level (i.i.d.) bootstrap 95 % confidence interval.

    Parameters
    ----------
    metric : one of the keys returned by ``score``.
    n_resamples : number of bootstrap replicates (1000 in the paper).
    alpha : two-sided coverage (0.05 → 95 % CI).
    seed : RNG seed.

    Returns
    -------
    (low, high) tuple of floats.
    """
    y_true = np.asarray(y_true).astype(int)
    y_prob = np.asarray(y_prob, dtype=float)
    n = len(y_true)
    if n < 5:
        return (float('nan'), float('nan'))
    rng = np.random.default_rng(seed)
    vals = []
    for _ in range(n_resamples):
        idx = rng.integers(0, n, size=n)
        try:
            s = score(y_true[idx], y_prob[idx])
            v = s[metric]
        except Exception:
            v = float('nan')
        if not np.isnan(v):
            vals.append(v)
    if not vals:
        return (float('nan'), float('nan'))
    vals.sort()
    lo = vals[int(round((alpha / 2)       * len(vals)))]
    hi = vals[int(round((1 - alpha / 2)   * len(vals))) - 1]
    return (lo, hi)


def score_with_ci(y_true, y_prob, metric: str = 'AUROC',
                  n_resamples: int = 1000, seed: int = 0) -> dict:
    """Convenience: point estimate + 95 % CI for a single metric."""
    s = score(y_true, y_prob)
    lo, hi = bootstrap_ci(y_true, y_prob, metric=metric,
                          n_resamples=n_resamples, seed=seed)
    return {metric: s[metric], f'{metric}_CI_lower_95': lo,
            f'{metric}_CI_upper_95': hi}
