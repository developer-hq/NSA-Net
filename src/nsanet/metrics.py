import math

import numpy as np
from sklearn.metrics import (accuracy_score, average_precision_score,
                             balanced_accuracy_score, brier_score_loss,
                             confusion_matrix, f1_score, fbeta_score,
                             matthews_corrcoef, precision_score, recall_score,
                             roc_auc_score)


def expected_calibration_error(labels, scores, n_bins=10):
    labels = np.asarray(labels, dtype=int)
    scores = np.asarray(scores, dtype=float)
    edges = np.linspace(0.0, 1.0, n_bins + 1)
    ece = 0.0
    for index in range(n_bins):
        if index == n_bins - 1:
            mask = (scores >= edges[index]) & (scores <= edges[index + 1])
        else:
            mask = (scores >= edges[index]) & (scores < edges[index + 1])
        if mask.any():
            ece += mask.mean() * abs(scores[mask].mean() - labels[mask].mean())
    return float(ece)


def ranking_at_fraction(labels, scores, fraction):
    labels = np.asarray(labels, dtype=int)
    scores = np.asarray(scores, dtype=float)
    k = max(1, int(math.ceil(len(labels) * fraction)))
    order = np.argsort(-scores, kind="stable")[:k]
    hits = int(labels[order].sum())
    positives = int(labels.sum())
    prevalence = positives / len(labels)
    precision = hits / k
    recall = hits / positives if positives else 0.0
    enrichment = precision / prevalence if prevalence else 0.0
    suffix = f"{int(round(fraction * 100))}pct"
    return {
        f"budget_k_{suffix}": k,
        f"precision_at_{suffix}": float(precision),
        f"recall_at_{suffix}": float(recall),
        f"enrichment_factor_at_{suffix}": float(enrichment),
    }


def compute_metrics(labels, scores, threshold=0.5):
    labels = np.asarray(labels, dtype=int)
    scores = np.asarray(scores, dtype=float)
    predictions = (scores >= threshold).astype(int)
    tn, fp, fn, tp = confusion_matrix(labels, predictions, labels=[0, 1]).ravel()
    prevalence = float(labels.mean())
    specificity = tn / (tn + fp) if tn + fp else 0.0
    npv = tn / (tn + fn) if tn + fn else 0.0

    result = {
        "n_samples": int(len(labels)),
        "n_positive": int(labels.sum()),
        "n_negative": int((labels == 0).sum()),
        "positive_prevalence": prevalence,
        "majority_accuracy_baseline": float(max(prevalence, 1.0 - prevalence)),
        "no_skill_ap_baseline": prevalence,
        "threshold": float(threshold),
        "tn": int(tn), "fp": int(fp), "fn": int(fn), "tp": int(tp),
        "roc_auc": float(roc_auc_score(labels, scores)),
        "partial_auc_fpr_0_10": float(roc_auc_score(labels, scores, max_fpr=0.10)),
        "average_precision": float(average_precision_score(labels, scores)),
        "accuracy": float(accuracy_score(labels, predictions)),
        "balanced_accuracy": float(balanced_accuracy_score(labels, predictions)),
        "precision": float(precision_score(labels, predictions, zero_division=0)),
        "recall": float(recall_score(labels, predictions, zero_division=0)),
        "specificity": float(specificity),
        "npv": float(npv),
        "f1": float(f1_score(labels, predictions, zero_division=0)),
        "f2": float(fbeta_score(labels, predictions, beta=2, zero_division=0)),
        "mcc": float(matthews_corrcoef(labels, predictions)),
        "brier": float(brier_score_loss(labels, scores)),
        "ece_10bin": expected_calibration_error(labels, scores, n_bins=10),
    }
    result.update(ranking_at_fraction(labels, scores, 0.10))
    result.update(ranking_at_fraction(labels, scores, 0.20))
    return result
