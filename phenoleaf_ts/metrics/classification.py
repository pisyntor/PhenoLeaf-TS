"""Growth-stage classification metrics for PhenoLeaf-TS."""

import numpy as np


def compute_classification_metrics(y_true, y_pred, num_classes=3):
    """Accuracy, macro-F1, precision/recall, specificity, G-mean, MCC, confusion matrix.

    Classes: 0 = Early, 1 = Intermediate, 2 = Mature.
    """
    from sklearn.metrics import (
        accuracy_score,
        f1_score,
        precision_score,
        recall_score,
        confusion_matrix,
        matthews_corrcoef,
    )

    cm = confusion_matrix(y_true, y_pred, labels=list(range(num_classes)))

    specificities = []
    for i in range(num_classes):
        tn = cm.sum() - cm[i, :].sum() - cm[:, i].sum() + cm[i, i]
        fp = cm[:, i].sum() - cm[i, i]
        specificities.append(tn / (tn + fp + 1e-8))

    per_class_recall = recall_score(
        y_true, y_pred, average=None, labels=list(range(num_classes))
    )
    gmean = float(np.prod(per_class_recall) ** (1.0 / num_classes))

    return {
        "accuracy": float(accuracy_score(y_true, y_pred)),
        "f1_macro": float(f1_score(y_true, y_pred, average="macro")),
        "f1_weighted": float(f1_score(y_true, y_pred, average="weighted")),
        "precision": float(precision_score(y_true, y_pred, average="macro")),
        "recall": float(recall_score(y_true, y_pred, average="macro")),
        "specificity": float(np.mean(specificities)),
        "geometric_mean": gmean,
        "mcc": float(matthews_corrcoef(y_true, y_pred)),
        "confusion_matrix": cm.tolist(),
        "per_class_recall": per_class_recall.tolist(),
        "per_class_specificity": [float(s) for s in specificities],
    }
