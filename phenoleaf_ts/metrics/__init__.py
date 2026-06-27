"""Evaluation metrics for the three PhenoLeaf-TS tasks.

Segmentation and classification metrics are implemented here. Tracking uses standard
MOT metrics (MOTA / IDF1 / HOTA) via py-motmetrics / TrackEval over the colour-consistent
ground truth — see the README.
"""

from phenoleaf_ts.metrics.segmentation import (
    compute_dice,
    compute_iou,
    extract_instances_from_colour_mask,
    compute_symmetric_best_dice,
    compute_mask_map,
    compute_segmentation_metrics,
)
from phenoleaf_ts.metrics.classification import compute_classification_metrics

__all__ = [
    "compute_dice",
    "compute_iou",
    "extract_instances_from_colour_mask",
    "compute_symmetric_best_dice",
    "compute_mask_map",
    "compute_segmentation_metrics",
    "compute_classification_metrics",
]
