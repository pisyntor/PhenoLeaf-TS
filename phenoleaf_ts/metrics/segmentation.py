"""Instance-segmentation metrics for PhenoLeaf-TS (Dice, IoU, SBD, mask AP)."""

import numpy as np
from PIL import Image


def compute_dice(pred, gt):
    """Dice coefficient between two binary masks."""
    intersection = np.logical_and(pred, gt).sum()
    return 2.0 * intersection / (pred.sum() + gt.sum() + 1e-8)


def compute_iou(pred, gt):
    """IoU between two binary masks."""
    intersection = np.logical_and(pred, gt).sum()
    union = np.logical_or(pred, gt).sum()
    return intersection / (union + 1e-8)


def extract_instances_from_colour_mask(colour_mask):
    """Extract per-leaf binary masks from a colour-coded instance mask.

    Args:
        colour_mask: H x W x 3 array with a unique colour per leaf.
    Returns:
        list of binary masks (H x W), one per leaf instance.
    """
    pixels = colour_mask.reshape(-1, 3)
    unique_colours = np.unique(pixels, axis=0)

    instances = []
    for colour in unique_colours:
        if np.all(colour == 0):  # skip background (black)
            continue
        instances.append(np.all(colour_mask == colour, axis=-1))
    return instances


def compute_symmetric_best_dice(pred_instances, gt_instances):
    """Symmetric Best Dice (SBD) — the standard CVPPP LSC metric.

    SBD = min(mean BestDice(pred->gt), mean BestDice(gt->pred)).
    """
    if len(pred_instances) == 0 and len(gt_instances) == 0:
        return 1.0
    if len(pred_instances) == 0 or len(gt_instances) == 0:
        return 0.0

    dice_matrix = np.zeros((len(pred_instances), len(gt_instances)))
    for i, p in enumerate(pred_instances):
        for j, g in enumerate(gt_instances):
            dice_matrix[i, j] = compute_dice(p, g)

    mean_bd_pred = dice_matrix.max(axis=1).mean()
    mean_bd_gt = dice_matrix.max(axis=0).mean()
    return min(mean_bd_pred, mean_bd_gt)


def compute_mask_map(pred_instances, pred_scores, gt_instances, iou_thresholds=None):
    """Mask mAP at COCO-style IoU thresholds.

    Args:
        pred_instances: list of predicted binary masks.
        pred_scores: confidence score per prediction.
        gt_instances: list of ground-truth binary masks.
        iou_thresholds: IoU thresholds (default 0.50:0.05:0.95).
    Returns:
        dict with AP, AP50, AP75.
    """
    if iou_thresholds is None:
        iou_thresholds = np.arange(0.5, 1.0, 0.05)

    if len(gt_instances) == 0:
        v = 1.0 if len(pred_instances) == 0 else 0.0
        return {"AP": v, "AP50": v, "AP75": v}
    if len(pred_instances) == 0:
        return {"AP": 0.0, "AP50": 0.0, "AP75": 0.0}

    sorted_indices = np.argsort(-np.array(pred_scores))
    pred_instances = [pred_instances[i] for i in sorted_indices]

    aps = {}
    for thresh in iou_thresholds:
        gt_matched = [False] * len(gt_instances)
        tp, fp = [], []
        for pred in pred_instances:
            best_iou, best_gt_idx = 0, -1
            for j, gt in enumerate(gt_instances):
                if gt_matched[j]:
                    continue
                iou = compute_iou(pred, gt)
                if iou > best_iou:
                    best_iou, best_gt_idx = iou, j
            if best_iou >= thresh and best_gt_idx >= 0:
                tp.append(1); fp.append(0); gt_matched[best_gt_idx] = True
            else:
                tp.append(0); fp.append(1)

        tp_cumsum, fp_cumsum = np.cumsum(tp), np.cumsum(fp)
        recalls = tp_cumsum / len(gt_instances)
        precisions = tp_cumsum / (tp_cumsum + fp_cumsum)
        ap = 0
        for r_thresh in np.arange(0, 1.1, 0.1):
            prec = precisions[recalls >= r_thresh]
            if len(prec) > 0:
                ap += prec.max() / 11
        aps[thresh] = ap

    return {
        "AP": float(np.mean(list(aps.values()))),
        "AP50": float(aps.get(0.5, 0.0)),
        "AP75": float(aps.get(0.75, 0.0)),
    }


def compute_segmentation_metrics(pred_mask_path, gt_mask_path):
    """All single-image segmentation metrics for a predicted vs GT colour mask."""
    pred_mask = np.array(Image.open(pred_mask_path).convert("RGB"))
    gt_mask = np.array(Image.open(gt_mask_path).convert("RGB"))

    pred_instances = extract_instances_from_colour_mask(pred_mask)
    gt_instances = extract_instances_from_colour_mask(gt_mask)

    pred_fg = np.any(pred_mask > 0, axis=-1)
    gt_fg = np.any(gt_mask > 0, axis=-1)

    return {
        "dice": compute_dice(pred_fg, gt_fg),
        "overall_iou": compute_iou(pred_fg, gt_fg),
        "leaf_iou": compute_iou(pred_fg, gt_fg),
        "background_iou": compute_iou(~pred_fg, ~gt_fg),
        "sbd": compute_symmetric_best_dice(pred_instances, gt_instances),
        "num_pred_instances": len(pred_instances),
        "num_gt_instances": len(gt_instances),
        "leaf_count_diff": abs(len(pred_instances) - len(gt_instances)),
    }
