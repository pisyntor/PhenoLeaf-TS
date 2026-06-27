#!/usr/bin/env python
"""CLI: evaluate predictions against PhenoLeaf-TS ground truth.

Segmentation — directory of predicted colour masks vs GT colour masks (matched by filename):
    python tools/evaluate.py --task segmentation --pred_dir preds/ --gt_dir gts/

Classification — JSON/CSV with predicted vs true growth-stage labels:
    python tools/evaluate.py --task classification --pred predictions.json
"""
import argparse
import glob
import json
import os

import numpy as np

from phenoleaf_ts.metrics import (
    compute_segmentation_metrics,
    compute_classification_metrics,
)


def eval_segmentation(pred_dir, gt_dir):
    keys = ["dice", "leaf_iou", "sbd", "leaf_count_diff"]
    agg = {k: [] for k in keys}
    for pred_path in sorted(glob.glob(os.path.join(pred_dir, "*.png"))):
        gt_path = os.path.join(gt_dir, os.path.basename(pred_path))
        if not os.path.exists(gt_path):
            continue
        m = compute_segmentation_metrics(pred_path, gt_path)
        for k in keys:
            agg[k].append(m[k])
    return {k: float(np.mean(v)) if v else None for k, v in agg.items()}


def eval_classification(pred_file):
    with open(pred_file) as f:
        data = json.load(f)
    return compute_classification_metrics(
        data["y_true"], data["y_pred"], num_classes=data.get("num_classes", 3)
    )


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--task", required=True, choices=["segmentation", "classification"])
    p.add_argument("--pred_dir"); p.add_argument("--gt_dir")
    p.add_argument("--pred")
    args = p.parse_args()

    if args.task == "segmentation":
        result = eval_segmentation(args.pred_dir, args.gt_dir)
    else:
        result = eval_classification(args.pred)
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
