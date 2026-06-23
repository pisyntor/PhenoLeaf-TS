"""Baseline model registry and loaders for PhenoLeaf-TS.

The registry is declared in ``configs/models.yaml``. Use :func:`list_models` to inspect
the 21 baselines and :func:`load_model` to instantiate one from a released checkpoint.

>>> from phenoleaf_ts.models import list_models, load_model
>>> list_models("classification")
>>> clf = load_model("CL-6", checkpoint="checkpoints/CL-6_swin_t.pth")
"""

import os
import yaml

_REGISTRY_PATH = os.path.join(
    os.path.dirname(os.path.dirname(os.path.dirname(__file__))), "configs", "models.yaml"
)


def _registry():
    with open(_REGISTRY_PATH) as f:
        return yaml.safe_load(f)


def list_models(task=None):
    """Return registry entries, optionally filtered by task
    ('segmentation' | 'tracking' | 'classification')."""
    reg = _registry()
    if task is not None:
        return reg[task]
    return [m for entries in reg.values() for m in entries]


def get_spec(model_id):
    """Look up a model's registry spec by id (e.g. 'IS-3', 'CL-6')."""
    for m in list_models():
        if m["id"] == model_id:
            return m
    raise KeyError(f"Unknown model id: {model_id}")


def load_model(model_id, checkpoint, device="cuda", **kwargs):
    """Load a baseline by id from a released checkpoint.

    Dispatches to the appropriate framework loader (Ultralytics / Detectron2 / timm /
    BoxMOT). Frameworks are imported lazily so only the one you use is required.
    """
    spec = get_spec(model_id)
    fw = spec.get("framework")

    if fw == "ultralytics":
        from ultralytics import YOLO
        return YOLO(checkpoint)

    if fw == "timm":
        import timm
        import torch
        model = timm.create_model(spec["timm_id"], num_classes=3, **kwargs)
        state = torch.load(checkpoint, map_location=device)
        model.load_state_dict(state.get("state_dict", state))
        return model.eval().to(device)

    if fw == "detectron2":
        from detectron2.config import get_cfg
        from detectron2.engine import DefaultPredictor
        cfg = get_cfg()
        if "config_file" in kwargs:
            cfg.merge_from_file(kwargs["config_file"])
        cfg.MODEL.WEIGHTS = checkpoint
        cfg.MODEL.DEVICE = device
        return DefaultPredictor(cfg)

    if fw in ("boxmot", "norfair"):
        raise NotImplementedError(
            f"{model_id} is a tracker; construct it directly (e.g. `from boxmot import "
            f"ByteTrack`) and run it on IS-1 (YOLOv11) detections — see the README."
        )

    raise ValueError(f"No loader for framework '{fw}' ({model_id}).")
