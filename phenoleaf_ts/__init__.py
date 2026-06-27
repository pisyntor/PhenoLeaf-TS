"""PhenoLeaf-TS — time-series leaf instance segmentation, tracking, and growth-stage benchmark.

Subpackages:
    phenoleaf_ts.data     — dataset preparation and loading (incl. Hugging Face)
    phenoleaf_ts.metrics  — segmentation / classification / tracking metrics
    phenoleaf_ts.models   — baseline model registry and loaders
"""

__version__ = "0.1.0"

HF_DATASET = "rick77a/PhenoLeaf_TS"
HF_DATASET_ZIP = "PhenoLeaf-TS_DS.zip"

from phenoleaf_ts import data, metrics, models  # noqa: E402,F401
