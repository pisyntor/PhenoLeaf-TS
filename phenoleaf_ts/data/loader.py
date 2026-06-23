"""Load PhenoLeaf-TS from the Hugging Face Hub.

The dataset is published at https://huggingface.co/datasets/basimazam/PhenoLeaf-TS,
exposing one example per frame with the RGB image, the colour-coded instance mask,
genotype / replicate / frame indices, leaf count, and growth-stage label.

Example
-------
>>> from phenoleaf_ts.data import load_phenoleaf
>>> ds = load_phenoleaf(split="test")
>>> ex = ds[0]
>>> ex["image"], ex["mask"], ex["growth_stage"]
"""

from phenoleaf_ts import HF_DATASET


def load_phenoleaf(split=None, streaming=False, **kwargs):
    """Return the PhenoLeaf-TS dataset via `datasets.load_dataset`.

    Args:
        split: "train" | "val" | "test", or None for a DatasetDict.
        streaming: stream examples instead of downloading everything up front
            (recommended for the full 17k-image set).
        **kwargs: forwarded to `datasets.load_dataset`.
    """
    try:
        from datasets import load_dataset
    except ImportError as e:  # pragma: no cover
        raise ImportError(
            "The `datasets` package is required: pip install datasets"
        ) from e

    return load_dataset(HF_DATASET, split=split, streaming=streaming, **kwargs)
