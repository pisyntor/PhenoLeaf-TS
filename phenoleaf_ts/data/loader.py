"""Download PhenoLeaf-TS from the Hugging Face Hub.

The dataset is released as a single archive (``PhenoLeaf-TS_DS.zip``, ~5.5 GB) at
https://huggingface.co/datasets/rick77a/PhenoLeaf_TS. This helper downloads and
extracts it; pass the extracted folder to ``tools/prepare_data.py`` to build the
model-ready splits.

Example
-------
>>> from phenoleaf_ts.data import download_phenoleaf
>>> path = download_phenoleaf("data")          # -> data/leaf_dataset_colour/...
"""

import os
import zipfile

from phenoleaf_ts import HF_DATASET, HF_DATASET_ZIP


def download_phenoleaf(dest="data", repo_id=HF_DATASET, filename=HF_DATASET_ZIP, extract=True):
    """Download (and optionally extract) the PhenoLeaf-TS archive from the HF Hub.

    Args:
        dest: directory to download/extract into.
        repo_id: HF dataset repo (default ``rick77a/PhenoLeaf_TS``).
        filename: archive name in the repo.
        extract: unzip the archive after download.
    Returns:
        path to the downloaded archive.
    """
    try:
        from huggingface_hub import hf_hub_download
    except ImportError as e:  # pragma: no cover
        raise ImportError("Install the Hub client: pip install huggingface_hub") from e

    os.makedirs(dest, exist_ok=True)
    zip_path = hf_hub_download(
        repo_id=repo_id, filename=filename, repo_type="dataset", local_dir=dest
    )
    if extract:
        with zipfile.ZipFile(zip_path) as z:
            z.extractall(dest)
    return zip_path
