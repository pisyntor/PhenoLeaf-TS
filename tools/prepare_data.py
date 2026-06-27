#!/usr/bin/env python
"""CLI: prepare the raw PhenoLeaf-TS dataset into model-ready form.

    python tools/prepare_data.py --data_root data/leaf_dataset_colour \
        --output_dir data/prepared --format all
"""
from phenoleaf_ts.data.prepare import main

if __name__ == "__main__":
    main()
