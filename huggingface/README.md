---
license: cc-by-nc-4.0
pretty_name: PhenoLeaf-TS
task_categories:
  - image-segmentation
  - image-classification
tags:
  - plant-phenotyping
  - leaf-instance-segmentation
  - tracking
  - arabidopsis
  - time-series
size_categories:
  - 10K<n<100K
---

# PhenoLeaf-TS

A time-series dataset of **17,082** top-down RGB images across **21** *Arabidopsis thaliana*
genotypes (**318** plant replicates), annotated with temporally consistent, colour-coded leaf
instance masks (fixed 31-colour palette). Supports instance segmentation, leaf tracking, and
growth-stage classification. Companion to the ECCV 2026 paper.

- Code & benchmark: https://github.com/pisyntor/PhenoLeaf-TS
- Project page: https://pisyntor.github.io/PhenoLeaf-TS/

## Schema

| field | type | description |
|-------|------|-------------|
| `image` | image | RGB frame (~530×525) |
| `mask` | image | colour-coded leaf instance mask (same size) |
| `genotype` | string | Arabidopsis accession |
| `replicate` | string | plant replicate id |
| `frame_idx` | int | temporal index within the replicate sequence |
| `leaf_count` | int | number of leaf instances |
| `growth_stage` | class label | 0 = Early (4–6), 1 = Intermediate (7–10), 2 = Mature (11+) |

Splits: `train` / `val` / `test` (70 / 15 / 15, within-sequence).

## Usage

```python
from datasets import load_dataset
ds = load_dataset("basimazam/PhenoLeaf-TS", split="test", streaming=True)
ex = next(iter(ds))
ex["image"], ex["mask"], ex["growth_stage"]
```

Or via the benchmark package: `from phenoleaf_ts.data import load_phenoleaf`.

## Citation

```bibtex
@InProceedings{saric2026phenoleafts,
  author    = {Sari\'c, Rijad and Azam, Basim and Khan, Sarmad and \v{C}ustovi\'c, Edhem},
  title     = {{PhenoLeaf-TS}: A Time-Series Benchmark for Leaf Instance Segmentation, Tracking, and Growth Stage Classification},
  booktitle = {Computer Vision -- ECCV 2026},
  year      = {2026},
  publisher = {Springer Nature Switzerland},
}
```
