# PhenoLeaf-TS

**A Time-Series Benchmark for Leaf Instance Segmentation, Tracking, and Growth Stage Classification**
*Rijad Sarić, Basim Azam, Sarmad Khan, Edhem Čustović* — ECCV 2026

🌐 **Project page:** https://pisyntor.github.io/PhenoLeaf-TS/

PhenoLeaf-TS is a time-series dataset of **17,082** top-down RGB images spanning **21** *Arabidopsis thaliana*
genotypes (**318** plant replicates), annotated with temporally consistent, colour-coded leaf instance masks
(fixed 31-colour palette). It supports three benchmark tasks in one dataset: **instance segmentation**,
**leaf tracking**, and **growth stage classification**.

## Key results

| Task | Best model | Score |
|------|-----------|-------|
| Instance segmentation | Mask R-CNN R50-FPN (IS-3) | **73.2 mAP** |
| Leaf tracking | ByteTrack (TR-5) | **84.1% MOTA / 91.0 IDF1** |
| Growth stage classification | Swin-T (CL-6) | **91.7% accuracy** |

Cross-dataset fine-tuning from PhenoLeaf-TS weights gives up to **+51.4 mAP** over zero-shot transfer on Komatsuna.

## Resources

- **Dataset (Figshare):** _DOI pending_
- **Paper PDF:** _link pending_
- **Trained models:** _host TBD_

## Repository layout

```
index.html        # GitHub Pages project site
assets/           # figures used by the site
# (code release — training/eval/transfer pipelines — to be added; see deliverables/REPO_STRUCTURE.md)
```

## Citation

```bibtex
@inproceedings{saric2026phenoleafts,
  title     = {PhenoLeaf-TS: A Time-Series Benchmark for Leaf Instance
               Segmentation, Tracking, and Growth Stage Classification},
  author    = {Sari\'c, Rijad and Azam, Basim and Khan, Sarmad and \v{C}ustovi\'c, Edhem},
  booktitle = {Proceedings of the European Conference on Computer Vision (ECCV)},
  year      = {2026}
}
```

## License

Code: [Apache-2.0](LICENSE). Dataset license: _to be confirmed_ (e.g. CC BY-NC 4.0).
