# 2p-NeuSeg

Code and evaluation scripts for **2P-NeuSeg**, a framework for neuronal instance segmentation, normal/injured
classification and morphology-associated phenotyping in label-free two-photon microscopy (TPM) images of ischaemic
rat brain tissue.

Manuscript: *Segmentation and Stratification of Ischaemic Neuronal Injury in Label-Free Two-Photon Images*
(Biomedical Signal Processing and Control, BSPC-D-26-13479).

## Repository layout

```
code/                       training and inference code (MyNet / HoVer-Net style pipeline)
  models/MyNet/             encoder, decoders, targets and post-processing
  dataloader/               data loaders and augmentation
  run_train.py              two-stage training
  run_infer.py              tile-based inference
  run_ablation.py           GIF/OGS 2x2 ablation
  eval/                     evaluation and robustness scripts added for the revision
data/                       example image (iii109.tif); the full dataset is not redistributed here
```

## Environment

Python 3.9+, PyTorch, and the packages in `code/requirements.txt` (plus `scikit-image`, `scipy`, `scikit-learn`,
`pandas` and `opencv-python`). The inference and evaluation scripts were tested with PyTorch 2.7 (CUDA 12.8).

## Data availability

The expert-consensus annotated dataset (1,386 label-free TPM images, instance masks, normal/injured labels, split
manifest and annotation protocol) is available from the corresponding author on reasonable request. The source code
and trained model weights are provided in this repository; a permanent data DOI will be added when deposition is
complete.

Reference annotations are expert-consensus references, not per-instance histopathological ground truth. The
morphological criteria follow the TPM–H&E correlation established in the companion MCAO study (Biomed. Opt. Express
9(5):2312, 2018).

## Ethics

All animal experiments were approved by the Institutional Animal Care and Use Committee of Fujian Medical University
and performed in accordance with the National Institutes of Health Guide for the Care and Use of Laboratory Animals.
Animals were supplied under licence SCXK 2014-007.

## Evaluation protocol

Unless stated otherwise, predicted and reference instances are matched one-to-one by centroid distance with a
**12-pixel radius**. Metrics implemented in `code/eval/`:

- `unified_metrics.py` – detection precision/recall/F1, matched-instance macro/micro/support-weighted F1,
  detection-aware F1, per-class F1, Dice and AJI, with bootstrap confidence intervals;
- `evaluate_12px.py` – uniform 12-pixel evaluation for one or more prediction directories;
- `boundary_robustness.py` – fixed-prediction sensitivity to ±1/2/3-pixel dilation, erosion and local jitter of the
  reference contours;
- `train_ogs_experiments.py` – multi-seed GIF/OGS ablation and OGS-loss comparison (weighted MSE, focal BCE,
  Dice–Tversky);
- `analyze_clustering_regions.py` – k = 2–5 clustering stability, image-level bootstrap and regional weight-scheme
  sensitivity.

Example:

```bash
python code/eval/unified_metrics.py \
  --gt-dir /path/to/labels \
  --method 2P-NeuSeg=/path/to/predictions/mat \
  --radius 12 --out-dir results/metrics
```

## Citation

If you use this code, please cite the manuscript above. The code is released under the MIT licence (see `code/LICENSE`).
