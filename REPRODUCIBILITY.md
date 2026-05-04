# Reproducibility Checklist

This file records what must be available to reproduce the TreeShift experiments for the paper.

## Code

- Use this repository for model wrappers, metrics, shift analysis, script generation, and aggregation.
- Use `scripts2/` as the canonical Slurm launch surface.
- Run `python tools/verify_generated_scripts.py` before submitting jobs or committing script changes.

## Environments

Two environments are defined:

- `tree-shift.def` / `Dockerfile`: Plain-DETR, DINOv3, GroundingDINO, and MMDetection.
- `detectron.def`: Detectron2, Faster R-CNN, and Mask R-CNN.

The built `.sif` files are external artifacts and should not be committed.

## DINOv3

Container builds clone `facebookresearch/dinov3` at commit `31703e4cbf1ccb7c4a72daa1350405f86754b6d1` into `/opt/dinov3`. Canonical scripts allow overriding the repo path with `DINOV3_REPO`.

Plain-DETR DINOv3 scripts resolve weights through `DINO_WEIGHTS_ROOT` and `DINO_WEIGHTS`:

- `plaindetr_dinov3`: `${DINO_WEIGHTS_ROOT}/dinov3_vits.pth` unless `DINO_WEIGHTS` is set.
- `plaindetr_dinov3_7b16`: `${DINO_WEIGHTS_ROOT}/dino_weights_7b16.pth` unless `DINO_WEIGHTS` is set.
- `plaindetr_dinov3_sat`: `${DINO_WEIGHTS_ROOT}/dino_weights_sat.pth` unless `DINO_WEIGHTS` is set.

The default `DINO_WEIGHTS_ROOT` in generated scripts is `${REPO_ROOT}/../dino_weights`. Public release notes should provide checksums and download URLs for each required checkpoint.

## External weights

Expected external weights include:

- MM-Grounding-DINO pretrained weights referenced by the GroundingDINO scripts.
- DINOv3 weights referenced by Plain-DETR DINOv3 scripts.
- Any custom DINOv3 or satellite-pretrained checkpoints used for ablations.

Store these outside Git and document their exact paths, checksums, and download URLs in the final release notes.

## Metrics and outputs

Each model run should produce:

- `eval_val/per_image_results.json`
- `eval_ood_test/per_image_results.json`
- `eval_ood_train/per_image_results.json` when the split exists
- `eval_summary.json`
- optional `shift_analysis/` outputs

Use `scripts2/aggregate_results.py` to build cross-model summary tables from a curated output directory.

## Pre-release checks

```bash
python tools/verify_generated_scripts.py
python -m py_compile shared_utils.py plain_detr.py grounding_dino.py detectron_fastrcnn.py detectron_maskrcnn.py shift_analysis.py
find . -type f -size +50M -not -path './.git/*'
git status --short
```

A GitHub-ready source release should not contain large files, data exports, logs, containers, or checkpoints.
