# TreeShift

TreeShift is a benchmark and experiment codebase for individual tree-crown detection under geographic distribution shift. This repository is the curated code release for running the detection experiments and reproducing paper tables. Data, model weights, containers, and experiment outputs are intentionally handled outside Git.

## What is included

- Model wrappers for Faster R-CNN, Mask R-CNN, Plain-DETR, and GroundingDINO.
- Shared COCO export/path handling, unified detection metrics, and distribution-shift analysis.
- Vendored Plain-DETR fork used by the experiment wrapper.
- Slurm experiment scripts in `scripts2/`, with generators and verification tooling.
- Container definitions for the Plain-DETR/GroundingDINO and Detectron2 environments.
- Dataset/config construction tools under `tools/data/`.

## What is not included

This repository should not track raw imagery, COCO exports, metadata CSVs, model checkpoints, pretrained weights, Apptainer `.sif` files, logs, or model outputs. See `DATA.md` for the expected external data layout and `REPRODUCIBILITY.md` for the experiment checklist.

## Repository layout

- `shared_utils.py`: COCO export/path helpers and unified per-image/tree metrics.
- `plain_detr.py`, `grounding_dino.py`, `detectron_fastrcnn.py`, `detectron_maskrcnn.py`: model-specific train/eval entrypoints.
- `shift_analysis.py`: univariate and Shapley-style ID/OOD covariate analysis.
- `Plain-DETR/`: local Plain-DETR fork imported by `plain_detr.py`.
- `scripts2/`: canonical 1-GPU, 40-epoch Slurm scripts used as the reproducible launch surface.
- `scripts/`: base generated scripts and generator source used to regenerate `scripts2/`.
- `tools/data/`: dataset split/config generation and COCO export tooling.
- `tools/verify_generated_scripts.py`: regenerates scripts in a temp directory and diffs them against committed scripts.

## Quick checks

From the repository root:

```bash
python tools/verify_generated_scripts.py
python -m py_compile shared_utils.py plain_detr.py grounding_dino.py detectron_fastrcnn.py detectron_maskrcnn.py shift_analysis.py
find . -type f -size +50M -not -path './.git/*'
```

The large-file check should print nothing for a GitHub-ready checkout.

## Running experiments

Data setup is separate. Once the TreeShift dataset is available through the `tree-shift` package or a local COCO export, use the model wrappers directly or submit one of the canonical scripts in `scripts2/`.

Example wrapper invocation:

```bash
python detectron_fastrcnn.py run \
  --config region_train_North__ood_South \
  --coco-dir ./coco_export \
  --train --eval-val --eval-ood --eval-ood-train \
  --output-dir ./outputs/fastrcnn_pretrained_region_North_South \
  --pretrained
```

Example Slurm invocation:

```bash
sbatch scripts2/fastrcnn_pretrained_region_North_South.sh
```

The Slurm scripts assume an HPC environment with Apptainer. DINOv3 scripts support `DINOV3_REPO`, `DINO_WEIGHTS_ROOT`, and `DINO_WEIGHTS` overrides; other data paths are documented in `DATA.md` and `REPRODUCIBILITY.md`.
