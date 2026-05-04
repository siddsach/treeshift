#!/bin/bash
#SBATCH --job-name=frcnn_pt_india_random_80_20
#SBATCH --partition=serc
#SBATCH --gres=gpu:2
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=16
#SBATCH --mem=96G
#SBATCH --time=36:00:00
#SBATCH --output=logs/fastrcnn_pretrained_india_random_80_20_%j.out
#SBATCH --error=logs/fastrcnn_pretrained_india_random_80_20_%j.err

# -----------------------------------------------------------------------
# Faster R-CNN (COCO pretrained): india_random_80_20
# 10794 train images, batch 20 (2 GPUs), 50 epochs = 27000 iterations.
# Early stopping: patience 5 epochs. train + eval ID + eval OOD + SHIFT.
# -----------------------------------------------------------------------

set -euo pipefail

REPO_ROOT="/scratch/groups/dlobell/siddsach/treeshift"
IMAGE_SIF="${DETECTRON_SIF:-/scratch/groups/dlobell/aadityan/tree-distribution-shift/detectron.sif}"
METADATA_CSV="${METADATA_CSV:-/scratch/groups/dlobell/aadityan/dataset/metadata.csv}"

CONFIG=india_random_80_20
OUTPUT_DIR="${REPO_ROOT}/outputs/fastrcnn_pretrained_india_random_80_20_${SLURM_JOB_ID}"

cd "${REPO_ROOT}"
mkdir -p logs outputs

if [[ ! -f "${IMAGE_SIF}" ]]; then
  echo "ERROR: Apptainer image not found: ${IMAGE_SIF}"
  echo "Run: apptainer build --fakeroot detectron.sif detectron.def"
  exit 1
fi

# ----- 1. Train + evaluate on ID test, OOD test, and OOD train -----
apptainer exec --nv \
  --bind "${REPO_ROOT}:/workspace" \
  "${IMAGE_SIF}" \
  python detectron_fastrcnn.py run \
  --config "${CONFIG}" \
  --train --eval-val \
  --output-dir "${OUTPUT_DIR}" \
  --batch-size 20 \
  --max-iterations 27000 \
  --learning-rate 0.005 \
  --pretrained \
  --early-stopping-patience 5

RUN_ERR=$?
if [[ $RUN_ERR -ne 0 ]]; then
  echo "detectron_fastrcnn.py failed with exit code $RUN_ERR"
  exit $RUN_ERR
fi

# ----- 2. ID-only baseline -----
# This config has no OOD split, so shift analysis is intentionally skipped.
echo ""
echo "Done. Outputs: ${OUTPUT_DIR}"
