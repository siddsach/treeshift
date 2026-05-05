#!/bin/bash
#SBATCH --job-name=mrcnn_biome_Rajasthan_WET_DRY_fs10
#SBATCH --partition=serc
#SBATCH --gres=gpu:1
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=16
#SBATCH --mem=64G
#SBATCH --time=12:00:00
#SBATCH --output=logs/maskrcnn_biome_Rajasthan_WET_DRY_fs10_%j.out
#SBATCH --error=logs/maskrcnn_biome_Rajasthan_WET_DRY_fs10_%j.err

# -----------------------------------------------------------------------
# Mask R-CNN: biome_Rajasthan_train_WET__ood_DRY__fs10
# 1724 train images, batch 8 (1 GPU), 50 epochs = 10800 iterations.
# Early stopping: patience 5 epochs. train + eval ID + eval OOD + SHIFT.
# -----------------------------------------------------------------------

set -euo pipefail

REPO_ROOT="/scratch/groups/dlobell/siddsach/treeshift"
IMAGE_SIF="${DETECTRON_SIF:-/scratch/groups/dlobell/aadityan/tree-distribution-shift/detectron.sif}"
METADATA_CSV="${METADATA_CSV:-/scratch/groups/dlobell/aadityan/dataset/metadata.csv}"

CONFIG=biome_Rajasthan_train_WET__ood_DRY__fs10
OUTPUT_DIR="${REPO_ROOT}/outputs/maskrcnn_biome_Rajasthan_WET_DRY_fs10_${SLURM_JOB_ID}"

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
  python detectron_maskrcnn.py run \
  --config "${CONFIG}" \
  --train --eval-val --eval-ood --eval-ood-train \
  --eval-mode distshift \
  --output-dir "${OUTPUT_DIR}" \
  --batch-size 8 \
  --max-iterations 10800 \
  --early-stopping-patience 5

RUN_ERR=$?
if [[ $RUN_ERR -ne 0 ]]; then
  echo "detectron_maskrcnn.py failed with exit code $RUN_ERR"
  exit $RUN_ERR
fi

# ----- 2. Post-hoc shift analysis -----
# Shift analysis is intentionally not run inside training jobs.
# Use saved per-image eval outputs for post-hoc analysis after jobs finish.
echo ""
echo "Done. Outputs: ${OUTPUT_DIR}"
echo "Post-hoc shift inputs, when available:"
echo "  ${OUTPUT_DIR}/eval_val/per_image_results.json"
echo "  ${OUTPUT_DIR}/eval_ood_train/per_image_results.json"
