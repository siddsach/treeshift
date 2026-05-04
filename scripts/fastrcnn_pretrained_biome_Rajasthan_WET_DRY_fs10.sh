#!/bin/bash
#SBATCH --job-name=frcnn_pt_biome_Rajasthan_WET_DRY_fs10
#SBATCH --partition=serc
#SBATCH --gres=gpu:2
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=16
#SBATCH --mem=64G
#SBATCH --time=12:00:00
#SBATCH --output=logs/fastrcnn_pretrained_biome_Rajasthan_WET_DRY_fs10_%j.out
#SBATCH --error=logs/fastrcnn_pretrained_biome_Rajasthan_WET_DRY_fs10_%j.err

# -----------------------------------------------------------------------
# Faster R-CNN (COCO pretrained): biome_Rajasthan_train_WET__ood_DRY__fs10
# 1724 train images, batch 16 (2 GPUs), 50 epochs = 5400 iterations.
# Early stopping: patience 5 epochs. train + eval ID + eval OOD + SHIFT.
# -----------------------------------------------------------------------

set -euo pipefail

REPO_ROOT="/scratch/groups/dlobell/aadityan/tree-distribution-shift"
IMAGE_SIF="${REPO_ROOT}/detectron.sif"
METADATA_CSV="/scratch/groups/dlobell/aadityan/tree-distribution-shift/../dataset/metadata.csv"

CONFIG=biome_Rajasthan_train_WET__ood_DRY__fs10
OUTPUT_DIR="${REPO_ROOT}/outputs/fastrcnn_pretrained_biome_Rajasthan_WET_DRY_fs10_${SLURM_JOB_ID}"

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
  --train --eval-val --eval-ood --eval-ood-train \
  --output-dir "${OUTPUT_DIR}" \
  --batch-size 16 \
  --max-iterations 5400 \
  --learning-rate 0.005 \
  --pretrained \
  --early-stopping-patience 5

RUN_ERR=$?
if [[ $RUN_ERR -ne 0 ]]; then
  echo "detectron_fastrcnn.py failed with exit code $RUN_ERR"
  exit $RUN_ERR
fi

# ----- 2. Shift analysis (univariate + Shapley) -----
# Uses eval_val (ID test) and eval_ood_train (OOD train pool) for shift analysis.
# eval_ood_test is the fixed held-out set used only for the reported metric.
ID_RESULTS="${OUTPUT_DIR}/eval_val/per_image_results.json"
OOD_RESULTS="${OUTPUT_DIR}/eval_ood_train/per_image_results.json"
SHIFT_OUT="${OUTPUT_DIR}/shift_analysis"

if [[ ! -f "${ID_RESULTS}" ]] || [[ ! -f "${OOD_RESULTS}" ]]; then
  echo "Skipping shift analysis: missing eval outputs."
  exit 0
fi

if [[ ! -f "${METADATA_CSV}" ]]; then
  echo "Skipping shift analysis: metadata not found at ${METADATA_CSV}"
  exit 0
fi

echo ""
echo "Running univariate + Shapley decomposition ..."
apptainer exec --nv \
  --bind "${REPO_ROOT}:/workspace" \
  "${IMAGE_SIF}" \
  python shift_analysis.py univariate \
  --id-results "${ID_RESULTS}" \
  --ood-results "${OOD_RESULTS}" \
  --metadata "${METADATA_CSV}" \
  --output-dir "${SHIFT_OUT}"

apptainer exec --nv \
  --bind "${REPO_ROOT}:/workspace" \
  "${IMAGE_SIF}" \
  python shift_analysis.py shapley \
  --id-results "${ID_RESULTS}" \
  --ood-results "${OOD_RESULTS}" \
  --metadata "${METADATA_CSV}" \
  --output-dir "${SHIFT_OUT}"

echo ""
echo ""
echo "Done. Outputs: ${OUTPUT_DIR}"
