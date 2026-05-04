#!/bin/bash
#SBATCH --job-name=fastrcnn_WET_DRY
#SBATCH --partition=serc
#SBATCH --gres=gpu:1
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=24G
#SBATCH --time=12:00:00
#SBATCH --output=logs/fastrcnn_WET_DRY_%j.out
#SBATCH --error=logs/fastrcnn_WET_DRY_%j.err

# -----------------------------------------------------------------------------
# Fast R-CNN: Biome shift — train Rajasthan WET, test Rajasthan DRY. 1 GPU.
# Use biome_Karnataka_train_WET__ood_DRY if your tree-shift has Karnataka config.
# ~3.5k train images, batch 8, 30 epochs -> 13.5k iterations.
# -----------------------------------------------------------------------------

ml load cuda/12.1.1
source /scratch/groups/dlobell/aadityan/tree-distribution-shift/.venv/bin/activate
cd /scratch/groups/dlobell/aadityan/tree-distribution-shift

mkdir -p logs outputs

CONFIG=biome_Rajasthan_train_WET__ood_DRY
OUTPUT_DIR=./outputs/fastrcnn_WET_DRY_${SLURM_JOB_ID}
COCO_DIR="${OUTPUT_DIR}/coco_export"
METADATA_CSV=/scratch/groups/dlobell/aadityan/dataset/metadata.csv

# ----- 1. Train + evaluate on val and OOD -----
python detectron_fastrcnn.py run \
  --coco-dir "${COCO_DIR}" \
  --config "${CONFIG}" \
  --train --eval-val --eval-ood \
  --output-dir "${OUTPUT_DIR}" \
  --batch-size 8 \
  --max-iterations 13500

RUN_ERR=$?
if [ $RUN_ERR -ne 0 ]; then
  echo "detectron_fastrcnn.py run failed with exit code $RUN_ERR"
  exit $RUN_ERR
fi

# ----- 2. Shift analysis (univariate + Shapley) -----
ID_RESULTS="${OUTPUT_DIR}/eval_val/per_image_results.json"
OOD_RESULTS="${OUTPUT_DIR}/eval_ood_test/per_image_results.json"
SHIFT_OUT="${OUTPUT_DIR}/shift_analysis"

if [ ! -f "${ID_RESULTS}" ] || [ ! -f "${OOD_RESULTS}" ]; then
  echo "Skipping shift analysis: missing eval outputs."
  exit 0
fi

if [ ! -f "${METADATA_CSV}" ]; then
  echo "Skipping shift analysis: metadata not found at ${METADATA_CSV}"
  exit 0
fi

echo "Running univariate + Shapley decomposition ..."
python shift_analysis.py univariate \
  --id-results "${ID_RESULTS}" \
  --ood-results "${OOD_RESULTS}" \
  --metadata "${METADATA_CSV}" \
  --output-dir "${SHIFT_OUT}"

python shift_analysis.py shapley \
  --id-results "${ID_RESULTS}" \
  --ood-results "${OOD_RESULTS}" \
  --metadata "${METADATA_CSV}" \
  --output-dir "${SHIFT_OUT}"

echo "Done. Outputs: ${OUTPUT_DIR}"
