#!/bin/bash
#SBATCH --job-name=maskrcnn_South_North
#SBATCH --partition=serc
#SBATCH --gres=gpu:1
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=24G
#SBATCH --time=12:00:00
#SBATCH --output=logs/maskrcnn_South_North_%j.out
#SBATCH --error=logs/maskrcnn_South_North_%j.err

set -euo pipefail

# -----------------------------------------------------------------------------
# Mask R-CNN: National shift — train South India, test North India. 1 GPU.
# Training + eval (val + OOD) + univariate + Shapley shift analysis.
# ~10k train images, batch 10, 30 epochs -> 30k iterations.
# -----------------------------------------------------------------------------

ml load cuda/12.1.1
source /scratch/groups/dlobell/aadityan/tree-distribution-shift/.venv/bin/activate
cd /scratch/groups/dlobell/aadityan/tree-distribution-shift

mkdir -p logs outputs

CONFIG=region_train_South__ood_North
OUTPUT_DIR=./outputs/maskrcnn_national_South_North_${SLURM_JOB_ID}
COCO_DIR="${OUTPUT_DIR}/coco_export"
METADATA_CSV=/scratch/groups/dlobell/aadityan/dataset/metadata.csv

# Export dataset (skips if exists)
echo "Exporting COCO dataset for ${CONFIG}..."
tree-shift export --config "${CONFIG}" --out "${COCO_DIR}"

python detectron_maskrcnn.py run \
  --config "${CONFIG}" \
  --coco-dir "${COCO_DIR}" \
  --train --eval-val --eval-ood \
  --eval-mode distshift \
  --output-dir "${OUTPUT_DIR}" \
  --batch-size 10 \
  --max-iterations 30000

# Shift analysis (univariate + Shapley)
ID_RESULTS="${OUTPUT_DIR}/eval_val/per_image_results.json"
OOD_RESULTS="${OUTPUT_DIR}/eval_ood_test/per_image_results.json"
SHIFT_OUT="${OUTPUT_DIR}/shift_analysis"

if [[ -f "${ID_RESULTS}" && -f "${OOD_RESULTS}" && -f "${METADATA_CSV}" ]]; then
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
else
  echo "Skipping shift analysis: missing results or metadata"
fi

echo "Done. Outputs: ${OUTPUT_DIR}"
