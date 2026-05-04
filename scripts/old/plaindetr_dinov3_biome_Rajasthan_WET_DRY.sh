#!/bin/bash
#SBATCH --job-name=plaindetr_dinov3_WET_DRY
#SBATCH --partition=serc
#SBATCH --gres=gpu:2
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=16
#SBATCH --mem=128G
#SBATCH --time=24:00:00
#SBATCH --output=logs/plaindetr_dinov3_WET_DRY_%j.out
#SBATCH --error=logs/plaindetr_dinov3_WET_DRY_%j.err

# -----------------------------------------------------------------------------
# Plain-DETR with DinoV3 backbone: train Rajasthan WET, test Rajasthan DRY.
# Uses tree-shift Apptainer container. Training + eval + univariate + Shapley.
# Uses dinov3_vits16 (~22M params) - fits on 40GB GPUs.
# -----------------------------------------------------------------------------

set -euo pipefail

REPO_ROOT="/scratch/groups/dlobell/aadityan/tree-distribution-shift"
IMAGE_SIF="${REPO_ROOT}/tree-shift.sif"
METADATA_CSV="/scratch/groups/dlobell/aadityan/dataset/metadata.csv"
DINO_WEIGHTS="${DINO_WEIGHTS:-/scratch/groups/dlobell/aadityan/dino_weights/dinov3_vits.pth}"

CONFIG=biome_Rajasthan_train_WET__ood_DRY
OUTPUT_DIR="${REPO_ROOT}/outputs/plaindetr_dinov3_WET_DRY_${SLURM_JOB_ID}"
COCO_DIR="${OUTPUT_DIR}/coco_export"

cd "${REPO_ROOT}"
mkdir -p logs outputs

if [[ ! -f "${IMAGE_SIF}" ]]; then
  echo "ERROR: Image not found: ${IMAGE_SIF}"
  echo "Run: ./scripts/pull_apptainer.sh"
  exit 1
fi

if [[ ! -f "${DINO_WEIGHTS}" ]]; then
  echo "ERROR: DinoV3 ViT-S weights not found: ${DINO_WEIGHTS}"
  exit 1
fi

echo "=== Plain-DETR DinoV3: train Rajasthan WET, OOD Rajasthan DRY ==="
echo "Container: ${IMAGE_SIF}"
echo "Config: ${CONFIG}"
echo "COCO export: ${COCO_DIR} (auto-download via tree-shift)"
echo "Output: ${OUTPUT_DIR}"
echo ""

# ----- 1. Train + evaluate (tree-shift auto-downloads/exports COCO on first run) -----
export NCCL_IB_DISABLE=1
export NCCL_SOCKET_IFNAME=^lo,docker0
NUM_GPUS=2
apptainer exec --nv \
  --bind "${REPO_ROOT}:/workspace" \
  --env "NCCL_IB_DISABLE=${NCCL_IB_DISABLE}" \
  --env "NCCL_SOCKET_IFNAME=${NCCL_SOCKET_IFNAME}" \
  "${IMAGE_SIF}" \
  torchrun --standalone --nproc_per_node=${NUM_GPUS} plain_detr.py run \
  --coco-dir "${COCO_DIR}" \
  --config "${CONFIG}" \
  --train --eval-val --eval-ood \
  --output-dir "${OUTPUT_DIR}" \
  --batch-size 2 \
  --decoder_use_checkpoint \
  --epochs 50 \
  --backbone dinov3 \
  --dinov3_repo /opt/dinov3 \
  --dinov3_model dinov3_vits16 \
  --dinov3_weights "${DINO_WEIGHTS}" \
  --layers_to_use 3 6 9 11 \
  --num_feature_levels 1 \
  --proposal_feature_levels 1 \
  --proposal_in_stride 16 \
  --proposal_tgt_strides 16 \
  --add_transformer_encoder \
  --num_encoder_layers 6 \
  --norm_type pre_norm \
  --hidden_dim 256 \
  --dim_feedforward 2048 \
  --nheads 8 \
  --decoder_type global_rpe_decomp \
  --decoder_rpe_type linear \
  --decoder_rpe_hidden_dim 256 \
  --two_stage \
  --mixed_selection \
  --with_box_refine \
  --num_queries_one2one 100 \
  --num_queries_one2many 0 \
  --k_one2many 0 \
  --n_windows_sqrt 3

RUN_ERR=$?
if [[ $RUN_ERR -ne 0 ]]; then
  echo "plain_detr.py failed with exit code $RUN_ERR"
  exit $RUN_ERR
fi

# ----- 2. Shift analysis (univariate + Shapley) -----
ID_RESULTS="${OUTPUT_DIR}/eval_val/per_image_results.json"
OOD_RESULTS="${OUTPUT_DIR}/eval_ood_test/per_image_results.json"
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
echo "Done. Outputs: ${OUTPUT_DIR}"
