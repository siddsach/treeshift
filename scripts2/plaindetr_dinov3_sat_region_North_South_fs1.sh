#!/bin/bash
#SBATCH --job-name=pd_dv3_sat_region_North_South_fs1
#SBATCH --partition=serc,gpu
#SBATCH --gres=gpu:1
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=36:00:00
#SBATCH --output=logs/plaindetr_dinov3_sat_region_North_South_fs1_%j.out
#SBATCH --error=logs/plaindetr_dinov3_sat_region_North_South_fs1_%j.err

# -----------------------------------------------------------------------
# Plain-DETR + DinoV3 (dino_weights_sat): region_train_North__ood_South__fs1
# 6119 train images, batch 2/GPU × 1 A100 GPU, 40 epochs max.
# train + eval ID + eval OOD + SHIFT.
# -----------------------------------------------------------------------

set -euo pipefail

REPO_ROOT="/scratch/groups/dlobell/siddsach/treeshift"
IMAGE_SIF="${TREE_SHIFT_SIF:-/scratch/groups/dlobell/aadityan/tree-distribution-shift/tree-shift.sif}"
METADATA_CSV="${METADATA_CSV:-/scratch/groups/dlobell/aadityan/dataset/metadata.csv}"
DINO_WEIGHTS_ROOT="${DINO_WEIGHTS_ROOT:-/scratch/groups/dlobell/aadityan/dino_weights}"
DINOV3_REPO="${DINOV3_REPO:-/opt/dinov3}"
DINO_WEIGHTS="${DINO_WEIGHTS:-${DINO_WEIGHTS_ROOT}/dino_weights_sat.pth}"

CONFIG=region_train_North__ood_South__fs1
OUTPUT_DIR="${REPO_ROOT}/outputs/plaindetr_dinov3_sat_region_North_South_fs1_1gpu40ep_${SLURM_JOB_ID}"

cd "${REPO_ROOT}"
mkdir -p logs outputs

if [[ ! -f "${IMAGE_SIF}" ]]; then
  echo "ERROR: Apptainer image not found: ${IMAGE_SIF}"
  echo "Run: ./scripts/pull_apptainer.sh"
  exit 1
fi

if [[ ! -f "${DINO_WEIGHTS}" ]]; then
  echo "ERROR: DinoV3 sat weights not found: ${DINO_WEIGHTS}"
  exit 1
fi

echo "=== Plain-DETR DinoV3 (sat): region_train_North__ood_South__fs1 ==="
echo "Container : ${IMAGE_SIF}"
echo "Config    : ${CONFIG}"
echo "Weights   : ${DINO_WEIGHTS}"
echo "Output    : ${OUTPUT_DIR}"
echo ""

# ----- 1. Train + evaluate -----
export NCCL_IB_DISABLE=1
export NCCL_SOCKET_IFNAME=^lo,docker0
NUM_GPUS=1
MASTER_PORT=$(( 29400 + SLURM_JOB_ID % 10000 ))
apptainer exec --nv \
  --bind "${REPO_ROOT}:/workspace" \
  --env "NCCL_IB_DISABLE=${NCCL_IB_DISABLE}" \
  --env "NCCL_SOCKET_IFNAME=${NCCL_SOCKET_IFNAME}" \
  "${IMAGE_SIF}" \
  torchrun --standalone --nproc_per_node=${NUM_GPUS} --master_port=${MASTER_PORT} plain_detr.py run \
  --config "${CONFIG}" \
  --train --eval-val --eval-ood --eval-ood-train \
  --output-dir "${OUTPUT_DIR}" \
  --batch-size 2 \
  --decoder_use_checkpoint \
  --epochs 40 \
  --backbone dinov3 \
  --dinov3_repo "${DINOV3_REPO}" \
  --dinov3_model dinov3_vit7b16 \
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

# ----- 2. Post-hoc shift analysis -----
# Shift analysis is intentionally not run inside training jobs.
# Use saved per-image eval outputs for post-hoc analysis after jobs finish.
echo ""
echo "Done. Outputs: ${OUTPUT_DIR}"
echo "Post-hoc shift inputs, when available:"
echo "  ${OUTPUT_DIR}/eval_val/per_image_results.json"
echo "  ${OUTPUT_DIR}/eval_ood_train/per_image_results.json"
