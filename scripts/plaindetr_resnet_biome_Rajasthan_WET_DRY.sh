#!/bin/bash
#SBATCH --job-name=pd_rn_biome_Rajasthan_WET_DRY
#SBATCH --partition=serc
#SBATCH --gres=gpu:3
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=24
#SBATCH --mem=192G
#SBATCH --time=24:00:00
#SBATCH --output=logs/plaindetr_resnet_biome_Rajasthan_WET_DRY_%j.out
#SBATCH --error=logs/plaindetr_resnet_biome_Rajasthan_WET_DRY_%j.err

# -----------------------------------------------------------------------
# Plain-DETR + ResNet-50 FPN backbone: biome_Rajasthan_train_WET__ood_DRY
# 1714 train images, batch 2/GPU × 3 A100 GPUs, 50 epochs max.
# Early stopping: patience 5 epochs. train + eval ID + eval OOD + SHIFT.
# -----------------------------------------------------------------------

set -euo pipefail

REPO_ROOT="/scratch/groups/dlobell/siddsach/treeshift"
IMAGE_SIF="${TREE_SHIFT_SIF:-/scratch/groups/dlobell/aadityan/tree-distribution-shift/tree-shift.sif}"
METADATA_CSV="${METADATA_CSV:-/scratch/groups/dlobell/aadityan/dataset/metadata.csv}"

CONFIG=biome_Rajasthan_train_WET__ood_DRY
OUTPUT_DIR="${REPO_ROOT}/outputs/plaindetr_resnet_biome_Rajasthan_WET_DRY_${SLURM_JOB_ID}"

cd "${REPO_ROOT}"
mkdir -p logs outputs

if [[ ! -f "${IMAGE_SIF}" ]]; then
  echo "ERROR: Apptainer image not found: ${IMAGE_SIF}"
  echo "Run: ./scripts/pull_apptainer.sh"
  exit 1
fi

echo "=== Plain-DETR ResNet50: biome_Rajasthan_train_WET__ood_DRY ==="
echo "Container : ${IMAGE_SIF}"
echo "Config    : ${CONFIG}"
echo "Output    : ${OUTPUT_DIR}"
echo ""

# ----- 1. Train + evaluate -----
export NCCL_IB_DISABLE=1
export NCCL_SOCKET_IFNAME=^lo,docker0
NUM_GPUS=3
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
  --epochs 50 \
  --early_stop \
  --early_stop_patience 5 \
  --early_stop_min_epochs 10 \
  --early_stop_min_delta 0.0 \
  --backbone resnet50 \
  --num_feature_levels 1 \
  --proposal_feature_levels 3 \
  --proposal_in_stride 32 \
  --proposal_tgt_strides 8 16 32 \
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
  --k_one2many 0

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
