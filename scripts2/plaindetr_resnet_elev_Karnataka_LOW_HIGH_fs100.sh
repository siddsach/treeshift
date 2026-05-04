#!/bin/bash
#SBATCH --job-name=pd_rn_elev_Karnataka_LOW_HIGH_fs100
#SBATCH --partition=serc,gpu
#SBATCH --gres=gpu:1
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=24:00:00
#SBATCH --output=logs/plaindetr_resnet_elev_Karnataka_LOW_HIGH_fs100_%j.out
#SBATCH --error=logs/plaindetr_resnet_elev_Karnataka_LOW_HIGH_fs100_%j.err

# -----------------------------------------------------------------------
# Plain-DETR + ResNet-50 FPN backbone: elev_Karnataka_train_LOW__ood_HIGH__fs100
# 2917 train images, batch 2/GPU × 1 A100 GPU, 40 epochs max.
# train + eval ID + eval OOD + SHIFT.
# -----------------------------------------------------------------------

set -euo pipefail

REPO_ROOT="/scratch/groups/dlobell/aadityan/tree-distribution-shift"
IMAGE_SIF="${REPO_ROOT}/tree-shift.sif"
METADATA_CSV="/scratch/groups/dlobell/aadityan/tree-distribution-shift/../dataset/metadata.csv"

CONFIG=elev_Karnataka_train_LOW__ood_HIGH__fs100
OUTPUT_DIR="${REPO_ROOT}/outputs/plaindetr_resnet_elev_Karnataka_LOW_HIGH_fs100_1gpu40ep_${SLURM_JOB_ID}"

cd "${REPO_ROOT}"
mkdir -p logs outputs

if [[ ! -f "${IMAGE_SIF}" ]]; then
  echo "ERROR: Apptainer image not found: ${IMAGE_SIF}"
  echo "Run: ./scripts/pull_apptainer.sh"
  exit 1
fi

echo "=== Plain-DETR ResNet50: elev_Karnataka_train_LOW__ood_HIGH__fs100 ==="
echo "Container : ${IMAGE_SIF}"
echo "Config    : ${CONFIG}"
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
