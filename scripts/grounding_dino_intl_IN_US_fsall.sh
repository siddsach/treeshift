#!/bin/bash
#SBATCH --job-name=gdino_intl_IN_US_fsall
#SBATCH --partition=serc
#SBATCH --gres=gpu:3
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=24
#SBATCH --mem=192G
#SBATCH --time=18:00:00
#SBATCH --output=logs/grounding_dino_intl_IN_US_fsall_%j.out
#SBATCH --error=logs/grounding_dino_intl_IN_US_fsall_%j.err

# -----------------------------------------------------------------------
# MM-Grounding-DINO: intl_train_IN__ood_US__fsall
# 19083 train images, batch 2/GPU × 3 GPUs, 50 epochs max.
# Early stopping: patience 5 epochs. train + eval ID + eval OOD + SHIFT.
# -----------------------------------------------------------------------

set -euo pipefail

REPO_ROOT="/scratch/groups/dlobell/siddsach/treeshift"
IMAGE_SIF="${TREE_SHIFT_SIF:-/scratch/groups/dlobell/aadityan/tree-distribution-shift/tree-shift.sif}"
METADATA_CSV="${METADATA_CSV:-/scratch/groups/dlobell/aadityan/dataset/metadata.csv}"
HF_CACHE_HOST="${REPO_ROOT}/.hf_cache/huggingface"
HF_HOME="/workspace/.hf_cache/huggingface"
PRETRAINED_WEIGHTS_REL="model_weights/grounding_dino_swin-t_pretrain_obj365_goldg_grit9m_v3det.pth"
PRETRAINED_WEIGHTS_HOST="${REPO_ROOT}/${PRETRAINED_WEIGHTS_REL}"
PRETRAINED_WEIGHTS_URL="https://download.openmmlab.com/mmdetection/v3.0/mm_grounding_dino/grounding_dino_swin-t_pretrain_obj365_goldg_grit9m_v3det/grounding_dino_swin-t_pretrain_obj365_goldg_grit9m_v3det_20231204_095047-b448804b.pth"

CONFIG=intl_train_IN__ood_US__fsall
OUTPUT_DIR="${REPO_ROOT}/outputs/grounding_dino_intl_IN_US_fsall_${SLURM_JOB_ID}"
NUM_GPUS=3

cd "${REPO_ROOT}"
mkdir -p logs outputs
mkdir -p "${HF_CACHE_HOST}"

if [[ ! -f "${IMAGE_SIF}" ]]; then
  echo "ERROR: Apptainer image not found: ${IMAGE_SIF}"
  echo "Run: ./scripts/pull_apptainer.sh"
  exit 1
fi

if [[ ! -f "${PRETRAINED_WEIGHTS_HOST}" ]]; then
  echo "Pretrained MM-Grounding-DINO weights not found. Downloading once ..."
  mkdir -p "$(dirname "${PRETRAINED_WEIGHTS_HOST}")"
  wget -O "${PRETRAINED_WEIGHTS_HOST}" "${PRETRAINED_WEIGHTS_URL}"
fi

echo "=== GroundingDINO: intl_train_IN__ood_US__fsall ==="
echo "Container : ${IMAGE_SIF}"
echo "Config    : ${CONFIG}"
echo "Output    : ${OUTPUT_DIR}"
echo ""

# Detect PyQt5 lib path (required by mmcv on headless nodes)
PYQT5_LIB=$(apptainer exec "${IMAGE_SIF}" python -c "
import os
try:
    import PyQt5
    for sub in ('Qt5', 'Qt'):
        p = os.path.join(os.path.dirname(PyQt5.__file__), sub, 'lib')
        if os.path.isdir(p):
            print(p)
            break
except Exception:
    pass
" 2>/dev/null | head -1)
: "${PYQT5_LIB:=/usr/local/lib/python3.9/dist-packages/PyQt5/Qt5/lib}"

export QT_QPA_PLATFORM=offscreen
export NCCL_IB_DISABLE=1
export NCCL_SOCKET_IFNAME=^lo,docker0
MASTER_PORT=$(( 29400 + SLURM_JOB_ID % 10000 ))

# ----- 1. Train (multi-GPU via torchrun) -----
apptainer exec --nv \
  --bind "${REPO_ROOT}:/workspace" \
  --env QT_QPA_PLATFORM=offscreen \
  --env "HF_HOME=${HF_HOME}" \
  --env "NCCL_IB_DISABLE=${NCCL_IB_DISABLE}" \
  --env "NCCL_SOCKET_IFNAME=${NCCL_SOCKET_IFNAME}" \
  --env "LD_LIBRARY_PATH=${PYQT5_LIB}:${LD_LIBRARY_PATH:-}" \
  "${IMAGE_SIF}" \
  torchrun --standalone --nproc_per_node=${NUM_GPUS} --master_port=${MASTER_PORT} grounding_dino.py run \
  --config "${CONFIG}" \
  --train \
  --output-dir "${OUTPUT_DIR}" \
  --batch-size 2 \
  --epochs 50 \
  --lr 0.00015 \
  --num-workers 4 \
  --early-stopping-patience 5 \
  --pretrained-weights "/workspace/${PRETRAINED_WEIGHTS_REL}"

TRAIN_ERR=$?
if [[ $TRAIN_ERR -ne 0 ]]; then
  echo "Training failed with exit code $TRAIN_ERR"
  exit $TRAIN_ERR
fi

# Find best checkpoint
CKPT=$(ls -t "${OUTPUT_DIR}"/best_coco_bbox_mAP*.pth 2>/dev/null | head -1)
if [[ -z "${CKPT}" ]]; then
  CKPT=$(ls -t "${OUTPUT_DIR}"/epoch_*.pth 2>/dev/null | head -1)
fi
if [[ -z "${CKPT}" ]]; then
  echo "ERROR: No checkpoint found in ${OUTPUT_DIR}"
  exit 1
fi
echo "Best checkpoint: ${CKPT}"

# ----- 2. Evaluate (single GPU, rank-safe) -----
apptainer exec --nv \
  --bind "${REPO_ROOT}:/workspace" \
  --env QT_QPA_PLATFORM=offscreen \
  --env "HF_HOME=${HF_HOME}" \
  --env "LD_LIBRARY_PATH=${PYQT5_LIB}:${LD_LIBRARY_PATH:-}" \
  "${IMAGE_SIF}" \
  python grounding_dino.py run \
  --config "${CONFIG}" \
  --eval-val --eval-ood \
  --model-path "${CKPT}" \
  --output-dir "${OUTPUT_DIR}" \
  --num-workers 4

EVAL_ERR=$?
if [[ $EVAL_ERR -ne 0 ]]; then
  echo "Evaluation failed with exit code $EVAL_ERR"
  exit $EVAL_ERR
fi

# ----- 2. Shift analysis (univariate + Shapley) -----
# fsall config: ood_train is empty (all OOD moved to train).
# Using eval_ood_test for shift analysis instead.
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
