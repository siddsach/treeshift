#!/bin/bash
#SBATCH --job-name=grounding_dino_South_North
#SBATCH --partition=serc
#SBATCH --gres=gpu:1
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --time=24:00:00
#SBATCH --output=logs/grounding_dino_South_North_%j.out
#SBATCH --error=logs/grounding_dino_South_North_%j.err

# -----------------------------------------------------------------------------
# MM-Grounding-DINO: train South India, test North India.
# Uses tree-shift Apptainer container. Training + eval + univariate + Shapley.
# -----------------------------------------------------------------------------

set -euo pipefail

REPO_ROOT="/scratch/groups/dlobell/aadityan/tree-distribution-shift"
IMAGE_SIF="${REPO_ROOT}/tree-shift.sif"
METADATA_CSV="/scratch/groups/dlobell/aadityan/dataset/metadata.csv"
HF_CACHE_HOST="${REPO_ROOT}/.hf_cache/huggingface"
HF_HOME="/workspace/.hf_cache/huggingface"
PRETRAINED_WEIGHTS_REL="model_weights/grounding_dino_swin-t_pretrain_obj365_goldg_grit9m_v3det.pth"
PRETRAINED_WEIGHTS_HOST="${REPO_ROOT}/${PRETRAINED_WEIGHTS_REL}"
PRETRAINED_WEIGHTS_URL="https://download.openmmlab.com/mmdetection/v3.0/mm_grounding_dino/grounding_dino_swin-t_pretrain_obj365_goldg_grit9m_v3det/grounding_dino_swin-t_pretrain_obj365_goldg_grit9m_v3det_20231204_095047-b448804b.pth"

CONFIG=region_train_South__ood_North
OUTPUT_DIR="${REPO_ROOT}/outputs/grounding_dino_national_South_North_${SLURM_JOB_ID}"
COCO_DIR="${OUTPUT_DIR}/coco_export"

cd "${REPO_ROOT}"
mkdir -p logs outputs
mkdir -p "${HF_CACHE_HOST}"

if [[ ! -f "${IMAGE_SIF}" ]]; then
  echo "ERROR: Image not found: ${IMAGE_SIF}"
  echo "Run: ./scripts/pull_apptainer.sh"
  exit 1
fi

if [[ ! -f "${PRETRAINED_WEIGHTS_HOST}" ]]; then
  echo "Pretrained MM-Grounding-DINO weights not found. Downloading once ..."
  mkdir -p "$(dirname "${PRETRAINED_WEIGHTS_HOST}")"
  wget -O "${PRETRAINED_WEIGHTS_HOST}" "${PRETRAINED_WEIGHTS_URL}"
fi

echo "=== GroundingDINO: train South India, OOD North India ==="
echo "Container: ${IMAGE_SIF}"
echo "Config: ${CONFIG}"
echo "COCO export: ${COCO_DIR} (auto-download via tree-shift)"
echo "Output: ${OUTPUT_DIR}"
echo "HF_HOME: ${HF_HOME}"
echo "HF cache host dir: ${HF_CACHE_HOST}"
echo "Pretrained weights: ${PRETRAINED_WEIGHTS_HOST}"
echo ""

# PyQt5's Qt5 libs must be in LD_LIBRARY_PATH for GroundingDINO/mmcv (libQt5Core-*.so.5.15)
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
echo "PyQt5 LD_LIBRARY_PATH: ${PYQT5_LIB}"

# ----- 1. Train + evaluate (tree-shift auto-downloads/exports COCO on first run) -----
# QT_QPA_PLATFORM=offscreen avoids Qt display issues on headless HPC nodes
export QT_QPA_PLATFORM=offscreen
apptainer exec --nv \
  --bind "${REPO_ROOT}:/workspace" \
  --env QT_QPA_PLATFORM=offscreen \
  --env "HF_HOME=${HF_HOME}" \
  --env "LD_LIBRARY_PATH=${PYQT5_LIB}:${LD_LIBRARY_PATH:-}" \
  "${IMAGE_SIF}" \
  python grounding_dino.py run \
  --coco-dir "${COCO_DIR}" \
  --config "${CONFIG}" \
  --train --eval-val --eval-ood \
  --output-dir "${OUTPUT_DIR}" \
  --batch-size 2 \
  --epochs 30 \
  --lr 0.00005 \
  --num-workers 4 \
  --pretrained-weights "/workspace/${PRETRAINED_WEIGHTS_REL}"

RUN_ERR=$?
if [[ $RUN_ERR -ne 0 ]]; then
  echo "grounding_dino.py failed with exit code $RUN_ERR"
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
