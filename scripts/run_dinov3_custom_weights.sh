#!/bin/bash
# ============================================================================
# Interactive runner for Plain-DETR + custom DINOv3 weights.
#
# Step 1 (MUST run first — no GPU needed, fast):
#   bash scripts/run_dinov3_custom_weights.sh probe
#
#   This identifies the correct --dinov3_model for each weight file.
#
# Step 2 (submit SLURM jobs after confirming model names):
#   DINOV3_MODEL_7B16=<from probe output>  \
#   DINOV3_MODEL_SAT=<from probe output>   \
#   bash scripts/run_dinov3_custom_weights.sh submit [CONFIG]
#
#   CONFIG defaults to biome_Rajasthan_train_WET__ood_DRY (single test run).
#   For all 40 configs: scripts/run_dinov3_custom_weights.sh submit all
# ============================================================================

set -euo pipefail

REPO_ROOT="/scratch/groups/dlobell/aadityan/tree-distribution-shift"
IMAGE_SIF="${REPO_ROOT}/tree-shift.sif"
DINO_WEIGHTS_DIR="${REPO_ROOT}/../dino_weights"
WEIGHTS_7B16="${DINO_WEIGHTS_DIR}/dino_weights_7b16.pth"
WEIGHTS_SAT="${DINO_WEIGHTS_DIR}/dino_weights_sat.pth"
WEIGHTS_VITS="${DINO_WEIGHTS_DIR}/dinov3_vits.pth"

COMMAND="${1:-probe}"

# ============================================================================
# PROBE
# ============================================================================
if [[ "${COMMAND}" == "probe" ]]; then
  echo "=== Probing DINOv3 weight files ==="
  echo ""

  if [[ ! -f "${IMAGE_SIF}" ]]; then
    echo "ERROR: Apptainer image not found: ${IMAGE_SIF}"
    exit 1
  fi

  apptainer exec --nv \
    --bind "${REPO_ROOT}:/workspace" \
    "${IMAGE_SIF}" \
    python /workspace/probe_weights.py \
      "${WEIGHTS_7B16}" \
      "${WEIGHTS_SAT}"  \
      "${WEIGHTS_VITS}"

  echo ""
  echo "=== Next steps ==="
  echo "  Look for the '✓ Recommended --dinov3_model' line above for each file."
  echo "  Then set DINOV3_MODEL_7B16 and DINOV3_MODEL_SAT and re-run:"
  echo ""
  echo "  DINOV3_MODEL_7B16=dinov3_vits16 \\"
  echo "  DINOV3_MODEL_SAT=dinov3_vits16 \\"
  echo "  bash scripts/run_dinov3_custom_weights.sh submit biome_Rajasthan_train_WET__ood_DRY"
  echo ""
  echo "  Or to submit all 40 configs:"
  echo "  DINOV3_MODEL_7B16=... DINOV3_MODEL_SAT=... bash scripts/run_dinov3_custom_weights.sh submit all"
  exit 0
fi

# ============================================================================
# SUBMIT
# ============================================================================
if [[ "${COMMAND}" == "submit" ]]; then
  # Model names (must be set by caller or env)
  DINOV3_MODEL_7B16="${DINOV3_MODEL_7B16:-}"
  DINOV3_MODEL_SAT="${DINOV3_MODEL_SAT:-}"

  if [[ -z "${DINOV3_MODEL_7B16}" ]] || [[ -z "${DINOV3_MODEL_SAT}" ]]; then
    echo "ERROR: Set DINOV3_MODEL_7B16 and DINOV3_MODEL_SAT before submitting."
    echo "  Run: bash scripts/run_dinov3_custom_weights.sh probe"
    exit 1
  fi

  CONFIG_ARG="${2:-biome_Rajasthan_train_WET__ood_DRY}"

  if [[ "${CONFIG_ARG}" == "all" ]]; then
    # Submit all 40 configs
    for script in "${REPO_ROOT}/scripts/plaindetr_dinov3_7b16_"*.sh \
                  "${REPO_ROOT}/scripts/plaindetr_dinov3_sat_"*.sh; do
      if [[ -f "${script}" ]]; then
        DINOV3_MODEL_7B16="${DINOV3_MODEL_7B16}" \
        DINOV3_MODEL_SAT="${DINOV3_MODEL_SAT}" \
        sbatch "${script}"
      fi
    done
  else
    # Single config
    SHORT=$(echo "${CONFIG_ARG}" \
      | sed 's/biome_Rajasthan_train_/biome_Rajasthan_/' \
      | sed 's/elev_Karnataka_train_/elev_Karnataka_/' \
      | sed 's/intl_train_/intl_/' \
      | sed 's/region_train_/region_/' \
      | sed 's/__ood_/_/' \
      | sed 's/__/_/')

    SCRIPT_7B16="${REPO_ROOT}/scripts/plaindetr_dinov3_7b16_${SHORT}.sh"
    SCRIPT_SAT="${REPO_ROOT}/scripts/plaindetr_dinov3_sat_${SHORT}.sh"

    for script in "${SCRIPT_7B16}" "${SCRIPT_SAT}"; do
      if [[ -f "${script}" ]]; then
        echo "Submitting: ${script}"
        DINOV3_MODEL_7B16="${DINOV3_MODEL_7B16}" \
        DINOV3_MODEL_SAT="${DINOV3_MODEL_SAT}" \
        sbatch "${script}"
      else
        echo "WARNING: Script not found: ${script}"
      fi
    done
  fi

  exit 0
fi

echo "Unknown command: ${COMMAND}"
echo "Usage: bash scripts/run_dinov3_custom_weights.sh [probe|submit] [CONFIG|all]"
exit 1
