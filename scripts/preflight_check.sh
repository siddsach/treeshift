#!/bin/bash
#SBATCH --job-name=preflight
#SBATCH --partition=serc
#SBATCH --gres=gpu:3
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=1:00:00
#SBATCH --output=logs/preflight_%j.out
#SBATCH --error=logs/preflight_%j.err

# ---------------------------------------------------------------------------
# Preflight check for tree-distribution-shift.
#
# Runs test_preflight.py inside tree-shift.sif (covers Plain-DETR / GDINO).
# Then runs the Detectron2-specific check inside detectron.sif.
#
# Usage (interactive):
#   bash scripts/preflight_check.sh
#   bash scripts/preflight_check.sh --skip-download   # no HF network
#   bash scripts/preflight_check.sh --skip-models     # faster, skip model build
#
# Usage (SLURM):
#   sbatch scripts/preflight_check.sh
# ---------------------------------------------------------------------------

set -euo pipefail

REPO_ROOT="/scratch/groups/dlobell/aadityan/tree-distribution-shift"
TREE_SIF="${REPO_ROOT}/tree-shift.sif"
DETECTRON_SIF="${REPO_ROOT}/detectron.sif"
PREFLIGHT_PY="${REPO_ROOT}/test_preflight.py"

# Pass through any extra flags (--skip-download, --skip-models, etc.)
EXTRA_FLAGS="${*}"

cd "${REPO_ROOT}"
mkdir -p logs

echo "================================================================"
echo "  tree-distribution-shift PREFLIGHT CHECK"
echo "  $(date)"
echo "  REPO_ROOT : ${REPO_ROOT}"
echo "  extra flags: ${EXTRA_FLAGS:-<none>}"
echo "================================================================"

# ── Sanity: containers must exist ────────────────────────────────────
for sif in "${TREE_SIF}" "${DETECTRON_SIF}"; do
  if [[ ! -f "${sif}" ]]; then
    echo ""
    echo "ERROR: Apptainer image not found: ${sif}"
    echo "Build it first (see scripts/pull_apptainer.sh / detectron.def)"
    exit 1
  fi
done

echo ""
echo "Both containers found:"
echo "  tree-shift.sif  : $(stat -c '%s' "${TREE_SIF}") bytes"
echo "  detectron.sif   : $(stat -c '%s' "${DETECTRON_SIF}") bytes"

# ════════════════════════════════════════════════════════════════════
# PART 1 — tree-shift.sif (Plain-DETR, Grounding-DINO, pycocotools,
#           tree_shift package, shared_utils, shift_analysis, etc.)
# ════════════════════════════════════════════════════════════════════
echo ""
echo "════════════════════════════════════════════════════════════════"
echo "  PART 1 — tree-shift.sif"
echo "════════════════════════════════════════════════════════════════"

apptainer exec --nv \
  --bind "${REPO_ROOT}:/workspace" \
  "${TREE_SIF}" \
  python /workspace/test_preflight.py \
    --repo-root /workspace \
    ${EXTRA_FLAGS}

PART1_ERR=$?

# ════════════════════════════════════════════════════════════════════
# PART 2 — detectron.sif (Detectron2, fastrcnn/maskrcnn imports)
# ════════════════════════════════════════════════════════════════════
echo ""
echo "════════════════════════════════════════════════════════════════"
echo "  PART 2 — detectron.sif (Detectron2 imports)"
echo "════════════════════════════════════════════════════════════════"

apptainer exec --nv \
  --bind "${REPO_ROOT}:/workspace" \
  "${DETECTRON_SIF}" \
  python /workspace/test_preflight.py \
    --repo-root /workspace \
    --detectron

PART2_ERR=$?

# ════════════════════════════════════════════════════════════════════
# Summary
# ════════════════════════════════════════════════════════════════════
echo ""
echo "================================================================"
echo "  PREFLIGHT SHELL SUMMARY"
echo "  Part 1 (tree-shift.sif)  : $([ $PART1_ERR -eq 0 ] && echo PASS || echo FAIL)"
echo "  Part 2 (detectron.sif)   : $([ $PART2_ERR -eq 0 ] && echo PASS || echo FAIL)"
echo "================================================================"

if [[ $PART1_ERR -ne 0 ]] || [[ $PART2_ERR -ne 0 ]]; then
  echo ""
  echo "One or more checks failed. Resolve errors before submitting training jobs."
  exit 1
fi

echo ""
echo "All preflight checks passed. Safe to submit training scripts."
echo ""
