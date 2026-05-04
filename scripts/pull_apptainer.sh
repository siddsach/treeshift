#!/bin/bash
# Pull tree-shift Apptainer image from Docker Hub into the repo.
# Run from project root on Sherlock: ./scripts/pull_apptainer.sh
#
# The image will be saved as tree-shift.sif in the repo root.

set -e

REPO_ROOT="$(cd "$(dirname "$0")/.." && pwd)"
cd "$REPO_ROOT"

IMAGE_SIF="${REPO_ROOT}/tree-shift.sif"
DOCKER_URI="docker://aadityabuilds/tree-shift:latest"

echo "=== Pulling tree-shift Apptainer image ==="
echo "  Source: ${DOCKER_URI}"
echo "  Output: ${IMAGE_SIF}"
echo ""
echo "This may take 5–15 minutes depending on network..."
echo ""

apptainer pull "${IMAGE_SIF}" "${DOCKER_URI}"

echo ""
echo "Done. Image: ${IMAGE_SIF}"
echo ""
echo "Test with: ./scripts/test_container.sh"
echo "  Or interactively: apptainer exec --nv -B \$(pwd):/workspace -w /workspace tree-shift.sif bash"
