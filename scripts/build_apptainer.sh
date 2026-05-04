#!/bin/bash
# Build tree-shift Apptainer image for cluster (e.g. Sherlock).
# Requires: apptainer (or singularity), sudo, network access
# Run from project root: ./scripts/build_apptainer.sh

set -euo pipefail
cd "$(dirname "$0")/.."

IMAGE_NAME="${1:-tree-shift.sif}"

echo "=== Building Apptainer image: ${IMAGE_NAME} ==="
echo "Building from tree-shift.def (this may take 15-30 min)..."

# Use --fakeroot if available; otherwise require sudo and fail loudly if unavailable.
if apptainer build --help 2>/dev/null | rg -q fakeroot; then
    if ! apptainer build --fakeroot --force "${IMAGE_NAME}" tree-shift.def; then
        if command -v sudo >/dev/null 2>&1; then
            sudo apptainer build --force "${IMAGE_NAME}" tree-shift.def
        else
            echo "ERROR: apptainer --fakeroot failed and sudo is unavailable on this cluster."
            echo "Use Docker->Hub->apptainer pull workflow instead:"
            echo "  1) Build+push image locally"
            echo "  2) ./scripts/pull_apptainer.sh on cluster"
            exit 1
        fi
    fi
else
    if command -v sudo >/dev/null 2>&1; then
        sudo apptainer build --force "${IMAGE_NAME}" tree-shift.def
    else
        echo "ERROR: apptainer build requires sudo here, and sudo is unavailable."
        echo "Use Docker->Hub->apptainer pull workflow instead."
        exit 1
    fi
fi

if [[ ! -s "${IMAGE_NAME}" ]]; then
    echo "ERROR: build finished but image is missing or empty: ${IMAGE_NAME}"
    exit 1
fi

echo ""
echo "Done. Image: $(pwd)/${IMAGE_NAME}"
echo ""
echo "Usage on cluster (example with GPU):"
echo "  apptainer exec --nv ${IMAGE_NAME} python your_script.py"
echo "  apptainer shell --nv ${IMAGE_NAME}"
echo ""
echo "Or with bind mounts:"
echo "  apptainer exec --nv -B /path/to/data:/data ${IMAGE_NAME} python /workspace/script.py"
