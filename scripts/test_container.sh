#!/bin/bash
#SBATCH --job-name=tree-shift-test
#SBATCH --partition=serc
#SBATCH --gres=gpu:1
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=16G
#SBATCH --time=0:30:00
#SBATCH --output=logs/test_container_%j.out
#SBATCH --error=logs/test_container_%j.err

# Test the tree-shift Apptainer container: imports, CUDA, and minimal training sanity check.
# Run from project root: sbatch scripts/test_container.sh
# Or run interactively: srun -p serc --gres gpu:1 -c 4 --mem 16G -t 0:30:00 --pty bash -c './scripts/test_container.sh'

set -euo pipefail

REPO_ROOT="/scratch/groups/dlobell/aadityan/tree-distribution-shift"
IMAGE_SIF="${REPO_ROOT}/tree-shift.sif"

cd "$REPO_ROOT"
mkdir -p logs

if [[ ! -f "${IMAGE_SIF}" ]]; then
  echo "ERROR: Image not found: ${IMAGE_SIF}"
  echo "Run: ./scripts/pull_apptainer.sh"
  exit 1
fi

echo "=== Testing tree-shift container ==="
echo "Image: ${IMAGE_SIF}"
echo ""

apptainer exec --nv \
  --bind "${REPO_ROOT}:/workspace" \
  "${IMAGE_SIF}" \
  python - <<'PYTEST'
"""Verify container can import deps and run models."""
import sys
import os

os.chdir("/workspace")

def ok(name):
    print(f"  [OK] {name}")

def fail(name, e):
    print(f"  [FAIL] {name}: {e}", file=sys.stderr)
    sys.exit(1)

# 1. PyTorch + CUDA
try:
    import torch
    ok("torch")
    cuda_ok = torch.cuda.is_available()
    if cuda_ok:
        ok(f"torch.cuda ({torch.cuda.get_device_name(0)})")
    else:
        print("  [WARN] CUDA not available (CPU-only run)")
except Exception as e:
    fail("torch", e)

# 2. GroundingDINO C extension
try:
    from groundingdino import _C
    ok("groundingdino._C")
except Exception as e:
    fail("groundingdino._C", e)

# 3. MMDetection
try:
    import mmdet
    from mmdet.apis import DetInferencer
    ok("mmdet")
except Exception as e:
    fail("mmdet", e)

# 4. tree_shift
try:
    import tree_shift
    ok("tree_shift")
except Exception as e:
    fail("tree_shift", e)

# 5. Plain-DETR imports (Plain-DETR internal)
try:
    import sys
    sys.path.insert(0, "/workspace/Plain-DETR")
    from models import build_model
    ok("Plain-DETR build_model")
except Exception as e:
    fail("Plain-DETR", e)

# 6. Minimal training sanity: tiny forward pass (no data)
if cuda_ok:
    try:
        x = torch.randn(1, 3, 224, 224, device="cuda")
        y = torch.nn.Conv2d(3, 64, 7)(x)
        ok("minimal CUDA forward pass")
    except Exception as e:
        fail("CUDA forward", e)

print("")
print("All checks passed. Container is ready for training.")
PYTEST

echo ""
echo "=== Container test complete ==="
