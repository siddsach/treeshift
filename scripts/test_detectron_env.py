#!/usr/bin/env python3
"""
Smoke-test for the .venv used by detectron_fastrcnn.py and detectron_maskrcnn.py.

Run from repo root (with .venv activated):
    python scripts/test_detectron_env.py
"""

import sys
import os

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
PASS = "\033[32mPASS\033[0m"
FAIL = "\033[31mFAIL\033[0m"
failures = []


def check(label, fn):
    try:
        result = fn()
        msg = f"  {result}" if result else ""
        print(f"[{PASS}] {label}{msg}")
    except Exception as e:
        print(f"[{FAIL}] {label}: {e}")
        failures.append(label)


# ── 1. Python version ────────────────────────────────────────────────────────
check(
    "Python >= 3.9",
    lambda: (
        None
        if sys.version_info >= (3, 9)
        else (_ for _ in ()).throw(RuntimeError(f"got {sys.version_info}"))
    ),
)

# ── 2. NumPy < 2 (detectron2 hard requirement) ───────────────────────────────
def _check_numpy():
    import numpy as np
    v = tuple(int(x) for x in np.__version__.split(".")[:2])
    if v >= (2, 0):
        raise RuntimeError(f"numpy {np.__version__} >= 2 — detectron2 will break")
    return f"numpy {np.__version__}"

check("numpy < 2", _check_numpy)

# ── 3. PyTorch + CUDA ────────────────────────────────────────────────────────
def _check_torch():
    import torch
    cuda_ok = torch.cuda.is_available()
    return f"torch {torch.__version__}, CUDA available: {cuda_ok}"

check("torch", _check_torch)

# ── 4. Core scientific stack ─────────────────────────────────────────────────
check("pandas",     lambda: __import__("pandas").__version__)
check("scipy.stats",lambda: __import__("scipy.stats", fromlist=["stats"]) and "ok")
check("matplotlib", lambda: __import__("matplotlib").__version__)
check("cv2",        lambda: __import__("cv2").__version__)
check("pycocotools",lambda: __import__("pycocotools") and "ok")
check("tqdm",       lambda: __import__("tqdm").__version__)

# ── 5. detectron2 imports (mirrors what the scripts do at startup) ────────────
def _check_detectron2():
    # Prepend local clone so configs are accessible
    d2_dir = os.path.join(REPO_ROOT, "detectron2")
    if os.path.isdir(d2_dir) and d2_dir not in sys.path:
        sys.path.insert(0, d2_dir)

    from detectron2 import model_zoo                                  # noqa
    from detectron2.data.datasets import register_coco_instances      # noqa
    from detectron2.utils.visualizer import Visualizer, ColorMode     # noqa
    from detectron2.data import DatasetCatalog, MetadataCatalog       # noqa
    from detectron2.engine import DefaultTrainer, DefaultPredictor, launch, HookBase  # noqa
    from detectron2.config import get_cfg                             # noqa
    from detectron2.evaluation import COCOEvaluator, inference_on_dataset  # noqa
    from detectron2.data import build_detection_test_loader           # noqa
    import detectron2.utils.comm                                      # noqa
    import detectron2
    return f"detectron2 {detectron2.__version__}"

check("detectron2 (all imports)", _check_detectron2)

# ── 6. Detectron2 config yaml files ──────────────────────────────────────────
def _check_faster_rcnn_yaml():
    p = os.path.join(REPO_ROOT, "detectron2", "configs",
                     "COCO-Detection", "faster_rcnn_R_50_FPN_3x.yaml")
    if not os.path.exists(p):
        raise FileNotFoundError(p)
    return "found"

def _check_mask_rcnn_yaml():
    p = os.path.join(REPO_ROOT, "detectron2", "configs",
                     "COCO-InstanceSegmentation", "mask_rcnn_R_50_FPN_3x.yaml")
    if not os.path.exists(p):
        raise FileNotFoundError(p)
    return "found"

check("detectron2 faster_rcnn_R_50_FPN_3x.yaml", _check_faster_rcnn_yaml)
check("detectron2 mask_rcnn_R_50_FPN_3x.yaml",   _check_mask_rcnn_yaml)

# ── 7. tree-distribution-shift package ───────────────────────────────────────
check("tree-distribution-shift", lambda: __import__("tree_shift") and "ok")

# ── 8. shared_utils (project file, not a pip package) ────────────────────────
def _check_shared_utils():
    sys.path.insert(0, REPO_ROOT)
    from shared_utils import ensure_tree_shift_export, resolve_coco_paths, evaluate_detections  # noqa
    return "ok"

check("shared_utils", _check_shared_utils)

# ── Summary ───────────────────────────────────────────────────────────────────
print()
if failures:
    print(f"FAILED ({len(failures)} checks): {', '.join(failures)}")
    sys.exit(1)
else:
    print("All checks passed — .venv is ready for fastrcnn/maskrcnn jobs.")
