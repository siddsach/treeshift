#!/usr/bin/env python3
"""
Generator for tree-distribution-shift v3.0 SBATCH training scripts.

Generates 200 scripts (40 configs × 5 models) in the scripts/ directory.
Run:  python scripts/generate_scripts.py
"""

import os
import math

# ---------------------------------------------------------------------------
# Config registry:  name → number of training images
# ---------------------------------------------------------------------------
CONFIGS = {
    # train counts from make_configs (v3.1 — ood split into ood_train 70% + ood_test 30%)
    "biome_Rajasthan_train_DRY__ood_WET":         1934,
    "biome_Rajasthan_train_DRY__ood_WET__fs1":    1935,
    "biome_Rajasthan_train_DRY__ood_WET__fs10":   1944,
    "biome_Rajasthan_train_DRY__ood_WET__fs100":  2034,
    "biome_Rajasthan_train_DRY__ood_WET__fsall":  3268,   # was 3839; now only 70% OOD
    "biome_Rajasthan_train_WET__ood_DRY":         1714,
    "biome_Rajasthan_train_WET__ood_DRY__fs1":    1715,
    "biome_Rajasthan_train_WET__ood_DRY__fs10":   1724,
    "biome_Rajasthan_train_WET__ood_DRY__fs100":  1814,
    "biome_Rajasthan_train_WET__ood_DRY__fsall":  3218,   # was 3863
    "elev_Karnataka_train_HIGH__ood_LOW":          2351,
    "elev_Karnataka_train_HIGH__ood_LOW__fs1":     2352,
    "elev_Karnataka_train_HIGH__ood_LOW__fs10":    2361,
    "elev_Karnataka_train_HIGH__ood_LOW__fs100":   2451,
    "elev_Karnataka_train_HIGH__ood_LOW__fsall":   4542,   # was 5481
    "elev_Karnataka_train_LOW__ood_HIGH":          2817,
    "elev_Karnataka_train_LOW__ood_HIGH__fs1":     2818,
    "elev_Karnataka_train_LOW__ood_HIGH__fs10":    2827,
    "elev_Karnataka_train_LOW__ood_HIGH__fs100":   2917,
    "elev_Karnataka_train_LOW__ood_HIGH__fsall":   4645,   # was 5429
    "intl_train_IN__ood_US":                      12143,
    "intl_train_IN__ood_US__fs1":                 12144,
    "intl_train_IN__ood_US__fs10":                12153,
    "intl_train_IN__ood_US__fs100":               12243,
    "intl_train_IN__ood_US__fsall":               19083,   # was 22057
    "intl_train_US__ood_IN":                       8923,
    "intl_train_US__ood_IN__fs1":                  8924,
    "intl_train_US__ood_IN__fs10":                 8933,
    "intl_train_US__ood_IN__fs100":                9023,
    "intl_train_US__ood_IN__fsall":               18367,   # was 22415
    "region_train_North__ood_South":               6118,
    "region_train_North__ood_South__fs1":          6119,
    "region_train_North__ood_South__fs10":         6128,
    "region_train_North__ood_South__fs100":        6218,
    "region_train_North__ood_South__fsall":       10804,   # was 12812
    "region_train_South__ood_North":               6025,
    "region_train_South__ood_North__fs1":          6026,
    "region_train_South__ood_North__fs10":         6035,
    "region_train_South__ood_North__fs100":        6125,
    "region_train_South__ood_North__fsall":       10784,   # was 12823
    "india_random_80_20":                          10794,
}

MODELS = ["fastrcnn", "maskrcnn", "fastrcnn_pretrained", "maskrcnn_pretrained",
          "plaindetr_dinov3", "plaindetr_dinov3_7b16", "plaindetr_dinov3_sat",
          "plaindetr_resnet", "grounding_dino"]

REPO_ROOT = "/scratch/groups/dlobell/siddsach/treeshift"

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def config_shortname(config: str) -> str:
    """Convert full config name to a short display name for filenames / job names."""
    s = config
    s = s.replace("biome_Rajasthan_train_", "biome_Rajasthan_")
    s = s.replace("elev_Karnataka_train_",  "elev_Karnataka_")
    s = s.replace("intl_train_",            "intl_")
    s = s.replace("region_train_",          "region_")
    s = s.replace("__ood_", "_")
    s = s.replace("__", "_")
    return s


def script_filename(model: str, config: str) -> str:
    short = config_shortname(config)
    return f"{model}_{short}.sh"


def job_name(model: str, config: str) -> str:
    short = config_shortname(config)
    abbr = {
        "fastrcnn":             "frcnn",
        "maskrcnn":             "mrcnn",
        "fastrcnn_pretrained":  "frcnn_pt",
        "maskrcnn_pretrained":  "mrcnn_pt",
        "plaindetr_dinov3":      "pd_dv3",
        "plaindetr_dinov3_7b16": "pd_dv3_7b16",
        "plaindetr_dinov3_sat":  "pd_dv3_sat",
        "plaindetr_resnet":      "pd_rn",
        "grounding_dino":        "gdino",
    }
    return f"{abbr[model]}_{short}"[:40]


def batch_size(model: str, n: int) -> int:
    if model in ("fastrcnn", "maskrcnn", "fastrcnn_pretrained", "maskrcnn_pretrained"):
        # 2 GPUs: doubled batch sizes (8→16, 10→20, 12→24); each divisible by 2
        if n < 4000:
            return 16
        elif n < 12000:
            return 20
        else:
            return 24
    elif model == "grounding_dino":
        return 2
    else:  # plain-detr (per GPU, 3 GPUs)
        return 2


def max_iterations(n: int, bs: int, epochs: int = 50) -> int:
    """Compute Detectron2 max iterations for `epochs` epochs."""
    return math.ceil(n / bs) * epochs


def time_limit(model: str, n: int) -> str:
    if model in ("plaindetr_dinov3", "plaindetr_dinov3_7b16", "plaindetr_dinov3_sat", "plaindetr_resnet"):
        if n < 5000:   return "24:00:00"
        elif n < 15000: return "36:00:00"
        else:           return "48:00:00"
    elif model == "grounding_dino":
        # 3-GPU: roughly 3× faster training + eval headroom
        if n < 4000:    return "8:00:00"
        elif n < 10000: return "12:00:00"
        elif n < 20000: return "18:00:00"
        else:           return "24:00:00"
    else:  # 1-GPU
        if n < 4000:    return "12:00:00"
        elif n < 10000: return "24:00:00"
        elif n < 20000: return "36:00:00"
        else:           return "48:00:00"


def memory(model: str, n: int) -> str:
    if model in ("plaindetr_dinov3", "plaindetr_dinov3_7b16", "plaindetr_dinov3_sat", "plaindetr_resnet"):
        return "192G"
    if model == "grounding_dino":
        # 3-GPU — generous
        if n < 5000:    return "96G"
        elif n < 15000: return "128G"
        else:           return "192G"
    # 2-GPU models (fastrcnn/maskrcnn) — scaled up from 1-GPU
    if n < 5000:    return "64G"
    elif n < 15000: return "96G"
    else:           return "128G"


# ---------------------------------------------------------------------------
# Script templates
# ---------------------------------------------------------------------------

SHIFT_ANALYSIS_BLOCK_VENV = """\
# ----- 2. Shift analysis (univariate + Shapley) -----
# Uses eval_val (ID test) and eval_ood_train (OOD train pool) for shift analysis.
# eval_ood_test is the fixed held-out set used only for the reported metric.
ID_RESULTS="${OUTPUT_DIR}/eval_val/per_image_results.json"
OOD_RESULTS="${OUTPUT_DIR}/eval_ood_train/per_image_results.json"
SHIFT_OUT="${OUTPUT_DIR}/shift_analysis"

if [ ! -f "${ID_RESULTS}" ] || [ ! -f "${OOD_RESULTS}" ]; then
  echo "Skipping shift analysis: missing eval outputs."
  exit 0
fi

if [ ! -f "${METADATA_CSV}" ]; then
  echo "Skipping shift analysis: metadata not found at ${METADATA_CSV}"
  exit 0
fi

echo "Running univariate + Shapley decomposition ..."
python shift_analysis.py univariate \\
  --id-results "${ID_RESULTS}" \\
  --ood-results "${OOD_RESULTS}" \\
  --metadata "${METADATA_CSV}" \\
  --output-dir "${SHIFT_OUT}"

python shift_analysis.py shapley \\
  --id-results "${ID_RESULTS}" \\
  --ood-results "${OOD_RESULTS}" \\
  --metadata "${METADATA_CSV}" \\
  --output-dir "${SHIFT_OUT}"

echo "Done. Outputs: ${OUTPUT_DIR}"
"""

SHIFT_ANALYSIS_BLOCK_APPTAINER = """\
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
apptainer exec --nv \\
  --bind "${REPO_ROOT}:/workspace" \\
  "${IMAGE_SIF}" \\
  python shift_analysis.py univariate \\
  --id-results "${ID_RESULTS}" \\
  --ood-results "${OOD_RESULTS}" \\
  --metadata "${METADATA_CSV}" \\
  --output-dir "${SHIFT_OUT}"

apptainer exec --nv \\
  --bind "${REPO_ROOT}:/workspace" \\
  "${IMAGE_SIF}" \\
  python shift_analysis.py shapley \\
  --id-results "${ID_RESULTS}" \\
  --ood-results "${OOD_RESULTS}" \\
  --metadata "${METADATA_CSV}" \\
  --output-dir "${SHIFT_OUT}"

echo ""
echo ""
echo "Done. Outputs: ${OUTPUT_DIR}"
"""

# fsall configs have an empty ood_train — use ood_test for shift analysis instead.
SHIFT_ANALYSIS_BLOCK_VENV_FSALL = """\
# ----- 2. Shift analysis (univariate + Shapley) -----
# fsall config: ood_train is empty (all OOD moved to train).
# Using eval_ood_test for shift analysis instead.
ID_RESULTS="${OUTPUT_DIR}/eval_val/per_image_results.json"
OOD_RESULTS="${OUTPUT_DIR}/eval_ood_test/per_image_results.json"
SHIFT_OUT="${OUTPUT_DIR}/shift_analysis"

if [ ! -f "${ID_RESULTS}" ] || [ ! -f "${OOD_RESULTS}" ]; then
  echo "Skipping shift analysis: missing eval outputs."
  exit 0
fi

if [ ! -f "${METADATA_CSV}" ]; then
  echo "Skipping shift analysis: metadata not found at ${METADATA_CSV}"
  exit 0
fi

echo "Running univariate + Shapley decomposition ..."
python shift_analysis.py univariate \\
  --id-results "${ID_RESULTS}" \\
  --ood-results "${OOD_RESULTS}" \\
  --metadata "${METADATA_CSV}" \\
  --output-dir "${SHIFT_OUT}"

python shift_analysis.py shapley \\
  --id-results "${ID_RESULTS}" \\
  --ood-results "${OOD_RESULTS}" \\
  --metadata "${METADATA_CSV}" \\
  --output-dir "${SHIFT_OUT}"

echo "Done. Outputs: ${OUTPUT_DIR}"
"""

SHIFT_ANALYSIS_BLOCK_APPTAINER_FSALL = """\
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
apptainer exec --nv \\
  --bind "${REPO_ROOT}:/workspace" \\
  "${IMAGE_SIF}" \\
  python shift_analysis.py univariate \\
  --id-results "${ID_RESULTS}" \\
  --ood-results "${OOD_RESULTS}" \\
  --metadata "${METADATA_CSV}" \\
  --output-dir "${SHIFT_OUT}"

apptainer exec --nv \\
  --bind "${REPO_ROOT}:/workspace" \\
  "${IMAGE_SIF}" \\
  python shift_analysis.py shapley \\
  --id-results "${ID_RESULTS}" \\
  --ood-results "${OOD_RESULTS}" \\
  --metadata "${METADATA_CSV}" \\
  --output-dir "${SHIFT_OUT}"

echo ""
echo "Done. Outputs: ${OUTPUT_DIR}"
"""


def _ood_train_flag(config: str) -> str:
    """Return ' --eval-ood-train' unless this is an fsall config (empty ood_train)."""
    return "" if config.endswith("__fsall") else " --eval-ood-train"


def _is_id_only_config(config: str) -> bool:
    return config == "india_random_80_20"


def _eval_flags(config: str) -> str:
    if _is_id_only_config(config):
        return " --eval-val"
    return f" --eval-val --eval-ood{_ood_train_flag(config)}"


ID_ONLY_DONE_BLOCK = """\
# ----- 2. ID-only baseline -----
# This config has no OOD split, so shift analysis is intentionally skipped.
echo ""
echo "Done. Outputs: ${OUTPUT_DIR}"
"""


def _shift_apptainer(config: str) -> str:
    if _is_id_only_config(config):
        return ID_ONLY_DONE_BLOCK
    return SHIFT_ANALYSIS_BLOCK_APPTAINER_FSALL if config.endswith("__fsall") \
        else SHIFT_ANALYSIS_BLOCK_APPTAINER


def _shift_venv(config: str) -> str:
    if _is_id_only_config(config):
        return ID_ONLY_DONE_BLOCK
    return SHIFT_ANALYSIS_BLOCK_VENV_FSALL if config.endswith("__fsall") \
        else SHIFT_ANALYSIS_BLOCK_VENV


def gen_fastrcnn(config: str, n: int) -> str:
    bs   = batch_size("fastrcnn", n)
    itr  = max_iterations(n, bs, 50)
    jname = job_name("fastrcnn", config)
    short = config_shortname(config)
    mem  = memory("fastrcnn", n)
    tl   = time_limit("fastrcnn", n)

    return f"""\
#!/bin/bash
#SBATCH --job-name={jname}
#SBATCH --partition=serc
#SBATCH --gres=gpu:2
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=16
#SBATCH --mem={mem}
#SBATCH --time={tl}
#SBATCH --output=logs/fastrcnn_{short}_%j.out
#SBATCH --error=logs/fastrcnn_{short}_%j.err

# -----------------------------------------------------------------------
# Faster R-CNN: {config}
# {n} train images, batch {bs} (2 GPUs), 50 epochs = {itr} iterations.
# Early stopping: patience 5 epochs. train + eval ID + eval OOD + SHIFT.
# -----------------------------------------------------------------------

set -euo pipefail

REPO_ROOT="{REPO_ROOT}"
IMAGE_SIF="${{DETECTRON_SIF:-/scratch/groups/dlobell/aadityan/tree-distribution-shift/detectron.sif}}"
METADATA_CSV="${{METADATA_CSV:-/scratch/groups/dlobell/aadityan/dataset/metadata.csv}}"

CONFIG={config}
OUTPUT_DIR="${{REPO_ROOT}}/outputs/fastrcnn_{short}_${{SLURM_JOB_ID}}"

cd "${{REPO_ROOT}}"
mkdir -p logs outputs

if [[ ! -f "${{IMAGE_SIF}}" ]]; then
  echo "ERROR: Apptainer image not found: ${{IMAGE_SIF}}"
  echo "Run: apptainer build --fakeroot detectron.sif detectron.def"
  exit 1
fi

# ----- 1. Train + evaluate on ID test, OOD test, and OOD train -----
apptainer exec --nv \\
  --bind "${{REPO_ROOT}}:/workspace" \\
  "${{IMAGE_SIF}}" \\
  python detectron_fastrcnn.py run \\
  --config "${{CONFIG}}" \\
  --train{_eval_flags(config)} \\
  --output-dir "${{OUTPUT_DIR}}" \\
  --batch-size {bs} \\
  --max-iterations {itr} \\
  --early-stopping-patience 5

RUN_ERR=$?
if [[ $RUN_ERR -ne 0 ]]; then
  echo "detectron_fastrcnn.py failed with exit code $RUN_ERR"
  exit $RUN_ERR
fi

{_shift_apptainer(config)}"""


def gen_maskrcnn(config: str, n: int) -> str:
    bs   = batch_size("maskrcnn", n)
    itr  = max_iterations(n, bs, 50)
    jname = job_name("maskrcnn", config)
    short = config_shortname(config)
    mem  = memory("maskrcnn", n)
    tl   = time_limit("maskrcnn", n)

    return f"""\
#!/bin/bash
#SBATCH --job-name={jname}
#SBATCH --partition=serc
#SBATCH --gres=gpu:2
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=16
#SBATCH --mem={mem}
#SBATCH --time={tl}
#SBATCH --output=logs/maskrcnn_{short}_%j.out
#SBATCH --error=logs/maskrcnn_{short}_%j.err

# -----------------------------------------------------------------------
# Mask R-CNN: {config}
# {n} train images, batch {bs} (2 GPUs), 50 epochs = {itr} iterations.
# Early stopping: patience 5 epochs. train + eval ID + eval OOD + SHIFT.
# -----------------------------------------------------------------------

set -euo pipefail

REPO_ROOT="{REPO_ROOT}"
IMAGE_SIF="${{DETECTRON_SIF:-/scratch/groups/dlobell/aadityan/tree-distribution-shift/detectron.sif}}"
METADATA_CSV="${{METADATA_CSV:-/scratch/groups/dlobell/aadityan/dataset/metadata.csv}}"

CONFIG={config}
OUTPUT_DIR="${{REPO_ROOT}}/outputs/maskrcnn_{short}_${{SLURM_JOB_ID}}"

cd "${{REPO_ROOT}}"
mkdir -p logs outputs

if [[ ! -f "${{IMAGE_SIF}}" ]]; then
  echo "ERROR: Apptainer image not found: ${{IMAGE_SIF}}"
  echo "Run: apptainer build --fakeroot detectron.sif detectron.def"
  exit 1
fi

# ----- 1. Train + evaluate on ID test, OOD test, and OOD train -----
apptainer exec --nv \\
  --bind "${{REPO_ROOT}}:/workspace" \\
  "${{IMAGE_SIF}}" \\
  python detectron_maskrcnn.py run \\
  --config "${{CONFIG}}" \\
  --train{_eval_flags(config)} \\
  --eval-mode distshift \\
  --output-dir "${{OUTPUT_DIR}}" \\
  --batch-size {bs} \\
  --max-iterations {itr} \\
  --early-stopping-patience 5

RUN_ERR=$?
if [[ $RUN_ERR -ne 0 ]]; then
  echo "detectron_maskrcnn.py failed with exit code $RUN_ERR"
  exit $RUN_ERR
fi

{_shift_apptainer(config)}"""


def gen_fastrcnn_pretrained(config: str, n: int) -> str:
    bs   = batch_size("fastrcnn_pretrained", n)
    itr  = max_iterations(n, bs, 50)
    jname = job_name("fastrcnn_pretrained", config)
    short = config_shortname(config)
    mem  = memory("fastrcnn_pretrained", n)
    tl   = time_limit("fastrcnn_pretrained", n)

    return f"""\
#!/bin/bash
#SBATCH --job-name={jname}
#SBATCH --partition=serc
#SBATCH --gres=gpu:2
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=16
#SBATCH --mem={mem}
#SBATCH --time={tl}
#SBATCH --output=logs/fastrcnn_pretrained_{short}_%j.out
#SBATCH --error=logs/fastrcnn_pretrained_{short}_%j.err

# -----------------------------------------------------------------------
# Faster R-CNN (COCO pretrained): {config}
# {n} train images, batch {bs} (2 GPUs), 50 epochs = {itr} iterations.
# Early stopping: patience 5 epochs. train + eval ID + eval OOD + SHIFT.
# -----------------------------------------------------------------------

set -euo pipefail

REPO_ROOT="{REPO_ROOT}"
IMAGE_SIF="${{DETECTRON_SIF:-/scratch/groups/dlobell/aadityan/tree-distribution-shift/detectron.sif}}"
METADATA_CSV="${{METADATA_CSV:-/scratch/groups/dlobell/aadityan/dataset/metadata.csv}}"

CONFIG={config}
OUTPUT_DIR="${{REPO_ROOT}}/outputs/fastrcnn_pretrained_{short}_${{SLURM_JOB_ID}}"

cd "${{REPO_ROOT}}"
mkdir -p logs outputs

if [[ ! -f "${{IMAGE_SIF}}" ]]; then
  echo "ERROR: Apptainer image not found: ${{IMAGE_SIF}}"
  echo "Run: apptainer build --fakeroot detectron.sif detectron.def"
  exit 1
fi

# ----- 1. Train + evaluate on ID test, OOD test, and OOD train -----
apptainer exec --nv \\
  --bind "${{REPO_ROOT}}:/workspace" \\
  "${{IMAGE_SIF}}" \\
  python detectron_fastrcnn.py run \\
  --config "${{CONFIG}}" \\
  --train{_eval_flags(config)} \\
  --output-dir "${{OUTPUT_DIR}}" \\
  --batch-size {bs} \\
  --max-iterations {itr} \\
  --learning-rate 0.005 \\
  --pretrained \\
  --early-stopping-patience 5

RUN_ERR=$?
if [[ $RUN_ERR -ne 0 ]]; then
  echo "detectron_fastrcnn.py failed with exit code $RUN_ERR"
  exit $RUN_ERR
fi

{_shift_apptainer(config)}"""


def gen_maskrcnn_pretrained(config: str, n: int) -> str:
    bs   = batch_size("maskrcnn_pretrained", n)
    itr  = max_iterations(n, bs, 50)
    jname = job_name("maskrcnn_pretrained", config)
    short = config_shortname(config)
    mem  = memory("maskrcnn_pretrained", n)
    tl   = time_limit("maskrcnn_pretrained", n)

    return f"""\
#!/bin/bash
#SBATCH --job-name={jname}
#SBATCH --partition=serc
#SBATCH --gres=gpu:2
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=16
#SBATCH --mem={mem}
#SBATCH --time={tl}
#SBATCH --output=logs/maskrcnn_pretrained_{short}_%j.out
#SBATCH --error=logs/maskrcnn_pretrained_{short}_%j.err

# -----------------------------------------------------------------------
# Mask R-CNN (COCO pretrained): {config}
# {n} train images, batch {bs} (2 GPUs), 50 epochs = {itr} iterations.
# Early stopping: patience 5 epochs. train + eval ID + eval OOD + SHIFT.
# -----------------------------------------------------------------------

set -euo pipefail

REPO_ROOT="{REPO_ROOT}"
IMAGE_SIF="${{DETECTRON_SIF:-/scratch/groups/dlobell/aadityan/tree-distribution-shift/detectron.sif}}"
METADATA_CSV="${{METADATA_CSV:-/scratch/groups/dlobell/aadityan/dataset/metadata.csv}}"

CONFIG={config}
OUTPUT_DIR="${{REPO_ROOT}}/outputs/maskrcnn_pretrained_{short}_${{SLURM_JOB_ID}}"

cd "${{REPO_ROOT}}"
mkdir -p logs outputs

if [[ ! -f "${{IMAGE_SIF}}" ]]; then
  echo "ERROR: Apptainer image not found: ${{IMAGE_SIF}}"
  echo "Run: apptainer build --fakeroot detectron.sif detectron.def"
  exit 1
fi

# ----- 1. Train + evaluate on ID test, OOD test, and OOD train -----
apptainer exec --nv \\
  --bind "${{REPO_ROOT}}:/workspace" \\
  "${{IMAGE_SIF}}" \\
  python detectron_maskrcnn.py run \\
  --config "${{CONFIG}}" \\
  --train{_eval_flags(config)} \\
  --eval-mode distshift \\
  --output-dir "${{OUTPUT_DIR}}" \\
  --batch-size {bs} \\
  --max-iterations {itr} \\
  --learning-rate 0.005 \\
  --pretrained \\
  --early-stopping-patience 5

RUN_ERR=$?
if [[ $RUN_ERR -ne 0 ]]; then
  echo "detectron_maskrcnn.py failed with exit code $RUN_ERR"
  exit $RUN_ERR
fi

{_shift_apptainer(config)}"""


def gen_plaindetr_dinov3(config: str, n: int) -> str:
    jname = job_name("plaindetr_dinov3", config)
    short = config_shortname(config)
    mem  = memory("plaindetr_dinov3", n)
    tl   = time_limit("plaindetr_dinov3", n)

    return f"""\
#!/bin/bash
#SBATCH --job-name={jname}
#SBATCH --partition=serc
#SBATCH --gres=gpu:3
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=24
#SBATCH --mem={mem}
#SBATCH --time={tl}
#SBATCH --output=logs/plaindetr_dinov3_{short}_%j.out
#SBATCH --error=logs/plaindetr_dinov3_{short}_%j.err

# -----------------------------------------------------------------------
# Plain-DETR + DinoV3 (ViT-S/16): {config}
# {n} train images, batch 2/GPU × 3 A100 GPUs, 50 epochs max.
# Early stopping: patience 5 epochs. train + eval ID + eval OOD + SHIFT.
# -----------------------------------------------------------------------

set -euo pipefail

REPO_ROOT="{REPO_ROOT}"
IMAGE_SIF="${{TREE_SHIFT_SIF:-/scratch/groups/dlobell/aadityan/tree-distribution-shift/tree-shift.sif}}"
METADATA_CSV="${{METADATA_CSV:-/scratch/groups/dlobell/aadityan/dataset/metadata.csv}}"
DINO_WEIGHTS_ROOT="${{DINO_WEIGHTS_ROOT:-/scratch/groups/dlobell/aadityan/dino_weights}}"
DINOV3_REPO="${{DINOV3_REPO:-/opt/dinov3}}"
DINO_WEIGHTS="${{DINO_WEIGHTS:-${{DINO_WEIGHTS_ROOT}}/dinov3_vits.pth}}"

CONFIG={config}
OUTPUT_DIR="${{REPO_ROOT}}/outputs/plaindetr_dinov3_{short}_${{SLURM_JOB_ID}}"

cd "${{REPO_ROOT}}"
mkdir -p logs outputs

if [[ ! -f "${{IMAGE_SIF}}" ]]; then
  echo "ERROR: Apptainer image not found: ${{IMAGE_SIF}}"
  echo "Run: ./scripts/pull_apptainer.sh"
  exit 1
fi

if [[ ! -f "${{DINO_WEIGHTS}}" ]]; then
  echo "ERROR: DinoV3 ViT-S weights not found: ${{DINO_WEIGHTS}}"
  exit 1
fi

echo "=== Plain-DETR DinoV3: {config} ==="
echo "Container : ${{IMAGE_SIF}}"
echo "Config    : ${{CONFIG}}"
echo "Output    : ${{OUTPUT_DIR}}"
echo ""

# ----- 1. Train + evaluate -----
export NCCL_IB_DISABLE=1
export NCCL_SOCKET_IFNAME=^lo,docker0
NUM_GPUS=3
MASTER_PORT=$(( 29400 + SLURM_JOB_ID % 10000 ))
apptainer exec --nv \\
  --bind "${{REPO_ROOT}}:/workspace" \\
  --env "NCCL_IB_DISABLE=${{NCCL_IB_DISABLE}}" \\
  --env "NCCL_SOCKET_IFNAME=${{NCCL_SOCKET_IFNAME}}" \\
  "${{IMAGE_SIF}}" \\
  torchrun --standalone --nproc_per_node=${{NUM_GPUS}} --master_port=${{MASTER_PORT}} plain_detr.py run \\
  --config "${{CONFIG}}" \\
  --train{_eval_flags(config)} \\
  --output-dir "${{OUTPUT_DIR}}" \\
  --batch-size 2 \\
  --decoder_use_checkpoint \\
  --epochs 50 \\
  --early_stop \\
  --early_stop_patience 5 \\
  --early_stop_min_epochs 10 \\
  --early_stop_min_delta 0.0 \\
  --backbone dinov3 \\
  --dinov3_repo "${{DINOV3_REPO}}" \\
  --dinov3_model dinov3_vits16 \\
  --dinov3_weights "${{DINO_WEIGHTS}}" \\
  --layers_to_use 3 6 9 11 \\
  --num_feature_levels 1 \\
  --proposal_feature_levels 1 \\
  --proposal_in_stride 16 \\
  --proposal_tgt_strides 16 \\
  --add_transformer_encoder \\
  --num_encoder_layers 6 \\
  --norm_type pre_norm \\
  --hidden_dim 256 \\
  --dim_feedforward 2048 \\
  --nheads 8 \\
  --decoder_type global_rpe_decomp \\
  --decoder_rpe_type linear \\
  --decoder_rpe_hidden_dim 256 \\
  --two_stage \\
  --mixed_selection \\
  --with_box_refine \\
  --num_queries_one2one 100 \\
  --num_queries_one2many 0 \\
  --k_one2many 0 \\
  --n_windows_sqrt 3

RUN_ERR=$?
if [[ $RUN_ERR -ne 0 ]]; then
  echo "plain_detr.py failed with exit code $RUN_ERR"
  exit $RUN_ERR
fi

{_shift_apptainer(config)}"""


def gen_plaindetr_dinov3_7b16(config: str, n: int) -> str:
    jname = job_name("plaindetr_dinov3_7b16", config)
    short = config_shortname(config)
    mem  = memory("plaindetr_dinov3_7b16", n)
    tl   = time_limit("plaindetr_dinov3_7b16", n)

    return f"""\
#!/bin/bash
#SBATCH --job-name={jname}
#SBATCH --partition=serc
#SBATCH --gres=gpu:3
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=24
#SBATCH --mem={mem}
#SBATCH --time={tl}
#SBATCH --output=logs/plaindetr_dinov3_7b16_{short}_%j.out
#SBATCH --error=logs/plaindetr_dinov3_7b16_{short}_%j.err

# -----------------------------------------------------------------------
# Plain-DETR + DinoV3 (dino_weights_7b16): {config}
# {n} train images, batch 2/GPU × 3 A100 GPUs, 50 epochs max.
# Early stopping: patience 5 epochs. train + eval ID + eval OOD + SHIFT.
# -----------------------------------------------------------------------

set -euo pipefail

REPO_ROOT="{REPO_ROOT}"
IMAGE_SIF="${{TREE_SHIFT_SIF:-/scratch/groups/dlobell/aadityan/tree-distribution-shift/tree-shift.sif}}"
METADATA_CSV="${{METADATA_CSV:-/scratch/groups/dlobell/aadityan/dataset/metadata.csv}}"
DINO_WEIGHTS_ROOT="${{DINO_WEIGHTS_ROOT:-/scratch/groups/dlobell/aadityan/dino_weights}}"
DINOV3_REPO="${{DINOV3_REPO:-/opt/dinov3}}"
DINO_WEIGHTS="${{DINO_WEIGHTS:-${{DINO_WEIGHTS_ROOT}}/dino_weights_7b16.pth}}"

CONFIG={config}
OUTPUT_DIR="${{REPO_ROOT}}/outputs/plaindetr_dinov3_7b16_{short}_${{SLURM_JOB_ID}}"

cd "${{REPO_ROOT}}"
mkdir -p logs outputs

if [[ ! -f "${{IMAGE_SIF}}" ]]; then
  echo "ERROR: Apptainer image not found: ${{IMAGE_SIF}}"
  echo "Run: ./scripts/pull_apptainer.sh"
  exit 1
fi

if [[ ! -f "${{DINO_WEIGHTS}}" ]]; then
  echo "ERROR: DinoV3 7b16 weights not found: ${{DINO_WEIGHTS}}"
  exit 1
fi

echo "=== Plain-DETR DinoV3 (7b16): {config} ==="
echo "Container : ${{IMAGE_SIF}}"
echo "Config    : ${{CONFIG}}"
echo "Weights   : ${{DINO_WEIGHTS}}"
echo "Output    : ${{OUTPUT_DIR}}"
echo ""

# ----- 1. Train + evaluate -----
export NCCL_IB_DISABLE=1
export NCCL_SOCKET_IFNAME=^lo,docker0
NUM_GPUS=3
MASTER_PORT=$(( 29400 + SLURM_JOB_ID % 10000 ))
apptainer exec --nv \\
  --bind "${{REPO_ROOT}}:/workspace" \\
  --env "NCCL_IB_DISABLE=${{NCCL_IB_DISABLE}}" \\
  --env "NCCL_SOCKET_IFNAME=${{NCCL_SOCKET_IFNAME}}" \\
  "${{IMAGE_SIF}}" \\
  torchrun --standalone --nproc_per_node=${{NUM_GPUS}} --master_port=${{MASTER_PORT}} plain_detr.py run \\
  --config "${{CONFIG}}" \\
  --train{_eval_flags(config)} \\
  --output-dir "${{OUTPUT_DIR}}" \\
  --batch-size 2 \\
  --decoder_use_checkpoint \\
  --epochs 50 \\
  --early_stop \\
  --early_stop_patience 5 \\
  --early_stop_min_epochs 10 \\
  --early_stop_min_delta 0.0 \\
  --backbone dinov3 \\
  --dinov3_repo "${{DINOV3_REPO}}" \\
  --dinov3_model dinov3_vit7b16 \\
  --dinov3_weights "${{DINO_WEIGHTS}}" \\
  --layers_to_use 3 6 9 11 \\
  --num_feature_levels 1 \\
  --proposal_feature_levels 1 \\
  --proposal_in_stride 16 \\
  --proposal_tgt_strides 16 \\
  --add_transformer_encoder \\
  --num_encoder_layers 6 \\
  --norm_type pre_norm \\
  --hidden_dim 256 \\
  --dim_feedforward 2048 \\
  --nheads 8 \\
  --decoder_type global_rpe_decomp \\
  --decoder_rpe_type linear \\
  --decoder_rpe_hidden_dim 256 \\
  --two_stage \\
  --mixed_selection \\
  --with_box_refine \\
  --num_queries_one2one 100 \\
  --num_queries_one2many 0 \\
  --k_one2many 0 \\
  --n_windows_sqrt 3

RUN_ERR=$?
if [[ $RUN_ERR -ne 0 ]]; then
  echo "plain_detr.py failed with exit code $RUN_ERR"
  exit $RUN_ERR
fi

{_shift_apptainer(config)}"""


def gen_plaindetr_dinov3_sat(config: str, n: int) -> str:
    jname = job_name("plaindetr_dinov3_sat", config)
    short = config_shortname(config)
    mem  = memory("plaindetr_dinov3_sat", n)
    tl   = time_limit("plaindetr_dinov3_sat", n)

    return f"""\
#!/bin/bash
#SBATCH --job-name={jname}
#SBATCH --partition=serc
#SBATCH --gres=gpu:3
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=24
#SBATCH --mem={mem}
#SBATCH --time={tl}
#SBATCH --output=logs/plaindetr_dinov3_sat_{short}_%j.out
#SBATCH --error=logs/plaindetr_dinov3_sat_{short}_%j.err

# -----------------------------------------------------------------------
# Plain-DETR + DinoV3 (dino_weights_sat): {config}
# {n} train images, batch 2/GPU × 3 A100 GPUs, 50 epochs max.
# Early stopping: patience 5 epochs. train + eval ID + eval OOD + SHIFT.
# -----------------------------------------------------------------------

set -euo pipefail

REPO_ROOT="{REPO_ROOT}"
IMAGE_SIF="${{TREE_SHIFT_SIF:-/scratch/groups/dlobell/aadityan/tree-distribution-shift/tree-shift.sif}}"
METADATA_CSV="${{METADATA_CSV:-/scratch/groups/dlobell/aadityan/dataset/metadata.csv}}"
DINO_WEIGHTS_ROOT="${{DINO_WEIGHTS_ROOT:-/scratch/groups/dlobell/aadityan/dino_weights}}"
DINOV3_REPO="${{DINOV3_REPO:-/opt/dinov3}}"
DINO_WEIGHTS="${{DINO_WEIGHTS:-${{DINO_WEIGHTS_ROOT}}/dino_weights_sat.pth}}"

CONFIG={config}
OUTPUT_DIR="${{REPO_ROOT}}/outputs/plaindetr_dinov3_sat_{short}_${{SLURM_JOB_ID}}"

cd "${{REPO_ROOT}}"
mkdir -p logs outputs

if [[ ! -f "${{IMAGE_SIF}}" ]]; then
  echo "ERROR: Apptainer image not found: ${{IMAGE_SIF}}"
  echo "Run: ./scripts/pull_apptainer.sh"
  exit 1
fi

if [[ ! -f "${{DINO_WEIGHTS}}" ]]; then
  echo "ERROR: DinoV3 sat weights not found: ${{DINO_WEIGHTS}}"
  exit 1
fi

echo "=== Plain-DETR DinoV3 (sat): {config} ==="
echo "Container : ${{IMAGE_SIF}}"
echo "Config    : ${{CONFIG}}"
echo "Weights   : ${{DINO_WEIGHTS}}"
echo "Output    : ${{OUTPUT_DIR}}"
echo ""

# ----- 1. Train + evaluate -----
export NCCL_IB_DISABLE=1
export NCCL_SOCKET_IFNAME=^lo,docker0
NUM_GPUS=3
MASTER_PORT=$(( 29400 + SLURM_JOB_ID % 10000 ))
apptainer exec --nv \\
  --bind "${{REPO_ROOT}}:/workspace" \\
  --env "NCCL_IB_DISABLE=${{NCCL_IB_DISABLE}}" \\
  --env "NCCL_SOCKET_IFNAME=${{NCCL_SOCKET_IFNAME}}" \\
  "${{IMAGE_SIF}}" \\
  torchrun --standalone --nproc_per_node=${{NUM_GPUS}} --master_port=${{MASTER_PORT}} plain_detr.py run \\
  --config "${{CONFIG}}" \\
  --train{_eval_flags(config)} \\
  --output-dir "${{OUTPUT_DIR}}" \\
  --batch-size 2 \\
  --decoder_use_checkpoint \\
  --epochs 50 \\
  --early_stop \\
  --early_stop_patience 5 \\
  --early_stop_min_epochs 10 \\
  --early_stop_min_delta 0.0 \\
  --backbone dinov3 \\
  --dinov3_repo "${{DINOV3_REPO}}" \\
  --dinov3_model dinov3_vit7b16 \\
  --dinov3_weights "${{DINO_WEIGHTS}}" \\
  --layers_to_use 3 6 9 11 \\
  --num_feature_levels 1 \\
  --proposal_feature_levels 1 \\
  --proposal_in_stride 16 \\
  --proposal_tgt_strides 16 \\
  --add_transformer_encoder \\
  --num_encoder_layers 6 \\
  --norm_type pre_norm \\
  --hidden_dim 256 \\
  --dim_feedforward 2048 \\
  --nheads 8 \\
  --decoder_type global_rpe_decomp \\
  --decoder_rpe_type linear \\
  --decoder_rpe_hidden_dim 256 \\
  --two_stage \\
  --mixed_selection \\
  --with_box_refine \\
  --num_queries_one2one 100 \\
  --num_queries_one2many 0 \\
  --k_one2many 0 \\
  --n_windows_sqrt 3

RUN_ERR=$?
if [[ $RUN_ERR -ne 0 ]]; then
  echo "plain_detr.py failed with exit code $RUN_ERR"
  exit $RUN_ERR
fi

{_shift_apptainer(config)}"""


def gen_plaindetr_resnet(config: str, n: int) -> str:
    jname = job_name("plaindetr_resnet", config)
    short = config_shortname(config)
    mem  = memory("plaindetr_resnet", n)
    tl   = time_limit("plaindetr_resnet", n)

    return f"""\
#!/bin/bash
#SBATCH --job-name={jname}
#SBATCH --partition=serc
#SBATCH --gres=gpu:3
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=24
#SBATCH --mem={mem}
#SBATCH --time={tl}
#SBATCH --output=logs/plaindetr_resnet_{short}_%j.out
#SBATCH --error=logs/plaindetr_resnet_{short}_%j.err

# -----------------------------------------------------------------------
# Plain-DETR + ResNet-50 FPN backbone: {config}
# {n} train images, batch 2/GPU × 3 A100 GPUs, 50 epochs max.
# Early stopping: patience 5 epochs. train + eval ID + eval OOD + SHIFT.
# -----------------------------------------------------------------------

set -euo pipefail

REPO_ROOT="{REPO_ROOT}"
IMAGE_SIF="${{TREE_SHIFT_SIF:-/scratch/groups/dlobell/aadityan/tree-distribution-shift/tree-shift.sif}}"
METADATA_CSV="${{METADATA_CSV:-/scratch/groups/dlobell/aadityan/dataset/metadata.csv}}"

CONFIG={config}
OUTPUT_DIR="${{REPO_ROOT}}/outputs/plaindetr_resnet_{short}_${{SLURM_JOB_ID}}"

cd "${{REPO_ROOT}}"
mkdir -p logs outputs

if [[ ! -f "${{IMAGE_SIF}}" ]]; then
  echo "ERROR: Apptainer image not found: ${{IMAGE_SIF}}"
  echo "Run: ./scripts/pull_apptainer.sh"
  exit 1
fi

echo "=== Plain-DETR ResNet50: {config} ==="
echo "Container : ${{IMAGE_SIF}}"
echo "Config    : ${{CONFIG}}"
echo "Output    : ${{OUTPUT_DIR}}"
echo ""

# ----- 1. Train + evaluate -----
export NCCL_IB_DISABLE=1
export NCCL_SOCKET_IFNAME=^lo,docker0
NUM_GPUS=3
MASTER_PORT=$(( 29400 + SLURM_JOB_ID % 10000 ))
apptainer exec --nv \\
  --bind "${{REPO_ROOT}}:/workspace" \\
  --env "NCCL_IB_DISABLE=${{NCCL_IB_DISABLE}}" \\
  --env "NCCL_SOCKET_IFNAME=${{NCCL_SOCKET_IFNAME}}" \\
  "${{IMAGE_SIF}}" \\
  torchrun --standalone --nproc_per_node=${{NUM_GPUS}} --master_port=${{MASTER_PORT}} plain_detr.py run \\
  --config "${{CONFIG}}" \\
  --train{_eval_flags(config)} \\
  --output-dir "${{OUTPUT_DIR}}" \\
  --batch-size 2 \\
  --decoder_use_checkpoint \\
  --epochs 50 \\
  --early_stop \\
  --early_stop_patience 5 \\
  --early_stop_min_epochs 10 \\
  --early_stop_min_delta 0.0 \\
  --backbone resnet50 \\
  --num_feature_levels 1 \\
  --proposal_feature_levels 3 \\
  --proposal_in_stride 32 \\
  --proposal_tgt_strides 8 16 32 \\
  --add_transformer_encoder \\
  --num_encoder_layers 6 \\
  --norm_type pre_norm \\
  --hidden_dim 256 \\
  --dim_feedforward 2048 \\
  --nheads 8 \\
  --decoder_type global_rpe_decomp \\
  --decoder_rpe_type linear \\
  --decoder_rpe_hidden_dim 256 \\
  --two_stage \\
  --mixed_selection \\
  --with_box_refine \\
  --num_queries_one2one 100 \\
  --num_queries_one2many 0 \\
  --k_one2many 0

RUN_ERR=$?
if [[ $RUN_ERR -ne 0 ]]; then
  echo "plain_detr.py failed with exit code $RUN_ERR"
  exit $RUN_ERR
fi

{_shift_apptainer(config)}"""


def gen_grounding_dino(config: str, n: int) -> str:
    jname = job_name("grounding_dino", config)
    short = config_shortname(config)
    mem  = memory("grounding_dino", n)
    tl   = time_limit("grounding_dino", n)

    return f"""\
#!/bin/bash
#SBATCH --job-name={jname}
#SBATCH --partition=serc
#SBATCH --gres=gpu:3
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=24
#SBATCH --mem={mem}
#SBATCH --time={tl}
#SBATCH --output=logs/grounding_dino_{short}_%j.out
#SBATCH --error=logs/grounding_dino_{short}_%j.err

# -----------------------------------------------------------------------
# MM-Grounding-DINO: {config}
# {n} train images, batch 2/GPU × 3 GPUs, 50 epochs max.
# Early stopping: patience 5 epochs. train + eval ID + eval OOD + SHIFT.
# -----------------------------------------------------------------------

set -euo pipefail

REPO_ROOT="{REPO_ROOT}"
IMAGE_SIF="${{TREE_SHIFT_SIF:-/scratch/groups/dlobell/aadityan/tree-distribution-shift/tree-shift.sif}}"
METADATA_CSV="${{METADATA_CSV:-/scratch/groups/dlobell/aadityan/dataset/metadata.csv}}"
HF_CACHE_HOST="${{REPO_ROOT}}/.hf_cache/huggingface"
HF_HOME="/workspace/.hf_cache/huggingface"
PRETRAINED_WEIGHTS_REL="model_weights/grounding_dino_swin-t_pretrain_obj365_goldg_grit9m_v3det.pth"
PRETRAINED_WEIGHTS_HOST="${{REPO_ROOT}}/${{PRETRAINED_WEIGHTS_REL}}"
PRETRAINED_WEIGHTS_URL="https://download.openmmlab.com/mmdetection/v3.0/mm_grounding_dino/grounding_dino_swin-t_pretrain_obj365_goldg_grit9m_v3det/grounding_dino_swin-t_pretrain_obj365_goldg_grit9m_v3det_20231204_095047-b448804b.pth"

CONFIG={config}
OUTPUT_DIR="${{REPO_ROOT}}/outputs/grounding_dino_{short}_${{SLURM_JOB_ID}}"
NUM_GPUS=3

cd "${{REPO_ROOT}}"
mkdir -p logs outputs
mkdir -p "${{HF_CACHE_HOST}}"

if [[ ! -f "${{IMAGE_SIF}}" ]]; then
  echo "ERROR: Apptainer image not found: ${{IMAGE_SIF}}"
  echo "Run: ./scripts/pull_apptainer.sh"
  exit 1
fi

if [[ ! -f "${{PRETRAINED_WEIGHTS_HOST}}" ]]; then
  echo "Pretrained MM-Grounding-DINO weights not found. Downloading once ..."
  mkdir -p "$(dirname "${{PRETRAINED_WEIGHTS_HOST}}")"
  wget -O "${{PRETRAINED_WEIGHTS_HOST}}" "${{PRETRAINED_WEIGHTS_URL}}"
fi

echo "=== GroundingDINO: {config} ==="
echo "Container : ${{IMAGE_SIF}}"
echo "Config    : ${{CONFIG}}"
echo "Output    : ${{OUTPUT_DIR}}"
echo ""

# Detect PyQt5 lib path (required by mmcv on headless nodes)
PYQT5_LIB=$(apptainer exec "${{IMAGE_SIF}}" python -c "
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
: "${{PYQT5_LIB:=/usr/local/lib/python3.9/dist-packages/PyQt5/Qt5/lib}}"

export QT_QPA_PLATFORM=offscreen
export NCCL_IB_DISABLE=1
export NCCL_SOCKET_IFNAME=^lo,docker0
MASTER_PORT=$(( 29400 + SLURM_JOB_ID % 10000 ))

# ----- 1. Train (multi-GPU via torchrun) -----
apptainer exec --nv \\
  --bind "${{REPO_ROOT}}:/workspace" \\
  --env QT_QPA_PLATFORM=offscreen \\
  --env "HF_HOME=${{HF_HOME}}" \\
  --env "NCCL_IB_DISABLE=${{NCCL_IB_DISABLE}}" \\
  --env "NCCL_SOCKET_IFNAME=${{NCCL_SOCKET_IFNAME}}" \\
  --env "LD_LIBRARY_PATH=${{PYQT5_LIB}}:${{LD_LIBRARY_PATH:-}}" \\
  "${{IMAGE_SIF}}" \\
  torchrun --standalone --nproc_per_node=${{NUM_GPUS}} --master_port=${{MASTER_PORT}} grounding_dino.py run \\
  --config "${{CONFIG}}" \\
  --train \\
  --output-dir "${{OUTPUT_DIR}}" \\
  --batch-size 2 \\
  --epochs 50 \\
  --lr 0.00015 \\
  --num-workers 4 \\
  --early-stopping-patience 5 \\
  --pretrained-weights "/workspace/${{PRETRAINED_WEIGHTS_REL}}"

TRAIN_ERR=$?
if [[ $TRAIN_ERR -ne 0 ]]; then
  echo "Training failed with exit code $TRAIN_ERR"
  exit $TRAIN_ERR
fi

# Find best checkpoint
CKPT=$(ls -t "${{OUTPUT_DIR}}"/best_coco_bbox_mAP*.pth 2>/dev/null | head -1)
if [[ -z "${{CKPT}}" ]]; then
  CKPT=$(ls -t "${{OUTPUT_DIR}}"/epoch_*.pth 2>/dev/null | head -1)
fi
if [[ -z "${{CKPT}}" ]]; then
  echo "ERROR: No checkpoint found in ${{OUTPUT_DIR}}"
  exit 1
fi
echo "Best checkpoint: ${{CKPT}}"

# ----- 2. Evaluate (single GPU, rank-safe) -----
apptainer exec --nv \\
  --bind "${{REPO_ROOT}}:/workspace" \\
  --env QT_QPA_PLATFORM=offscreen \\
  --env "HF_HOME=${{HF_HOME}}" \\
  --env "LD_LIBRARY_PATH=${{PYQT5_LIB}}:${{LD_LIBRARY_PATH:-}}" \\
  "${{IMAGE_SIF}}" \\
  python grounding_dino.py run \\
  --config "${{CONFIG}}" \\
  {_eval_flags(config).strip()} \\
  --model-path "${{CKPT}}" \\
  --output-dir "${{OUTPUT_DIR}}" \\
  --num-workers 4

EVAL_ERR=$?
if [[ $EVAL_ERR -ne 0 ]]; then
  echo "Evaluation failed with exit code $EVAL_ERR"
  exit $EVAL_ERR
fi

{_shift_apptainer(config)}"""


GENERATORS = {
    "fastrcnn":              gen_fastrcnn,
    "maskrcnn":              gen_maskrcnn,
    "fastrcnn_pretrained":   gen_fastrcnn_pretrained,
    "maskrcnn_pretrained":   gen_maskrcnn_pretrained,
    "plaindetr_dinov3":      gen_plaindetr_dinov3,
    "plaindetr_dinov3_7b16": gen_plaindetr_dinov3_7b16,
    "plaindetr_dinov3_sat":  gen_plaindetr_dinov3_sat,
    "plaindetr_resnet":      gen_plaindetr_resnet,
    "grounding_dino":        gen_grounding_dino,
}

# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    script_dir = os.path.dirname(os.path.abspath(__file__))
    generated = 0
    for config, n_images in CONFIGS.items():
        for model in MODELS:
            content = GENERATORS[model](config, n_images)
            fname   = script_filename(model, config)
            fpath   = os.path.join(script_dir, fname)
            with open(fpath, "w") as f:
                f.write(content)
            os.chmod(fpath, 0o755)
            generated += 1

    print(f"Generated {generated} scripts in {script_dir}/")
    print(f"  {len(CONFIGS)} configs × {len(MODELS)} models = {generated}")


if __name__ == "__main__":
    main()
