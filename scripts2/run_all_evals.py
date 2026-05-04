#!/usr/bin/env python3
"""
Submit SLURM eval-only jobs for model outputs missing evaluation results.

Scans the given outputs directory, identifies model type and dataset config
from each subdirectory name, finds the best checkpoint, and submits a 1-GPU
eval job that runs all three splits:

  - eval_val         (ID test):   per_image_results.json + COCO AP metrics
  - eval_ood_test    (OOD test):  per_image_results.json + COCO AP metrics
  - eval_ood_train   (OOD train): per_image_results.json (for analysis)

Produces per output directory:
  eval_val/per_image_results.json
  eval_ood_test/per_image_results.json
  eval_ood_train/per_image_results.json
  eval_summary.json           (combined metrics for all splits)
  eval_summary_table.png      (table image of ID + OOD test metrics)

Usage:
    python scripts2/run_all_evals.py --outputs-dir /path/to/final_outputs [--dry-run]
"""

import argparse
import os
import re
import subprocess
import sys
from pathlib import Path

# ── Paths ─────────────────────────────────────────────────────────────────────
REPO_ROOT         = "/scratch/groups/dlobell/aadityan/tree-distribution-shift"
DETECTRON_SIF     = f"{REPO_ROOT}/detectron.sif"
TREESHIFT_SIF     = f"{REPO_ROOT}/tree-shift.sif"
DINO_WEIGHTS_VITS = "/scratch/groups/dlobell/aadityan/dino_weights/dinov3_vits.pth"
DINO_WEIGHTS_SAT  = "/scratch/groups/dlobell/aadityan/dino_weights/dino_weights_sat.pth"
DINO_WEIGHTS_7B16 = "/scratch/groups/dlobell/aadityan/dino_weights/dino_weights_7b16.pth"

# ── Config shortname → full config ────────────────────────────────────────────
CONFIG_MAP = {
    "biome_Rajasthan_WET_DRY":  "biome_Rajasthan_train_WET__ood_DRY",
    "biome_Rajasthan_DRY_WET":  "biome_Rajasthan_train_DRY__ood_WET",
    "intl_US_IN":               "intl_train_US__ood_IN",
    "intl_IN_US":               "intl_train_IN__ood_US",
    "region_North_South":       "region_train_North__ood_South",
    "region_South_North":       "region_train_South__ood_North",
    "elev_Karnataka_HIGH_LOW":  "elev_Karnataka_train_HIGH__ood_LOW",
    "elev_Karnataka_LOW_HIGH":  "elev_Karnataka_train_LOW__ood_HIGH",
}

# ── Model prefixes (order: longer/more-specific first) ───────────────────────
MODEL_PREFIXES = [
    ("fastrcnn_pretrained_",    "fastrcnn_pretrained"),
    ("maskrcnn_pretrained_",    "maskrcnn_pretrained"),
    ("fastrcnn_",               "fastrcnn"),
    ("maskrcnn_",               "maskrcnn"),
    ("grounding_dino_",         "grounding_dino"),
    ("plaindetr_dinov3_sat_",   "plaindetr_dinov3_sat"),
    ("plaindetr_dinov3_7b16_",  "plaindetr_dinov3_7b16"),
    ("plaindetr_dinov3_",       "plaindetr_dinov3"),
    ("plaindetr_resnet_",       "plaindetr_resnet"),
]

# ── Per-model SLURM resources (1 GPU each) ───────────────────────────────────
# tree-shift.sif PyTorch supports CUDA sm_37–sm_86 only (no H100 / sm_90).
# Set GPU_CONSTRAINT below to match your cluster's feature flags.
# Run `sinfo -o "%N %G %f" -p serc,gpu | sort -u` to see available features.
GPU_CONSTRAINT = "GPU_SKU:A100_SXM4"

RESOURCES = {
    "fastrcnn":              {"cpus": 4,  "mem": "16G", "time": "4:00:00"},
    "fastrcnn_pretrained":   {"cpus": 4,  "mem": "16G", "time": "4:00:00"},
    "maskrcnn":              {"cpus": 4,  "mem": "16G", "time": "4:00:00"},
    "maskrcnn_pretrained":   {"cpus": 4,  "mem": "16G", "time": "4:00:00"},
    "grounding_dino":        {"cpus": 4,  "mem": "32G", "time": "12:00:00"},
    "plaindetr_dinov3":      {"cpus": 8,  "mem": "48G", "time": "12:00:00"},
    "plaindetr_dinov3_7b16": {"cpus": 8,  "mem": "48G", "time": "12:00:00"},
    "plaindetr_dinov3_sat":  {"cpus": 8,  "mem": "48G", "time": "12:00:00"},
    "plaindetr_resnet":      {"cpus": 4,  "mem": "32G", "time": "8:00:00"},
}

# Models using tree-shift.sif (needs GPU constraint to avoid H100/sm_90)
_TREESHIFT_MODELS = {"grounding_dino", "plaindetr_dinov3", "plaindetr_dinov3_7b16",
                     "plaindetr_dinov3_sat", "plaindetr_resnet"}


# ═══════════════════════════════════════════════════════════════════════════════
# Helpers
# ═══════════════════════════════════════════════════════════════════════════════

def parse_output_dir(dirname):
    """Return (model_type, full_config) or None if dirname is unrecognised."""
    name = dirname
    name = re.sub(r"_\d+$", "", name)       # strip trailing SLURM job ID
    name = name.replace("_1gpu40ep", "")     # strip training-time suffix

    model_type = shortname = None
    for prefix, mtype in MODEL_PREFIXES:
        if name.startswith(prefix):
            model_type = mtype
            shortname = name[len(prefix):]
            break
    if model_type is None:
        return None

    fewshot = ""
    for fs in ("_fsall", "_fs100", "_fs10", "_fs1"):
        if shortname.endswith(fs):
            fewshot = "__" + fs[1:]
            shortname = shortname[: -len(fs)]
            break

    base_config = CONFIG_MAP.get(shortname)
    if base_config is None:
        return None

    return model_type, base_config + fewshot


def _required_splits(config):
    splits = ["eval_val", "eval_ood_test"]
    if not config.endswith("__fsall"):
        splits.append("eval_ood_train")
    return splits


def clean_stale_summary(out_path, config):
    """Remove eval_summary.json if it exists but required per-image results are missing.

    This handles the case where a previous run wrote the summary after partial
    success (e.g. val+ood_test passed but ood_train crashed).  Removing the
    stale summary lets the idempotency check correctly re-submit the job.
    """
    summary = out_path / "eval_summary.json"
    if not summary.exists():
        return False
    required = _required_splits(config)
    missing = [
        sub for sub in required
        if not (out_path / sub / "per_image_results.json").exists()
    ]
    if missing:
        summary.unlink()
        table = out_path / "eval_summary_table.png"
        if table.exists():
            table.unlink()
        print(f"  [CLEAN] removed stale eval_summary.json from {out_path.name}"
              f" (missing: {', '.join(missing)})")
        return True
    return False


def eval_already_done(out_path, config):
    """True if the expected per_image_results.json files + summary exist.

    For fsall configs there is no ood_train split, so we only require val + ood_test.
    """
    required = _required_splits(config)
    has_splits = all(
        (out_path / sub / "per_image_results.json").exists()
        for sub in required
    )
    has_summary = (out_path / "eval_summary.json").exists()
    return has_splits and has_summary


# ═══════════════════════════════════════════════════════════════════════════════
# Checkpoint finders (bash snippets)
# ═══════════════════════════════════════════════════════════════════════════════

def _find_ckpt_detectron(out):
    return f"""\
CKPT=""
if [[ -f "{out}/last_checkpoint" ]]; then
    LAST=$(cat "{out}/last_checkpoint")
    CKPT="{out}/$LAST"
fi
if [[ -z "$CKPT" || ! -f "$CKPT" ]]; then
    CKPT=$(ls -t "{out}"/model_final.pth "{out}"/model_*.pth 2>/dev/null | head -1)
fi
if [[ -z "$CKPT" || ! -f "$CKPT" ]]; then
    echo "ERROR: no checkpoint in {out}" && exit 1
fi"""


def _find_ckpt_gdino(out):
    return f"""\
CKPT=$(ls -t "{out}"/best_coco_bbox_mAP*.pth 2>/dev/null | head -1)
if [[ -z "$CKPT" ]]; then
    CKPT=$(ls -t "{out}"/epoch_*.pth 2>/dev/null | head -1)
fi
if [[ -z "$CKPT" || ! -f "$CKPT" ]]; then
    echo "ERROR: no checkpoint in {out}" && exit 1
fi"""


def _find_ckpt_plaindetr(out):
    return f"""\
if [[ -f "{out}/checkpoint_best.pth" ]]; then
    CKPT="{out}/checkpoint_best.pth"
elif [[ -f "{out}/checkpoint.pth" ]]; then
    CKPT="{out}/checkpoint.pth"
else
    echo "ERROR: no checkpoint in {out}" && exit 1
fi"""


# ═══════════════════════════════════════════════════════════════════════════════
# Eval commands (bash snippets)
# ═══════════════════════════════════════════════════════════════════════════════

def _eval_detectron(out, config, py_script):
    return f"""\
apptainer exec --nv \\
  --bind "{REPO_ROOT}:/workspace" \\
  "{DETECTRON_SIF}" \\
  python {py_script} run \\
  --config "{config}" \\
  --eval-val --eval-ood --eval-ood-train \\
  --model-path "$CKPT" \\
  --output-dir "{out}"
"""


def _eval_gdino(out, config):
    return f"""\
PYQT5_LIB=$(apptainer exec --nv "{TREESHIFT_SIF}" python -c "
import os
try:
    import PyQt5
    for sub in ('Qt5', 'Qt'):
        p = os.path.join(os.path.dirname(PyQt5.__file__), sub, 'lib')
        if os.path.isdir(p):
            print(p); break
except Exception:
    pass
" 2>/dev/null | head -1)
: "${{PYQT5_LIB:=/usr/local/lib/python3.9/dist-packages/PyQt5/Qt5/lib}}"

apptainer exec --nv \\
  --bind "{REPO_ROOT}:/workspace" \\
  --env QT_QPA_PLATFORM=offscreen \\
  --env "HF_HOME=/workspace/.hf_cache/huggingface" \\
  --env "LD_LIBRARY_PATH=${{PYQT5_LIB}}:${{LD_LIBRARY_PATH:-}}" \\
  "{TREESHIFT_SIF}" \\
  python grounding_dino.py run \\
  --config "{config}" \\
  --eval-val --eval-ood --eval-ood-train \\
  --model-path "$CKPT" \\
  --output-dir "{out}" \\
  --num-workers 4
"""


# ── Plain-DETR architecture blocks ───────────────────────────────────────────

_PLAINDETR_COMMON_TAIL = """\
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
  --k_one2many 0"""


def _eval_plaindetr(out, config, model_type):
    if model_type == "plaindetr_dinov3":
        backbone = f"""\
  --backbone dinov3 \\
  --dinov3_repo /opt/dinov3 \\
  --dinov3_model dinov3_vits16 \\
  --dinov3_weights "{DINO_WEIGHTS_VITS}" \\
  --layers_to_use 3 6 9 11 \\
  --num_feature_levels 1 \\
  --proposal_feature_levels 1 \\
  --proposal_in_stride 16 \\
  --proposal_tgt_strides 16 \\
  --n_windows_sqrt 3 \\"""
    elif model_type == "plaindetr_dinov3_7b16":
        backbone = f"""\
  --backbone dinov3 \\
  --dinov3_repo /opt/dinov3 \\
  --dinov3_model dinov3_vit7b16 \\
  --dinov3_weights "{DINO_WEIGHTS_7B16}" \\
  --layers_to_use 3 6 9 11 \\
  --num_feature_levels 1 \\
  --proposal_feature_levels 1 \\
  --proposal_in_stride 16 \\
  --proposal_tgt_strides 16 \\
  --n_windows_sqrt 3 \\"""
    elif model_type == "plaindetr_dinov3_sat":
        backbone = f"""\
  --backbone dinov3 \\
  --dinov3_repo /opt/dinov3 \\
  --dinov3_model dinov3_vit7b16 \\
  --dinov3_weights "{DINO_WEIGHTS_SAT}" \\
  --layers_to_use 3 6 9 11 \\
  --num_feature_levels 1 \\
  --proposal_feature_levels 1 \\
  --proposal_in_stride 16 \\
  --proposal_tgt_strides 16 \\
  --n_windows_sqrt 3 \\"""
    else:  # plaindetr_resnet
        backbone = """\
  --backbone resnet50 \\
  --num_feature_levels 1 \\
  --proposal_feature_levels 3 \\
  --proposal_in_stride 32 \\
  --proposal_tgt_strides 8 16 32 \\"""

    return f"""\
MASTER_PORT=$(( 29400 + ${{SLURM_JOB_ID:-$$}} % 10000 ))
apptainer exec --nv \\
  --bind "{REPO_ROOT}:/workspace" \\
  --env "HF_HOME=/workspace/.hf_cache/huggingface" \\
  "{TREESHIFT_SIF}" \\
  torchrun --standalone --nproc_per_node=1 --master_port=$MASTER_PORT \\
  plain_detr.py run \\
  --config "{config}" \\
  --eval-val --eval-ood --eval-ood-train \\
  --model-path "$CKPT" \\
  --output-dir "{out}" \\
  --decoder_use_checkpoint \\
{backbone}
{_PLAINDETR_COMMON_TAIL}
"""


# ═══════════════════════════════════════════════════════════════════════════════
# SBATCH script builder
# ═══════════════════════════════════════════════════════════════════════════════

def make_sbatch(model_type, config, out_dir, dirname):
    res = RESOURCES.get(model_type, {"cpus": 8, "mem": "32G", "time": "12:00:00"})
    job_name = f"ev_{dirname}"[:72]
    log_base = f"{REPO_ROOT}/logs/eval_{dirname}"

    # Checkpoint finder
    if model_type in ("fastrcnn", "fastrcnn_pretrained"):
        find_ckpt = _find_ckpt_detectron(out_dir)
        eval_cmd  = _eval_detectron(out_dir, config, "detectron_fastrcnn.py")
    elif model_type in ("maskrcnn", "maskrcnn_pretrained"):
        find_ckpt = _find_ckpt_detectron(out_dir)
        eval_cmd  = _eval_detectron(out_dir, config, "detectron_maskrcnn.py")
    elif model_type == "grounding_dino":
        find_ckpt = _find_ckpt_gdino(out_dir)
        eval_cmd  = _eval_gdino(out_dir, config)
    elif model_type.startswith("plaindetr_"):
        find_ckpt = _find_ckpt_plaindetr(out_dir)
        eval_cmd  = _eval_plaindetr(out_dir, config, model_type)
    else:
        return None

    # Build skip-check: require val + ood_test + summary; also ood_train unless fsall
    skip_conditions = (
        f'[[ -f "{out_dir}/eval_val/per_image_results.json" ]] \\\n'
        f'&& [[ -f "{out_dir}/eval_ood_test/per_image_results.json" ]] \\\n'
    )
    if not config.endswith("__fsall"):
        skip_conditions += f'&& [[ -f "{out_dir}/eval_ood_train/per_image_results.json" ]] \\\n'
    skip_conditions += f'&& [[ -f "{out_dir}/eval_summary.json" ]]'

    constraint_line = ""
    if model_type in _TREESHIFT_MODELS and GPU_CONSTRAINT:
        constraint_line = f"\n#SBATCH --constraint={GPU_CONSTRAINT}"

    return f"""#!/bin/bash
#SBATCH --job-name={job_name}
#SBATCH --partition=serc,gpu
#SBATCH --gres=gpu:1{constraint_line}
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task={res['cpus']}
#SBATCH --mem={res['mem']}
#SBATCH --time={res['time']}
#SBATCH --output={log_base}_%j.out
#SBATCH --error={log_base}_%j.err

# Eval-only: {dirname}
# Model  : {model_type}
# Config : {config}

set -uo pipefail
cd "{REPO_ROOT}"
mkdir -p logs

# ---- skip if already evaluated ----
if {skip_conditions}; then
    echo "Eval already done for {dirname}, exiting."
    exit 0
fi

{find_ckpt}
echo "=== Eval: {dirname} ==="
echo "Model      : {model_type}"
echo "Config     : {config}"
echo "Output     : {out_dir}"
echo "Checkpoint : $CKPT"
echo ""

{eval_cmd}
ERR=$?
if [[ $ERR -ne 0 ]]; then
    echo "Eval exited with code $ERR (partial results may have been saved)"
fi

echo ""
echo "Done: {out_dir}"
"""


# ═══════════════════════════════════════════════════════════════════════════════
# Main
# ═══════════════════════════════════════════════════════════════════════════════

def main():
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument(
        "--outputs-dir", required=True,
        help="Directory containing model output subdirectories to evaluate",
    )
    parser.add_argument(
        "--dry-run", action="store_true",
        help="Print what would be submitted without actually submitting.",
    )
    args = parser.parse_args()

    outputs_path = Path(args.outputs_dir)
    if not outputs_path.exists():
        print(f"ERROR: outputs dir not found: {args.outputs_dir}", file=sys.stderr)
        sys.exit(1)

    dirs = sorted(d for d in outputs_path.iterdir() if d.is_dir())

    submitted       = 0
    skipped_done    = 0
    skipped_noparse = 0
    cleaned         = 0

    for d in dirs:
        dirname = d.name

        parsed = parse_output_dir(dirname)
        if parsed is None:
            print(f"  [SKIP/unrecognised] {dirname}")
            skipped_noparse += 1
            continue

        model_type, config = parsed

        if clean_stale_summary(d, config):
            cleaned += 1

        if eval_already_done(d, config):
            print(f"  [SKIP/done]         {dirname}  ({model_type})")
            skipped_done += 1
            continue

        script = make_sbatch(model_type, config, str(d), dirname)
        if script is None:
            print(f"  [SKIP/no-handler]   {dirname}  ({model_type})")
            skipped_noparse += 1
            continue

        if args.dry_run:
            print(f"  [DRY-RUN]           {dirname}  ({model_type}, {config})")
            submitted += 1
            continue

        result = subprocess.run(
            ["sbatch"], input=script, capture_output=True, text=True,
        )
        if result.returncode != 0:
            print(f"  [ERROR]             {dirname}: {result.stderr.strip()}")
        else:
            job_id = result.stdout.strip().split()[-1]
            print(f"  [SUBMITTED] job {job_id:<10} {dirname}  ({model_type})")
            submitted += 1

    print()
    if cleaned:
        print(f"Cleaned stale : {cleaned}")
    if args.dry_run:
        print(f"Would submit  : {submitted}")
    else:
        print(f"Submitted     : {submitted}")
    print(f"Already done  : {skipped_done}")
    print(f"Unrecognised  : {skipped_noparse}")
    print(f"Total dirs    : {len(dirs)}")
    if args.dry_run:
        print("(dry-run: no jobs actually submitted)")


if __name__ == "__main__":
    main()
