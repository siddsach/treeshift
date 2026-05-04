#!/usr/bin/env python3
"""
Submit eval-only SLURM jobs for grounding_dino runs that are missing eval results.

Scans outputs/ for grounding_dino_* directories, deduplicates by config (picks
the most recent run per config), skips currently running jobs, and submits a
lightweight 1-GPU eval + shift-analysis job for each missing result.

Usage (run on login node):
    python scripts/submit_gdino_evals.py [--outputs-dir DIR] [--dry-run]
"""
import argparse
import glob
import os
import re
import subprocess
import sys
import tempfile

REPO_ROOT    = "/scratch/groups/dlobell/aadityan/tree-distribution-shift"
OUTPUTS_DIR  = os.path.join(REPO_ROOT, "outputs")
IMAGE_SIF    = os.path.join(REPO_ROOT, "tree-shift.sif")
METADATA_CSV = os.path.join(REPO_ROOT, "../dataset/metadata.csv")
HF_HOME_CTR  = "/workspace/.hf_cache/huggingface"

# Reuse config registry and shortname helper from the script generator
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from generate_scripts import CONFIGS, config_shortname

SHORTNAME_TO_CONFIG = {config_shortname(c): c for c in CONFIGS}


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def get_running_job_ids():
    try:
        out = subprocess.check_output(
            ["squeue", "-u", os.environ["USER"], "-h", "-o", "%i"],
            text=True, stderr=subprocess.DEVNULL,
        )
        return set(out.split())
    except Exception:
        return set()


def find_best_ckpt(output_dir):
    patterns = [
        "best_coco_bbox_mAP*.pth",
        "best_coco_bbox_mAP_*.pth",
        "epoch_*.pth",
    ]
    candidates = []
    for p in patterns:
        candidates.extend(glob.glob(os.path.join(output_dir, p)))
    if not candidates:
        return None
    return max(candidates, key=os.path.getmtime)


def make_eval_script(output_dir, config, ckpt):
    short        = config_shortname(config)
    id_results   = os.path.join(output_dir, "eval_val",       "per_image_results.json")
    ood_train    = os.path.join(output_dir, "eval_ood_train", "per_image_results.json")
    shift_out    = os.path.join(output_dir, "shift_analysis")

    # PyQt5 detection as a one-liner (safe inside f-string — no {...} placeholders)
    pyqt5_oneliner = (
        r'PYQT5_LIB=$(apptainer exec "${IMAGE_SIF}" python -c "'
        r"import os;"
        r"[print(p) for s in ('Qt5','Qt')"
        r" for p in [os.path.join(os.path.dirname(__import__('PyQt5').__file__),s,'lib')]"
        r' if os.path.isdir(p)]" 2>/dev/null | head -1)'
        "\n"
        r': "${PYQT5_LIB:=/usr/local/lib/python3.9/dist-packages/PyQt5/Qt5/lib}"'
    )

    return f"""\
#!/bin/bash
#SBATCH --job-name=gde_{short[:35]}
#SBATCH --partition=serc
#SBATCH --gres=gpu:1
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --time=4:00:00
#SBATCH --output={REPO_ROOT}/logs/gde_{short}_%j.out
#SBATCH --error={REPO_ROOT}/logs/gde_{short}_%j.err

set -euo pipefail

IMAGE_SIF={IMAGE_SIF}
cd {REPO_ROOT}
mkdir -p {REPO_ROOT}/logs
export QT_QPA_PLATFORM=offscreen

{pyqt5_oneliner}

# ----- Evaluate ID val, OOD test (metrics), OOD train (shift analysis only) -----
apptainer exec --nv \\
  --bind {REPO_ROOT}:/workspace \\
  --env QT_QPA_PLATFORM=offscreen \\
  --env "HF_HOME={HF_HOME_CTR}" \\
  --env "LD_LIBRARY_PATH=${{PYQT5_LIB}}:${{LD_LIBRARY_PATH:-}}" \\
  {IMAGE_SIF} \\
  python grounding_dino.py run \\
  --config {config} \\
  --eval-val --eval-ood --eval-ood-train \\
  --model-path {ckpt} \\
  --output-dir {output_dir} \\
  --num-workers 4

EVAL_ERR=$?
if [[ $EVAL_ERR -ne 0 ]]; then
  echo "Eval failed with exit code $EVAL_ERR"
  exit $EVAL_ERR
fi

# ----- Shift analysis (ID vs OOD train; OOD test is held-out for reported metric) -----
if [[ -f "{id_results}" ]] && [[ -f "{ood_train}" ]]; then
  apptainer exec --nv \\
    --bind {REPO_ROOT}:/workspace \\
    {IMAGE_SIF} \\
    python shift_analysis.py univariate \\
    --id-results "{id_results}" \\
    --ood-results "{ood_train}" \\
    --metadata "{METADATA_CSV}" \\
    --output-dir "{shift_out}"

  apptainer exec --nv \\
    --bind {REPO_ROOT}:/workspace \\
    {IMAGE_SIF} \\
    python shift_analysis.py shapley \\
    --id-results "{id_results}" \\
    --ood-results "{ood_train}" \\
    --metadata "{METADATA_CSV}" \\
    --output-dir "{shift_out}"

  echo "Shift analysis complete."
else
  echo "WARNING: eval outputs not found (ID or OOD train), skipping shift analysis."
fi

echo "Done: {output_dir}"
"""


def submit(script_content, dry_run):
    if dry_run:
        return None
    with tempfile.NamedTemporaryFile(mode="w", suffix=".sh", delete=False) as f:
        f.write(script_content)
        tmp = f.name
    try:
        result = subprocess.run(["sbatch", tmp], capture_output=True, text=True)
        if result.returncode == 0:
            return result.stdout.strip().split()[-1]   # job ID
        else:
            print(f"  sbatch error: {result.stderr.strip()}")
            return None
    finally:
        os.unlink(tmp)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--outputs-dir", default=None,
                        help=f"Outputs directory to scan (default: {OUTPUTS_DIR})")
    parser.add_argument("--dry-run", action="store_true",
                        help="Print what would be submitted without actually submitting.")
    args = parser.parse_args()

    outputs_dir = args.outputs_dir or OUTPUTS_DIR
    if not os.path.isdir(outputs_dir):
        print(f"ERROR: outputs dir not found: {outputs_dir}", file=sys.stderr)
        sys.exit(1)

    running_jobs = get_running_job_ids()

    # ------------------------------------------------------------------
    # Scan outputs/, group by shortname — keep ALL runs sorted newest-first
    # ------------------------------------------------------------------
    # Map: shortname → list of (job_id_int, job_id_str, output_dir) newest first
    runs_by_short = {}

    for entry in sorted(os.listdir(outputs_dir)):
        if not entry.startswith("grounding_dino_"):
            continue
        output_dir = os.path.join(outputs_dir, entry)
        if not os.path.isdir(output_dir):
            continue

        m = re.search(r"_(\d+)$", entry)
        if not m:
            continue
        job_id     = m.group(1)
        job_id_int = int(job_id)

        short = entry[len("grounding_dino_"):]          # strip prefix
        short = short[:-(len(job_id) + 1)]               # strip _JOBID suffix

        runs_by_short.setdefault(short, []).append((job_id_int, job_id, output_dir))

    # Sort each config's runs newest → oldest
    for short in runs_by_short:
        runs_by_short[short].sort(key=lambda x: x[0], reverse=True)

    # For each config, pick the best candidate:
    #   - Already has eval → skip entirely
    #   - Most-recent run that has a checkpoint (or is running)
    # Fall back through older runs if the newest has no checkpoint and isn't running.
    latest_by_short = {}   # shortname → (output_dir, job_id_str)
    for short, runs in runs_by_short.items():
        chosen = None
        for job_id_int, job_id, output_dir in runs:
            # Already done? Skip the whole config.
            if os.path.exists(os.path.join(output_dir, "eval_val", "per_image_results.json")):
                chosen = None
                break
            # Running — will produce results eventually, skip
            if job_id in running_jobs:
                chosen = None
                break
            # Has a checkpoint — this is our candidate
            if find_best_ckpt(output_dir):
                chosen = (output_dir, job_id)
                break
            # No checkpoint, not running → failed before epoch 1; try older run
        if chosen:
            latest_by_short[short] = chosen

    # ------------------------------------------------------------------
    # Process each config
    # ------------------------------------------------------------------
    n_submitted      = 0
    n_already_done   = 0
    n_running        = 0
    n_no_ckpt        = 0
    n_unknown_config = 0

    # Count skipped configs (already done / running / all runs have no ckpt)
    for short, runs in runs_by_short.items():
        has_eval = any(
            os.path.exists(os.path.join(d, "eval_val", "per_image_results.json"))
            for _, _, d in runs
        )
        if has_eval:
            n_already_done += 1
            continue
        is_running = any(j in running_jobs for _, j, _ in runs)
        if is_running:
            n_running += 1
            continue
        if short not in latest_by_short:
            # All runs for this config have no checkpoint
            n_no_ckpt += 1
            dirs = [os.path.basename(d) for _, _, d in runs]
            print(f"  SKIP (no checkpoint in any run): {', '.join(dirs)}")

    for short, (output_dir, job_id) in sorted(latest_by_short.items()):
        if short not in SHORTNAME_TO_CONFIG:
            n_unknown_config += 1
            print(f"  SKIP (unknown shortname '{short}'): {os.path.basename(output_dir)}")
            continue
        config = SHORTNAME_TO_CONFIG[short]
        ckpt   = find_best_ckpt(output_dir)   # already confirmed to exist

        script = make_eval_script(output_dir, config, ckpt)

        if args.dry_run:
            print(f"  [DRY RUN] {os.path.basename(output_dir)}")
            print(f"            config={config}")
            print(f"            ckpt  ={os.path.basename(ckpt)}")
        else:
            submitted_id = submit(script, dry_run=False)
            if submitted_id:
                print(f"  Submitted job {submitted_id}: {os.path.basename(output_dir)}")
                n_submitted += 1

    print()
    if args.dry_run:
        print(f"Dry run — would submit {len(latest_by_short)} eval jobs.")
    else:
        print(f"Submitted : {n_submitted}")
    print(f"Already done  : {n_already_done}")
    print(f"Still running : {n_running}")
    print(f"No checkpoint : {n_no_ckpt}")
    if n_unknown_config:
        print(f"Unknown config: {n_unknown_config}")


if __name__ == "__main__":
    main()
