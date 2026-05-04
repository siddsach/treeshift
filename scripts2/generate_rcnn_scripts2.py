#!/usr/bin/env python3
"""
Generate scripts2/ — 1-GPU, 40-epoch backup copies of selected
fastrcnn_pretrained and maskrcnn_pretrained scripts.

Transformations applied vs the original 2-GPU scripts:
  - --gres=gpu:2  →  --gres=gpu:1
  - --cpus-per-task=16  →  --cpus-per-task=8
  - Memory:  64G→32G  96G→48G  128G→64G
  - Batch size: halved  (16→8  20→10  24→12)
  - Max iterations: recalculated  ceil(n / new_batch) * 40
  - Early stopping removed  (--early-stopping-patience 5)
  - Output dir gains _1gpu40ep suffix

Run from repo root:
    python scripts2/generate_rcnn_scripts2.py
"""

import math
import os
import re
import stat

SCRIPTS_DIR  = os.path.join(os.path.dirname(__file__), "..", "scripts")
SCRIPTS2_DIR = os.path.dirname(__file__)

# ── Exact list of scripts to generate ────────────────────────────────────────
FASTRCNN_BASE = [
    "fastrcnn_pretrained_biome_Rajasthan_WET_DRY.sh",
    "fastrcnn_pretrained_elev_Karnataka_HIGH_LOW.sh",
    "fastrcnn_pretrained_intl_US_IN.sh",
    "fastrcnn_pretrained_region_North_South.sh",
]

MASKRCNN_BASE = [
    "maskrcnn_pretrained_biome_Rajasthan_WET_DRY.sh",
    "maskrcnn_pretrained_elev_Karnataka_HIGH_LOW.sh",
    "maskrcnn_pretrained_intl_US_IN.sh",
    "maskrcnn_pretrained_region_North_South.sh",
]

FASTRCNN_FEWSHOT = [
    f"fastrcnn_pretrained_{cfg}_{fs}.sh"
    for cfg in [
        "biome_Rajasthan_WET_DRY",
        "elev_Karnataka_HIGH_LOW",
        "intl_US_IN",
        "region_North_South",
    ]
    for fs in ["fs1", "fs10", "fs100", "fsall"]
]

ALL_SCRIPTS = FASTRCNN_BASE + MASKRCNN_BASE + FASTRCNN_FEWSHOT


# ── Transformation ────────────────────────────────────────────────────────────

def transform(content: str) -> str:
    # ── Extract n_images and current batch_size from header comment ──────────
    m_n     = re.search(r"#\s+(\d+) train images", content)
    m_batch = re.search(r"--batch-size\s+(\d+)", content)
    if not m_n or not m_batch:
        raise ValueError("Could not parse n_images or batch_size from script")

    n_images   = int(m_n.group(1))
    old_batch  = int(m_batch.group(1))
    new_batch  = old_batch // 2
    new_iters  = math.ceil(n_images / new_batch) * 40

    # ── SBATCH directives ────────────────────────────────────────────────────
    content = content.replace("--partition=serc", "--partition=serc,gpu")
    content = content.replace("--partition=owners,serc", "--partition=serc,gpu")
    content = content.replace("--partition=gpu", "--partition=serc,gpu")
    content = content.replace("--gres=gpu:2",       "--gres=gpu:1")
    content = content.replace("--cpus-per-task=16", "--cpus-per-task=8")
    content = content.replace("--mem=64G",  "--mem=32G")
    content = content.replace("--mem=96G",  "--mem=48G")
    content = content.replace("--mem=128G", "--mem=64G")

    # ── Batch size ───────────────────────────────────────────────────────────
    content = re.sub(
        r"(--batch-size\s+)\d+",
        lambda m: f"{m.group(1)}{new_batch}",
        content,
    )

    # ── Max iterations ───────────────────────────────────────────────────────
    content = re.sub(
        r"(--max-iterations\s+)\d+",
        lambda m: f"{m.group(1)}{new_iters}",
        content,
    )

    # ── Remove early stopping (last arg — no trailing backslash) ─────────────
    # Also strip the trailing \ from the line immediately before it (--pretrained)
    content = re.sub(
        r"(  --pretrained) \\\n  --early-stopping-patience \d+\n",
        r"\1\n",
        content,
    )
    # Fallback: in case it appears mid-command with a trailing backslash
    content = re.sub(r"\s+--early-stopping-patience \d+ \\\n", "\n", content)

    # ── Header comment ───────────────────────────────────────────────────────
    # e.g. "batch 16 (2 GPUs), 50 epochs = 5400 iterations."
    content = re.sub(
        r"batch \d+ \(2 GPUs\), 50 epochs = \d+ iterations\.",
        f"batch {new_batch} (1 GPU), 40 epochs = {new_iters} iterations.",
        content,
    )
    content = content.replace("Early stopping: patience 5 epochs. ", "")

    # ── Output dir: add _1gpu40ep suffix before job-id placeholder ───────────
    content = content.replace('_${SLURM_JOB_ID}"', '_1gpu40ep_${SLURM_JOB_ID}"')

    return content


# ── Main ──────────────────────────────────────────────────────────────────────

def main():
    missing = [s for s in ALL_SCRIPTS
               if not os.path.isfile(os.path.join(SCRIPTS_DIR, s))]
    if missing:
        print("ERROR: source scripts not found in scripts/:")
        for m in missing:
            print(f"  {m}")
        return

    count = 0
    for fname in ALL_SCRIPTS:
        src = os.path.join(SCRIPTS_DIR, fname)
        dst = os.path.join(SCRIPTS2_DIR, fname)

        with open(src) as f:
            content = f.read()

        new_content = transform(content)

        with open(dst, "w") as f:
            f.write(new_content)

        os.chmod(dst, os.stat(src).st_mode | stat.S_IXUSR | stat.S_IXGRP | stat.S_IXOTH)
        count += 1
        print(f"  {fname}")

    print(f"\nCreated {count} scripts in {SCRIPTS2_DIR}/")


if __name__ == "__main__":
    main()
