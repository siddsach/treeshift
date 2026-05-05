#!/usr/bin/env python3
"""
Generate scripts2/ — backup copies of all grounding_dino + plaindetr scripts
with 1 GPU, reduced memory, reduced CPUs, and fixed 40 epochs (no early stopping).
"""

import os
import stat

SCRIPTS_DIR = os.path.join(os.path.dirname(__file__), "..", "scripts")
SCRIPTS2_DIR = os.path.dirname(__file__)


def transform(content: str, is_gdino: bool) -> str:
    lines = content.split("\n")
    out = []
    for line in lines:
        stripped = line.strip()

        # ── Skip early-stopping argument lines ──────────────────────────────
        if is_gdino:
            if stripped == "--early-stopping-patience 5 \\":
                continue
        else:  # plaindetr variants
            if stripped in (
                "--early_stop \\",
                "--early_stop_patience 5 \\",
                "--early_stop_min_epochs 10 \\",
                "--early_stop_min_delta 0.0 \\",
            ):
                continue

        # ── GPU / CPU / memory SBATCH directives ───────────────────────────
        line = line.replace("--partition=serc", "--partition=serc,gpu")
        line = line.replace("--partition=owners,serc", "--partition=serc,gpu")
        line = line.replace("--partition=gpu", "--partition=serc,gpu")
        line = line.replace("--gres=gpu:3", "--gres=gpu:1")
        line = line.replace("--cpus-per-task=24", "--cpus-per-task=8")
        if is_gdino:
            line = line.replace("--mem=96G", "--mem=32G")
            line = line.replace("--mem=128G", "--mem=48G")
            line = line.replace("--mem=192G", "--mem=64G")
        else:
            line = line.replace("--mem=192G", "--mem=64G")
            line = line.replace("--mem=128G", "--mem=48G")
            line = line.replace("--mem=96G", "--mem=32G")

        # ── NUM_GPUS variable ───────────────────────────────────────────────
        line = line.replace("NUM_GPUS=3", "NUM_GPUS=1")

        # ── Epochs ─────────────────────────────────────────────────────────
        line = line.replace("--epochs 50", "--epochs 40")

        # ── Header comment updates ──────────────────────────────────────────
        line = line.replace("batch 2/GPU × 3 GPUs, 50 epochs max.", "batch 2/GPU × 1 GPU, 40 epochs max.")
        line = line.replace("batch 2/GPU × 3 A100 GPUs, 50 epochs max.", "batch 2/GPU × 1 A100 GPU, 40 epochs max.")
        line = line.replace("Early stopping: patience 5 epochs. ", "")

        # ── Output dir: add _1gpu40ep suffix before job-id placeholder ─────
        line = line.replace("_${SLURM_JOB_ID}\"", "_1gpu40ep_${SLURM_JOB_ID}\"")

        out.append(line)

    return "\n".join(out)


def main():
    os.makedirs(SCRIPTS2_DIR, exist_ok=True)
    count = 0
    for fname in sorted(os.listdir(SCRIPTS_DIR)):
        if not fname.endswith(".sh"):
            continue
        is_gdino = fname.startswith("grounding_dino_")
        is_plaindetr = fname.startswith("plaindetr_")
        if not (is_gdino or is_plaindetr):
            continue

        src = os.path.join(SCRIPTS_DIR, fname)
        dst = os.path.join(SCRIPTS2_DIR, fname)

        with open(src, encoding="utf-8") as f:
            content = f.read()

        new_content = transform(content, is_gdino)

        with open(dst, "w", encoding="utf-8") as f:
            f.write(new_content)

        # preserve executable bit
        os.chmod(dst, os.stat(src).st_mode | stat.S_IXUSR | stat.S_IXGRP | stat.S_IXOTH)
        count += 1

    print(f"Created {count} scripts in {SCRIPTS2_DIR}/")


if __name__ == "__main__":
    main()
