#!/usr/bin/env python3
"""
Submit pending few-shot jobs (non-Karnataka splits only).

Group 1 — Truly pending (no output exists at all):
  grounding_dino: intl_US_IN {fs10, fsall}, region_North_South {fs1, fs100, fsall}

Group 2 — Not yet run as 1gpu40ep (standard runs exist, but no 1gpu40ep output):
  fastrcnn: biome_Rajasthan_WET_DRY {fs10, fs100},
            intl_US_IN {fs1, fs10, fsall},
            region_North_South {fs1, fs10, fs100, fsall}

Group 1 is submitted first, Group 2 after.
"""

import subprocess
import sys
import os

SCRIPTS_DIR = os.path.join(os.path.dirname(__file__))

# ── Group 1: Pending grounding_dino few-shots ────────────────────────────────
PENDING = [
    "grounding_dino_intl_US_IN_fs10.sh",
    "grounding_dino_intl_US_IN_fsall.sh",
    "grounding_dino_region_North_South_fs1.sh",
    "grounding_dino_region_North_South_fs100.sh",
    "grounding_dino_region_North_South_fsall.sh",
]

# ── Group 2: fastrcnn 1gpu40ep backups not yet submitted ─────────────────────
NOT_1GPU40EP = [
    "fastrcnn_pretrained_biome_Rajasthan_WET_DRY_fs10.sh",
    "fastrcnn_pretrained_biome_Rajasthan_WET_DRY_fs100.sh",
    "fastrcnn_pretrained_intl_US_IN_fs1.sh",
    "fastrcnn_pretrained_intl_US_IN_fs10.sh",
    "fastrcnn_pretrained_intl_US_IN_fsall.sh",
    "fastrcnn_pretrained_region_North_South_fs1.sh",
    "fastrcnn_pretrained_region_North_South_fs10.sh",
    "fastrcnn_pretrained_region_North_South_fs100.sh",
    "fastrcnn_pretrained_region_North_South_fsall.sh",
]


def sbatch(script_name: str):
    path = os.path.join(SCRIPTS_DIR, script_name)
    if not os.path.isfile(path):
        print(f"  [ERROR] Script not found: {path}", file=sys.stderr)
        return None
    result = subprocess.run(["sbatch", path], capture_output=True, text=True)
    if result.returncode != 0:
        print(f"  [ERROR] sbatch failed for {script_name}:\n{result.stderr.strip()}", file=sys.stderr)
        return None
    job_id = result.stdout.strip().split()[-1]
    print(f"  Submitted {script_name} → job {job_id}")
    return job_id


def main():
    print("=" * 60)
    print("Group 1: Pending grounding_dino few-shots")
    print("=" * 60)
    for script in PENDING:
        sbatch(script)

    print()
    print("=" * 60)
    print("Group 2: fastrcnn 1gpu40ep backups")
    print("=" * 60)
    for script in NOT_1GPU40EP:
        sbatch(script)

    print()
    print("Done. Check with: squeue -u $USER")


if __name__ == "__main__":
    main()
