#!/usr/bin/env python3
"""
Submit backup (1-GPU, 40-epoch) jobs from scripts2/ for the selected configs.

Submits:
  - Base grounding_dino scripts     (4 scripts)
  - Base plaindetr_dinov3 scripts   (4 scripts)
  - All grounding_dino few-shot scripts (fs1/fs10/fs100/fsall × 4 configs = 16 scripts)

Total: 24 scripts

Run from the repo root:
    python scripts2/submit_backup_batch.py [--dry-run]
"""

import argparse
import os
import subprocess
import sys

SCRIPTS2_DIR = os.path.join(os.path.dirname(__file__))

# ── Script list ────────────────────────────────────────────────────────────────
# 1. Base grounding_dino (no few-shot)
GDINO_BASE = [
    "grounding_dino_biome_Rajasthan_WET_DRY.sh",       # Rajasthan WET → DRY
    "grounding_dino_elev_Karnataka_HIGH_LOW.sh",         # Karnataka HIGH → LOW
    "grounding_dino_intl_US_IN.sh",                      # International US → IN
    "grounding_dino_region_North_South.sh",              # Region North → South
]

# 2. Base plaindetr + DinoV3 backbone (no few-shot)
PLAINDETR_DINOV3_BASE = [
    "plaindetr_dinov3_biome_Rajasthan_WET_DRY.sh",
    "plaindetr_dinov3_elev_Karnataka_HIGH_LOW.sh",
    "plaindetr_dinov3_intl_US_IN.sh",
    "plaindetr_dinov3_region_North_South.sh",
]

# 3. All grounding_dino few-shot variants for the same 4 configs
FS_VARIANTS = ["fs1", "fs10", "fs100", "fsall"]
CONFIGS = [
    "biome_Rajasthan_WET_DRY",
    "elev_Karnataka_HIGH_LOW",
    "intl_US_IN",
    "region_North_South",
]
GDINO_FEWSHOT = [
    f"grounding_dino_{cfg}_{fs}.sh"
    for cfg in CONFIGS
    for fs in FS_VARIANTS
]

ALL_SCRIPTS = GDINO_BASE + PLAINDETR_DINOV3_BASE + GDINO_FEWSHOT

# ── Submission ─────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--dry-run", action="store_true", help="Print commands without submitting")
    args = parser.parse_args()

    print(f"{'DRY RUN — ' if args.dry_run else ''}Submitting {len(ALL_SCRIPTS)} jobs from scripts2/\n")

    # Verify all scripts exist before submitting any
    missing = [s for s in ALL_SCRIPTS if not os.path.isfile(os.path.join(SCRIPTS2_DIR, s))]
    if missing:
        print("ERROR: The following scripts were not found in scripts2/:")
        for m in missing:
            print(f"  {m}")
        sys.exit(1)

    groups = [
        ("Base grounding_dino", GDINO_BASE),
        ("Base plaindetr_dinov3", PLAINDETR_DINOV3_BASE),
        ("grounding_dino few-shot", GDINO_FEWSHOT),
    ]

    job_ids = []
    for group_name, scripts in groups:
        print(f"── {group_name} ({len(scripts)} scripts) ──")
        for script in scripts:
            path = os.path.join(SCRIPTS2_DIR, script)
            cmd = ["sbatch", path]
            if args.dry_run:
                print(f"  [dry-run] sbatch {path}")
            else:
                result = subprocess.run(cmd, capture_output=True, text=True)
                if result.returncode != 0:
                    print(f"  ERROR submitting {script}: {result.stderr.strip()}")
                    sys.exit(result.returncode)
                job_id = result.stdout.strip().split()[-1]
                job_ids.append(job_id)
                print(f"  Submitted {script} → job {job_id}")
        print()

    if not args.dry_run:
        print(f"All {len(job_ids)} jobs submitted. IDs: {', '.join(job_ids)}")
        print("Monitor with: squeue -u $USER")


if __name__ == "__main__":
    main()
