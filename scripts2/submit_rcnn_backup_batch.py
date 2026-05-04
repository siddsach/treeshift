#!/usr/bin/env python3
"""
Submit 1-GPU, 40-epoch fastrcnn_pretrained and maskrcnn_pretrained backup jobs
from scripts2/.

Submits:
  - Base fastrcnn_pretrained scripts    (4 scripts)
  - Base maskrcnn_pretrained scripts    (4 scripts)
  - fastrcnn_pretrained few-shot scripts  (fs1/fs10/fs100/fsall × 4 configs = 16 scripts)

Total: 24 scripts

Run from the repo root:
    python scripts2/submit_rcnn_backup_batch.py [--dry-run]
"""

import argparse
import os
import subprocess
import sys

SCRIPTS2_DIR = os.path.dirname(__file__)

# ── Script list ────────────────────────────────────────────────────────────────
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

# ── Submission ─────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--dry-run", action="store_true",
                        help="Print commands without submitting")
    args = parser.parse_args()

    print(f"{'DRY RUN — ' if args.dry_run else ''}Submitting {len(ALL_SCRIPTS)} jobs from scripts2/\n")

    missing = [s for s in ALL_SCRIPTS
               if not os.path.isfile(os.path.join(SCRIPTS2_DIR, s))]
    if missing:
        print("ERROR: The following scripts were not found in scripts2/:")
        for m in missing:
            print(f"  {m}")
        sys.exit(1)

    groups = [
        ("Base fastrcnn_pretrained", FASTRCNN_BASE),
        ("Base maskrcnn_pretrained", MASKRCNN_BASE),
        ("fastrcnn_pretrained few-shot", FASTRCNN_FEWSHOT),
    ]

    job_ids = []
    for group_name, scripts in groups:
        print(f"── {group_name} ({len(scripts)} scripts) ──")
        for script in scripts:
            path = os.path.join(SCRIPTS2_DIR, script)
            if args.dry_run:
                print(f"  [dry-run] sbatch {path}")
            else:
                result = subprocess.run(["sbatch", path], capture_output=True, text=True)
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
