#!/usr/bin/env python3
"""
Submit the four baseline Plain-DETR + DinoV3-sat jobs for the four base configs.

Uses scripts2/ scripts: 1 GPU, batch-size 1, mem=32G, serc,gpu partition.
DinoV3 backbone: dinov3_vit7b16 with dino_weights_sat.pth.

Run from the HPC login node (from any directory):
    python scripts2/run_baseline_plaindetr_sat.py
    python scripts2/run_baseline_plaindetr_sat.py --dry-run
"""

import subprocess
import os
from pathlib import Path

REPO_ROOT = Path("/scratch/groups/dlobell/aadityan/tree-distribution-shift")
SCRIPTS2_DIR = REPO_ROOT / "scripts2"

# Four base configs (shortname → full config name, n_train)
BASE_CONFIGS = {
    "biome_Rajasthan_WET_DRY":    ("biome_Rajasthan_train_WET__ood_DRY",  1714),
    "region_North_South":          ("region_train_North__ood_South",        6118),
    "intl_US_IN":                  ("intl_train_US__ood_IN",                8923),
    "elev_Karnataka_HIGH_LOW":     ("elev_Karnataka_train_HIGH__ood_LOW",   2351),
}


def submit_baseline_plaindetr_sat(dry_run: bool = False) -> dict:
    """
    Submit Plain-DETR + DinoV3-sat baseline jobs for the four base configs
    via the scripts2/ SBATCH scripts (1 GPU, batch=1, mem=32G, gpu partition).

    Parameters
    ----------
    dry_run : bool
        If True, print the sbatch commands without actually submitting.

    Returns
    -------
    dict
        Maps config shortname → SLURM job ID (or -1 in dry-run mode).
    """
    job_ids = {}

    for short, (full_config, n_train) in BASE_CONFIGS.items():
        script = SCRIPTS2_DIR / f"plaindetr_dinov3_sat_{short}.sh"

        if not script.exists():
            raise FileNotFoundError(
                f"Script not found: {script}\n"
                f"Expected scripts2/plaindetr_dinov3_sat_{short}.sh"
            )

        print(f"  {short}")
        print(f"    config  : {full_config}  ({n_train} train images)")
        print(f"    script  : {script.name}")

        if dry_run:
            print(f"    command : sbatch {script}")
            job_ids[short] = -1
        else:
            result = subprocess.run(
                ["sbatch", str(script)],
                capture_output=True,
                text=True,
                check=True,
                cwd=str(REPO_ROOT),
            )
            # sbatch stdout: "Submitted batch job <JOB_ID>"
            job_id = int(result.stdout.strip().split()[-1])
            job_ids[short] = job_id
            print(f"    job ID  : {job_id}")

        print()

    return job_ids


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(
        description="Submit 4 baseline Plain-DETR + DinoV3-sat jobs (scripts2)."
    )
    parser.add_argument(
        "--dry-run", action="store_true",
        help="Print sbatch commands without submitting.",
    )
    args = parser.parse_args()

    dino_weights = REPO_ROOT.parent / "dino_weights" / "dino_weights_sat.pth"
    image_sif    = REPO_ROOT / "tree-shift.sif"

    print("=== Baseline Plain-DETR + DinoV3-sat (scripts2) ===")
    print(f"Repo      : {REPO_ROOT}")
    print(f"Scripts   : {SCRIPTS2_DIR}")
    print(f"Weights   : {dino_weights}")
    print(f"Container : {image_sif}")
    print(f"GPU       : 1 × gpu partition  (batch=1, mem=32G)")
    print(f"Backbone  : dinov3_vit7b16 + dino_weights_sat.pth")
    print()

    if not args.dry_run:
        if not image_sif.exists():
            raise FileNotFoundError(f"Apptainer image not found: {image_sif}")
        if not dino_weights.exists():
            raise FileNotFoundError(f"DinoV3-sat weights not found: {dino_weights}")

    ids = submit_baseline_plaindetr_sat(dry_run=args.dry_run)

    if not args.dry_run:
        print("Submitted jobs:")
        for config, jid in ids.items():
            print(f"  {config:<40s}  job {jid}")
        print()
        print("Monitor with:")
        job_list = " ".join(str(j) for j in ids.values())
        print(f"  squeue -j {job_list}")
        print(f"  tail -f logs/plaindetr_dinov3_sat_*_<JOB_ID>.out")
