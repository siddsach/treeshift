#!/usr/bin/env python3
"""
Collect per_image_results.json from all evaluated outputs into a single
organised directory tree.

Output structure:
    <out_dir>/
      <config>/                          # e.g. biome_Rajasthan_train_WET__ood_DRY
        <model>/                         # e.g. fastrcnn_pretrained
          id_test.json
          ood_test.json
          ood_train.json                 # when available
          ood_combined.json              # ood_test + ood_train merged
        <model>__fs1/                    # few-shot variant
          ...

Each JSON file is a list of per-image records as produced by shared_utils.

Usage:
    python scripts2/collect_per_image.py \
        --outputs-dir /path/to/final_outputs \
        --out-dir /path/to/per_result_json
"""

import argparse
import json
import os
import re
import shutil
import sys
from pathlib import Path

# ── Directory parsing (same logic as run_all_evals.py) ────────────────────────

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

SPLIT_MAP = {
    "eval_val":       "id_test.json",
    "eval_ood_test":  "ood_test.json",
    "eval_ood_train": "ood_train.json",
}


def parse_output_dir(dirname):
    """Return (model_type, config, fewshot_suffix) or None."""
    name = dirname
    name = re.sub(r"_\d+$", "", name)
    name = name.replace("_1gpu40ep", "")

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
            fewshot = fs[1:]   # e.g. "fs1"
            shortname = shortname[: -len(fs)]
            break

    base_config = CONFIG_MAP.get(shortname)
    if base_config is None:
        return None

    return model_type, base_config, fewshot


def main():
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument("--outputs-dir", required=True)
    parser.add_argument("--out-dir", default="per_result_json")
    args = parser.parse_args()

    outputs_path = Path(args.outputs_dir)
    out_root = Path(args.out_dir)

    if not outputs_path.exists():
        print(f"ERROR: {args.outputs_dir} not found", file=sys.stderr)
        sys.exit(1)

    copied = 0
    combined = 0
    skipped = 0

    for d in sorted(outputs_path.iterdir()):
        if not d.is_dir():
            continue

        parsed = parse_output_dir(d.name)
        if parsed is None:
            skipped += 1
            continue

        model_type, config, fewshot = parsed
        model_label = f"{model_type}__{fewshot}" if fewshot else model_type
        dest_dir = out_root / config / model_label
        dest_dir.mkdir(parents=True, exist_ok=True)

        ood_parts = []

        for split_subdir, dest_name in SPLIT_MAP.items():
            src = d / split_subdir / "per_image_results.json"
            if not src.exists():
                continue
            shutil.copy2(src, dest_dir / dest_name)
            copied += 1

            if split_subdir in ("eval_ood_test", "eval_ood_train"):
                with open(src) as f:
                    ood_parts.extend(json.load(f))

        if ood_parts:
            with open(dest_dir / "ood_combined.json", "w") as f:
                json.dump(ood_parts, f, indent=2)
            combined += 1

    print(f"Copied   : {copied} per-image JSON files")
    print(f"Combined : {combined} ood_combined.json files")
    print(f"Skipped  : {skipped} unrecognised directories")
    print(f"Output   : {out_root}")


if __name__ == "__main__":
    main()
