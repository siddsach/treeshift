#!/usr/bin/env python3
"""Compile corrected detector per-image outputs into split and stratified CSVs."""

import argparse
import csv
import json
import math
import os
from collections import defaultdict
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional, Tuple


GROUPS = ("country", "state", "region", "zone", "biome")

CORRECTED_DETECTRON_RUNS = {
    "fastrcnn_pretrained_india_random_80_20_23870600_best_t03": ("eval_val",),
    "fastrcnn_pretrained_region_North_South_23870631_best_t03": (
        "eval_val",
        "eval_ood_test",
        "eval_ood_train",
    ),
    "fastrcnn_pretrained_region_South_North_23870772_best_t03": (
        "eval_val",
        "eval_ood_test",
        "eval_ood_train",
    ),
    "maskrcnn_pretrained_india_random_80_20_23870601_best_t03": ("eval_val",),
    "maskrcnn_pretrained_region_North_South_23870632_best_t03": (
        "eval_val",
        "eval_ood_test",
        "eval_ood_train",
    ),
    "maskrcnn_pretrained_region_South_North_23870773_best_t03": (
        "eval_val",
        "eval_ood_test",
        "eval_ood_train",
    ),
}

PLAINTDETR_RUNS = {
    "plaindetr_resnet_india_random_80_20_23868507": ("eval_val",),
    "plaindetr_resnet_region_North_South_23870673": (
        "eval_val",
        "eval_ood_test",
        "eval_ood_train",
    ),
    "plaindetr_resnet_region_South_North_23870776": (
        "eval_val",
        "eval_ood_test",
        "eval_ood_train",
    ),
}

SPLIT_NAMES = {
    "eval_val": "val",
    "eval_ood_test": "ood_test",
    "eval_ood_train": "ood_train",
}

SPLIT_COLUMNS = (
    "run_name",
    "model",
    "job_id",
    "split",
    "n_images",
    "tp",
    "fp",
    "fn",
    "micro_precision",
    "micro_recall",
    "mean_precision",
    "mean_recall",
    "mean_n_gt",
    "mean_n_pred",
    "count_r2",
    "count_mae",
    "count_rmse",
    "count_bias",
    "result_file",
)

STRATIFIED_COLUMNS = (
    "run_name",
    "model",
    "job_id",
    "split",
    "stratum",
    "value",
    "n_images",
    "tp",
    "fp",
    "fn",
    "micro_precision",
    "micro_recall",
    "mean_precision",
    "mean_recall",
    "mean_n_gt",
    "mean_n_pred",
    "count_r2",
    "count_mae",
    "count_rmse",
    "count_bias",
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--repo-root",
        default="/scratch/groups/dlobell/siddsach/treeshift",
        help="Treeshift repo root.",
    )
    parser.add_argument(
        "--corrected-root",
        default=None,
        help="Root containing corrected Detectron eval output directories.",
    )
    parser.add_argument(
        "--metadata",
        default="/scratch/groups/dlobell/aadityan/dataset/metadata.csv",
        help="Metadata CSV with filename and stratification columns.",
    )
    parser.add_argument(
        "--output-dir",
        default=None,
        help="Directory for compiled CSVs and manifest.",
    )
    parser.add_argument(
        "--no-plaindetr",
        action="store_true",
        help="Do not include existing Plain-DETR ResNet outputs.",
    )
    return parser.parse_args()


def safe_div(num, den):
    if den == 0:
        return None
    return num / den


def mean(values):
    if not values:
        return None
    return sum(values) / len(values)


def corr_squared(xs, ys):
    if len(xs) < 2 or len(xs) != len(ys):
        return None
    mx = sum(xs) / len(xs)
    my = sum(ys) / len(ys)
    dx = [x - mx for x in xs]
    dy = [y - my for y in ys]
    ssx = sum(x * x for x in dx)
    ssy = sum(y * y for y in dy)
    if ssx == 0 or ssy == 0:
        return None
    cov = sum(x * y for x, y in zip(dx, dy))
    return (cov * cov) / (ssx * ssy)


def fmt(value):
    if value is None:
        return ""
    if isinstance(value, float):
        return f"{value:.6f}"
    return value


def load_metadata(path):
    out = {}
    with path.open(newline="") as f:
        for row in csv.DictReader(f):
            filename = row.get("filename")
            if filename:
                out[filename] = {group: row.get(group, "") for group in GROUPS}
    return out


def derive_model_and_job(run_name):
    base = run_name[:-9] if run_name.endswith("_best_t03") else run_name
    prefix, _, suffix = base.rpartition("_")
    if suffix.isdigit():
        return prefix, suffix
    return base, ""


def load_result_file(path, split_dir):
    with path.open() as f:
        records = json.load(f)
    split = SPLIT_NAMES.get(split_dir, split_dir[5:] if split_dir.startswith("eval_") else split_dir)
    for rec in records:
        rec.setdefault("split", split)
    return records


def summarize(records):
    n_images = len(records)
    tp = sum(int(rec.get("tp", 0)) for rec in records)
    fp = sum(int(rec.get("fp", 0)) for rec in records)
    fn = sum(int(rec.get("fn", 0)) for rec in records)
    n_gt = [float(rec.get("n_gt", 0)) for rec in records]
    n_pred = [float(rec.get("n_pred", 0)) for rec in records]
    diffs = [pred - gt for gt, pred in zip(n_gt, n_pred)]
    precisions = [
        float(rec["precision"])
        for rec in records
        if rec.get("precision") is not None
    ]
    recalls = [
        float(rec["recall"])
        for rec in records
        if rec.get("recall") is not None
    ]

    return {
        "n_images": n_images,
        "tp": tp,
        "fp": fp,
        "fn": fn,
        "micro_precision": safe_div(tp, tp + fp),
        "micro_recall": safe_div(tp, tp + fn),
        "mean_precision": mean(precisions),
        "mean_recall": mean(recalls),
        "mean_n_gt": mean(n_gt),
        "mean_n_pred": mean(n_pred),
        "count_r2": corr_squared(n_gt, n_pred),
        "count_mae": mean([abs(diff) for diff in diffs]),
        "count_rmse": math.sqrt(mean([diff * diff for diff in diffs])) if diffs else None,
        "count_bias": mean(diffs),
    }


def result_files(repo_root, corrected_root, include_plaindetr):
    files = []
    for run_name, split_dirs in CORRECTED_DETECTRON_RUNS.items():
        for split_dir in split_dirs:
            files.append((run_name, split_dir, corrected_root / run_name / split_dir / "per_image_results.json"))

    if include_plaindetr:
        outputs_root = repo_root / "outputs"
        for run_name, split_dirs in PLAINTDETR_RUNS.items():
            for split_dir in split_dirs:
                files.append((run_name, split_dir, outputs_root / run_name / split_dir / "per_image_results.json"))

    return files


def main() -> None:
    args = parse_args()
    repo_root = Path(args.repo_root)
    corrected_root = (
        Path(args.corrected_root)
        if args.corrected_root
        else repo_root / "outputs_corrected" / "detection_eval_v2_t03_best"
    )
    output_dir = (
        Path(args.output_dir)
        if args.output_dir
        else repo_root / "analysis_outputs" / "detection_stratified_20260507_corrected_t03_best"
    )
    output_dir.mkdir(parents=True, exist_ok=True)

    metadata_path = Path(args.metadata)
    metadata = load_metadata(metadata_path)
    files = result_files(repo_root, corrected_root, not args.no_plaindetr)
    missing = [str(path) for _, _, path in files if not path.exists()]
    if missing:
        raise FileNotFoundError("Missing expected per-image result files:\n" + "\n".join(missing))

    split_rows = []
    stratified_rows = []
    missing_metadata_records = 0
    files_used = []

    for run_name, split_dir, path in files:
        model, job_id = derive_model_and_job(run_name)
        split = SPLIT_NAMES.get(split_dir, split_dir[5:] if split_dir.startswith("eval_") else split_dir)
        records = load_result_file(path, split_dir)
        files_used.append(str(path))

        row = {
            "run_name": run_name,
            "model": model,
            "job_id": job_id,
            "split": split,
            "result_file": str(path),
            **summarize(records),
        }
        split_rows.append(row)

        by_group_value = defaultdict(list)
        for rec in records:
            meta = metadata.get(str(rec.get("filename", "")))
            if meta is None:
                missing_metadata_records += 1
                continue
            for group in GROUPS:
                value = meta.get(group, "")
                if value:
                    by_group_value[(group, value)].append(rec)

        for (group, value), group_records in sorted(by_group_value.items()):
            stratified_rows.append(
                {
                    "run_name": run_name,
                    "model": model,
                    "job_id": job_id,
                    "split": split,
                    "stratum": group,
                    "value": value,
                    **summarize(group_records),
                }
            )

    with (output_dir / "detection_split_metrics.csv").open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=SPLIT_COLUMNS)
        writer.writeheader()
        for row in split_rows:
            writer.writerow({column: fmt(row.get(column)) for column in SPLIT_COLUMNS})

    with (output_dir / "detection_stratified_metrics.csv").open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=STRATIFIED_COLUMNS)
        writer.writeheader()
        for row in stratified_rows:
            writer.writerow({column: fmt(row.get(column)) for column in STRATIFIED_COLUMNS})

    manifest = {
        "created_at": datetime.now().isoformat(timespec="seconds"),
        "metadata_path": str(metadata_path),
        "corrected_root": str(corrected_root),
        "groups": list(GROUPS),
        "run_prefixes": [
            "fastrcnn_pretrained_",
            "maskrcnn_pretrained_",
            "plaindetr_resnet_",
        ],
        "n_run_dirs_seen": len({str(path.parents[1]) for _, _, path in files}),
        "n_result_files_used": len(files_used),
        "n_split_rows": len(split_rows),
        "n_stratified_rows": len(stratified_rows),
        "missing_metadata_records": missing_metadata_records,
        "files_used": files_used,
        "notes": (
            "Corrected Detectron metrics use best validation checkpoint, COCO/AP prediction "
            "generation at score threshold 0.05, and tree/count metrics filtered at 0.30. "
            "Plain-DETR ResNet rows are the existing completed runs for comparison."
        ),
    }
    with (output_dir / "manifest.json").open("w") as f:
        json.dump(manifest, f, indent=2)

    print(f"Wrote {output_dir / 'detection_split_metrics.csv'}")
    print(f"Wrote {output_dir / 'detection_stratified_metrics.csv'}")
    print(f"Wrote {output_dir / 'manifest.json'}")


if __name__ == "__main__":
    main()
