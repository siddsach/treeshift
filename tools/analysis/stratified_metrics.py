#!/usr/bin/env python3
"""Compute per-image detection metrics stratified by metadata groups or bins."""

import argparse
import json
from pathlib import Path
from typing import List

import pandas as pd


def _load_results(paths: List[str]) -> pd.DataFrame:
    frames = []
    for path in paths:
        with Path(path).open() as f:
            data = json.load(f)
        if isinstance(data, dict) and "per_image" in data:
            data = data["per_image"]
        df = pd.DataFrame(data)
        df["result_file"] = path
        frames.append(df)
    return pd.concat(frames, ignore_index=True)


def _micro_metrics(g: pd.DataFrame) -> pd.Series:
    tp = g["tp"].sum()
    fp = g["fp"].sum()
    fn = g["fn"].sum()
    return pd.Series(
        {
            "n_images": int(len(g)),
            "tp": int(tp),
            "fp": int(fp),
            "fn": int(fn),
            "micro_precision": tp / (tp + fp) if (tp + fp) else 0.0,
            "micro_recall": tp / (tp + fn) if (tp + fn) else 0.0,
            "mean_precision": g["precision"].mean(),
            "mean_recall": g["recall"].mean(),
            "mean_n_gt": g["n_gt"].mean(),
            "mean_n_pred": g["n_pred"].mean(),
        }
    )


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--metadata", required=True)
    ap.add_argument("--results", nargs="+", required=True)
    ap.add_argument("--out-csv", required=True)
    ap.add_argument("--groups", nargs="+", default=["country", "state", "region", "biome"])
    ap.add_argument("--bin", dest="bins", action="append", default=[], help="Numeric bin spec: column:num_bins")
    args = ap.parse_args()

    meta = pd.read_csv(args.metadata)
    df = _load_results(args.results).merge(meta, on="filename", how="left")

    out_rows = []
    for group in args.groups:
        if group not in df.columns:
            continue
        summary = df.groupby(group, dropna=False).apply(_micro_metrics).reset_index()
        summary.insert(0, "stratum", group)
        summary = summary.rename(columns={group: "value"})
        out_rows.append(summary)

    for spec in args.bins:
        col, n_s = spec.split(":", 1)
        n_bins = int(n_s)
        if col not in df.columns:
            continue
        values = pd.to_numeric(df[col], errors="coerce")
        binned = df.copy()
        binned["_bin"] = pd.qcut(values, q=n_bins, duplicates="drop")
        summary = binned.groupby("_bin", dropna=False).apply(_micro_metrics).reset_index()
        summary.insert(0, "stratum", col)
        summary = summary.rename(columns={"_bin": "value"})
        summary["value"] = summary["value"].astype(str)
        out_rows.append(summary)

    if out_rows:
        out = pd.concat(out_rows, ignore_index=True)
    else:
        out = pd.DataFrame()
    Path(args.out_csv).parent.mkdir(parents=True, exist_ok=True)
    out.to_csv(args.out_csv, index=False)
    print(f"Wrote {len(out)} rows to {args.out_csv}")


if __name__ == "__main__":
    main()
