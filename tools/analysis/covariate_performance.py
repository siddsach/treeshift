#!/usr/bin/env python3
"""Join per-image detection results with metadata and summarize covariates."""

import argparse
import json
from pathlib import Path
from typing import Dict, List

import pandas as pd


DEFAULT_GROUPS = ["country", "state", "region", "biome", "elevation_class_zonewise", "zone"]
DEFAULT_NUMERIC = [
    "trees",
    "elevation_m",
    "pop_density_per_km2",
    "precip_annual_mm",
    "slope_deg",
    "solar_rad_wm2",
    "temp_mean_annual_c",
]
METRICS = ["tp", "fp", "fn", "n_gt", "n_pred", "recall", "precision"]


def _load_results(path: Path) -> pd.DataFrame:
    with path.open() as f:
        data = json.load(f)
    if isinstance(data, dict) and "per_image" in data:
        data = data["per_image"]
    df = pd.DataFrame(data)
    if "filename" not in df.columns:
        raise ValueError(f"{path} does not contain per-image filename records")
    df["result_file"] = str(path)
    return df


def _safe_corr(df: pd.DataFrame, x: str, y: str) -> float:
    sub = df[[x, y]].dropna()
    if len(sub) < 3:
        return float("nan")
    return float(sub[x].corr(sub[y], method="spearman"))


def _summarize_groups(df: pd.DataFrame, group_cols: List[str]) -> pd.DataFrame:
    rows = []
    for group in group_cols:
        if group not in df.columns:
            continue
        grouped = df.groupby(group, dropna=False)
        for value, g in grouped:
            tp = g["tp"].sum() if "tp" in g.columns else 0
            fp = g["fp"].sum() if "fp" in g.columns else 0
            fn = g["fn"].sum() if "fn" in g.columns else 0
            rows.append(
                {
                    "group": group,
                    "value": value,
                    "n_images": int(len(g)),
                    "tp": int(tp),
                    "fp": int(fp),
                    "fn": int(fn),
                    "micro_precision": tp / (tp + fp) if (tp + fp) else 0.0,
                    "micro_recall": tp / (tp + fn) if (tp + fn) else 0.0,
                    "mean_precision": float(g["precision"].mean()) if "precision" in g.columns else None,
                    "mean_recall": float(g["recall"].mean()) if "recall" in g.columns else None,
                }
            )
    return pd.DataFrame(rows)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--metadata", required=True)
    ap.add_argument("--results", nargs="+", required=True, help="One or more per_image_results.json files")
    ap.add_argument("--out-dir", required=True)
    ap.add_argument("--group-cols", nargs="*", default=DEFAULT_GROUPS)
    ap.add_argument("--numeric-cols", nargs="*", default=DEFAULT_NUMERIC)
    args = ap.parse_args()

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    meta = pd.read_csv(args.metadata)
    results = pd.concat([_load_results(Path(p)) for p in args.results], ignore_index=True)
    joined = results.merge(meta, on="filename", how="left", suffixes=("", "_meta"))
    joined.to_csv(out_dir / "per_image_with_metadata.csv", index=False)

    group_summary = _summarize_groups(joined, args.group_cols)
    group_summary.to_csv(out_dir / "metrics_by_group.csv", index=False)

    corr_rows: List[Dict[str, object]] = []
    metric_cols = [c for c in ["recall", "precision", "tp", "fp", "fn", "n_gt", "n_pred"] if c in joined.columns]
    for covar in args.numeric_cols:
        if covar not in joined.columns:
            continue
        joined[covar] = pd.to_numeric(joined[covar], errors="coerce")
        for metric in metric_cols:
            corr_rows.append(
                {
                    "covariate": covar,
                    "metric": metric,
                    "spearman_corr": _safe_corr(joined, covar, metric),
                    "n": int(joined[[covar, metric]].dropna().shape[0]),
                }
            )
    pd.DataFrame(corr_rows).to_csv(out_dir / "metric_covariate_correlations.csv", index=False)

    manifest = {
        "metadata": args.metadata,
        "results": args.results,
        "n_rows": int(joined.shape[0]),
        "outputs": ["per_image_with_metadata.csv", "metrics_by_group.csv", "metric_covariate_correlations.csv"],
    }
    (out_dir / "analysis_manifest.json").write_text(json.dumps(manifest, indent=2))
    print(json.dumps(manifest, indent=2))


if __name__ == "__main__":
    main()
