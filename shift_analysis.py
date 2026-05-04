#!/usr/bin/env python3
"""
Distribution-shift analysis for the tree-distribution-shift project.

Two subcommands:
    univariate  – Per-covariate regression of ACC_ratio ~ N_ratio.
    shapley     – Shapley-value decomposition of the ID/OOD performance gap.

Both require:
    1.  per_image_results.json files produced by any model's evaluation
        (stored in eval_val/ and eval_ood_test/ by the unified
        ``evaluate_detections`` in shared_utils).
    2.  metadata.csv mapping each image filename to covariate values.

Usage examples:
    python shift_analysis.py univariate \
        --id-results  ./output/eval_val/per_image_results.json \
        --ood-results ./output/eval_ood_test/per_image_results.json \
        --metadata    /path/to/metadata.csv \
        --output-dir  ./shift_analysis_output

    python shift_analysis.py shapley \
        --id-results  ./output/eval_val/per_image_results.json \
        --ood-results ./output/eval_ood_test/per_image_results.json \
        --metadata    /path/to/metadata.csv \
        --output-dir  ./shift_analysis_output
"""

import os
import sys
import json
import argparse
import warnings
from itertools import combinations
from math import factorial

import numpy as np
import pandas as pd

# ── Covariates ────────────────────────────────────────────────────────────

CONTINUOUS_COVARIATES = [
    "pop_density_per_km2",
    "precip_annual_mm",
    "elevation_m",
    "slope_deg",
    "solar_rad_wm2",
    "temp_mean_annual_c",
]

CATEGORICAL_COVARIATES = [
    "soil_texture_class",
    "biome",
]

ALL_COVARIATES = CONTINUOUS_COVARIATES + CATEGORICAL_COVARIATES


# ── Helpers ───────────────────────────────────────────────────────────────

def load_per_image_results(path):
    with open(path) as f:
        return json.load(f)


def load_metadata(path):
    """Load metadata.csv, deduplicating the repeated column groups."""
    df = pd.read_csv(path)
    df = df.loc[:, ~df.columns.duplicated()]
    return df


def merge_results_with_metadata(per_image, metadata_df):
    """Inner-join per-image results with metadata on filename."""
    res_df = pd.DataFrame(per_image)
    meta_dedup = metadata_df.drop_duplicates(subset=["filename"], keep="first")
    if len(meta_dedup) < len(metadata_df):
        print(f"  Warning: metadata had {len(metadata_df) - len(meta_dedup)} duplicate filename rows — kept first.")
    merged = res_df.merge(meta_dedup, on="filename", how="inner")
    n_missed = len(res_df) - len(merged)
    if n_missed > 0:
        print(f"  Warning: {n_missed}/{len(res_df)} images had no metadata match.")
    return merged


def _bin_continuous(series, n_bins, method):
    """Bin a continuous series into discrete levels."""
    if method == "quantile":
        bins = pd.qcut(series, q=n_bins, duplicates="drop")
    else:
        bins = pd.cut(series, bins=n_bins, duplicates="drop")
    return bins


# ── Univariate shift analysis ────────────────────────────────────────────

def univariate_analysis(
    id_df, ood_df, covariates, n_bins=10, bin_method="quantile", output_dir=".",
):
    """For each covariate: bin → N_ratio, ACC_ratio → regress → report.

    Returns a dict mapping covariate → {R2, slope, intercept, p_value}.
    """
    from scipy import stats as sp_stats

    os.makedirs(output_dir, exist_ok=True)
    summary = {}

    for cov in covariates:
        if cov not in id_df.columns or cov not in ood_df.columns:
            print(f"  Skipping {cov}: not found in merged data.")
            continue

        id_vals = id_df[cov].dropna()
        ood_vals = ood_df[cov].dropna()
        if len(id_vals) == 0 or len(ood_vals) == 0:
            print(f"  Skipping {cov}: no valid values.")
            continue

        is_categorical = cov in CATEGORICAL_COVARIATES

        if is_categorical:
            id_copy = id_df.dropna(subset=[cov]).copy()
            ood_copy = ood_df.dropna(subset=[cov]).copy()
            id_copy["_level"] = id_copy[cov].astype(str)
            ood_copy["_level"] = ood_copy[cov].astype(str)
        else:
            combined = pd.concat(
                [id_vals, ood_vals], keys=["id", "ood"],
            )
            if bin_method == "quantile":
                bins = pd.qcut(combined, q=n_bins, duplicates="drop")
            else:
                bins = pd.cut(combined, bins=n_bins, duplicates="drop")

            id_copy = id_df.loc[id_vals.index].copy()
            ood_copy = ood_df.loc[ood_vals.index].copy()
            id_copy["_level"] = bins.loc["id"].values
            ood_copy["_level"] = bins.loc["ood"].values

        id_grouped = id_copy.groupby("_level")
        ood_grouped = ood_copy.groupby("_level")

        levels = sorted(
            set(id_copy["_level"].dropna().unique()) & set(ood_copy["_level"].dropna().unique()),
            key=str,
        )

        n_ratios = []
        acc_ratios = []
        level_labels = []

        for lev in levels:
            id_sub = id_grouped.get_group(lev) if lev in id_grouped.groups else None
            ood_sub = ood_grouped.get_group(lev) if lev in ood_grouped.groups else None
            if id_sub is None or ood_sub is None:
                continue

            n_id = len(id_sub)
            n_ood = len(ood_sub)
            if n_ood == 0:
                continue

            acc_id = id_sub["recall"].mean()
            acc_ood = ood_sub["recall"].mean()
            if acc_ood == 0:
                continue

            n_ratios.append(n_id / n_ood)
            acc_ratios.append(acc_id / acc_ood)
            level_labels.append(str(lev))

        if len(n_ratios) < 3:
            print(f"  Skipping {cov}: fewer than 3 overlapping levels.")
            continue

        n_ratios = np.array(n_ratios)
        acc_ratios = np.array(acc_ratios)

        slope, intercept, r_value, p_value, std_err = sp_stats.linregress(
            n_ratios, acc_ratios
        )
        r2 = r_value ** 2

        summary[cov] = {
            "R2": round(r2, 4),
            "slope": round(slope, 6),
            "intercept": round(intercept, 6),
            "p_value": round(p_value, 6),
            "n_levels": len(n_ratios),
        }

        print(f"  {cov:30s}  R²={r2:.4f}  slope={slope:+.4f}  p={p_value:.4f}  (n={len(n_ratios)} levels)")

        # Scatter plot
        try:
            import matplotlib
            matplotlib.use("Agg")
            import matplotlib.pyplot as plt

            fig, ax = plt.subplots(figsize=(6, 5))
            ax.scatter(n_ratios, acc_ratios, s=40, zorder=3)
            xline = np.linspace(n_ratios.min(), n_ratios.max(), 50)
            ax.plot(xline, slope * xline + intercept, "r--", linewidth=1.5,
                    label=f"R²={r2:.3f}  p={p_value:.3f}")
            ax.set_xlabel("N_ID / N_OOD  (sample-size ratio)")
            ax.set_ylabel("Recall_ID / Recall_OOD  (accuracy ratio)")
            ax.set_title(f"Univariate shift: {cov}")
            ax.legend()
            ax.grid(True, alpha=0.3)
            fig.tight_layout()
            fig.savefig(os.path.join(output_dir, f"univariate_{cov}.png"), dpi=150)
            plt.close(fig)
        except ImportError:
            pass

    summary_path = os.path.join(output_dir, "univariate_summary.json")
    with open(summary_path, "w") as f:
        json.dump(summary, f, indent=2)
    print(f"\nUnivariate summary saved to: {summary_path}")
    return summary


# ── Shapley decomposition ────────────────────────────────────────────────

def _reweight_recall(ood_df, id_df, covariate_subset, n_bins=10):
    """Compute importance-weighted mean recall for OOD images.

    Weights are chosen so that the joint distribution of *covariate_subset*
    in the reweighted OOD matches the ID distribution.  We estimate
    densities via histogram binning.

    Returns the reweighted mean recall (float).
    """
    if len(covariate_subset) == 0:
        return ood_df["recall"].mean()

    cov_list = list(covariate_subset)

    id_work = id_df[cov_list + ["recall"]].dropna().copy()
    ood_work = ood_df[cov_list + ["recall"]].dropna().copy()

    if len(id_work) == 0 or len(ood_work) == 0:
        return ood_df["recall"].mean()

    # Discretise each covariate
    for cov in cov_list:
        if cov in CATEGORICAL_COVARIATES:
            id_work[cov] = id_work[cov].astype(str)
            ood_work[cov] = ood_work[cov].astype(str)
        else:
            combined = pd.concat(
                [id_work[cov], ood_work[cov]], keys=["id", "ood"],
            )
            bins = pd.qcut(combined, q=min(n_bins, len(combined)), duplicates="drop")
            id_work[cov] = bins.loc["id"].values
            ood_work[cov] = bins.loc["ood"].values

    # Joint cell key
    id_work["_cell"] = id_work[cov_list].astype(str).agg("|".join, axis=1)
    ood_work["_cell"] = ood_work[cov_list].astype(str).agg("|".join, axis=1)

    id_counts = id_work["_cell"].value_counts()
    ood_counts = ood_work["_cell"].value_counts()

    id_total = id_counts.sum()
    ood_total = ood_counts.sum()

    id_freq = id_counts / id_total
    ood_freq = ood_counts / ood_total

    # Importance weight per cell = P_ID(cell) / P_OOD(cell)
    ratio = id_freq / ood_freq
    ratio = ratio.clip(upper=10.0)

    ood_work["_w"] = ood_work["_cell"].map(ratio).fillna(0.0)

    w_sum = ood_work["_w"].sum()
    if w_sum == 0:
        return ood_df["recall"].mean()

    return float((ood_work["recall"] * ood_work["_w"]).sum() / w_sum)


def shapley_decomposition(id_df, ood_df, covariates, n_bins=10, output_dir="."):
    """Exact Shapley-value attribution of the ID-OOD recall gap.

    For n covariates we enumerate all 2^n subsets (feasible for n<=10).

    Returns dict with per-covariate Shapley values, total gap, and
    explained / unexplained breakdown.
    """
    os.makedirs(output_dir, exist_ok=True)
    n = len(covariates)

    recall_id = id_df["recall"].mean()
    recall_ood = ood_df["recall"].mean()
    total_gap = recall_id - recall_ood
    print(f"  Recall ID  = {recall_id:.4f}")
    print(f"  Recall OOD = {recall_ood:.4f}")
    print(f"  Total gap  = {total_gap:+.4f}")

    # Pre-compute v(S) for every subset S
    subset_values = {}
    for size in range(n + 1):
        for combo in combinations(range(n), size):
            cov_subset = tuple(covariates[i] for i in combo)
            v = _reweight_recall(ood_df, id_df, cov_subset, n_bins=n_bins)
            subset_values[frozenset(combo)] = v

    # Shapley formula (exact)
    shapley_vals = {}
    for i in range(n):
        phi = 0.0
        others = [j for j in range(n) if j != i]
        for size in range(n):
            for combo in combinations(others, size):
                s = frozenset(combo)
                s_with_i = s | {i}
                weight = factorial(len(s)) * factorial(n - len(s) - 1) / factorial(n)
                marginal = subset_values[s_with_i] - subset_values[s]
                phi += weight * marginal
        shapley_vals[covariates[i]] = phi

    # The Shapley values sum to v(all) - v(empty)
    v_all = subset_values[frozenset(range(n))]
    explained = v_all - recall_ood
    unexplained = total_gap - explained

    result = {
        "recall_id": round(recall_id, 6),
        "recall_ood": round(recall_ood, 6),
        "total_gap": round(total_gap, 6),
        "explained": round(explained, 6),
        "unexplained": round(unexplained, 6),
        "shapley_values": {k: round(v, 6) for k, v in shapley_vals.items()},
    }

    print(f"\n  Explained (by covariates) = {explained:+.4f}")
    print(f"  Unexplained              = {unexplained:+.4f}")
    print(f"\n  Shapley values:")
    for cov, val in sorted(shapley_vals.items(), key=lambda x: -abs(x[1])):
        print(f"    {cov:30s}  {val:+.6f}")

    # Save
    out_path = os.path.join(output_dir, "shapley_results.json")
    with open(out_path, "w") as f:
        json.dump(result, f, indent=2)
    print(f"\nShapley results saved to: {out_path}")

    # Bar chart
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt

        names = list(shapley_vals.keys())
        vals = [shapley_vals[nm] for nm in names]
        order = np.argsort(np.abs(vals))[::-1]
        names = [names[i] for i in order]
        vals = [vals[i] for i in order]

        fig, axes = plt.subplots(1, 2, figsize=(14, 5))

        # Bar chart of Shapley values
        colors = ["#2196F3" if v >= 0 else "#F44336" for v in vals]
        axes[0].barh(names, vals, color=colors)
        axes[0].set_xlabel("Shapley value (contribution to reweighted gap)")
        axes[0].set_title("Covariate Shapley Values")
        axes[0].axvline(0, color="black", linewidth=0.5)
        axes[0].invert_yaxis()

        # Pie chart: explained vs unexplained
        pie_vals = [abs(explained), abs(unexplained)]
        pie_labels = [
            f"Explained\n({explained:+.4f})",
            f"Unexplained\n({unexplained:+.4f})",
        ]
        axes[1].pie(pie_vals, labels=pie_labels, autopct="%1.1f%%",
                     colors=["#4CAF50", "#FF9800"], startangle=90)
        axes[1].set_title(f"Gap Decomposition (total={total_gap:+.4f})")

        fig.tight_layout()
        fig.savefig(os.path.join(output_dir, "shapley_decomposition.png"), dpi=150)
        plt.close(fig)
    except ImportError:
        pass

    return result


# ── CLI ───────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(
        description="Distribution-shift analysis for tree detection."
    )
    sub = parser.add_subparsers(dest="command")

    # -- univariate --
    uni = sub.add_parser("univariate", help="Per-covariate ACC_ratio ~ N_ratio regression")
    uni.add_argument("--id-results", required=True, help="per_image_results.json for ID (val) split")
    uni.add_argument("--ood-results", required=True, help="per_image_results.json for OOD split")
    uni.add_argument("--metadata", required=True, help="Path to metadata.csv")
    uni.add_argument("--covariates", default=None,
                     help="Comma-separated covariate names (default: all 8)")
    uni.add_argument("--n-bins", type=int, default=10, help="Number of bins for continuous covariates")
    uni.add_argument("--bin-method", choices=["quantile", "equal_width"], default="quantile")
    uni.add_argument("--output-dir", default="./shift_analysis_output")

    # -- shapley --
    shp = sub.add_parser("shapley", help="Shapley decomposition of ID/OOD gap")
    shp.add_argument("--id-results", required=True, help="per_image_results.json for ID (val) split")
    shp.add_argument("--ood-results", required=True, help="per_image_results.json for OOD split")
    shp.add_argument("--metadata", required=True, help="Path to metadata.csv")
    shp.add_argument("--covariates", default=None,
                     help="Comma-separated covariate names (default: all 8)")
    shp.add_argument("--n-bins", type=int, default=10, help="Number of bins for reweighting")
    shp.add_argument("--output-dir", default="./shift_analysis_output")

    args = parser.parse_args()
    if args.command is None:
        parser.print_help()
        sys.exit(1)

    covariates = ALL_COVARIATES
    if args.covariates:
        covariates = [c.strip() for c in args.covariates.split(",")]

    print(f"Loading metadata from {args.metadata} ...")
    meta_df = load_metadata(args.metadata)
    print(f"  {len(meta_df)} rows, columns: {list(meta_df.columns)}")

    print(f"Loading ID results from {args.id_results} ...")
    id_raw = load_per_image_results(args.id_results)
    print(f"  {len(id_raw)} images")

    print(f"Loading OOD results from {args.ood_results} ...")
    ood_raw = load_per_image_results(args.ood_results)
    print(f"  {len(ood_raw)} images")

    print("Merging with metadata ...")
    id_df = merge_results_with_metadata(id_raw, meta_df)
    ood_df = merge_results_with_metadata(ood_raw, meta_df)
    print(f"  ID merged: {len(id_df)} images")
    print(f"  OOD merged: {len(ood_df)} images")

    if args.command == "univariate":
        print(f"\n{'=' * 60}")
        print("UNIVARIATE SHIFT ANALYSIS")
        print(f"{'=' * 60}")
        univariate_analysis(
            id_df, ood_df,
            covariates=covariates,
            n_bins=args.n_bins,
            bin_method=args.bin_method,
            output_dir=args.output_dir,
        )

    elif args.command == "shapley":
        available = [c for c in covariates if c in id_df.columns and c in ood_df.columns]
        if len(available) < len(covariates):
            missing = set(covariates) - set(available)
            print(f"  Warning: dropping covariates not in data: {missing}")
        covariates = available

        print(f"\n{'=' * 60}")
        print(f"SHAPLEY DECOMPOSITION  ({len(covariates)} covariates, {2**len(covariates)} subsets)")
        print(f"{'=' * 60}")
        shapley_decomposition(
            id_df, ood_df,
            covariates=covariates,
            n_bins=args.n_bins,
            output_dir=args.output_dir,
        )


if __name__ == "__main__":
    main()
