#!/usr/bin/env python3
"""Build one-off TreeShift dataset summary tables for paper reporting."""

import argparse
import glob
import json
import math
from pathlib import Path
from typing import Dict, Iterable, List

import pandas as pd


GROUP_COLS = [
    "country",
    "state",
    "region",
    "biome",
    "elevation_class_zonewise",
    "zone",
]

NUMERIC_COLS = [
    "trees",
    "elevation_m",
    "pop_density_per_km2",
    "precip_annual_mm",
    "slope_deg",
    "solar_rad_wm2",
    "temp_mean_annual_c",
]


def _clean_metadata(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    if "filename" not in df.columns:
        raise ValueError("metadata must contain a filename column")

    for col in GROUP_COLS:
        if col in df.columns:
            df[col] = df[col].fillna("unknown").astype(str).replace({"": "unknown"})

    for col in NUMERIC_COLS:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors="coerce")

    return df


def _write_group_counts(df: pd.DataFrame, out_dir: Path) -> None:
    for col in GROUP_COLS:
        if col not in df.columns:
            continue
        counts = (
            df.groupby(col, dropna=False)
            .agg(images=("filename", "count"), trees=("trees", "sum") if "trees" in df.columns else ("filename", "count"))
            .reset_index()
            .sort_values(["images", col], ascending=[False, True])
        )
        counts.to_csv(out_dir / f"count_by_{col}.csv", index=False)


def _write_numeric_summary(df: pd.DataFrame, out_dir: Path) -> None:
    rows = []
    for col in NUMERIC_COLS:
        if col not in df.columns:
            continue
        s = df[col].dropna()
        if s.empty:
            continue
        rows.append(
            {
                "variable": col,
                "count": int(s.shape[0]),
                "mean": float(s.mean()),
                "std": float(s.std()),
                "min": float(s.min()),
                "p25": float(s.quantile(0.25)),
                "median": float(s.median()),
                "p75": float(s.quantile(0.75)),
                "max": float(s.max()),
            }
        )
    pd.DataFrame(rows).to_csv(out_dir / "numeric_summary.csv", index=False)


def _iter_coco_bboxes(paths: Iterable[Path]) -> Iterable[Dict[str, float]]:
    for path in paths:
        with path.open() as f:
            data = json.load(f)
        split = path.parent.parent.name
        config = path.parent.parent.parent.name
        for ann in data.get("annotations", []):
            bbox = ann.get("bbox") or []
            if len(bbox) != 4:
                continue
            w = float(bbox[2])
            h = float(bbox[3])
            yield {
                "config": config,
                "split": split,
                "bbox_width": w,
                "bbox_height": h,
                "bbox_area_px": w * h,
            }


def _write_bbox_summary(coco_ann_globs: List[str], out_dir: Path) -> None:
    paths: List[Path] = []
    for pattern in coco_ann_globs:
        paths.extend(Path(p) for p in glob.glob(pattern))
    paths = sorted(set(p for p in paths if p.is_file()))
    if not paths:
        return

    boxes = pd.DataFrame(_iter_coco_bboxes(paths))
    if boxes.empty:
        return
    boxes.to_csv(out_dir / "bbox_annotations.csv", index=False)
    summary = (
        boxes.groupby(["config", "split"], dropna=False)["bbox_area_px"]
        .describe(percentiles=[0.25, 0.5, 0.75])
        .reset_index()
    )
    summary.to_csv(out_dir / "bbox_area_summary.csv", index=False)


def _read_image_array(path: Path):
    try:
        from PIL import Image

        with Image.open(path) as img:
            img = img.convert("L")
            return list(img.getdata())
    except Exception:
        return None


def _image_stats_for_file(path: Path) -> Dict[str, float]:
    vals = _read_image_array(path)
    if not vals:
        return {"brightness_mean": math.nan, "contrast_std": math.nan}
    n = len(vals)
    mean = sum(vals) / n
    var = sum((v - mean) ** 2 for v in vals) / n
    return {"brightness_mean": mean, "contrast_std": math.sqrt(var)}


def _write_image_stats(df: pd.DataFrame, image_root: Path, out_dir: Path) -> None:
    rows = []
    for fn in df["filename"].dropna().astype(str):
        path = image_root / fn
        stats = _image_stats_for_file(path)
        rows.append({"filename": fn, **stats})
    stats_df = pd.DataFrame(rows)
    stats_df.to_csv(out_dir / "image_brightness_contrast.csv", index=False)

    merged = df.merge(stats_df, on="filename", how="left")
    cols = [c for c in ["country", "state", "region", "biome"] if c in merged.columns]
    if cols:
        (
            merged.groupby(cols, dropna=False)[["brightness_mean", "contrast_std"]]
            .agg(["count", "mean", "std", "median"])
            .reset_index()
            .to_csv(out_dir / "image_stats_by_group.csv", index=False)
        )


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--metadata", required=True, help="Path to metadata.csv")
    ap.add_argument("--out-dir", required=True, help="Output directory for CSV summaries")
    ap.add_argument("--image-root", default=None, help="Optional world_images/ directory")
    ap.add_argument("--compute-image-stats", action="store_true", help="Compute brightness/contrast from image files")
    ap.add_argument(
        "--coco-ann-glob",
        action="append",
        default=[],
        help="Optional glob for COCO annotation JSONs, e.g. 'coco_export/*/*/annotations/*.json'",
    )
    args = ap.parse_args()

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    df = _clean_metadata(pd.read_csv(args.metadata))
    df.to_csv(out_dir / "metadata_clean.csv", index=False)
    _write_group_counts(df, out_dir)
    _write_numeric_summary(df, out_dir)
    _write_bbox_summary(args.coco_ann_glob, out_dir)

    if args.compute_image_stats:
        if not args.image_root:
            raise ValueError("--compute-image-stats requires --image-root")
        _write_image_stats(df, Path(args.image_root), out_dir)

    manifest = {
        "metadata": str(args.metadata),
        "n_images": int(df.shape[0]),
        "n_trees": int(df["trees"].sum()) if "trees" in df.columns else None,
        "outputs": sorted(p.name for p in out_dir.iterdir() if p.is_file()),
    }
    (out_dir / "summary_manifest.json").write_text(json.dumps(manifest, indent=2))
    print(json.dumps(manifest, indent=2))


if __name__ == "__main__":
    main()
