#!/usr/bin/env python3
"""
Sample example images from each distribution-shift pair in the dataset.

Reads from the already-exported COCO directory (produced by training scripts)
to avoid hitting the HuggingFace API rate limit. Falls back to a fresh
tree-shift export if a bucket's id_test split is not yet on disk.

Saves 5 images per domain (both sides of each shift) to example_images/<domain>/.
Images are sourced from the id_test split (held-out ID test set) of each config.
Images are chosen to maximise annotation count (≥10 annotations preferred) and
are guaranteed to be unique across all 8 domains.

Each image is saved in two versions:
  <stem>.png           — clean image, no annotations
  <stem>_annotated.png — same image with green bounding boxes overlaid

Output layout:
  example_images/
    rajasthan_wet/  (5 × 2 images)
    rajasthan_dry/  (5 × 2 images)
    karnataka_high/ (5 × 2 images)
    karnataka_low/  (5 × 2 images)
    india/          (5 × 2 images)
    us/             (5 × 2 images)
    north/          (5 × 2 images)
    south/          (5 × 2 images)
    manifest.json

Usage:
    python sample_example_images.py [--coco-dir ./coco_export]
"""

import argparse
import io
import json
import sys
from pathlib import Path

# Patch datasets' internal download threadpool to use a single worker before
# anything else imports it.  The default threadpool fires hundreds of concurrent
# HTTP requests and exhausts HuggingFace's 2500 req/5min rate limit almost
# instantly.  Forcing max_workers=1 makes downloads serial and safe.
import datasets.download.download_manager as _dm
from tqdm.contrib.concurrent import thread_map as _orig_thread_map

def _serial_thread_map(fn, *iterables, **kwargs):
    kwargs["max_workers"] = 1
    return _orig_thread_map(fn, *iterables, **kwargs)

_dm.thread_map = _serial_thread_map

import datasets as hf_ds
from datasets import DownloadConfig
from PIL import Image, ImageDraw

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------

N_IMAGES = 5
MIN_ANNOTATIONS = 10   # preferred minimum; lowered progressively if needed

# Each entry: (bucket_label, config_name)
# id_test is the fixed held-out ID test set for each domain.
BUCKETS = [
    # Pair 1: Rajasthan biome (WET vs DRY)
    ("rajasthan_wet",   "biome_Rajasthan_train_WET__ood_DRY"),
    ("rajasthan_dry",   "biome_Rajasthan_train_DRY__ood_WET"),
    # Pair 2: Karnataka elevation (HIGH vs LOW)
    ("karnataka_high",  "elev_Karnataka_train_HIGH__ood_LOW"),
    ("karnataka_low",   "elev_Karnataka_train_LOW__ood_HIGH"),
    # Pair 3: International (India vs US)
    ("india",           "intl_train_IN__ood_US"),
    ("us",              "intl_train_US__ood_IN"),
    # Pair 4: Region (North vs South India)
    ("north",           "region_train_North__ood_South"),
    ("south",           "region_train_South__ood_North"),
]

HF_REPO = "aadityabuilds/tree-distribution-shift"

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def find_id_test_paths(coco_dir: Path, config: str):
    """Return (ann_file, img_dir) for the id_test split of a config.

    Looks for id_test first, then falls back to val (which is symlinked to
    id_test by ensure_tree_shift_export).  Returns (None, None) if not found.
    """
    base = coco_dir / config
    for split in ("id_test", "val"):
        ann_dir = base / split / "annotations"
        img_dir = base / split / "images"
        if not ann_dir.exists() or not img_dir.exists():
            continue
        candidates = sorted(ann_dir.glob("instances_*.json"))
        if candidates:
            return candidates[0], img_dir
    return None, None


def load_id_test_from_hf(config: str) -> list:
    """Download id_test from HuggingFace with minimal concurrency to avoid rate limits.

    Uses num_proc=1 so files are fetched one at a time (~serial), keeping API
    requests well under the 2500/5min quota even when loading many configs.
    """
    dl_cfg = DownloadConfig(num_proc=1)
    print(f"  Downloading from HuggingFace (num_proc=1 to stay under rate limit) ...")
    ds = hf_ds.load_dataset(
        HF_REPO, config, split="id_test",
        download_config=dl_cfg,
    )
    rows = []
    for row in ds:
        anns = json.loads(row["coco_annotations"])
        rows.append({
            "image_id":    row["image_id"],
            "filename":    row["filename"],
            "width":       row["width"],
            "height":      row["height"],
            "n_ann":       len(anns),
            "annotations": anns,
            "img_bytes":   row["image_bytes"],  # used instead of img_path
            "img_path":    None,
        })
    return rows


def ensure_id_test_exported(coco_dir: Path, config: str) -> tuple:
    """Return (ann_file, img_dir) if the split exists on disk, else None, None."""
    return find_id_test_paths(coco_dir, config)


def load_coco_split(ann_file: Path, img_dir: Path):
    """Load a COCO split into a list of dicts with image + annotation info."""
    with open(ann_file) as f:
        data = json.load(f)

    # Build image_id → annotations map
    ann_by_img = {}
    for ann in data["annotations"]:
        ann_by_img.setdefault(ann["image_id"], []).append(ann)

    rows = []
    for img_info in data["images"]:
        img_id = img_info["id"]
        anns = ann_by_img.get(img_id, [])
        rows.append({
            "image_id":    img_id,
            "filename":    img_info["file_name"],
            "width":       img_info.get("width"),
            "height":      img_info.get("height"),
            "n_ann":       len(anns),
            "annotations": anns,
            "img_path":    img_dir / img_info["file_name"],
        })
    return rows


def pick_images(rows: list, used_ids: set, n: int = N_IMAGES, min_ann: int = MIN_ANNOTATIONS):
    """Sort by annotation count desc, skip used ids, apply progressive threshold."""
    scored = sorted(rows, key=lambda r: r["n_ann"], reverse=True)

    for threshold in [min_ann, 5, 1, 0]:
        selected = [r for r in scored if r["n_ann"] >= threshold and r["image_id"] not in used_ids]
        if len(selected) >= n:
            return selected[:n], threshold

    # Edge case: take whatever is available
    selected = [r for r in scored if r["image_id"] not in used_ids]
    return selected[:n], 0


def draw_boxes(img: Image.Image, annotations: list) -> Image.Image:
    """Draw COCO bounding boxes on a copy of img."""
    out = img.copy().convert("RGBA")
    overlay = Image.new("RGBA", out.size, (0, 0, 0, 0))
    draw = ImageDraw.Draw(overlay)
    for ann in annotations:
        x, y, w, h = ann["bbox"]
        draw.rectangle([x, y, x + w, y + h], outline=(0, 255, 0, 220), width=2)
    return Image.alpha_composite(out, overlay).convert("RGB")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--coco-dir", default="./coco_export",
                        help="Path to tree-shift COCO export directory (default: ./coco_export)")
    parser.add_argument("--out", default="./example_images",
                        help="Output directory (default: ./example_images)")
    args = parser.parse_args()

    coco_dir   = Path(args.coco_dir)
    output_dir = Path(args.out)
    output_dir.mkdir(parents=True, exist_ok=True)

    manifest  = {}
    used_ids: set = set()

    for bucket_label, config in BUCKETS:
        print(f"\n{'='*60}")
        print(f"  Bucket : {bucket_label}")
        print(f"  Config : {config}  |  split: id_test")
        print(f"{'='*60}")

        ann_file, img_dir = ensure_id_test_exported(coco_dir, config)

        if ann_file is not None:
            print(f"  Annotation file : {ann_file}")
            rows = load_coco_split(ann_file, img_dir)
        else:
            print(f"  Not on disk — falling back to HuggingFace download")
            rows = load_id_test_from_hf(config)

        print(f"  Images in split : {len(rows)}")

        selected, threshold = pick_images(rows, used_ids)
        print(f"  Annotation threshold used : ≥{threshold}")
        print(f"  Images selected : {len(selected)}")

        if len(selected) < N_IMAGES:
            print(f"  WARNING: only found {len(selected)} unique images (wanted {N_IMAGES})")

        bucket_dir = output_dir / bucket_label
        bucket_dir.mkdir(parents=True, exist_ok=True)

        bucket_manifest = []
        for row in selected:
            if row.get("img_bytes") is not None:
                import io
                img = Image.open(io.BytesIO(row["img_bytes"])).convert("RGB")
            else:
                img_path = row["img_path"]
                if not img_path.exists():
                    print(f"  WARNING: image file not found: {img_path} — skipping")
                    continue
                img = Image.open(img_path).convert("RGB")
            stem = Path(row["filename"]).stem

            # Clean version
            clean_path = bucket_dir / f"{stem}.png"
            img.save(clean_path, format="PNG")

            # Annotated version
            img_annotated = draw_boxes(img, row["annotations"])
            ann_path = bucket_dir / f"{stem}_annotated.png"
            img_annotated.save(ann_path, format="PNG")

            used_ids.add(row["image_id"])
            print(f"    {stem}.png + {stem}_annotated.png  "
                  f"({row['n_ann']} annotations, {img.width}×{img.height})")

            bucket_manifest.append({
                "image_id":      row["image_id"],
                "clean":         str(clean_path.relative_to(output_dir)),
                "annotated":     str(ann_path.relative_to(output_dir)),
                "source_config": config,
                "source_split":  "id_test",
                "n_annotations": row["n_ann"],
                "width":         img.width,
                "height":        img.height,
            })

        manifest[bucket_label] = bucket_manifest

    # Write manifest
    manifest_path = output_dir / "manifest.json"
    with open(manifest_path, "w") as f:
        json.dump(manifest, f, indent=2)
    print(f"\nManifest saved to {manifest_path}")

    # Summary
    print("\n" + "="*60)
    print("SUMMARY")
    print("="*60)
    all_ids = []
    for label, entries in manifest.items():
        ids = [e["image_id"] for e in entries]
        all_ids.extend(ids)
        counts = [e["n_annotations"] for e in entries]
        print(f"  {label:<20}  {len(entries)} images  "
              f"annotations: {min(counts)}–{max(counts)}")

    duplicates = len(all_ids) - len(set(all_ids))
    if duplicates:
        print(f"\n  WARNING: {duplicates} duplicate image_id(s) across buckets!")
    else:
        print(f"\n  All {len(all_ids)} sampled images are unique across all buckets.")

    print(f"\nImages saved to: {output_dir.resolve()}")


if __name__ == "__main__":
    main()
