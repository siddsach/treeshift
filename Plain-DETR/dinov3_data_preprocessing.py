#!/usr/bin/env python3
"""
Prepare a flat/unlabeled image directory into ImageNet-style structure for DINOv3 SSL pretraining.

This version matches the DINOv3 ImageNet loader expectation observed in your error:
  /train/<class_id>/<class_id>_<index>.JPEG

So it creates:
<OUT>/train/n000/n000_0.JPEG
<OUT>/train/n000/n000_1.JPEG
...
<OUT>/val/n000/n000_0.JPEG
...
<OUT>/test/n000_0.JPEG   (flat by default, like ImageNet-1k)

Also creates:
<OUT>/labels.txt  (two columns, comma-separated by default: class_id,class_name)
and dumps DINOv3 ImageNet extra metadata files into <OUT>/ (or <OUT>/extra if specified).

NOTE:
- Because the ImageNet loader expects .JPEG, we ALWAYS write JPEG outputs (convert if needed).
- That means --mode is effectively ignored (symlink/copy/move), because we must create new JPEG files.
"""

from __future__ import annotations

import argparse
import random
import shutil
import sys
from pathlib import Path
from typing import List, Tuple

from PIL import Image

IMAGE_EXTS_DEFAULT = {
    ".jpg", ".jpeg", ".png", ".bmp", ".webp",
    ".tif", ".tiff",
}


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser()
    p.add_argument("--src", type=str, required=True,
                   help="Source directory containing images in nested subdirectories.")
    p.add_argument("--out", type=str, required=True,
                   help="Output root directory to create ImageNet-style dataset.")
    p.add_argument("--extra", type=str, default=None,
                   help="Extra directory for metadata files. If omitted, uses --out.")
    p.add_argument("--class_id", type=str, default="n000",
                   help="Fake class folder name (default: n000).")
    p.add_argument("--class_name", type=str, default="tree",
                   help="Human-readable class name for labels.txt (default: tree).")
    p.add_argument("--labels_delim", type=str, default=",",
                   help="Delimiter used in labels.txt between class_id and class_name (default: ',').")

    p.add_argument("--split", type=str, default="0.99,0.005,0.005",
                   help="Train,Val,Test fractions (default: 0.99,0.005,0.005).")
    p.add_argument("--seed", type=int, default=42, help="RNG seed for deterministic split.")

    # kept for CLI compatibility, but ignored (we always write JPEGs)
    p.add_argument("--mode", choices=["symlink", "copy", "move"], default="symlink",
                   help="Ignored in this version (we always convert/write JPEG outputs).")

    p.add_argument("--overwrite", action="store_true",
                   help="If set, deletes existing output directory before writing.")
    p.add_argument("--exts", type=str, default="",
                   help="Comma-separated list of allowed extensions (e.g. .jpg,.jpeg,.tif). "
                        "If omitted, uses a default set.")
    p.add_argument("--test_has_classdir", action="store_true",
                   help="If set, places test images under <OUT>/test/<class_id>/ instead of flat <OUT>/test/")
    p.add_argument("--dry_run", action="store_true",
                   help="If set, only prints counts and planned actions; does not write files.")
    p.add_argument("--no_metadata", action="store_true",
                   help="If set, skip generating DINOv3 ImageNet metadata files.")
    return p.parse_args()


def gather_images(src: Path, exts: set[str]) -> List[Path]:
    paths: List[Path] = []
    for p in src.rglob("*"):
        if p.is_file() and p.suffix.lower() in exts:
            paths.append(p)
    return paths


def compute_split_counts(n: int, frac_train: float, frac_val: float, frac_test: float) -> Tuple[int, int, int]:
    n_val = int(round(n * frac_val))
    n_test = int(round(n * frac_test))
    n_train = n - n_val - n_test

    if n >= 200:
        if n_val == 0:
            n_val = 1
            n_train -= 1
        if n_test == 0:
            n_test = 1
            n_train -= 1

    if n_train < 0:
        n_train = max(0, n_train)
        rem = n - n_train
        n_val = rem // 2
        n_test = rem - n_val

    assert n_train + n_val + n_test == n, (n_train, n_val, n_test, n)
    return n_train, n_val, n_test


def ensure_empty_dir(path: Path, overwrite: bool) -> None:
    if path.exists():
        if overwrite:
            shutil.rmtree(path)
        else:
            raise FileExistsError(f"Path already exists: {path} (use --overwrite to delete it)")
    path.mkdir(parents=True, exist_ok=True)


def write_labels_txt(out_root: Path, class_id: str, class_name: str, delim: str = ",") -> None:
    safe_name = class_name.replace(delim, " ").strip()
    (out_root / "labels.txt").write_text(f"{class_id}{delim}{safe_name}\n", encoding="utf-8")


def write_as_jpeg(src: Path, dst: Path) -> None:
    dst.parent.mkdir(parents=True, exist_ok=True)
    with Image.open(src) as im:
        im = im.convert("RGB")
        im.save(dst, format="JPEG", quality=95)


def generate_metadata(out_root: Path, extra_dir: Path) -> None:
    try:
        from dinov3.data.datasets import ImageNet  # type: ignore
    except Exception as e:
        raise RuntimeError(
            "Failed to import dinov3. Run metadata generation inside your dinov3 container/env.\n"
            f"Import error: {e}"
        )

    for split in ImageNet.Split:
        ds = ImageNet(split=split, root=str(out_root), extra=str(extra_dir))
        ds.dump_extra()


def main() -> int:
    args = parse_args()
    src = Path(args.src).expanduser().resolve()
    out_root = Path(args.out).expanduser().resolve()
    extra_dir = Path(args.extra).expanduser().resolve() if args.extra else out_root

    if not src.exists() or not src.is_dir():
        print(f"ERROR: --src does not exist or is not a directory: {src}", file=sys.stderr)
        return 2

    exts = IMAGE_EXTS_DEFAULT if not args.exts.strip() else {
        e.strip().lower() for e in args.exts.split(",") if e.strip()
    }
    if not all(e.startswith(".") for e in exts):
        print("ERROR: All extensions must start with '.', e.g. .jpg,.tif", file=sys.stderr)
        return 2

    images = gather_images(src, exts)
    n = len(images)
    if n == 0:
        print(f"ERROR: Found 0 images under {src} with extensions: {sorted(exts)}", file=sys.stderr)
        return 2

    frac_train, frac_val, frac_test = (float(x) for x in args.split.split(","))
    s = frac_train + frac_val + frac_test
    if abs(s - 1.0) > 1e-6:
        print(f"ERROR: split fractions must sum to 1.0, got {s}", file=sys.stderr)
        return 2

    rng = random.Random(args.seed)
    rng.shuffle(images)

    n_train, n_val, n_test = compute_split_counts(n, frac_train, frac_val, frac_test)
    train_imgs = images[:n_train]
    val_imgs = images[n_train:n_train + n_val]
    test_imgs = images[n_train + n_val:]

    class_id = args.class_id
    class_name = args.class_name

    print("=== Plan ===")
    print(f"Source: {src}")
    print(f"Output root: {out_root}")
    print(f"Extra dir: {extra_dir}")
    print(f"Allowed extensions: {sorted(exts)}")
    print(f"Total images found: {n}")
    print(f"Split counts: train={n_train}, val={n_val}, test={n_test} (seed={args.seed})")
    print(f"Class id: {class_id}")
    print(f"Class name: {class_name}")
    print(f"labels.txt delimiter: {args.labels_delim!r}")
    print(f"Test layout: {'test/<class_id>/' if args.test_has_classdir else 'test/ (flat)'}")
    print("Output naming: <split>/<class_id>/<class_id>_<index>.JPEG (train/val), and test is flat by default")
    print("NOTE: writing JPEG outputs (conversion). --mode is ignored in this version.")
    print(f"Dry run: {args.dry_run}")
    print(f"Generate metadata: {not args.no_metadata}")

    if args.dry_run:
        return 0

    ensure_empty_dir(out_root, overwrite=args.overwrite)

    train_dir = out_root / "train" / class_id
    val_dir = out_root / "val" / class_id
    test_dir = (out_root / "test" / class_id) if args.test_has_classdir else (out_root / "test")

    train_dir.mkdir(parents=True, exist_ok=True)
    val_dir.mkdir(parents=True, exist_ok=True)
    test_dir.mkdir(parents=True, exist_ok=True)

    write_labels_txt(out_root, class_id, class_name, delim=args.labels_delim)

    def place(split_paths: List[Path], dst_dir: Path, start_idx: int = 0) -> int:
        idx = start_idx
        for p in split_paths:
            dst_name = f"{class_id}_{idx}.JPEG"
            dst = dst_dir / dst_name
            write_as_jpeg(p, dst)
            idx += 1
        return idx

    print("Writing train...")
    place(train_imgs, train_dir, start_idx=0)

    print("Writing val...")
    place(val_imgs, val_dir, start_idx=0)

    print("Writing test...")
    place(test_imgs, test_dir, start_idx=0)

    if not args.no_metadata:
        extra_dir.mkdir(parents=True, exist_ok=True)
        print("Generating DINOv3 ImageNet metadata files...")
        generate_metadata(out_root, extra_dir)
        print("Metadata generation complete.")

    print("Done.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
