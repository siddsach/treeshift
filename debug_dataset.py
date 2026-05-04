#!/usr/bin/env python3
"""
Debug script for tree-distribution-shift COCO export.
Run in Colab (or locally) to inspect dataset, images, and annotations.

Usage:
    python debug_dataset.py --config biome_Rajasthan_train_WET__ood_DRY --coco-dir ./coco_export
"""
import argparse
import json
import os
from pathlib import Path


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="biome_Rajasthan_train_WET__ood_DRY")
    parser.add_argument("--coco-dir", default="./coco_export")
    parser.add_argument("--split", default="val")
    parser.add_argument("--n-samples", type=int, default=3)
    parser.add_argument("--save-viz", action="store_true", help="Save images with bbox overlays")
    args = parser.parse_args()

    base = Path(args.coco_dir) / args.config / args.split
    ann_file = base / "annotations" / f"instances_{args.split}.json"
    img_dir = base / "images"

    if not ann_file.exists():
        print(f"Error: {ann_file} not found")
        return
    if not img_dir.exists():
        print(f"Error: {img_dir} not found")
        return

    with open(ann_file) as f:
        data = json.load(f)

    print("=" * 60)
    print("COCO ANNOTATION SUMMARY")
    print("=" * 60)
    print(f"Images: {len(data['images'])}")
    print(f"Annotations: {len(data['annotations'])}")
    print(f"Categories: {data['categories']}")

    # Stats
    from collections import Counter
    cat_counts = Counter(a["category_id"] for a in data["annotations"])
    print(f"Annotations per category_id: {dict(cat_counts)}")

    # Sample annotations
    print("\n" + "=" * 60)
    print("SAMPLE ANNOTATIONS (first 5)")
    print("=" * 60)
    for i, ann in enumerate(data["annotations"][:5]):
        print(f"  {i}: image_id={ann['image_id']} category_id={ann['category_id']} "
              f"bbox={ann['bbox']} area={ann.get('area', 'N/A')}")

    # Image info
    img_ids = list({a["image_id"] for a in data["annotations"]})[:args.n_samples]
    if not img_ids:
        img_ids = [data["images"][0]["id"]]

    print("\n" + "=" * 60)
    print("SAMPLE IMAGES")
    print("=" * 60)

    try:
        from PIL import Image
        import matplotlib.pyplot as plt
        import matplotlib.patches as mpatches
    except ImportError:
        print("Install: pip install pillow matplotlib")
        return

    for idx, img_id in enumerate(img_ids):
        img_info = next((i for i in data["images"] if i["id"] == img_id), None)
        if not img_info:
            continue
        fn = img_info["file_name"]
        path = img_dir / fn
        if not path.exists():
            print(f"  Image not found: {path}")
            continue
        img = Image.open(path).convert("RGB")
        anns = [a for a in data["annotations"] if a["image_id"] == img_id]

        fig, ax = plt.subplots(1, figsize=(8, 8))
        ax.imshow(img)
        for ann in anns:
            x, y, w, h = ann["bbox"]
            rect = mpatches.Rectangle((x, y), w, h, linewidth=2, edgecolor="lime", facecolor="none")
            ax.add_patch(rect)
            ax.text(x, y - 2, f"cat={ann['category_id']}", color="lime", fontsize=8)
        ax.set_title(f"{fn}\n{len(anns)} annotations")
        ax.axis("off")
        plt.tight_layout()
        if args.save_viz:
            out = Path("debug_viz")
            out.mkdir(exist_ok=True)
            plt.savefig(out / f"sample_{idx}.png", dpi=100, bbox_inches="tight")
            print(f"  Saved: {out}/sample_{idx}.png")
        plt.show()
        plt.close()

    print("\nDone. If bboxes look correct, the dataset is fine.")


if __name__ == "__main__":
    main()
