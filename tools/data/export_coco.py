import argparse
from pathlib import Path
import orjson
from datasets import load_dataset, get_dataset_split_names


def export_split(ds, out_dir: Path, split_name: str):
    images_dir = out_dir / split_name / "images"
    ann_dir = out_dir / split_name / "annotations"
    images_dir.mkdir(parents=True, exist_ok=True)
    ann_dir.mkdir(parents=True, exist_ok=True)

    images = []
    annotations = []
    categories = None
    ann_id = 1

    for i, ex in enumerate(ds, 1):
        image_id = int(ex["image_id"])
        filename = ex["filename"]
        width = int(ex["width"])
        height = int(ex["height"])

        (images_dir / filename).write_bytes(ex["image_bytes"])

        images.append(
            {"id": image_id, "file_name": filename, "width": width, "height": height}
        )

        annos = orjson.loads(ex["coco_annotations"])
        if categories is None:
            categories = orjson.loads(ex["coco_categories"])

        for a in annos:
            a = dict(a)
            a["id"] = ann_id
            ann_id += 1
            annotations.append(a)

        if i % 1000 == 0:
            print(f"[{split_name}] exported {i} images...")

    coco = {"images": images, "annotations": annotations, "categories": categories or []}
    (ann_dir / f"instances_{split_name}.json").write_bytes(orjson.dumps(coco))
    print(f"[{split_name}] images={len(images)} annotations={len(annotations)}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--repo", required=True, help="HF dataset repo id (e.g. aadityabuilds/tree-distribution-shift)")
    ap.add_argument("--config", required=True, help="Config name (e.g. in_state_train_Karnataka__ood_Rajasthan)")
    ap.add_argument("--out", required=True, help="Output directory (writes <out>/<config>/...)")
    ap.add_argument("--revision", default=None, help="Optional HF revision/commit sha")
    args = ap.parse_args()

    out_dir = Path(args.out) / args.config
    out_dir.mkdir(parents=True, exist_ok=True)

    # Discover splits from the Hub (so we don't hardcode)
    splits = get_dataset_split_names(args.repo, args.config, revision=args.revision)
    print("Splits:", splits)

    for split in splits:
        # streaming=True avoids huge RAM mmap / pyarrow pressure
        ds = load_dataset(
            args.repo,
            args.config,
            split=split,
            streaming=True,
            revision=args.revision,
        )
        export_split(ds, out_dir, split)

    print(f"COCO export written to: {out_dir}")


if __name__ == "__main__":
    main()
