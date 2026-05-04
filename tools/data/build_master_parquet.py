import argparse
from pathlib import Path
from typing import Dict, Any, List
import orjson
import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq
from tqdm import tqdm


def read_json(p: Path) -> Any:
    return orjson.loads(p.read_bytes())


def build_indices(coco: Dict[str, Any]):
    images_by_filename = {img["file_name"]: img for img in coco.get("images", [])}
    annos_by_image_id: Dict[int, List[Dict[str, Any]]] = {}
    for a in coco.get("annotations", []):
        annos_by_image_id.setdefault(int(a["image_id"]), []).append(a)
    return images_by_filename, annos_by_image_id, coco.get("categories", [])


def write_shard(rows: List[Dict[str, Any]], out_path: Path):
    table = pa.Table.from_pylist(rows)
    pq.write_table(
        table,
        out_path,
        compression="zstd",
        use_dictionary=True,
        write_statistics=True,
    )


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--src_root", required=True, help="dir with metadata.csv world_annotations.json world_images/")
    ap.add_argument("--out_dir", required=True, help="hf_repo/data/master")
    ap.add_argument("--rows_per_shard", type=int, default=256)
    ap.add_argument(
        "--strict_filenames",
        action="store_true",
        help="if set, fail when metadata filenames are missing from COCO/images; default is to skip",
    )
    args = ap.parse_args()

    src = Path(args.src_root)
    out = Path(args.out_dir)
    out.mkdir(parents=True, exist_ok=True)

    meta = pd.read_csv(src / "metadata.csv", low_memory=False)
    coco = read_json(src / "world_annotations.json")
    images_by_filename, annos_by_image_id, categories = build_indices(coco)

    missing = set(meta["filename"]) - set(images_by_filename.keys())
    if missing:
        ex = next(iter(missing))
        if args.strict_filenames:
            raise RuntimeError(f"{len(missing)} filenames in metadata missing from COCO images[]. Example: {ex}")
        print(
            f"Warning: {len(missing)} filenames in metadata missing from COCO images[]; "
            f"they will be skipped. Example: {ex}"
        )

    rows = []
    shard_idx = 0

    skipped_not_in_coco = 0
    skipped_missing_image_file = 0
    written_rows = 0

    for r in tqdm(meta.itertuples(index=False), total=len(meta), desc="Building master parquet"):
        filename = r.filename
        if filename in missing:
            skipped_not_in_coco += 1
            continue
        img_info = images_by_filename[filename]
        image_id = int(img_info["id"])
        img_path = src / "world_images" / filename

        if not img_path.exists():
            skipped_missing_image_file += 1
            continue
        b = img_path.read_bytes()
        annos = annos_by_image_id.get(image_id, [])

        rows.append({
            "image_id": image_id,
            "filename": filename,
            "country": r.country,
            "state": r.state,
            "zone": str(r.zone),
            "region": str(getattr(r, "region", "")),
            "width": int(img_info.get("width", -1)),
            "height": int(img_info.get("height", -1)),
            "image_bytes": b,
            "coco_annotations": orjson.dumps(annos).decode("utf-8"),
            "coco_categories": orjson.dumps(categories).decode("utf-8"),
        })
        written_rows += 1

        if len(rows) >= args.rows_per_shard:
            shard_path = out / f"all-{shard_idx:05d}.parquet"
            write_shard(rows, shard_path)
            rows = []
            shard_idx += 1

    if rows:
        shard_path = out / f"all-{shard_idx:05d}.parquet"
        write_shard(rows, shard_path)

    print(
        f"Done. Wrote master shards to {out} | written={written_rows} "
        f"skipped_not_in_coco={skipped_not_in_coco} skipped_missing_image_file={skipped_missing_image_file}"
    )


if __name__ == "__main__":
    main()

