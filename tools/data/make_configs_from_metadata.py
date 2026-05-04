import argparse
from pathlib import Path
from typing import Dict, List, Tuple, Set, Any
import random

import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq
import pyarrow.dataset as ds


def split_list(items: List[str], train_ratio: float, seed: int) -> Tuple[Set[str], Set[str]]:
    rng = random.Random(seed)
    items = items.copy()
    rng.shuffle(items)
    n_train = int(round(train_ratio * len(items)))
    return set(items[:n_train]), set(items[n_train:])


def write_rows_to_parquet(rows_iter, out_dir: Path, split: str, rows_per_shard: int):
    out_dir.mkdir(parents=True, exist_ok=True)
    buf = []
    shard = 0
    for row in rows_iter:
        buf.append(row)
        if len(buf) >= rows_per_shard:
            table = pa.Table.from_pylist(buf)
            pq.write_table(table, out_dir / f"{split}-{shard:05d}.parquet", compression="zstd")
            buf = []
            shard += 1
    if buf:
        table = pa.Table.from_pylist(buf)
        pq.write_table(table, out_dir / f"{split}-{shard:05d}.parquet", compression="zstd")


def fetch_rows_by_filenames(master: ds.Dataset, filenames: Set[str]):
    # For 27K rows, scanning + filtering in Python is acceptable and robust.
    scanner = master.scanner(columns=[
        "image_id","filename","country","state","zone","region","width","height",
        "image_bytes","coco_annotations","coco_categories"
    ])
    for batch in scanner.to_batches():
        b = batch.to_pydict()
        n = len(b["filename"])
        for i in range(n):
            if b["filename"][i] in filenames:
                yield {k: b[k][i] for k in b.keys()}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--src_root", required=True, help="original root with metadata.csv")
    ap.add_argument("--master_dir", required=True, help="hf_repo/data/master")
    ap.add_argument("--out_configs_dir", required=True, help="hf_repo/data/configs")
    ap.add_argument("--mode", choices=["state_shift", "zone_shift"], required=True)
    ap.add_argument("--train_ratio", type=float, default=0.9)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--rows_per_shard", type=int, default=256)
    ap.add_argument("--country", default=None, help="optional: only generate configs for this country (e.g. US)")
    ap.add_argument("--min_images_per_region", type=int, default=50, help="skip regions with fewer images")
    ap.add_argument("--max_configs", type=int, default=50, help="cap number of configs generated")
    args = ap.parse_args()

    meta = pd.read_csv(Path(args.src_root) / "metadata.csv")
    if args.country:
        meta = meta[meta["country"] == args.country].copy()

    master = ds.dataset(args.master_dir, format="parquet")
    out_configs = Path(args.out_configs_dir)
    out_configs.mkdir(parents=True, exist_ok=True)

    configs: List[Tuple[str, Set[str], Set[str], Set[str]]] = []

    if args.mode == "state_shift":
        regions = meta[["country", "state"]].drop_duplicates()
        region_to_files: Dict[Tuple[str, str], List[str]] = {}
        for (c, s) in regions.itertuples(index=False):
            files = meta[(meta.country == c) & (meta.state == s)]["filename"].tolist()
            region_to_files[(c, s)] = files

        keys = sorted(region_to_files.keys())
        for (c_id, s_id) in keys:
            id_files = region_to_files[(c_id, s_id)]
            if len(id_files) < args.min_images_per_region:
                continue

            train_files, val_files = split_list(id_files, args.train_ratio, args.seed)

            for (c_ood, s_ood) in keys:
                if c_ood != c_id or s_ood == s_id:
                    continue
                ood_files = region_to_files[(c_ood, s_ood)]
                if len(ood_files) < args.min_images_per_region:
                    continue

                cfg = f"state_{c_id}_{s_id}__ood_state_{c_ood}_{s_ood}"
                configs.append((cfg, train_files, val_files, set(ood_files)))
                if len(configs) >= args.max_configs:
                    break
            if len(configs) >= args.max_configs:
                break

    else:
        meta["zone"] = meta["zone"].astype(str)
        zones = sorted(meta["zone"].unique().tolist())
        zone_to_files = {z: meta[meta["zone"] == z]["filename"].tolist() for z in zones}

        for z_id in zones:
            id_files = zone_to_files[z_id]
            if len(id_files) < args.min_images_per_region:
                continue

            train_files, val_files = split_list(id_files, args.train_ratio, args.seed)

            for z_ood in zones:
                if z_ood == z_id:
                    continue
                ood_files = zone_to_files[z_ood]
                if len(ood_files) < args.min_images_per_region:
                    continue

                cfg = f"zone_{z_id}__ood_zone_{z_ood}"
                configs.append((cfg, train_files, val_files, set(ood_files)))
                if len(configs) >= args.max_configs:
                    break
            if len(configs) >= args.max_configs:
                break

    for cfg_name, train_files, val_files, ood_files in configs:
        cfg_dir = out_configs / cfg_name
        cfg_dir.mkdir(parents=True, exist_ok=True)

        write_rows_to_parquet(fetch_rows_by_filenames(master, train_files), cfg_dir, "train", args.rows_per_shard)
        write_rows_to_parquet(fetch_rows_by_filenames(master, val_files), cfg_dir, "val", args.rows_per_shard)
        write_rows_to_parquet(fetch_rows_by_filenames(master, ood_files), cfg_dir, "ood_test", args.rows_per_shard)

        print(f"Wrote config: {cfg_name}")

    print(f"Done. Created {len(configs)} configs in {out_configs}")


if __name__ == "__main__":
    main()

