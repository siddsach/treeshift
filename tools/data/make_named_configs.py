import argparse
from pathlib import Path
from typing import List, Set, Tuple, Dict
import random

import pandas as pd
import pyarrow.dataset as ds
import pyarrow as pa
import pyarrow.parquet as pq


def split_pool(items: List[str], train_ratio: float, seed: int) -> Tuple[List[str], List[str]]:
    """Deterministically split a pool into (train, test) portions."""
    rng = random.Random(seed)
    items = list(items)
    rng.shuffle(items)
    n_train = int(round(train_ratio * len(items)))
    return items[:n_train], items[n_train:]


def ensure_min(name: str, files, min_count: int):
    if len(files) < min_count:
        raise RuntimeError(f"{name}: only {len(files)} files (<{min_count}).")


def build_light_index(master_dir: str, scan_batch_size: int) -> Dict[str, Dict]:
    """
    Build a filename->row dict WITHOUT reading image_bytes.
    This avoids Arrow OOM during scanning.
    """
    master = ds.dataset(master_dir, format="parquet")
    cols = [
        "image_id", "filename", "country", "state", "zone", "region",
        "width", "height", "coco_annotations", "coco_categories",
    ]
    scanner = master.scanner(columns=cols, batch_size=scan_batch_size)

    idx: Dict[str, Dict] = {}
    for batch in scanner.to_batches():
        b = batch.to_pydict()
        n = len(b["filename"])
        for i in range(n):
            fn = b["filename"][i]
            idx[fn] = {k: b[k][i] for k in cols}
    return idx


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


def rows_from_files(filenames, light_index: Dict[str, Dict], images_root: Path):
    """Materialize full rows by reading image bytes directly from disk."""
    for fn in filenames:
        base = light_index.get(fn)
        if base is None:
            continue
        img_path = images_root / fn
        yield {**base, "image_bytes": img_path.read_bytes()}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--src_root", required=True,
                    help="root containing metadata.csv and world_images/")
    ap.add_argument("--master_dir", required=True,
                    help="hf_repo/data/master (parquet shards)")
    ap.add_argument("--out_configs_dir", required=True,
                    help="hf_repo/data/configs")
    ap.add_argument("--train_ratio", type=float, default=0.9)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--rows_per_shard", type=int, default=16)
    ap.add_argument("--scan_batch_size", type=int, default=32)
    ap.add_argument("--min_pool", type=int, default=200,
                    help="minimum images required in a pool to create a config")
    args = ap.parse_args()

    src_root = Path(args.src_root)
    meta_path = src_root / "metadata.csv"
    if not meta_path.exists():
        raise FileNotFoundError(f"metadata.csv not found at: {meta_path}")

    images_root = src_root / "world_images"
    if not images_root.exists():
        raise FileNotFoundError(f"world_images/ not found at: {images_root}")

    meta = pd.read_csv(meta_path)

    required = {"filename", "country", "state", "zone", "biome", "region"}
    missing = required - set(meta.columns)
    if missing:
        raise RuntimeError(f"metadata.csv missing required columns: {sorted(missing)}")

    meta["biome"] = meta["biome"].astype(str).str.upper().str.strip()
    meta["region"] = meta["region"].astype(str).str.strip()

    print("Building lightweight master index (no image_bytes)...")
    light_index = build_light_index(args.master_dir, args.scan_batch_size)
    print(f"Indexed {len(light_index)} rows from master.")

    out_root = Path(args.out_configs_dir)
    out_root.mkdir(parents=True, exist_ok=True)

    # Pre-split every pool deterministically so that the test portion is
    # identical regardless of whether the pool is used as ID or OOD.
    pool_splits: Dict[str, Tuple[List[str], List[str]]] = {}

    def get_pool_split(pool_key: str, filenames: List[str]) -> Tuple[List[str], List[str]]:
        if pool_key not in pool_splits:
            pool_splits[pool_key] = split_pool(filenames, args.train_ratio, args.seed)
        return pool_splits[pool_key]

    def materialize_pair(cfg_name_a: str, cfg_name_b: str,
                         pool_key_a: str, files_a: List[str],
                         pool_key_b: str, files_b: List[str]):
        """
        Create two paired configs where the test portion of each pool is
        constant across both configs.

        Config A: train on pool A, OOD = pool B
          train    = A's 90%
          val      = A's 10%   (same images as ood_test in config B)
          ood_test = B's 10%   (same images as val in config B)

        Config B: symmetric.
        """
        train_a, test_a = get_pool_split(pool_key_a, files_a)
        train_b, test_b = get_pool_split(pool_key_b, files_b)

        ensure_min(cfg_name_a + ":train", train_a, args.min_pool)
        ensure_min(cfg_name_a + ":val", test_a, 10)
        ensure_min(cfg_name_a + ":ood_test", test_b, 10)
        ensure_min(cfg_name_b + ":train", train_b, args.min_pool)
        ensure_min(cfg_name_b + ":val", test_b, 10)
        ensure_min(cfg_name_b + ":ood_test", test_a, 10)

        for cfg_name, train_files, val_files, ood_files in [
            (cfg_name_a, train_a, test_a, test_b),
            (cfg_name_b, train_b, test_b, test_a),
        ]:
            cfg_dir = out_root / cfg_name
            cfg_dir.mkdir(parents=True, exist_ok=True)

            write_rows_to_parquet(
                rows_from_files(train_files, light_index, images_root),
                cfg_dir, "train", args.rows_per_shard)
            write_rows_to_parquet(
                rows_from_files(val_files, light_index, images_root),
                cfg_dir, "val", args.rows_per_shard)
            write_rows_to_parquet(
                rows_from_files(ood_files, light_index, images_root),
                cfg_dir, "ood_test", args.rows_per_shard)

            print(f"  {cfg_name}: train={len(train_files)}  val={len(val_files)}  ood_test={len(ood_files)}")

    # ---- 1) US vs India (country field) ----
    print("\n=== Country shift: US vs India ===")
    files_in = meta[meta["country"] == "India"]["filename"].tolist()
    files_us = meta[meta["country"] == "US"]["filename"].tolist()
    materialize_pair(
        "intl_train_IN__ood_US", "intl_train_US__ood_IN",
        "country:India", files_in,
        "country:US", files_us,
    )

    # ---- 2) Wet vs Dry biome in Rajasthan ----
    print("\n=== Biome shift: Rajasthan WET vs DRY ===")
    raj = meta[(meta["country"] == "India") & (meta["state"] == "Rajasthan")]
    raj_wet = raj[raj["biome"] == "WET"]["filename"].tolist()
    raj_dry = raj[raj["biome"] == "DRY"]["filename"].tolist()
    materialize_pair(
        "biome_Rajasthan_train_WET__ood_DRY", "biome_Rajasthan_train_DRY__ood_WET",
        "biome:Rajasthan:WET", raj_wet,
        "biome:Rajasthan:DRY", raj_dry,
    )

    # ---- 3) Wet vs Dry biome in Karnataka ----
    print("\n=== Biome shift: Karnataka WET vs DRY ===")
    kar = meta[(meta["country"] == "India") & (meta["state"] == "Karnataka")]
    kar_wet = kar[kar["biome"] == "WET"]["filename"].tolist()
    kar_dry = kar[kar["biome"] == "DRY"]["filename"].tolist()
    materialize_pair(
        "biome_Karnataka_train_WET__ood_DRY", "biome_Karnataka_train_DRY__ood_WET",
        "biome:Karnataka:WET", kar_wet,
        "biome:Karnataka:DRY", kar_dry,
    )

    # ---- 4) North vs South India (region field) ----
    print("\n=== Region shift: North vs South India ===")
    files_north = meta[meta["region"] == "North"]["filename"].tolist()
    files_south = meta[meta["region"] == "South"]["filename"].tolist()
    materialize_pair(
        "region_train_North__ood_South", "region_train_South__ood_North",
        "region:North", files_north,
        "region:South", files_south,
    )

    print(f"\nDone. All configs written under: {out_root}")


if __name__ == "__main__":
    main()
