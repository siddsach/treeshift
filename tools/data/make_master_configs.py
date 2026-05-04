import argparse
import json
import random
from pathlib import Path
from typing import Dict, List, Tuple

import pandas as pd
import pyarrow as pa
import pyarrow.dataset as ds
import pyarrow.parquet as pq


def split_pool(items: List[str], train_ratio: float, seed: int) -> Tuple[List[str], List[str]]:
    rng = random.Random(seed)
    items = list(items)
    rng.shuffle(items)
    n_train = int(round(train_ratio * len(items)))
    return items[:n_train], items[n_train:]


def ensure_min(name: str, files: List[str], min_count: int) -> None:
    if len(files) < min_count:
        raise RuntimeError(f"{name}: only {len(files)} files (<{min_count}).")


def build_light_index(master_dir: str, scan_batch_size: int) -> Dict[str, Dict]:
    master = ds.dataset(master_dir, format="parquet")
    cols = [
        "image_id",
        "filename",
        "country",
        "state",
        "zone",
        "region",
        "width",
        "height",
        "coco_annotations",
        "coco_categories",
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


def write_rows_to_parquet(rows_iter, out_dir: Path, split: str, rows_per_shard: int) -> None:
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


def rows_from_files(filenames: List[str], light_index: Dict[str, Dict], images_root: Path):
    for fn in filenames:
        base = light_index.get(fn)
        if base is None:
            continue
        img_path = images_root / fn
        yield {**base, "image_bytes": img_path.read_bytes()}


def _normalize_label(s: str) -> str:
    return str(s).strip().lower()


def _karnataka_elevation_files(meta: pd.DataFrame, label: str) -> List[str]:
    kar = meta[(meta["country"] == "India") & (meta["state"] == "Karnataka")].copy()
    kar["elevation_class_zonewise"] = kar["elevation_class_zonewise"].astype(str).map(_normalize_label)
    return kar[kar["elevation_class_zonewise"] == label]["filename"].tolist()


def collect_base_splits(
    meta: pd.DataFrame,
    train_ratio: float,
    seed: int,
    min_pool: int,
    ood_train_ratio: float = 0.7,
) -> Dict[str, Dict[str, List[str]]]:
    meta = meta.copy()
    meta["biome"] = meta["biome"].astype(str).str.upper().str.strip()
    meta["region"] = meta["region"].astype(str).str.strip()

    required = {"filename", "country", "state", "zone", "biome", "region", "elevation_class_zonewise"}
    missing = required - set(meta.columns)
    if missing:
        raise RuntimeError(f"metadata.csv missing required columns: {sorted(missing)}")

    split_map: Dict[str, Dict[str, List[str]]] = {}

    def add_single_config(cfg_name: str, id_pool: List[str], ood_pool: List[str]) -> None:
        train_files, id_test_files = split_pool(id_pool, train_ratio, seed)
        # Split OOD pool 70:30 into ood_train (for few-shot) and ood_test (held-out eval)
        ood_train_files, ood_test_files = split_pool(ood_pool, ood_train_ratio, seed)

        ensure_min(cfg_name + ":train", train_files, min_pool)
        ensure_min(cfg_name + ":id_test", id_test_files, 10)
        ensure_min(cfg_name + ":ood_train", ood_train_files, 10)
        ensure_min(cfg_name + ":ood_test", ood_test_files, 10)

        split_map[cfg_name] = {
            "train": sorted(train_files),
            "id_test": sorted(id_test_files),
            "ood_train": sorted(ood_train_files),
            "ood_test": sorted(ood_test_files),
        }

    # 1) Country shift (India <-> US)
    files_in = meta[meta["country"] == "India"]["filename"].tolist()
    files_us = meta[meta["country"] == "US"]["filename"].tolist()
    add_single_config("intl_train_IN__ood_US", files_in, files_us)
    add_single_config("intl_train_US__ood_IN", files_us, files_in)

    # 2) Rajasthan biome shift (WET <-> DRY)
    raj = meta[(meta["country"] == "India") & (meta["state"] == "Rajasthan")]
    raj_wet = raj[raj["biome"] == "WET"]["filename"].tolist()
    raj_dry = raj[raj["biome"] == "DRY"]["filename"].tolist()
    add_single_config("biome_Rajasthan_train_WET__ood_DRY", raj_wet, raj_dry)
    add_single_config("biome_Rajasthan_train_DRY__ood_WET", raj_dry, raj_wet)

    # 3) Karnataka elevation shift (HIGH <-> LOW)
    kar_high = _karnataka_elevation_files(meta, "high")
    kar_low = _karnataka_elevation_files(meta, "low")
    add_single_config("elev_Karnataka_train_HIGH__ood_LOW", kar_high, kar_low)
    add_single_config("elev_Karnataka_train_LOW__ood_HIGH", kar_low, kar_high)

    # 4) India region shift (North <-> South)
    north = meta[(meta["country"] == "India") & (meta["region"] == "North")]["filename"].tolist()
    south = meta[(meta["country"] == "India") & (meta["region"] == "South")]["filename"].tolist()
    add_single_config("region_train_North__ood_South", north, south)
    add_single_config("region_train_South__ood_North", south, north)

    # 5) Whole-India regular non-shift baseline.
    # This is an ID-only config: train on a fixed 80% of India and evaluate on
    # the held-out 20% as id_test/val. It intentionally has no ood_* splits.
    india_files = sorted(meta[meta["country"] == "India"]["filename"].tolist())
    train_files, id_test_files = split_pool(india_files, 0.8, seed)
    ensure_min("india_random_80_20:train", train_files, min_pool)
    ensure_min("india_random_80_20:id_test", id_test_files, 10)
    split_map["india_random_80_20"] = {
        "train": sorted(train_files),
        "id_test": sorted(id_test_files),
    }

    return split_map


def write_config_from_split_map(
    split_map: Dict[str, Dict[str, List[str]]],
    out_configs_dir: Path,
    rows_per_shard: int,
    light_index: Dict[str, Dict],
    images_root: Path,
) -> None:
    out_configs_dir.mkdir(parents=True, exist_ok=True)
    total = len(split_map)
    print(f"[make_master_configs] Writing {total} base configs to {out_configs_dir}")
    for cfg_name, splits in split_map.items():
        cfg_dir = out_configs_dir / cfg_name
        cfg_dir.mkdir(parents=True, exist_ok=True)
        train_n = len(splits["train"])
        id_n = len(splits["id_test"])
        ood_train_n = len(splits.get("ood_train", []))
        ood_n = len(splits.get("ood_test", []))
        print(f"[make_master_configs] -> {cfg_name} (train={train_n}, id_test={id_n}, ood_train={ood_train_n}, ood_test={ood_n})")
        for split in ("train", "id_test", "ood_train", "ood_test"):
            if split not in splits or not splits[split]:
                continue
            write_rows_to_parquet(
                rows_from_files(splits[split], light_index, images_root),
                cfg_dir,
                split,
                rows_per_shard,
            )
        print(f"[make_master_configs] <- {cfg_name} done")


def build_master_configs(
    src_root: Path,
    master_dir: Path,
    out_configs_dir: Path,
    train_ratio: float,
    seed: int,
    rows_per_shard: int,
    scan_batch_size: int,
    min_pool: int,
    ood_train_ratio: float = 0.7,
    write_parquet: bool = True,
) -> Dict[str, Dict[str, List[str]]]:
    meta_path = src_root / "metadata.csv"
    images_root = src_root / "world_images"
    if not meta_path.exists():
        raise FileNotFoundError(f"metadata.csv not found at: {meta_path}")
    if not images_root.exists():
        raise FileNotFoundError(f"world_images/ not found at: {images_root}")

    meta = pd.read_csv(meta_path)
    split_map = collect_base_splits(
        meta,
        train_ratio=train_ratio,
        seed=seed,
        min_pool=min_pool,
        ood_train_ratio=ood_train_ratio,
    )

    if write_parquet:
        print("[make_master_configs] Building lightweight master index...")
        light_index = build_light_index(str(master_dir), scan_batch_size)
        print(f"[make_master_configs] Indexed {len(light_index)} master rows")
        write_config_from_split_map(
            split_map=split_map,
            out_configs_dir=out_configs_dir,
            rows_per_shard=rows_per_shard,
            light_index=light_index,
            images_root=images_root,
        )

    return split_map


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--src_root", required=True, help="root containing metadata.csv and world_images/")
    ap.add_argument("--master_dir", required=True, help="hf_repo/data/master (parquet shards)")
    ap.add_argument("--out_configs_dir", required=True, help="hf_repo/data/configs")
    ap.add_argument("--train_ratio", type=float, default=0.9)
    ap.add_argument("--ood_train_ratio", type=float, default=0.7,
                    help="Fraction of OOD pool used for ood_train (few-shot source); remainder is ood_test")
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--rows_per_shard", type=int, default=16)
    ap.add_argument("--scan_batch_size", type=int, default=32)
    ap.add_argument("--min_pool", type=int, default=200)
    ap.add_argument("--manifest_path", default=None, help="optional path to write JSON manifest")
    args = ap.parse_args()

    split_map = build_master_configs(
        src_root=Path(args.src_root),
        master_dir=Path(args.master_dir),
        out_configs_dir=Path(args.out_configs_dir),
        train_ratio=args.train_ratio,
        seed=args.seed,
        rows_per_shard=args.rows_per_shard,
        scan_batch_size=args.scan_batch_size,
        min_pool=args.min_pool,
        ood_train_ratio=args.ood_train_ratio,
        write_parquet=True,
    )

    manifest = {
        "base_config_count": len(split_map),
        "base_configs": sorted(split_map.keys()),
        "splits": ["train", "id_test", "ood_train", "ood_test"],
        "split_sizes": {
            cfg: {split: len(files) for split, files in splits.items()}
            for cfg, splits in split_map.items()
        },
    }

    if args.manifest_path:
        manifest_path = Path(args.manifest_path)
        manifest_path.parent.mkdir(parents=True, exist_ok=True)
        manifest_path.write_text(json.dumps(manifest, indent=2))

    print(json.dumps(manifest, indent=2))


if __name__ == "__main__":
    main()
