import argparse
import json
import random
import zlib
from pathlib import Path
from typing import Dict, List

from make_master_configs import (
    build_light_index,
    build_master_configs,
    rows_from_files,
    write_rows_to_parquet,
)


def _stable_seed(base_seed: int, cfg_name: str, tag: str) -> int:
    h = zlib.adler32(f"{cfg_name}::{tag}".encode("utf-8")) & 0xFFFFFFFF
    return (base_seed * 1_000_003 + h * 97) & 0xFFFFFFFF


def _sample_k(items: List[str], k: int, seed: int) -> List[str]:
    if k <= 0:
        return []
    if k >= len(items):
        return sorted(items)
    rng = random.Random(seed)
    return sorted(rng.sample(sorted(items), k))


def build_fewshot_split_map(
    base_split_map: Dict[str, Dict[str, List[str]]],
    seed: int,
) -> Dict[str, Dict[str, List[str]]]:
    out: Dict[str, Dict[str, List[str]]] = {}

    for base_cfg, splits in base_split_map.items():
        if "ood_train" not in splits or "ood_test" not in splits:
            continue

        base_train = sorted(splits["train"])
        base_id_test = sorted(splits["id_test"])
        base_ood_train = sorted(splits["ood_train"])
        base_ood_test = sorted(splits["ood_test"])  # never changes across few-shot variants

        # Build a single deterministic ordering of ood_train so that few-shot sets
        # are guaranteed to be progressive subsets: fs1 ⊂ fs10 ⊂ fs100 ⊂ fsall
        ordering_seed = _stable_seed(seed, base_cfg, "fsordering")
        ordered_ood_train = list(base_ood_train)
        rng = random.Random(ordering_seed)
        rng.shuffle(ordered_ood_train)

        for label, k in [("1", 1), ("10", 10), ("100", 100)]:
            k_actual = min(k, len(ordered_ood_train))
            chosen = ordered_ood_train[:k_actual]
            chosen_set = set(chosen)

            fs_cfg_name = f"{base_cfg}__fs{label}"
            out[fs_cfg_name] = {
                "train": sorted(set(base_train) | chosen_set),
                "id_test": list(base_id_test),
                "ood_train": sorted(set(base_ood_train) - chosen_set),
                "ood_test": list(base_ood_test),  # unchanged
            }

        # fsall: move all ood_train into train; ood_test still unchanged
        fs_cfg_name = f"{base_cfg}__fsall"
        out[fs_cfg_name] = {
            "train": sorted(set(base_train) | set(base_ood_train)),
            "id_test": list(base_id_test),
            "ood_train": [],  # all moved to train
            "ood_test": list(base_ood_test),  # unchanged
        }

    return out


def write_configs_from_split_map(
    split_map: Dict[str, Dict[str, List[str]]],
    out_configs_dir: Path,
    rows_per_shard: int,
    light_index: Dict[str, Dict],
    images_root: Path,
) -> None:
    out_configs_dir.mkdir(parents=True, exist_ok=True)
    total = len(split_map)
    print(f"[make_configs] Writing {total} configs to {out_configs_dir}")
    for cfg_name, splits in split_map.items():
        cfg_dir = out_configs_dir / cfg_name
        cfg_dir.mkdir(parents=True, exist_ok=True)
        train_n = len(splits["train"])
        id_n = len(splits["id_test"])
        ood_train_n = len(splits.get("ood_train", []))
        ood_n = len(splits.get("ood_test", []))
        print(f"[make_configs] -> {cfg_name} (train={train_n}, id_test={id_n}, ood_train={ood_train_n}, ood_test={ood_n})")
        for split in ("train", "id_test", "ood_train", "ood_test"):
            if split not in splits or not splits[split]:
                continue
            write_rows_to_parquet(
                rows_from_files(splits[split], light_index, images_root),
                cfg_dir,
                split,
                rows_per_shard,
            )
        print(f"[make_configs] <- {cfg_name} done")


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

    src_root = Path(args.src_root)
    out_configs_dir = Path(args.out_configs_dir)
    images_root = src_root / "world_images"

    base_split_map = build_master_configs(
        src_root=src_root,
        master_dir=Path(args.master_dir),
        out_configs_dir=out_configs_dir,
        train_ratio=args.train_ratio,
        seed=args.seed,
        rows_per_shard=args.rows_per_shard,
        scan_batch_size=args.scan_batch_size,
        min_pool=args.min_pool,
        ood_train_ratio=args.ood_train_ratio,
        write_parquet=False,
    )

    fewshot_split_map = build_fewshot_split_map(base_split_map, seed=args.seed)

    all_split_map = {}
    all_split_map.update(base_split_map)
    all_split_map.update(fewshot_split_map)
    print(
        f"[make_configs] Prepared split plans: base={len(base_split_map)} "
        f"fewshot={len(fewshot_split_map)} total={len(all_split_map)}"
    )

    print("[make_configs] Building lightweight master index...")
    light_index = build_light_index(str(args.master_dir), args.scan_batch_size)
    print(f"[make_configs] Indexed {len(light_index)} master rows")
    write_configs_from_split_map(
        split_map=all_split_map,
        out_configs_dir=out_configs_dir,
        rows_per_shard=args.rows_per_shard,
        light_index=light_index,
        images_root=images_root,
    )

    base_cfgs = sorted(base_split_map.keys())
    fs_cfgs = sorted(fewshot_split_map.keys())
    manifest = {
        "base_config_count": len(base_cfgs),
        "fewshot_per_base": ["fs1", "fs10", "fs100", "fsall"],
        "total_config_count": len(all_split_map),
        "splits": ["train", "id_test", "ood_train", "ood_test"],
        "split_notes": {
            "train": "ID training set (90% of ID pool)",
            "id_test": "ID test set (10% of ID pool, fixed across all configs)",
            "ood_train": "OOD training pool (70% of OOD pool); source for few-shot images",
            "ood_test": "OOD test set (30% of OOD pool, fixed across all configs including few-shot variants)",
        },
        "base_configs": base_cfgs,
        "fewshot_configs": fs_cfgs,
        "all_configs": sorted(all_split_map.keys()),
    }

    if args.manifest_path:
        manifest_path = Path(args.manifest_path)
        manifest_path.parent.mkdir(parents=True, exist_ok=True)
        manifest_path.write_text(json.dumps(manifest, indent=2))

    print(json.dumps(manifest, indent=2))


if __name__ == "__main__":
    main()
