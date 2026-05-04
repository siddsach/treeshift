# tools/make_fewshot_configs.py
#
# Create few-shot variants of existing configs by moving K images from OOD -> TRAIN.
# Works with density-aware configs (ood_same_density / ood_diff_density) if present.
#
# Output config name: <base_cfg>__fs<K>
# Output splits written:
#   - train
#   - id_test
#   - ood_test
#   - ood_same_density (if present in base)
#   - ood_diff_density (if present in base)
#
# Memory-safe:
# - Never builds giant Python lists of rows
# - Streams master parquet via pyarrow.dataset + record batches
# - Writes parquet shards with lazy-open writers + safe rotation

from __future__ import annotations

import argparse
import json
import os
import random
import zlib
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Set, Tuple

import pyarrow as pa
import pyarrow.compute as pc
import pyarrow.dataset as ds
import pyarrow.parquet as pq


def _stable_seed(base_seed: int, cfg_name: str, k: int) -> int:
    # deterministic across runs/machines
    h = zlib.adler32(cfg_name.encode("utf-8")) & 0xFFFFFFFF
    return (base_seed * 1_000_003 + h * 97 + k * 7919) & 0xFFFFFFFF


def _list_parquet_files(cfg_dir: Path, split: str) -> List[Path]:
    return sorted(cfg_dir.glob(f"{split}-*.parquet"))


def _read_filenames_from_split(cfg_dir: Path, split: str) -> Set[str]:
    files = _list_parquet_files(cfg_dir, split)
    if not files:
        return set()
    d = ds.dataset([str(p) for p in files], format="parquet")
    # only pull filename column to keep memory low
    col = d.to_table(columns=["filename"]).column("filename")
    return set(col.to_pylist())


def _sample_k_from_set(items: Set[str], k: int, seed: int) -> List[str]:
    if k <= 0:
        return []
    if len(items) < k:
        raise ValueError(f"Not enough items to sample k={k}. pool={len(items)}")
    rng = random.Random(seed)
    items_list = sorted(items)  # stable ordering
    return rng.sample(items_list, k)


@dataclass
class _SplitWriter:
    out_dir: Path
    split_name: str
    rows_per_shard: int
    compression: str = "zstd"

    # internal state
    shard_idx: int = 0
    rows_in_shard: int = 0
    total_rows: int = 0
    schema: Optional[pa.Schema] = None
    writer: Optional[pq.ParquetWriter] = None

    def _open(self, schema: pa.Schema) -> None:
        self.schema = schema
        out_path = self.out_dir / f"{self.split_name}-{self.shard_idx:05d}.parquet"
        self.writer = pq.ParquetWriter(out_path, schema, compression=self.compression)
        self.rows_in_shard = 0

    def _close(self) -> None:
        if self.writer is not None:
            self.writer.close()
            self.writer = None

    def write_batch(self, batch: pa.RecordBatch) -> None:
        if batch is None or batch.num_rows == 0:
            return

        if self.schema is None:
            self._open(batch.schema)

        # rotate before writing if already full
        if self.rows_in_shard >= self.rows_per_shard:
            self._close()
            self.shard_idx += 1
            # lazy-open on next write, but we can open now (schema known)
            self._open(self.schema)

        offset = 0
        while offset < batch.num_rows:
            remaining = self.rows_per_shard - self.rows_in_shard
            take = min(remaining, batch.num_rows - offset)
            sub = batch.slice(offset, take)

            if self.writer is None:
                # can happen after rotation
                if self.schema is None:
                    self.schema = sub.schema
                self._open(self.schema)

            # write
            self.writer.write_table(pa.Table.from_batches([sub]))

            self.rows_in_shard += sub.num_rows
            self.total_rows += sub.num_rows
            offset += take

            # rotate if filled exactly
            if self.rows_in_shard >= self.rows_per_shard:
                self._close()
                self.shard_idx += 1
                # keep lazy-open until next write

    def finalize(self) -> int:
        self._close()
        return self.total_rows


def _stream_master_and_write_splits(
    master_dir: Path,
    out_cfg_dir: Path,
    split_to_files: Dict[str, Set[str]],
    batch_rows: int,
    compression: str = "zstd",
) -> Dict[str, int]:
    """
    Single-pass scan over master parquet. Routes each row into 1..N splits
    depending on membership of filename in split_to_files[split].

    Writes parquet shards per split under out_cfg_dir.
    Returns counts per split.
    """
    master = ds.dataset(str(master_dir), format="parquet")
    writers: Dict[str, _SplitWriter] = {}

    # Build Arrow arrays for membership checks per batch:
    # We'll compute masks via pc.is_in(filename_col, value_set=array_of_values)
    # For large sets this could be heavy; our sets are moderate and batch_rows is controlled.
    split_value_arrays: Dict[str, pa.Array] = {}
    for split, files in split_to_files.items():
        if not files:
            continue
        split_value_arrays[split] = pa.array(list(files), type=pa.string())

    # Ensure output dir exists
    out_cfg_dir.mkdir(parents=True, exist_ok=True)

    # scan with a reasonable batch size
    scanner = master.scanner(batch_size=batch_rows)
    for rb in scanner.to_batches():
        # must have filename column for routing
        if "filename" not in rb.schema.names:
            raise RuntimeError("Master parquet must contain a 'filename' column.")

        fn = rb.column(rb.schema.get_field_index("filename"))

        for split, values_arr in split_value_arrays.items():
            mask = pc.is_in(fn, value_set=values_arr)
            # If no matches, skip
            if pc.sum(mask).as_py() == 0:
                continue

            filtered = rb.filter(mask)

            if split not in writers:
                split_out = out_cfg_dir
                writers[split] = _SplitWriter(
                    out_dir=split_out,
                    split_name=split,
                    rows_per_shard=ROWS_PER_SHARD,
                    compression=compression,
                )
            writers[split].write_batch(filtered)

    counts: Dict[str, int] = {}
    for split, w in writers.items():
        counts[split] = w.finalize()
    return counts


def materialize_fewshot_config(
    repo_root: Path,
    base_cfg: str,
    k: int,
    seed: int,
    rows_per_shard: int,
    scan_batch_size: int,
) -> Dict[str, object]:
    """
    Creates <base_cfg>__fs<k> under data/configs by moving k images
    from base ood_train into train.

    ood_test is NEVER modified and stays identical across the base config
    and all few-shot variants, ensuring consistent OOD evaluation.

    Few-shot images are drawn from a single deterministic ordering of ood_train
    (seeded by base_cfg name only, not k), so that
      fs1_images ⊂ fs10_images ⊂ fs100_images ⊂ fsall_images.
    Pass k=-1 (or k >= len(ood_train)) to move all ood_train images (fsall).
    """
    data_dir = repo_root / "data"
    cfgs_dir = data_dir / "configs"
    master_dir = data_dir / "master"

    base_dir = cfgs_dir / base_cfg
    if not base_dir.exists():
        raise FileNotFoundError(f"Base config not found: {base_dir}")

    out_cfg = f"{base_cfg}__fs{k}" if k >= 0 else f"{base_cfg}__fsall"
    out_dir = cfgs_dir / out_cfg

    if out_dir.exists():
        return {"status": "skip_exists", "out_cfg": out_cfg}

    # Load filename sets from base config
    train_files = _read_filenames_from_split(base_dir, "train")
    id_test_files = _read_filenames_from_split(base_dir, "id_test")
    ood_train_files = _read_filenames_from_split(base_dir, "ood_train")
    ood_test_files = _read_filenames_from_split(base_dir, "ood_test")  # never changes

    if not ood_train_files:
        raise RuntimeError(
            f"Base config '{base_cfg}' has no ood_train split. "
            "Regenerate base configs with the updated make_configs.py."
        )

    # Build a deterministic ordering seeded only on (seed, base_cfg) so that
    # all k values draw progressive prefixes of the same list.
    ordering_seed = _stable_seed(seed, base_cfg, 0)  # k=0 → ordering never depends on actual k
    ordered_ood_train = sorted(ood_train_files)  # stable base ordering
    rng = random.Random(ordering_seed)
    rng.shuffle(ordered_ood_train)

    # Determine how many to move
    k_actual = len(ordered_ood_train) if k < 0 else min(k, len(ordered_ood_train))
    chosen = ordered_ood_train[:k_actual]
    chosen_set = set(chosen)

    # Move chosen from ood_train -> train; ood_test is completely untouched
    new_train = set(train_files) | chosen_set
    new_id_test = set(id_test_files)
    new_ood_train = set(ood_train_files) - chosen_set
    new_ood_test = set(ood_test_files)  # unchanged

    # Build split membership map (skip empty splits)
    split_to_files: Dict[str, Set[str]] = {
        "train": new_train,
        "id_test": new_id_test,
        "ood_test": new_ood_test,
    }
    if new_ood_train:
        split_to_files["ood_train"] = new_ood_train

    # Write protocol/manifest
    out_dir.mkdir(parents=True, exist_ok=True)
    proto = {
        "base_config": base_cfg,
        "fewshot_k": k_actual,
        "seed": seed,
        "ordering_seed": ordering_seed,
        "moved_from_ood_train_to_train": sorted(chosen),
        "counts": {
            "train": len(new_train),
            "id_test": len(new_id_test),
            "ood_train": len(new_ood_train),
            "ood_test": len(new_ood_test),
        },
    }
    (out_dir / "protocol.json").write_text(json.dumps(proto, indent=2))

    # Stream master once and write all splits
    # Interpret scan_batch_size as a multiplier; enforce a reasonable minimum.
    batch_rows = max(2048, int(scan_batch_size) * 2048)

    global ROWS_PER_SHARD
    ROWS_PER_SHARD = rows_per_shard

    counts_written = _stream_master_and_write_splits(
        master_dir=master_dir,
        out_cfg_dir=out_dir,
        split_to_files=split_to_files,
        batch_rows=batch_rows,
        compression="zstd",
    )

    proto["counts_written"] = counts_written
    (out_dir / "protocol.json").write_text(json.dumps(proto, indent=2))

    return {"status": "ok", "out_cfg": out_cfg, "counts_written": counts_written}


def parse_fewshot_ks(s: str) -> List[int]:
    """Parse comma-separated k values. Use 'all' or -1 to indicate fsall (all ood_train)."""
    s = s.strip()
    if not s:
        return []
    parts = [p.strip() for p in s.split(",")]
    out: List[int] = []
    for p in parts:
        if not p:
            continue
        if p.lower() == "all":
            out.append(-1)
        else:
            out.append(int(p))
    return out


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--repo_root", required=True, help="path to hf_repo root (contains data/ and tools/)")
    ap.add_argument(
        "--base_configs",
        nargs="+",
        required=True,
        help="base config names (dirs under data/configs)",
    )
    ap.add_argument("--fewshot_ks", required=True, help="comma-separated list, e.g. 10,50,100")
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--rows_per_shard", type=int, default=256)
    ap.add_argument("--scan_batch_size", type=int, default=8, help="controls streaming batch rows (multiplier)")
    args = ap.parse_args()

    repo_root = Path(args.repo_root).resolve()
    if not (repo_root / "data" / "configs").exists():
        raise FileNotFoundError(f"Expected data/configs under {repo_root}")

    ks = parse_fewshot_ks(args.fewshot_ks)
    if not ks:
        raise ValueError("fewshot_ks is empty. Example: --fewshot_ks 10,50,100")

    # Ensure master exists
    master_dir = repo_root / "data" / "master"
    if not master_dir.exists():
        raise FileNotFoundError(f"Master dir missing: {master_dir}")

    for base_cfg in args.base_configs:
        for k in ks:
            out = materialize_fewshot_config(
                repo_root=repo_root,
                base_cfg=base_cfg,
                k=k,
                seed=args.seed,
                rows_per_shard=args.rows_per_shard,
                scan_batch_size=args.scan_batch_size,
            )
            if out["status"] == "skip_exists":
                print(f"Skip (exists): {out['out_cfg']}")
            else:
                print(f"Wrote: {out['out_cfg']}  counts={out.get('counts_written')}")

    print("Done.")


if __name__ == "__main__":
    # global used inside streaming writer constructor to avoid passing around everywhere
    ROWS_PER_SHARD = 256
    main()

