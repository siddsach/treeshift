# Data and External Artifacts

Raw data and generated COCO exports are intentionally not stored in this Git repository.

## Expected external inputs

The experiment code expects TreeShift data in COCO layout, usually produced by the `tree-shift` package or by the local dataset tools in `tools/data/`.

Distribution-shift config exports contain these splits:

- `train/images` and `train/annotations/instances_train.json`
- `id_test/images` and `id_test/annotations/instances_id_test.json`
- `ood_train/images` and `ood_train/annotations/instances_ood_train.json`
- `ood_test/images` and `ood_test/annotations/instances_ood_test.json`

The fixed India non-shift baseline config is `india_random_80_20`. It contains
only `train` and `id_test`: 80% of India for training and 20% of India for the
held-out ID test split. It intentionally has no `ood_train` or `ood_test` split,
and generated experiment scripts for this config run only `--train --eval-val`.

The wrappers also support a legacy `val` split. If `val` is missing, `shared_utils.ensure_tree_shift_export()` creates or resolves `val` as the ID test split where possible.

## Dataset/config tooling

The files in `tools/data/` reproduce the split/config construction logic used for the experiments. They require external raw inputs such as `metadata.csv`, `world_images/`, and master parquet shards. Those inputs should be released separately, for example through Hugging Face, Stanford storage, or Zenodo.

Important tools:

- `make_master_configs.py`: builds base country, biome, elevation, region shift, and fixed India non-shift configs.
- `make_configs.py`: adds few-shot variants by moving deterministic samples from `ood_train` into `train`.
- `export_coco.py`: exports a named dataset config into COCO layout.

## Artifacts intentionally excluded from Git

- Raw images and annotations.
- `coco_export/` and other generated data folders.
- `metadata.csv` unless a small public sample is added intentionally.
- Model outputs and final paper tables.
- Pretrained model weights and checkpoints.
- Apptainer `.sif` images.

## Canonical configs

The final benchmark centers on these base shifts:

- `biome_Rajasthan_train_WET__ood_DRY`
- `elev_Karnataka_train_HIGH__ood_LOW`
- `intl_train_US__ood_IN`
- `region_train_North__ood_South`

Few-shot variants use suffixes `__fs1`, `__fs10`, `__fs100`, and `__fsall`.
The fixed non-shift India baseline is `india_random_80_20`.
