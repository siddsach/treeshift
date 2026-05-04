# Container Setup (Docker)

Docker image for running Plain-DETR and GroundingDINO on the tree-distribution-shift benchmark.

## Requirements

- Host with NVIDIA GPU (e.g. Tesla T4)
- Docker + NVIDIA Container Toolkit
- CUDA driver ≥ 11.7 (`nvidia-smi` → check "CUDA Version")

## Step 1 — Build the image

```bash
cd ~/tree-distribution-shift
sudo chown -R $(whoami):$(whoami) .
git pull

# Public repo (or PyPI fallback)
docker build -t tree-shift:latest .

# Private repo — pass a GitHub token
docker build --build-arg GH_TOKEN=ghp_YOURTOKEN -t tree-shift:latest .
```

## Step 2 — Start the container

```bash
docker run -it --gpus all \
  --shm-size=2g \
  -v $(pwd):/workspace \
  -w /workspace \
  tree-shift:latest \
  bash
```

> `--shm-size=2g` prevents "bus error" / "No space left on device" from PyTorch DataLoader shared memory.

## Step 3 — Export data

```bash
# Delete stale exports
rm -rf ./coco_export

# Export (downloads from HuggingFace on first run)
tree-shift export --config biome_Rajasthan_train_WET__ood_DRY --out ./coco_export

# Verify
python -c "
import json
with open('coco_export/biome_Rajasthan_train_WET__ood_DRY/val/annotations/instances_val.json') as f:
    d = json.load(f)
print('category_ids:', set(a['category_id'] for a in d['annotations']))
print('Categories:', d['categories'])
print('Annotations:', len(d['annotations']))
"
```

Expected: `category_ids: {1}`, Categories: `[{'id': 1, 'name': 'tree', ...}]`

## Step 4 — Train + Eval Plain-DETR

```bash
rm -rf ./plain_detr_output

python plain_detr.py run --config biome_Rajasthan_train_WET__ood_DRY \
  --coco-dir ./coco_export --train --eval-val --eval-ood \
  --output-dir ./plain_detr_output --backbone resnet50 \
  --epochs 30 --not_auto_resume
```

## Step 5 — Eval GroundingDINO (zero-shot, no training needed)

```bash
python grounding_dino.py run --config biome_Rajasthan_train_WET__ood_DRY \
  --coco-dir ./coco_export --eval-val --eval-ood
```

## Step 6 — Fix file ownership (on host, after exiting container)

```bash
sudo chown -R $(whoami):$(whoami) coco_export plain_detr_output output_grounding_dino
```

## Available configs

```bash
tree-shift list-configs
```

| Config | Shift | Train on | OOD from |
|--------|-------|----------|----------|
| `biome_Rajasthan_train_WET__ood_DRY` | Biome | Rajasthan WET | Rajasthan DRY |
| `biome_Rajasthan_train_DRY__ood_WET` | Biome | Rajasthan DRY | Rajasthan WET |
| `intl_train_IN__ood_US` | Country | India | US |
| `intl_train_US__ood_IN` | Country | US | India |

## Host CUDA Mismatch

The default image uses **Ubuntu 22.04 + Python 3.10 + CUDA 11.7** (DinoV3 compatibility). If your host driver differs:

| Host CUDA | Change Dockerfile `FROM` to |
|-----------|----------------------------|
| 11.6 | `nvidia/cuda:11.6.2-cudnn8-devel-ubuntu20.04` + Python 3.9 (requires DinoV3 patch) |
| 11.7 | Default: `nvidia/cuda:11.7.1-cudnn8-devel-ubuntu22.04` |
| 11.8 | `nvidia/cuda:11.8.0-cudnn8-devel-ubuntu22.04` + `cu118` for PyTorch |
| 12.1 | `nvidia/cuda:12.1.0-cudnn8-devel-ubuntu22.04` + `cu121` for PyTorch |
