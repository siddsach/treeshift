#!/bin/bash
# Clean and rebuild tree-shift Docker image.
# Run from project root: ./scripts/rebuild_docker.sh

set -e
cd "$(dirname "$0")/.."

echo "=== Cleaning Docker ==="
docker system prune -f
docker builder prune -f 2>/dev/null || true

echo ""
echo "=== Reducing build context (moving large dirs aside) ==="
for d in coco_export dino_weights plain_detr_output output_grounding_dino outputs; do
  if [ -d "$d" ] && [ ! -L "$d" ]; then
    echo "  Moving $d to /tmp/${d}_backup"
    mv "$d" "/tmp/${d}_backup" 2>/dev/null || rm -rf "$d"
  fi
done

echo ""
echo "=== Build context size ==="
du -sh . 2>/dev/null || true

echo ""
echo "=== Building tree-shift:latest ==="
docker build -t tree-shift:latest .

echo ""
echo "=== Restoring moved dirs ==="
for d in coco_export dino_weights plain_detr_output output_grounding_dino outputs; do
  if [ -d "/tmp/${d}_backup" ]; then
    echo "  Restoring $d"
    mv "/tmp/${d}_backup" "$d"
  fi
done

echo ""
echo "Done. Run: docker run -it --gpus all --shm-size=2g -v \$(pwd):/workspace -w /workspace tree-shift:latest bash"
