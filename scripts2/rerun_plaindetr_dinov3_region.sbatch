#!/bin/bash
#SBATCH --job-name=ev_plaindetr_dinov3_region
#SBATCH --partition=serc
#SBATCH --gres=gpu:1
#SBATCH --constraint=GPU_SKU:A100_SXM4
#SBATCH --cpus-per-task=8
#SBATCH --mem=48G
#SBATCH --time=12:00:00
#SBATCH --exclude=sh03-14n13,sh03-14n07
#SBATCH --output=/scratch/groups/dlobell/aadityan/tree-distribution-shift/logs/eval_plaindetr_dinov3_region_fix_%j.out
#SBATCH --error=/scratch/groups/dlobell/aadityan/tree-distribution-shift/logs/eval_plaindetr_dinov3_region_fix_%j.err

set -uo pipefail
REPO="/scratch/groups/dlobell/aadityan/tree-distribution-shift"
OUT="$REPO/final_outputs/plaindetr_dinov3_region_North_South_1gpu40ep_17593285"
cd "$REPO"

CKPT="$OUT/checkpoint.pth"
[[ -f "$OUT/checkpoint_best.pth" ]] && CKPT="$OUT/checkpoint_best.pth"
echo "Checkpoint: $CKPT"

MASTER_PORT=$(( 29400 + ${SLURM_JOB_ID:-$$} % 10000 ))

apptainer exec --nv \
  --bind "$REPO:/workspace" \
  --env "HF_HOME=/workspace/.hf_cache/huggingface" \
  "$REPO/tree-shift.sif" \
  torchrun --standalone --nproc_per_node=1 --master_port=$MASTER_PORT \
  plain_detr.py run \
  --config "region_train_North__ood_South" \
  --eval-val --eval-ood --eval-ood-train \
  --model-path "$CKPT" \
  --output-dir "$OUT" \
  --decoder_use_checkpoint \
  --backbone dinov3 \
  --dinov3_repo /opt/dinov3 \
  --dinov3_model dinov3_vits16 \
  --dinov3_weights "/scratch/groups/dlobell/aadityan/dino_weights/dinov3_vits16_pretrain_lvd1689m-08c60483.pth" \
  --layers_to_use 3 6 9 11 \
  --num_feature_levels 1 \
  --proposal_feature_levels 1 \
  --proposal_in_stride 16 \
  --proposal_tgt_strides 16 \
  --n_windows_sqrt 3 \
  --add_transformer_encoder \
  --num_encoder_layers 6 \
  --norm_type pre_norm \
  --hidden_dim 256 \
  --dim_feedforward 2048 \
  --nheads 8 \
  --decoder_type global_rpe_decomp \
  --decoder_rpe_type linear \
  --decoder_rpe_hidden_dim 256 \
  --two_stage \
  --mixed_selection \
  --with_box_refine \
  --num_queries_one2one 100 \
  --num_queries_one2many 0 \
  --k_one2many 0

echo "Done: $OUT"
