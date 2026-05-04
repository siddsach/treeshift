#!/bin/bash
#SBATCH --job-name=plaindetr-dinov3-train
#SBATCH --partition=serc
#SBATCH --gres=gpu:4
#SBATCH --cpus-per-task=48
#SBATCH --mem=512G
#SBATCH --time=72:00:00
#SBATCH --output=slurm-train-%j.out
#SBATCH --error=slurm-train-%j.err

set -euo pipefail

# If your cluster needs modules, uncomment/edit:
# module load apptainer

apptainer exec --nv \
  --bind /scratch/groups/dlobell/aadityan/Plain-DETR:/opt/Plain-DETR \
  --bind /scratch/groups/dlobell/aadityan/coco2:/data/coco \
  --bind /scratch/groups/dlobell/aadityan/dinov3:/data/dinov3 \
  --bind /scratch/groups/dlobell/aadityan/dino_weights/dino_weights_7b16.pth:/data/dino_weights.pth \
  --bind /scratch/groups/dlobell/aadityan/new_detr_outputs:/data/output \
  plaindetr_dinov3.sif \
  torchrun --standalone --nproc_per_node=4 /opt/Plain-DETR/main.py \
    --dataset_file coco \
    --coco_path /data/coco \
    --output_dir /data/output/dinov3_7b_plain_detr_run \
    --device cuda \
    --num_classes 1 \
    --epochs 50 \
    --early_stop \
    --early_stop_min_epochs 30 \
    --early_stop_patience 15 \
    --early_stop_min_delta 0.0 \
    --lr 2e-4 \
    --lr_backbone 0 \
    --lr_drop 40 \
    --batch_size 3 \
    --weight_decay 1e-4 \
    --num_workers 4 \
    --clip_max_norm 0.1 \
    --save_ckpt_every_images 400 \
    --backbone dinov3 \
    --dinov3_repo /data/dinov3 \
    --dinov3_model dinov3_vit7b16 \
    --dinov3_weights /data/dino_weights.pth \
    --layers_to_use 9 19 29 39 \
    --num_feature_levels 1 \
    --proposal_feature_levels 1 \
    --proposal_in_stride 16 \
    --proposal_tgt_strides 16 \
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
    --k_one2many 0 \
    --n_windows_sqrt 3