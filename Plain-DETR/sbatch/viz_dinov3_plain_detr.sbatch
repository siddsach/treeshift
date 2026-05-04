#!/bin/bash
#SBATCH --job-name=plaindetr-dinov3-viz
#SBATCH --partition=serc
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=16
#SBATCH --mem=256G
#SBATCH --time=72:00:00
#SBATCH --output=slurm-viz-%j.out
#SBATCH --error=slurm-viz-%j.err

set -euo pipefail

# module load apptainer

apptainer exec --nv \
  --bind /scratch/groups/dlobell/aadityan/Plain-DETR:/opt/Plain-DETR \
  --bind /scratch/groups/dlobell/aadityan/coco2:/data/coco \
  --bind /scratch/groups/dlobell/aadityan/dinov3:/data/dinov3 \
  --bind /scratch/groups/dlobell/aadityan/dino_weights/dino_weights_7b16.pth:/data/dino_weights.pth \
  --bind /scratch/groups/dlobell/aadityan/new_detr_outputs:/data/output \
  plaindetr_dinov3.sif \
  python /opt/Plain-DETR/viz_pred_vs_gt_coco.py \
    --resume /data/output/dinov3_7b_plain_detr_run/checkpoint.pth \
    --dataset_file coco \
    --coco_path /data/coco \
    --device cuda \
    --num_classes 1 \
    --num_workers 16 \
    --batch_size 1 \
    --backbone dinov3 \
    --dinov3_repo /data/dinov3 \
    --dinov3_model dinov3_vit7b16 \
    --dinov3_weights /data/dino_weights.pth \
    --layers_to_use 9 19 29 39 \
    --n_windows_sqrt 3 \
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
    --viz_output_dir /data/output/dinov3_7b_plain_detr_run/viz \
    --do_eval \
    --eval_output_dir /data/output/dinov3_7b_plain_detr_run/eval \
    --num_images 50 \
    --eval_max_batches 50 \
    --score_thresh 0.3