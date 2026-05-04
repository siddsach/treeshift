# Self-contained MM-Grounding-DINO config for tree detection fine-tuning.
# No _base_ inheritance — all settings are defined here so mmdet's config
# tree is not required at runtime.
#
# Data paths (data_root, ann_file) are PLACEHOLDERS and must be overridden
# programmatically before training.  See grounding_dino.py for details.

default_scope = 'mmdet'

# ---------------------------------------------------------------------------
# Model
# ---------------------------------------------------------------------------

lang_model_name = 'bert-base-uncased'

model = dict(
    type='GroundingDINO',
    num_queries=900,
    with_box_refine=True,
    as_two_stage=True,
    data_preprocessor=dict(
        type='DetDataPreprocessor',
        mean=[123.675, 116.28, 103.53],
        std=[58.395, 57.12, 57.375],
        bgr_to_rgb=True,
        pad_mask=False,
        pad_size_divisor=32,  # Swin backbone expects divisible dims; ensures consistent batch shapes
    ),
    language_model=dict(
        type='BertModel',
        name=lang_model_name,
        max_tokens=256,
        pad_to_max=False,
        use_sub_sentence_represent=True,
        special_tokens_list=['[CLS]', '[SEP]', '.', '?'],
        add_pooling_layer=False,
    ),
    backbone=dict(
        type='SwinTransformer',
        embed_dims=96,
        depths=[2, 2, 6, 2],
        num_heads=[3, 6, 12, 24],
        window_size=7,
        mlp_ratio=4,
        qkv_bias=True,
        qk_scale=None,
        drop_rate=0.,
        attn_drop_rate=0.,
        drop_path_rate=0.2,
        patch_norm=True,
        out_indices=(1, 2, 3),
        with_cp=False,
        convert_weights=False,
    ),
    neck=dict(
        type='ChannelMapper',
        in_channels=[192, 384, 768],
        kernel_size=1,
        out_channels=256,
        act_cfg=None,
        bias=True,
        norm_cfg=dict(type='GN', num_groups=32),
        num_outs=4,
    ),
    encoder=dict(
        num_layers=6,
        # Avoid fairscale dependency in HPC container runtime.
        num_cp=0,
        layer_cfg=dict(
            self_attn_cfg=dict(embed_dims=256, num_levels=4, dropout=0.0),
            ffn_cfg=dict(
                embed_dims=256, feedforward_channels=2048, ffn_drop=0.0)),
        text_layer_cfg=dict(
            self_attn_cfg=dict(num_heads=4, embed_dims=256, dropout=0.0),
            ffn_cfg=dict(
                embed_dims=256, feedforward_channels=1024, ffn_drop=0.0)),
        fusion_layer_cfg=dict(
            v_dim=256, l_dim=256, embed_dim=1024,
            num_heads=4, init_values=1e-4),
    ),
    decoder=dict(
        num_layers=6,
        return_intermediate=True,
        layer_cfg=dict(
            self_attn_cfg=dict(embed_dims=256, num_heads=8, dropout=0.0),
            cross_attn_text_cfg=dict(embed_dims=256, num_heads=8, dropout=0.0),
            cross_attn_cfg=dict(embed_dims=256, num_heads=8, dropout=0.0),
            ffn_cfg=dict(
                embed_dims=256, feedforward_channels=2048, ffn_drop=0.0)),
        post_norm_cfg=None,
    ),
    positional_encoding=dict(
        num_feats=128, normalize=True, offset=0.0, temperature=20),
    bbox_head=dict(
        type='GroundingDINOHead',
        num_classes=1,
        sync_cls_avg_factor=True,
        contrastive_cfg=dict(max_text_len=256, log_scale='auto', bias=True),
        loss_cls=dict(
            type='FocalLoss', use_sigmoid=True,
            gamma=2.0, alpha=0.25, loss_weight=1.0),
        loss_bbox=dict(type='L1Loss', loss_weight=5.0),
    ),
    dn_cfg=dict(
        label_noise_scale=0.5,
        box_noise_scale=1.0,
        group_cfg=dict(dynamic=True, num_groups=None, num_dn_queries=100)),
    train_cfg=dict(
        assigner=dict(
            type='HungarianAssigner',
            match_costs=[
                dict(type='BinaryFocalLossCost', weight=2.0),
                dict(type='BBoxL1Cost', weight=5.0, box_format='xywh'),
                dict(type='IoUCost', iou_mode='giou', weight=2.0),
            ],
        ),
    ),
    test_cfg=dict(max_per_img=300),
)

# ---------------------------------------------------------------------------
# Data pipelines
# ---------------------------------------------------------------------------

# Fixed resize (800, 1333) avoids multi_scale_deform_attn AssertionError:
# (spatial_shapes[:, 0] * spatial_shapes[:, 1]).sum() == num_value.
# Complex RandomChoiceResize + RandomCrop with variable scales can produce
# incompatible spatial shapes for deformable attention.
train_pipeline = [
    dict(type='LoadImageFromFile'),
    dict(type='LoadAnnotations', with_bbox=True),
    dict(type='RandomFlip', prob=0.5),
    dict(type='FilterAnnotations', min_gt_bbox_wh=(1e-2, 1e-2)),
    # Force all chips to exactly 800×800 (no keep_ratio) so every image in a
    # batch has identical spatial shapes — prevents the deformable-attention
    # assertion: (spatial_shapes[:,0]*spatial_shapes[:,1]).sum() == num_value.
    dict(type='Resize', scale=(800, 800), keep_ratio=False),
    dict(
        type='PackDetInputs',
        meta_keys=('img_id', 'img_path', 'ori_shape', 'img_shape',
                   'scale_factor', 'flip', 'flip_direction', 'text',
                   'custom_entities')),
]

test_pipeline = [
    dict(type='LoadImageFromFile', backend_args=None, imdecode_backend='pillow'),
    dict(type='FixScaleResize', scale=(800, 1333), keep_ratio=True,
         backend='pillow'),
    dict(type='LoadAnnotations', with_bbox=True),
    dict(
        type='PackDetInputs',
        meta_keys=('img_id', 'img_path', 'ori_shape', 'img_shape',
                   'scale_factor', 'text', 'custom_entities',
                   'tokens_positive')),
]

# ---------------------------------------------------------------------------
# Data loaders  (data_root is overridden at runtime)
# ---------------------------------------------------------------------------

data_root = 'PLACEHOLDER'

_metainfo = dict(classes=('tree', ), palette=[(34, 139, 34)])

train_dataloader = dict(
    batch_size=4,
    num_workers=2,
    persistent_workers=False,
    drop_last=True,
    sampler=dict(type='DefaultSampler', shuffle=True),
    dataset=dict(
        type='CocoDataset',
        data_root=data_root,
        metainfo=_metainfo,
        ann_file='train/annotations/instances_train.json',
        data_prefix=dict(img='train/images/'),
        return_classes=True,
        filter_cfg=dict(filter_empty_gt=False, min_size=32),
        pipeline=train_pipeline,
    ),
)

val_dataloader = dict(
    batch_size=1,
    num_workers=2,
    persistent_workers=False,
    drop_last=False,
    sampler=dict(type='DefaultSampler', shuffle=False),
    dataset=dict(
        type='CocoDataset',
        data_root=data_root,
        metainfo=_metainfo,
        ann_file='val/annotations/instances_val.json',
        data_prefix=dict(img='val/images/'),
        return_classes=True,
        test_mode=True,
        pipeline=test_pipeline,
    ),
)

test_dataloader = val_dataloader

# ---------------------------------------------------------------------------
# Evaluators
# ---------------------------------------------------------------------------

val_evaluator = dict(
    type='CocoMetric',
    ann_file='PLACEHOLDER',
    metric='bbox',
    format_only=False,
)
test_evaluator = val_evaluator

# ---------------------------------------------------------------------------
# Optimizer — freeze backbone and language model for small-dataset fine-tuning
# ---------------------------------------------------------------------------

optim_wrapper = dict(
    type='OptimWrapper',
    optimizer=dict(type='AdamW', lr=0.0001, weight_decay=0.0001),
    clip_grad=dict(max_norm=0.1, norm_type=2),
    paramwise_cfg=dict(
        custom_keys={
            'absolute_pos_embed': dict(decay_mult=0.),
            'backbone': dict(lr_mult=0.0),
            'language_model': dict(lr_mult=0.0),
        }),
)

# ---------------------------------------------------------------------------
# LR schedule
# ---------------------------------------------------------------------------

max_epochs = 20

param_scheduler = [
    dict(
        type='MultiStepLR',
        begin=0,
        end=max_epochs,
        by_epoch=True,
        milestones=[15],
        gamma=0.1),
]

# ---------------------------------------------------------------------------
# Training / validation loop
# ---------------------------------------------------------------------------

train_cfg = dict(type='EpochBasedTrainLoop', max_epochs=max_epochs,
                 val_interval=1)
val_cfg = dict(type='ValLoop')
test_cfg = dict(type='TestLoop')

# ---------------------------------------------------------------------------
# Runtime
# ---------------------------------------------------------------------------

default_hooks = dict(
    timer=dict(type='IterTimerHook'),
    logger=dict(type='LoggerHook', interval=50),
    param_scheduler=dict(type='ParamSchedulerHook'),
    checkpoint=dict(type='CheckpointHook', interval=1,
                    max_keep_ckpts=3, save_best='auto'),
    sampler_seed=dict(type='DistSamplerSeedHook'),
    visualization=dict(type='DetVisualizationHook'),
)

env_cfg = dict(
    cudnn_benchmark=False,
    mp_cfg=dict(mp_start_method='fork', opencv_num_threads=0),
    dist_cfg=dict(backend='nccl'),
)

vis_backends = [dict(type='LocalVisBackend')]
visualizer = dict(
    type='DetLocalVisualizer', vis_backends=vis_backends, name='visualizer')
log_processor = dict(type='LogProcessor', window_size=50, by_epoch=True)
log_level = 'INFO'
resume = False

# ---------------------------------------------------------------------------
# Pretrained checkpoint (MM-Grounding-DINO-T, O365+GoldG+GRIT+V3Det)
# ---------------------------------------------------------------------------

load_from = (
    'https://download.openmmlab.com/mmdetection/v3.0/mm_grounding_dino/'
    'grounding_dino_swin-t_pretrain_obj365_goldg_grit9m_v3det/'
    'grounding_dino_swin-t_pretrain_obj365_goldg_grit9m_v3det_20231204_095047-b448804b.pth'
)
