#!/usr/bin/env python3
"""
GroundingDINO Training and Evaluation Script for tree-distribution-shift project.
Data is loaded via the ``tree_shift`` pip package (>= 0.2.0).

Supports two modes:

1. **Zero-shot** evaluation (IDEA-Research GroundingDINO, no training):
       python grounding_dino.py run \
           --config intl_train_IN__ood_US \
           --eval-val --eval-ood

2. **Fine-tune** MM-Grounding-DINO (MMDetection) then evaluate:
       python grounding_dino.py run \
           --config intl_train_IN__ood_US \
           --train --eval-val --eval-ood

   Evaluate a previously fine-tuned checkpoint:
       python grounding_dino.py run \
           --config intl_train_IN__ood_US \
           --eval-val --eval-ood \
           --model-path ./output_grounding_dino/best_coco_bbox_mAP_epoch_*.pth

   Customise training hyper-parameters:
       python grounding_dino.py run \
           --config intl_train_IN__ood_US \
           --train --eval-val --eval-ood \
           --epochs 30 --batch-size 2 --lr 0.00005
"""

import sys
import os
import json
import argparse
import time

os.environ["PYTHONUNBUFFERED"] = "1"

import numpy as np
import torch
from pathlib import Path

# Force transformers to recognise PyTorch before any GroundingDINO import
import transformers  # noqa: F401

# ── IDEA-Research GroundingDINO (zero-shot eval) ──────────────────────────
_GDINO_DIR = os.environ.get(
    "GROUNDING_DINO_DIR",
    os.path.join(os.path.dirname(os.path.abspath(__file__)), "GroundingDINO"),
)
sys.path.insert(0, _GDINO_DIR)

try:
    from groundingdino.util.inference import load_model as _gdino_load_model
    from groundingdino.util.misc import collate_fn as _gdino_collate_fn
    from groundingdino.util import box_ops as _gdino_box_ops
    import groundingdino.datasets.transforms as GDT
    GDINO_AVAILABLE = True
except ImportError as e:
    print(f"Warning: IDEA-Research GroundingDINO not available: {e}")
    GDINO_AVAILABLE = False

# ── MMDetection MM-Grounding-DINO (training + fine-tuned eval) ────────────
try:
    import mmdet  # noqa: F401  – registers model types
    from mmengine.config import Config as MMConfig
    from mmengine.runner import Runner
    from mmdet.apis import DetInferencer
    MMDET_AVAILABLE = True
except ImportError:
    MMDET_AVAILABLE = False

# ── HuggingFace Hub defaults (zero-shot) ─────────────────────────────────
_HF_REPO_ID = "ShilongLiu/GroundingDINO"
_HF_FILENAME = "groundingdino_swint_ogc.pth"

# ── Shared project utilities ──────────────────────────────────────────────
from shared_utils import (
    resolve_coco_paths,
    ensure_tree_shift_export,
    compute_tree_detection_metrics_from_boxes,
    generate_eval_summary_table,
    evaluate_detections,
)

_CONFIG_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "configs")


# ═══════════════════════════════════════════════════════════════════════════
# Zero-shot helpers (IDEA-Research GroundingDINO)
# ═══════════════════════════════════════════════════════════════════════════

def _load_model_hf(model_config_path, repo_id, filename, device="cuda"):
    """Load IDEA-Research GroundingDINO model from HuggingFace Hub."""
    from huggingface_hub import hf_hub_download
    from groundingdino.models import build_model
    from groundingdino.util.slconfig import SLConfig
    from groundingdino.util.misc import clean_state_dict

    args = SLConfig.fromfile(model_config_path)
    args.device = device
    model = build_model(args)
    cache_file = hf_hub_download(repo_id=repo_id, filename=filename)
    checkpoint = torch.load(cache_file, map_location="cpu")
    model.load_state_dict(clean_state_dict(checkpoint["model"]), strict=False)
    model.eval()
    return model


class _CocoDetectionZeroShot(torch.utils.data.Dataset):
    """Minimal COCO dataset for zero-shot GroundingDINO evaluation."""

    def __init__(self, img_folder, ann_file, transforms=None):
        from pycocotools.coco import COCO
        self.img_folder = img_folder
        self.coco = COCO(ann_file)
        self.ids = sorted(self.coco.imgs.keys())
        self._transforms = transforms

    def __len__(self):
        return len(self.ids)

    def __getitem__(self, idx):
        from PIL import Image
        image_id = self.ids[idx]
        img_info = self.coco.loadImgs(image_id)[0]
        img_path = os.path.join(self.img_folder, img_info["file_name"])
        img = Image.open(img_path).convert("RGB")

        ann_ids = self.coco.getAnnIds(imgIds=[image_id])
        anns = self.coco.loadAnns(ann_ids)

        w, h = img.size
        boxes = [obj["bbox"] for obj in anns]
        if len(boxes) > 0:
            boxes = torch.as_tensor(boxes, dtype=torch.float32).reshape(-1, 4)
            boxes[:, 2:] += boxes[:, :2]
            boxes[:, 0::2].clamp_(min=0, max=w)
            boxes[:, 1::2].clamp_(min=0, max=h)
            keep = (boxes[:, 3] > boxes[:, 1]) & (boxes[:, 2] > boxes[:, 0])
            boxes = boxes[keep]
        else:
            boxes = torch.zeros((0, 4), dtype=torch.float32)

        target = {
            "image_id": image_id,
            "boxes": boxes,
            "orig_size": torch.as_tensor([int(h), int(w)]),
        }
        if self._transforms is not None:
            img, target = self._transforms(img, target)
        return img, target


def evaluate_split_zeroshot(
    model, split_name, img_folder, ann_file, text_prompt,
    box_threshold, text_threshold, device, output_dir,
    score_threshold=0.3, num_workers=0,
):
    """Zero-shot evaluation using IDEA-Research GroundingDINO."""
    from pycocotools.cocoeval import COCOeval
    from torch.utils.data import DataLoader

    transform = GDT.Compose([
        GDT.RandomResize([800], max_size=1333),
        GDT.ToTensor(),
        GDT.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
    ])

    dataset = _CocoDetectionZeroShot(img_folder, ann_file, transforms=transform)
    cat_ids = dataset.coco.getCatIds()
    category_id = int(cat_ids[0]) if cat_ids else 1

    data_loader = DataLoader(
        dataset, batch_size=1, shuffle=False, num_workers=num_workers,
        pin_memory=False, collate_fn=_gdino_collate_fn,
    )

    eval_output = os.path.join(output_dir, f"eval_{split_name}")
    os.makedirs(eval_output, exist_ok=True)

    print(f"  Text prompt: \"{text_prompt}\"")
    print(f"  Box threshold: {box_threshold}, Text threshold: {text_threshold}")
    print(f"  Images: {len(dataset)}")

    model.eval()
    all_coco_results = []
    image_records = []

    start = time.time()
    for i, (images, targets) in enumerate(data_loader):
        images = images.tensors.to(device)
        bs = images.shape[0]
        captions = [text_prompt] * bs

        with torch.no_grad():
            outputs = model(images, captions=captions)

        pred_logits = outputs["pred_logits"].sigmoid()
        pred_boxes_out = outputs["pred_boxes"]

        for b in range(bs):
            target = targets[b]
            image_id = target["image_id"]
            orig_h, orig_w = target["orig_size"].tolist()

            logits = pred_logits[b]
            boxes_cxcywh = pred_boxes_out[b]

            max_logits, _ = logits.max(dim=-1)
            keep = max_logits > box_threshold
            scores = max_logits[keep].cpu().numpy()
            boxes_kept = boxes_cxcywh[keep]

            boxes_xyxy = _gdino_box_ops.box_cxcywh_to_xyxy(boxes_kept)
            boxes_xyxy[:, 0::2] *= orig_w
            boxes_xyxy[:, 1::2] *= orig_h
            boxes_xyxy = boxes_xyxy.cpu().numpy()

            boxes_xywh = boxes_xyxy.copy()
            boxes_xywh[:, 2] -= boxes_xywh[:, 0]
            boxes_xywh[:, 3] -= boxes_xywh[:, 1]

            for j in range(len(scores)):
                all_coco_results.append({
                    "image_id": int(image_id),
                    "category_id": category_id,
                    "bbox": boxes_xywh[j].tolist(),
                    "score": float(scores[j]),
                })

            ann_ids = dataset.coco.getAnnIds(imgIds=[int(image_id)])
            anns = dataset.coco.loadAnns(ann_ids)
            gt_boxes = [ann["bbox"] for ann in anns]

            sort_idx = np.argsort(-scores)
            scores_sorted = scores[sort_idx]
            pred_xywh_sorted = boxes_xywh[sort_idx].tolist()
            score_mask = scores_sorted >= score_threshold
            pred_filtered = [pb for pb, m in zip(pred_xywh_sorted, score_mask) if m]
            scores_filtered = [float(s) for s, m in zip(scores_sorted, score_mask) if m]

            img_info = dataset.coco.loadImgs(int(image_id))[0]
            image_records.append({
                "image_id": int(image_id),
                "filename": img_info.get("file_name", ""),
                "gt_boxes": gt_boxes,
                "pred_boxes": pred_filtered,
                "pred_scores": scores_filtered,
            })

        if (i + 1) % 50 == 0:
            elapsed = time.time() - start
            eta = elapsed / (i + 1) * (len(data_loader) - i - 1)
            print(f"  [{i+1}/{len(data_loader)}] elapsed={elapsed:.1f}s ETA={eta:.1f}s")

    elapsed = time.time() - start
    print(f"  Inference done in {elapsed:.1f}s ({len(dataset)} images)")

    coco_bbox_stats = None
    if len(all_coco_results) > 0:
        coco_dt_path = os.path.join(eval_output, "coco_dt_results.json")
        with open(coco_dt_path, "w") as f:
            json.dump(all_coco_results, f)
        coco_gt = dataset.coco
        coco_dt = coco_gt.loadRes(coco_dt_path)
        coco_eval = COCOeval(coco_gt, coco_dt, "bbox")
        coco_eval.evaluate()
        coco_eval.accumulate()
        coco_eval.summarize()
        s = coco_eval.stats
        # pycocotools returns stats in [0, 1]; convert to [0, 100] to match
        # Detectron2's COCOEvaluator convention. -1.0 means no objects → None.
        def _ap_pct(v):
            return None if float(v) < 0 else float(v) * 100
        coco_bbox_stats = {
            "AP": _ap_pct(s[0]), "AP50": _ap_pct(s[1]), "AP75": _ap_pct(s[2]),
            "APs": _ap_pct(s[3]), "APm": _ap_pct(s[4]), "APl": _ap_pct(s[5]),
        }
    else:
        print("  Warning: no detections produced.")
        coco_bbox_stats = {"AP": 0, "AP50": 0, "AP75": 0, "APs": 0, "APm": 0, "APl": 0}

    return evaluate_detections(
        image_records, split_name=split_name, output_dir=eval_output,
        iou_threshold=0.50, coco_bbox_stats=coco_bbox_stats,
    )


# ═══════════════════════════════════════════════════════════════════════════
# MMDetection helpers (training + fine-tuned eval)
# ═══════════════════════════════════════════════════════════════════════════

def _build_mmdet_cfg(
    data_root, output_dir, train_ann_file, train_img_dir, val_ann_file, val_img_dir,
    epochs=20, batch_size=4, lr=0.0001, num_workers=2, pretrained_weights=None,
    early_stopping_patience=0, gradient_checkpointing=False, train_val_interval=0,
):
    """Load the shipped config and override data paths / hyper-parameters."""
    cfg_path = os.path.join(_CONFIG_DIR, "mm_grounding_dino_tree.py")
    if not os.path.exists(cfg_path):
        raise FileNotFoundError(
            f"MM-Grounding-DINO config not found: {cfg_path}\n"
            "Expected at configs/mm_grounding_dino_tree.py"
        )

    cfg = MMConfig.fromfile(cfg_path)
    train_ann_rel = os.path.relpath(train_ann_file, data_root)
    train_img_rel = os.path.relpath(train_img_dir, data_root)
    val_ann_rel = os.path.relpath(val_ann_file, data_root)
    val_img_rel = os.path.relpath(val_img_dir, data_root)

    # ── Data paths ────────────────────────────────────────────────────────
    cfg.data_root = data_root
    cfg.train_dataloader.dataset.data_root = data_root
    cfg.train_dataloader.dataset.ann_file = train_ann_rel
    cfg.train_dataloader.dataset.data_prefix = dict(img=train_img_rel.rstrip("/") + "/")
    cfg.val_dataloader.dataset.data_root = data_root
    cfg.val_dataloader.dataset.ann_file = val_ann_rel
    cfg.val_dataloader.dataset.data_prefix = dict(img=val_img_rel.rstrip("/") + "/")
    cfg.test_dataloader.dataset.data_root = data_root
    cfg.test_dataloader.dataset.ann_file = val_ann_rel
    cfg.test_dataloader.dataset.data_prefix = dict(img=val_img_rel.rstrip("/") + "/")
    cfg.val_evaluator.ann_file = val_ann_file
    cfg.test_evaluator.ann_file = val_ann_file

    tree_metainfo = dict(classes=("tree",), palette=[(34, 139, 34)])
    cfg.train_dataloader.dataset.metainfo = tree_metainfo
    cfg.val_dataloader.dataset.metainfo = tree_metainfo
    cfg.test_dataloader.dataset.metainfo = tree_metainfo
    cfg.model.bbox_head.num_classes = 1

    # ── Hyper-parameters ──────────────────────────────────────────────────
    cfg.train_dataloader.batch_size = batch_size
    cfg.train_dataloader.num_workers = num_workers
    cfg.val_dataloader.num_workers = num_workers
    cfg.test_dataloader.num_workers = num_workers
    cfg.train_dataloader.persistent_workers = num_workers > 0
    cfg.val_dataloader.persistent_workers = num_workers > 0
    cfg.test_dataloader.persistent_workers = num_workers > 0
    cfg.optim_wrapper.optimizer.lr = lr
    cfg.train_cfg.max_epochs = epochs
    cfg.param_scheduler[0].end = epochs
    cfg.param_scheduler[0].milestones = [max(1, int(epochs * 0.75))]
    if pretrained_weights:
        cfg.load_from = pretrained_weights

    # MMDetection's internal CocoMetric consumes GroundingDINO token labels
    # directly. With our one-class tree dataset that can produce label indices
    # outside cat_ids and crash during train-time validation. We do unified
    # post-training evaluation below, so default to skipping internal val.
    if train_val_interval and train_val_interval > 0:
        cfg.train_cfg.val_interval = int(train_val_interval)
    else:
        cfg.train_cfg.val_interval = epochs + 1
        cfg.default_hooks.checkpoint.save_best = None
        cfg.default_hooks.checkpoint.save_last = True

    # ── Gradient checkpointing (saves activation memory at ~15% speed cost)
    if gradient_checkpointing:
        cfg.model.backbone.with_cp = True

    # ── Output / runtime ──────────────────────────────────────────────────
    cfg.work_dir = output_dir
    cfg.default_hooks.logger.interval = max(1, 50 // max(batch_size, 1))

    # ── Early stopping ────────────────────────────────────────────────────
    if early_stopping_patience > 0:
        # MMEngine EarlyStoppingHook (available since mmengine >= 0.8.0)
        es_hook = dict(
            type='EarlyStoppingHook',
            monitor='coco/bbox_mAP',
            patience=early_stopping_patience,
            min_delta=0.001,
        )
        if not hasattr(cfg, 'custom_hooks') or cfg.custom_hooks is None:
            cfg.custom_hooks = [es_hook]
        else:
            cfg.custom_hooks = list(cfg.custom_hooks) + [es_hook]

    return cfg


def train_grounding_dino(
    paths, output_dir, epochs=20, batch_size=4, lr=0.0001, num_workers=2,
    pretrained_weights=None, early_stopping_patience=0, gradient_checkpointing=False,
    train_val_interval=0,
):
    """Fine-tune MM-Grounding-DINO on tree-shift COCO data.

    Returns the path to the best checkpoint.
    """
    data_root = str(Path(paths["train_images"]).parent.parent)
    train_ann = paths["train_annotations"]
    train_img_dir = paths["train_images"]
    val_ann = paths["val_annotations"]
    val_img_dir = paths["val_images"]

    cfg = _build_mmdet_cfg(
        data_root=data_root,
        output_dir=output_dir,
        train_ann_file=train_ann,
        train_img_dir=train_img_dir,
        val_ann_file=val_ann,
        val_img_dir=val_img_dir,
        epochs=epochs,
        batch_size=batch_size,
        lr=lr,
        num_workers=num_workers,
        pretrained_weights=pretrained_weights,
        early_stopping_patience=early_stopping_patience,
        gradient_checkpointing=gradient_checkpointing,
        train_val_interval=train_val_interval,
    )

    runner = Runner.from_cfg(cfg)
    runner.train()

    # Find best checkpoint written by CheckpointHook(save_best='auto')
    import glob
    best_candidates = sorted(
        glob.glob(os.path.join(output_dir, "best_coco_bbox_mAP*.pth"))
        + glob.glob(os.path.join(output_dir, "best_coco_bbox_mAP_*.pth"))
        + glob.glob(os.path.join(output_dir, "best_coco", "*.pth"))
        + glob.glob(os.path.join(output_dir, "best_*.pth")),
        key=os.path.getmtime,
    )
    if best_candidates:
        return best_candidates[-1]

    latest = os.path.join(output_dir, "epoch_{}.pth".format(epochs))
    if os.path.exists(latest):
        return latest

    for ep in range(epochs, 0, -1):
        p = os.path.join(output_dir, f"epoch_{ep}.pth")
        if os.path.exists(p):
            return p

    raise FileNotFoundError(
        f"No checkpoint found in {output_dir} after training."
    )


def evaluate_split_finetuned(
    checkpoint_path, split_name, img_folder, ann_file,
    device, output_dir, score_threshold=0.3, num_workers=0,
):
    """Evaluate a fine-tuned MM-Grounding-DINO checkpoint on a COCO split.

    Produces the same ``image_records`` format as the zero-shot path and
    delegates to ``evaluate_detections`` for unified metrics.
    """
    from pycocotools.coco import COCO
    from pycocotools.cocoeval import COCOeval

    cfg_path = os.path.join(_CONFIG_DIR, "mm_grounding_dino_tree.py")
    inferencer = DetInferencer(
        model=cfg_path,
        weights=checkpoint_path,
        device=device,
    )

    coco = COCO(ann_file)
    img_ids = sorted(coco.getImgIds())
    cat_ids = coco.getCatIds()
    category_id = int(cat_ids[0]) if cat_ids else 1

    eval_output = os.path.join(output_dir, f"eval_{split_name}")
    os.makedirs(eval_output, exist_ok=True)

    print(f"  Checkpoint: {checkpoint_path}")
    print(f"  Images: {len(img_ids)}")

    all_coco_results = []
    image_records = []

    start = time.time()
    for i, img_id in enumerate(img_ids):
        img_info = coco.loadImgs(img_id)[0]
        img_path = os.path.join(img_folder, img_info["file_name"])

        result = inferencer(
            img_path, texts="tree", custom_entities=True,
            pred_score_thr=0.0, return_datasamples=False,
            no_save_vis=True,
        )
        pred = result["predictions"][0]
        bboxes_xyxy = np.array(pred["bboxes"]) if len(pred["bboxes"]) else np.zeros((0, 4))
        scores = np.array(pred["scores"]) if len(pred["scores"]) else np.zeros((0,))

        if len(bboxes_xyxy) > 0:
            bboxes_xywh = bboxes_xyxy.copy()
            bboxes_xywh[:, 2] -= bboxes_xywh[:, 0]
            bboxes_xywh[:, 3] -= bboxes_xywh[:, 1]
        else:
            bboxes_xywh = np.zeros((0, 4))

        for j in range(len(scores)):
            all_coco_results.append({
                "image_id": int(img_id),
                "category_id": category_id,
                "bbox": bboxes_xywh[j].tolist(),
                "score": float(scores[j]),
            })

        sort_idx = np.argsort(-scores) if len(scores) > 0 else np.array([], dtype=int)
        scores_sorted = scores[sort_idx] if len(scores) > 0 else scores
        pred_xywh_sorted = bboxes_xywh[sort_idx].tolist() if len(scores) > 0 else []
        score_mask = scores_sorted >= score_threshold
        pred_filtered = [pb for pb, m in zip(pred_xywh_sorted, score_mask) if m]
        scores_filtered = [float(s) for s, m in zip(scores_sorted, score_mask) if m]

        ann_ids = coco.getAnnIds(imgIds=[int(img_id)])
        anns = coco.loadAnns(ann_ids)
        gt_boxes = [ann["bbox"] for ann in anns]

        image_records.append({
            "image_id": int(img_id),
            "filename": img_info.get("file_name", ""),
            "gt_boxes": gt_boxes,
            "pred_boxes": pred_filtered,
            "pred_scores": scores_filtered,
        })

        if (i + 1) % 50 == 0:
            elapsed = time.time() - start
            eta = elapsed / (i + 1) * (len(img_ids) - i - 1)
            print(f"  [{i+1}/{len(img_ids)}] elapsed={elapsed:.1f}s ETA={eta:.1f}s")

    elapsed = time.time() - start
    print(f"  Inference done in {elapsed:.1f}s ({len(img_ids)} images)")

    coco_bbox_stats = None
    if len(all_coco_results) > 0:
        coco_dt_path = os.path.join(eval_output, "coco_dt_results.json")
        with open(coco_dt_path, "w") as f:
            json.dump(all_coco_results, f)
        coco_dt = coco.loadRes(coco_dt_path)
        coco_eval = COCOeval(coco, coco_dt, "bbox")
        coco_eval.evaluate()
        coco_eval.accumulate()
        coco_eval.summarize()
        s = coco_eval.stats
        # pycocotools returns stats in [0, 1]; convert to [0, 100] to match
        # Detectron2's COCOEvaluator convention. -1.0 means no objects → None.
        def _ap_pct(v):
            return None if float(v) < 0 else float(v) * 100
        coco_bbox_stats = {
            "AP": _ap_pct(s[0]), "AP50": _ap_pct(s[1]), "AP75": _ap_pct(s[2]),
            "APs": _ap_pct(s[3]), "APm": _ap_pct(s[4]), "APl": _ap_pct(s[5]),
        }
    else:
        print("  Warning: no detections produced.")
        coco_bbox_stats = {"AP": 0, "AP50": 0, "AP75": 0, "APs": 0, "APm": 0, "APl": 0}

    return evaluate_detections(
        image_records, split_name=split_name, output_dir=eval_output,
        iou_threshold=0.50, coco_bbox_stats=coco_bbox_stats,
    )


# ═══════════════════════════════════════════════════════════════════════════
# CLI
# ═══════════════════════════════════════════════════════════════════════════

def main():
    parser = argparse.ArgumentParser(
        description="GroundingDINO Training & Evaluation (tree-distribution-shift)"
    )
    subparsers = parser.add_subparsers(dest="command", help="Command to execute")

    run_parser = subparsers.add_parser(
        "run", help="Train / evaluate using tree_shift package"
    )

    # ── Data / splits ─────────────────────────────────────────────────────
    run_parser.add_argument("--config", required=True,
                            help="tree_shift benchmark config (e.g. intl_train_IN__ood_US)")
    run_parser.add_argument("--coco-dir", default=None,
                            help="COCO export directory (default: ./coco_export)")
    run_parser.add_argument("--train", action="store_true",
                            help="Fine-tune MM-Grounding-DINO on the train split (requires mmdet)")
    run_parser.add_argument("--eval-val", action="store_true",
                            help="Evaluate on the validation (ID holdout) split")
    run_parser.add_argument("--eval-ood", action="store_true",
                            help="Evaluate on the OOD test split (reported metric)")
    run_parser.add_argument("--eval-ood-train", action="store_true",
                            help="Evaluate on the OOD train split (for shift analysis only)")

    # ── Model / checkpoint ────────────────────────────────────────────────
    run_parser.add_argument(
        "--model-config", default=None,
        help="[zero-shot] Path to IDEA-Research GroundingDINO config .py")
    run_parser.add_argument(
        "--model-path", default=None,
        help="Path to checkpoint. Fine-tuned (mmdet) or local IDEA-Research checkpoint.")
    run_parser.add_argument(
        "--pretrained-weights", default=None,
        help="[train] Optional local checkpoint path to override MMDetection load_from.")
    run_parser.add_argument(
        "--hf-repo-id", default=_HF_REPO_ID,
        help=f"[zero-shot] HuggingFace Hub repo ID (default: {_HF_REPO_ID})")
    run_parser.add_argument(
        "--hf-filename", default=_HF_FILENAME,
        help=f"[zero-shot] HF checkpoint filename (default: {_HF_FILENAME})")

    # ── Zero-shot parameters ──────────────────────────────────────────────
    run_parser.add_argument("--text-prompt", default="tree .",
                            help="[zero-shot] Text prompt (default: 'tree .')")
    run_parser.add_argument("--box-threshold", type=float, default=0.35,
                            help="[zero-shot] Box confidence threshold")
    run_parser.add_argument("--text-threshold", type=float, default=0.25,
                            help="[zero-shot] Text confidence threshold")

    # ── Training hyper-parameters ─────────────────────────────────────────
    run_parser.add_argument("--epochs", type=int, default=20,
                            help="[train] Number of fine-tuning epochs (default: 20)")
    run_parser.add_argument("--batch-size", type=int, default=4,
                            help="[train] Batch size (default: 4)")
    run_parser.add_argument("--lr", type=float, default=0.0001,
                            help="[train] Learning rate (default: 0.0001)")
    run_parser.add_argument("--train-val-interval", type=int, default=0,
                            help="[train] MMDetection validation interval in epochs (0=disabled; post-training eval still runs).")

    # ── Common ────────────────────────────────────────────────────────────
    run_parser.add_argument("--score-threshold", type=float, default=0.3,
                            help="Score threshold for tree metrics")
    run_parser.add_argument("--output-dir", default="./output_grounding_dino",
                            help="Output directory")
    run_parser.add_argument("--device", default="cuda", help="Device (cuda or cpu)")
    run_parser.add_argument("--num-workers", type=int, default=0,
                            help="DataLoader workers (default: 0)")
    run_parser.add_argument("--early-stopping-patience", type=int, default=0,
                            help="Early stopping patience in epochs (0=disabled). Uses coco/bbox_mAP.")
    run_parser.add_argument("--gradient-checkpointing", action="store_true",
                            help="[train] Enable gradient checkpointing on the backbone (saves GPU memory).")

    args = parser.parse_args()

    if args.command is None:
        parser.print_help()
        sys.exit(1)

    if args.command != "run":
        parser.print_help()
        sys.exit(1)

    if not args.train and not args.eval_val and not args.eval_ood and not args.eval_ood_train:
        print("Error: specify at least one of --train, --eval-val, --eval-ood, --eval-ood-train")
        sys.exit(1)

    # ── Determine mode: fine-tune (mmdet) vs zero-shot ────────────────────
    use_mmdet = args.train or (args.model_path is not None)

    if args.train and not MMDET_AVAILABLE:
        print("Error: --train requires MMDetection. Install with:")
        print("  pip install mmengine mmcv mmdet")
        sys.exit(1)

    if not use_mmdet and not GDINO_AVAILABLE:
        print("Error: IDEA-Research GroundingDINO not available for zero-shot eval.")
        print("Install it or use --train / --model-path for MMDetection mode.")
        sys.exit(1)

    device = args.device
    if device == "cuda" and not torch.cuda.is_available():
        print("CUDA not available, falling back to CPU")
        device = "cpu"

    # ── Data export ───────────────────────────────────────────────────────
    paths = ensure_tree_shift_export(args.config, coco_dir=args.coco_dir)
    print(f"Config: {args.config}")
    for k, v in paths.items():
        exists = os.path.exists(v)
        print(f"  {k}: {v}  {'OK' if exists else 'MISSING'}")

    os.makedirs(args.output_dir, exist_ok=True)

    # ══════════════════════════════════════════════════════════════════════
    # TRAIN (MMDetection)
    # ══════════════════════════════════════════════════════════════════════
    checkpoint_path = args.model_path

    if args.train:
        if args.pretrained_weights and not os.path.exists(args.pretrained_weights):
            print(f"Error: --pretrained-weights not found: {args.pretrained_weights}")
            sys.exit(1)
        print(f"\n{'=' * 60}")
        print("PHASE: TRAINING (MM-Grounding-DINO)")
        print("=" * 60)

        checkpoint_path = train_grounding_dino(
            paths=paths,
            output_dir=args.output_dir,
            epochs=args.epochs,
            batch_size=args.batch_size,
            lr=args.lr,
            num_workers=args.num_workers,
            pretrained_weights=args.pretrained_weights,
            early_stopping_patience=getattr(args, 'early_stopping_patience', 0),
            gradient_checkpointing=getattr(args, 'gradient_checkpointing', False),
            train_val_interval=getattr(args, 'train_val_interval', 0),
        )
        print(f"Training complete. Best checkpoint: {checkpoint_path}")

    # ══════════════════════════════════════════════════════════════════════
    # EVALUATE
    # ══════════════════════════════════════════════════════════════════════
    eval_splits = []
    if args.eval_val:
        eval_splits.append(("val", "Validation (ID holdout)"))
    if args.eval_ood:
        eval_splits.append(("ood_test", "Out-of-Distribution (OOD test — reported metric)"))
    if args.eval_ood_train:
        eval_splits.append(("ood_train", "OOD Train (shift analysis only)"))

    if not eval_splits:
        return

    # ── Load model (zero-shot path only) ──────────────────────────────────
    zs_model = None
    if not use_mmdet:
        gdino_root = _GDINO_DIR
        if args.model_config is None:
            args.model_config = os.path.join(
                gdino_root, "groundingdino", "config",
                "GroundingDINO_SwinT_OGC.py",
            )
        if not os.path.exists(args.model_config):
            print(f"Error: model config not found: {args.model_config}")
            sys.exit(1)

        print(f"\nLoading GroundingDINO model (zero-shot)...")
        print(f"  Config: {args.model_config}")
        print(f"  Checkpoint (HuggingFace Hub): {args.hf_repo_id}/{args.hf_filename}")
        zs_model = _load_model_hf(
            args.model_config, args.hf_repo_id, args.hf_filename, device=device,
        )
        zs_model = zs_model.to(device)
        zs_model.eval()
        print("Model loaded successfully.")

    # ── Run evaluation per split ──────────────────────────────────────────
    all_results = {}
    for split_name, split_label in eval_splits:
        img_key = f"{split_name}_images"
        ann_key = f"{split_name}_annotations"
        img_folder = paths[img_key]
        ann_file = paths[ann_key]

        if not os.path.exists(ann_file):
            print(f"\nSkipping {split_label} ({split_name}): annotation file not found")
            continue
        if not os.path.isdir(img_folder):
            print(f"\nSkipping {split_label} ({split_name}): image directory not found")
            continue

        print(f"\n{'=' * 60}")
        print(f"PHASE: EVALUATE {split_label.upper()} ({split_name})")
        print("=" * 60)

        try:
            if use_mmdet:
                if checkpoint_path is None or not os.path.exists(checkpoint_path):
                    print(f"Error: checkpoint not found at '{checkpoint_path}'.")
                    print("Train first (--train) or specify --model-path.")
                    sys.exit(1)
                results = evaluate_split_finetuned(
                    checkpoint_path=checkpoint_path,
                    split_name=split_name,
                    img_folder=img_folder,
                    ann_file=ann_file,
                    device=device,
                    output_dir=args.output_dir,
                    score_threshold=args.score_threshold,
                    num_workers=args.num_workers,
                )
            else:
                results = evaluate_split_zeroshot(
                    model=zs_model,
                    split_name=split_name,
                    img_folder=img_folder,
                    ann_file=ann_file,
                    text_prompt=args.text_prompt,
                    box_threshold=args.box_threshold,
                    text_threshold=args.text_threshold,
                    device=device,
                    output_dir=args.output_dir,
                    score_threshold=args.score_threshold,
                    num_workers=args.num_workers,
                )

            if results:
                all_results[split_name] = results
        except Exception as exc:
            print(f"ERROR evaluating split '{split_name}': {exc}")
            import traceback; traceback.print_exc()

    # ── Summary ───────────────────────────────────────────────────────────
    if all_results:
        mode_label = "MM-Grounding-DINO (fine-tuned)" if use_mmdet else "GroundingDINO (zero-shot)"
        print(f"\n{'=' * 60}")
        print(f"EVALUATION SUMMARY  [{mode_label}]")
        print("=" * 60)
        for split_name, res in all_results.items():
            bbox = res.get("bbox", {})
            print(f"  {split_name}:")
            print(f"    bbox  — AP={bbox.get('AP', 0):.3f}  AP50={bbox.get('AP50', 0):.3f}  AP75={bbox.get('AP75', 0):.3f}")
            tm = res.get("tree_metrics", {}).get("overall", {})
            if tm:
                print(f"    tree  — Precision={tm.get('Precision', 'N/A')}  Recall={tm.get('Recall', 'N/A')}")
                for k, v in tm.items():
                    if "RMSE" in k and v is not None:
                        print(f"    tree  — {k}={v}")

        summary_path = os.path.join(args.output_dir, "eval_summary.json")
        with open(summary_path, "w") as f:
            json.dump(all_results, f, indent=2)
        print(f"\nResults saved to: {summary_path}")

        # Generate table image — only ID test + OOD test (consistent reported metrics)
        table_results = {k: v for k, v in all_results.items() if k in ("val", "ood_test")}
        table_path = os.path.join(args.output_dir, "eval_summary_table.png")
        generate_eval_summary_table(
            table_results, args.config, table_path,
            model_type=mode_label,
        )


if __name__ == "__main__":
    main()
