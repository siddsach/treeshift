#!/usr/bin/env python3
"""
Detectron2 Faster R-CNN Training and Inference Script
Data is loaded via the ``tree_shift`` pip package (>= 0.2.0).

Usage (recommended — tree_shift auto-download):
    # Train only
    python detectron_fastrcnn.py run --config intl_train_IN__ood_US --train

    # Train + evaluate val + evaluate OOD
    python detectron_fastrcnn.py run --config intl_train_IN__ood_US --train --eval-val --eval-ood

    # Evaluate only (requires trained model)
    python detectron_fastrcnn.py run --config intl_train_IN__ood_US --eval-val --eval-ood --model-path ./output/model_final.pth

    # With explicit COCO export directory
    python detectron_fastrcnn.py run --config intl_train_IN__ood_US --coco-dir ./my_coco --train --eval-val --eval-ood

Usage (legacy — manual paths):
    python detectron_fastrcnn.py train --train-annotations train/annotations.json --train-images train/images
    python detectron_fastrcnn.py predict --model-path ./output/model_final.pth --input-image test.jpg --output-image output.jpg
    python detectron_fastrcnn.py evaluate --model-path ./output/model_final.pth --test-annotations test/annotations.json --test-images test/images
"""

import sys
import os
import json
import random
import argparse
import numpy as np
import pandas as pd
from pathlib import Path
from collections import defaultdict

# Add local detectron2 submodule to path (before any detectron2 imports)
_DETECTRON2_SUBMODULE = os.path.join(os.path.dirname(os.path.abspath(__file__)), "detectron2")
if os.path.isdir(_DETECTRON2_SUBMODULE):
    sys.path.insert(0, _DETECTRON2_SUBMODULE)

# Force unbuffered output
sys.stdout = os.fdopen(sys.stdout.fileno(), 'w', buffering=1)
sys.stderr = os.fdopen(sys.stderr.fileno(), 'w', buffering=1)

# Optional imports - only import when needed
try:
    import cv2
    CV2_AVAILABLE = True
except ImportError:
    CV2_AVAILABLE = False

try:
    import matplotlib.pyplot as plt
    MATPLOTLIB_AVAILABLE = True
except ImportError:
    MATPLOTLIB_AVAILABLE = False

print("DEBUG: Starting detectron2 imports...")
sys.stdout.flush()

try:
    from detectron2 import model_zoo
    print("DEBUG: Imported model_zoo")
    from detectron2.data.datasets import register_coco_instances
    print("DEBUG: Imported register_coco_instances")
    from detectron2.utils.visualizer import Visualizer, ColorMode
    print("DEBUG: Imported Visualizer, ColorMode")
    from detectron2.data import DatasetCatalog, MetadataCatalog
    print("DEBUG: Imported DatasetCatalog, MetadataCatalog")
    from detectron2.engine import DefaultTrainer, DefaultPredictor, launch, HookBase
    print("DEBUG: Imported DefaultTrainer, DefaultPredictor, launch, HookBase")
    from detectron2.config import get_cfg
    print("DEBUG: Imported get_cfg")
    from detectron2.evaluation import COCOEvaluator, inference_on_dataset
    print("DEBUG: Imported COCOEvaluator, inference_on_dataset")
    from detectron2.data import build_detection_test_loader
    print("DEBUG: Imported build_detection_test_loader")
    import detectron2.utils.comm as _d2comm
    print("DEBUG: Imported detectron2.utils.comm")
    DETECTRON2_AVAILABLE = True
    print("DEBUG: All detectron2 imports successful!")
except ImportError as e:
    print(f"ERROR: Could not import detectron2: {e}", file=sys.stderr)
    sys.stderr.flush()
    DETECTRON2_AVAILABLE = False
except Exception as e:
    print(f"ERROR: Exception importing detectron2: {e}", file=sys.stderr)
    import traceback
    traceback.print_exc()
    sys.stderr.flush()
    DETECTRON2_AVAILABLE = False

print(f"DEBUG: DETECTRON2_AVAILABLE = {DETECTRON2_AVAILABLE}")
sys.stdout.flush()


from shared_utils import resolve_coco_paths, ensure_tree_shift_export, evaluate_detections


# ---------------------------------------------------------------------------
# Early-stopping support for Detectron2
# ---------------------------------------------------------------------------

class EarlyStoppingHook(HookBase if DETECTRON2_AVAILABLE else object):
    """Stop training when val AP50 does not improve for `patience` evaluations."""

    def __init__(self, eval_period, patience=5, metric="bbox/AP50", min_delta=0.001):
        self._eval_period = eval_period
        self._patience = patience
        self._metric = metric
        self._min_delta = min_delta
        self._best = float("-inf")
        self._bad_evals = 0
        self._eval_count = 0

    def after_step(self):
        if not DETECTRON2_AVAILABLE:
            return
        next_iter = self.trainer.iter + 1
        if self._eval_period <= 0 or next_iter % self._eval_period != 0:
            return
        if not _d2comm.is_main_process():
            return
        self._eval_count += 1
        try:
            latest_ap = self.trainer.storage.history(self._metric).latest()[0]
        except Exception:
            return
        if latest_ap > (self._best + self._min_delta):
            self._best = latest_ap
            self._bad_evals = 0
            print(f"[EarlyStop] eval={self._eval_count} {self._metric}={latest_ap:.4f} (improved, best={self._best:.4f})")
        else:
            self._bad_evals += 1
            print(f"[EarlyStop] eval={self._eval_count} {self._metric}={latest_ap:.4f} "
                  f"(no improvement, bad={self._bad_evals}/{self._patience})")
            if self._bad_evals >= self._patience:
                print(f"[EarlyStop] Stopping: {self._patience} evals with no improvement. "
                      f"Best {self._metric}={self._best:.4f}")
                self.trainer.max_iter = next_iter

    def after_train(self):
        if _d2comm.is_main_process():
            print(f"[EarlyStop] Done. Best {self._metric}={self._best:.4f} "
                  f"after {self._eval_count} evaluations.")


class EarlyStoppingTrainer(DefaultTrainer if DETECTRON2_AVAILABLE else object):
    """DefaultTrainer subclass with optional early stopping."""

    def __init__(self, cfg, early_stopping_patience=0):
        self._es_patience = early_stopping_patience
        super().__init__(cfg)

    def build_hooks(self):
        hooks = super().build_hooks()
        eval_period = self.cfg.TEST.EVAL_PERIOD
        if self._es_patience > 0 and eval_period > 0:
            hooks.append(EarlyStoppingHook(
                eval_period=eval_period,
                patience=self._es_patience,
                metric="bbox/AP50",
                min_delta=0.001,
            ))
        return hooks

    @classmethod
    def build_evaluator(cls, cfg, dataset_name, output_folder=None):
        if output_folder is None:
            output_folder = os.path.join(cfg.OUTPUT_DIR, "eval_val_es")
        return COCOEvaluator(dataset_name, output_dir=output_folder)


def _distributed_train(cfg, resume=True):
    """Helper function for distributed training."""
    trainer = DefaultTrainer(cfg)
    trainer.resume_or_load(resume=resume)
    trainer.train()


def register_tree_shift_datasets(paths, register_train=True, register_val=True,
                                 register_ood_test=True, register_ood_train=False):
    """Register datasets from tree_shift COCO export paths with Detectron2.

    Args:
        paths: dict returned by :func:`shared_utils.resolve_coco_paths`.
        register_train: register the training split.
        register_val: register the validation (ID holdout) split.
        register_ood_test: register the OOD test split (fixed held-out set).
        register_ood_train: register the OOD train split (for shift analysis).

    Returns:
        dict mapping dataset names to True/False (registered or not).
    """
    if not DETECTRON2_AVAILABLE:
        print("Error: detectron2 is not installed.")
        return {}

    status = {}

    splits = []
    if register_train:
        splits.append(("train", paths["train_annotations"], paths["train_images"]))
    if register_val:
        splits.append(("val", paths["val_annotations"], paths["val_images"]))
    if register_ood_test:
        splits.append(("ood_test", paths["ood_test_annotations"], paths["ood_test_images"]))
    if register_ood_train:
        splits.append(("ood_train", paths["ood_train_annotations"], paths["ood_train_images"]))

    for name, ann, imgs in splits:
        if not os.path.exists(ann):
            print(f"Warning: annotation file not found: {ann}")
            status[name] = False
            continue
        if not os.path.isdir(imgs):
            print(f"Warning: images directory not found: {imgs}")
            status[name] = False
            continue

        if name in DatasetCatalog.list():
            print(f"Dataset '{name}' already registered, skipping.")
        else:
            debug_coco_annotations(ann)
            register_coco_instances(name, {}, ann, imgs)
            print(f"Registered dataset '{name}': {ann}")
        status[name] = True

    return status


def debug_coco_annotations(annotation_file):
    """Debug COCO annotations to see what's being filtered."""
    if not os.path.exists(annotation_file):
        print(f"Annotation file not found: {annotation_file}")
        return
    
    import json
    with open(annotation_file, 'r') as f:
        data = json.load(f)
    
    print(f"=== Debugging {annotation_file} ===")
    print(f"Total images: {len(data['images'])}")
    print(f"Total annotations: {len(data['annotations'])}")
    print(f"Categories: {[cat['name'] for cat in data['categories']]}")
    
    # Check annotation types
    has_bbox = any('bbox' in ann for ann in data['annotations'])
    has_segmentation = any('segmentation' in ann for ann in data['annotations'])
    has_area = any('area' in ann for ann in data['annotations'])
    
    print(f"Has bbox: {has_bbox}")
    print(f"Has segmentation: {has_segmentation}")
    print(f"Has area: {has_area}")
    
    # Check a few sample annotations
    for i, ann in enumerate(data['annotations'][:3]):
        print(f"Annotation {i}: keys={list(ann.keys())}")
        if 'bbox' in ann:
            print(f"  bbox: {ann['bbox']}")
        if 'segmentation' in ann:
            print(f"  segmentation type: {type(ann['segmentation'])}")
        if 'area' in ann:
            print(f"  area: {ann['area']}")
    print("=" * 50)


def register_datasets(train_annotations=None, train_images=None, test_annotations=None, test_images=None):
    """Register COCO format datasets with Detectron2."""
    
    if not DETECTRON2_AVAILABLE:
        print("Error: detectron2 is not installed. Please install with: pip install -r requirements.txt")
        return False
    
    # Default paths if not provided
    if train_annotations is None:
        train_annotations = "train/annotations.json"
    if train_images is None:
        train_images = "train/images"
    if test_annotations is None:
        test_annotations = "test/annotations.json"
    if test_images is None:
        test_images = "test/images"
    
    # Debug annotations before registering
    if os.path.exists(train_annotations):
        debug_coco_annotations(train_annotations)
    
    # Check if files exist before registering
    if os.path.exists(train_annotations) and os.path.exists(train_images):
        register_coco_instances("train", {}, train_annotations, train_images)
        print(f"Registered training dataset: {train_annotations}")
    else:
        print(f"Warning: Training dataset not found at {train_annotations}")
    
    if os.path.exists(test_annotations) and os.path.exists(test_images):
        register_coco_instances("test", {}, test_annotations, test_images)
        print(f"Registered test dataset: {test_annotations}")
    else:
        print(f"Warning: Test dataset not found at {test_annotations}")
    
    return True


def visualize_dataset(dataset_name="train", num_samples=9, output_path=None):
    """Visualize random samples from the dataset."""
    
    if not DETECTRON2_AVAILABLE:
        print("Error: detectron2 is not installed. Please install with: pip install -r requirements.txt")
        return
    
    if not CV2_AVAILABLE:
        print("Error: cv2 is not installed. Please install with: pip install opencv-python")
        return
    
    if not MATPLOTLIB_AVAILABLE:
        print("Error: matplotlib is not installed. Please install with: pip install matplotlib")
        return
    
    if dataset_name not in DatasetCatalog.list():
        print(f"Dataset '{dataset_name}' not registered.")
        return
    
    dataset_dicts = DatasetCatalog.get(dataset_name)
    if len(dataset_dicts) == 0:
        print(f"No samples found in dataset '{dataset_name}'.")
        return
    
    # Calculate grid size
    cols = rows = int(np.sqrt(num_samples))
    figure = plt.figure(figsize=(8, 8))
    
    samples = random.sample(dataset_dicts, min(num_samples, len(dataset_dicts)))
    
    for i, d in enumerate(samples):
        img = cv2.imread(d["file_name"])
        if img is None:
            print(f"Could not read image: {d['file_name']}")
            continue
            
        visualizer = Visualizer(img[:, :, ::-1], metadata=MetadataCatalog.get(dataset_name), 
                               scale=0.5, font_size_scale=0.3)
        out = visualizer.draw_dataset_dict(d)
        
        figure.add_subplot(rows, cols, i+1)
        plt.axis("off")
        plt.imshow(out.get_image()[:, :, ::-1])
    
    plt.tight_layout()
    
    if output_path:
        plt.savefig(output_path, dpi=150, bbox_inches='tight')
        print(f"Visualization saved to: {output_path}")
    else:
        plt.show()


def create_trainer_config(num_classes=1, batch_size=4, learning_rate=0.0002,
                         max_iterations=90000, output_dir="./output",
                         config_file=None, eval_period=0, pretrained=False):
    """Create and configure Detectron2 trainer configuration."""
    
    if not DETECTRON2_AVAILABLE:
        print("Error: detectron2 is not installed. Please install with: pip install -r requirements.txt")
        return None
    
    cfg = get_cfg()
    
    # Use custom config file if provided, otherwise use default
    if config_file and os.path.exists(config_file):
        cfg.merge_from_file(config_file)
        print(f"Using custom config file: {config_file}")
    else:
        # Use Faster R-CNN for bounding box detection only (no segmentation masks required)
        _config_yaml = model_zoo.get_config_file("COCO-Detection/faster_rcnn_R_50_FPN_3x.yaml")
        cfg.merge_from_file(_config_yaml)
        if pretrained:
            cfg.MODEL.WEIGHTS = model_zoo.get_checkpoint_url("COCO-Detection/faster_rcnn_R_50_FPN_3x.yaml")
            print("Using Faster R-CNN config with COCO pretrained weights")
        else:
            cfg.MODEL.WEIGHTS = ""  # train from scratch
            print("Using Faster R-CNN config (train from scratch)")
    
    # Dataset configuration
    if "train" in DatasetCatalog.list():
        cfg.DATASETS.TRAIN = ("train",)
    else:
        cfg.DATASETS.TRAIN = ()
        print("Warning: No training dataset registered")
    
    cfg.DATASETS.TEST = ()
    try:
        import torch
        _device = "cuda" if torch.cuda.is_available() else "cpu"
    except Exception:
        _device = "cpu"
    # Use 0 workers to avoid multiprocessing issues
    cfg.DATALOADER.NUM_WORKERS = 0
    
    # Mask format not needed for Faster R-CNN (bbox-only detection)
    # cfg.INPUT.MASK_FORMAT = "bitmask"
    
    # Training parameters
    cfg.SOLVER.IMS_PER_BATCH = batch_size
    cfg.SOLVER.BASE_LR = learning_rate
    cfg.SOLVER.MAX_ITER = max_iterations
    cfg.SOLVER.STEPS = []  # Do not decay learning rate
    # Gradient clipping — essential for training from random init
    cfg.SOLVER.CLIP_GRADIENTS.ENABLED = True
    cfg.SOLVER.CLIP_GRADIENTS.CLIP_TYPE = "norm"
    cfg.SOLVER.CLIP_GRADIENTS.CLIP_VALUE = 1.0
    cfg.SOLVER.CLIP_GRADIENTS.NORM_TYPE = 2.0

    # Model parameters
    cfg.MODEL.ROI_HEADS.BATCH_SIZE_PER_IMAGE = 512
    cfg.MODEL.ROI_HEADS.NUM_CLASSES = num_classes
    cfg.MODEL.DEVICE = _device
    cfg.DATALOADER.FILTER_EMPTY_ANNOTATIONS = True
    
    # Output configuration
    cfg.OUTPUT_DIR = output_dir

    # Early stopping: register val for periodic evaluation
    if eval_period > 0:
        cfg.TEST.EVAL_PERIOD = eval_period
        if "val" in DatasetCatalog.list():
            cfg.DATASETS.TEST = ("val",)

    return cfg


def train_model(cfg, resume=True, early_stopping_patience=0):
    """Train the Detectron2 model."""
    
    if not DETECTRON2_AVAILABLE:
        print("Error: detectron2 is not installed. Please install with: pip install -r requirements.txt")
        return None
    
    if cfg is None:
        print("Error: Configuration is None. Cannot train.")
        return None
    
    if len(cfg.DATASETS.TRAIN) == 0:
        print("No training dataset available. Cannot train.")
        return None
    
    os.makedirs(cfg.OUTPUT_DIR, exist_ok=True)

    if early_stopping_patience > 0:
        trainer = EarlyStoppingTrainer(cfg, early_stopping_patience=early_stopping_patience)
        print(f"Early stopping enabled: patience={early_stopping_patience} evals (metric=bbox/AP50)")
    else:
        trainer = DefaultTrainer(cfg)
    trainer.resume_or_load(resume=resume)

    print(f"Starting training for {cfg.SOLVER.MAX_ITER} iterations...")
    print(f"Batch size: {cfg.SOLVER.IMS_PER_BATCH}")
    print(f"Learning rate: {cfg.SOLVER.BASE_LR}")
    print(f"Device: {cfg.MODEL.DEVICE}")

    trainer.train()

    print(f"Training completed. Model saved to: {cfg.OUTPUT_DIR}")
    return trainer


def create_predictor(model_path, num_classes=1, score_threshold=0.7, config_file=None):
    """Create a predictor for inference."""
    
    if not DETECTRON2_AVAILABLE:
        print("Error: detectron2 is not installed. Please install with: pip install -r requirements.txt")
        return None
    
    cfg = get_cfg()
    
    if config_file and os.path.exists(config_file):
        cfg.merge_from_file(config_file)
    else:
        _config_yaml = model_zoo.get_config_file("COCO-Detection/faster_rcnn_R_50_FPN_3x.yaml")
        cfg.merge_from_file(_config_yaml)
    
    cfg.MODEL.WEIGHTS = model_path
    cfg.MODEL.ROI_HEADS.NUM_CLASSES = num_classes
    cfg.MODEL.ROI_HEADS.SCORE_THRESH_TEST = score_threshold
    try:
        import torch
        _device = "cuda" if torch.cuda.is_available() else "cpu"
    except Exception:
        _device = "cpu"
    cfg.MODEL.DEVICE = _device
    
    return DefaultPredictor(cfg)


def predict_image(predictor, image_path, output_path=None, visualize=True):
    """Run prediction on a single image."""
    
    if not CV2_AVAILABLE:
        print("Error: cv2 is not installed. Please install with: pip install opencv-python")
        return None
    
    if not MATPLOTLIB_AVAILABLE and visualize and output_path is None:
        print("Error: matplotlib is not installed. Please install with: pip install matplotlib")
        return None
    
    if predictor is None:
        print("Error: Predictor is None. Cannot predict.")
        return None
    
    if not os.path.exists(image_path):
        print(f"Image not found: {image_path}")
        return None
    
    im = cv2.imread(image_path)
    if im is None:
        print(f"Could not read image: {image_path}")
        return None
    
    outputs = predictor(im)
    
    print(f"Predictions for {image_path}:")
    print(f"  Number of instances: {len(outputs['instances'])}")
    if len(outputs['instances']) > 0:
        print(f"  Classes: {outputs['instances'].pred_classes.cpu().numpy()}")
        print(f"  Scores: {outputs['instances'].scores.cpu().numpy()}")
    
    if visualize or output_path:
        # Get metadata for visualization (use test dataset if available, otherwise empty metadata)
        if "test" in DatasetCatalog.list():
            metadata = MetadataCatalog.get("test")
        else:
            metadata = MetadataCatalog.get("train") if "train" in DatasetCatalog.list() else None
        
        if metadata:
            v = Visualizer(im[:, :, ::-1], metadata=metadata, scale=0.8, instance_mode=ColorMode.IMAGE_BW)
        else:
            v = Visualizer(im[:, :, ::-1], scale=0.8, instance_mode=ColorMode.IMAGE_BW)
        
        out = v.draw_instance_predictions(outputs["instances"].to("cpu"))
        result_image = out.get_image()[:, :, ::-1]
        
        if output_path:
            cv2.imwrite(output_path, result_image)
            print(f"Result saved to: {output_path}")
        else:
            plt.figure(figsize=(12, 8))
            plt.imshow(result_image)
            plt.axis("off")
            plt.show()
    
    return outputs


def _compute_iou(box_a, box_b):
    """Compute IoU between two boxes in [x, y, w, h] COCO format."""
    ax, ay, aw, ah = box_a
    bx, by, bw, bh = box_b
    # Convert to [x1, y1, x2, y2]
    a_x1, a_y1, a_x2, a_y2 = ax, ay, ax + aw, ay + ah
    b_x1, b_y1, b_x2, b_y2 = bx, by, bx + bw, by + bh
    inter_x1 = max(a_x1, b_x1)
    inter_y1 = max(a_y1, b_y1)
    inter_x2 = min(a_x2, b_x2)
    inter_y2 = min(a_y2, b_y2)
    inter_area = max(0, inter_x2 - inter_x1) * max(0, inter_y2 - inter_y1)
    union_area = aw * ah + bw * bh - inter_area
    return inter_area / union_area if union_area > 0 else 0.0


def compute_tree_detection_metrics(predictor, dataset_name, coco_results=None,
                                   iou_threshold=0.50, gsd=None):
    """Compute detailed tree-detection metrics with per-size breakdown.

    Metrics produced (overall and per size class small / large):
        - AP@50            (from COCO evaluator results if provided)
        - Precision         at the predictor's score threshold
        - Recall            at the predictor's score threshold
        - Crown Diameter RMSE  (bbox width as proxy, pixels or metres if gsd given)
        - Geolocation RMSE     (bbox-centre distance, pixels or metres if gsd given)

    Tree size split: GT boxes with area < dataset median → *small*, else → *large*.

    Args:
        predictor: DefaultPredictor instance.
        dataset_name: registered Detectron2 dataset name.
        coco_results: dict returned by inference_on_dataset (optional, used to
                      pull AP50 from the standard COCO evaluator).
        iou_threshold: IoU threshold for a true-positive match (default 0.50).
        gsd: ground sampling distance in metres/pixel.  When provided the RMSE
             values are reported in metres; otherwise in pixels.

    Returns:
        dict with keys 'overall', 'small', 'large', each containing the metrics.
    """
    if not DETECTRON2_AVAILABLE:
        return None
    if predictor is None or dataset_name not in DatasetCatalog.list():
        return None

    dataset_dicts = DatasetCatalog.get(dataset_name)
    if len(dataset_dicts) == 0:
        return None

    # ------------------------------------------------------------------
    # 1. Collect all GT boxes and determine the median area for size split
    # ------------------------------------------------------------------
    all_gt_areas = []
    for d in dataset_dicts:
        for ann in d.get("annotations", []):
            if "bbox" in ann:
                _, _, w, h = ann["bbox"]
                all_gt_areas.append(w * h)
    if len(all_gt_areas) == 0:
        print("No GT annotations found – skipping tree detection metrics.")
        return None
    area_median = float(np.median(all_gt_areas))
    print(f"Tree-size split: median GT bbox area = {area_median:.1f} px²  "
          f"(small < median, large >= median)")

    # ------------------------------------------------------------------
    # 2. Per-image: run predictor, greedy-match preds to GT at IoU >= thr
    # ------------------------------------------------------------------
    # Accumulators per size bucket
    buckets = {"overall": [], "small": [], "large": []}
    tp_counts = {"overall": 0, "small": 0, "large": 0}
    fp_counts = {"overall": 0, "small": 0, "large": 0}
    fn_counts = {"overall": 0, "small": 0, "large": 0}
    crown_errors = {"overall": [], "small": [], "large": []}
    centre_errors = {"overall": [], "small": [], "large": []}

    for d in dataset_dicts:
        # --- ground truth ---
        gt_boxes = []  # list of [x, y, w, h]
        gt_sizes = []  # 'small' or 'large'
        for ann in d.get("annotations", []):
            if "bbox" not in ann:
                continue
            bx, by, bw, bh = ann["bbox"]
            gt_boxes.append([bx, by, bw, bh])
            gt_sizes.append("small" if bw * bh < area_median else "large")

        # --- predictions ---
        if CV2_AVAILABLE:
            im = cv2.imread(d["file_name"])
        else:
            im = np.zeros((d.get("height", 100), d.get("width", 100), 3), dtype=np.uint8)
        if im is None:
            continue
        outputs = predictor(im)
        instances = outputs["instances"].to("cpu")
        pred_boxes_xyxy = instances.pred_boxes.tensor.numpy()  # (N, 4) x1y1x2y2
        scores = instances.scores.numpy()

        # Convert pred boxes to COCO [x, y, w, h]
        pred_boxes = []
        for pb in pred_boxes_xyxy:
            x1, y1, x2, y2 = pb
            pred_boxes.append([float(x1), float(y1), float(x2 - x1), float(y2 - y1)])

        # Sort predictions by descending score for greedy matching
        order = np.argsort(-scores)
        pred_boxes = [pred_boxes[i] for i in order]

        matched_gt = set()
        matched_pred = set()

        for pi in range(len(pred_boxes)):
            best_iou = 0.0
            best_gi = -1
            for gi in range(len(gt_boxes)):
                if gi in matched_gt:
                    continue
                iou = _compute_iou(pred_boxes[pi], gt_boxes[gi])
                if iou > best_iou:
                    best_iou = iou
                    best_gi = gi
            if best_iou >= iou_threshold and best_gi >= 0:
                matched_gt.add(best_gi)
                matched_pred.add(pi)
                size_key = gt_sizes[best_gi]

                # Crown diameter error (width as proxy)
                gt_w = gt_boxes[best_gi][2]
                pd_w = pred_boxes[pi][2]
                crown_errors["overall"].append((pd_w - gt_w) ** 2)
                crown_errors[size_key].append((pd_w - gt_w) ** 2)

                # Centre-distance error
                gt_cx = gt_boxes[best_gi][0] + gt_boxes[best_gi][2] / 2
                gt_cy = gt_boxes[best_gi][1] + gt_boxes[best_gi][3] / 2
                pd_cx = pred_boxes[pi][0] + pred_boxes[pi][2] / 2
                pd_cy = pred_boxes[pi][1] + pred_boxes[pi][3] / 2
                dist_sq = (pd_cx - gt_cx) ** 2 + (pd_cy - gt_cy) ** 2
                centre_errors["overall"].append(dist_sq)
                centre_errors[size_key].append(dist_sq)

                tp_counts["overall"] += 1
                tp_counts[size_key] += 1
            else:
                # False positive (no matching GT)
                fp_counts["overall"] += 1
                # Assign FP to the size bucket of the closest GT if any
                if len(gt_boxes) > 0:
                    closest_gi = min(range(len(gt_boxes)),
                                     key=lambda g: _compute_iou(pred_boxes[pi], gt_boxes[g]),
                                     default=0)
                    fp_counts[gt_sizes[closest_gi]] += 1

        # Unmatched GT → false negatives
        for gi in range(len(gt_boxes)):
            if gi not in matched_gt:
                fn_counts["overall"] += 1
                fn_counts[gt_sizes[gi]] += 1

    # ------------------------------------------------------------------
    # 3. Aggregate metrics
    # ------------------------------------------------------------------
    unit = "m" if gsd else "px"
    scale = gsd if gsd else 1.0

    def _build_bucket(key):
        tp = tp_counts[key]
        fp = fp_counts[key]
        fn = fn_counts[key]
        precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
        recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0

        crown_rmse = float(np.sqrt(np.mean(crown_errors[key])) * scale) if crown_errors[key] else None
        geo_rmse = float(np.sqrt(np.mean(centre_errors[key])) * scale) if centre_errors[key] else None

        m = {
            "TP": tp, "FP": fp, "FN": fn,
            "Precision": round(precision, 4),
            "Recall": round(recall, 4),
            f"Crown_Diameter_RMSE_{unit}": round(crown_rmse, 3) if crown_rmse is not None else None,
            f"Geolocation_RMSE_{unit}": round(geo_rmse, 3) if geo_rmse is not None else None,
        }
        return m

    metrics = {}
    for key in ["overall", "small", "large"]:
        metrics[key] = _build_bucket(key)

    # Inject AP@50 from COCO results if available
    if coco_results:
        bbox_res = coco_results.get("bbox", {})
        metrics["overall"]["AP50"] = bbox_res.get("AP50")
        # COCO evaluator already provides AP by size (APs, APm, APl)
        metrics["small"]["AP50_COCO_APs"] = bbox_res.get("APs")
        metrics["large"]["AP50_COCO_APl"] = bbox_res.get("APl")

    # ------------------------------------------------------------------
    # 4. Pretty-print
    # ------------------------------------------------------------------
    print("\n" + "=" * 70)
    print("TREE DETECTION METRICS  (IoU >= {:.2f})".format(iou_threshold))
    if gsd:
        print(f"  GSD = {gsd} m/px  →  spatial metrics in metres")
    else:
        print("  No GSD provided  →  spatial metrics in pixels")
    print("=" * 70)
    for key in ["overall", "small", "large"]:
        m = metrics[key]
        label = key.upper()
        print(f"\n  [{label}]  TP={m['TP']}  FP={m['FP']}  FN={m['FN']}")
        print(f"    Precision = {m['Precision']:.4f}    Recall = {m['Recall']:.4f}")
        if "AP50" in m and m["AP50"] is not None:
            print(f"    AP@50     = {m['AP50']:.4f}")
        cd_key = f"Crown_Diameter_RMSE_{unit}"
        gl_key = f"Geolocation_RMSE_{unit}"
        if m.get(cd_key) is not None:
            print(f"    Crown Diameter RMSE = {m[cd_key]:.3f} {unit}")
        else:
            print(f"    Crown Diameter RMSE = N/A (no matched detections)")
        if m.get(gl_key) is not None:
            print(f"    Geolocation RMSE    = {m[gl_key]:.3f} {unit}")
        else:
            print(f"    Geolocation RMSE    = N/A (no matched detections)")
    print("=" * 70)

    return metrics


def generate_eval_summary_table(all_results, config_name, output_path, model_type="Faster R-CNN"):
    """Generate a table image summarising AP metrics across eval splits.

    Columns (bbox-only for Faster R-CNN):
        Train - Test | AP@0.50 [bbox] | AP@[0.50:0.95] [bbox] | APs [bbox] | APm [bbox] | APl [bbox]
                     | Precision | Recall | Crown Diam RMSE | Geoloc RMSE
    """
    if not MATPLOTLIB_AVAILABLE:
        print("matplotlib not available — skipping table image generation")
        return

    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    # Build rows
    columns = [
        "Train - Test",
        "AP@0.50\n[bbox]",
        "AP@[0.50:0.95]\n[bbox]",
        "APs\n[bbox]",
        "APm\n[bbox]",
        "APl\n[bbox]",
        "Precision",
        "Recall",
        "Crown Diam\nRMSE",
        "Geoloc\nRMSE",
    ]

    rows = []
    for split_name, res in all_results.items():
        bbox = res.get("bbox", {})
        tm = res.get("tree_metrics", {}).get("overall", {})

        def _fmt(v):
            if v is None or v == "N/A":
                return "—"
            try:
                return f"{float(v):.3f}"
            except (TypeError, ValueError):
                return str(v)

        row = [
            f"{config_name}\n{split_name}",
            _fmt(bbox.get("AP50")),
            _fmt(bbox.get("AP")),
            _fmt(bbox.get("APs")),
            _fmt(bbox.get("APm")),
            _fmt(bbox.get("APl")),
            _fmt(tm.get("Precision")),
            _fmt(tm.get("Recall")),
            _fmt(tm.get("Crown_Diameter_RMSE_pixels") or tm.get("Crown Diameter RMSE (pixels)")),
            _fmt(tm.get("Geolocation_RMSE_pixels") or tm.get("Geolocation RMSE (pixels)")),
        ]
        rows.append(row)

    # Create figure
    n_rows = len(rows)
    fig_width = max(14, len(columns) * 1.5)
    fig_height = max(1.5, 0.6 * (n_rows + 1) + 0.8)
    fig, ax = plt.subplots(figsize=(fig_width, fig_height))
    ax.axis("off")
    ax.set_title(f"{model_type} — Evaluation Summary", fontsize=14, fontweight="bold", pad=12)

    table = ax.table(
        cellText=rows,
        colLabels=columns,
        cellLoc="center",
        loc="center",
    )
    table.auto_set_font_size(False)
    table.set_fontsize(9)
    table.scale(1.0, 1.6)

    # Style header
    for j in range(len(columns)):
        cell = table[0, j]
        cell.set_facecolor("#4472C4")
        cell.set_text_props(color="white", fontweight="bold")

    # Alternate row colours
    for i in range(1, n_rows + 1):
        colour = "#D9E2F3" if i % 2 == 1 else "white"
        for j in range(len(columns)):
            table[i, j].set_facecolor(colour)

    os.makedirs(os.path.dirname(output_path) if os.path.dirname(output_path) else ".", exist_ok=True)
    fig.savefig(output_path, dpi=150, bbox_inches="tight", facecolor="white")
    plt.close(fig)
    print(f"Evaluation summary table saved to: {output_path}")


def evaluate_model(predictor, dataset_name="test", output_dir="./detectron_output",
                   score_threshold=None):
    """Evaluate model on test dataset using COCO metrics + unified per-image eval."""
    
    if not DETECTRON2_AVAILABLE:
        print("Error: detectron2 is not installed. Please install with: pip install -r requirements.txt")
        return None
    
    if predictor is None:
        print("Error: Predictor is None. Cannot evaluate.")
        return None
    
    if dataset_name not in DatasetCatalog.list():
        print(f"Dataset '{dataset_name}' not registered for evaluation.")
        return
    
    os.makedirs(output_dir, exist_ok=True)

    evaluator = COCOEvaluator(dataset_name, output_dir=output_dir)
    val_loader = build_detection_test_loader(predictor.cfg, dataset_name)
    
    print(f"Evaluating model on {dataset_name} dataset...")
    coco_results = inference_on_dataset(predictor.model, val_loader, evaluator)
    
    print("COCO Evaluation results:")
    print(json.dumps(coco_results, indent=2))

    coco_bbox_stats = coco_results.get("bbox", {})

    # Collect per-image predictions for unified evaluation
    dataset_dicts = DatasetCatalog.get(dataset_name)
    image_records = []
    for d in dataset_dicts:
        if CV2_AVAILABLE:
            im = cv2.imread(d["file_name"])
        else:
            im = np.zeros((d.get("height", 100), d.get("width", 100), 3), dtype=np.uint8)
        if im is None:
            continue

        outputs = predictor(im)
        instances = outputs["instances"].to("cpu")
        pred_boxes_xyxy = instances.pred_boxes.tensor.numpy()
        scores = instances.scores.numpy()
        if score_threshold is not None:
            keep = scores >= float(score_threshold)
            pred_boxes_xyxy = pred_boxes_xyxy[keep]
            scores = scores[keep]

        order = np.argsort(-scores)
        pred_xywh = []
        for idx in order:
            x1, y1, x2, y2 = pred_boxes_xyxy[idx]
            pred_xywh.append([float(x1), float(y1), float(x2 - x1), float(y2 - y1)])

        gt_boxes = []
        for ann in d.get("annotations", []):
            if "bbox" in ann:
                gt_boxes.append(list(ann["bbox"]))

        image_records.append({
            "image_id": d.get("image_id", 0),
            "filename": os.path.basename(d["file_name"]),
            "gt_boxes": gt_boxes,
            "pred_boxes": pred_xywh,
            "pred_scores": [float(scores[i]) for i in order],
        })

    results = evaluate_detections(
        image_records,
        split_name=dataset_name,
        output_dir=output_dir,
        iou_threshold=0.50,
        coco_bbox_stats=coco_bbox_stats,
    )
    return results


def evaluate_model_by_zone(predictor, dataset_name="test", output_dir="./detectron_output",
                          zone_csv_path=None):
    """Evaluate model with per-zone metrics using zone mapping CSV."""

    if not DETECTRON2_AVAILABLE:
        print("Error: detectron2 is not installed. Please install with: pip install -r requirements.txt")
        return None

    if predictor is None:
        print("Error: Predictor is None. Cannot evaluate.")
        return None

    if dataset_name not in DatasetCatalog.list():
        print(f"Dataset '{dataset_name}' not registered for evaluation.")
        return

    # CLEAR any existing zone datasets to prevent registration conflicts
    print("Clearing existing zone datasets...")
    existing_datasets = list(DatasetCatalog.keys())
    for dataset_key in existing_datasets:
        if dataset_key.startswith(f"{dataset_name}_"):
            try:
                DatasetCatalog.remove(dataset_key)
                MetadataCatalog.remove(dataset_key)
                print(f"Cleared existing dataset: {dataset_key}")
            except KeyError:
                pass  # Dataset doesn't exist, skip

    # Load zone mapping if provided
    zone_mapping = {}
    if zone_csv_path and os.path.exists(zone_csv_path):
        zone_df = pd.read_csv(zone_csv_path)
        zone_mapping = dict(zip(zone_df['filename'], zone_df['zone']))
        print(f"Loaded zone mapping for {len(zone_mapping)} images")
    else:
        print("No zone CSV provided, running standard evaluation")
        return evaluate_model(predictor, dataset_name, output_dir)

    # Create output directory
    os.makedirs(output_dir, exist_ok=True)

    # Get dataset information
    dataset_dicts = DatasetCatalog.get(dataset_name)
    print(f"Evaluating {len(dataset_dicts)} images...")

    # Group dataset by zones
    zone_datasets = defaultdict(list)
    for d in dataset_dicts:
        filename = os.path.basename(d["file_name"])
        # Extract base filename (remove .tiff extension if present)
        base_filename = filename.replace('.tiff', '').replace('.tif', '')

        zone = zone_mapping.get(filename, zone_mapping.get(base_filename, 'unknown'))
        zone_datasets[zone].append(d)

    print(f"Found {len(zone_datasets)} zones: {list(zone_datasets.keys())}")

    # Evaluate each zone separately
    zone_results = {}
    overall_results = None

    for zone, zone_dicts in zone_datasets.items():
        if len(zone_dicts) == 0:
            continue

        print(f"\n=== Evaluating Zone: {zone} ({len(zone_dicts)} images) ===")

        # Create temporary dataset for this zone with unique name
        import uuid
        zone_dataset_name = f"{dataset_name}_zone_{zone.replace(' ', '_').replace('-', '_').lower()}_{str(uuid.uuid4())[:8]}"

        # Register zone-specific dataset
        DatasetCatalog.register(zone_dataset_name, lambda zone_dicts=zone_dicts: zone_dicts)
        MetadataCatalog.get(zone_dataset_name).set(thing_classes=["tree"])  # Adjust for your classes

        # Create evaluator for this zone
        zone_output_dir = os.path.join(output_dir, zone.replace(' ', '_').lower())
        evaluator = COCOEvaluator(zone_dataset_name, output_dir=zone_output_dir)
        val_loader = build_detection_test_loader(predictor.cfg, zone_dataset_name)

        # Run evaluation
        try:
            results = inference_on_dataset(predictor.model, val_loader, evaluator)
            zone_results[zone] = {
                'metrics': results,
                'num_images': len(zone_dicts),
                'output_dir': zone_output_dir
            }
            print(f"Zone {zone} results: {results.get('bbox', {})}")

            # Accumulate overall results (weighted by number of images)
            if overall_results is None:
                overall_results = {k: v * len(zone_dicts) for k, v in results.items() if isinstance(v, (int, float))}
            else:
                for k, v in results.items():
                    if isinstance(v, (int, float)):
                        overall_results[k] = overall_results.get(k, 0) + v * len(zone_dicts)

        except Exception as e:
            print(f"Error evaluating zone {zone}: {e}")
            zone_results[zone] = {'error': str(e), 'num_images': len(zone_dicts)}

    # Compute weighted average for overall results
    if overall_results and zone_results:
        total_images = sum(zone['num_images'] for zone in zone_results.values() if 'num_images' in zone)
        if total_images > 0:
            overall_results = {k: v / total_images for k, v in overall_results.items()}

    # Save comprehensive results
    final_results = {
        'overall': overall_results,
        'per_zone': zone_results,
        'zone_mapping_file': zone_csv_path,
        'total_images': sum(len(zone_dicts) for zone_dicts in zone_datasets.values()),
        'num_zones': len(zone_results)
    }

    # Save to JSON
    results_file = os.path.join(output_dir, "zone_evaluation_results.json")
    with open(results_file, 'w') as f:
        json.dump(final_results, f, indent=2)

    # Print summary
    print("\n" + "="*60)
    print("PER-ZONE EVALUATION SUMMARY")
    print("="*60)
    print(f"Total images evaluated: {final_results['total_images']}")
    print(f"Number of zones: {final_results['num_zones']}")
    print(f"Results saved to: {results_file}")

    print("\nPer-Zone Results:")
    for zone, data in zone_results.items():
        if 'metrics' in data:
            bbox_ap = data['metrics'].get('bbox', {}).get('AP', 'N/A')
            print(f"  {zone}: {data['num_images']} images, AP={bbox_ap}")
        else:
            print(f"  {zone}: {data['num_images']} images, Error: {data.get('error', 'Unknown')}")

    return final_results


def visualize_predictions_vs_ground_truth(predictor, dataset_name="test", num_samples=3, output_path=None):
    """Visualize model predictions compared to ground truth annotations."""
    
    if not DETECTRON2_AVAILABLE:
        print("Error: detectron2 is not installed. Please install with: pip install -r requirements.txt")
        return
    
    if not CV2_AVAILABLE:
        print("Error: cv2 is not installed. Please install with: pip install opencv-python")
        return
    
    if not MATPLOTLIB_AVAILABLE:
        print("Error: matplotlib is not installed. Please install with: pip install matplotlib")
        return
    
    if predictor is None:
        print("Error: Predictor is None. Cannot visualize.")
        return
    
    if dataset_name not in DatasetCatalog.list():
        print(f"Dataset '{dataset_name}' not registered.")
        return
    
    dataset_dicts = DatasetCatalog.get(dataset_name)
    if len(dataset_dicts) == 0:
        print(f"No samples found in dataset '{dataset_name}'.")
        return
    
    samples = random.sample(dataset_dicts, min(num_samples, len(dataset_dicts)))
    figure = plt.figure(figsize=(12, 8))
    
    for i, d in enumerate(samples):
        im = cv2.imread(d["file_name"])
        if im is None:
            continue
        
        # Model prediction
        outputs = predictor(im)
        v = Visualizer(im[:, :, ::-1],
                      metadata=MetadataCatalog.get(dataset_name), 
                      scale=0.5, 
                      instance_mode=ColorMode.IMAGE_BW,
                      font_size_scale=0)
        out = v.draw_instance_predictions(outputs["instances"].to("cpu"))
        
        # Show prediction
        figure.add_subplot(num_samples, 2, 2*i+1)
        plt.axis("off")
        plt.title("Prediction")
        plt.imshow(out.get_image()[:, :, ::-1])
        
        # Show ground truth
        v = Visualizer(im[:, :, ::-1],
                      metadata=MetadataCatalog.get(dataset_name), 
                      scale=0.5, 
                      instance_mode=ColorMode.IMAGE_BW,
                      font_size_scale=0)
        
        # Draw ground truth annotations
        for ann in d['annotations']:
            for segm in ann['segmentation']:
                v = v.draw_polygon(np.reshape(segm, (int(len(segm)/2), 2)), "tab:green")
        
        figure.add_subplot(num_samples, 2, 2*i+2)
        plt.axis("off")
        plt.title("Ground Truth")
        
        if hasattr(v, 'get_image'):
            gt_image = v.get_image()[:, :, ::-1]
        else:
            gt_image = im[:, :, ::-1]
        
        plt.imshow(gt_image)
    
    plt.tight_layout()
    
    if output_path:
        plt.savefig(output_path, dpi=150, bbox_inches='tight')
        print(f"Comparison visualization saved to: {output_path}")
    else:
        plt.show()


def main():
    parser = argparse.ArgumentParser(description="Detectron2 Faster R-CNN Training and Inference Script")
    subparsers = parser.add_subparsers(dest='command', help='Command to execute')
    
    # ---- tree_shift workflow (recommended) ----
    run_parser = subparsers.add_parser('run', help='Train / evaluate using tree_shift package')
    run_parser.add_argument('--config', required=True, help='tree_shift benchmark config (e.g. intl_train_IN__ood_US)')
    run_parser.add_argument('--coco-dir', default=None, help='COCO export directory (default: ./coco_export)')
    run_parser.add_argument('--train', action='store_true', help='Train on the train split')
    run_parser.add_argument('--eval-val', action='store_true', help='Evaluate on the validation (ID holdout) split')
    run_parser.add_argument('--eval-ood', action='store_true', help='Evaluate on the OOD test split (reported metric)')
    run_parser.add_argument('--eval-ood-train', action='store_true', help='Evaluate on the OOD train split (for shift analysis only)')
    run_parser.add_argument(
        '--eval-mode', choices=['distshift', 'conventional'], default='distshift',
        help='distshift: train on ID only (default). conventional: train on ID + OOD.',
    )
    run_parser.add_argument('--model-path', default=None, help='Path to trained model (required if --train is not set but eval is requested)')
    run_parser.add_argument('--num-classes', type=int, default=1, help='Number of classes')
    run_parser.add_argument('--batch-size', type=int, default=4, help='Batch size')
    run_parser.add_argument('--learning-rate', type=float, default=0.0002, help='Learning rate')
    run_parser.add_argument('--max-iterations', type=int, default=90000, help='Maximum training iterations')
    run_parser.add_argument('--output-dir', default='./output', help='Output directory for model checkpoints')
    run_parser.add_argument('--config-file', default=None, help='Custom Detectron2 config file')
    run_parser.add_argument('--score-threshold', type=float, default=0.7, help='Confidence threshold for tree/count metrics')
    run_parser.add_argument('--coco-score-threshold', type=float, default=0.05,
                            help='Low confidence threshold for COCO AP prediction generation')
    run_parser.add_argument('--resume', action='store_true', default=True, help='Resume from checkpoint')
    run_parser.add_argument('--early-stopping-patience', type=int, default=0,
                            help='Early stopping patience in epochs (0=disabled). Requires val split.')
    run_parser.add_argument('--pretrained', action='store_true', default=False,
                            help='Load COCO pretrained weights (default: train from scratch)')

    # ---- Legacy subcommands ----
    # Train command
    train_parser = subparsers.add_parser('train', help='[Legacy] Train the model with manual paths')
    train_parser.add_argument('--train-annotations', default=None, help='Path to training annotations JSON')
    train_parser.add_argument('--train-images', default=None, help='Path to training images directory')
    train_parser.add_argument('--num-classes', type=int, default=1, help='Number of classes (1 for Kejri Tree only)')
    train_parser.add_argument('--batch-size', type=int, default=4, help='Batch size')
    train_parser.add_argument('--learning-rate', type=float, default=0.0002, help='Learning rate')
    train_parser.add_argument('--max-iterations', type=int, default=90000, help='Maximum iterations')
    train_parser.add_argument('--output-dir', default='./output', help='Output directory')
    train_parser.add_argument('--config-file', default=None, help='Custom config file path')
    train_parser.add_argument('--resume', action='store_true', default=True, help='Resume from checkpoint')
    train_parser.add_argument('--visualize', action='store_true', help='Visualize training data')
    
    # Predict command
    predict_parser = subparsers.add_parser('predict', help='Run prediction on an image')
    predict_parser.add_argument('--model-path', required=True, help='Path to trained model')
    predict_parser.add_argument('--input-image', required=True, help='Input image path')
    predict_parser.add_argument('--output-image', default=None, help='Output image path')
    predict_parser.add_argument('--num-classes', type=int, default=1, help='Number of classes (1 for Kejri Tree only)')
    predict_parser.add_argument('--score-threshold', type=float, default=0.7, help='Confidence threshold')
    predict_parser.add_argument('--config-file', default=None, help='Custom config file path')
    
    # Evaluate command
    evaluate_parser = subparsers.add_parser('evaluate', help='Evaluate model on test dataset')
    evaluate_parser.add_argument('--model-path', required=True, help='Path to trained model')
    evaluate_parser.add_argument('--test-annotations', default=None, help='Path to test annotations JSON')
    evaluate_parser.add_argument('--test-images', default=None, help='Path to test images directory')
    evaluate_parser.add_argument('--num-classes', type=int, default=1, help='Number of classes (1 for Kejri Tree only)')
    evaluate_parser.add_argument('--score-threshold', type=float, default=0.7, help='Confidence threshold')
    evaluate_parser.add_argument('--output-dir', default='./detectron_output', help='Evaluation output directory')
    evaluate_parser.add_argument('--config-file', default=None, help='Custom config file path')
    evaluate_parser.add_argument('--visualize-comparison', action='store_true', help='Visualize predictions vs ground truth')
    evaluate_parser.add_argument('--zone-csv', default=None, help='Path to CSV file mapping images to zones (filename,zone columns)')
    
    args = parser.parse_args()
    
    if args.command == 'run':
        # ---- tree_shift workflow ----
        if not args.train and not args.eval_val and not args.eval_ood and not args.eval_ood_train:
            print("Error: specify at least one of --train, --eval-val, --eval-ood, --eval-ood-train")
            sys.exit(1)

        # Auto-download & export via tree_shift
        paths = ensure_tree_shift_export(args.config, coco_dir=args.coco_dir)
        print(f"Config: {args.config}")
        for k, v in paths.items():
            exists = os.path.exists(v)
            print(f"  {k}: {v}  {'OK' if exists else 'MISSING'}")

        # Register only the splits we need (val also needed for early stopping)
        needs_val = args.eval_val or args.early_stopping_patience > 0
        status = register_tree_shift_datasets(
            paths,
            register_train=args.train,
            register_val=needs_val,
            register_ood_test=args.eval_ood,
            register_ood_train=args.eval_ood_train,
        )

        # --- TRAIN ---
        model_path = args.model_path
        if args.train:
            if not status.get("train"):
                print("Error: training dataset could not be registered. Aborting.")
                sys.exit(1)

            print("\n" + "=" * 60)
            print("PHASE: TRAINING")
            print("=" * 60)

            # eval_period = one epoch worth of iterations (for early stopping)
            early_stopping_patience = getattr(args, 'early_stopping_patience', 0)
            eval_period = max(1, args.max_iterations // 50) if early_stopping_patience > 0 else 0

            cfg = create_trainer_config(
                num_classes=args.num_classes,
                batch_size=args.batch_size,
                learning_rate=args.learning_rate,
                max_iterations=args.max_iterations,
                output_dir=args.output_dir,
                config_file=args.config_file,
                eval_period=eval_period,
                pretrained=getattr(args, 'pretrained', False),
            )

            trainer = train_model(cfg, resume=args.resume,
                                  early_stopping_patience=early_stopping_patience)
            if not trainer:
                print("Training failed!")
                sys.exit(1)
            print("Training completed successfully!")
            model_path = os.path.join(args.output_dir, "model_final.pth")

        # --- EVALUATE ---
        eval_splits = []
        if args.eval_val:
            eval_splits.append(("val", "Validation (ID holdout)"))
        if args.eval_ood:
            eval_splits.append(("ood_test", "Out-of-Distribution (OOD test — reported metric)"))
        if args.eval_ood_train:
            eval_splits.append(("ood_train", "OOD Train (shift analysis only)"))

        if eval_splits:
            if model_path is None or not os.path.exists(model_path):
                print(f"Error: model not found at '{model_path}'. Train first or pass --model-path.")
                sys.exit(1)

            predictor = create_predictor(
                model_path=model_path,
                num_classes=args.num_classes,
                score_threshold=args.coco_score_threshold,
                config_file=args.config_file,
            )
            if not predictor:
                print("Failed to create predictor!")
                sys.exit(1)

            all_results = {}
            for split_name, split_label in eval_splits:
                if not status.get(split_name):
                    print(f"Warning: {split_label} dataset not registered, skipping.")
                    continue

                print(f"\n{'=' * 60}")
                print(f"PHASE: EVALUATE {split_label.upper()} ({split_name})")
                print("=" * 60)

                try:
                    eval_output = os.path.join(args.output_dir, f"eval_{split_name}")
                    results = evaluate_model(
                        predictor, split_name, eval_output,
                        score_threshold=args.score_threshold,
                    )
                    if results:
                        all_results[split_name] = results
                except Exception as exc:
                    print(f"ERROR evaluating split '{split_name}': {exc}")
                    import traceback; traceback.print_exc()

            # Print summary
            print("\n" + "=" * 60)
            print("EVALUATION SUMMARY")
            print("=" * 60)
            for split_name, res in all_results.items():
                bbox = res.get("bbox", {})
                print(f"  {split_name}:")
                print(f"    bbox  — AP={bbox.get('AP', 'N/A'):.3f}  AP50={bbox.get('AP50', 'N/A'):.3f}  AP75={bbox.get('AP75', 'N/A'):.3f}")
                tm = res.get("tree_metrics", {}).get("overall", {})
                if tm:
                    print(f"    tree  — Precision={tm.get('Precision', 'N/A')}  Recall={tm.get('Recall', 'N/A')}")
                    for k, v in tm.items():
                        if "RMSE" in k and v is not None:
                            print(f"    tree  — {k}={v}")

            if all_results:
                summary_path = os.path.join(args.output_dir, "eval_summary.json")
                with open(summary_path, "w") as f:
                    json.dump(all_results, f, indent=2)
                print(f"Results saved to: {summary_path}")

                table_results = {k: v for k, v in all_results.items() if k in ("val", "ood_test")}
                table_path = os.path.join(args.output_dir, "eval_summary_table.png")
                generate_eval_summary_table(table_results, args.config, table_path, model_type="Faster R-CNN")
            else:
                print("ERROR: No splits evaluated successfully — no summary written.")
                sys.exit(1)

    elif args.command == 'train':
        print("=== Training Mode ===")
        
        # Register datasets
        register_datasets(args.train_annotations, args.train_images)
        
        # Visualize training data if requested
        if args.visualize:
            print("Visualizing training data...")
            visualize_dataset("train", output_path="./training_data_visualization.png")
        
        # Create config and train
        cfg = create_trainer_config(
            num_classes=args.num_classes,
            batch_size=args.batch_size,
            learning_rate=args.learning_rate,
            max_iterations=args.max_iterations,
            output_dir=args.output_dir,
            config_file=args.config_file
        )
        
        try:
            import torch
            _ngpu = torch.cuda.device_count() if torch.cuda.is_available() else 0
        except Exception:
            _ngpu = 0
        if _ngpu and _ngpu > 1:
            print(f"Using distributed training on {_ngpu} GPUs")
            launch(_distributed_train, num_gpus_per_machine=_ngpu, num_machines=1, machine_rank=0, dist_url="auto", args=(cfg, args.resume))
            trainer = None
        else:
            trainer = train_model(cfg, resume=args.resume)
        
        if trainer:
            print("Training completed successfully!")
        else:
            print("Training failed!")
    
    elif args.command == 'predict':
        print("=== Prediction Mode ===")
        
        # Create predictor
        predictor = create_predictor(
            model_path=args.model_path,
            num_classes=args.num_classes,
            score_threshold=args.score_threshold,
            config_file=args.config_file
        )
        
        # Run prediction
        outputs = predict_image(
            predictor, 
            args.input_image, 
            output_path=args.output_image,
            visualize=args.output_image is None
        )
        
        if outputs:
            print("Prediction completed successfully!")
        else:
            print("Prediction failed!")
    
    elif args.command == 'evaluate':
        print("=== Evaluation Mode ===")
        
        # Register test dataset
        register_datasets(test_annotations=args.test_annotations, test_images=args.test_images)
        
        # Create predictor
        predictor = create_predictor(
            model_path=args.model_path,
            num_classes=args.num_classes,
            score_threshold=args.score_threshold,
            config_file=args.config_file
        )
        
        # Run evaluation (zone-aware if zone CSV provided)
        results = evaluate_model_by_zone(predictor, "test", args.output_dir, getattr(args, 'zone_csv', None))
        
        # Visualize comparison if requested
        if args.visualize_comparison:
            print("Creating prediction vs ground truth visualization...")
            visualize_predictions_vs_ground_truth(predictor, "test", output_path="./evaluation_comparison.png")
        
        if results:
            print("Evaluation completed successfully!")
        else:
            print("Evaluation failed!")
    
    else:
        parser.print_help()


if __name__ == "__main__":
    main()
