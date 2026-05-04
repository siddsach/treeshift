#!/usr/bin/env python3
"""
Shared utilities for tree-distribution-shift project.
Used by detectron_fastrcnn.py, detectron_maskrcnn.py, plain_detr.py, and grounding_dino.py.

Data is loaded via the ``tree_shift`` pip package (>= 0.2.0).
"""

import os
import json
import subprocess
import numpy as np
from pathlib import Path
from collections import defaultdict
from math import factorial

_DEFAULT_COCO_EXPORT_DIR = os.path.join(".", "coco_export")


def resolve_coco_paths(coco_dir, config_name):
    """Resolve standard paths from a tree_shift COCO export directory.

    Expected layout (produced by ``tree-shift export``):
        <coco_dir>/<config_name>/train/images/
        <coco_dir>/<config_name>/train/annotations/instances_train.json
        <coco_dir>/<config_name>/id_test/images/
        <coco_dir>/<config_name>/id_test/annotations/instances_id_test.json
        <coco_dir>/<config_name>/ood_train/images/
        <coco_dir>/<config_name>/ood_train/annotations/instances_ood_train.json
        <coco_dir>/<config_name>/ood_test/images/
        <coco_dir>/<config_name>/ood_test/annotations/instances_ood_test.json
    """
    base = Path(coco_dir) / config_name
    paths = {
        "train_images": str(base / "train" / "images"),
        "train_annotations": str(base / "train" / "annotations" / "instances_train.json"),
        "val_images": str(base / "val" / "images"),
        "val_annotations": str(base / "val" / "annotations" / "instances_val.json"),
        "ood_train_images": str(base / "ood_train" / "images"),
        "ood_train_annotations": str(base / "ood_train" / "annotations" / "instances_ood_train.json"),
        "ood_test_images": str(base / "ood_test" / "images"),
        "ood_test_annotations": str(base / "ood_test" / "annotations" / "instances_ood_test.json"),
    }

    # The new dataset structure uses id_test as the canonical ID eval split.
    # Prefer id_test over the legacy val path; fall back to id_val / train.
    val_ann = Path(paths["val_annotations"])
    if not val_ann.exists():
        alt_splits = ["id_test", "id_val", "validation", "test"]
        for split in alt_splits:
            ann_dir = base / split / "annotations"
            img_dir = base / split / "images"
            if not ann_dir.exists():
                continue
            ann_candidates = sorted(ann_dir.glob("instances_*.json"))
            if ann_candidates and img_dir.exists():
                paths["val_images"] = str(img_dir)
                paths["val_annotations"] = str(ann_candidates[0])
                print(
                    f"Using '{split}' as ID eval split fallback "
                    f"(val not available for config '{config_name}')."
                )
                break
        else:
            train_ann = Path(paths["train_annotations"])
            train_imgs = Path(paths["train_images"])
            if train_ann.exists() and train_imgs.exists():
                paths["val_images"] = str(train_imgs)
                paths["val_annotations"] = str(train_ann)
                print(
                    f"Warning: val split missing for config '{config_name}'. "
                    "Falling back to train split for ID evaluation."
                )

    return paths


def ensure_tree_shift_export(config, coco_dir=None):
    """Download tree_shift data (if needed) and export to COCO format.

    Skips the export when the annotation files already exist on disk.

    Args:
        config: tree_shift benchmark config string (e.g. ``intl_train_IN__ood_US``).
        coco_dir: directory to write COCO exports into.  Defaults to ``./coco_export``.

    Returns:
        dict of resolved COCO paths (same as :func:`resolve_coco_paths`).
    """
    if coco_dir is None:
        coco_dir = _DEFAULT_COCO_EXPORT_DIR

    base = Path(coco_dir) / config
    train_ann = base / "train" / "annotations" / "instances_train.json"
    id_test_ann = base / "id_test" / "annotations" / "instances_id_test.json"
    val_ann = base / "val" / "annotations" / "instances_val.json"
    ood_test_ann = base / "ood_test" / "annotations" / "instances_ood_test.json"
    ood_train_ann = base / "ood_train" / "annotations" / "instances_ood_train.json"

    if train_ann.exists() and (
        (ood_test_ann.exists() and ood_train_ann.exists()) or id_test_ann.exists() or val_ann.exists()
    ):
        print(f"tree_shift COCO export already exists at {base}")
    else:
        print(f"Downloading and exporting tree_shift data ...")
        print(f"  Config : {config}")
        print(f"  Output : {coco_dir}")
        # tree-shift expects HuggingFace 'datasets'; Plain-DETR's datasets shadows it.
        # Use a clean env so tree-shift finds the correct package.
        env = os.environ.copy()
        if "PYTHONPATH" in env:
            paths = [
                p for p in env["PYTHONPATH"].split(os.pathsep)
                if p and "Plain-DETR" not in p and "plain_detr" not in p.lower()
            ]
            env["PYTHONPATH"] = os.pathsep.join(paths) if paths else ""
        # New dataset structure: train, id_test, ood_train, ood_test
        requested_splits = ["train", "ood_train", "ood_test"]
        try:
            out = subprocess.check_output(
                [
                    "python", "-c",
                    (
                        "import json, datasets; "
                        "print(json.dumps(datasets.get_dataset_split_names("
                        "'aadityabuilds/tree-distribution-shift', "
                        f"'{config}'"
                        ")))"
                    ),
                ],
                env=env,
                text=True,
            ).strip()
            available_splits = set(json.loads(out))
            requested_splits = [
                s for s in ["train", "id_test", "ood_train", "ood_test"]
                if s in available_splits
            ]
            if "train" not in requested_splits or (
                "ood_test" not in requested_splits and "id_test" not in requested_splits
            ):
                raise RuntimeError(
                    f"Config '{config}' must contain 'train' and either 'ood_test' or 'id_test'. "
                    f"Available: {sorted(available_splits)}"
                )
            print(f"  Export splits: {requested_splits}")
        except Exception as e:
            print(
                f"Warning: could not infer available splits ({e}); "
                "falling back to ['train', 'ood_train', 'ood_test']."
            )
            requested_splits = ["train", "ood_train", "ood_test"]

        export_cmd = ["tree-shift", "export", "--config", config, "--out", str(coco_dir), "--splits"] + requested_splits
        export_proc = subprocess.run(
            export_cmd,
            env=env,
            text=True,
        )
        if export_proc.returncode != 0:
            # Retry with the legacy distribution-shift split set. ID-only
            # configs such as india_random_80_20 should have succeeded above
            # after split discovery because they do not expose ood_* splits.
            print("tree-shift export failed; retrying with ['train', 'ood_train', 'ood_test'].")
            subprocess.check_call(
                ["tree-shift", "export", "--config", config, "--out", str(coco_dir),
                 "--splits", "train", "ood_train", "ood_test"],
                env=env,
            )
        print("Export complete.")

    # Ensure a 'val' split always exists.  New dataset configs expose id_test
    # as the canonical ID eval split.  Models such as Plain-DETR hard-code the
    # path <coco_dir>/<config>/val/images; when 'val' was not exported we create
    # symlinks pointing to id_test (preferred) or train as a last resort.
    val_images_dir   = base / "val" / "images"
    val_ann_file     = base / "val" / "annotations" / "instances_val.json"
    id_test_images   = base / "id_test" / "images"
    id_test_ann_candidates = sorted((base / "id_test" / "annotations").glob("instances_*.json")) \
        if (base / "id_test" / "annotations").exists() else []
    train_images_dir = base / "train" / "images"
    train_ann_file   = base / "train" / "annotations" / "instances_train.json"

    if not val_images_dir.exists():
        (base / "val" / "annotations").mkdir(parents=True, exist_ok=True)
        if id_test_images.exists() and id_test_ann_candidates:
            # Preferred: symlink val → id_test
            val_images_dir.symlink_to(id_test_images.resolve())
            if not val_ann_file.exists():
                val_ann_file.symlink_to(id_test_ann_candidates[0].resolve())
            print(
                f"Note: no 'val' split in export for config '{config}'; "
                "created val→id_test symlinks."
            )
        elif train_images_dir.exists():
            # Fallback: symlink val → train
            val_images_dir.symlink_to(train_images_dir.resolve())
            if not val_ann_file.exists():
                val_ann_file.symlink_to(train_ann_file.resolve())
            print(
                f"Note: no 'val' or 'id_test' split for config '{config}'; "
                "created val→train symlinks."
            )

    return resolve_coco_paths(coco_dir, config)


def debug_coco_annotations(annotation_file):
    """Debug COCO annotations to see what's being loaded."""
    if not os.path.exists(annotation_file):
        print(f"Annotation file not found: {annotation_file}")
        return

    with open(annotation_file, 'r') as f:
        data = json.load(f)

    print(f"=== Debugging {annotation_file} ===")
    print(f"Total images: {len(data['images'])}")
    print(f"Total annotations: {len(data['annotations'])}")
    print(f"Categories: {[cat['name'] for cat in data['categories']]}")

    has_bbox = any('bbox' in ann for ann in data['annotations'])
    has_segmentation = any('segmentation' in ann for ann in data['annotations'])
    has_area = any('area' in ann for ann in data['annotations'])

    print(f"Has bbox: {has_bbox}")
    print(f"Has segmentation: {has_segmentation}")
    print(f"Has area: {has_area}")

    for i, ann in enumerate(data['annotations'][:3]):
        print(f"Annotation {i}: keys={list(ann.keys())}")
        if 'bbox' in ann:
            print(f"  bbox: {ann['bbox']}")
        if 'segmentation' in ann:
            print(f"  segmentation type: {type(ann['segmentation'])}")
        if 'area' in ann:
            print(f"  area: {ann['area']}")
    print("=" * 50)


def _compute_iou(box_a, box_b):
    """Compute IoU between two boxes in [x, y, w, h] COCO format."""
    ax, ay, aw, ah = box_a
    bx, by, bw, bh = box_b
    a_x1, a_y1, a_x2, a_y2 = ax, ay, ax + aw, ay + ah
    b_x1, b_y1, b_x2, b_y2 = bx, by, bx + bw, by + bh
    inter_x1 = max(a_x1, b_x1)
    inter_y1 = max(a_y1, b_y1)
    inter_x2 = min(a_x2, b_x2)
    inter_y2 = min(a_y2, b_y2)
    inter_area = max(0, inter_x2 - inter_x1) * max(0, inter_y2 - inter_y1)
    union_area = aw * ah + bw * bh - inter_area
    return inter_area / union_area if union_area > 0 else 0.0


def compute_tree_detection_metrics_from_boxes(
    all_gt_boxes, all_pred_boxes, iou_threshold=0.50, gsd=None
):
    """Compute detailed tree-detection metrics with per-size breakdown.

    This is a framework-agnostic version that works with raw box lists.

    Args:
        all_gt_boxes: list of lists, each inner list contains GT boxes as [x, y, w, h].
        all_pred_boxes: list of lists, each inner list contains pred boxes as [x, y, w, h]
                        sorted by descending confidence.
        iou_threshold: IoU threshold for a true-positive match.
        gsd: ground sampling distance in metres/pixel.

    Returns:
        dict with keys 'overall', 'small', 'large', each containing the metrics.
    """
    # 1. Determine median GT area for size split
    all_gt_areas = []
    for gt_boxes in all_gt_boxes:
        for bx, by, bw, bh in gt_boxes:
            all_gt_areas.append(bw * bh)
    if len(all_gt_areas) == 0:
        print("No GT annotations found – skipping tree detection metrics.")
        return None
    area_median = float(np.median(all_gt_areas))
    print(f"Tree-size split: median GT bbox area = {area_median:.1f} px²  "
          f"(small < median, large >= median)")

    # 2. Per-image greedy matching
    tp_counts = {"overall": 0, "small": 0, "large": 0}
    fp_counts = {"overall": 0, "small": 0, "large": 0}
    fn_counts = {"overall": 0, "small": 0, "large": 0}
    crown_errors = {"overall": [], "small": [], "large": []}
    centre_errors = {"overall": [], "small": [], "large": []}

    for gt_boxes, pred_boxes in zip(all_gt_boxes, all_pred_boxes):
        gt_sizes = []
        for bx, by, bw, bh in gt_boxes:
            gt_sizes.append("small" if bw * bh < area_median else "large")

        matched_gt = set()
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
                size_key = gt_sizes[best_gi]

                gt_w = gt_boxes[best_gi][2]
                pd_w = pred_boxes[pi][2]
                crown_errors["overall"].append((pd_w - gt_w) ** 2)
                crown_errors[size_key].append((pd_w - gt_w) ** 2)

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
                fp_counts["overall"] += 1
                if len(gt_boxes) > 0:
                    closest_gi = min(range(len(gt_boxes)),
                                     key=lambda g: _compute_iou(pred_boxes[pi], gt_boxes[g]),
                                     default=0)
                    fp_counts[gt_sizes[closest_gi]] += 1

        for gi in range(len(gt_boxes)):
            if gi not in matched_gt:
                fn_counts["overall"] += 1
                fn_counts[gt_sizes[gi]] += 1

    # 3. Aggregate
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
        return {
            "TP": tp, "FP": fp, "FN": fn,
            "Precision": round(precision, 4),
            "Recall": round(recall, 4),
            f"Crown_Diameter_RMSE_{unit}": round(crown_rmse, 3) if crown_rmse is not None else None,
            f"Geolocation_RMSE_{unit}": round(geo_rmse, 3) if geo_rmse is not None else None,
        }

    metrics = {}
    for key in ["overall", "small", "large"]:
        metrics[key] = _build_bucket(key)

    # 4. Pretty-print
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


def generate_eval_summary_table(all_results, config_name, output_path, model_type="Model"):
    """Generate a table image summarising AP metrics across eval splits.

    Columns:
        Train - Test | AP@0.50 [bbox] | AP@[0.50:0.95] [bbox] | APs [bbox] | APm [bbox] | APl [bbox]
                     | Precision | Recall | Crown Diam RMSE | Geoloc RMSE
    """
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except ImportError:
        print("matplotlib not available — skipping table image generation")
        return

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
            _fmt(tm.get("Crown_Diameter_RMSE_px") or tm.get("Crown_Diameter_RMSE_m")),
            _fmt(tm.get("Geolocation_RMSE_px") or tm.get("Geolocation_RMSE_m")),
        ]
        rows.append(row)

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

    for j in range(len(columns)):
        cell = table[0, j]
        cell.set_facecolor("#4472C4")
        cell.set_text_props(color="white", fontweight="bold")

    for i in range(1, n_rows + 1):
        colour = "#D9E2F3" if i % 2 == 1 else "white"
        for j in range(len(columns)):
            table[i, j].set_facecolor(colour)

    os.makedirs(os.path.dirname(output_path) if os.path.dirname(output_path) else ".", exist_ok=True)
    fig.savefig(output_path, dpi=150, bbox_inches="tight", facecolor="white")
    plt.close(fig)
    print(f"Evaluation summary table saved to: {output_path}")


# ---------------------------------------------------------------------------
# Unified per-image evaluation used by all model scripts
# ---------------------------------------------------------------------------

def _match_single_image(gt_boxes, pred_boxes, iou_threshold=0.50):
    """Greedy-match predicted boxes to GT boxes for one image.

    Args:
        gt_boxes: list of [x, y, w, h] ground-truth boxes.
        pred_boxes: list of [x, y, w, h] predicted boxes **sorted by
                    descending confidence**.
        iou_threshold: minimum IoU for a true-positive match.

    Returns:
        (tp, fp, fn) counts for this image.
    """
    matched_gt = set()
    tp = fp = 0
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
            tp += 1
        else:
            fp += 1
    fn = len(gt_boxes) - len(matched_gt)
    return tp, fp, fn


def evaluate_detections(
    image_records,
    split_name,
    output_dir,
    iou_threshold=0.50,
    gsd=None,
    coco_bbox_stats=None,
):
    """Unified evaluation producing aggregate + per-image metrics.

    This is the **single evaluation function** used by all model scripts
    (Plain-DETR, Grounding DINO, Faster R-CNN, Mask R-CNN).

    Args:
        image_records: list of dicts, each with keys:
            ``image_id``   (int)
            ``filename``   (str)
            ``gt_boxes``   (list of [x,y,w,h])
            ``pred_boxes`` (list of [x,y,w,h], sorted by descending score)
            ``pred_scores``(list of float, same order as pred_boxes)
        split_name: string label for the split (e.g. "val", "ood_test").
        output_dir: directory to write results into.
        iou_threshold: IoU threshold for TP matching.
        gsd: ground-sampling distance (m/px); ``None`` → pixel units.
        coco_bbox_stats: optional pre-computed COCO AP dict with keys
            ``AP``, ``AP50``, ``AP75``, ``APs``, ``APm``, ``APl``.

    Returns:
        dict with keys ``bbox`` (COCO AP), ``tree_metrics`` (aggregate),
        and ``per_image`` (list of per-image metric dicts).
    """
    os.makedirs(output_dir, exist_ok=True)

    # ------------------------------------------------------------------
    # 1.  Per-image matching
    # ------------------------------------------------------------------
    per_image = []
    for rec in image_records:
        gt = rec["gt_boxes"]
        pred = rec["pred_boxes"]
        tp, fp, fn = _match_single_image(gt, pred, iou_threshold)
        recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
        precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
        per_image.append({
            "image_id": rec["image_id"],
            "filename": rec["filename"],
            "split": split_name,
            "tp": tp,
            "fp": fp,
            "fn": fn,
            "n_gt": len(gt),
            "n_pred": len(pred),
            "recall": round(recall, 6),
            "precision": round(precision, 6),
        })

    # ------------------------------------------------------------------
    # 2.  Aggregate tree-detection metrics (with size breakdown)
    # ------------------------------------------------------------------
    all_gt_boxes = [r["gt_boxes"] for r in image_records]
    all_pred_boxes = [r["pred_boxes"] for r in image_records]
    tree_metrics = compute_tree_detection_metrics_from_boxes(
        all_gt_boxes, all_pred_boxes, iou_threshold=iou_threshold, gsd=gsd,
    )

    # ------------------------------------------------------------------
    # 3.  Assemble results
    # ------------------------------------------------------------------
    results = {}
    if coco_bbox_stats is not None:
        results["bbox"] = coco_bbox_stats
    if tree_metrics:
        results["tree_metrics"] = tree_metrics
        if coco_bbox_stats:
            tree_metrics["overall"]["AP50"] = coco_bbox_stats.get("AP50")

    results["per_image"] = per_image

    # ------------------------------------------------------------------
    # 4.  Save artefacts
    # ------------------------------------------------------------------
    per_image_path = os.path.join(output_dir, "per_image_results.json")
    with open(per_image_path, "w") as f:
        json.dump(per_image, f, indent=2)
    print(f"Per-image results ({len(per_image)} images) saved to: {per_image_path}")

    if tree_metrics:
        tree_path = os.path.join(output_dir, "tree_detection_metrics.json")
        with open(tree_path, "w") as f:
            json.dump(tree_metrics, f, indent=2)
        print(f"Tree detection metrics saved to: {tree_path}")

    return results
