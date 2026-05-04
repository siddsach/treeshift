#!/usr/bin/env python3
"""
Plain-DETR Training and Inference Script for tree-distribution-shift project.
Wraps the Plain-DETR codebase to match the CLI interface of the other model scripts.
Data is loaded via the ``tree_shift`` pip package (>= 0.2.0).

Usage (recommended — tree_shift auto-download):
    # Train only
    python plain_detr.py run --config intl_train_IN__ood_US --train

    # Train + evaluate val + evaluate OOD
    python plain_detr.py run --config intl_train_IN__ood_US --train --eval-val --eval-ood

    # Evaluate only (requires trained model)
    python plain_detr.py run --config intl_train_IN__ood_US --eval-val --eval-ood --model-path ./output/checkpoint.pth

    # With explicit COCO export directory
    python plain_detr.py run --config intl_train_IN__ood_US --coco-dir ./my_coco --train --eval-val --eval-ood

    All extra Plain-DETR flags (e.g. --backbone, --epochs, --lr, --two_stage, etc.)
    can be appended after the run arguments above.
"""

import sys
import os
import json
import argparse
import time
import datetime
import random
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader

# Add Plain-DETR to sys.path so its internal imports work
_PLAIN_DETR_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "Plain-DETR")
sys.path.insert(0, _PLAIN_DETR_DIR)

import datasets
import datasets.samplers as samplers
import util.misc as utils
from datasets import build_dataset, get_coco_api_from_dataset
from engine import evaluate, train_one_epoch
from models import build_model
from main import get_args_parser as get_plain_detr_args_parser

# Shared project utilities
from shared_utils import (
    resolve_coco_paths,
    ensure_tree_shift_export,
    debug_coco_annotations,
    compute_tree_detection_metrics_from_boxes,
    generate_eval_summary_table,
    evaluate_detections,
)


def _torch_load(path, map_location=None):
    try:
        return torch.load(path, map_location=map_location, weights_only=False)
    except TypeError:
        return torch.load(path, map_location=map_location)


def build_plain_detr_args(run_args, extra_args, coco_dir):
    """Merge the wrapper's run_args with Plain-DETR's native argparse defaults.

    Args:
        run_args: parsed wrapper CLI namespace.
        extra_args: passthrough Plain-DETR flags.
        coco_dir: resolved COCO export directory (from tree_shift).

    Returns a namespace that Plain-DETR's main() / build_model() can consume.
    """
    parser = argparse.ArgumentParser(
        "Plain-DETR (internal)", parents=[get_plain_detr_args_parser()]
    )
    args = parser.parse_args(extra_args)

    args.hf_coco_dir = coco_dir
    args.hf_coco_config = run_args.config
    args.dataset_file = "coco"

    if run_args.num_classes is not None:
        args.num_classes = run_args.num_classes
    if run_args.batch_size is not None:
        args.batch_size = run_args.batch_size
    if run_args.output_dir is not None:
        args.output_dir = run_args.output_dir
    if run_args.num_workers is not None:
        args.num_workers = run_args.num_workers

    return args


# ---------------------------------------------------------------------------
# Evaluation helper: run Plain-DETR evaluate() on a given split and collect
# COCO AP metrics + tree-specific metrics.
# ---------------------------------------------------------------------------

def evaluate_split(
    model, criterion, postprocessors, args, split_name, device, output_dir,
    score_threshold=0.3,
):
    """Evaluate on a single split (val / ood_test) and return results dict.

    Uses the unified ``evaluate_detections`` from shared_utils so that all
    models produce the same per-image result format.
    """
    dataset = build_dataset(image_set=split_name, args=args)
    sampler = torch.utils.data.SequentialSampler(dataset)
    data_loader = DataLoader(
        dataset,
        args.batch_size,
        sampler=sampler,
        drop_last=False,
        collate_fn=utils.collate_fn,
        num_workers=args.num_workers,
        pin_memory=True,
    )
    base_ds = get_coco_api_from_dataset(dataset)

    eval_output = os.path.join(output_dir, f"eval_{split_name}")
    os.makedirs(eval_output, exist_ok=True)

    # --- COCO AP via Plain-DETR's own evaluator ---
    stats, coco_evaluator = evaluate(
        model,
        criterion,
        postprocessors,
        data_loader,
        base_ds,
        device,
        eval_output,
        step=0,
        use_wandb=False,
        reparam=args.reparam,
        max_batches=int(getattr(args, "eval_max_batches", 0) or 0),
    )

    coco_bbox_stats = None
    if "coco_eval_bbox" in stats:
        s = stats["coco_eval_bbox"]
        # pycocotools returns stats in [0, 1]; convert to [0, 100] to match
        # Detectron2's COCOEvaluator convention used by fastrcnn/maskrcnn.
        # Values of -1.0 mean "no objects of that size" → store as None.
        def _ap_pct(v):
            return None if float(v) < 0 else float(v) * 100
        coco_bbox_stats = {
            "AP": _ap_pct(s[0]), "AP50": _ap_pct(s[1]), "AP75": _ap_pct(s[2]),
            "APs": _ap_pct(s[3]), "APm": _ap_pct(s[4]), "APl": _ap_pct(s[5]),
        }

    # --- Collect per-image predictions for unified evaluation ---
    model.eval()
    coco_api = base_ds
    image_records = []

    for samples, targets in data_loader:
        samples = samples.to(device)
        targets_dev = [{k: v.to(device) for k, v in t.items()} for t in targets]

        with torch.no_grad():
            outputs = model(samples)

        orig_target_sizes = torch.stack([t["orig_size"] for t in targets_dev], dim=0)
        target_sizes = torch.stack([t["size"] for t in targets_dev], dim=0)
        if args.reparam:
            batch_results = postprocessors["bbox"](outputs, target_sizes, orig_target_sizes)
        else:
            batch_results = postprocessors["bbox"](outputs, orig_target_sizes)

        for target, res in zip(targets_dev, batch_results):
            image_id = int(target["image_id"].item())

            ann_ids = coco_api.getAnnIds(imgIds=[image_id])
            anns = coco_api.loadAnns(ann_ids)
            gt_boxes = [ann["bbox"] for ann in anns]

            pred_scores = res["scores"].cpu().numpy()
            pred_boxes_xyxy = res["boxes"].cpu().numpy()
            keep = pred_scores >= score_threshold
            order = np.argsort(-pred_scores[keep])
            kept_scores = pred_scores[keep][order]
            kept_boxes = pred_boxes_xyxy[keep][order]
            pred_boxes_xywh = []
            for box in kept_boxes:
                x1, y1, x2, y2 = box
                pred_boxes_xywh.append([float(x1), float(y1), float(x2 - x1), float(y2 - y1)])

            img_info = coco_api.loadImgs(image_id)[0]
            image_records.append({
                "image_id": image_id,
                "filename": img_info.get("file_name", ""),
                "gt_boxes": gt_boxes,
                "pred_boxes": pred_boxes_xywh,
                "pred_scores": [float(s) for s in kept_scores],
            })

    return evaluate_detections(
        image_records,
        split_name=split_name,
        output_dir=eval_output,
        iou_threshold=0.50,
        coco_bbox_stats=coco_bbox_stats,
    )


# ---------------------------------------------------------------------------
# Main training loop (adapted from Plain-DETR main.py)
# ---------------------------------------------------------------------------

def train(model, criterion, postprocessors, args, device):
    """Run the full Plain-DETR training loop."""
    model_without_ddp = model
    n_parameters = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print("number of params:", n_parameters)

    scaler = None
    if args.use_fp16:
        scaler = torch.cuda.amp.GradScaler()

    dataset_train = build_dataset(image_set="train", args=args)
    dataset_val = build_dataset(image_set="val", args=args)

    if args.distributed:
        if args.cache_mode:
            sampler_train = samplers.NodeDistributedSampler(dataset_train)
            sampler_val = samplers.NodeDistributedSampler(dataset_val, shuffle=False)
        else:
            sampler_train = samplers.DistributedSampler(dataset_train)
            sampler_val = samplers.DistributedSampler(dataset_val, shuffle=False)
    else:
        sampler_train = torch.utils.data.RandomSampler(dataset_train)
        sampler_val = torch.utils.data.SequentialSampler(dataset_val)

    batch_sampler_train = torch.utils.data.BatchSampler(
        sampler_train, args.batch_size, drop_last=True
    )

    data_loader_train = DataLoader(
        dataset_train,
        batch_sampler=batch_sampler_train,
        collate_fn=utils.collate_fn,
        num_workers=args.num_workers,
        pin_memory=True,
    )
    data_loader_val = DataLoader(
        dataset_val,
        args.batch_size,
        sampler=sampler_val,
        drop_last=False,
        collate_fn=utils.collate_fn,
        num_workers=args.num_workers,
        pin_memory=True,
    )

    param_dicts = utils.get_param_dict(
        model_without_ddp, args, use_layerwise_decay=args.use_layerwise_decay
    )
    if args.sgd:
        optimizer = torch.optim.SGD(
            param_dicts, lr=args.lr, momentum=0.9, weight_decay=args.weight_decay
        )
    else:
        optimizer = torch.optim.AdamW(
            param_dicts, lr=args.lr, weight_decay=args.weight_decay
        )

    epoch_iter = len(data_loader_train)
    if args.warmup:
        lambda0 = lambda cur_iter: cur_iter / args.warmup if cur_iter < args.warmup else (0.1 if cur_iter > args.lr_drop * epoch_iter else 1)
    else:
        lambda0 = lambda cur_iter: 0.1 if cur_iter > args.lr_drop * epoch_iter else 1
    lr_scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda=lambda0)

    if args.distributed:
        model = torch.nn.parallel.DistributedDataParallel(model, device_ids=[args.gpu])
        model_without_ddp = model.module

    base_ds = get_coco_api_from_dataset(dataset_val)

    if args.use_wandb and utils.get_rank() == 0:
        import wandb
        wandb.init(
            entity=getattr(args, "wandb_entity", None),
            project="Plain-DETR",
            id=getattr(args, "wandb_name", None),
            name=getattr(args, "wandb_name", None),
        )

    output_dir = Path(args.output_dir)

    # Auto-resume
    if args.auto_resume:
        resume_from = utils.find_latest_checkpoint(output_dir)
        if resume_from is not None:
            print(f"Use autoresume, overwrite args.resume with {resume_from}")
            args.resume = resume_from
        else:
            print(f"Use autoresume, but can not find checkpoint in {output_dir}")

    if args.resume and os.path.exists(args.resume):
        checkpoint = _torch_load(args.resume, map_location="cpu")
        missing_keys, unexpected_keys = model_without_ddp.load_state_dict(
            checkpoint["model"], strict=False
        )
        if len(missing_keys) > 0:
            print("Missing Keys: {}".format(missing_keys))
        if len(unexpected_keys) > 0:
            print("Unexpected Keys: {}".format(unexpected_keys))
        if (
            "optimizer" in checkpoint
            and "lr_scheduler" in checkpoint
            and "epoch" in checkpoint
        ):
            import copy as _copy
            p_groups = _copy.deepcopy(optimizer.param_groups)
            optimizer.load_state_dict(checkpoint["optimizer"])
            for pg, pg_old in zip(optimizer.param_groups, p_groups):
                pg["lr"] = pg_old["lr"]
                pg["initial_lr"] = pg_old["initial_lr"]
            lr_scheduler.load_state_dict(checkpoint["lr_scheduler"])
            lr_scheduler.step(lr_scheduler.last_epoch)
            if "iteration" in checkpoint and checkpoint["iteration"] is not None:
                args.start_epoch = checkpoint["epoch"]
                args.start_iter = int(checkpoint["iteration"]) + 1
            else:
                args.start_epoch = checkpoint["epoch"] + 1
            if args.use_fp16 and scaler and "scaler" in checkpoint:
                scaler.load_state_dict(checkpoint["scaler"])

        # Quick validation after resume
        test_stats, _ = evaluate(
            model, criterion, postprocessors, data_loader_val, base_ds,
            device, args.output_dir, step=args.start_epoch * len(data_loader_train),
            use_wandb=args.use_wandb, reparam=args.reparam,
            max_batches=int(getattr(args, "eval_max_batches", 0) or 0),
        )

    print("Start training")
    start_time = time.time()

    early_stop_best = float("-inf")
    early_stop_best_epoch = -1
    early_stop_bad_epochs = 0
    if args.resume and os.path.exists(args.resume):
        ckpt = _torch_load(args.resume, map_location="cpu")
        if isinstance(ckpt, dict) and "early_stop" in ckpt and isinstance(ckpt["early_stop"], dict):
            es = ckpt["early_stop"]
            early_stop_best = float(es.get("best", early_stop_best))
            early_stop_best_epoch = int(es.get("best_epoch", early_stop_best_epoch))
            early_stop_bad_epochs = int(es.get("bad_epochs", early_stop_bad_epochs))

    def _save_periodic_checkpoint(epoch, iteration):
        if not args.output_dir:
            return
        save_dict = {
            "model": model_without_ddp.state_dict(),
            "optimizer": optimizer.state_dict(),
            "lr_scheduler": lr_scheduler.state_dict(),
            "epoch": epoch,
            "iteration": iteration,
            "args": args,
            "early_stop": {
                "best": early_stop_best,
                "best_epoch": early_stop_best_epoch,
                "bad_epochs": early_stop_bad_epochs,
            },
        }
        if args.use_fp16 and scaler:
            save_dict["scaler"] = scaler.state_dict()
        utils.save_on_master(save_dict, output_dir / "checkpoint.pth")

    for epoch in range(args.start_epoch, args.epochs):
        if args.distributed:
            sampler_train.set_epoch(epoch)
        start_iter = args.start_iter if epoch == args.start_epoch else 0
        train_stats = train_one_epoch(
            model, criterion, data_loader_train, optimizer, device, epoch,
            lr_scheduler, args.clip_max_norm,
            k_one2many=args.k_one2many,
            lambda_one2many=args.lambda_one2many,
            use_wandb=args.use_wandb,
            use_fp16=args.use_fp16,
            scaler=scaler if args.use_fp16 else None,
            save_ckpt_every_images=args.save_ckpt_every_images,
            save_checkpoint=_save_periodic_checkpoint,
            start_iter=start_iter,
        )
        if epoch == args.start_epoch:
            args.start_iter = 0

        if args.output_dir:
            checkpoint_paths = [output_dir / "checkpoint.pth"]
            checkpoint_paths.append(output_dir / f"checkpoint{epoch:04}.pth")
            for checkpoint_path in checkpoint_paths:
                save_dict = {
                    "model": model_without_ddp.state_dict(),
                    "optimizer": optimizer.state_dict(),
                    "lr_scheduler": lr_scheduler.state_dict(),
                    "epoch": epoch,
                    "args": args,
                    "early_stop": {
                        "best": early_stop_best,
                        "best_epoch": early_stop_best_epoch,
                        "bad_epochs": early_stop_bad_epochs,
                    },
                }
                if args.use_fp16 and scaler:
                    save_dict["scaler"] = scaler.state_dict()
                utils.save_on_master(save_dict, checkpoint_path)

        test_stats, coco_evaluator = evaluate(
            model, criterion, postprocessors, data_loader_val, base_ds,
            device, args.output_dir,
            step=(epoch + 1) * len(data_loader_train),
            use_wandb=args.use_wandb, reparam=args.reparam,
            max_batches=int(getattr(args, "eval_max_batches", 0) or 0),
        )

        log_stats = {
            **{f"train_{k}": v for k, v in train_stats.items()},
            **{f"test_{k}": v for k, v in test_stats.items()},
            "epoch": epoch,
            "n_parameters": n_parameters,
        }

        if args.output_dir and utils.is_main_process():
            with (output_dir / "log.txt").open("a") as f:
                f.write(json.dumps(log_stats) + "\n")
            if coco_evaluator is not None:
                (output_dir / "eval").mkdir(exist_ok=True)
                if "bbox" in coco_evaluator.coco_eval:
                    filenames = ["latest.pth"]
                    if epoch % 50 == 0:
                        filenames.append(f"{epoch:03}.pth")
                    for name in filenames:
                        torch.save(
                            coco_evaluator.coco_eval["bbox"].eval,
                            output_dir / "eval" / name,
                        )

        # Early stopping
        if args.early_stop and ("coco_eval_bbox" in test_stats):
            cur_ap = float(test_stats["coco_eval_bbox"][0])
            improved = cur_ap > (early_stop_best + float(args.early_stop_min_delta))
            if improved:
                early_stop_best = cur_ap
                early_stop_best_epoch = epoch
                early_stop_bad_epochs = 0
                if args.output_dir:
                    utils.save_on_master(
                        {
                            "model": model_without_ddp.state_dict(),
                            "optimizer": optimizer.state_dict(),
                            "lr_scheduler": lr_scheduler.state_dict(),
                            "epoch": epoch,
                            "args": args,
                            "early_stop": {
                                "best": early_stop_best,
                                "best_epoch": early_stop_best_epoch,
                                "bad_epochs": early_stop_bad_epochs,
                            },
                        },
                        output_dir / "checkpoint_best.pth",
                    )
            else:
                if (epoch + 1) >= int(args.early_stop_min_epochs):
                    early_stop_bad_epochs += 1

            stop = False
            if (epoch + 1) >= int(args.early_stop_min_epochs) and early_stop_bad_epochs >= int(args.early_stop_patience):
                stop = True

            if args.distributed:
                from torch import distributed as dist
                stop_tensor = torch.tensor([1 if stop else 0], device=device)
                dist.broadcast(stop_tensor, src=0)
                stop = bool(int(stop_tensor.item()))

            if stop:
                if utils.is_main_process():
                    print(
                        f"Early stopping triggered at epoch {epoch}. "
                        f"Best AP={early_stop_best:.4f} at epoch {early_stop_best_epoch}"
                    )
                break

    total_time = time.time() - start_time
    total_time_str = str(datetime.timedelta(seconds=int(total_time)))
    print("Training time {}".format(total_time_str))

    return model, model_without_ddp


# ---------------------------------------------------------------------------
# CLI entry point
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(
        description="Plain-DETR Training and Inference Script (tree-distribution-shift)"
    )
    subparsers = parser.add_subparsers(dest="command", help="Command to execute")

    # ---- run subcommand ----
    run_parser = subparsers.add_parser(
        "run", help="Train / evaluate using tree_shift package"
    )
    run_parser.add_argument(
        "--config", required=True,
        help="tree_shift benchmark config (e.g. intl_train_IN__ood_US)",
    )
    run_parser.add_argument(
        "--coco-dir", default=None,
        help="COCO export directory (default: ./coco_export)",
    )
    run_parser.add_argument("--train", action="store_true", help="Train on the train split")
    run_parser.add_argument("--eval-val", action="store_true", help="Evaluate on the validation (ID holdout) split")
    run_parser.add_argument("--eval-ood", action="store_true", help="Evaluate on the OOD test split (reported metric)")
    run_parser.add_argument("--eval-ood-train", action="store_true", help="Evaluate on the OOD train split (for shift analysis only)")
    run_parser.add_argument(
        "--eval-mode", choices=["distshift", "conventional"], default="distshift",
        help="distshift: train on ID only (default). conventional: train on ID + OOD.",
    )
    run_parser.add_argument(
        "--model-path", default=None,
        help="Path to trained model checkpoint (required if --train is not set but eval is requested)",
    )
    run_parser.add_argument("--num-classes", type=int, default=1, help="Number of object classes")
    run_parser.add_argument("--batch-size", type=int, default=None, help="Override batch size")
    run_parser.add_argument("--output-dir", default="./plain_detr_output", help="Output directory")
    run_parser.add_argument("--score-threshold", type=float, default=0.3, help="Score threshold for tree metrics")
    run_parser.add_argument("--num-workers", type=int, default=None, help="Override num_workers")

    args, extra_args = parser.parse_known_args()

    if args.command is None:
        parser.print_help()
        sys.exit(1)

    if args.command == "run":
        if not args.train and not args.eval_val and not args.eval_ood and not args.eval_ood_train:
            print("Error: specify at least one of --train, --eval-val, --eval-ood, --eval-ood-train")
            sys.exit(1)

        # Auto-download & export via tree_shift
        paths = ensure_tree_shift_export(args.config, coco_dir=args.coco_dir)
        coco_dir = args.coco_dir or os.path.join(".", "coco_export")
        print(f"Config: {args.config}")
        for k, v in paths.items():
            exists = os.path.exists(v)
            print(f"  {k}: {v}  {'OK' if exists else 'MISSING'}")

        # Build the full Plain-DETR args namespace
        detr_args = build_plain_detr_args(args, extra_args, coco_dir=coco_dir)

        # Ensure output dir exists
        if detr_args.output_dir:
            Path(detr_args.output_dir).mkdir(parents=True, exist_ok=True)

        # Init distributed mode
        utils.init_distributed_mode(detr_args)
        print(detr_args)

        device = torch.device(detr_args.device)

        # Seed
        seed = detr_args.seed + utils.get_rank()
        torch.manual_seed(seed)
        np.random.seed(seed)
        random.seed(seed)

        # Build model
        model, criterion, postprocessors = build_model(detr_args)
        model.to(device)

        # --- TRAIN ---
        model_path = args.model_path
        if args.train:
            print("\n" + "=" * 60)
            print("PHASE: TRAINING (Plain-DETR)")
            print("=" * 60)

            model, model_without_ddp = train(
                model, criterion, postprocessors, detr_args, device
            )
            print("Training completed successfully!")
            # Default to the latest checkpoint
            model_path = os.path.join(detr_args.output_dir, "checkpoint.pth")
            # Prefer best checkpoint if early stopping was used
            best_ckpt = os.path.join(detr_args.output_dir, "checkpoint_best.pth")
            if os.path.exists(best_ckpt):
                model_path = best_ckpt

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

            # Load checkpoint if we didn't just train (or if a specific path was given)
            if not args.train or args.model_path:
                checkpoint = _torch_load(model_path, map_location="cpu")
                model_without_ddp = model.module if hasattr(model, "module") else model
                missing, unexpected = model_without_ddp.load_state_dict(
                    checkpoint["model"], strict=False
                )
                if missing:
                    print(f"Missing Keys: {missing}")
                if unexpected:
                    print(f"Unexpected Keys: {unexpected}")

            # Disable one-to-many branch for eval
            model_for_eval = model.module if hasattr(model, "module") else model
            model_for_eval.num_queries = model_for_eval.num_queries_one2one
            model_for_eval.transformer.two_stage_num_proposals = model_for_eval.num_queries

            all_results = {}
            for split_name, split_label in eval_splits:
                print(f"\n{'=' * 60}")
                print(f"PHASE: EVALUATE {split_label.upper()} ({split_name})")
                print("=" * 60)

                try:
                    results = evaluate_split(
                        model, criterion, postprocessors, detr_args,
                        split_name, device, detr_args.output_dir,
                        score_threshold=args.score_threshold,
                    )
                    if results:
                        all_results[split_name] = results
                except (FileNotFoundError, ValueError) as exc:
                    print(f"WARNING: skipping split '{split_name}': {exc}")
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
                summary_path = os.path.join(detr_args.output_dir, "eval_summary.json")
                with open(summary_path, "w") as f:
                    json.dump(all_results, f, indent=2)
                print(f"Results saved to: {summary_path}")

                table_results = {k: v for k, v in all_results.items() if k in ("val", "ood_test")}
                table_path = os.path.join(detr_args.output_dir, "eval_summary_table.png")
                generate_eval_summary_table(
                    table_results, args.config, table_path, model_type="Plain-DETR"
                )
            else:
                print("ERROR: No splits evaluated successfully — no summary written.")
                sys.exit(1)

    else:
        parser.print_help()


if __name__ == "__main__":
    main()
