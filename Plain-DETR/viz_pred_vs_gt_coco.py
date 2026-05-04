import argparse
import os
from pathlib import Path

import json
import torch
from PIL import Image, ImageDraw, ImageFont

import util.misc as utils
from datasets import build_dataset, get_coco_api_from_dataset
from engine import evaluate
from models import build_model
from main import get_args_parser

def _xywh_to_xyxy(box):
    x, y, w, h = box
    return [x, y, x + w, y + h]

def _save_ap_table_image(metrics, out_path: Path):
    rows = [(str(k), f"{float(v):.4f}") for k, v in metrics.items()]
    width = 520
    pad = 16
    row_h = 28
    header_h = 40
    height = header_h + row_h * len(rows) + pad

    img = Image.new("RGB", (width, height), (255, 255, 255))
    draw = ImageDraw.Draw(img)
    font = ImageFont.load_default()

    draw.text((pad, 12), "COCO bbox metrics", fill=(0, 0, 0), font=font)
    y = header_h
    for k, v in rows:
        draw.text((pad, y), k, fill=(0, 0, 0), font=font)
        draw.text((width // 2, y), v, fill=(0, 0, 0), font=font)
        y += row_h

    out_path.parent.mkdir(parents=True, exist_ok=True)
    img.save(out_path)

@torch.no_grad()
def main_viz(args):
    device = torch.device(args.device)
    model, criterion, postprocessors = build_model(args)
    model.to(device)
    model.eval()
    criterion.eval()

    assert args.resume is not None and os.path.exists(args.resume), f"Checkpoint not found: {args.resume}"
    checkpoint = torch.load(args.resume, map_location="cpu")
    model.load_state_dict(checkpoint["model"], strict=False)

    dataset_val = build_dataset(image_set="val", args=args)
    coco = get_coco_api_from_dataset(dataset_val)

    sampler = torch.utils.data.SequentialSampler(dataset_val)
    data_loader = torch.utils.data.DataLoader(
        dataset_val,
        batch_size=1,
        sampler=sampler,
        drop_last=False,
        collate_fn=utils.collate_fn,
        num_workers=args.num_workers,
        pin_memory=True,
    )

    viz_dir = Path(args.viz_output_dir)
    viz_dir.mkdir(parents=True, exist_ok=True)

    eval_dir = None
    if getattr(args, "do_eval", False):
        eval_dir = Path(args.eval_output_dir) if args.eval_output_dir else (viz_dir.parent / "eval")
        eval_dir.mkdir(parents=True, exist_ok=True)

    coco_root = Path(args.coco_path)
    img_root = coco_root / "val2017"

    saved = 0
    for samples, targets in data_loader:
        samples = samples.to(device)
        targets = [{k: v.to(device) for k, v in t.items()} for t in targets]

        outputs = model(samples)

        orig_target_sizes = torch.stack([t["orig_size"] for t in targets], dim=0)
        results = postprocessors["bbox"](outputs, orig_target_sizes)[0]

        image_id = int(targets[0]["image_id"].item())
        img_info = coco.loadImgs([image_id])[0]
        file_name = img_info["file_name"]
        img_path = img_root / file_name
        img = Image.open(img_path).convert("RGB")
        draw = ImageDraw.Draw(img)

        # GT from COCO annotations (absolute xywh)
        ann_ids = coco.getAnnIds(imgIds=[image_id])
        anns = coco.loadAnns(ann_ids)
        for ann in anns:
            gt_xyxy = _xywh_to_xyxy(ann["bbox"])
            draw.rectangle(gt_xyxy, outline=(0, 255, 0), width=3)

        # Predictions (absolute xyxy in original image coords)
        pred_boxes = results["boxes"].detach().cpu()
        pred_scores = results["scores"].detach().cpu()
        pred_labels = results["labels"].detach().cpu()

        for box, score, lab in zip(pred_boxes, pred_scores, pred_labels):
            if float(score) < float(args.score_thresh):
                continue
            x1, y1, x2, y2 = [float(v) for v in box.tolist()]
            draw.rectangle([x1, y1, x2, y2], outline=(255, 0, 0), width=3)
            draw.text((x1, max(0.0, y1 - 10.0)), f"{int(lab)}:{float(score):.3f}", fill=(255, 0, 0))

        out_path = viz_dir / f"{image_id:012d}.jpg"
        img.save(out_path)
        saved += 1
        if saved >= int(args.num_images):
            break

    print(f"Saved {saved} visualizations to: {viz_dir}")

    if eval_dir is not None:
        stats, _ = evaluate(
            model=model,
            criterion=criterion,
            postprocessors=postprocessors,
            data_loader=data_loader,
            base_ds=coco,
            device=device,
            output_dir=str(eval_dir),
            step=0,
            use_wandb=False,
            reparam=getattr(args, "reparam", False),
            max_batches=int(getattr(args, "eval_max_batches", 0) or 0),
        )
        if "coco_eval_bbox" in stats:
            s = stats["coco_eval_bbox"]
            ap_metrics = {
                "AP": s[0],
                "AP50": s[1],
                "AP75": s[2],
                "APs": s[3],
                "APm": s[4],
                "APl": s[5],
            }
            with open(eval_dir / "viz_eval_metrics.json", "w") as f:
                json.dump(ap_metrics, f, indent=2)
            _save_ap_table_image(ap_metrics, eval_dir / "ap_table.png")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        "Plain-DETR inference + GT/pred visualization",
        parents=[get_args_parser()],
    )
    parser.add_argument("--viz_output_dir", type=str, required=True)
    parser.add_argument("--do_eval", action="store_true", default=False)
    parser.add_argument("--eval_output_dir", type=str, default=None)
    parser.add_argument("--num_images", type=int, default=20)
    parser.add_argument("--score_thresh", type=float, default=0.3)
    args = parser.parse_args()
    main_viz(args)