# ------------------------------------------------------------------------
# Plain-DETR
# Copyright (c) 2023 Xi'an Jiaotong University & Microsoft Research Asia.
# Licensed under The MIT License [see LICENSE for details]
# ------------------------------------------------------------------------
# Deformable DETR
# Copyright (c) 2020 SenseTime. All Rights Reserved.
# Licensed under the Apache License, Version 2.0 [see LICENSE for details]
# ------------------------------------------------------------------------
# Modified from DETR (https://github.com/facebookresearch/detr)
# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved
# ------------------------------------------------------------------------

"""
COCO dataset which returns image_id for evaluation.

Mostly copy-paste from https://github.com/pytorch/vision/blob/13b35ff/references/detection/coco_utils.py
"""
from pathlib import Path

import torch
import torch.utils.data
from pycocotools import mask as coco_mask

from .torchvision_datasets import CocoDetection as TvCocoDetection
from util.misc import get_local_rank, get_local_size
import datasets.transforms as T


class CocoDetection(TvCocoDetection):
    def __init__(
        self,
        img_folder,
        ann_file,
        transforms,
        return_masks,
        cache_mode=False,
        local_rank=0,
        local_size=1,
    ):
        super(CocoDetection, self).__init__(
            img_folder,
            ann_file,
            cache_mode=cache_mode,
            local_rank=local_rank,
            local_size=local_size,
        )
        self._transforms = transforms
        self.prepare = ConvertCocoPolysToMask(return_masks)

    def __getitem__(self, idx):
        img, target = super(CocoDetection, self).__getitem__(idx)
        image_id = self.ids[idx]
        target = {"image_id": image_id, "annotations": target}
        img, target = self.prepare(img, target)
        if self._transforms is not None:
            img, target = self._transforms(img, target)
        return img, target


def convert_coco_poly_to_mask(segmentations, height, width):
    masks = []
    for polygons in segmentations:
        rles = coco_mask.frPyObjects(polygons, height, width)
        mask = coco_mask.decode(rles)
        if len(mask.shape) < 3:
            mask = mask[..., None]
        mask = torch.as_tensor(mask, dtype=torch.uint8)
        mask = mask.any(dim=2)
        masks.append(mask)
    if masks:
        masks = torch.stack(masks, dim=0)
    else:
        masks = torch.zeros((0, height, width), dtype=torch.uint8)
    return masks


class ConvertCocoPolysToMask(object):
    def __init__(self, return_masks=False):
        self.return_masks = return_masks

    def __call__(self, image, target):
        w, h = image.size

        image_id = target["image_id"]
        image_id = torch.tensor([image_id])

        anno = target["annotations"]

        anno = [obj for obj in anno if "iscrowd" not in obj or obj["iscrowd"] == 0]

        boxes = [obj["bbox"] for obj in anno]
        # guard against no boxes via resizing
        boxes = torch.as_tensor(boxes, dtype=torch.float32).reshape(-1, 4)
        boxes[:, 2:] += boxes[:, :2]
        boxes[:, 0::2].clamp_(min=0, max=w)
        boxes[:, 1::2].clamp_(min=0, max=h)

        # COCO uses 1-indexed category_id; model expects 0-indexed labels.
        # Map to [0, num_classes-1] (e.g. category_id 1 -> 0 for single-class tree dataset).
        classes = [max(0, obj["category_id"] - 1) for obj in anno]
        classes = torch.tensor(classes, dtype=torch.int64)

        if self.return_masks:
            segmentations = [obj["segmentation"] for obj in anno]
            masks = convert_coco_poly_to_mask(segmentations, h, w)

        keypoints = None
        if anno and "keypoints" in anno[0]:
            keypoints = [obj["keypoints"] for obj in anno]
            keypoints = torch.as_tensor(keypoints, dtype=torch.float32)
            num_keypoints = keypoints.shape[0]
            if num_keypoints:
                keypoints = keypoints.view(num_keypoints, -1, 3)

        keep = (boxes[:, 3] > boxes[:, 1]) & (boxes[:, 2] > boxes[:, 0])
        boxes = boxes[keep]
        classes = classes[keep]
        if self.return_masks:
            masks = masks[keep]
        if keypoints is not None:
            keypoints = keypoints[keep]

        target = {}
        target["boxes"] = boxes
        target["labels"] = classes
        if self.return_masks:
            target["masks"] = masks
        target["image_id"] = image_id
        if keypoints is not None:
            target["keypoints"] = keypoints

        # for conversion to coco api
        area = torch.tensor([obj["area"] for obj in anno])
        iscrowd = torch.tensor(
            [obj["iscrowd"] if "iscrowd" in obj else 0 for obj in anno]
        )
        target["area"] = area[keep]
        target["iscrowd"] = iscrowd[keep]

        target["orig_size"] = torch.as_tensor([int(h), int(w)])
        target["size"] = torch.as_tensor([int(h), int(w)])

        return image, target


def make_coco_transforms(image_set, args):

    normalize = T.Compose(
        [T.ToTensor(), T.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225], args.reparam)]
    )

    scales = [480, 512, 544, 576, 608, 640, 672, 704, 736, 768, 800]

    if image_set == "train":
        return T.Compose(
            [
                T.RandomHorizontalFlip(),
                T.RandomSelect(
                    T.RandomResize(scales, max_size=1333),
                    T.Compose(
                        [
                            T.RandomResize([400, 500, 600]),
                            T.RandomSizeCrop(384, 600),
                            T.RandomResize(scales, max_size=1333),
                        ]
                    ),
                ),
                normalize,
            ]
        )

    if image_set == "val":
        return T.Compose([T.RandomResize([800], max_size=1333), normalize,])

    raise ValueError(f"unknown {image_set}")


def build(image_set, args):
    # If using HF COCO export layout, delegate to build_hf_coco
    if getattr(args, "hf_coco_dir", None) and getattr(args, "hf_coco_config", None):
        return build_hf_coco(image_set, args)

    root = Path(args.coco_path)
    assert root.exists(), f"provided COCO path {root} does not exist"
    mode = "instances"
    PATHS = {
        "train": (root / "train2017", root / "annotations" / f"{mode}_train2017.json"),
        "val": (root / "val2017", root / "annotations" / f"{mode}_val2017.json"),
    }

    img_folder, ann_file = PATHS[image_set]
    dataset = CocoDetection(
        img_folder,
        ann_file,
        transforms=make_coco_transforms(image_set, args),
        return_masks=args.masks,
        cache_mode=args.cache_mode,
        local_rank=get_local_rank(),
        local_size=get_local_size(),
    )
    return dataset


def build_hf_coco(image_set, args):
    """Build dataset from tree_shift COCO export layout.

    Expected layout (produced by ``tree-shift export``):
        <hf_coco_dir>/<hf_coco_config>/train/images/
        <hf_coco_dir>/<hf_coco_config>/train/annotations/instances_train.json
        <hf_coco_dir>/<hf_coco_config>/val/images/
        <hf_coco_dir>/<hf_coco_config>/val/annotations/instances_val.json
        <hf_coco_dir>/<hf_coco_config>/ood_test/images/
        <hf_coco_dir>/<hf_coco_config>/ood_test/annotations/instances_ood_test.json

    image_set should be one of: "train", "val", "ood_test", "ood_train".
    """
    base = Path(args.hf_coco_dir) / args.hf_coco_config

    PATHS = {
        "train": (
            base / "train" / "images",
            base / "train" / "annotations" / "instances_train.json",
        ),
        "val": (
            base / "val" / "images",
            base / "val" / "annotations" / "instances_val.json",
        ),
        "ood_test": (
            base / "ood_test" / "images",
            base / "ood_test" / "annotations" / "instances_ood_test.json",
        ),
        "ood_train": (
            base / "ood_train" / "images",
            base / "ood_train" / "annotations" / "instances_ood_train.json",
        ),
    }

    if image_set not in PATHS:
        raise ValueError(f"Unknown image_set '{image_set}', expected one of {list(PATHS.keys())}")
    img_folder, ann_file = PATHS[image_set]
    if not img_folder.exists():
        raise FileNotFoundError(f"Image folder not found: {img_folder}")
    if not ann_file.exists():
        raise FileNotFoundError(f"Annotation file not found: {ann_file}")

    transform_key = "train" if image_set == "train" else "val"
    dataset = CocoDetection(
        img_folder,
        ann_file,
        transforms=make_coco_transforms(transform_key, args),
        return_masks=args.masks,
        cache_mode=args.cache_mode,
        local_rank=get_local_rank(),
        local_size=get_local_size(),
    )
    return dataset
