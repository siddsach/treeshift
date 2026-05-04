"""Windowed backbone wrapper."""

# ------------------------------------------------------------------------
# Portions of this file are adapted from DINOv3:
# https://github.com/facebookresearch/dinov3
# Copyright (c) Meta Platforms, Inc. and affiliates.
# Licensed under the DINOv3 License (see THIRD_PARTY_LICENSES/DINOV3_LICENSE.txt).
# If you use this functionality in research, please cite:
# Siméoni et al., "DINOv3", arXiv:2508.10104 (2025).
# ------------------------------------------------------------------------

import math
from itertools import accumulate

import torch
import torch.nn.functional as F

from util.misc import NestedTensor


class WindowsWrapper(torch.nn.Module):
    def __init__(self, backbone, n_windows_w, n_windows_h, patch_size):
        super().__init__()
        self._backbone = backbone
        self._n_windows_w = n_windows_w
        self._n_windows_h = n_windows_h
        self._patch_size = patch_size
        self.strides = backbone.strides
        self.num_channels = [el * 2 for el in backbone.num_channels]

    def forward(self, tensor_list: NestedTensor):
        tensors = tensor_list.tensors
        mask_in = tensor_list.mask
        original_h, original_w = tensors.shape[2], tensors.shape[3]

        window_h = math.ceil((original_h // self._n_windows_h) / self._patch_size) * self._patch_size
        window_w = math.ceil((original_w // self._n_windows_w) / self._patch_size) * self._patch_size

        all_h = [window_h] * (self._n_windows_h - 1) + [original_h - window_h * (self._n_windows_h - 1)]
        all_w = [window_w] * (self._n_windows_w - 1) + [original_w - window_w * (self._n_windows_w - 1)]
        all_h_cumsum = [0] + list(accumulate(all_h))
        all_w_cumsum = [0] + list(accumulate(all_w))

        window_patch_features = [[None for _ in range(self._n_windows_w)] for _ in range(self._n_windows_h)]

        for ih in range(self._n_windows_h):
            for iw in range(self._n_windows_w):
                top, left = all_h_cumsum[ih], all_w_cumsum[iw]
                height, width = all_h[ih], all_w[iw]
                window_tensor = tensors[:, :, top : top + height, left : left + width]
                window_mask = mask_in[:, top : top + height, left : left + width]
                window_patch_features[ih][iw] = self._backbone(NestedTensor(tensors=window_tensor, mask=window_mask))[0]

        window_tensors = torch.cat(
            [torch.cat([el.tensors for el in window_patch_features[ih]], dim=-1) for ih in range(len(window_patch_features))],
            dim=-2,
        )

        resized_global_tensor = F.interpolate(tensors, size=(window_h, window_w), mode="bilinear", align_corners=False)
        global_features = self._backbone(NestedTensor(tensors=resized_global_tensor, mask=mask_in))
        global_tensors = F.interpolate(
            global_features[0].tensors, size=window_tensors.shape[-2:], mode="bilinear", align_corners=False
        )

        concat_tensors = torch.cat([global_tensors, window_tensors], dim=1)
        global_mask = F.interpolate(mask_in[None].float(), size=concat_tensors.shape[-2:]).to(torch.bool)[0]
        return [NestedTensor(tensors=concat_tensors, mask=global_mask)]
