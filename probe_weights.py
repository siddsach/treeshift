#!/usr/bin/env python3
"""
Inspect DINOv3 checkpoint files and determine the correct --dinov3_model flag.

Usage (run inside the apptainer container or venv that has PyTorch):

    python probe_weights.py /path/to/dino_weights_7b16.pth \
                            /path/to/dino_weights_sat.pth   \
                            /path/to/dinov3_vits.pth          # sanity-check

It will print, for each file:
  - Top-level checkpoint keys
  - Which sub-key was used as the state dict
  - Inferred embed_dim and n_blocks
  - The recommended --dinov3_model value
  - Available model names in dinov3.hub.backbones (if the repo is on-path)
"""

import sys
import os
import argparse
import torch

REPO_ROOT = "/scratch/groups/dlobell/aadityan/tree-distribution-shift"
DINOV3_REPO = "/opt/dinov3"


# ---------------------------------------------------------------------------
# Helpers (mirrors backbone.py logic)
# ---------------------------------------------------------------------------

def _unwrap_state_dict(checkpoint):
    if not isinstance(checkpoint, dict):
        return checkpoint
    for k in ["model", "state_dict", "teacher", "student", "backbone"]:
        if k in checkpoint and isinstance(checkpoint[k], dict):
            return checkpoint[k], k
    return checkpoint, "(top-level)"


def _maybe_strip_prefix(state_dict, prefix):
    keys = list(state_dict.keys())
    if not keys:
        return state_dict
    if all(k.startswith(prefix) for k in keys[:min(len(keys), 100)]):
        return {k[len(prefix):]: v for k, v in state_dict.items()}, prefix
    return state_dict, None


# ---------------------------------------------------------------------------
# Architecture inference
# ---------------------------------------------------------------------------

_EMBED_DIM_TO_MODEL = {
    384:  "dinov3_vits16",
    768:  "dinov3_vitb16",
    1024: "dinov3_vitl16",
    1280: "dinov3_vith14",
    1536: "dinov3_vitg14",
}


def _infer_arch(state_dict):
    """Return (embed_dim, n_blocks, patch_size, suggested_model_name)."""
    embed_dim = None
    # norm.weight is the final LayerNorm of the ViT backbone
    for key in ["norm.weight", "blocks.0.norm1.weight", "patch_embed.proj.bias"]:
        if key in state_dict:
            embed_dim = state_dict[key].shape[0]
            break

    # count blocks
    n_blocks = 0
    for i in range(200):
        if f"blocks.{i}.norm1.weight" in state_dict:
            n_blocks = i + 1
        else:
            if n_blocks > 0:
                break

    # patch size from patch_embed.proj.weight  [out_ch, in_ch, H, W]
    patch_size = None
    pe = state_dict.get("patch_embed.proj.weight")
    if pe is not None and pe.ndim == 4:
        patch_size = pe.shape[2]

    suggestion = _EMBED_DIM_TO_MODEL.get(embed_dim)
    if suggestion and patch_size and patch_size != 16:
        # swap the patch size suffix if needed
        suggestion = suggestion[:-2] + str(patch_size)

    return embed_dim, n_blocks, patch_size, suggestion


# ---------------------------------------------------------------------------
# Available models in dinov3.hub.backbones
# ---------------------------------------------------------------------------

def _list_dinov3_models(dinov3_repo):
    if not os.path.exists(dinov3_repo):
        return None
    try:
        if dinov3_repo not in sys.path:
            sys.path.insert(0, dinov3_repo)
        import importlib
        mod = importlib.import_module("dinov3.hub.backbones")
        names = [n for n in dir(mod) if not n.startswith("_") and callable(getattr(mod, n))]
        return names
    except Exception as e:
        return f"(could not import dinov3.hub.backbones: {e})"


# ---------------------------------------------------------------------------
# Per-file probe
# ---------------------------------------------------------------------------

def probe(path, dinov3_repo=DINOV3_REPO):
    print(f"\n{'='*70}")
    print(f"FILE: {path}")
    print('='*70)

    if not os.path.exists(path):
        print("  ERROR: file not found")
        return

    checkpoint = torch.load(path, map_location="cpu")

    if not isinstance(checkpoint, dict):
        print(f"  Not a dict: {type(checkpoint)}")
        return

    top_keys = list(checkpoint.keys())
    print(f"  Top-level keys ({len(top_keys)}): {top_keys[:10]}"
          + (" ..." if len(top_keys) > 10 else ""))

    # unwrap
    state_dict, used_key = _unwrap_state_dict(checkpoint)
    print(f"  Using sub-key: '{used_key}'")

    # strip prefixes
    stripped_prefix = None
    for prefix in ["module.", "backbone.", "teacher.", "student.", "model."]:
        state_dict, p = _maybe_strip_prefix(state_dict, prefix)
        if p:
            stripped_prefix = p
            break
    if stripped_prefix:
        print(f"  Stripped prefix: '{stripped_prefix}'")

    n_params = len(state_dict)
    print(f"  State dict size: {n_params} tensors")

    # show some keys
    all_keys = sorted(state_dict.keys())
    print(f"  First 5 keys: {all_keys[:5]}")
    print(f"  Last  5 keys: {all_keys[-5:]}")

    # infer architecture
    embed_dim, n_blocks, patch_size, suggestion = _infer_arch(state_dict)
    print()
    print(f"  Inferred embed_dim : {embed_dim}")
    print(f"  Inferred n_blocks  : {n_blocks}")
    print(f"  Inferred patch_size: {patch_size}")
    print()
    if suggestion:
        print(f"  ✓ Recommended --dinov3_model : {suggestion}")
    else:
        print(f"  ✗ No standard model mapping for embed_dim={embed_dim}")
        print(f"    Check dinov3.hub.backbones for custom model names.")

    # check if suggestion exists in backbones module
    models = _list_dinov3_models(dinov3_repo)
    if isinstance(models, list):
        print(f"\n  Available models in dinov3.hub.backbones:")
        for m in sorted(models):
            marker = " ← MATCH" if m == suggestion else ""
            print(f"    {m}{marker}")
        if suggestion and suggestion not in models:
            print(f"\n  WARNING: '{suggestion}' not found in backbones module.")
            print(f"  Closest matches: {[m for m in models if 'vit' in m.lower()]}")
    elif models:
        print(f"  {models}")
    else:
        print(f"  (dinov3 repo not found at {dinov3_repo}; run on the cluster)")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(description="Probe DINOv3 checkpoint files.")
    parser.add_argument("weights", nargs="+", help="Paths to .pth weight files")
    parser.add_argument(
        "--dinov3-repo", default=DINOV3_REPO,
        help=f"Path to dinov3 repo (default: {DINOV3_REPO})",
    )
    args = parser.parse_args()

    for path in args.weights:
        probe(path, dinov3_repo=args.dinov3_repo)

    print()


if __name__ == "__main__":
    main()
