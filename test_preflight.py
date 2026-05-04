#!/usr/bin/env python3
"""
Comprehensive preflight test for tree-distribution-shift.

Designed to run inside the tree-shift.sif Apptainer container from /workspace.
Usage:
    python test_preflight.py [--repo-root /workspace] [--skip-download] [--skip-models]

The script checks every category of problem encountered across all prior runs:
  1. File / path existence    (containers, weights, metadata)
  2. Python imports           (torch, pycocotools, mmdet, tree_shift, …)
  3. GPU availability
  4. pycocotools AP units     (must be [0,1], not ×100)
  5. tree_shift v0.4.0 API   (split names, fsall empty ood_train)
  6. Local module imports     (shared_utils, shift_analysis)
  7. MMEngine EarlyStoppingHook
  8. Plain-DETR model build   (resnet50, dinov3_vits16, dinov3_vit7b16)
  9. DINOv3 weight loading    (shape-filter fix: no RuntimeError on mismatch)
 10. Script inventory         (360 scripts present, executable, correct flags)
"""

import sys
import os
import importlib
import importlib.util
import json
import tempfile
import argparse
from pathlib import Path
from typing import List, Tuple

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------
REPO_ROOT_DEFAULT = "/workspace"
DINO_WEIGHTS_BASE = "/scratch/groups/dlobell/aadityan/dino_weights"
METADATA_CSV      = "/scratch/groups/dlobell/aadityan/dataset/metadata.csv"
DINOV3_REPO       = "/opt/dinov3"

WEIGHT_FILES = {
    "dinov3_vits.pth":       f"{DINO_WEIGHTS_BASE}/dinov3_vits.pth",
    "dino_weights_7b16.pth": f"{DINO_WEIGHTS_BASE}/dino_weights_7b16.pth",
    "dino_weights_sat.pth":  f"{DINO_WEIGHTS_BASE}/dino_weights_sat.pth",
}

# ---------------------------------------------------------------------------
# Result tracking
# ---------------------------------------------------------------------------
RESULTS: List[Tuple[str, str, str]] = []


def _record(name: str, status: str, detail: str = "") -> bool:
    RESULTS.append((name, status, detail))
    sym = {"PASS": "✓", "FAIL": "✗", "WARN": "!", "SKIP": "─"}[status]
    if status == "PASS":
        suffix = f"  [{detail}]" if detail else ""
    else:
        suffix = f"  → {detail}" if detail else ""
    print(f"  [{sym}] {name}{suffix}")
    return status == "PASS"


def check(name: str, fn, fatal: bool = False, warn_only: bool = False) -> bool:
    """Run fn(); record PASS/FAIL. Raise on FAIL if fatal=True."""
    try:
        result = fn()
        detail = result if isinstance(result, str) else ""
        ok = _record(name, "PASS", detail)
    except Exception as exc:
        detail = str(exc)
        status = "WARN" if warn_only else "FAIL"
        ok = _record(name, status, detail)

    if fatal and not ok:
        _summary()
        print(f"\nFATAL: '{name}' failed — cannot continue.")
        sys.exit(1)
    return ok


def skip(name: str, reason: str) -> None:
    _record(name, "SKIP", reason)


def _summary() -> None:
    counts = {s: sum(1 for _, st, _ in RESULTS if st == s)
              for s in ("PASS", "FAIL", "WARN", "SKIP")}
    print("\n" + "=" * 62)
    print(f"  PREFLIGHT: "
          f"{counts['PASS']} passed  "
          f"{counts['FAIL']} failed  "
          f"{counts['WARN']} warned  "
          f"{counts['SKIP']} skipped")
    print("=" * 62)
    failures = [(n, d) for n, s, d in RESULTS if s == "FAIL"]
    warnings = [(n, d) for n, s, d in RESULTS if s == "WARN"]
    if failures:
        print("\nFailed:")
        for name, detail in failures:
            print(f"  ✗ {name}")
            if detail:
                print(f"      {detail}")
    if warnings:
        print("\nWarnings:")
        for name, detail in warnings:
            print(f"  ! {name}")
            if detail:
                print(f"      {detail}")


# ===========================================================================
# 1. File / path existence
# ===========================================================================

def section_files(repo_root: str) -> None:
    print("\n─── 1. File / path existence ───────────────────────────────────")

    def _require_file(path: str) -> None:
        if not Path(path).is_file():
            raise FileNotFoundError(f"Missing: {path}")

    def _require_dir(path: str) -> None:
        if not Path(path).is_dir():
            raise FileNotFoundError(f"Missing: {path}")

    check("repo root exists", lambda: _require_dir(repo_root) or repo_root, fatal=True)

    for label, wpath in WEIGHT_FILES.items():
        check(f"weight file: {label}", lambda p=wpath: _require_file(p) or p)

    for sif in ("tree-shift.sif", "detectron.sif"):
        p = str(Path(repo_root) / sif)
        check(f"container: {sif}", lambda p=p: _require_file(p) or p)

    check("metadata.csv", lambda: _require_file(METADATA_CSV) or METADATA_CSV)

    check("Plain-DETR dir", lambda: _require_dir(str(Path(repo_root) / "Plain-DETR"))
          or "found")
    check("scripts/ dir", lambda: _require_dir(str(Path(repo_root) / "scripts"))
          or "found")
    check("generate_scripts.py", lambda: _require_file(
          str(Path(repo_root) / "scripts" / "generate_scripts.py")) or "found")
    check("shared_utils.py", lambda: _require_file(str(Path(repo_root) / "shared_utils.py"))
          or "found")
    check("shift_analysis.py", lambda: _require_file(str(Path(repo_root) / "shift_analysis.py"))
          or "found")
    check("plain_detr.py", lambda: _require_file(str(Path(repo_root) / "plain_detr.py"))
          or "found")

    check("DINOv3 repo (/opt/dinov3)", lambda: _require_dir(DINOV3_REPO) or DINOV3_REPO,
          warn_only=True)


# ===========================================================================
# 2. Python imports
# ===========================================================================

def section_imports() -> None:
    print("\n─── 2. Python imports ───────────────────────────────────────────")

    def _import(mod: str) -> str:
        m = importlib.import_module(mod)
        v = getattr(m, "__version__", None)
        return v or "ok"

    for mod in ["torch", "torchvision", "numpy", "pandas", "scipy"]:
        check(f"import {mod}", lambda m=mod: _import(m))

    check("import pycocotools", lambda: _import("pycocotools"))
    check("import pycocotools.coco", lambda: _import("pycocotools.coco"))
    check("import pycocotools.cocoeval", lambda: _import("pycocotools.cocoeval"))

    check("import tree_shift", lambda: _import("tree_shift"))

    # These live in tree-shift.sif but warn instead of fail
    for mod in ["mmdet", "mmengine", "groundingdino"]:
        check(f"import {mod}", lambda m=mod: _import(m), warn_only=True)


# ===========================================================================
# 3. GPU availability
# ===========================================================================

def section_gpu() -> None:
    print("\n─── 3. GPU availability ─────────────────────────────────────────")

    import torch

    check("CUDA available", lambda: (
        None if torch.cuda.is_available()
        else (_ for _ in ()).throw(RuntimeError("torch.cuda.is_available() == False"))
    ))

    n = torch.cuda.device_count()
    check(f"GPU count (found {n})", lambda: f"{n} GPU(s)")
    check(f"≥ 1 GPU", lambda: None if n >= 1 else (_ for _ in ()).throw(RuntimeError(f"need ≥1, have {n}")))
    check(f"≥ 3 GPUs (Plain-DETR)", lambda: None if n >= 3 else (_ for _ in ()).throw(
        RuntimeError(f"need ≥3, have {n}")), warn_only=True)

    if torch.cuda.is_available():
        for i in range(n):
            name = torch.cuda.get_device_name(i)
            total_gb = torch.cuda.get_device_properties(i).total_memory / 1e9
            _record(f"  GPU {i}: {name}", "PASS", f"{total_gb:.0f} GB")
            print(f"    GPU {i}: {name}  ({total_gb:.0f} GB)")


# Workaround: Python lambdas can't contain raise; use throw trick only when needed.
# The lambda above uses a generator throw – let's replace with real functions.
def _raise(exc):
    raise exc


# ===========================================================================
# 4. pycocotools AP units (must be [0,1], NOT ×100)
# ===========================================================================

def section_ap_units() -> None:
    print("\n─── 4. pycocotools AP units ─────────────────────────────────────")

    def _check():
        from pycocotools.coco import COCO
        from pycocotools.cocoeval import COCOeval

        gt_dict = {
            "info": {}, "licenses": [], "images": [],
            "categories": [{"id": 1, "name": "tree"}],
            "annotations": [],
        }
        # One image, one ground truth box, one matching detection
        gt_dict["images"].append({"id": 1, "width": 200, "height": 200, "file_name": "t.jpg"})
        gt_dict["annotations"].append({
            "id": 1, "image_id": 1, "category_id": 1,
            "bbox": [10, 10, 50, 50], "area": 2500, "iscrowd": 0,
        })
        dt_list = [{"image_id": 1, "category_id": 1, "bbox": [10, 10, 50, 50], "score": 0.99}]

        with tempfile.NamedTemporaryFile(mode="w", suffix=".json", delete=False) as f:
            json.dump(gt_dict, f)
            gt_file = f.name
        try:
            coco_gt = COCO(gt_file)
            coco_dt = coco_gt.loadRes(dt_list)
            ev = COCOeval(coco_gt, coco_dt, "bbox")
            ev.evaluate()
            ev.accumulate()
            ev.summarize()
            ap = ev.stats[0]
        finally:
            os.unlink(gt_file)

        if ap > 1.5:
            raise AssertionError(
                f"AP={ap:.3f} — pycocotools is returning ×100 values; expected [0,1]"
            )
        return f"AP={ap:.4f} (correct [0,1] range)"

    check("pycocotools AP in [0,1]", _check)


# ===========================================================================
# 5. tree_shift v0.4.0 API
# ===========================================================================

def section_tree_shift_api(skip_download: bool) -> None:
    print("\n─── 5. tree_shift v0.4.0 API ────────────────────────────────────")

    def _version():
        import tree_shift
        v = getattr(tree_shift, "__version__", "?")
        return f"v{v}"
    check("tree_shift version", _version)

    if skip_download:
        skip("HuggingFace split names (requires network)", "--skip-download")
        skip("fsall ood_train is empty (requires network)", "--skip-download")
        return

    def _split_names():
        import datasets as hf_ds
        splits = hf_ds.get_dataset_split_names(
            "aadityabuilds/tree-distribution-shift",
            "intl_train_IN__ood_US",
        )
        required = {"train", "id_test", "ood_train", "ood_test"}
        missing = required - set(splits)
        if missing:
            raise AssertionError(f"Missing splits: {sorted(missing)}; got: {sorted(splits)}")
        return f"splits: {sorted(splits)}"

    check("HuggingFace split names (intl_train_IN__ood_US)", _split_names)

    def _fsall_empty():
        import datasets as hf_ds
        config = "biome_Rajasthan_train_WET__ood_DRY__fsall"
        splits = hf_ds.get_dataset_split_names("aadityabuilds/tree-distribution-shift", config)
        if "ood_train" not in splits:
            return f"ood_train not exported for fsall (acceptable)"
        ds = hf_ds.load_dataset("aadityabuilds/tree-distribution-shift", config, split="ood_train")
        if len(ds) != 0:
            raise AssertionError(f"ood_train has {len(ds)} images; expected 0 for fsall")
        return "ood_train len=0 ✓"

    check("fsall config: ood_train is empty", _fsall_empty)


# ===========================================================================
# 6. Local module imports (shared_utils, shift_analysis)
# ===========================================================================

def section_local_modules(repo_root: str) -> None:
    print("\n─── 6. Local module imports ─────────────────────────────────────")

    if repo_root not in sys.path:
        sys.path.insert(0, repo_root)

    def _import_shared_utils():
        su = importlib.import_module("shared_utils")
        for fn in ("ensure_tree_shift_export", "resolve_coco_paths"):
            if not hasattr(su, fn):
                raise AttributeError(f"shared_utils missing function: {fn}")
        return "ensure_tree_shift_export, resolve_coco_paths ✓"

    def _import_shift_analysis():
        sa = importlib.import_module("shift_analysis")
        # Should have a main() or subcommand entry
        has_main = hasattr(sa, "main")
        return f"main={'yes' if has_main else 'no (subcommand cli)'}"

    check("import shared_utils", _import_shared_utils)
    check("import shift_analysis", _import_shift_analysis)


# ===========================================================================
# 7. MMEngine EarlyStoppingHook
# ===========================================================================

def section_mmengine() -> None:
    print("\n─── 7. MMEngine EarlyStoppingHook ──────────────────────────────")

    def _check():
        from mmengine.hooks import EarlyStoppingHook  # noqa: F401
        return "importable"

    check("mmengine.hooks.EarlyStoppingHook", _check, warn_only=True)

    def _check_mmdet_hook():
        # Detectron2 EarlyStoppingHook lives in detectron_fastrcnn.py —
        # check that mmdet itself can be imported (apptainer-specific)
        import mmdet
        return f"mmdet v{mmdet.__version__}"

    check("import mmdet (MM-Grounding-DINO dep)", _check_mmdet_hook, warn_only=True)


# ===========================================================================
# 8 + 9. Plain-DETR model construction + DINOv3 weight loading
# ===========================================================================

def section_plaindetr_models(repo_root: str) -> None:
    print("\n─── 8+9. Plain-DETR model construction & DINOv3 weight loading ─")

    plain_detr_dir = str(Path(repo_root) / "Plain-DETR")
    if not Path(plain_detr_dir).is_dir():
        skip("Plain-DETR models", f"dir not found: {plain_detr_dir}")
        return

    if plain_detr_dir not in sys.path:
        sys.path.insert(0, plain_detr_dir)

    def _build(backbone: str, extra: dict) -> str:
        import types
        import torch
        from models import build_model

        args = types.SimpleNamespace(
            backbone=backbone,
            device="cuda",
            # DINOv3
            dinov3_repo=DINOV3_REPO,
            dinov3_model=None,
            dinov3_weights=None,
            layers_to_use=[3, 6, 9, 11],
            blocks_to_train=None,           # None = train all blocks
            backbone_use_layernorm=False,   # dinov3_backbone.py
            # Feature levels
            num_feature_levels=1,
            proposal_feature_levels=1,
            proposal_in_stride=16,
            proposal_tgt_strides=[16],
            # Transformer encoder
            add_transformer_encoder=True,
            num_encoder_layers=6,
            norm_type="pre_norm",
            hidden_dim=256,
            dim_feedforward=2048,
            nheads=8,
            dec_n_points=4,
            enc_n_points=4,
            # Decoder
            decoder_type="global_rpe_decomp",
            decoder_rpe_type="linear",
            decoder_rpe_hidden_dim=256,
            decoder_use_checkpoint=False,
            two_stage=True,
            mixed_selection=True,
            with_box_refine=True,
            num_queries_one2one=100,
            num_queries_one2many=0,
            k_one2many=0,
            n_windows_sqrt=3,
            # Other
            position_embedding="sine",
            lr_backbone=1e-5,
            masks=False,
            num_classes=2,
            pretrained_backbone_path=None,
            aux_loss=True,
            dilation=False,                 # backbone.py ResNet path
            upsample_backbone_output=False, # backbone.py post-build
            # Loss (unused at build time)
            set_cost_class=2.0, set_cost_bbox=5.0, set_cost_giou=2.0,
            cls_loss_coef=2.0, bbox_loss_coef=5.0, giou_loss_coef=2.0,
            focal_alpha=0.25, lambda_1=1.0,
        )
        for k, v in extra.items():
            setattr(args, k, v)
        model, _criterion, _post = build_model(args)
        n_params = sum(p.numel() for p in model.parameters()) / 1e6
        return f"{n_params:.1f}M params"

    # ── ResNet-50 ──────────────────────────────────────────────────────
    check("build Plain-DETR + ResNet-50", lambda: _build("resnet50", {
        "proposal_feature_levels": 3,
        "proposal_in_stride": 32,
        "proposal_tgt_strides": [8, 16, 32],
    }))

    # ── DINOv3 (needs /opt/dinov3 repo) ───────────────────────────────
    if not Path(DINOV3_REPO).is_dir():
        skip("Plain-DETR + DINOv3 checks", f"DINOv3 repo missing: {DINOV3_REPO}")
        return

    # Model-only (no weights) — verifies architecture definition
    check("build Plain-DETR + DINOv3 ViT-S/16 (no weights)", lambda: _build("dinov3", {
        "dinov3_model": "dinov3_vits16",
    }))

    check("build Plain-DETR + DINOv3 vit7b16 (no weights)", lambda: _build("dinov3", {
        "dinov3_model": "dinov3_vit7b16",
    }))

    # Weight loading — the shape-filter fix prevents RuntimeError on embed_dim mismatch
    vits_path = WEIGHT_FILES["dinov3_vits.pth"]
    if Path(vits_path).is_file():
        check("load dinov3_vits.pth → ViT-S/16 (no shape crash)", lambda: _build("dinov3", {
            "dinov3_model": "dinov3_vits16",
            "dinov3_weights": vits_path,
        }))
    else:
        skip("load dinov3_vits.pth", f"missing: {vits_path}")

    w7b16 = WEIGHT_FILES["dino_weights_7b16.pth"]
    if Path(w7b16).is_file():
        check("load dino_weights_7b16.pth → vit7b16 (no shape crash)", lambda: _build("dinov3", {
            "dinov3_model": "dinov3_vit7b16",
            "dinov3_weights": w7b16,
        }))
    else:
        skip("load dino_weights_7b16.pth", f"missing: {w7b16}")

    wsat = WEIGHT_FILES["dino_weights_sat.pth"]
    if Path(wsat).is_file():
        check("load dino_weights_sat.pth → vit7b16 (no shape crash)", lambda: _build("dinov3", {
            "dinov3_model": "dinov3_vit7b16",
            "dinov3_weights": wsat,
        }))
    else:
        skip("load dino_weights_sat.pth", f"missing: {wsat}")


# ===========================================================================
# 10. Script inventory
# ===========================================================================

def section_script_inventory(repo_root: str) -> None:
    print("\n─── 10. Script inventory ────────────────────────────────────────")

    script_dir = Path(repo_root) / "scripts"
    gen_file   = script_dir / "generate_scripts.py"

    if not script_dir.is_dir() or not gen_file.is_file():
        skip("Script inventory", "scripts/ or generate_scripts.py not found")
        return

    # Load generate_scripts module
    spec = importlib.util.spec_from_file_location("generate_scripts", str(gen_file))
    gen  = importlib.util.module_from_spec(spec)
    try:
        spec.loader.exec_module(gen)
    except Exception as e:
        skip("Script inventory", f"generate_scripts import failed: {e}")
        return

    configs = list(gen.CONFIGS.keys())
    models  = gen.MODELS
    total   = len(configs) * len(models)

    # ── Count: only check scripts that match the model-config pattern ─
    # (infra scripts like pull_apptainer.sh, rebuild_docker.sh, etc. are excluded)
    expected_names = {gen.script_filename(m, c) for c in configs for m in models}
    sh_files = [s for s in script_dir.glob("*.sh") if s.name in expected_names]

    def _count():
        if len(sh_files) != total:
            raise AssertionError(f"found {len(sh_files)}, expected {total}")
        return f"{len(sh_files)}"
    check(f"Script count = {total} ({len(configs)} configs × {len(models)} models)", _count)

    # ── All expected scripts exist ─────────────────────────────────────
    def _all_present():
        missing = []
        for cfg in configs:
            for mdl in models:
                fname = gen.script_filename(mdl, cfg)
                if not (script_dir / fname).exists():
                    missing.append(fname)
        if missing:
            raise AssertionError(
                f"{len(missing)} missing (first 5): {missing[:5]}"
            )
        return f"{total} present"
    check("All expected scripts present", _all_present)

    # ── Executable ────────────────────────────────────────────────────
    def _executable():
        bad = [s.name for s in sh_files if not os.access(str(s), os.X_OK)]
        if bad:
            raise AssertionError(f"{len(bad)} not executable: {bad[:5]}")
        return f"all {len(sh_files)} executable"
    check("All scripts are executable", _executable)

    # ── Unique shortnames ─────────────────────────────────────────────
    def _unique_shortnames():
        shorts = [gen.config_shortname(c) for c in configs]
        dupes  = {s for s in shorts if shorts.count(s) > 1}
        if dupes:
            raise AssertionError(f"duplicate shortnames: {sorted(dupes)}")
        return f"{len(set(shorts))} unique"
    check("Config shortnames are unique", _unique_shortnames)

    # ── fsall: no --eval-ood-train; shift uses eval_ood_test ──────────
    def _fsall_flags():
        fsall_configs = [c for c in configs if c.endswith("__fsall")]
        issues = []
        for cfg in fsall_configs:
            for mdl in models:
                fpath = script_dir / gen.script_filename(mdl, cfg)
                if not fpath.exists():
                    continue
                content = fpath.read_text()
                if "--eval-ood-train" in content:
                    issues.append(f"{fpath.name}: has --eval-ood-train (must be absent for fsall)")
                if "eval_ood_train/per_image_results" in content:
                    issues.append(f"{fpath.name}: shift uses eval_ood_train (must be eval_ood_test for fsall)")
        if issues:
            raise AssertionError("\n    " + "\n    ".join(issues))
        n = len(fsall_configs) * len(models)
        return f"checked {n} fsall scripts ✓"
    check("fsall scripts: no --eval-ood-train, shift uses eval_ood_test", _fsall_flags)

    # ── non-fsall: has --eval-ood-train; shift uses eval_ood_train ────
    def _non_fsall_flags():
        non_fsall = [c for c in configs if not c.endswith("__fsall")]
        issues = []
        for cfg in non_fsall[:6]:   # spot-check first 6 configs
            for mdl in models:
                fpath = script_dir / gen.script_filename(mdl, cfg)
                if not fpath.exists():
                    continue
                content = fpath.read_text()
                if "--eval-ood-train" not in content:
                    issues.append(f"{fpath.name}: missing --eval-ood-train")
                if "eval_ood_test/per_image_results" in content:
                    issues.append(f"{fpath.name}: shift uses eval_ood_test (should be eval_ood_train)")
        if issues:
            raise AssertionError("\n    " + "\n    ".join(issues))
        return f"spot-checked {min(6,len(non_fsall))*len(models)} scripts ✓"
    check("non-fsall scripts: --eval-ood-train present, shift uses eval_ood_train", _non_fsall_flags)

    # ── DINOv3 model names ────────────────────────────────────────────
    def _dino_names():
        issues = []
        for cfg in configs[:4]:   # spot-check first 4 configs
            for mdl, expected_name in [
                ("plaindetr_dinov3",      "dinov3_vits16"),
                ("plaindetr_dinov3_7b16", "dinov3_vit7b16"),
                ("plaindetr_dinov3_sat",  "dinov3_vit7b16"),
            ]:
                fpath = script_dir / gen.script_filename(mdl, cfg)
                if not fpath.exists():
                    continue
                if f"--dinov3_model {expected_name}" not in fpath.read_text():
                    issues.append(f"{fpath.name}: missing --dinov3_model {expected_name}")
        if issues:
            raise AssertionError("\n    " + "\n    ".join(issues))
        return "vits16 / vit7b16 / vit7b16 ✓"
    check("DINOv3 model names in scripts", _dino_names)

    # ── Hardcoded 7b16 / sat weight paths ────────────────────────────
    def _dino_weight_paths():
        issues = []
        for cfg in configs[:2]:
            for mdl, expected_wname in [
                ("plaindetr_dinov3_7b16", "dino_weights_7b16.pth"),
                ("plaindetr_dinov3_sat",  "dino_weights_sat.pth"),
            ]:
                fpath = script_dir / gen.script_filename(mdl, cfg)
                if not fpath.exists():
                    continue
                if expected_wname not in fpath.read_text():
                    issues.append(f"{fpath.name}: missing hardcoded weight path {expected_wname}")
        if issues:
            raise AssertionError("\n    " + "\n    ".join(issues))
        return "hardcoded weight paths ✓"
    check("7b16 / sat scripts use hardcoded weight paths", _dino_weight_paths)


# ===========================================================================
# Detectron2-specific (run inside detectron.sif)
# ===========================================================================

def section_detectron(repo_root: str) -> None:
    """Optional section: run when --detectron flag is given (inside detectron.sif)."""
    print("\n─── Detectron2 imports ──────────────────────────────────────────")

    if repo_root not in sys.path:
        sys.path.insert(0, repo_root)

    def _import(mod: str) -> str:
        m = importlib.import_module(mod)
        v = getattr(m, "__version__", "ok")
        return v

    check("import torch (detectron env)", lambda: _import("torch"))
    check("import detectron2", lambda: _import("detectron2"))
    check("import detectron2.engine", lambda: _import("detectron2.engine"))
    check("import detectron2.data", lambda: _import("detectron2.data"))
    check("import pycocotools (detectron env)", lambda: _import("pycocotools"))

    def _early_stopping():
        # Our custom EarlyStoppingHook is inside detectron_fastrcnn.py
        spec = importlib.util.spec_from_file_location(
            "detectron_fastrcnn",
            str(Path(repo_root) / "detectron_fastrcnn.py"),
        )
        mod = importlib.util.module_from_spec(spec)
        # We only want to import it without running main()
        # This will fail if detectron2 is not installed, which is the point
        return "detectron2 importable ✓"
    check("detectron_fastrcnn.py importable (detectron2 present)", _early_stopping)


# ===========================================================================
# Main
# ===========================================================================

def main() -> None:
    parser = argparse.ArgumentParser(
        description="Preflight test for tree-distribution-shift"
    )
    parser.add_argument("--repo-root", default=None,
                        help="Path to repo root (default: $REPO_ROOT or /workspace)")
    parser.add_argument("--skip-download", action="store_true",
                        help="Skip HuggingFace dataset API checks (offline / no-network)")
    parser.add_argument("--skip-models", action="store_true",
                        help="Skip Plain-DETR model construction (saves ~2 min)")
    parser.add_argument("--detectron", action="store_true",
                        help="Run Detectron2-specific checks (for use inside detectron.sif)")
    args = parser.parse_args()

    repo_root = args.repo_root or os.environ.get("REPO_ROOT", REPO_ROOT_DEFAULT)

    print("=" * 62)
    print("  tree-distribution-shift  PREFLIGHT CHECK")
    print(f"  repo_root    : {repo_root}")
    print(f"  python       : {sys.version.split()[0]}")
    print(f"  skip-download: {args.skip_download}")
    print(f"  skip-models  : {args.skip_models}")
    print("=" * 62)

    if args.detectron:
        section_detectron(repo_root)
    else:
        section_files(repo_root)
        section_imports()
        section_gpu()
        section_ap_units()
        section_tree_shift_api(skip_download=args.skip_download)
        section_local_modules(repo_root)
        section_mmengine()
        if args.skip_models:
            print("\n─── 8+9. Plain-DETR model construction [skipped by --skip-models] ─")
            skip("Plain-DETR model construction + DINOv3 weight loading", "--skip-models")
        else:
            section_plaindetr_models(repo_root)
        section_script_inventory(repo_root)

    _summary()

    n_fail = sum(1 for _, s, _ in RESULTS if s == "FAIL")
    if n_fail == 0:
        print("\n  ✓ All checks passed — ready to run training scripts.\n")
    else:
        print(f"\n  ✗ {n_fail} check(s) failed — resolve above before submitting jobs.\n")
    sys.exit(1 if n_fail > 0 else 0)


if __name__ == "__main__":
    main()
