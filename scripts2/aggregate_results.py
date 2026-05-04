#!/usr/bin/env python3
"""
Aggregate evaluation results from final_outputs/ into cross-model tables.

Produces two sets of outputs:

Part 1 — Base configs (no few-shot):
  For each of the 4 distribution shifts, a table with all 5 models showing
  ID test + OOD test metrics. Also a combined JSON.

Part 2 — Few-shot configs:
  fastrcnn_pretrained across all 4 shifts × (base, fs1, fs10, fs100, fsall)
  grounding_dino for region_North_South × (base, fs1, fs10, fs100, fsall)
  Tables show OOD test metrics.

Usage:
    python scripts2/aggregate_results.py --outputs-dir /path/to/final_outputs [--out-dir ./aggregate_results]
"""

import argparse
import json
import os
import re
import sys
from pathlib import Path

# ═══════════════════════════════════════════════════════════════════════════════
# Constants
# ═══════════════════════════════════════════════════════════════════════════════

SHIFTS = {
    "biome_Rajasthan_WET_DRY": {
        "label": "Biome: Rajasthan Wet → Dry",
        "config": "biome_Rajasthan_train_WET__ood_DRY",
    },
    "elev_Karnataka_HIGH_LOW": {
        "label": "Elevation: Karnataka High → Low",
        "config": "elev_Karnataka_train_HIGH__ood_LOW",
    },
    "intl_US_IN": {
        "label": "International: US → India",
        "config": "intl_train_US__ood_IN",
    },
    "region_North_South": {
        "label": "Region: North → South",
        "config": "region_train_North__ood_South",
    },
}

MODEL_DISPLAY = {
    "fastrcnn_pretrained": "Faster R-CNN (pretrained)",
    "maskrcnn_pretrained":  "Mask R-CNN (pretrained)",
    "plaindetr_resnet":     "Plain-DETR (ResNet-50)",
    "plaindetr_dinov3":     "Plain-DETR (DINOv3)",
    "grounding_dino":       "Grounding DINO",
}

BASE_MODELS = list(MODEL_DISPLAY.keys())

FEWSHOT_SUFFIXES = ["", "_fs1", "_fs10", "_fs100", "_fsall"]
FEWSHOT_LABELS   = ["0-shot (base)", "1-shot", "10-shot", "100-shot", "All OOD"]

# ── Directory parsing (same logic as run_all_evals.py) ────────────────────────

MODEL_PREFIXES = [
    ("fastrcnn_pretrained_",    "fastrcnn_pretrained"),
    ("maskrcnn_pretrained_",    "maskrcnn_pretrained"),
    ("fastrcnn_",               "fastrcnn"),
    ("maskrcnn_",               "maskrcnn"),
    ("grounding_dino_",         "grounding_dino"),
    ("plaindetr_dinov3_sat_",   "plaindetr_dinov3_sat"),
    ("plaindetr_dinov3_7b16_",  "plaindetr_dinov3_7b16"),
    ("plaindetr_dinov3_",       "plaindetr_dinov3"),
    ("plaindetr_resnet_",       "plaindetr_resnet"),
]

CONFIG_MAP = {
    "biome_Rajasthan_WET_DRY":  "biome_Rajasthan_train_WET__ood_DRY",
    "biome_Rajasthan_DRY_WET":  "biome_Rajasthan_train_DRY__ood_WET",
    "intl_US_IN":               "intl_train_US__ood_IN",
    "intl_IN_US":               "intl_train_IN__ood_US",
    "region_North_South":       "region_train_North__ood_South",
    "region_South_North":       "region_train_South__ood_North",
    "elev_Karnataka_HIGH_LOW":  "elev_Karnataka_train_HIGH__ood_LOW",
    "elev_Karnataka_LOW_HIGH":  "elev_Karnataka_train_LOW__ood_HIGH",
}


def parse_output_dir(dirname):
    """Return (model_type, full_config) or None."""
    name = dirname
    name = re.sub(r"_\d+$", "", name)
    name = name.replace("_1gpu40ep", "")

    model_type = shortname = None
    for prefix, mtype in MODEL_PREFIXES:
        if name.startswith(prefix):
            model_type = mtype
            shortname = name[len(prefix):]
            break
    if model_type is None:
        return None

    fewshot = ""
    for fs in ("_fsall", "_fs100", "_fs10", "_fs1"):
        if shortname.endswith(fs):
            fewshot = "__" + fs[1:]
            shortname = shortname[:-len(fs)]
            break

    base_config = CONFIG_MAP.get(shortname)
    if base_config is None:
        return None

    return model_type, base_config + fewshot


# ═══════════════════════════════════════════════════════════════════════════════
# Data loading
# ═══════════════════════════════════════════════════════════════════════════════

def load_eval_summary(output_dir):
    """Load eval_summary.json from a model output directory."""
    p = os.path.join(output_dir, "eval_summary.json")
    if not os.path.exists(p):
        return None
    with open(p) as f:
        return json.load(f)


def scan_outputs(outputs_dir):
    """Scan outputs directory and return indexed results.

    Returns:
        dict: (model_type, full_config) → eval_summary dict
    """
    results = {}
    for d in sorted(Path(outputs_dir).iterdir()):
        if not d.is_dir():
            continue
        parsed = parse_output_dir(d.name)
        if parsed is None:
            continue
        model_type, config = parsed
        summary = load_eval_summary(str(d))
        if summary is None:
            print(f"  [WARN] no eval_summary.json: {d.name}")
            continue
        results[(model_type, config)] = summary
    return results


# ═══════════════════════════════════════════════════════════════════════════════
# Metric extraction
# ═══════════════════════════════════════════════════════════════════════════════

def _fmt(v):
    if v is None or v == "N/A":
        return "—"
    try:
        return f"{float(v):.3f}"
    except (TypeError, ValueError):
        return str(v)


def extract_split_metrics(summary, split):
    """Extract key metrics from a split in eval_summary."""
    if summary is None or split not in summary:
        return None
    res = summary[split]
    bbox = res.get("bbox", {})
    tm = res.get("tree_metrics", {}).get("overall", {})
    return {
        "AP50":       bbox.get("AP50"),
        "AP":         bbox.get("AP"),
        "APs":        bbox.get("APs"),
        "APm":        bbox.get("APm"),
        "APl":        bbox.get("APl"),
        "Precision":  tm.get("Precision"),
        "Recall":     tm.get("Recall"),
        "Crown_RMSE": tm.get("Crown_Diameter_RMSE_px") or tm.get("Crown_Diameter_RMSE_m"),
        "Geoloc_RMSE": tm.get("Geolocation_RMSE_px") or tm.get("Geolocation_RMSE_m"),
    }


# ═══════════════════════════════════════════════════════════════════════════════
# Table generation
# ═══════════════════════════════════════════════════════════════════════════════

def _make_table_image(title, columns, rows, output_path, row_colors=None):
    """Render a matplotlib table to PNG."""
    import matplotlib
    try:
        matplotlib.use("Agg")
    except Exception:
        pass
    import matplotlib.pyplot as plt

    n_rows = len(rows)
    fig_width = max(14, len(columns) * 1.6)
    fig_height = max(2.0, 0.55 * (n_rows + 1) + 1.0)
    fig, ax = plt.subplots(figsize=(fig_width, fig_height))
    ax.axis("off")
    ax.set_title(title, fontsize=13, fontweight="bold", pad=14)

    table = ax.table(
        cellText=rows,
        colLabels=columns,
        cellLoc="center",
        loc="center",
    )
    table.auto_set_font_size(False)
    table.set_fontsize(8.5)
    table.scale(1.0, 1.55)

    for j in range(len(columns)):
        cell = table[0, j]
        cell.set_facecolor("#4472C4")
        cell.set_text_props(color="white", fontweight="bold")

    for i in range(1, n_rows + 1):
        colour = "#D9E2F3" if i % 2 == 1 else "white"
        if row_colors and i - 1 < len(row_colors) and row_colors[i - 1]:
            colour = row_colors[i - 1]
        for j in range(len(columns)):
            table[i, j].set_facecolor(colour)

    os.makedirs(os.path.dirname(output_path) or ".", exist_ok=True)
    fig.savefig(output_path, dpi=150, bbox_inches="tight", facecolor="white")
    plt.close(fig)
    print(f"  Table saved: {output_path}")


# ═══════════════════════════════════════════════════════════════════════════════
# Part 1: Base configs
# ═══════════════════════════════════════════════════════════════════════════════

def generate_base_tables(results, out_dir):
    """Generate per-shift tables with all 5 models, for ID + OOD test."""
    base_dir = os.path.join(out_dir, "base_configs")
    os.makedirs(base_dir, exist_ok=True)

    columns = [
        "Model",
        "Split",
        "AP@0.50",
        "AP@[.50:.95]",
        "APs",
        "APm",
        "APl",
        "Precision",
        "Recall",
        "Crown\nRMSE",
        "Geoloc\nRMSE",
    ]

    all_shift_data = {}

    for shift_key, shift_info in SHIFTS.items():
        config = shift_info["config"]
        label = shift_info["label"]
        rows = []
        shift_json = {}

        for model_key in BASE_MODELS:
            display_name = MODEL_DISPLAY[model_key]
            summary = results.get((model_key, config))

            for split, split_label in [("val", "ID Test"), ("ood_test", "OOD Test")]:
                m = extract_split_metrics(summary, split)
                if m is None:
                    rows.append([display_name, split_label] + ["—"] * 9)
                else:
                    rows.append([
                        display_name, split_label,
                        _fmt(m["AP50"]), _fmt(m["AP"]),
                        _fmt(m["APs"]), _fmt(m["APm"]), _fmt(m["APl"]),
                        _fmt(m["Precision"]), _fmt(m["Recall"]),
                        _fmt(m["Crown_RMSE"]), _fmt(m["Geoloc_RMSE"]),
                    ])
                    shift_json.setdefault(model_key, {})[split] = m

        all_shift_data[shift_key] = shift_json

        # Row colors: alternate by model (2 rows per model)
        row_colors = []
        for i, model in enumerate(BASE_MODELS):
            c = "#D9E2F3" if i % 2 == 0 else "white"
            row_colors.extend([c, c])

        table_path = os.path.join(base_dir, f"{shift_key}_table.png")
        _make_table_image(
            title=f"Distribution Shift: {label}",
            columns=columns,
            rows=rows,
            output_path=table_path,
            row_colors=row_colors,
        )

    json_path = os.path.join(base_dir, "base_configs_results.json")
    with open(json_path, "w") as f:
        json.dump(all_shift_data, f, indent=2)
    print(f"  JSON saved: {json_path}")

    return all_shift_data


# ═══════════════════════════════════════════════════════════════════════════════
# Part 2: Few-shot configs
# ═══════════════════════════════════════════════════════════════════════════════

def generate_fewshot_tables(results, out_dir):
    """Generate few-shot tables showing OOD test metrics across shot counts."""
    fs_dir = os.path.join(out_dir, "fewshot_configs")
    os.makedirs(fs_dir, exist_ok=True)

    columns = [
        "Model",
        "Few-shot",
        "AP@0.50\n(OOD)",
        "AP@[.50:.95]\n(OOD)",
        "Precision\n(OOD)",
        "Recall\n(OOD)",
        "Crown RMSE\n(OOD)",
        "Geoloc RMSE\n(OOD)",
    ]

    all_fs_data = {}

    # ── fastrcnn_pretrained: all 4 shifts ─────────────────────────────────
    for shift_key, shift_info in SHIFTS.items():
        base_config = shift_info["config"]
        label = shift_info["label"]
        rows = []
        fs_json = {}

        for suffix, fs_label in zip(FEWSHOT_SUFFIXES, FEWSHOT_LABELS):
            config = base_config + (f"__{suffix[1:]}" if suffix else "")
            summary = results.get(("fastrcnn_pretrained", config))
            m = extract_split_metrics(summary, "ood_test")

            if m is None:
                rows.append(["Faster R-CNN (pt)", fs_label] + ["—"] * 6)
            else:
                rows.append([
                    "Faster R-CNN (pt)", fs_label,
                    _fmt(m["AP50"]), _fmt(m["AP"]),
                    _fmt(m["Precision"]), _fmt(m["Recall"]),
                    _fmt(m["Crown_RMSE"]), _fmt(m["Geoloc_RMSE"]),
                ])
                fs_json.setdefault("fastrcnn_pretrained", {})[fs_label] = m

        all_fs_data[f"fastrcnn_{shift_key}"] = fs_json

        table_path = os.path.join(fs_dir, f"fastrcnn_fewshot_{shift_key}_table.png")
        _make_table_image(
            title=f"Few-shot: Faster R-CNN — {label}\n(OOD Test Metrics)",
            columns=columns,
            rows=rows,
            output_path=table_path,
        )

    # ── grounding_dino: region_North_South only ───────────────────────────
    shift_key = "region_North_South"
    shift_info = SHIFTS[shift_key]
    base_config = shift_info["config"]
    label = shift_info["label"]
    rows = []
    fs_json = {}

    for suffix, fs_label in zip(FEWSHOT_SUFFIXES, FEWSHOT_LABELS):
        config = base_config + (f"__{suffix[1:]}" if suffix else "")
        summary = results.get(("grounding_dino", config))
        m = extract_split_metrics(summary, "ood_test")

        if m is None:
            rows.append(["Grounding DINO", fs_label] + ["—"] * 6)
        else:
            rows.append([
                "Grounding DINO", fs_label,
                _fmt(m["AP50"]), _fmt(m["AP"]),
                _fmt(m["Precision"]), _fmt(m["Recall"]),
                _fmt(m["Crown_RMSE"]), _fmt(m["Geoloc_RMSE"]),
            ])
            fs_json.setdefault("grounding_dino", {})[fs_label] = m

    all_fs_data[f"grounding_dino_{shift_key}"] = fs_json

    table_path = os.path.join(fs_dir, f"grounding_dino_fewshot_{shift_key}_table.png")
    _make_table_image(
        title=f"Few-shot: Grounding DINO — {label}\n(OOD Test Metrics)",
        columns=columns,
        rows=rows,
        output_path=table_path,
    )

    json_path = os.path.join(fs_dir, "fewshot_results.json")
    with open(json_path, "w") as f:
        json.dump(all_fs_data, f, indent=2)
    print(f"  JSON saved: {json_path}")

    return all_fs_data


# ═══════════════════════════════════════════════════════════════════════════════
# Main
# ═══════════════════════════════════════════════════════════════════════════════

def main():
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument(
        "--outputs-dir", required=True,
        help="Path to final_outputs directory containing model subdirectories",
    )
    parser.add_argument(
        "--out-dir", default="./aggregate_results",
        help="Output directory for aggregate tables and JSON (default: ./aggregate_results)",
    )
    args = parser.parse_args()

    if not os.path.isdir(args.outputs_dir):
        print(f"ERROR: directory not found: {args.outputs_dir}", file=sys.stderr)
        sys.exit(1)

    print(f"Scanning: {args.outputs_dir}")
    results = scan_outputs(args.outputs_dir)
    print(f"Loaded {len(results)} eval summaries.\n")

    if not results:
        print("No results found. Exiting.")
        sys.exit(1)

    # Show what we found
    print("Found results for:")
    for (model, config), _ in sorted(results.items()):
        print(f"  {model:25s} {config}")
    print()

    os.makedirs(args.out_dir, exist_ok=True)

    print("=" * 60)
    print("PART 1: Base config tables (no few-shot)")
    print("=" * 60)
    base_data = generate_base_tables(results, args.out_dir)
    print()

    print("=" * 60)
    print("PART 2: Few-shot tables")
    print("=" * 60)
    fs_data = generate_fewshot_tables(results, args.out_dir)
    print()

    # Combined JSON
    combined = {"base_configs": base_data, "fewshot_configs": fs_data}
    combined_path = os.path.join(args.out_dir, "all_results.json")
    with open(combined_path, "w") as f:
        json.dump(combined, f, indent=2)
    print(f"Combined JSON: {combined_path}")
    print(f"\nAll outputs in: {args.out_dir}")


if __name__ == "__main__":
    main()
