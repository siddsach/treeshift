#!/usr/bin/env python3
"""
Summarize all completed evaluation runs into a single AP50 results table.

Scans outputs/ for directories with eval results and produces:
  - A printed ASCII table sorted by model / shift / config
  - A CSV file saved alongside the outputs directory

Usage (run on login node or compute node — no GPU needed):
    python scripts/summarize_results.py
    python scripts/summarize_results.py --outputs-dir /path/to/outputs --out results.csv
"""

import argparse
import csv
import json
import os
import re
import sys

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

REPO_ROOT   = "/scratch/groups/dlobell/aadityan/tree-distribution-shift"
OUTPUTS_DIR = os.path.join(REPO_ROOT, "outputs")

# Order matters: longest/most-specific prefixes first so fastrcnn_pretrained
# doesn't get matched as fastrcnn.
MODEL_PREFIXES = [
    "fastrcnn_pretrained",
    "maskrcnn_pretrained",
    "plaindetr_dinov3_7b16",
    "plaindetr_dinov3_sat",
    "plaindetr_dinov3",
    "plaindetr_resnet",
    "grounding_dino",
    "fastrcnn",
    "maskrcnn",
]

# Preferred display order for the final table
MODEL_ORDER = [
    "fastrcnn",
    "fastrcnn_pretrained",
    "maskrcnn",
    "maskrcnn_pretrained",
    "plaindetr_resnet",
    "plaindetr_dinov3",
    "plaindetr_dinov3_7b16",
    "plaindetr_dinov3_sat",
    "grounding_dino",
]

SHIFT_ORDER = ["biome", "elev", "intl", "region"]

FS_ORDER = {"base": 0, "fs1": 1, "fs10": 2, "fs100": 3, "fsall": 4}


# ---------------------------------------------------------------------------
# Parsing helpers
# ---------------------------------------------------------------------------

def parse_dir_name(entry):
    """Parse an output directory name into (model, shortname, job_id_int).

    Returns (None, None, None) if the entry doesn't match any known model.
    """
    for model in MODEL_PREFIXES:
        prefix = model + "_"
        if entry.startswith(prefix):
            rest = entry[len(prefix):]                   # e.g. biome_Rajasthan_WET_DRY_17469351
            m = re.search(r"_(\d+)$", rest)
            if not m:
                continue
            job_id  = int(m.group(1))
            short   = rest[:-(len(m.group(1)) + 1)]      # e.g. biome_Rajasthan_WET_DRY
            return model, short, job_id
    return None, None, None


def parse_shortname(short):
    """Break a config shortname into (shift_type, train_domain, ood_domain, few_shot).

    Examples:
        biome_Rajasthan_WET_DRY        → (biome, Rajasthan_WET, DRY, base)
        biome_Rajasthan_WET_DRY_fs1    → (biome, Rajasthan_WET, DRY, fs1)
        intl_IN_US                     → (intl,  IN, US, base)
        region_North_South_fs100       → (region, North, South, fs100)
    """
    # Detect and strip few-shot suffix
    fs_match = re.search(r"_(fs(?:all|\d+))$", short)
    if fs_match:
        few_shot = fs_match.group(1)
        core     = short[:fs_match.start()]
    else:
        few_shot = "base"
        core     = short

    # Detect shift type by prefix
    for st in SHIFT_ORDER:
        if core.startswith(st + "_"):
            remainder = core[len(st) + 1:]   # e.g. Rajasthan_WET_DRY  or  IN_US
            # Split on last underscore to get ood_domain
            idx = remainder.rfind("_")
            if idx == -1:
                return st, remainder, "", few_shot
            train_domain = remainder[:idx]
            ood_domain   = remainder[idx + 1:]
            return st, train_domain, ood_domain, few_shot

    return "unknown", core, "", few_shot


# ---------------------------------------------------------------------------
# Eval result extraction
# ---------------------------------------------------------------------------

def get_ap50(output_dir, split):
    """Return AP50 (float %) for the given split, or None if unavailable.

    Primary  : eval_summary.json → [split]["bbox"]["AP50"]
    Fallback : eval_{split}/tree_detection_metrics.json → ["overall"]["AP50"]
    """
    # Primary
    summary = os.path.join(output_dir, "eval_summary.json")
    if os.path.isfile(summary):
        try:
            with open(summary) as f:
                d = json.load(f)
            v = d.get(split, {}).get("bbox", {}).get("AP50")
            if v is not None:
                return round(float(v), 2)
        except Exception:
            pass

    # Fallback
    tree = os.path.join(output_dir, f"eval_{split}", "tree_detection_metrics.json")
    if os.path.isfile(tree):
        try:
            with open(tree) as f:
                d = json.load(f)
            v = d.get("overall", {}).get("AP50")
            if v is not None:
                return round(float(v), 2)
        except Exception:
            pass

    return None


def has_any_eval(output_dir):
    """True if the directory contains at least partial eval results."""
    return (
        os.path.isfile(os.path.join(output_dir, "eval_summary.json"))
        or os.path.isfile(os.path.join(output_dir, "eval_val",      "per_image_results.json"))
        or os.path.isfile(os.path.join(output_dir, "eval_ood_test", "per_image_results.json"))
    )


# ---------------------------------------------------------------------------
# Scan outputs/
# ---------------------------------------------------------------------------

def scan_outputs(outputs_dir):
    """Return a list of result dicts, one per (model, shortname) combination.

    For configs with multiple runs (re-submissions), keeps the most-recent run
    that has eval results.
    """
    # key → best (job_id, output_dir)
    best = {}

    for entry in sorted(os.listdir(outputs_dir)):
        full = os.path.join(outputs_dir, entry)
        if not os.path.isdir(full):
            continue

        model, short, job_id = parse_dir_name(entry)
        if model is None:
            continue

        if not has_any_eval(full):
            continue

        key = (model, short)
        prev_id, _ = best.get(key, (-1, None))
        if job_id > prev_id:
            best[key] = (job_id, full)

    rows = []
    for (model, short), (job_id, output_dir) in best.items():
        ap50_id  = get_ap50(output_dir, "val")
        ap50_ood = get_ap50(output_dir, "ood_test")

        shift_type, train_domain, ood_domain, few_shot = parse_shortname(short)

        rows.append({
            "model":        model,
            "shift_type":   shift_type,
            "train_domain": train_domain,
            "ood_domain":   ood_domain,
            "few_shot":     few_shot,
            "config":       short,
            "AP50_ID":      ap50_id,
            "AP50_OOD":     ap50_ood,
            "job_id":       job_id,
            "output_dir":   output_dir,
        })

    return rows


# ---------------------------------------------------------------------------
# Sorting
# ---------------------------------------------------------------------------

def sort_key(row):
    model_rank = MODEL_ORDER.index(row["model"]) if row["model"] in MODEL_ORDER else 99
    shift_rank = SHIFT_ORDER.index(row["shift_type"]) if row["shift_type"] in SHIFT_ORDER else 99
    fs_rank    = FS_ORDER.get(row["few_shot"], 99)
    return (model_rank, shift_rank, row["train_domain"], row["ood_domain"], fs_rank)


# ---------------------------------------------------------------------------
# Formatting
# ---------------------------------------------------------------------------

def fmt(v):
    if v is None:
        return "—"
    return f"{v:.1f}"


def print_table(rows):
    cols = ["model", "shift_type", "train_domain", "ood_domain", "few_shot", "AP50_ID", "AP50_OOD"]
    headers = ["Model", "Shift", "Train", "OOD", "Few-shot", "AP50 ID", "AP50 OOD"]

    # Compute column widths
    widths = [len(h) for h in headers]
    display_rows = []
    for r in rows:
        vals = [
            r["model"],
            r["shift_type"],
            r["train_domain"],
            r["ood_domain"],
            r["few_shot"],
            fmt(r["AP50_ID"]),
            fmt(r["AP50_OOD"]),
        ]
        display_rows.append(vals)
        for i, v in enumerate(vals):
            widths[i] = max(widths[i], len(str(v)))

    sep = "+-" + "-+-".join("-" * w for w in widths) + "-+"
    hdr = "| " + " | ".join(h.ljust(widths[i]) for i, h in enumerate(headers)) + " |"

    print(sep)
    print(hdr)
    print(sep)

    prev_model = None
    for vals in display_rows:
        if vals[0] != prev_model and prev_model is not None:
            print(sep)          # blank separator between models
        prev_model = vals[0]
        line = "| " + " | ".join(str(v).ljust(widths[i]) for i, v in enumerate(vals)) + " |"
        print(line)

    print(sep)


def save_csv(rows, path):
    fieldnames = ["model", "shift_type", "train_domain", "ood_domain", "few_shot",
                  "config", "AP50_ID", "AP50_OOD", "job_id", "output_dir"]
    with open(path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)
    print(f"CSV saved to: {path}")


# ---------------------------------------------------------------------------
# Image generation
# ---------------------------------------------------------------------------

# Colours matching the existing eval_summary_table style
_HDR_BG    = "#4472C4"   # header row
_HDR_FG    = "white"
_GRP_BG    = "#1F3864"   # model-group separator rows (dark navy)
_GRP_FG    = "white"
_ROW_ODD   = "#D9E2F3"   # alternating data rows
_ROW_EVEN  = "white"


def save_table_image(rows, path):
    """Render the results table as a PNG, matching eval_summary_table style."""
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except ImportError:
        print("matplotlib not available — skipping image generation")
        return

    COLUMNS = ["Model", "Shift", "Train → OOD", "Few-shot", "AP50 ID (%)", "AP50 OOD (%)"]

    # Build cell data, inserting a separator row at each model-group boundary
    cell_data  = []   # list of cell-value lists
    row_styles = []   # "header" | "group" | "odd" | "even"

    prev_model = None
    parity     = 0   # alternates within each model group

    for r in rows:
        if r["model"] != prev_model:
            # Separator row: full model name, other cells empty
            cell_data.append([r["model"], "", "", "", "", ""])
            row_styles.append("group")
            prev_model = r["model"]
            parity     = 0

        cell_data.append([
            "",                                               # model col blank (group row has it)
            r["shift_type"],
            f"{r['train_domain']} → {r['ood_domain']}",
            r["few_shot"],
            fmt(r["AP50_ID"]),
            fmt(r["AP50_OOD"]),
        ])
        row_styles.append("odd" if parity % 2 == 0 else "even")
        parity += 1

    n_data_cols = len(COLUMNS)
    n_rows      = len(cell_data)

    # Sizing: ~0.38 inches per row, min 4 inches tall
    fig_width  = max(13, n_data_cols * 2.0)
    fig_height = max(4,  0.38 * (n_rows + 1) + 1.2)

    fig, ax = plt.subplots(figsize=(fig_width, fig_height))
    ax.axis("off")
    ax.set_title(
        "AP50 Results — All Models & Configs",
        fontsize=14, fontweight="bold", pad=14,
    )

    table = ax.table(
        cellText=cell_data,
        colLabels=COLUMNS,
        cellLoc="center",
        loc="center",
    )
    table.auto_set_font_size(False)
    table.set_fontsize(8)
    table.scale(1.2, 1.6)

    # Style header row (row index 0 in ax.table)
    for j in range(n_data_cols):
        cell = table[0, j]
        cell.set_facecolor(_HDR_BG)
        cell.set_text_props(color=_HDR_FG, fontweight="bold")

    # Style data rows (row indices 1 … n_rows)
    for i, style in enumerate(row_styles, start=1):
        if style == "group":
            for j in range(n_data_cols):
                table[i, j].set_facecolor(_GRP_BG)
                table[i, j].set_text_props(color=_GRP_FG, fontweight="bold")
            # Merge visually: make the model name left-aligned and span the full width
            table[i, 0].set_text_props(
                color=_GRP_FG, fontweight="bold", ha="left"
            )
        elif style == "odd":
            for j in range(n_data_cols):
                table[i, j].set_facecolor(_ROW_ODD)
        else:
            for j in range(n_data_cols):
                table[i, j].set_facecolor(_ROW_EVEN)

    # Widen first two columns slightly
    table.auto_set_column_width([0, 1, 2, 3, 4, 5])

    os.makedirs(os.path.dirname(path) if os.path.dirname(path) else ".", exist_ok=True)
    fig.savefig(path, dpi=150, bbox_inches="tight", facecolor="white")
    plt.close(fig)
    print(f"Table image saved to: {path}")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--outputs-dir", default=OUTPUTS_DIR,
        help=f"Path to the outputs directory (default: {OUTPUTS_DIR})"
    )
    parser.add_argument(
        "--out", default=None,
        help="Path for the output CSV (default: <outputs-dir>/../results_ap50.csv)"
    )
    parser.add_argument(
        "--img", default=None,
        help="Path for the output PNG (default: <outputs-dir>/../results_ap50.png)"
    )
    args = parser.parse_args()

    if not os.path.isdir(args.outputs_dir):
        print(f"ERROR: outputs directory not found: {args.outputs_dir}")
        sys.exit(1)

    base     = os.path.abspath(os.path.join(args.outputs_dir, ".."))
    csv_path = args.out or os.path.join(base, "results_ap50.csv")
    img_path = args.img or os.path.join(base, "results_ap50.png")
    csv_path = os.path.abspath(csv_path)
    img_path = os.path.abspath(img_path)

    print(f"Scanning: {args.outputs_dir}")
    rows = scan_outputs(args.outputs_dir)

    if not rows:
        print("No completed evaluation runs found.")
        sys.exit(0)

    rows.sort(key=sort_key)

    n_id  = sum(1 for r in rows if r["AP50_ID"]  is not None)
    n_ood = sum(1 for r in rows if r["AP50_OOD"] is not None)
    print(f"Found {len(rows)} runs with eval results  "
          f"({n_id} with AP50 ID,  {n_ood} with AP50 OOD)\n")

    print_table(rows)
    print()
    save_csv(rows, csv_path)
    save_table_image(rows, img_path)


if __name__ == "__main__":
    main()
