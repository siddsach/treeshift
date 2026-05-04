import argparse
from pathlib import Path

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--repo_root", required=True)
    ap.add_argument("--repo_id", required=True)
    args = ap.parse_args()

    repo_root = Path(args.repo_root)
    cfg_root = repo_root / "data" / "configs"
    cfgs = sorted([p.name for p in cfg_root.iterdir() if p.is_dir()])
    if not cfgs:
        raise RuntimeError("No configs found under data/configs. Generate configs first.")

    # YAML header for HF data-only configs
    out_lines = []
    out_lines.append("---")
    out_lines.append("configs:")
    for cfg in cfgs:
        out_lines.append(f"- config_name: {cfg}")
        out_lines.append("  data_files:")
        out_lines.append("  - split: train")
        out_lines.append(f"    path: \"data/configs/{cfg}/train-*.parquet\"")
        out_lines.append("  - split: val")
        out_lines.append(f"    path: \"data/configs/{cfg}/val-*.parquet\"")
        out_lines.append("  - split: ood_test")
        out_lines.append(f"    path: \"data/configs/{cfg}/ood_test-*.parquet\"")
    out_lines.append("---")
    out_lines.append("")

    example = cfgs[0]

    # Body (no triple-quoted strings)
    out_lines.append("# Tree Distribution Shift — Satellite Tree Detection (COCO export)")
    out_lines.append("")
    out_lines.append("This dataset is organized as **configs** (distribution shift settings).")
    out_lines.append("Each config provides 3 splits:")
    out_lines.append("")
    out_lines.append("- `train` (90% of the in-distribution pool)")
    out_lines.append("- `val` (10% held-out from the in-distribution pool)")
    out_lines.append("- `ood_test` (held-out from the out-of-distribution pool)")
    out_lines.append("")
    out_lines.append("## Super simple COCO UX")
    out_lines.append("")
    out_lines.append("Export any config into standard COCO folder structure:")
    out_lines.append("")
    out_lines.append("```bash")
    out_lines.append(f"python tools/export_coco.py --repo {args.repo_id} --config {example} --out ./coco_out")
    out_lines.append("```")
    out_lines.append("")
    out_lines.append("This writes:")
    out_lines.append("")
    out_lines.append(f"- `./coco_out/{example}/train/images/*.tiff`")
    out_lines.append(f"- `./coco_out/{example}/train/annotations/instances_train.json`")
    out_lines.append(f"- `./coco_out/{example}/val/...`")
    out_lines.append(f"- `./coco_out/{example}/ood_test/...`")
    out_lines.append("")
    out_lines.append("## Programmatic load (optional)")
    out_lines.append("")
    out_lines.append("```python")
    out_lines.append("from datasets import load_dataset")
    out_lines.append(f"ds = load_dataset(\"{args.repo_id}\", \"{example}\")")
    out_lines.append("print(ds)")
    out_lines.append("```")
    out_lines.append("")

    (repo_root / "README.md").write_text("\n".join(out_lines), encoding="utf-8")
    print(f"Wrote README.md with {len(cfgs)} configs. Example config: {example}")

if __name__ == "__main__":
    main()
