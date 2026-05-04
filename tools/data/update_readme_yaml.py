from __future__ import annotations
from pathlib import Path
import re

REPO_ROOT = Path(__file__).resolve().parents[1]
README = REPO_ROOT / "README.md"
CONFIGS_DIR = REPO_ROOT / "data" / "configs"

SPLITS = [
    "train",
    "id_test",
    "ood_train",
    "ood_test",
]

def has_files(cfg_dir: Path, split: str) -> bool:
    return any(cfg_dir.glob(f"{split}-*.parquet"))

def build_yaml() -> str:
    cfg_dirs = sorted([p for p in CONFIGS_DIR.iterdir() if p.is_dir()], key=lambda p: p.name)
    lines = ["---", "configs:"]
    for cfg in cfg_dirs:
        present = [s for s in SPLITS if has_files(cfg, s)]
        # always include at least train/id_test/ood_test, but only write what's present on disk
        lines.append(f"- config_name: {cfg.name}")
        lines.append("  data_files:")
        for s in present:
            lines.append(f"  - split: {s}")
            lines.append(f"    path: \"data/configs/{cfg.name}/{s}-*.parquet\"")
    lines.append("---")
    return "\n".join(lines) + "\n"

def strip_existing_yaml(text: str) -> str:
    # Remove existing top YAML block if present
    if text.startswith("---"):
        m = re.search(r"^---\s*\n.*?\n---\s*\n", text, flags=re.S)
        if m:
            return text[m.end():]
    return text

def main():
    old = README.read_text() if README.exists() else ""
    body = strip_existing_yaml(old).lstrip("\n")
    new = build_yaml() + "\n" + body
    README.write_text(new)
    print(f"Wrote YAML configs to {README}")

if __name__ == "__main__":
    main()
