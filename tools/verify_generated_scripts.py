#!/usr/bin/env python3
"""Verify generated Slurm scripts are in sync with their generators.

This command does not write into the repository's script directories. It copies
`scripts/` and `scripts2/` into a temporary workspace, runs the generators there,
and diffs the canonical generated `scripts2/` result against committed `scripts2/`.
"""

import argparse
import difflib
import filecmp
import os
import shutil
import subprocess
import sys
import tempfile
from pathlib import Path


IGNORE_DIRS = {"__pycache__", "logs", ".pytest_cache"}
IGNORE_SUFFIXES = {".pyc", ".pyo"}


def copy_script_tree(src, dst):
    def _ignore(_dir, names):
        ignored = set()
        for name in names:
            p = Path(name)
            if name in IGNORE_DIRS or p.suffix in IGNORE_SUFFIXES:
                ignored.add(name)
        return ignored

    shutil.copytree(str(src), str(dst), ignore=_ignore)


def run_generator(script, cwd):
    print("[verify] running {}".format(script.relative_to(cwd)))
    env = os.environ.copy()
    env["LANG"] = "en_US.utf8"
    env["LC_ALL"] = "en_US.utf8"
    subprocess.run([sys.executable, str(script)], cwd=str(cwd), env=env, check=True)


def iter_files(root):
    files = []
    for path in root.rglob("*"):
        if path.is_dir():
            continue
        if any(part in IGNORE_DIRS for part in path.relative_to(root).parts):
            continue
        if path.suffix in IGNORE_SUFFIXES:
            continue
        files.append(path.relative_to(root))
    return sorted(files)


def unified_diff(expected, actual, rel):
    try:
        expected_lines = expected.read_text(encoding='utf-8').splitlines(keepends=True)
    except UnicodeDecodeError:
        expected_lines = []
    try:
        actual_lines = actual.read_text(encoding='utf-8').splitlines(keepends=True)
    except UnicodeDecodeError:
        actual_lines = []
    if expected_lines or actual_lines:
        return "".join(
            difflib.unified_diff(
                expected_lines,
                actual_lines,
                fromfile="committed/{}".format(rel),
                tofile="generated/{}".format(rel),
                n=3,
            )
        )
    return "Binary files differ: {}\n".format(rel)


def compare_tree(committed, generated, label, max_diffs):
    committed_files = set(iter_files(committed))
    generated_files = set(iter_files(generated))
    failures = 0

    for rel in sorted(committed_files - generated_files):
        print("[diff:{}] committed-only file: {}".format(label, rel))
        failures += 1
    for rel in sorted(generated_files - committed_files):
        print("[diff:{}] generated-only file: {}".format(label, rel))
        failures += 1

    for rel in sorted(committed_files & generated_files):
        left = committed / rel
        right = generated / rel
        if filecmp.cmp(str(left), str(right), shallow=False):
            continue
        failures += 1
        if failures <= max_diffs:
            diff = unified_diff(left, right, rel)
            print("[diff:{}] content differs: {}".format(label, rel))
            print(diff[:8000].encode('ascii', 'backslashreplace').decode('ascii'))

    return failures


def main():
    default_root = Path(__file__).resolve().parents[1]
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--repo-root",
        default=default_root,
        type=Path,
        help="Repository root to verify (default: parent of this tool).",
    )
    parser.add_argument(
        "--max-diffs",
        default=10,
        type=int,
        help="Maximum content diffs to print before only counting failures.",
    )
    args = parser.parse_args()

    repo = args.repo_root.resolve()
    scripts = repo / "scripts"
    scripts2 = repo / "scripts2"

    required = [
        scripts / "generate_scripts.py",
        scripts2 / "generate_scripts2.py",
        scripts2 / "generate_rcnn_scripts2.py",
    ]
    missing = [str(p) for p in required if not p.exists()]
    if missing:
        print("Missing generator(s):")
        for path in missing:
            print("  {}".format(path))
        return 2

    with tempfile.TemporaryDirectory(prefix="treeshift-script-verify-") as tmp_s:
        tmp = Path(tmp_s)
        tmp_scripts = tmp / "scripts"
        tmp_scripts2 = tmp / "scripts2"
        copy_script_tree(scripts, tmp_scripts)
        copy_script_tree(scripts2, tmp_scripts2)

        run_generator(tmp_scripts / "generate_scripts.py", cwd=tmp)
        run_generator(tmp_scripts2 / "generate_scripts2.py", cwd=tmp)
        run_generator(tmp_scripts2 / "generate_rcnn_scripts2.py", cwd=tmp)

        failures = 0
        failures += compare_tree(scripts2, tmp_scripts2, "scripts2", args.max_diffs)

    if failures:
        print("[verify] FAILED: {} generated-script difference(s) found".format(failures))
        return 1

    print("[verify] OK: generated scripts match committed scripts")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
