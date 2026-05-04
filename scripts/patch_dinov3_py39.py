#!/usr/bin/env python3
"""Patch DinoV3 for Python 3.9 compatibility.

Adds 'from __future__ import annotations' (PEP 563) to all .py files.
This defers annotation evaluation, allowing float|None and tuple[X, Y]
type-hint syntax that otherwise requires Python 3.10+.
"""
import pathlib
import sys

FUTURE_LINE = "from __future__ import annotations\n"


def patch(root: str) -> int:
    patched = 0
    for p in pathlib.Path(root).rglob("*.py"):
        text = p.read_text()
        if FUTURE_LINE.strip() not in text:
            p.write_text(FUTURE_LINE + text)
            patched += 1
    return patched


if __name__ == "__main__":
    root = sys.argv[1] if len(sys.argv) > 1 else "/opt/dinov3"
    n = patch(root)
    print(f"Patched {n} files in {root}")
