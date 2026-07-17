#!/usr/bin/env python3
"""Fail when a public MoE class has more than one implementation source."""

from __future__ import annotations

import ast
from collections import defaultdict
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1] / "ultralytics/nn/modules/moe"


def find_duplicate_classes(root: Path = ROOT) -> dict[str, list[str]]:
    """Return public top-level class names defined in more than one MoE file."""
    definitions: dict[str, list[str]] = defaultdict(list)
    for path in sorted(root.glob("*.py")):
        tree = ast.parse(path.read_text(encoding="utf-8"), filename=str(path))
        for node in tree.body:
            if isinstance(node, ast.ClassDef) and not node.name.startswith("_"):
                definitions[node.name].append(f"{path.name}:{node.lineno}")
    return {name: locations for name, locations in definitions.items() if len(locations) > 1}


def main() -> int:
    duplicates = find_duplicate_classes()
    if duplicates:
        detail = "; ".join(f"{name}={locations}" for name, locations in sorted(duplicates.items()))
        raise SystemExit(f"MoE SSOT violation: {detail}")
    print("MoE SSOT check: PASS")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
