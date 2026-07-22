#!/usr/bin/env python3
"""Deterministically mirror RepoWiki content into the GitHub Pages content tree."""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import shutil
import tempfile
import unicodedata
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
LANGS = ("en", "zh")


def files(root: Path) -> list[Path]:
    return sorted(root.rglob("*.md"), key=lambda p: p.relative_to(root).as_posix())


def digest(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def key_for(path: str) -> str:
    stem = Path(path).stem
    return "".join(c for c in unicodedata.normalize("NFKC", stem).casefold() if c.isalnum())


def build_mapping(paths: dict[str, list[str]]) -> tuple[dict[str, dict[str, str]], list[dict[str, str]]]:
    result = {lang: {} for lang in LANGS}
    other_by_key = {}
    for lang in LANGS:
        other = "zh" if lang == "en" else "en"
        buckets: dict[str, list[str]] = {}
        for path in paths[other]:
            buckets.setdefault(key_for(path), []).append(path)
        other_by_key[lang] = buckets
    missing = []
    for lang in LANGS:
        other = "zh" if lang == "en" else "en"
        other_set = set(paths[other])
        for path in paths[lang]:
            match = path if path in other_set else None
            if not match:
                candidates = other_by_key[lang].get(key_for(path), [])
                match = candidates[0] if len(candidates) == 1 else None
            if match:
                result[lang][path] = f"{other}/{match}"
            else:
                missing.append({"language": lang, "path": path, "fallback": f"{other}/"})
    return result, missing


def category(directory: Path, content_root: Path, lang: str, mapping: dict[str, str]) -> dict:
    rel_dir = directory.relative_to(content_root).as_posix()
    pages = []
    for page in sorted(directory.glob("*.md"), key=lambda p: p.name):
        rel = page.relative_to(content_root).as_posix()
        pages.append({"name": page.stem, "path": f"{lang}/{rel}", "translationKey": f"{lang}:{rel}",
                      "alternatePath": mapping.get(rel)})
    children = [category(p, content_root, lang, mapping) for p in sorted(directory.iterdir(), key=lambda p: p.name)
                if p.is_dir()]
    landing = next((p["path"] for p in pages if Path(p["path"]).stem == directory.name), None)
    return {"name": directory.name, "path": rel_dir, "landingPage": landing,
            "children": children, "pages": pages}


def index_for(root: Path, lang: str, mapping: dict[str, str]) -> dict:
    children = [category(p, root, lang, mapping) for p in sorted(root.iterdir(), key=lambda p: p.name) if p.is_dir()]
    return {"name": lang.upper(), "language": lang, "children": children}


def json_bytes(value: object) -> bytes:
    return (json.dumps(value, ensure_ascii=False, indent=2, sort_keys=False) + "\n").encode()


def expected(source: Path) -> tuple[dict[str, dict[str, bytes]], dict[str, bytes]]:
    source_files: dict[str, dict[str, bytes]] = {}
    paths: dict[str, list[str]] = {}
    for lang in LANGS:
        root = source / lang / "content"
        if not root.is_dir():
            raise SystemExit(f"missing source: {root}")
        source_files[lang] = {p.relative_to(root).as_posix(): p.read_bytes() for p in files(root)}
        paths[lang] = list(source_files[lang])
    mapping, missing = build_mapping(paths)
    indexes = {f"index-{lang}.json": json_bytes(index_for(source / lang / "content", lang, mapping[lang])) for lang in LANGS}
    indexes["repowiki-language-map.json"] = json_bytes({"schemaVersion": 1, "missing": missing, "mapping": mapping})
    return source_files, indexes


def check(dest: Path, source_files: dict[str, dict[str, bytes]], indexes: dict[str, bytes]) -> list[str]:
    errors = []
    for lang in LANGS:
        actual_root = dest / lang
        actual = {p.relative_to(actual_root).as_posix(): p.read_bytes() for p in files(actual_root)} if actual_root.exists() else {}
        if actual != source_files[lang]:
            errors.append(f"{lang}: content differs ({len(actual)} != {len(source_files[lang])} or hashes changed)")
    for name, data in indexes.items():
        if not (dest / name).is_file() or (dest / name).read_bytes() != data:
            errors.append(f"{name}: stale")
    return errors


def sync(dest: Path, source_files: dict[str, dict[str, bytes]], indexes: dict[str, bytes]) -> None:
    dest.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(prefix="repowiki-sync-", dir=dest.parent) as td:
        stage = Path(td)
        for lang, entries in source_files.items():
            for rel, data in entries.items():
                target = stage / lang / rel
                target.parent.mkdir(parents=True, exist_ok=True)
                target.write_bytes(data)
        for name, data in indexes.items():
            (stage / name).write_bytes(data)
        for lang in LANGS:
            target = dest / lang
            backup = dest / f".{lang}.old"
            if backup.exists(): shutil.rmtree(backup)
            if target.exists(): os.replace(target, backup)
            os.replace(stage / lang, target)
            if backup.exists(): shutil.rmtree(backup)
        for name in indexes:
            os.replace(stage / name, dest / name)


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--source", type=Path, default=ROOT / "wiki/repowiki")
    parser.add_argument("--dest", type=Path, default=ROOT / "wiki-content")
    parser.add_argument("--check", action="store_true")
    args = parser.parse_args()
    source_files, indexes = expected(args.source.resolve())
    if args.check:
        errors = check(args.dest.resolve(), source_files, indexes)
        print("\n".join(errors) if errors else f"RepoWiki sync OK: en={len(source_files['en'])}, zh={len(source_files['zh'])}")
        return bool(errors)
    sync(args.dest.resolve(), source_files, indexes)
    print(f"Synced RepoWiki: en={len(source_files['en'])}, zh={len(source_files['zh'])}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
