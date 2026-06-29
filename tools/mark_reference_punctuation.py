#!/usr/bin/env python3
import argparse
import json
import re
from pathlib import Path


PUNCT_RE = re.compile(r"[，。！？；：,.!?;:]")


def has_no_punctuation_reference(case: dict, root: Path) -> bool:
    reference = case.get("reference", {})
    if reference.get("status") != "ready":
        return False
    if reference.get("method") == "empty_reference":
        return False
    ref_path = reference.get("path")
    if not ref_path:
        return False
    path = root / ref_path
    if not path.exists():
        return False
    text = path.read_text(encoding="utf-8").strip()
    return bool(text) and not PUNCT_RE.search(text)


def main() -> None:
    parser = argparse.ArgumentParser(description="Mark ready references without sentence punctuation as punctuation=none.")
    parser.add_argument("--manifest", default="data/benchmark_manifest.v1.json")
    parser.add_argument("--write", action="store_true", help="Update the manifest in place.")
    args = parser.parse_args()

    manifest_path = Path(args.manifest)
    root = manifest_path.resolve().parents[1]
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))

    changed = []
    for case in manifest.get("cases", []):
        reference = case.get("reference", {})
        if not has_no_punctuation_reference(case, root):
            continue
        if reference.get("punctuation") == "none":
            continue
        reference["punctuation"] = "none"
        changed.append(case.get("case_id", ""))

    if args.write and changed:
        manifest_path.write_text(json.dumps(manifest, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")

    action = "updated" if args.write else "would_update"
    print(f"{action}: {len(changed)}")
    for case_id in changed:
        print(case_id)


if __name__ == "__main__":
    main()
