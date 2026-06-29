#!/usr/bin/env python3
import argparse
import json
from pathlib import Path


def main() -> None:
    parser = argparse.ArgumentParser(description="List manifest cases that have not explicitly reviewed hotwords.")
    parser.add_argument("--manifest", default="data/gold_manifest.v1.json")
    parser.add_argument("--ready-only", action="store_true")
    args = parser.parse_args()

    manifest = json.loads(Path(args.manifest).read_text(encoding="utf-8"))
    missing = []
    with_hotwords = []
    empty_hotwords = []
    for case in manifest["cases"]:
        if args.ready_only and case.get("reference", {}).get("status") != "ready":
            continue
        if "hotwords" not in case:
            missing.append(case["case_id"])
        elif case["hotwords"]:
            with_hotwords.append(case["case_id"])
        else:
            empty_hotwords.append(case["case_id"])

    print(f"manifest: {args.manifest}")
    print(f"cases checked: {len(missing) + len(with_hotwords) + len(empty_hotwords)}")
    print(f"missing hotwords field: {len(missing)}")
    print(f"with hotwords: {len(with_hotwords)}")
    print(f"explicit empty hotwords: {len(empty_hotwords)}")
    if missing:
        print("\nMissing:")
        for case_id in missing:
            print(f"- {case_id}")


if __name__ == "__main__":
    main()
