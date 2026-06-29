#!/usr/bin/env python3
import argparse
import json


def main() -> None:
    parser = argparse.ArgumentParser(description="List user-confirmed real-audio ASR cases.")
    parser.add_argument("--manifest", default="data/gold_manifest.v1.json")
    args = parser.parse_args()

    with open(args.manifest, "r", encoding="utf-8") as f:
        manifest = json.load(f)

    cases = [
        case
        for case in manifest.get("cases", [])
        if case.get("reference", {}).get("review_level") == "user_confirmed_real_audio"
    ]

    print(f"user_confirmed_real_audio: {len(cases)}")
    for case in cases:
        ref = case.get("reference", {})
        source = case.get("source", {})
        segment = case.get("segment", {})
        print(
            "\t".join(
                [
                    case.get("case_id", ""),
                    case.get("language", ""),
                    case.get("scenario", ""),
                    source.get("bvid") or source.get("url", ""),
                    f"{segment.get('start_sec')}-{segment.get('end_sec')}s",
                    ref.get("reviewed_at", ""),
                ]
            )
        )


if __name__ == "__main__":
    main()
