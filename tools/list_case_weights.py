#!/usr/bin/env python3
import argparse
import json
from collections import Counter


DEFAULT_CONFIDENCE_WEIGHT_MULTIPLIERS = {
    "user_confirmed_real_audio": 3.0,
    "auto_screened_public_subtitle": 2.0,
    "ready_reference": 1.0,
    "other": 0.2,
}
BENCHMARK_REVIEW_LEVELS = {"user_confirmed_real_audio", "auto_screened_public_subtitle"}


def confidence_weight_tier(case: dict) -> str:
    reference = case.get("reference", {})
    if reference.get("review_level") in BENCHMARK_REVIEW_LEVELS:
        return reference.get("review_level")
    if reference.get("status") == "ready":
        return "ready_reference"
    return "other"


def is_benchmark_case(case: dict) -> bool:
    reference = case.get("reference", {})
    return reference.get("status") == "ready" and reference.get("review_level") in BENCHMARK_REVIEW_LEVELS


def include_case_for_scope(case: dict, scope: str) -> bool:
    reference = case.get("reference", {})
    if scope == "all":
        return True
    if scope == "ready":
        return reference.get("status") == "ready"
    if scope == "benchmark":
        return is_benchmark_case(case)
    raise ValueError(f"unknown scope: {scope}")


def main() -> None:
    parser = argparse.ArgumentParser(description="List ASR case base weights and effective weights.")
    parser.add_argument("--manifest", default="data/benchmark_manifest.v1.json")
    parser.add_argument("--tier", choices=["user_confirmed_real_audio", "auto_screened_public_subtitle", "ready_reference", "other"])
    parser.add_argument(
        "--scope",
        choices=["benchmark", "ready", "all"],
        default="benchmark",
        help="Default is the curated benchmark set only. Use ready/all for backup/candidate audits.",
    )
    args = parser.parse_args()

    with open(args.manifest, "r", encoding="utf-8") as f:
        manifest = json.load(f)

    configured = (
        (manifest.get("policy", {}).get("case_weighting", {}) or {}).get("confidence_weight_multipliers", {})
    )
    multipliers = {**DEFAULT_CONFIDENCE_WEIGHT_MULTIPLIERS, **configured}
    rows = []
    for case in manifest.get("cases", []):
        if not include_case_for_scope(case, args.scope):
            continue
        tier = confidence_weight_tier(case)
        if args.tier and tier != args.tier:
            continue
        base_weight = float(case.get("weight", 0.0))
        multiplier = float(multipliers[tier])
        rows.append(
            {
                "case_id": case.get("case_id", ""),
                "tier": tier,
                "base_weight": base_weight,
                "multiplier": multiplier,
                "effective_weight": base_weight * multiplier,
                "reference_status": case.get("reference", {}).get("status", ""),
                "scenario": case.get("scenario", ""),
                "language": case.get("language", ""),
            }
        )

    rows.sort(key=lambda row: (-row["effective_weight"], row["tier"], row["case_id"]))
    counts = Counter(row["tier"] for row in rows)
    print(f"cases: {len(rows)}")
    for tier in ["user_confirmed_real_audio", "auto_screened_public_subtitle", "ready_reference", "other"]:
        if not args.tier or args.tier == tier:
            print(f"{tier}: {counts[tier]}")
    print("case_id\ttier\tbase_weight\tmultiplier\teffective_weight\treference_status\tscenario\tlanguage")
    for row in rows:
        print(
            "\t".join(
                [
                    row["case_id"],
                    row["tier"],
                    f"{row['base_weight']:.2f}",
                    f"{row['multiplier']:.2f}",
                    f"{row['effective_weight']:.2f}",
                    row["reference_status"],
                    row["scenario"],
                    row["language"],
                ]
            )
        )


if __name__ == "__main__":
    main()
