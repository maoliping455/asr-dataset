#!/usr/bin/env python3
import argparse
import json
from collections import Counter, defaultdict
from pathlib import Path
from typing import Iterable


ROOT = Path(__file__).resolve().parents[1]
DEFAULT_MANIFEST = "data/benchmark_manifest.v1.json"
BENCHMARK_REVIEW_LEVELS = {"user_confirmed_real_audio", "auto_screened_public_subtitle"}
DEFAULT_BACKTEST_ROOTS = (
    ("qwen3_short", "results/qwen3_asr_1_7b_4bit_case_backtests"),
    ("qwen3_long_vad", "results/qwen3_asr_1_7b_4bit_long_audio_vad_chunked_backtests"),
)


def read_json(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def rel_path(path: str) -> Path:
    raw = Path(path)
    if raw.is_absolute():
        return raw
    return ROOT / raw


def is_benchmark(case: dict) -> bool:
    reference = case.get("reference") or {}
    return reference.get("status") == "ready" and reference.get("review_level") in BENCHMARK_REVIEW_LEVELS


def case_bucket(case: dict) -> str:
    if is_benchmark(case):
        if case.get("case_type") == "real_long_benchmark":
            return "benchmark_long"
        return "benchmark_short"
    if case.get("case_type") == "composed_benchmark_stress" or case.get("scenario") == "long_form_stress":
        return "engineering_stress"
    reference = case.get("reference") or {}
    if reference.get("status") == "ready":
        return "backup_ready"
    if reference.get("status") == "deferred":
        return "candidate_deferred"
    return "candidate_needs_reference"


def case_duration(case: dict) -> float | None:
    segment = case.get("segment") or {}
    duration = segment.get("duration_sec")
    if isinstance(duration, (int, float)):
        return float(duration)
    start = segment.get("start_sec")
    end = segment.get("end_sec")
    if isinstance(start, (int, float)) and isinstance(end, (int, float)) and end > start:
        return float(end - start)
    return None


def duration_bucket(duration: float | None) -> str:
    if duration is None:
        return "unknown"
    if duration < 30:
        return "<30s"
    if duration < 60:
        return "30-60s"
    if duration < 180:
        return "60-180s"
    if duration < 600:
        return "3-10m"
    if duration < 1200:
        return "10-20m"
    if duration < 2700:
        return "20-45m"
    if duration < 5400:
        return "45-90m"
    if duration < 9000:
        return "90-150m"
    if duration < 16200:
        return "3.5-4.5h"
    return ">4.5h"


def backtest_labels(case_id: str, roots: Iterable[tuple[str, str]]) -> list[str]:
    labels = []
    for label, root in roots:
        if (ROOT / root / case_id / "summary.md").exists():
            labels.append(label)
    return labels


def backtest_check_available(roots: Iterable[tuple[str, str]]) -> bool:
    return any((ROOT / root).exists() for _, root in roots)


def print_counter(title: str, counter: Counter, limit: int) -> None:
    print(title)
    if not counter:
        print("  none")
        return
    for key, count in counter.most_common(limit):
        print(f"  {key}: {count}")


def summarize_group(name: str, cases: list[dict], roots: Iterable[tuple[str, str]], limit: int) -> None:
    print(f"\n## {name}")
    print(f"count: {len(cases)}")
    print_counter("language:", Counter(case.get("language", "unknown") for case in cases), limit)
    print_counter("scenario:", Counter(case.get("scenario", "unknown") for case in cases), limit)
    print_counter("source_type:", Counter(case.get("source_type", "unknown") for case in cases), limit)
    print_counter("priority:", Counter(case.get("priority", "unknown") for case in cases), limit)
    print_counter(
        "duration:",
        Counter(duration_bucket(case_duration(case)) for case in cases),
        limit,
    )
    print_counter(
        "reference_punctuation:",
        Counter((case.get("reference") or {}).get("punctuation", "normal") for case in cases),
        limit,
    )
    missing_hotwords = [case["case_id"] for case in cases if "hotwords" not in case]
    explicit_empty_hotwords = [case["case_id"] for case in cases if case.get("hotwords") == []]
    with_hotwords = [case["case_id"] for case in cases if case.get("hotwords")]
    check_backtests = backtest_check_available(roots)
    missing_backtest = [case["case_id"] for case in cases if check_backtests and not backtest_labels(case["case_id"], roots)]
    print(f"hotwords: with={len(with_hotwords)} explicit_empty={len(explicit_empty_hotwords)} missing_field={len(missing_hotwords)}")
    if check_backtests:
        print(f"qwen_backtest_summary: present={len(cases) - len(missing_backtest)} missing={len(missing_backtest)}")
    else:
        print("qwen_backtest_summary: not_checked_local_results_not_present")
    if missing_hotwords:
        print("missing_hotwords_sample:")
        for case_id in missing_hotwords[:limit]:
            print(f"  - {case_id}")
    if missing_backtest:
        print("missing_backtest_sample:")
        for case_id in missing_backtest[:limit]:
            print(f"  - {case_id}")


def main() -> None:
    parser = argparse.ArgumentParser(description="Summarize ASR dataset with benchmark-first buckets.")
    parser.add_argument("--manifest", default=DEFAULT_MANIFEST)
    parser.add_argument("--limit", type=int, default=20)
    parser.add_argument("--json", action="store_true", help="Emit machine-readable JSON instead of text.")
    args = parser.parse_args()

    manifest_path = rel_path(args.manifest)
    manifest = read_json(manifest_path)
    cases = manifest.get("cases", [])
    buckets: dict[str, list[dict]] = defaultdict(list)
    for case in cases:
        buckets[case_bucket(case)].append(case)

    roots = DEFAULT_BACKTEST_ROOTS
    if args.json:
        payload = {
            "manifest": args.manifest,
            "case_count": len(cases),
            "buckets": {},
        }
        for name, group in sorted(buckets.items()):
            missing_hotwords = [case["case_id"] for case in group if "hotwords" not in case]
            check_backtests = backtest_check_available(roots)
            missing_backtest = [case["case_id"] for case in group if check_backtests and not backtest_labels(case["case_id"], roots)]
            payload["buckets"][name] = {
                "count": len(group),
                "language": Counter(case.get("language", "unknown") for case in group),
                "scenario": Counter(case.get("scenario", "unknown") for case in group),
                "source_type": Counter(case.get("source_type", "unknown") for case in group),
                "duration": Counter(duration_bucket(case_duration(case)) for case in group),
                "missing_hotwords": missing_hotwords,
                "backtest_check_available": check_backtests,
                "missing_backtest": missing_backtest,
            }
        print(json.dumps(payload, ensure_ascii=False, indent=2))
        return

    print(f"manifest: {args.manifest}")
    print(f"cases: {len(cases)}")
    for name in ["benchmark_short", "benchmark_long", "backup_ready", "candidate_needs_reference", "candidate_deferred", "engineering_stress"]:
        print(f"{name}: {len(buckets.get(name, []))}")
    for name in ["benchmark_short", "benchmark_long", "backup_ready", "candidate_needs_reference", "candidate_deferred", "engineering_stress"]:
        summarize_group(name, buckets.get(name, []), roots, args.limit)


if __name__ == "__main__":
    main()
