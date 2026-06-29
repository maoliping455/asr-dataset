#!/usr/bin/env python3
import json
import os
import sys
from collections import Counter, defaultdict
import re


REQUIRED_CASE_FIELDS = {
    "case_id",
    "phase",
    "priority",
    "status",
    "scenario",
    "language",
    "source_type",
    "source",
    "segment",
    "speakers",
    "acoustic_conditions",
    "primary_error_metric",
    "weight",
    "reference",
    "keywords",
    "numbers",
}

VALID_METRICS = {"cer", "wer", "hybrid_ter"}
VALID_REFERENCE_STATUS = {"ready", "needs_reference", "deferred"}
VALID_REFERENCE_PUNCTUATION = {"normal", "none"}
VALID_REFERENCE_REVIEW_LEVEL = {"user_confirmed_real_audio", "auto_screened_public_subtitle"}
DEFAULT_CONFIDENCE_WEIGHT_MULTIPLIERS = {
    "user_confirmed_real_audio": 3.0,
    "auto_screened_public_subtitle": 2.0,
    "ready_reference": 1.0,
    "other": 0.2,
}
GOLD_REVIEW_LEVELS = {"user_confirmed_real_audio", "auto_screened_public_subtitle"}
PUNCT_RE = re.compile(r"[，。！？；：!?;:]|(?<!\d)[,.](?!\d)")


def strip_technical_punctuation_context(text: str) -> str:
    text = re.sub(r"https?://\S+", " ", text)
    return re.sub(r"\b[\w-]+(?:\.[\w-]+)+\b", " ", text)


def has_sentence_punctuation(text: str) -> bool:
    return bool(PUNCT_RE.search(strip_technical_punctuation_context(text)))


def fail(message: str) -> None:
    print(f"ERROR: {message}", file=sys.stderr)
    raise SystemExit(1)


def warn(message: str) -> None:
    print(f"WARN: {message}", file=sys.stderr)


def confidence_weight_tier(case: dict) -> str:
    reference = case.get("reference", {})
    if reference.get("review_level") in GOLD_REVIEW_LEVELS:
        return reference.get("review_level")
    if reference.get("status") == "ready":
        return "ready_reference"
    return "other"


def main() -> None:
    manifest_path = sys.argv[1] if len(sys.argv) > 1 else "data/gold_manifest.v1.json"
    with open(manifest_path, "r", encoding="utf-8") as f:
        manifest = json.load(f)

    cases = manifest.get("cases")
    if not isinstance(cases, list) or not cases:
        fail("manifest.cases must be a non-empty list")

    seen = set()
    ready_count = 0
    candidate_count = 0
    total_weight = 0.0
    tier_counts = Counter()
    tier_effective_weights = defaultdict(list)
    configured_multipliers = (
        (manifest.get("policy", {}).get("case_weighting", {}) or {}).get("confidence_weight_multipliers", {})
    )
    confidence_multipliers = {**DEFAULT_CONFIDENCE_WEIGHT_MULTIPLIERS, **configured_multipliers}
    for tier, multiplier in confidence_multipliers.items():
        if tier not in DEFAULT_CONFIDENCE_WEIGHT_MULTIPLIERS:
            fail(f"policy.case_weighting has unknown confidence tier {tier!r}")
        if not isinstance(multiplier, (int, float)) or multiplier < 0:
            fail(f"policy.case_weighting multiplier for {tier!r} must be a non-negative number")

    root = os.path.abspath(os.path.join(os.path.dirname(manifest_path), ".."))

    for idx, case in enumerate(cases):
        missing = REQUIRED_CASE_FIELDS - set(case)
        if missing:
            fail(f"case #{idx} missing required fields: {sorted(missing)}")

        case_id = case["case_id"]
        if not isinstance(case_id, str) or not case_id:
            fail(f"case #{idx} has invalid case_id")
        if case_id in seen:
            fail(f"duplicate case_id: {case_id}")
        seen.add(case_id)

        metric = case["primary_error_metric"]
        if metric not in VALID_METRICS:
            fail(f"{case_id}: invalid primary_error_metric {metric!r}")

        ref = case["reference"]
        if not isinstance(ref, dict):
            fail(f"{case_id}: reference must be an object")
        if ref.get("status") not in VALID_REFERENCE_STATUS:
            fail(f"{case_id}: invalid reference.status {ref.get('status')!r}")
        if "punctuation" in ref and ref.get("punctuation") not in VALID_REFERENCE_PUNCTUATION:
            fail(f"{case_id}: invalid reference.punctuation {ref.get('punctuation')!r}")
        if "review_level" in ref and ref.get("review_level") not in VALID_REFERENCE_REVIEW_LEVEL:
            fail(f"{case_id}: invalid reference.review_level {ref.get('review_level')!r}")
        ref_path = ref.get("path")
        if not ref_path or not ref_path.startswith("data/gold_references/"):
            fail(f"{case_id}: reference.path must be under data/gold_references/")

        weight = case["weight"]
        if not isinstance(weight, (int, float)) or weight < 0:
            fail(f"{case_id}: weight must be a non-negative number")
        total_weight += float(weight)
        tier = confidence_weight_tier(case)
        tier_counts[tier] += 1
        tier_effective_weights[tier].append(float(weight) * float(confidence_multipliers[tier]))

        segment = case["segment"]
        if not isinstance(segment, dict):
            fail(f"{case_id}: segment must be an object")
        start = segment.get("start_sec")
        end = segment.get("end_sec")
        duration = segment.get("duration_sec")
        if not isinstance(start, (int, float)):
            fail(f"{case_id}: segment.start_sec must be numeric")
        if end is not None and (not isinstance(end, (int, float)) or end <= start):
            fail(f"{case_id}: segment.end_sec must be null or > start_sec")
        if not isinstance(duration, (int, float)) or duration <= 0:
            fail(f"{case_id}: segment.duration_sec must be positive")

        if not isinstance(case["keywords"], list):
            fail(f"{case_id}: keywords must be a list")
        if not isinstance(case["numbers"], list):
            fail(f"{case_id}: numbers must be a list")
        if "hotwords" in case:
            if not isinstance(case["hotwords"], list):
                fail(f"{case_id}: hotwords must be a list")
            for hotword in case["hotwords"]:
                if not isinstance(hotword, str) or not hotword.strip():
                    fail(f"{case_id}: hotwords must contain non-empty strings")

        if ref.get("status") == "ready":
            ready_count += 1
            full_ref_path = os.path.join(root, ref_path)
            if not os.path.exists(full_ref_path):
                fail(f"{case_id}: ready reference missing on disk: {ref_path}")
            elif ref.get("method") != "empty_reference":
                ref_text = open(full_ref_path, "r", encoding="utf-8").read().strip()
                if ref_text and not has_sentence_punctuation(ref_text) and ref.get("punctuation") != "none":
                    warn(f"{case_id}: ready reference has no sentence punctuation; mark reference.punctuation as 'none'")
                if ref.get("punctuation") == "none" and has_sentence_punctuation(ref_text):
                    warn(f"{case_id}: reference.punctuation is 'none' but reference text contains sentence punctuation")
        else:
            candidate_count += 1

    print(f"manifest: {manifest_path}")
    print(f"cases: {len(cases)}")
    print(f"ready references: {ready_count}")
    print(f"candidate/deferred references: {candidate_count}")
    print(f"total raw weight: {total_weight:.2f}")
    for tier in ["user_confirmed_real_audio", "auto_screened_public_subtitle", "ready_reference", "other"]:
        weights = tier_effective_weights[tier]
        if weights:
            print(
                f"{tier}: count={tier_counts[tier]} multiplier={confidence_multipliers[tier]:.2f} "
                f"effective_weight_range={min(weights):.2f}-{max(weights):.2f}"
            )
        else:
            print(f"{tier}: count=0 multiplier={confidence_multipliers[tier]:.2f}")

    phase_weights = manifest.get("phase1_scenario_weights", {})
    if phase_weights:
        weight_sum = sum(phase_weights.values())
        if abs(weight_sum - 1.0) > 0.001:
            warn(f"phase1_scenario_weights sum to {weight_sum:.3f}, expected 1.0")


if __name__ == "__main__":
    main()
