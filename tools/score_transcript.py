#!/usr/bin/env python3
import argparse
import json
import math
import os
import re
import sys
import unicodedata
from collections import Counter
from difflib import SequenceMatcher
from typing import Dict, Iterable, List, Sequence, Tuple

try:
    from rapidfuzz.distance import Levenshtein as RapidFuzzLevenshtein
except ImportError:
    RapidFuzzLevenshtein = None

try:
    import opencc
except ImportError:
    opencc = None


CJK_RE = re.compile(r"[\u3400-\u4dbf\u4e00-\u9fff\u3040-\u30ff\uac00-\ud7af]")
ARABIC_RE = re.compile(r"[\u0600-\u06ff]+")
LATIN_WORD_RE = re.compile(r"[a-zA-Z]+(?:[-'][a-zA-Z]+)*")
JAPANESE_KANA_PAREN_RE = re.compile(r"（[ぁ-んァ-ヶー\s]+）|\([ぁ-んァ-ヶー\s]+\)")
NUMBER_RE = re.compile(
    r"(?:\d+(?:[,.]\d+)*(?:[万亿]\d+(?:[,.]\d+)*)+(?:[万亿])?)"
    r"|(?:\d+(?:[,.]\d+)*[万亿])"
    r"|(?:\d+(?:[,.]\d+)*(?:st|nd|rd|th)?%?)"
    r"|(?:[一二三四五六七八九十百千万亿零〇两点]+)",
    re.IGNORECASE,
)
PUNCT_RE = re.compile(r"[，。！？；：!?;:]|(?<!\d)[,.](?!\d)")
BRACKET_EVENT_RE = re.compile(r"\[[^\]]+\]")
CHINESE_DIGITS = {
    "零": 0,
    "〇": 0,
    "一": 1,
    "二": 2,
    "两": 2,
    "三": 3,
    "四": 4,
    "五": 5,
    "六": 6,
    "七": 7,
    "八": 8,
    "九": 9,
}
CHINESE_SMALL_UNITS = {"十": 10, "百": 100, "千": 1000}
CHINESE_LARGE_UNITS = {"万": 10000, "亿": 100000000}
CHINESE_SPOKEN_DIGIT_SEQUENCE_RE = re.compile(r"[零〇一二三四五六七八九](?:[\s,，、.。．·\-:：;；!！?？]+[零〇一二三四五六七八九]){2,}")
CHINESE_NUMBER_SEQUENCE_RE = re.compile(r"[零〇一二三四五六七八九十百千万亿两]+")
MATH_FORMULA_SYMBOL_RE = re.compile(r"[0-9×xX*＊+\-=÷/]")
DEFAULT_CONFIDENCE_WEIGHT_MULTIPLIERS = {
    "user_confirmed_real_audio": 3.0,
    "auto_screened_public_subtitle": 2.0,
    "ready_reference": 1.0,
    "other": 0.2,
}
GOLD_REVIEW_LEVELS = {"user_confirmed_real_audio", "auto_screened_public_subtitle"}
CONTEXT_TERM_MODES = {"with_context_terms", "prompt_context"}
NATIVE_HOTWORD_MODES = {"native_hotwords"}
LEGACY_HOTWORD_MODE_ALIASES = {"with_hotwords": "with_context_terms"}
MAX_PYTHON_DP_CELLS = 10_000_000
SCRIPT_NORMALIZED_LANGUAGES = {"yue"}
_OPENCC_T2S = None


def read_text(path: str) -> str:
    with open(path, "r", encoding="utf-8") as f:
        return f.read().strip()


def strip_events(text: str) -> str:
    return BRACKET_EVENT_RE.sub(" ", text)


def normalize_text(text: str, keep_punctuation: bool = False) -> str:
    text = unicodedata.normalize("NFKC", strip_events(text)).lower()
    if not keep_punctuation:
        text = re.sub(r"[^\w\s\u3400-\u4dbf\u4e00-\u9fff\u3040-\u30ff\uac00-\ud7af\u0600-\u06ff.%-]", " ", text)
    text = re.sub(r"\s+", " ", text).strip()
    return text


def chars_for_cer(text: str) -> List[str]:
    text = normalize_text(text)
    text = re.sub(r"\s+", "", text)
    text = re.sub(r"[^\w\u3400-\u4dbf\u4e00-\u9fff\u3040-\u30ff\uac00-\ud7af\u0600-\u06ff.%-]", "", text)
    return list(text)


def words_for_wer(text: str) -> List[str]:
    text = normalize_text(text)
    return re.findall(r"[a-zA-Z]+(?:[-'][a-zA-Z]+)*|\d+(?:[,.]\d+)*(?:st|nd|rd|th)?", text)


def hybrid_tokens(text: str) -> List[str]:
    text = normalize_text(text)
    tokens: List[str] = []
    i = 0
    while i < len(text):
        ch = text[i]
        if ch.isspace():
            i += 1
            continue
        if CJK_RE.match(ch):
            tokens.append(ch)
            i += 1
            continue
        m_num = re.match(r"\d+(?:[,.]\d+)*(?:st|nd|rd|th)?%?", text[i:], re.IGNORECASE)
        if m_num:
            tokens.append(m_num.group(0))
            i += len(m_num.group(0))
            continue
        m_ar = ARABIC_RE.match(text[i:])
        if m_ar:
            tokens.append(m_ar.group(0))
            i += len(m_ar.group(0))
            continue
        m_word = LATIN_WORD_RE.match(text[i:])
        if m_word:
            tokens.append(m_word.group(0))
            i += len(m_word.group(0))
            continue
        i += 1
    return tokens


def levenshtein(a: Sequence[str], b: Sequence[str]) -> int:
    if RapidFuzzLevenshtein is not None:
        return int(RapidFuzzLevenshtein.distance(a, b))

    cells = len(a) * len(b)
    if cells > MAX_PYTHON_DP_CELLS:
        raise RuntimeError(
            "rapidfuzz is required for this long transcript scoring run. "
            f"Fallback Python DP would need {cells:,} edit-distance cells. "
            "Install rapidfuzz in the active Python environment."
        )

    if len(a) < len(b):
        a, b = b, a
    prev = list(range(len(b) + 1))
    for i, ca in enumerate(a, 1):
        curr = [i]
        for j, cb in enumerate(b, 1):
            curr.append(
                min(
                    prev[j] + 1,
                    curr[j - 1] + 1,
                    prev[j - 1] + (0 if ca == cb else 1),
                )
            )
        prev = curr
    return prev[-1]


def edit_distance_backend() -> str:
    return "rapidfuzz" if RapidFuzzLevenshtein is not None else "python_dp"


def error_rate(ref_tokens: Sequence[str], hyp_tokens: Sequence[str]) -> float:
    if not ref_tokens:
        return 0.0 if not hyp_tokens else 1.0
    return levenshtein(ref_tokens, hyp_tokens) / len(ref_tokens)


def metric_tokens(metric: str, text: str) -> List[str]:
    if metric == "cer":
        return chars_for_cer(text)
    if metric == "wer":
        return words_for_wer(text)
    if metric == "hybrid_ter":
        return hybrid_tokens(text)
    raise ValueError(f"unknown metric: {metric}")


def script_normalization_backend() -> str:
    return "opencc_t2s" if opencc is not None else "unavailable"


def traditional_to_simplified(text: str) -> str | None:
    global _OPENCC_T2S
    if opencc is None:
        return None
    if _OPENCC_T2S is None:
        _OPENCC_T2S = opencc.OpenCC("t2s")
    return _OPENCC_T2S.convert(text)


def should_script_normalize(case: dict) -> bool:
    return case.get("language") in SCRIPT_NORMALIZED_LANGUAGES


def strip_japanese_furigana_parentheses(text: str) -> str:
    return JAPANESE_KANA_PAREN_RE.sub("", text)


def orthographic_alias_groups(case: dict) -> List[List[str]]:
    groups = []
    for item in case.get("orthographic_aliases", []) or case.get("acceptable_aliases", []) or []:
        if isinstance(item, dict):
            values = [item.get("canonical", ""), *item.get("aliases", [])]
        elif isinstance(item, (list, tuple)):
            values = list(item)
        else:
            values = []
        normalized_values = []
        seen = set()
        for value in values:
            value = str(value).strip()
            if value and value not in seen:
                normalized_values.append(value)
                seen.add(value)
        if len(normalized_values) >= 2:
            groups.append(normalized_values)
    return groups


def apply_orthographic_aliases(text: str, alias_groups: List[List[str]]) -> str:
    for group in alias_groups:
        canonical = group[0]
        for alias in sorted(group, key=len, reverse=True):
            text = text.replace(alias, canonical)
    return text


def normalize_keyword(keyword: str) -> str:
    return normalize_text(keyword)


def keyword_scores(expected: Iterable[str], reference: str, prediction: str) -> Dict[str, float]:
    ref_norm = normalize_text(reference)
    expected_norm = [
        normalize_keyword(k)
        for k in expected
        if normalize_keyword(k) and normalize_keyword(k) in ref_norm
    ]
    if not expected_norm:
        return {"keyword_recall": 1.0, "keyword_precision": 1.0, "keyword_f1": 1.0, "keyword_matched": 0, "keyword_total": 0}
    pred_norm = normalize_text(prediction)
    matched = sum(1 for kw in expected_norm if kw in pred_norm)
    recall = matched / len(expected_norm)
    # Closed-set keyword precision. False positives require a per-case forbidden list, so v1 keeps this conservative.
    precision = recall
    f1 = 0.0 if precision + recall == 0 else 2 * precision * recall / (precision + recall)
    return {
        "keyword_recall": recall,
        "keyword_precision": precision,
        "keyword_f1": f1,
        "keyword_matched": matched,
        "keyword_total": len(expected_norm),
    }


def hotword_scores(expected: Iterable[str], reference: str, prediction: str) -> Dict[str, object]:
    ref_norm = normalize_text(reference)
    candidates = []
    for term in expected:
        norm = normalize_keyword(term)
        if norm and norm in ref_norm:
            candidates.append((term, norm))
    if not candidates:
        return {
            "hotword_recall": 1.0,
            "hotword_precision": 1.0,
            "hotword_f1": 1.0,
            "hotword_matched": 0,
            "hotword_total": 0,
            "hotword_expected_terms": [],
            "hotword_matched_terms": [],
            "hotword_missing_terms": [],
        }

    pred_norm = normalize_text(prediction)
    matched_terms = [term for term, norm in candidates if norm in pred_norm]
    expected_terms = [term for term, _ in candidates]
    missing_terms = [term for term in expected_terms if term not in matched_terms]
    recall = len(matched_terms) / len(candidates)
    # Closed-set hotword precision. Insertion-style false positives require a per-case forbidden list.
    precision = recall
    f1 = 0.0 if precision + recall == 0 else 2 * precision * recall / (precision + recall)
    return {
        "hotword_recall": recall,
        "hotword_precision": precision,
        "hotword_f1": f1,
        "hotword_matched": len(matched_terms),
        "hotword_total": len(candidates),
        "hotword_expected_terms": expected_terms,
        "hotword_matched_terms": matched_terms,
        "hotword_missing_terms": missing_terms,
    }


def normalize_hotword_mode(mode: str) -> str:
    return LEGACY_HOTWORD_MODE_ALIASES.get(mode, mode)


def term_prompt_mechanism(mode: str) -> str:
    normalized = normalize_hotword_mode(mode)
    if normalized in CONTEXT_TERM_MODES:
        return "prompt_context_terms"
    if normalized in NATIVE_HOTWORD_MODES:
        return "native_hotword_bias"
    return normalized


def quality_flags(hallucination: Dict[str, float], hotword_mode: str) -> Dict[str, object]:
    flags = []
    length_ratio = hallucination.get("length_ratio", 1.0)
    excess_rep_rate = hallucination.get("excess_repetition_rate", 0.0)
    repetition_score = hallucination.get("repetition_score", 1.0)

    if length_ratio < 0.80:
        flags.append("under_transcription")
    elif length_ratio > 1.30:
        flags.append("over_transcription")
    if excess_rep_rate > 0.08 or repetition_score < 0.70:
        flags.append("excess_repetition")

    normalized_mode = normalize_hotword_mode(hotword_mode)
    context_prompt_failure = normalized_mode in CONTEXT_TERM_MODES and bool(flags)
    if context_prompt_failure:
        flags.append("context_prompt_instability")

    return {
        "quality_flags": flags,
        "context_prompt_failure": context_prompt_failure,
    }


def parse_chinese_number(token: str) -> int | None:
    if not token or token == "点":
        return None
    if token == "一两":
        return None
    if all(ch in CHINESE_DIGITS for ch in token):
        return int("".join(str(CHINESE_DIGITS[ch]) for ch in token))
    if not all(ch in CHINESE_DIGITS or ch in CHINESE_SMALL_UNITS or ch in CHINESE_LARGE_UNITS for ch in token):
        return None

    total = 0
    section = 0
    number = 0
    seen_value = False
    for ch in token:
        if ch in CHINESE_DIGITS:
            number = CHINESE_DIGITS[ch]
            seen_value = True
        elif ch in CHINESE_SMALL_UNITS:
            section += (number or 1) * CHINESE_SMALL_UNITS[ch]
            number = 0
            seen_value = True
        elif ch in CHINESE_LARGE_UNITS:
            section += number
            total += (section or 1) * CHINESE_LARGE_UNITS[ch]
            section = 0
            number = 0
            seen_value = True
    if not seen_value:
        return None
    return total + section + number


def normalize_number_token(token: str) -> str | None:
    token = token.lower().replace(",", "")
    if token == "点":
        return None
    mixed = parse_mixed_chinese_unit_number(token)
    if mixed is not None:
        return mixed
    chinese_decimal = parse_chinese_decimal_number(token)
    if chinese_decimal is not None:
        return chinese_decimal
    if re.fullmatch(r"\d+(?:\.\d+)?(?:st|nd|rd|th)?%?", token, re.IGNORECASE):
        return token
    parsed = parse_chinese_number(token)
    if parsed is not None:
        return str(parsed)
    return token


def parse_mixed_chinese_unit_number(token: str) -> str | None:
    if not re.search(r"\d.*[万亿]|[万亿].*\d", token):
        return None
    parts = re.findall(r"\d+(?:\.\d+)?|[万亿]", token)
    if not parts:
        return None
    total = 0.0
    i = 0
    while i < len(parts):
        part = parts[i]
        if not re.fullmatch(r"\d+(?:\.\d+)?", part):
            return None
        value = float(part)
        if i + 1 < len(parts) and parts[i + 1] in CHINESE_LARGE_UNITS:
            total += value * CHINESE_LARGE_UNITS[parts[i + 1]]
            i += 2
        else:
            total += value
            i += 1
    if total.is_integer():
        return str(int(total))
    return str(total)


def parse_chinese_decimal_number(token: str) -> str | None:
    if "点" not in token:
        return None
    integer_part, decimal_part = token.split("点", 1)
    if not decimal_part or any(ch not in CHINESE_DIGITS for ch in decimal_part):
        return None
    integer = parse_chinese_number(integer_part) if integer_part else 0
    if integer is None:
        return None
    decimal = "".join(str(CHINESE_DIGITS[ch]) for ch in decimal_part)
    return f"{integer}.{decimal}"


def normalize_spoken_digit_sequences(text: str) -> str:
    def replace(match: re.Match[str]) -> str:
        return "".join(ch for ch in match.group(0) if ch in CHINESE_DIGITS)

    return CHINESE_SPOKEN_DIGIT_SEQUENCE_RE.sub(replace, text)


def case_math_formula_itn_applicable(case: dict, reference: str) -> bool:
    formatting = case.get("formatting", {}) or {}
    domains = formatting.get("itn_domains") or []
    if formatting.get("itn_domain") == "math_formula" or "math_formula" in domains:
        return True
    keyword_blob = " ".join(str(item) for item in case.get("keywords", []))
    return bool(
        case.get("scenario") == "student_lecture"
        and case.get("numbers")
        and (MATH_FORMULA_SYMBOL_RE.search(reference) or MATH_FORMULA_SYMBOL_RE.search(keyword_blob))
    )


def normalize_math_formula_text(text: str) -> str:
    text = unicodedata.normalize("NFKC", strip_events(text)).lower()
    text = text.replace("＊", "×").replace("*", "×").replace("x", "×")

    def replace_number(match: re.Match[str]) -> str:
        parsed = parse_chinese_number(match.group(0))
        return str(parsed) if parsed is not None else match.group(0)

    text = CHINESE_NUMBER_SEQUENCE_RE.sub(replace_number, text)
    text = re.sub(r"(?<=\d)\s*乘以?\s*(?=\d)", "×", text)
    text = re.sub(r"(?<=再)\s*乘以?\s*(?=\d)", "×", text)
    text = re.sub(r"(?<=\d)\s*加上?\s*(?=\d)", "+", text)
    text = re.sub(r"(?<=\d)\s*减去?\s*(?=\d)", "-", text)
    text = re.sub(r"(?<=\d)\s*除以?\s*(?=\d)", "÷", text)
    text = re.sub(r"\s+", "", text)
    return re.sub(r"[^0-9a-z\u3400-\u4dbf\u4e00-\u9fff\u3040-\u30ff\uac00-\ud7af×+\-=÷/%]", "", text)


def math_formula_itn_scores(case: dict, reference: str, prediction: str, raw_error_rate: float) -> Dict[str, object]:
    if not case_math_formula_itn_applicable(case, reference):
        return {
            "math_formula_itn_evaluation": "not_applicable",
            "math_normalized_error_rate": None,
            "math_itn_format_gap": None,
            "math_itn_format_score_10": None,
        }
    ref_norm = normalize_math_formula_text(reference)
    pred_norm = normalize_math_formula_text(prediction)
    norm_error = error_rate(list(ref_norm), list(pred_norm))
    format_gap = max(0.0, raw_error_rate - norm_error)
    # This is a formatting diagnostic, not an ASR content score. A 20-point CER gap
    # from written/spoken math-form mismatch maps to zero on this 0-10 scale.
    format_score = 10.0 * max(0.0, 1.0 - min(1.0, format_gap / 0.20))
    return {
        "math_formula_itn_evaluation": "scored",
        "math_normalized_error_rate": norm_error,
        "math_itn_format_gap": format_gap,
        "math_itn_format_score_10": format_score,
        "math_normalized_reference": ref_norm,
        "math_normalized_prediction": pred_norm,
    }


def extract_numbers(text: str) -> List[str]:
    norm = normalize_spoken_digit_sequences(normalize_text(text, keep_punctuation=True))
    numbers = []
    for match in NUMBER_RE.finditer(norm):
        normalized = normalize_number_token(match.group(0))
        if normalized:
            numbers.append(normalized)
    return numbers


def f1_from_counters(ref_items: List[str], hyp_items: List[str]) -> Tuple[float, float, float]:
    if not ref_items and not hyp_items:
        return 1.0, 1.0, 1.0
    if not ref_items or not hyp_items:
        return 0.0, 0.0, 0.0
    ref_counter = Counter(ref_items)
    hyp_counter = Counter(hyp_items)
    true_positive = sum((ref_counter & hyp_counter).values())
    precision = true_positive / max(1, sum(hyp_counter.values()))
    recall = true_positive / max(1, sum(ref_counter.values()))
    f1 = 0.0 if precision + recall == 0 else 2 * precision * recall / (precision + recall)
    return precision, recall, f1


def is_plain_numeric_form(value: str) -> bool:
    return bool(re.fullmatch(r"\d+(?:\.\d+)?(?:st|nd|rd|th)?%?", value, re.IGNORECASE))


def number_target_forms(value: object) -> Tuple[str, List[str]]:
    display = str(value).strip()
    forms = []
    normalized_display = normalize_text(display)
    if normalized_display:
        forms.append(normalized_display)
    for number in extract_numbers(display):
        if number and number not in forms:
            forms.append(number)
    return display, forms


def form_matches(form: str, normalized_text_value: str, extracted_numbers: set[str]) -> bool:
    if is_plain_numeric_form(form):
        return form in extracted_numbers
    return form in normalized_text_value or form in extracted_numbers


def compact_normalized_text(text: str) -> str:
    return re.sub(r"\s+", "", normalize_text(text))


def best_substring_similarity(needle: str, haystack: str) -> float:
    needle = re.sub(r"\s+", "", normalize_text(needle))
    haystack = re.sub(r"\s+", "", normalize_text(haystack))
    if not needle or not haystack:
        return 0.0
    if needle in haystack:
        return 1.0

    best = 0.0
    needle_len = len(needle)
    min_len = max(1, needle_len - 1)
    max_len = min(len(haystack), needle_len + 1)
    for window_len in range(min_len, max_len + 1):
        for start in range(0, len(haystack) - window_len + 1):
            candidate = haystack[start : start + window_len]
            best = max(best, SequenceMatcher(None, needle, candidate).ratio())
    return best


def number_target_match_score(forms: List[str], hyp_text: str, hyp_extracted: set[str]) -> float:
    if any(form_matches(form, hyp_text, hyp_extracted) for form in forms):
        return 1.0

    # Give partial credit for near misses inside number-like phrases. This avoids
    # turning one digit/Chinese-number character error into a full 10-point loss.
    scores = []
    for form in forms:
        scores.append(best_substring_similarity(form, hyp_text))
        for extracted in hyp_extracted:
            scores.append(SequenceMatcher(None, form, extracted).ratio())
    return max(scores) if scores else 0.0


def number_scores(reference: str, prediction: str, manifest_numbers: Iterable[str]) -> Dict[str, float]:
    targets = [number_target_forms(item) for item in manifest_numbers]
    targets = [(display, forms) for display, forms in targets if display and forms]
    if not targets:
        return {
            "number_precision": 1.0,
            "number_recall": 1.0,
            "number_f1": 1.0,
            "reference_numbers": [],
            "prediction_numbers": [],
        }

    ref_text = normalize_text(reference)
    hyp_text = normalize_text(prediction)
    ref_extracted = set(extract_numbers(reference))
    hyp_extracted = set(extract_numbers(prediction))

    ref_numbers = []
    hyp_numbers = []
    scored_targets = []
    for display, forms in targets:
        if not any(form_matches(form, ref_text, ref_extracted) for form in forms):
            continue
        ref_numbers.append(display)
        score = number_target_match_score(forms, hyp_text, hyp_extracted)
        scored_targets.append({"target": display, "score": score})
        if score >= 0.999:
            hyp_numbers.append(display)

    ref_numbers = sorted(set(ref_numbers))
    hyp_numbers = sorted(set(hyp_numbers))
    if scored_targets:
        f1 = sum(item["score"] for item in scored_targets) / len(scored_targets)
        precision = f1
        recall = f1
    else:
        precision, recall, f1 = f1_from_counters(ref_numbers, hyp_numbers)
    return {
        "number_precision": precision,
        "number_recall": recall,
        "number_f1": f1,
        "reference_numbers": ref_numbers,
        "prediction_numbers": hyp_numbers,
        "number_target_scores": scored_targets,
    }


def punctuation_scores(reference: str, prediction: str) -> Dict[str, float]:
    ref_punct = PUNCT_RE.findall(reference)
    hyp_punct = PUNCT_RE.findall(prediction)
    precision, recall, f1 = f1_from_counters(ref_punct, hyp_punct)
    return {
        "punctuation_evaluation": "scored",
        "punctuation_precision": precision,
        "punctuation_recall": recall,
        "punctuation_f1": f1,
    }


def repetition_rate(tokens: Sequence[str], n: int = 4) -> float:
    if len(tokens) < n * 2:
        return 0.0
    grams = [tuple(tokens[i : i + n]) for i in range(len(tokens) - n + 1)]
    counts = Counter(grams)
    repeated = sum(count - 1 for count in counts.values() if count > 1)
    return repeated / max(1, len(grams))


def hallucination_scores(reference: str, prediction: str, metric: str) -> Dict[str, float]:
    ref_tokens = metric_tokens(metric, reference)
    hyp_tokens = metric_tokens(metric, prediction)
    ref_len = len(ref_tokens)
    hyp_len = len(hyp_tokens)
    if ref_len == 0:
        length_ratio = math.inf if hyp_len else 1.0
        length_score = 1.0 if hyp_len == 0 else max(0.0, 1.0 - min(1.0, hyp_len / 50))
    else:
        length_ratio = hyp_len / ref_len
        if 0.6 <= length_ratio <= 1.6:
            length_score = 1.0
        elif length_ratio < 0.6:
            length_score = max(0.0, length_ratio / 0.6)
        else:
            length_score = max(0.0, 1.0 - min(1.0, (length_ratio - 1.6) / 1.4))
    ref_rep_rate = repetition_rate(ref_tokens)
    rep_rate = repetition_rate(hyp_tokens)
    excess_rep_rate = max(0.0, rep_rate - ref_rep_rate)
    repetition_score = max(0.0, 1.0 - min(1.0, excess_rep_rate * 4))
    hallucination_score = 0.65 * length_score + 0.35 * repetition_score
    return {
        "reference_token_count": ref_len,
        "prediction_token_count": hyp_len,
        "length_ratio": length_ratio,
        "length_score": length_score,
        "reference_repetition_rate": ref_rep_rate,
        "repetition_rate": rep_rate,
        "excess_repetition_rate": excess_rep_rate,
        "repetition_score": repetition_score,
        "hallucination_score": hallucination_score,
    }


def confidence_weight_tier(case: dict) -> str:
    reference = case.get("reference", {})
    if reference.get("review_level") in GOLD_REVIEW_LEVELS:
        return reference.get("review_level")
    if reference.get("status") == "ready":
        return "ready_reference"
    return "other"


def is_gold_case(case: dict) -> bool:
    reference = case.get("reference", {})
    return reference.get("status") == "ready" and reference.get("review_level") in GOLD_REVIEW_LEVELS


def include_case_for_scope(case: dict, scope: str) -> bool:
    reference = case.get("reference", {})
    if scope == "all":
        return True
    if scope == "ready":
        return reference.get("status") == "ready"
    if scope == "gold":
        return is_gold_case(case)
    raise ValueError(f"unknown scope: {scope}")


def confidence_weight_multiplier(case: dict, policy: dict | None = None) -> float:
    tier = confidence_weight_tier(case)
    configured = ((policy or {}).get("case_weighting", {}) or {}).get("confidence_weight_multipliers", {})
    multipliers = {**DEFAULT_CONFIDENCE_WEIGHT_MULTIPLIERS, **configured}
    return float(multipliers.get(tier, DEFAULT_CONFIDENCE_WEIGHT_MULTIPLIERS["other"]))


def text_score_90_from_components(
    primary_error_rate: float,
    keyword_f1: float,
    number_f1: float,
    hallucination_score: float,
    punctuation_f1: float,
) -> float:
    content_points = 50.0 * max(0.0, 1.0 - min(1.0, primary_error_rate))
    keyword_points = 15.0 * keyword_f1
    number_points = 10.0 * number_f1
    hallucination_points = 10.0 * hallucination_score
    punctuation_points = 5.0 * punctuation_f1
    return content_points + keyword_points + number_points + hallucination_points + punctuation_points


def script_normalized_scores(
    case: dict,
    reference: str,
    prediction: str,
    metric: str,
    effective_weight: float,
) -> dict:
    if not should_script_normalize(case):
        return {
            "script_normalization": "not_applicable",
            "script_normalized_primary_error_rate": None,
            "script_normalized_text_score_90": None,
            "script_normalized_weighted_text_score_90": None,
            "script_normalized_delta_text_score_90": None,
        }

    normalized_reference = traditional_to_simplified(reference)
    normalized_prediction = traditional_to_simplified(prediction)
    if normalized_reference is None or normalized_prediction is None:
        return {
            "script_normalization": "unavailable_opencc_t2s",
            "script_normalized_primary_error_rate": None,
            "script_normalized_text_score_90": None,
            "script_normalized_weighted_text_score_90": None,
            "script_normalized_delta_text_score_90": None,
        }

    ref_tokens = metric_tokens(metric, normalized_reference)
    hyp_tokens = metric_tokens(metric, normalized_prediction)
    err = error_rate(ref_tokens, hyp_tokens)
    keyword = keyword_scores(case.get("keywords", []), normalized_reference, normalized_prediction)
    numbers = number_scores(normalized_reference, normalized_prediction, case.get("numbers", []))
    if case.get("reference", {}).get("punctuation") == "none":
        punct_f1 = 1.0
    else:
        punct_f1 = punctuation_scores(normalized_reference, normalized_prediction)["punctuation_f1"]
    hallucination = hallucination_scores(normalized_reference, normalized_prediction, metric)
    score = text_score_90_from_components(
        err,
        float(keyword["keyword_f1"]),
        float(numbers["number_f1"]),
        float(hallucination["hallucination_score"]),
        float(punct_f1),
    )
    return {
        "script_normalization": "opencc_t2s",
        "script_normalized_primary_error_rate": err,
        "script_normalized_text_score_90": score,
        "script_normalized_weighted_text_score_90": score * effective_weight,
        "script_normalized_delta_text_score_90": None,
    }


def japanese_diagnostic_score(
    case: dict,
    reference: str,
    prediction: str,
    metric: str,
    effective_weight: float,
    *,
    apply_furigana_strip: bool,
    apply_aliases: bool,
) -> dict:
    if case.get("language") != "ja":
        return {}

    diagnostic_reference = reference
    diagnostic_prediction = prediction
    if apply_furigana_strip:
        diagnostic_reference = strip_japanese_furigana_parentheses(diagnostic_reference)
        diagnostic_prediction = strip_japanese_furigana_parentheses(diagnostic_prediction)
    alias_groups = orthographic_alias_groups(case) if apply_aliases else []
    if alias_groups:
        diagnostic_reference = apply_orthographic_aliases(diagnostic_reference, alias_groups)
        diagnostic_prediction = apply_orthographic_aliases(diagnostic_prediction, alias_groups)

    ref_tokens = metric_tokens(metric, diagnostic_reference)
    hyp_tokens = metric_tokens(metric, diagnostic_prediction)
    err = error_rate(ref_tokens, hyp_tokens)
    keyword = keyword_scores(case.get("keywords", []), diagnostic_reference, diagnostic_prediction)
    numbers = number_scores(diagnostic_reference, diagnostic_prediction, case.get("numbers", []))
    if case.get("reference", {}).get("punctuation") == "none":
        punct_f1 = 1.0
    else:
        punct_f1 = punctuation_scores(diagnostic_reference, diagnostic_prediction)["punctuation_f1"]
    hallucination = hallucination_scores(diagnostic_reference, diagnostic_prediction, metric)
    score = text_score_90_from_components(
        err,
        float(keyword["keyword_f1"]),
        float(numbers["number_f1"]),
        float(hallucination["hallucination_score"]),
        float(punct_f1),
    )
    return {
        "primary_error_rate": err,
        "text_score_90": score,
        "weighted_text_score_90": score * effective_weight,
        "alias_group_count": len(alias_groups),
    }


def japanese_diagnostic_scores(
    case: dict,
    reference: str,
    prediction: str,
    metric: str,
    effective_weight: float,
    raw_text_score_90: float,
) -> dict:
    if case.get("language") != "ja":
        return {
            "japanese_text_normalization": "not_applicable",
            "japanese_furigana_stripped_primary_error_rate": None,
            "japanese_furigana_stripped_text_score_90": None,
            "japanese_furigana_stripped_delta_text_score_90": None,
            "japanese_orthographic_alias_primary_error_rate": None,
            "japanese_orthographic_alias_text_score_90": None,
            "japanese_orthographic_alias_weighted_text_score_90": None,
            "japanese_orthographic_alias_delta_text_score_90": None,
            "japanese_orthographic_alias_group_count": 0,
        }

    furigana = japanese_diagnostic_score(
        case,
        reference,
        prediction,
        metric,
        effective_weight,
        apply_furigana_strip=True,
        apply_aliases=False,
    )
    alias = japanese_diagnostic_score(
        case,
        reference,
        prediction,
        metric,
        effective_weight,
        apply_furigana_strip=True,
        apply_aliases=True,
    )
    alias_group_count = int(alias["alias_group_count"])
    normalization = "furigana_and_orthographic_aliases" if alias_group_count else "furigana_only"
    return {
        "japanese_text_normalization": normalization,
        "japanese_furigana_stripped_primary_error_rate": furigana["primary_error_rate"],
        "japanese_furigana_stripped_text_score_90": furigana["text_score_90"],
        "japanese_furigana_stripped_delta_text_score_90": furigana["text_score_90"] - raw_text_score_90,
        "japanese_orthographic_alias_primary_error_rate": alias["primary_error_rate"] if alias_group_count else None,
        "japanese_orthographic_alias_text_score_90": alias["text_score_90"] if alias_group_count else None,
        "japanese_orthographic_alias_weighted_text_score_90": alias["weighted_text_score_90"] if alias_group_count else None,
        "japanese_orthographic_alias_delta_text_score_90": (
            alias["text_score_90"] - raw_text_score_90 if alias_group_count else None
        ),
        "japanese_orthographic_alias_group_count": alias_group_count,
    }


def score_case(case: dict, reference: str, prediction: str, hotword_mode: str = "zero_shot", policy: dict | None = None) -> dict:
    hotword_mode = normalize_hotword_mode(hotword_mode)
    metric = case["primary_error_metric"]
    ref_tokens = metric_tokens(metric, reference)
    hyp_tokens = metric_tokens(metric, prediction)
    err = error_rate(ref_tokens, hyp_tokens)
    base_weight = float(case["weight"])
    confidence_tier = confidence_weight_tier(case)
    confidence_multiplier = confidence_weight_multiplier(case, policy)
    effective_weight = base_weight * confidence_multiplier

    keyword = keyword_scores(case.get("keywords", []), reference, prediction)
    hotword = hotword_scores(case.get("hotwords", []), reference, prediction)
    context_terms = {
        "context_term_recall": hotword["hotword_recall"],
        "context_term_precision": hotword["hotword_precision"],
        "context_term_f1": hotword["hotword_f1"],
        "context_term_matched": hotword["hotword_matched"],
        "context_term_total": hotword["hotword_total"],
        "context_term_expected_terms": hotword["hotword_expected_terms"],
        "context_term_matched_terms": hotword["hotword_matched_terms"],
        "context_term_missing_terms": hotword["hotword_missing_terms"],
    }
    numbers = number_scores(reference, prediction, case.get("numbers", []))
    math_itn = math_formula_itn_scores(case, reference, prediction, err)
    if case.get("reference", {}).get("punctuation") == "none":
        punct = {
            "punctuation_evaluation": "ignored_reference_has_no_punctuation",
            "punctuation_precision": 1.0,
            "punctuation_recall": 1.0,
            "punctuation_f1": 1.0,
        }
    else:
        punct = punctuation_scores(reference, prediction)
    hallucination = hallucination_scores(reference, prediction, metric)
    guardrails = quality_flags(hallucination, hotword_mode)

    text_score_90 = text_score_90_from_components(
        err,
        float(keyword["keyword_f1"]),
        float(numbers["number_f1"]),
        float(hallucination["hallucination_score"]),
        float(punct["punctuation_f1"]),
    )
    script_normalized = script_normalized_scores(case, reference, prediction, metric, effective_weight)
    if script_normalized["script_normalized_text_score_90"] is not None:
        script_normalized["script_normalized_delta_text_score_90"] = (
            float(script_normalized["script_normalized_text_score_90"]) - text_score_90
        )
    japanese_diagnostics = japanese_diagnostic_scores(case, reference, prediction, metric, effective_weight, text_score_90)

    return {
        "case_id": case["case_id"],
        "scenario": case["scenario"],
        "language": case["language"],
        "priority": case["priority"],
        "weight": base_weight,
        "confidence_weight_tier": confidence_tier,
        "confidence_weight_multiplier": confidence_multiplier,
        "effective_weight": effective_weight,
        "hotword_mode": hotword_mode,
        "term_prompt_mechanism": term_prompt_mechanism(hotword_mode),
        "context_terms_source_field": "hotwords",
        "primary_error_metric": metric,
        "primary_error_rate": err,
        "text_score_90": text_score_90,
        "weighted_text_score_90": text_score_90 * effective_weight,
        **script_normalized,
        **japanese_diagnostics,
        **keyword,
        **hotword,
        **context_terms,
        **numbers,
        **math_itn,
        **punct,
        **hallucination,
        **guardrails,
    }


def result_effective_weight(result: dict) -> float:
    return float(result.get("effective_weight", result["weight"]))


def load_manifest(path: str) -> dict:
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def find_case(manifest: dict, case_id: str) -> dict:
    for case in manifest["cases"]:
        if case["case_id"] == case_id:
            return case
    raise KeyError(case_id)


def ref_path_for(manifest_path: str, case: dict) -> str:
    root = os.path.abspath(os.path.join(os.path.dirname(manifest_path), ".."))
    return os.path.join(root, case["reference"]["path"])


def aggregate(results: List[dict]) -> dict:
    if not results:
        return {
            "case_count": 0,
            "weighted_text_score_90": None,
            "unweighted_text_score_90": None,
            "script_normalized_case_count": 0,
            "weighted_script_normalized_text_score_90": None,
            "unweighted_script_normalized_text_score_90": None,
            "weighted_script_normalized_delta_text_score_90": None,
            "unweighted_script_normalized_delta_text_score_90": None,
            "japanese_furigana_stripped_case_count": 0,
            "weighted_japanese_furigana_stripped_text_score_90": None,
            "weighted_japanese_furigana_stripped_delta_text_score_90": None,
            "japanese_orthographic_alias_case_count": 0,
            "weighted_japanese_orthographic_alias_text_score_90": None,
            "weighted_japanese_orthographic_alias_delta_text_score_90": None,
            "hotword_case_count": 0,
            "weighted_hotword_f1": None,
            "unweighted_hotword_f1": None,
        }
    total_weight = sum(result_effective_weight(r) for r in results)
    weighted = sum(float(r["weighted_text_score_90"]) for r in results) / max(total_weight, 1e-9)
    unweighted = sum(float(r["text_score_90"]) for r in results) / len(results)
    script_normalized_results = [r for r in results if r.get("script_normalized_text_score_90") is not None]
    if script_normalized_results:
        script_normalized_weight = sum(result_effective_weight(r) for r in script_normalized_results)
        weighted_script_normalized = (
            sum(float(r["script_normalized_text_score_90"]) * result_effective_weight(r) for r in script_normalized_results)
            / max(script_normalized_weight, 1e-9)
        )
        unweighted_script_normalized = (
            sum(float(r["script_normalized_text_score_90"]) for r in script_normalized_results)
            / len(script_normalized_results)
        )
        weighted_script_normalized_delta = (
            sum(float(r["script_normalized_delta_text_score_90"]) * result_effective_weight(r) for r in script_normalized_results)
            / max(script_normalized_weight, 1e-9)
        )
        unweighted_script_normalized_delta = (
            sum(float(r["script_normalized_delta_text_score_90"]) for r in script_normalized_results)
            / len(script_normalized_results)
        )
    else:
        weighted_script_normalized = None
        unweighted_script_normalized = None
        weighted_script_normalized_delta = None
        unweighted_script_normalized_delta = None
    japanese_furigana_results = [r for r in results if r.get("japanese_furigana_stripped_text_score_90") is not None]
    if japanese_furigana_results:
        japanese_furigana_weight = sum(result_effective_weight(r) for r in japanese_furigana_results)
        weighted_japanese_furigana = (
            sum(float(r["japanese_furigana_stripped_text_score_90"]) * result_effective_weight(r) for r in japanese_furigana_results)
            / max(japanese_furigana_weight, 1e-9)
        )
        weighted_japanese_furigana_delta = (
            sum(float(r["japanese_furigana_stripped_delta_text_score_90"]) * result_effective_weight(r) for r in japanese_furigana_results)
            / max(japanese_furigana_weight, 1e-9)
        )
    else:
        weighted_japanese_furigana = None
        weighted_japanese_furigana_delta = None
    japanese_alias_results = [r for r in results if r.get("japanese_orthographic_alias_text_score_90") is not None]
    if japanese_alias_results:
        japanese_alias_weight = sum(result_effective_weight(r) for r in japanese_alias_results)
        weighted_japanese_alias = (
            sum(float(r["japanese_orthographic_alias_text_score_90"]) * result_effective_weight(r) for r in japanese_alias_results)
            / max(japanese_alias_weight, 1e-9)
        )
        weighted_japanese_alias_delta = (
            sum(float(r["japanese_orthographic_alias_delta_text_score_90"]) * result_effective_weight(r) for r in japanese_alias_results)
            / max(japanese_alias_weight, 1e-9)
        )
    else:
        weighted_japanese_alias = None
        weighted_japanese_alias_delta = None
    hotword_results = [r for r in results if int(r.get("hotword_total", 0)) > 0]
    if hotword_results:
        hotword_weight = sum(result_effective_weight(r) for r in hotword_results)
        weighted_hotword = sum(float(r["hotword_f1"]) * result_effective_weight(r) for r in hotword_results) / max(hotword_weight, 1e-9)
        unweighted_hotword = sum(float(r["hotword_f1"]) for r in hotword_results) / len(hotword_results)
    else:
        weighted_hotword = None
        unweighted_hotword = None
    by_scenario = {}
    for r in results:
        bucket = by_scenario.setdefault(
            r["scenario"],
            {
                "weight": 0.0,
                "raw_weight": 0.0,
                "score_sum": 0.0,
                "case_count": 0,
                "hotword_weight": 0.0,
                "raw_hotword_weight": 0.0,
                "hotword_score_sum": 0.0,
                "hotword_case_count": 0,
                "script_normalized_weight": 0.0,
                "script_normalized_score_sum": 0.0,
                "script_normalized_delta_sum": 0.0,
                "script_normalized_case_count": 0,
                "japanese_furigana_weight": 0.0,
                "japanese_furigana_score_sum": 0.0,
                "japanese_furigana_delta_sum": 0.0,
                "japanese_furigana_case_count": 0,
                "japanese_alias_weight": 0.0,
                "japanese_alias_score_sum": 0.0,
                "japanese_alias_delta_sum": 0.0,
                "japanese_alias_case_count": 0,
            },
        )
        effective_weight = result_effective_weight(r)
        bucket["weight"] += effective_weight
        bucket["raw_weight"] += float(r["weight"])
        bucket["score_sum"] += float(r["weighted_text_score_90"])
        bucket["case_count"] += 1
        if int(r.get("hotword_total", 0)) > 0:
            bucket["hotword_weight"] += effective_weight
            bucket["raw_hotword_weight"] += float(r["weight"])
            bucket["hotword_score_sum"] += float(r["hotword_f1"]) * effective_weight
            bucket["hotword_case_count"] += 1
        if r.get("script_normalized_text_score_90") is not None:
            bucket["script_normalized_weight"] += effective_weight
            bucket["script_normalized_score_sum"] += float(r["script_normalized_text_score_90"]) * effective_weight
            bucket["script_normalized_delta_sum"] += float(r["script_normalized_delta_text_score_90"]) * effective_weight
            bucket["script_normalized_case_count"] += 1
        if r.get("japanese_furigana_stripped_text_score_90") is not None:
            bucket["japanese_furigana_weight"] += effective_weight
            bucket["japanese_furigana_score_sum"] += float(r["japanese_furigana_stripped_text_score_90"]) * effective_weight
            bucket["japanese_furigana_delta_sum"] += float(r["japanese_furigana_stripped_delta_text_score_90"]) * effective_weight
            bucket["japanese_furigana_case_count"] += 1
        if r.get("japanese_orthographic_alias_text_score_90") is not None:
            bucket["japanese_alias_weight"] += effective_weight
            bucket["japanese_alias_score_sum"] += float(r["japanese_orthographic_alias_text_score_90"]) * effective_weight
            bucket["japanese_alias_delta_sum"] += float(r["japanese_orthographic_alias_delta_text_score_90"]) * effective_weight
            bucket["japanese_alias_case_count"] += 1
    for bucket in by_scenario.values():
        bucket["weighted_text_score_90"] = bucket["score_sum"] / max(bucket["weight"], 1e-9)
        if bucket["script_normalized_case_count"]:
            bucket["weighted_script_normalized_text_score_90"] = (
                bucket["script_normalized_score_sum"] / max(bucket["script_normalized_weight"], 1e-9)
            )
            bucket["weighted_script_normalized_delta_text_score_90"] = (
                bucket["script_normalized_delta_sum"] / max(bucket["script_normalized_weight"], 1e-9)
            )
        else:
            bucket["weighted_script_normalized_text_score_90"] = None
            bucket["weighted_script_normalized_delta_text_score_90"] = None
        if bucket["japanese_furigana_case_count"]:
            bucket["weighted_japanese_furigana_stripped_text_score_90"] = (
                bucket["japanese_furigana_score_sum"] / max(bucket["japanese_furigana_weight"], 1e-9)
            )
            bucket["weighted_japanese_furigana_stripped_delta_text_score_90"] = (
                bucket["japanese_furigana_delta_sum"] / max(bucket["japanese_furigana_weight"], 1e-9)
            )
        else:
            bucket["weighted_japanese_furigana_stripped_text_score_90"] = None
            bucket["weighted_japanese_furigana_stripped_delta_text_score_90"] = None
        if bucket["japanese_alias_case_count"]:
            bucket["weighted_japanese_orthographic_alias_text_score_90"] = (
                bucket["japanese_alias_score_sum"] / max(bucket["japanese_alias_weight"], 1e-9)
            )
            bucket["weighted_japanese_orthographic_alias_delta_text_score_90"] = (
                bucket["japanese_alias_delta_sum"] / max(bucket["japanese_alias_weight"], 1e-9)
            )
        else:
            bucket["weighted_japanese_orthographic_alias_text_score_90"] = None
            bucket["weighted_japanese_orthographic_alias_delta_text_score_90"] = None
        if bucket["hotword_case_count"]:
            bucket["weighted_hotword_f1"] = bucket["hotword_score_sum"] / max(bucket["hotword_weight"], 1e-9)
        else:
            bucket["weighted_hotword_f1"] = None
        del bucket["score_sum"]
        del bucket["hotword_score_sum"]
        del bucket["script_normalized_score_sum"]
        del bucket["script_normalized_delta_sum"]
        del bucket["japanese_furigana_score_sum"]
        del bucket["japanese_furigana_delta_sum"]
        del bucket["japanese_alias_score_sum"]
        del bucket["japanese_alias_delta_sum"]
    return {
        "case_count": len(results),
        "weighted_text_score_90": weighted,
        "unweighted_text_score_90": unweighted,
        "script_normalized_case_count": len(script_normalized_results),
        "weighted_script_normalized_text_score_90": weighted_script_normalized,
        "unweighted_script_normalized_text_score_90": unweighted_script_normalized,
        "weighted_script_normalized_delta_text_score_90": weighted_script_normalized_delta,
        "unweighted_script_normalized_delta_text_score_90": unweighted_script_normalized_delta,
        "japanese_furigana_stripped_case_count": len(japanese_furigana_results),
        "weighted_japanese_furigana_stripped_text_score_90": weighted_japanese_furigana,
        "weighted_japanese_furigana_stripped_delta_text_score_90": weighted_japanese_furigana_delta,
        "japanese_orthographic_alias_case_count": len(japanese_alias_results),
        "weighted_japanese_orthographic_alias_text_score_90": weighted_japanese_alias,
        "weighted_japanese_orthographic_alias_delta_text_score_90": weighted_japanese_alias_delta,
        "hotword_case_count": len(hotword_results),
        "weighted_hotword_f1": weighted_hotword,
        "unweighted_hotword_f1": unweighted_hotword,
        "by_scenario": by_scenario,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Score ASR prediction text against local references.")
    parser.add_argument("--manifest", default="data/gold_manifest.v1.json")
    parser.add_argument("--case", dest="case_id")
    parser.add_argument("--prediction", help="Prediction text file for --case mode")
    parser.add_argument("--pred-dir", help="Directory containing <case_id>.txt files")
    parser.add_argument(
        "--hotword-mode",
        choices=[
            "zero_shot",
            "with_context_terms",
            "prompt_context",
            "native_hotwords",
            "with_hotwords",
            "not_supported",
            "post_correction",
        ],
        default="zero_shot",
        help="How the prediction was generated; scores are comparable only within the same mode.",
    )
    parser.add_argument("--out", help="Write JSON result to this path")
    parser.add_argument(
        "--print-full",
        action="store_true",
        help="Print the full JSON payload to stdout even when --out is used.",
    )
    parser.add_argument(
        "--scope",
        choices=["gold", "ready", "all"],
        default="gold",
        help="Cases included in --pred-dir aggregate mode. Defaults to strict Gold only.",
    )
    args = parser.parse_args()

    manifest = load_manifest(args.manifest)

    results: List[dict] = []
    skipped: List[dict] = []

    if args.case_id:
        if not args.prediction:
            parser.error("--prediction is required with --case")
        case = find_case(manifest, args.case_id)
        reference = read_text(ref_path_for(args.manifest, case))
        prediction = read_text(args.prediction)
        results.append(score_case(case, reference, prediction, args.hotword_mode, manifest.get("policy", {})))
    else:
        if not args.pred_dir:
            parser.error("either --case/--prediction or --pred-dir is required")
        for case in manifest["cases"]:
            ref = case["reference"]
            if not include_case_for_scope(case, args.scope):
                skipped.append({"case_id": case["case_id"], "reason": f"outside_scope_{args.scope}"})
                continue
            if ref.get("status") != "ready":
                skipped.append({"case_id": case["case_id"], "reason": "reference_not_ready"})
                continue
            pred_path = os.path.join(args.pred_dir, f"{case['case_id']}.txt")
            if not os.path.exists(pred_path):
                skipped.append({"case_id": case["case_id"], "reason": "prediction_missing"})
                continue
            reference = read_text(ref_path_for(args.manifest, case))
            prediction = read_text(pred_path)
            results.append(score_case(case, reference, prediction, args.hotword_mode, manifest.get("policy", {})))

    output = {
        "manifest": args.manifest,
        "prediction": args.prediction,
        "pred_dir": args.pred_dir,
        "hotword_mode": args.hotword_mode,
        "scope": args.scope if not args.case_id else "single_case",
        "edit_distance_backend": edit_distance_backend(),
        "script_normalization_backend": script_normalization_backend(),
        "aggregate": aggregate(results),
        "cases": results,
        "skipped": skipped,
        "performance_fields_expected_elsewhere": [
            "rtf",
            "peak_memory_gb",
            "model_disk_gb",
            "first_token_latency_ms",
            "install_friction",
        ],
    }

    if args.out:
        out_dir = os.path.dirname(args.out)
        if out_dir:
            os.makedirs(out_dir, exist_ok=True)
        with open(args.out, "w", encoding="utf-8") as f:
            json.dump(output, f, ensure_ascii=False, indent=2)
            f.write("\n")
    if args.out and not args.print_full:
        summary = {
            "out": args.out,
            "case_count": output["aggregate"]["case_count"],
            "weighted_text_score_90": output["aggregate"]["weighted_text_score_90"],
            "unweighted_text_score_90": output["aggregate"]["unweighted_text_score_90"],
            "skipped_count": len(skipped),
            "edit_distance_backend": output["edit_distance_backend"],
        }
        print(json.dumps(summary, ensure_ascii=False, indent=2))
    else:
        print(json.dumps(output, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
