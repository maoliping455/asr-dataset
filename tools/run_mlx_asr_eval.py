#!/usr/bin/env python3
import argparse
import inspect
import json
import os
import subprocess
import sys
import time
from pathlib import Path

import psutil
from mlx_audio.stt import load


def read_json(path: str | Path) -> dict:
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def ffprobe_duration(path: str | Path) -> float:
    out = subprocess.check_output(
        [
            "ffprobe",
            "-hide_banner",
            "-loglevel",
            "error",
            "-show_entries",
            "format=duration",
            "-of",
            "default=nk=1:nw=1",
            str(path),
        ],
        text=True,
    ).strip()
    return float(out)


def dir_size_bytes(local: Path) -> int:
    if local.is_file():
        return local.stat().st_size
    total = 0
    for item in local.rglob("*"):
        if item.is_file():
            total += item.stat().st_size
    return total


def model_size_bytes(path: str | Path) -> int | None:
    model_id = str(path)
    candidates = [Path(model_id)]
    if "/" in model_id:
        candidates.append(Path("models") / model_id)
        candidates.append(Path.home() / ".cache" / "huggingface" / "hub" / f"models--{model_id.replace('/', '--')}")
    for local in candidates:
        if local.exists():
            return dir_size_bytes(local)
    return None


def output_text(result) -> str:
    segments = getattr(result, "segments", None)
    if isinstance(result, str):
        return result
    text = getattr(result, "text", None)
    if isinstance(text, str):
        if segments and text.lstrip().startswith(("[", "{", "```json")):
            segment_text = " ".join(
                str(item.get("text", "")).strip()
                for item in segments
                if isinstance(item, dict) and item.get("text")
            ).strip()
            if segment_text:
                return segment_text
        return text
    if isinstance(result, dict):
        segments = result.get("segments")
        if segments:
            segment_text = " ".join(
                str(item.get("text", "")).strip()
                for item in segments
                if isinstance(item, dict) and item.get("text")
            ).strip()
            if segment_text:
                return segment_text
        value = result.get("text") or result.get("transcription") or result.get("result")
        if isinstance(value, str):
            return value
    return str(result)


def normalize_language(lang: str, mode: str = "manifest") -> str | None:
    if mode == "auto":
        return None
    if lang.startswith("zh"):
        return "zh"
    if lang.startswith("en"):
        return "en"
    if mode == "zh-en":
        return None
    if lang.startswith("ja"):
        return "ja"
    return None


def build_system_prompt(args: argparse.Namespace, case_hotwords: list[str]) -> str | None:
    prompt_parts = []
    if args.system_prompt:
        prompt_parts.append(args.system_prompt.strip())
    hotwords = [item.strip() for item in args.hotword if item.strip()]
    if args.hotword_mode == "with_hotwords":
        hotwords.extend(item.strip() for item in case_hotwords if item.strip())
    if hotwords:
        prompt_parts.append("请转写音频。以下是可能出现的专有名词和热词，请优先按这些写法输出：" + "、".join(hotwords) + "。")
    return "\n".join(part for part in prompt_parts if part) or None


def generate_one(model, audio_path: str, language: str | None, args: argparse.Namespace, system_prompt: str | None):
    signature = inspect.signature(model.generate)
    params = signature.parameters
    common_kwargs = {
        "max_tokens": args.max_tokens,
        "verbose": args.verbose,
    }
    if "language" in params:
        common_kwargs["language"] = language
    if "source_lang" in params:
        common_kwargs["source_lang"] = language or args.default_source_lang
    if "target_lang" in params:
        common_kwargs["target_lang"] = language or args.default_target_lang
    if "chunk_duration" in params:
        common_kwargs["chunk_duration"] = args.chunk_duration
    if system_prompt and "system_prompt" in params:
        common_kwargs["system_prompt"] = system_prompt
    if "audio" in signature.parameters:
        return model.generate(audio=audio_path, **common_kwargs), "keyword_audio"
    return model.generate(audio_path, **common_kwargs), "positional_audio"


def main() -> None:
    parser = argparse.ArgumentParser(description="Run MLX-Audio ASR model over local audio manifest.")
    parser.add_argument("--model", required=True)
    parser.add_argument("--model-type", help="Explicit MLX-Audio model_type override for repos without model_type in config.")
    parser.add_argument("--audio-tokenizer-dir")
    parser.add_argument("--audio-manifest", default="data/audio/gold/audio_manifest.json")
    parser.add_argument("--manifest", default="data/gold_manifest.v1.json")
    parser.add_argument("--out-dir", required=True)
    parser.add_argument("--max-cases", type=int)
    parser.add_argument("--case", action="append")
    parser.add_argument("--max-tokens", type=int, default=8192)
    parser.add_argument("--chunk-duration", type=float, default=1200.0)
    parser.add_argument(
        "--language-mode",
        choices=["manifest", "zh-en", "auto"],
        default="manifest",
        help="manifest=use zh/en/ja hints; zh-en=only pass zh/en; auto=never pass a language hint.",
    )
    parser.add_argument("--verbose", action="store_true")
    parser.add_argument("--default-source-lang", default="en", help="Fallback source_lang for models such as Canary.")
    parser.add_argument("--default-target-lang", default="en", help="Fallback target_lang for models such as Canary.")
    parser.add_argument(
        "--hotword-mode",
        choices=["zero_shot", "with_hotwords"],
        default="zero_shot",
        help="zero_shot=do not use manifest hotwords; with_hotwords=pass case.hotwords via system_prompt when supported.",
    )
    parser.add_argument("--hotword", action="append", default=[], help="Global hotword appended to every case prompt.")
    parser.add_argument("--system-prompt", help="Global system prompt for models whose generate() supports system_prompt.")
    args = parser.parse_args()

    out_dir = Path(args.out_dir)
    pred_dir = out_dir / "predictions"
    pred_dir.mkdir(parents=True, exist_ok=True)

    audio_manifest = read_json(args.audio_manifest)
    eval_manifest = read_json(args.manifest)
    case_meta = {case["case_id"]: case for case in eval_manifest["cases"]}
    audio_cases = audio_manifest["cases"]
    if args.case:
        wanted = set(args.case)
        audio_cases = [c for c in audio_cases if c["case_id"] in wanted]
    if args.max_cases:
        audio_cases = audio_cases[: args.max_cases]

    proc = psutil.Process()
    run_started = time.time()
    load_started = time.time()
    load_kwargs = {}
    if args.audio_tokenizer_dir:
        load_kwargs["audio_tokenizer_dir"] = args.audio_tokenizer_dir
    if args.model_type:
        load_kwargs["model_type"] = args.model_type
    model = load(args.model, **load_kwargs)
    load_sec = time.time() - load_started
    peak_rss = proc.memory_info().rss
    generate_style = None

    case_results = []
    total_audio_sec = 0.0
    total_infer_sec = 0.0

    for item in audio_cases:
        case_id = item["case_id"]
        meta = case_meta.get(case_id, {})
        audio_path = item["audio_path"]
        duration = float(item.get("duration_sec") or ffprobe_duration(audio_path))
        language = normalize_language(meta.get("language", ""), args.language_mode)
        case_hotwords = list(meta.get("hotwords", [])) if isinstance(meta.get("hotwords", []), list) else []
        system_prompt = build_system_prompt(args, case_hotwords)
        started = time.time()
        try:
            result, generate_style = generate_one(model, audio_path, language, args, system_prompt)
            text = output_text(result).strip()
            error = None
        except Exception as exc:
            text = ""
            error = repr(exc)
        infer_sec = time.time() - started
        peak_rss = max(peak_rss, proc.memory_info().rss)
        total_audio_sec += duration
        total_infer_sec += infer_sec
        (pred_dir / f"{case_id}.txt").write_text(text + "\n", encoding="utf-8")
        case_results.append(
            {
                "case_id": case_id,
                "audio_path": audio_path,
                "duration_sec": duration,
                "infer_sec": infer_sec,
                "rtf": infer_sec / duration if duration else None,
                "language_arg": language,
                "hotword_mode": args.hotword_mode,
                "hotwords_sent": case_hotwords if args.hotword_mode == "with_hotwords" else [],
                "system_prompt_sent": bool(system_prompt),
                "error": error,
                "prediction_chars": len(text),
            }
        )
        print(f"{case_id}\tduration={duration:.2f}s\tinfer={infer_sec:.2f}s\trtf={(infer_sec / duration if duration else 0):.3f}\terror={error}", flush=True)

    size = model_size_bytes(args.model)
    tokenizer_size = model_size_bytes(args.audio_tokenizer_dir) if args.audio_tokenizer_dir else None
    metrics = {
        "model": args.model,
        "audio_manifest": args.audio_manifest,
        "case_count": len(case_results),
        "load_sec": load_sec,
        "total_audio_sec": total_audio_sec,
        "total_infer_sec": total_infer_sec,
        "overall_rtf": total_infer_sec / total_audio_sec if total_audio_sec else None,
        "peak_rss_gb": peak_rss / (1024**3),
        "wall_sec": time.time() - run_started,
        "generate_style": generate_style,
        "model_type_override": args.model_type,
        "hotword_mode": args.hotword_mode,
        "global_hotwords": args.hotword,
        "system_prompt": args.system_prompt,
        "model_disk_gb": size / (1024**3) if size is not None else None,
        "audio_tokenizer_disk_gb": tokenizer_size / (1024**3) if tokenizer_size is not None else None,
        "pipeline_disk_gb": (sum(v for v in [size, tokenizer_size] if v is not None) / (1024**3)) if size is not None or tokenizer_size is not None else None,
        "cases": case_results,
        "note": "RSS does not fully capture Apple unified GPU memory; use as an approximate process memory signal.",
    }
    (out_dir / "run_metrics.json").write_text(json.dumps(metrics, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(f"pred_dir={pred_dir}")
    print(f"metrics={out_dir / 'run_metrics.json'}")


if __name__ == "__main__":
    main()
