#!/usr/bin/env python3
import argparse
import json
import os
import re
import subprocess
import time
from pathlib import Path
from typing import Any

import psutil
import torch
from funasr import AutoModel


SENSEVOICE_TAG_RE = re.compile(r"<\|[^|]+?\|>")


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
    alias_map = {
        "paraformer-zh": "iic/speech_seaco_paraformer_large_asr_nat-zh-cn-16k-common-vocab8404-pytorch",
        "fsmn-vad": "iic/speech_fsmn_vad_zh-cn-16k-common-pytorch",
        "ct-punc": "iic/punc_ct-transformer_cn-en-common-vocab471067-large",
    }
    candidates = [Path(model_id)]
    resolved_id = alias_map.get(model_id, model_id)
    if "/" in resolved_id:
        candidates.append(Path.home() / ".cache" / "modelscope" / "hub" / "models" / resolved_id)
        candidates.append(Path.home() / ".cache" / "huggingface" / "hub" / f"models--{resolved_id.replace('/', '--')}")
    for local in candidates:
        if local.exists():
            return dir_size_bytes(local)
    return None


def select_device(device: str) -> str:
    if device != "auto":
        return device
    if torch.backends.mps.is_available():
        return "mps"
    return "cpu"


def normalize_language(lang: str, default: str, style: str) -> str:
    if default != "manifest":
        return default
    if style == "funasr-nano":
        if lang.startswith("zh") and lang != "zh-en":
            return "中文"
        if lang.startswith("en"):
            return "英文"
        if lang.startswith("ja"):
            return "日文"
        return "auto"
    if lang.startswith("zh"):
        return "zh"
    if lang.startswith("en"):
        return "en"
    if lang.startswith("ja"):
        return "ja"
    return "auto"


def strip_model_tags(text: str) -> str:
    text = SENSEVOICE_TAG_RE.sub(" ", text)
    return re.sub(r"\s+", " ", text).strip()


def patch_glm_asr_loader(model_id: str) -> bool:
    """Route FunASR's GLM wrapper to the generative HF class.

    FunASR 1.3.12 imports `transformers.AutoModel` inside its GLM wrapper.
    For GLM-ASR, AutoModel resolves to GlmAsrModel, which lacks generate().
    """
    if "glm-asr" not in model_id.lower():
        return False
    from transformers import AutoModel, GlmAsrConfig, GlmAsrForConditionalGeneration

    AutoModel.register(GlmAsrConfig, GlmAsrForConditionalGeneration, exist_ok=True)
    return True


def output_text(result: Any) -> str:
    if isinstance(result, str):
        return strip_model_tags(result)
    if isinstance(result, dict):
        for key in ("text", "transcription", "result"):
            value = result.get(key)
            if isinstance(value, str):
                return strip_model_tags(value)
        return strip_model_tags(str(result))
    if isinstance(result, list):
        parts = []
        for item in result:
            text = output_text(item)
            if text:
                parts.append(text)
        return strip_model_tags(" ".join(parts))
    text = getattr(result, "text", None)
    if isinstance(text, str):
        return strip_model_tags(text)
    return strip_model_tags(str(result))


def build_generate_cfg(args: argparse.Namespace, language: str, case_hotwords: list[str]) -> dict:
    cfg: dict[str, Any] = {
        "batch_size_s": args.batch_size_s,
    }
    if args.generate_batch_size is not None:
        cfg["batch_size"] = args.generate_batch_size
    if args.cache_arg:
        cfg["cache"] = {}
    if args.vad_max_single_segment_time:
        cfg["vad_kwargs"] = {"max_single_segment_time": args.vad_max_single_segment_time}
    if args.use_itn is not None:
        cfg["use_itn"] = args.use_itn
    if args.merge_vad is not None:
        cfg["merge_vad"] = args.merge_vad
    hotwords = []
    if args.hotword:
        hotwords.append(args.hotword)
    if args.hotword_mode == "with_hotwords":
        hotwords.extend(case_hotwords)
    if hotwords:
        cfg["hotword"] = " ".join(hotwords)
    if args.language_arg:
        cfg["language"] = language
    return cfg


def main() -> None:
    parser = argparse.ArgumentParser(description="Run FunASR AutoModel over local audio manifest.")
    parser.add_argument("--model", required=True)
    parser.add_argument("--audio-manifest", default="data/audio/benchmark/audio_manifest.json")
    parser.add_argument("--manifest", default="data/benchmark_manifest.v1.json")
    parser.add_argument("--out-dir", required=True)
    parser.add_argument("--max-cases", type=int)
    parser.add_argument("--case", action="append")
    parser.add_argument("--device", default="auto", choices=["auto", "cpu", "mps", "cuda"])
    parser.add_argument("--language", default="manifest", help="manifest, auto, zh, en, ja, yue, ko, nospeech")
    parser.add_argument("--language-style", default="iso", choices=["iso", "funasr-nano"])
    parser.add_argument("--language-arg", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--batch-size-s", type=int, default=60)
    parser.add_argument("--generate-batch-size", type=int)
    parser.add_argument("--cache-arg", action=argparse.BooleanOptionalAction, default=False)
    parser.add_argument("--use-itn", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--merge-vad", action=argparse.BooleanOptionalAction, default=False)
    parser.add_argument("--vad-model")
    parser.add_argument("--vad-max-single-segment-time", type=int)
    parser.add_argument("--punc-model")
    parser.add_argument("--hub")
    parser.add_argument("--model-revision")
    parser.add_argument("--hotword")
    parser.add_argument(
        "--hotword-mode",
        choices=["zero_shot", "with_hotwords"],
        default="zero_shot",
        help="zero_shot=do not use per-case manifest hotwords; with_hotwords=append case.hotwords to FunASR hotword.",
    )
    parser.add_argument("--trust-remote-code", action=argparse.BooleanOptionalAction, default=False)
    parser.add_argument("--remote-code")
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
    device = select_device(args.device)

    load_kwargs: dict[str, Any] = {
        "model": args.model,
        "device": device,
        "disable_update": True,
    }
    if args.vad_model:
        load_kwargs["vad_model"] = args.vad_model
    if args.punc_model:
        load_kwargs["punc_model"] = args.punc_model
    if args.hub:
        load_kwargs["hub"] = args.hub
    if args.model_revision:
        load_kwargs["model_revision"] = args.model_revision
    if args.trust_remote_code:
        load_kwargs["trust_remote_code"] = True
    if args.remote_code:
        load_kwargs["remote_code"] = args.remote_code

    load_started = time.time()
    glm_loader_patch = patch_glm_asr_loader(args.model)
    model = AutoModel(**load_kwargs)
    load_sec = time.time() - load_started
    peak_rss = proc.memory_info().rss

    case_results = []
    total_audio_sec = 0.0
    total_infer_sec = 0.0

    for item in audio_cases:
        case_id = item["case_id"]
        meta = case_meta.get(case_id, {})
        audio_path = item["audio_path"]
        duration = float(item.get("duration_sec") or ffprobe_duration(audio_path))
        language = normalize_language(meta.get("language", ""), args.language, args.language_style)
        case_hotwords = list(meta.get("hotwords", [])) if isinstance(meta.get("hotwords", []), list) else []
        generate_cfg = build_generate_cfg(args, language, case_hotwords)
        started = time.time()
        try:
            result = model.generate(input=audio_path, **generate_cfg)
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
                "language_arg": language if args.language_arg else None,
                "hotword_mode": args.hotword_mode,
                "hotwords_sent": case_hotwords if args.hotword_mode == "with_hotwords" else [],
                "device": device,
                "error": error,
                "prediction_chars": len(text),
            }
        )
        print(
            f"{case_id}\tduration={duration:.2f}s\tinfer={infer_sec:.2f}s\t"
            f"rtf={(infer_sec / duration if duration else 0):.3f}\tdevice={device}\terror={error}",
            flush=True,
        )

    size = model_size_bytes(args.model)
    component_sizes = {
        name: model_size_bytes(model_id)
        for name, model_id in [
            ("model", args.model),
            ("vad_model", args.vad_model),
            ("punc_model", args.punc_model),
        ]
        if model_id
    }
    pipeline_size = sum(size for size in component_sizes.values() if size is not None)
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
        "device": device,
        "load_kwargs": load_kwargs,
        "glm_loader_patch": glm_loader_patch,
        "generate_defaults": {
            "language": args.language,
            "language_style": args.language_style,
            "language_arg": args.language_arg,
            "batch_size_s": args.batch_size_s,
            "generate_batch_size": args.generate_batch_size,
            "cache_arg": args.cache_arg,
            "use_itn": args.use_itn,
            "merge_vad": args.merge_vad,
            "hotword": args.hotword,
            "hotword_mode": args.hotword_mode,
            "vad_max_single_segment_time": args.vad_max_single_segment_time,
        },
        "model_disk_gb": size / (1024**3) if size is not None else None,
        "component_disk_gb": {
            name: value / (1024**3) if value is not None else None for name, value in component_sizes.items()
        },
        "pipeline_disk_gb": pipeline_size / (1024**3) if pipeline_size else None,
        "cases": case_results,
        "note": "RSS does not fully capture Apple unified GPU memory; use as an approximate process memory signal.",
    }
    (out_dir / "run_metrics.json").write_text(json.dumps(metrics, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(f"pred_dir={pred_dir}")
    print(f"metrics={out_dir / 'run_metrics.json'}")


if __name__ == "__main__":
    main()
