#!/usr/bin/env python3
import argparse
import inspect
import json
import os
import sys
import time
from pathlib import Path

from mlx_audio.stt import load

TOOL_DIR = Path(__file__).resolve().parent
if str(TOOL_DIR) not in sys.path:
    sys.path.insert(0, str(TOOL_DIR))

from qwen3_asr_transcribe_long import ffprobe_duration, transcribe_long_audio


DEFAULT_MODEL = Path.home() / ".local" / "share" / "asr-models" / "qwen3-asr-1.7b-4bit"
DEFAULT_LONG_THRESHOLD_SEC = 1200.0


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


def normalize_language(language: str) -> str | None:
    if language == "auto":
        return None
    if language.startswith("zh"):
        return "zh"
    if language.startswith("en"):
        return "en"
    if language.startswith("ja"):
        return "ja"
    return language or None


def build_system_prompt(system_prompt: str | None, context_terms: list[str]) -> str | None:
    prompt_parts = []
    if system_prompt:
        prompt_parts.append(system_prompt.strip())
    clean_context_terms = [item.strip() for item in context_terms if item.strip()]
    if clean_context_terms:
        prompt_parts.append(
            "请完整逐字转写整段音频，不要摘要，不要只输出热词。"
            "以下词只用于纠正常见写法；只有音频中实际出现时才按这些写法输出："
            + "、".join(clean_context_terms)
            + "。"
        )
    return "\n".join(part for part in prompt_parts if part) or None


def probe_duration(audio_path: Path) -> tuple[float | None, str | None]:
    try:
        return ffprobe_duration(audio_path), None
    except Exception as exc:
        return None, repr(exc)


def resolve_pipeline(audio_path: Path, requested: str, long_threshold_sec: float) -> tuple[str, float | None, str | None]:
    duration, probe_error = probe_duration(audio_path)
    if requested == "single_pass":
        return "single_pass", duration, probe_error
    if requested == "vad_chunked":
        return "vad_chunked", duration, probe_error
    if duration is not None and duration >= long_threshold_sec:
        return "vad_chunked", duration, None
    return "single_pass", duration, probe_error


def generate_one(
    model,
    audio_path: str,
    language: str | None,
    max_tokens: int,
    verbose: bool,
    system_prompt: str | None,
):
    signature = inspect.signature(model.generate)
    kwargs = {
        "max_tokens": max_tokens,
        "verbose": verbose,
    }
    if "language" in signature.parameters:
        kwargs["language"] = language
    if "source_lang" in signature.parameters:
        kwargs["source_lang"] = language or "en"
    if "target_lang" in signature.parameters:
        kwargs["target_lang"] = language or "en"
    if system_prompt and "system_prompt" in signature.parameters:
        kwargs["system_prompt"] = system_prompt
    if "audio" in signature.parameters:
        return model.generate(audio=audio_path, **kwargs)
    return model.generate(audio_path, **kwargs)


def main() -> None:
    parser = argparse.ArgumentParser(
        description=(
            "Transcribe audio with the globally preferred Qwen3-ASR-1.7B 4bit local model. "
            "By default, short files use single-pass decoding and long files use VAD chunking."
        )
    )
    parser.add_argument("audio", nargs="+", help="Audio/video files accepted by the MLX-Audio loader.")
    parser.add_argument(
        "--model",
        default=os.environ.get("QWEN3_ASR_MODEL", str(DEFAULT_MODEL)),
        help="Model path. Defaults to ~/.local/share/asr-models/qwen3-asr-1.7b-4bit.",
    )
    parser.add_argument("--language", default="auto", help="auto, zh, en, ja, zh-en, etc.")
    parser.add_argument("--out-dir", help="Write <audio stem>.txt files here instead of only printing JSON lines.")
    parser.add_argument(
        "--pipeline",
        choices=["auto", "single_pass", "vad_chunked"],
        default="auto",
        help="auto chooses vad_chunked when duration >= --long-threshold-sec.",
    )
    parser.add_argument(
        "--long-threshold-sec",
        type=float,
        default=DEFAULT_LONG_THRESHOLD_SEC,
        help="Duration threshold for auto long-audio chunking. Default: 1200 seconds.",
    )
    parser.add_argument("--max-tokens", type=int, default=8192)
    parser.add_argument("--target-sec", type=float, default=120.0, help="Preferred chunk length for vad_chunked.")
    parser.add_argument("--max-sec", type=float, default=180.0, help="Hard maximum chunk length for vad_chunked.")
    parser.add_argument("--min-sec", type=float, default=20.0, help="Minimum chunk length for vad_chunked.")
    parser.add_argument(
        "--segmenter",
        choices=["auto", "silero", "ffmpeg"],
        default="auto",
        help="Long-audio segmentation backend. auto tries Silero VAD and falls back to ffmpeg silencedetect.",
    )
    parser.add_argument("--overlap-sec", type=float, default=0.0, help="Audio overlap for vad_chunked.")
    parser.add_argument("--silence-db", type=float, default=-35.0, help="ffmpeg silencedetect noise threshold for vad_chunked.")
    parser.add_argument("--min-silence-sec", type=float, default=0.4, help="Minimum silence duration for vad_chunked.")
    parser.add_argument("--max-tokens-per-chunk", type=int, default=4096)
    parser.add_argument("--repeat-clean-threshold", type=int, default=20)
    parser.add_argument("--no-repeat-clean", action="store_true")
    parser.add_argument("--system-prompt", help="Optional system prompt for models that support context prompting.")
    parser.add_argument(
        "--context-term",
        action="append",
        default=[],
        help="Optional context term for prompt-based terminology guidance. Repeat this flag for multiple terms.",
    )
    parser.add_argument(
        "--hotword",
        action="append",
        default=[],
        help="Legacy alias for --context-term. Qwen3-ASR MLX uses system_prompt context, not native hotword bias.",
    )
    parser.add_argument("--verbose", action="store_true")
    parser.add_argument("--tmp-dir", help="Directory for vad_chunked temporary chunk wav files.")
    parser.add_argument("--keep-chunks", action="store_true", help="Keep extracted vad_chunked chunk wav files.")
    parser.add_argument("--min-overlap-chars", type=int, default=12)
    parser.add_argument("--max-overlap-chars", type=int, default=160)
    parser.add_argument("--srt", action="store_true", help="Write rough chunk-level SRT for vad_chunked when --out-dir is set.")
    parser.add_argument("--progress", action="store_true", help="Print per-chunk progress for vad_chunked.")
    parser.add_argument("--json", dest="json", action="store_true", default=True, help="Print JSON lines with timing metadata. This is the default.")
    parser.add_argument("--text", dest="json", action="store_false", help="Print plain transcription text instead of JSON.")
    args = parser.parse_args()

    model_path = Path(args.model).expanduser()
    if not model_path.exists():
        raise SystemExit(f"model path not found: {model_path}")

    out_dir = Path(args.out_dir).expanduser() if args.out_dir else None
    if out_dir:
        out_dir.mkdir(parents=True, exist_ok=True)

    started_load = time.time()
    model = load(str(model_path))
    load_sec = time.time() - started_load
    language = normalize_language(args.language)
    context_terms = [term for term in [*args.context_term, *args.hotword] if term.strip()]
    system_prompt = build_system_prompt(args.system_prompt, context_terms)

    for audio in args.audio:
        audio_path = Path(audio).expanduser()
        pipeline, duration_sec, probe_error = resolve_pipeline(audio_path, args.pipeline, args.long_threshold_sec)
        if pipeline == "vad_chunked":
            record = transcribe_long_audio(args, audio_path, model, load_sec)
            record["pipeline"] = "vad_chunked"
            record["auto_pipeline_threshold_sec"] = args.long_threshold_sec
            if duration_sec is not None:
                record["duration_sec"] = duration_sec
        else:
            started = time.time()
            try:
                result = generate_one(model, str(audio_path), language, args.max_tokens, args.verbose, system_prompt)
                text = output_text(result).strip()
                error = None
            except Exception as exc:
                text = ""
                error = repr(exc)
            infer_sec = time.time() - started

            if out_dir:
                (out_dir / f"{audio_path.stem}.txt").write_text(text + "\n", encoding="utf-8")

            record = {
                "audio": str(audio_path),
                "model": str(model_path),
                "mode": "single_pass",
                "pipeline": "single_pass",
                "duration_sec": duration_sec,
                "duration_probe_error": probe_error,
                "auto_pipeline_threshold_sec": args.long_threshold_sec,
                "language": language,
                "context_terms": context_terms,
                "hotwords": args.hotword,
                "context_prompt_method": "system_prompt",
                "native_hotword_bias": False,
                "system_prompt": system_prompt,
                "load_sec": load_sec,
                "infer_sec": infer_sec,
                "rtf_infer": infer_sec / duration_sec if duration_sec else None,
                "error": error,
                "text": text,
            }
        if args.json:
            print(json.dumps(record, ensure_ascii=False), flush=True)
        else:
            if len(args.audio) > 1:
                print(f"## {audio_path}", flush=True)
            print(record.get("text", ""), flush=True)
            if record.get("error"):
                print(f"[error] {record['error']}", file=sys.stderr, flush=True)


if __name__ == "__main__":
    main()
