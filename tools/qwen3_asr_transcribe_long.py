#!/usr/bin/env python3
import argparse
import inspect
import json
import os
import re
import subprocess
import sys
import tempfile
import time
from pathlib import Path

import numpy as np
from mlx_audio.stt import load


DEFAULT_MODEL = Path.home() / ".local" / "share" / "asr-models" / "qwen3-asr-1.7b-4bit"


def run_command(cmd: list[str]) -> subprocess.CompletedProcess:
    return subprocess.run(cmd, text=True, capture_output=True)


def ffprobe_duration(path: Path) -> float:
    proc = run_command(
        [
            "ffprobe",
            "-v",
            "error",
            "-show_entries",
            "format=duration",
            "-of",
            "default=noprint_wrappers=1:nokey=1",
            str(path),
        ]
    )
    if proc.returncode != 0:
        raise RuntimeError(proc.stderr.strip() or f"ffprobe failed for {path}")
    return float(proc.stdout.strip())


def load_audio_np(path: Path, sample_rate: int = 16000) -> np.ndarray:
    proc = subprocess.run(
        [
            "ffmpeg",
            "-hide_banner",
            "-nostdin",
            "-loglevel",
            "error",
            "-i",
            str(path),
            "-vn",
            "-ac",
            "1",
            "-ar",
            str(sample_rate),
            "-f",
            "f32le",
            "-",
        ],
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
    )
    if proc.returncode != 0:
        raise RuntimeError(proc.stderr.decode("utf-8", errors="ignore").strip() or f"ffmpeg failed for {path}")
    return np.frombuffer(proc.stdout, dtype=np.float32).copy()


def detect_silences(path: Path, silence_db: float, min_silence_sec: float) -> list[tuple[float, float]]:
    proc = run_command(
        [
            "ffmpeg",
            "-hide_banner",
            "-nostdin",
            "-i",
            str(path),
            "-af",
            f"silencedetect=noise={silence_db}dB:d={min_silence_sec}",
            "-f",
            "null",
            "-",
        ]
    )
    # ffmpeg returns 0 for successful null output, but keep parsed stderr even if
    # a codec warning causes a non-zero exit; later extraction will fail loudly.
    silences: list[tuple[float, float]] = []
    current_start: float | None = None
    for line in proc.stderr.splitlines():
        start_match = re.search(r"silence_start:\s*([0-9.]+)", line)
        if start_match:
            current_start = float(start_match.group(1))
            continue
        end_match = re.search(r"silence_end:\s*([0-9.]+)", line)
        if end_match and current_start is not None:
            silences.append((current_start, float(end_match.group(1))))
            current_start = None
    return silences


def choose_cut(
    start: float,
    duration: float,
    silences: list[tuple[float, float]],
    target_sec: float,
    max_sec: float,
    min_sec: float,
) -> float:
    target = min(duration, start + target_sec)
    lower = min(duration, start + min_sec)
    upper = min(duration, start + max_sec)
    candidates = []
    for silence_start, silence_end in silences:
        midpoint = (silence_start + silence_end) / 2.0
        if lower <= midpoint <= upper:
            candidates.append(midpoint)
    if candidates:
        return min(candidates, key=lambda item: abs(item - target))
    return target if lower <= target <= upper else upper


def build_segments(
    duration: float,
    silences: list[tuple[float, float]],
    target_sec: float,
    max_sec: float,
    min_sec: float,
    overlap_sec: float,
) -> list[dict]:
    if target_sec <= 0 or max_sec <= 0 or min_sec <= 0:
        raise ValueError("target/max/min segment durations must be positive")
    if min_sec > target_sec or target_sec > max_sec:
        raise ValueError("--min-sec must be <= --target-sec <= --max-sec")
    if overlap_sec < 0:
        raise ValueError("--overlap-sec must be non-negative")

    boundaries = [0.0]
    start = 0.0
    while duration - start > max_sec:
        cut = choose_cut(start, duration, silences, target_sec, max_sec, min_sec)
        if cut <= start + 1.0:
            cut = min(duration, start + target_sec)
        boundaries.append(cut)
        start = cut
    if duration > boundaries[-1]:
        boundaries.append(duration)

    if len(boundaries) > 2 and boundaries[-1] - boundaries[-2] < min_sec:
        boundaries.pop(-2)

    segments = []
    for index, (start, end) in enumerate(zip(boundaries, boundaries[1:]), 1):
        extract_start = max(0.0, start - (overlap_sec if index > 1 else 0.0))
        extract_end = min(duration, end + (overlap_sec if index < len(boundaries) - 1 else 0.0))
        segments.append(
            {
                "index": index,
                "start": round(start, 3),
                "end": round(end, 3),
                "extract_start": round(extract_start, 3),
                "extract_end": round(extract_end, 3),
                "duration": round(end - start, 3),
                "extract_duration": round(extract_end - extract_start, 3),
            }
        )
    return segments


def build_segments_silero(
    audio_path: Path,
    duration: float,
    target_sec: float,
    max_sec: float,
    min_sec: float,
) -> tuple[list[dict], dict]:
    from silero_vad import get_speech_timestamps, load_silero_vad

    sample_rate = 16000
    wav = load_audio_np(audio_path, sample_rate)
    worker_vad_model = load_silero_vad(onnx=False)
    speech_timestamps = get_speech_timestamps(
        wav,
        worker_vad_model,
        sampling_rate=sample_rate,
        return_seconds=False,
        min_speech_duration_ms=1500,
        min_silence_duration_ms=500,
    )
    if not speech_timestamps:
        raise ValueError("No speech segments detected by Silero VAD.")

    total_samples = len(wav)
    potential_split_points = {0, total_samples}
    for item in speech_timestamps:
        potential_split_points.add(int(item["start"]))
    sorted_potential_splits = sorted(potential_split_points)

    final_split_points = {0, total_samples}
    target_samples = int(target_sec * sample_rate)
    target = target_samples
    while target < total_samples:
        closest = min(sorted_potential_splits, key=lambda point: abs(point - target))
        final_split_points.add(closest)
        target += target_samples

    ordered = sorted(final_split_points)
    max_samples = int(max_sec * sample_rate)
    min_samples = int(min_sec * sample_rate)
    split_points = [0]
    for start, end in zip(ordered, ordered[1:]):
        segment_len = end - start
        if segment_len <= max_samples:
            if end > split_points[-1]:
                split_points.append(end)
            continue
        subsegments = int(np.ceil(segment_len / max_samples))
        subsegment_len = segment_len / subsegments
        for index in range(1, subsegments):
            split = int(start + index * subsegment_len)
            if split > split_points[-1]:
                split_points.append(split)
        if end > split_points[-1]:
            split_points.append(end)

    if len(split_points) > 2 and split_points[-1] - split_points[-2] < min_samples:
        split_points.pop(-2)

    segments = []
    for index, (start_sample, end_sample) in enumerate(zip(split_points, split_points[1:]), 1):
        start = start_sample / sample_rate
        end = end_sample / sample_rate
        segments.append(
            {
                "index": index,
                "start": round(start, 3),
                "end": round(end, 3),
                "extract_start": round(start, 3),
                "extract_end": round(end, 3),
                "duration": round(end - start, 3),
                "extract_duration": round(end - start, 3),
            }
        )

    return segments, {
        "method": "silero_vad_speech_start",
        "speech_segment_count": len(speech_timestamps),
    }


def build_long_segments(
    audio_path: Path,
    duration: float,
    args: argparse.Namespace,
) -> tuple[list[dict], dict]:
    if args.segmenter in {"auto", "silero"}:
        try:
            segments, metadata = build_segments_silero(
                audio_path,
                duration,
                target_sec=args.target_sec,
                max_sec=args.max_sec,
                min_sec=args.min_sec,
            )
            return segments, metadata
        except Exception as exc:
            if args.segmenter == "silero":
                raise
            fallback_reason = repr(exc)
    else:
        fallback_reason = None

    silences = detect_silences(audio_path, args.silence_db, args.min_silence_sec)
    segments = build_segments(
        duration,
        silences,
        target_sec=args.target_sec,
        max_sec=args.max_sec,
        min_sec=args.min_sec,
        overlap_sec=args.overlap_sec,
    )
    return segments, {
        "method": "ffmpeg_silencedetect_or_target_duration",
        "detected_silence_count": len(silences),
        "fallback_reason": fallback_reason,
    }


def extract_chunk(source: Path, chunk_path: Path, start_sec: float, duration_sec: float) -> None:
    chunk_path.parent.mkdir(parents=True, exist_ok=True)
    proc = run_command(
        [
            "ffmpeg",
            "-hide_banner",
            "-nostdin",
            "-loglevel",
            "error",
            "-ss",
            f"{start_sec:.3f}",
            "-t",
            f"{duration_sec:.3f}",
            "-i",
            str(source),
            "-vn",
            "-ac",
            "1",
            "-ar",
            "16000",
            "-f",
            "wav",
            "-y",
            str(chunk_path),
        ]
    )
    if proc.returncode != 0:
        raise RuntimeError(proc.stderr.strip() or f"ffmpeg failed for chunk {chunk_path}")


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


def post_text_process(text: str, threshold: int = 20, max_pattern_len: int = 20) -> str:
    def fix_char_repeats(value: str) -> str:
        result = []
        index = 0
        while index < len(value):
            count = 1
            while index + count < len(value) and value[index + count] == value[index]:
                count += 1
            if count > threshold:
                result.append(value[index])
            else:
                result.append(value[index : index + count])
            index += count
        return "".join(result)

    def fix_pattern_repeats(value: str) -> str:
        min_repeat_chars = threshold * 2
        if len(value) < min_repeat_chars:
            return value
        index = 0
        result = []
        while index <= len(value) - min_repeat_chars:
            found = False
            for pattern_len in range(1, max_pattern_len + 1):
                if index + pattern_len * threshold > len(value):
                    break
                pattern = value[index : index + pattern_len]
                if all(
                    value[index + rep * pattern_len : index + (rep + 1) * pattern_len] == pattern
                    for rep in range(1, threshold)
                ):
                    end = index + threshold * pattern_len
                    while end + pattern_len <= len(value) and value[end : end + pattern_len] == pattern:
                        end += pattern_len
                    result.append(pattern)
                    result.append(fix_pattern_repeats(value[end:]))
                    index = len(value)
                    found = True
                    break
            if found:
                break
            result.append(value[index])
            index += 1
        if index < len(value):
            result.append(value[index:])
        return "".join(result)

    return fix_pattern_repeats(fix_char_repeats(text))


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


def generate_one(
    model,
    audio_path: Path,
    language: str | None,
    max_tokens: int,
    verbose: bool,
    system_prompt: str | None,
):
    signature = inspect.signature(model.generate)
    kwargs = {"max_tokens": max_tokens, "verbose": verbose}
    if "language" in signature.parameters:
        kwargs["language"] = language
    if "source_lang" in signature.parameters:
        kwargs["source_lang"] = language or "en"
    if "target_lang" in signature.parameters:
        kwargs["target_lang"] = language or "en"
    if "chunk_duration" in signature.parameters:
        kwargs["chunk_duration"] = 1200.0
    if system_prompt and "system_prompt" in signature.parameters:
        kwargs["system_prompt"] = system_prompt
    if "audio" in signature.parameters:
        return model.generate(audio=str(audio_path), **kwargs)
    return model.generate(str(audio_path), **kwargs)


def compact_chars_with_index(text: str) -> list[tuple[str, int]]:
    chars = []
    for index, char in enumerate(text):
        if char.isspace():
            continue
        chars.append((char.lower(), index))
    return chars


def trim_prefix_overlap(previous: str, current: str, min_chars: int, max_chars: int) -> str:
    if not previous or not current or min_chars <= 0:
        return current
    prev_chars = compact_chars_with_index(previous)
    curr_chars = compact_chars_with_index(current)
    max_overlap = min(max_chars, len(prev_chars), len(curr_chars))
    for size in range(max_overlap, min_chars - 1, -1):
        prev_suffix = "".join(char for char, _ in prev_chars[-size:])
        curr_prefix = "".join(char for char, _ in curr_chars[:size])
        if prev_suffix == curr_prefix:
            cut_index = curr_chars[size - 1][1] + 1
            return current[cut_index:].lstrip()
    return current


def merge_texts(texts: list[str], min_overlap_chars: int, max_overlap_chars: int) -> str:
    merged = ""
    for text in texts:
        clean = text.strip()
        if not clean:
            continue
        if not merged:
            merged = clean
            continue
        clean = trim_prefix_overlap(merged[-max_overlap_chars * 3 :], clean, min_overlap_chars, max_overlap_chars)
        if clean:
            merged = merged.rstrip() + "\n" + clean
    return merged.strip()


def format_timestamp(seconds: float) -> str:
    hours = int(seconds // 3600)
    minutes = int((seconds % 3600) // 60)
    seconds = seconds % 60
    return f"{hours:02d}:{minutes:02d}:{seconds:06.3f}".replace(".", ",")


def write_srt(path: Path, segments: list[dict]) -> None:
    lines = []
    index = 1
    for segment in segments:
        text = (segment.get("text") or "").strip()
        if not text:
            continue
        lines.extend(
            [
                str(index),
                f"{format_timestamp(float(segment['start']))} --> {format_timestamp(float(segment['end']))}",
                text,
                "",
            ]
        )
        index += 1
    path.write_text("\n".join(lines), encoding="utf-8")


def transcribe_long_audio(args: argparse.Namespace, audio_path: Path, model, load_sec: float) -> dict:
    duration = ffprobe_duration(audio_path)
    segments, segment_metadata = build_long_segments(audio_path, duration, args)

    out_dir = Path(args.out_dir).expanduser() if args.out_dir else None
    if out_dir:
        out_dir.mkdir(parents=True, exist_ok=True)

    temp_context = None
    if args.tmp_dir:
        chunk_root = Path(args.tmp_dir).expanduser() / audio_path.stem
        chunk_root.mkdir(parents=True, exist_ok=True)
    elif args.keep_chunks and out_dir:
        chunk_root = out_dir / f"{audio_path.stem}.chunks"
        chunk_root.mkdir(parents=True, exist_ok=True)
    else:
        temp_context = tempfile.TemporaryDirectory(prefix="qwen3_asr_long_")
        chunk_root = Path(temp_context.name)

    language = normalize_language(args.language)
    context_terms = [term for term in [*args.context_term, *args.hotword] if term.strip()]
    system_prompt = build_system_prompt(args.system_prompt, context_terms)

    started_audio = time.time()
    total_infer_sec = 0.0
    texts: list[str] = []
    chunk_records = []
    try:
        for segment in segments:
            chunk_path = chunk_root / f"{audio_path.stem}.{int(segment['index']):04d}.wav"
            extract_chunk(
                audio_path,
                chunk_path,
                float(segment["extract_start"]),
                float(segment["extract_duration"]),
            )
            started_chunk = time.time()
            try:
                result = generate_one(
                    model,
                    chunk_path,
                    language,
                    args.max_tokens_per_chunk,
                    args.verbose,
                    system_prompt,
                )
                text = output_text(result).strip()
                if not args.no_repeat_clean:
                    text = post_text_process(text, threshold=args.repeat_clean_threshold).strip()
                error = None
            except Exception as exc:
                text = ""
                error = repr(exc)
            infer_sec = time.time() - started_chunk
            total_infer_sec += infer_sec
            texts.append(text)
            record = {
                **segment,
                "chunk_path": str(chunk_path) if args.keep_chunks or args.tmp_dir else None,
                "infer_sec": infer_sec,
                "rtf_infer": infer_sec / max(float(segment["extract_duration"]), 1e-9),
                "text": text,
                "error": error,
            }
            chunk_records.append(record)
            if args.progress:
                print(
                    f"[{audio_path.name}] chunk {segment['index']}/{len(segments)} "
                    f"{segment['start']:.1f}-{segment['end']:.1f}s "
                    f"infer={infer_sec:.2f}s error={error}",
                    file=sys.stderr,
                    flush=True,
                )
    finally:
        if temp_context is not None and not args.keep_chunks:
            temp_context.cleanup()

    merged_text = merge_texts(texts, args.min_overlap_chars, args.max_overlap_chars)
    wall_sec = time.time() - started_audio

    record = {
        "audio": str(audio_path),
        "model": str(Path(args.model).expanduser()),
        "mode": "chunked_long_audio",
        "segmentation": {
            **segment_metadata,
            "duration_sec": duration,
            "target_sec": args.target_sec,
            "max_sec": args.max_sec,
            "min_sec": args.min_sec,
            "overlap_sec": args.overlap_sec,
            "silence_db": args.silence_db,
            "min_silence_sec": args.min_silence_sec,
            "chunk_count": len(segments),
        },
        "language": language,
        "context_terms": context_terms,
        "hotwords": args.hotword,
        "context_prompt_method": "system_prompt",
        "native_hotword_bias": False,
        "system_prompt": system_prompt,
        "load_sec": load_sec,
        "infer_sec": total_infer_sec,
        "wall_sec": wall_sec,
        "rtf_infer": total_infer_sec / max(duration, 1e-9),
        "rtf_wall": wall_sec / max(duration, 1e-9),
        "max_tokens_per_chunk": args.max_tokens_per_chunk,
        "text": merged_text,
        "segments": chunk_records,
    }

    if out_dir:
        stem = audio_path.stem
        (out_dir / f"{stem}.txt").write_text(merged_text + "\n", encoding="utf-8")
        (out_dir / f"{stem}.segments.json").write_text(
            json.dumps(record, ensure_ascii=False, indent=2) + "\n",
            encoding="utf-8",
        )
        if args.srt:
            write_srt(out_dir / f"{stem}.srt", chunk_records)

    return record


def main() -> None:
    parser = argparse.ArgumentParser(
        description=(
            "Transcribe long audio with Qwen3-ASR-1.7B 4bit by explicit "
            "silence-aware chunking and per-chunk token budgets."
        )
    )
    parser.add_argument("audio", nargs="+", help="Audio/video files accepted by ffmpeg.")
    parser.add_argument(
        "--model",
        default=os.environ.get("QWEN3_ASR_MODEL", str(DEFAULT_MODEL)),
        help="Model path. Defaults to ~/.local/share/asr-models/qwen3-asr-1.7b-4bit.",
    )
    parser.add_argument("--language", default="auto", help="auto, zh, en, ja, zh-en, etc.")
    parser.add_argument("--out-dir", help="Write <audio stem>.txt and <audio stem>.segments.json here.")
    parser.add_argument("--target-sec", type=float, default=120.0, help="Preferred chunk length.")
    parser.add_argument("--max-sec", type=float, default=180.0, help="Hard maximum chunk length before overlap.")
    parser.add_argument("--min-sec", type=float, default=20.0, help="Minimum chunk length.")
    parser.add_argument(
        "--segmenter",
        choices=["auto", "silero", "ffmpeg"],
        default="auto",
        help="Segmentation backend. auto tries Silero VAD and falls back to ffmpeg silencedetect.",
    )
    parser.add_argument("--overlap-sec", type=float, default=0.0, help="Audio overlap added to each side of interior chunks.")
    parser.add_argument("--silence-db", type=float, default=-35.0, help="ffmpeg silencedetect noise threshold in dB.")
    parser.add_argument("--min-silence-sec", type=float, default=0.4, help="Minimum silence duration used for cut candidates.")
    parser.add_argument("--max-tokens-per-chunk", type=int, default=4096)
    parser.add_argument("--repeat-clean-threshold", type=int, default=20, help="Collapse character/pattern repeats longer than this count.")
    parser.add_argument("--no-repeat-clean", action="store_true", help="Disable Qwen3-ASR-Toolkit-style repeat cleanup.")
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
    parser.add_argument("--tmp-dir", help="Directory for chunk wav files. Defaults to an auto-cleaned temp directory.")
    parser.add_argument("--keep-chunks", action="store_true", help="Keep extracted chunk wav files.")
    parser.add_argument("--min-overlap-chars", type=int, default=12, help="Minimum text overlap to remove during merge.")
    parser.add_argument("--max-overlap-chars", type=int, default=160, help="Maximum text overlap to check during merge.")
    parser.add_argument("--srt", action="store_true", help="Write rough chunk-level SRT when --out-dir is set.")
    parser.add_argument("--progress", action="store_true", help="Print per-chunk progress to stderr.")
    parser.add_argument("--verbose", action="store_true")
    parser.add_argument("--json", dest="json", action="store_true", default=True, help="Print JSON records. This is the default.")
    parser.add_argument("--text", dest="json", action="store_false", help="Print plain transcription text instead of JSON.")
    args = parser.parse_args()

    model_path = Path(args.model).expanduser()
    if not model_path.exists():
        raise SystemExit(f"model path not found: {model_path}")

    started_load = time.time()
    model = load(str(model_path))
    load_sec = time.time() - started_load

    for audio in args.audio:
        audio_path = Path(audio).expanduser()
        record = transcribe_long_audio(args, audio_path, model, load_sec)
        if args.json:
            print(json.dumps(record, ensure_ascii=False), flush=True)
        else:
            if len(args.audio) > 1:
                print(f"## {audio_path}", flush=True)
            print(record["text"], flush=True)


if __name__ == "__main__":
    main()
