#!/usr/bin/env python3
import argparse
import json
import subprocess
import time
from pathlib import Path

import mlx.core as mx
import psutil
from parakeet_mlx import Beam, DecodingConfig, Greedy, SentenceConfig, from_pretrained


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
        candidates.append(
            Path.home()
            / ".cache"
            / "huggingface"
            / "hub"
            / f"models--{model_id.replace('/', '--')}"
        )
    for local in candidates:
        if local.exists():
            return dir_size_bytes(local)
    return None


def build_decoding_config(args: argparse.Namespace) -> DecodingConfig:
    decoding = (
        Beam(
            beam_size=args.beam_size,
            length_penalty=args.length_penalty,
            patience=args.patience,
            duration_reward=args.duration_reward,
        )
        if args.decoding == "beam"
        else Greedy()
    )
    return DecodingConfig(
        decoding=decoding,
        sentence=SentenceConfig(
            max_words=args.max_words,
            silence_gap=args.silence_gap,
            max_duration=args.max_duration,
        ),
    )


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Run parakeet-mlx over local audio manifest."
    )
    parser.add_argument("--model", required=True)
    parser.add_argument("--audio-manifest", default="data/audio/benchmark/audio_manifest.json")
    parser.add_argument("--manifest", default="data/benchmark_manifest.v1.json")
    parser.add_argument("--out-dir", required=True)
    parser.add_argument("--max-cases", type=int)
    parser.add_argument("--case", action="append")
    parser.add_argument("--dtype", choices=["bf16", "fp32"], default="bf16")
    parser.add_argument("--chunk-duration", type=float, default=0.0)
    parser.add_argument("--overlap-duration", type=float, default=15.0)
    parser.add_argument("--decoding", choices=["greedy", "beam"], default="greedy")
    parser.add_argument("--beam-size", type=int, default=5)
    parser.add_argument("--length-penalty", type=float, default=0.013)
    parser.add_argument("--patience", type=float, default=3.5)
    parser.add_argument("--duration-reward", type=float, default=0.67)
    parser.add_argument("--max-words", type=int)
    parser.add_argument("--silence-gap", type=float)
    parser.add_argument("--max-duration", type=float)
    args = parser.parse_args()

    out_dir = Path(args.out_dir)
    pred_dir = out_dir / "predictions"
    pred_dir.mkdir(parents=True, exist_ok=True)

    audio_manifest = read_json(args.audio_manifest)
    audio_cases = audio_manifest["cases"]
    if args.case:
        wanted = set(args.case)
        audio_cases = [c for c in audio_cases if c["case_id"] in wanted]
    if args.max_cases:
        audio_cases = audio_cases[: args.max_cases]

    dtype = mx.float32 if args.dtype == "fp32" else mx.bfloat16
    decoding_config = build_decoding_config(args)
    proc = psutil.Process()
    run_started = time.time()
    load_started = time.time()
    model = from_pretrained(args.model, dtype=dtype)
    load_sec = time.time() - load_started
    peak_rss = proc.memory_info().rss

    case_results = []
    total_audio_sec = 0.0
    total_infer_sec = 0.0
    chunk_duration = args.chunk_duration if args.chunk_duration > 0 else None

    for item in audio_cases:
        case_id = item["case_id"]
        audio_path = item["audio_path"]
        duration = float(item.get("duration_sec") or ffprobe_duration(audio_path))
        started = time.time()
        try:
            result = model.transcribe(
                audio_path,
                dtype=dtype,
                chunk_duration=chunk_duration,
                overlap_duration=args.overlap_duration,
                decoding_config=decoding_config,
            )
            text = result.text.strip()
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
                "error": error,
                "prediction_chars": len(text),
            }
        )
        print(
            f"{case_id}\tduration={duration:.2f}s\tinfer={infer_sec:.2f}s\t"
            f"rtf={(infer_sec / duration if duration else 0):.3f}\terror={error}",
            flush=True,
        )

    size = model_size_bytes(args.model)
    metrics = {
        "model": args.model,
        "runtime": "parakeet-mlx",
        "audio_manifest": args.audio_manifest,
        "case_count": len(case_results),
        "dtype": args.dtype,
        "decoding": args.decoding,
        "chunk_duration": chunk_duration,
        "load_sec": load_sec,
        "total_audio_sec": total_audio_sec,
        "total_infer_sec": total_infer_sec,
        "overall_rtf": total_infer_sec / total_audio_sec if total_audio_sec else None,
        "peak_rss_gb": peak_rss / (1024**3),
        "wall_sec": time.time() - run_started,
        "model_disk_gb": size / (1024**3) if size is not None else None,
        "pipeline_disk_gb": size / (1024**3) if size is not None else None,
        "cases": case_results,
        "note": "RSS does not fully capture Apple unified GPU memory; use as an approximate process memory signal.",
    }
    (out_dir / "run_metrics.json").write_text(
        json.dumps(metrics, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    print(f"pred_dir={pred_dir}")
    print(f"metrics={out_dir / 'run_metrics.json'}")


if __name__ == "__main__":
    main()
