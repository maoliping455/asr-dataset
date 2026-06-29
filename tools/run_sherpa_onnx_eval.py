#!/usr/bin/env python3
import argparse
import json
import subprocess
import time
from pathlib import Path
from typing import Any

import numpy as np
import psutil
import sherpa_onnx
from scipy.io import wavfile


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


def load_wav(path: str | Path) -> tuple[int, np.ndarray]:
    sample_rate, samples = wavfile.read(path)
    if samples.ndim > 1:
        samples = samples[:, 0]
    if np.issubdtype(samples.dtype, np.integer):
        max_abs = float(np.iinfo(samples.dtype).max)
        samples = samples.astype(np.float32) / max_abs
    else:
        samples = samples.astype(np.float32)
    return sample_rate, np.ascontiguousarray(samples)


def output_text(result: Any) -> str:
    text = getattr(result, "text", None)
    if isinstance(text, str):
        return text
    if isinstance(result, dict):
        value = result.get("text") or result.get("result") or result.get("transcription")
        if isinstance(value, str):
            return value
    if isinstance(result, str):
        return result
    return str(result)


def build_recognizer(args: argparse.Namespace):
    model_dir = Path(args.model_dir)
    if args.model_kind == "fire-red-asr-ctc":
        return sherpa_onnx.OfflineRecognizer.from_fire_red_asr_ctc(
            model=str(model_dir / "model.int8.onnx"),
            tokens=str(model_dir / "tokens.txt"),
            num_threads=args.num_threads,
            decoding_method=args.decoding_method,
            debug=args.debug,
            provider=args.provider,
        )
    raise ValueError(f"Unsupported model kind: {args.model_kind}")


def main() -> None:
    parser = argparse.ArgumentParser(description="Run sherpa-onnx ASR model over local audio manifest.")
    parser.add_argument("--model-kind", required=True, choices=["fire-red-asr-ctc"])
    parser.add_argument("--model-dir", required=True)
    parser.add_argument("--audio-manifest", default="data/audio/benchmark/audio_manifest.json")
    parser.add_argument("--manifest", default="data/benchmark_manifest.v1.json")
    parser.add_argument("--out-dir", required=True)
    parser.add_argument("--max-cases", type=int)
    parser.add_argument("--case", action="append")
    parser.add_argument("--num-threads", type=int, default=2)
    parser.add_argument("--provider", default="cpu", choices=["cpu", "coreml", "cuda"])
    parser.add_argument("--decoding-method", default="greedy_search")
    parser.add_argument("--debug", action="store_true")
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
    recognizer = build_recognizer(args)
    load_sec = time.time() - load_started
    peak_rss = proc.memory_info().rss

    case_results = []
    total_audio_sec = 0.0
    total_infer_sec = 0.0

    for item in audio_cases:
        case_id = item["case_id"]
        audio_path = item["audio_path"]
        duration = float(item.get("duration_sec") or ffprobe_duration(audio_path))
        started = time.time()
        try:
            sample_rate, samples = load_wav(audio_path)
            stream = recognizer.create_stream()
            stream.accept_waveform(sample_rate, samples)
            recognizer.decode_stream(stream)
            text = output_text(stream.result).strip()
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
                "scenario": case_meta.get(case_id, {}).get("scenario"),
                "language": case_meta.get(case_id, {}).get("language"),
                "audio_path": audio_path,
                "duration_sec": duration,
                "infer_sec": infer_sec,
                "rtf": infer_sec / duration if duration else None,
                "sample_rate": sample_rate if error is None else None,
                "provider": args.provider,
                "num_threads": args.num_threads,
                "error": error,
                "prediction_chars": len(text),
            }
        )
        print(
            f"{case_id}\tduration={duration:.2f}s\tinfer={infer_sec:.2f}s\t"
            f"rtf={(infer_sec / duration if duration else 0):.3f}\tprovider={args.provider}\terror={error}",
            flush=True,
        )

    model_dir = Path(args.model_dir)
    size = dir_size_bytes(model_dir) if model_dir.exists() else None
    metrics = {
        "model_kind": args.model_kind,
        "model_dir": args.model_dir,
        "audio_manifest": args.audio_manifest,
        "case_count": len(case_results),
        "load_sec": load_sec,
        "total_audio_sec": total_audio_sec,
        "total_infer_sec": total_infer_sec,
        "overall_rtf": total_infer_sec / total_audio_sec if total_audio_sec else None,
        "peak_rss_gb": peak_rss / (1024**3),
        "wall_sec": time.time() - run_started,
        "provider": args.provider,
        "num_threads": args.num_threads,
        "decoding_method": args.decoding_method,
        "model_disk_gb": size / (1024**3) if size is not None else None,
        "cases": case_results,
        "note": "sherpa-onnx runs through ONNX Runtime; RSS is process memory and may not include all provider-specific memory.",
    }
    (out_dir / "run_metrics.json").write_text(json.dumps(metrics, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(f"pred_dir={pred_dir}")
    print(f"metrics={out_dir / 'run_metrics.json'}")


if __name__ == "__main__":
    main()
