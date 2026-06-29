#!/usr/bin/env python3
import argparse
import json
import subprocess
import time
from pathlib import Path

import psutil
import soundfile as sf
import torch
from transformers import AutoModelForSpeechSeq2Seq, AutoProcessor


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


def load_audio(path: str | Path) -> tuple:
    wav, sr = sf.read(str(path), dtype="float32", always_2d=False)
    if getattr(wav, "ndim", 1) > 1:
        wav = wav.mean(axis=1)
    if sr != 16000:
        raise ValueError(f"Expected 16 kHz audio, got {sr} Hz for {path}")
    return wav, sr


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Run Transformers Granite Speech ASR over local audio manifest."
    )
    parser.add_argument("--model", required=True)
    parser.add_argument("--audio-manifest", default="data/audio/gold/audio_manifest.json")
    parser.add_argument("--manifest", default="data/gold_manifest.v1.json")
    parser.add_argument("--out-dir", required=True)
    parser.add_argument("--max-cases", type=int)
    parser.add_argument("--case", action="append")
    parser.add_argument("--device", choices=["auto", "mps", "cpu"], default="auto")
    parser.add_argument("--dtype", choices=["bf16", "fp16", "fp32"], default="bf16")
    parser.add_argument("--max-new-tokens", type=int, default=200)
    parser.add_argument(
        "--prompt",
        default="<|audio|>transcribe the speech with proper punctuation and capitalization.",
    )
    parser.add_argument("--system-prompt")
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

    if args.device == "auto":
        device = "mps" if torch.backends.mps.is_available() else "cpu"
    else:
        device = args.device

    dtype_map = {
        "bf16": torch.bfloat16,
        "fp16": torch.float16,
        "fp32": torch.float32,
    }
    dtype = dtype_map[args.dtype]

    proc = psutil.Process()
    run_started = time.time()
    load_started = time.time()
    processor = AutoProcessor.from_pretrained(args.model)
    tokenizer = processor.tokenizer
    model = AutoModelForSpeechSeq2Seq.from_pretrained(args.model, dtype=dtype)
    model.eval().to(device)
    load_sec = time.time() - load_started
    peak_rss = proc.memory_info().rss

    chat = []
    if args.system_prompt:
        chat.append({"role": "system", "content": args.system_prompt})
    chat.append({"role": "user", "content": args.prompt})
    prompt_text = tokenizer.apply_chat_template(
        chat, tokenize=False, add_generation_prompt=True
    )

    case_results = []
    total_audio_sec = 0.0
    total_infer_sec = 0.0

    for item in audio_cases:
        case_id = item["case_id"]
        audio_path = item["audio_path"]
        duration = float(item.get("duration_sec") or ffprobe_duration(audio_path))
        started = time.time()
        try:
            wav, _ = load_audio(audio_path)
            inputs = processor(
                prompt_text, wav, device=device, return_tensors="pt"
            ).to(device)
            with torch.inference_mode():
                outputs = model.generate(
                    **inputs,
                    max_new_tokens=args.max_new_tokens,
                    do_sample=False,
                    num_beams=1,
                )
            num_input_tokens = inputs["input_ids"].shape[-1]
            new_tokens = outputs[0, num_input_tokens:].unsqueeze(0)
            text = tokenizer.batch_decode(
                new_tokens, add_special_tokens=False, skip_special_tokens=True
            )[0].strip()
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
        "runtime": "transformers",
        "audio_manifest": args.audio_manifest,
        "case_count": len(case_results),
        "device": device,
        "dtype": args.dtype,
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
