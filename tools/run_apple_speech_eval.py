#!/usr/bin/env python3
import argparse
import json
import subprocess
import tempfile
import time
from pathlib import Path

import psutil


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


def language_to_locale(language: str) -> str:
    if language.startswith("zh"):
        return "zh-CN"
    if language.startswith("ja"):
        return "ja-JP"
    return "en-US"


def apple_speech_rss(app_binary_marker: str) -> int:
    total = 0
    for proc in psutil.process_iter(["name", "cmdline", "memory_info"]):
        try:
            name = proc.info.get("name") or ""
            cmdline = " ".join(proc.info.get("cmdline") or [])
            if "apple_speech_transcribe" in name or app_binary_marker in cmdline:
                total += proc.info["memory_info"].rss
        except (psutil.NoSuchProcess, psutil.AccessDenied):
            continue
    return total


def run_case(app_path: Path, item: dict, locale: str, mode: str, timeout_sec: float) -> tuple[dict, int, str, str]:
    audio_path = str(Path(item["audio_path"]).resolve())
    app_binary_marker = f"{app_path.name}/Contents/MacOS/apple_speech_transcribe"
    with tempfile.NamedTemporaryFile(prefix="apple_speech_", suffix=".json", delete=False) as tmp:
        out_json = Path(tmp.name)
    out_json.unlink(missing_ok=True)

    mode_arg = "--on-device" if mode == "on-device" else "--allow-server"
    cmd = [
        "open",
        "-W",
        "-n",
        str(app_path),
        "--args",
        "--audio",
        audio_path,
        "--locale",
        locale,
        mode_arg,
        "--timeout-sec",
        str(timeout_sec),
        "--out",
        str(out_json),
    ]
    proc = subprocess.Popen(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)
    peak_rss = 0
    while proc.poll() is None:
        peak_rss = max(peak_rss, apple_speech_rss(app_binary_marker))
        time.sleep(0.02)
    stdout, stderr = proc.communicate()
    peak_rss = max(peak_rss, apple_speech_rss(app_binary_marker))

    if out_json.exists():
        result = json.loads(out_json.read_text(encoding="utf-8"))
    else:
        result = {
            "ok": False,
            "text": "",
            "error": "missing_output_json",
            "infer_sec": None,
            "locale": locale,
            "on_device": mode == "on-device",
        }
    out_json.unlink(missing_ok=True)
    result["open_returncode"] = proc.returncode
    return result, peak_rss, stdout, stderr


def main() -> None:
    parser = argparse.ArgumentParser(description="Run Apple Speech framework over a local ASR audio manifest.")
    parser.add_argument("--app", default="build/AppleSpeechTranscribe.app")
    parser.add_argument("--audio-manifest", default="data/audio/benchmark/audio_manifest.json")
    parser.add_argument("--manifest", default="data/benchmark_manifest.v1.json")
    parser.add_argument("--out-dir", required=True)
    parser.add_argument("--mode", choices=["on-device", "allow-server"], default="on-device")
    parser.add_argument("--timeout-sec", type=float, default=180.0)
    parser.add_argument("--max-cases", type=int)
    parser.add_argument("--case", action="append")
    args = parser.parse_args()

    app_path = Path(args.app).resolve()
    if not app_path.exists():
        raise FileNotFoundError(app_path)

    out_dir = Path(args.out_dir)
    pred_dir = out_dir / "predictions"
    pred_dir.mkdir(parents=True, exist_ok=True)

    audio_manifest = read_json(args.audio_manifest)
    eval_manifest = read_json(args.manifest)
    case_meta = {case["case_id"]: case for case in eval_manifest["cases"]}
    audio_cases = audio_manifest["cases"]
    if args.case:
        wanted = set(args.case)
        audio_cases = [case for case in audio_cases if case["case_id"] in wanted]
    if args.max_cases:
        audio_cases = audio_cases[: args.max_cases]

    run_started = time.time()
    peak_rss = 0
    total_audio_sec = 0.0
    total_infer_sec = 0.0
    case_results = []

    for item in audio_cases:
        case_id = item["case_id"]
        meta = case_meta.get(case_id, {})
        language = meta.get("language") or item.get("language") or ""
        locale = language_to_locale(language)
        duration = float(item.get("duration_sec") or ffprobe_duration(item["audio_path"]))
        started = time.time()
        result, child_peak_rss, stdout, stderr = run_case(app_path, item, locale, args.mode, args.timeout_sec)
        wall_infer_sec = time.time() - started
        infer_sec = float(result.get("infer_sec") or wall_infer_sec)
        text = str(result.get("text") or "").strip()
        error = None if result.get("ok") else str(result.get("error") or "unknown_error")
        peak_rss = max(peak_rss, child_peak_rss)
        total_audio_sec += duration
        total_infer_sec += infer_sec
        (pred_dir / f"{case_id}.txt").write_text(text + "\n", encoding="utf-8")
        case_results.append(
            {
                "case_id": case_id,
                "audio_path": item["audio_path"],
                "duration_sec": duration,
                "infer_sec": infer_sec,
                "rtf": infer_sec / duration if duration else None,
                "locale": locale,
                "mode": args.mode,
                "error": error,
                "prediction_chars": len(text),
                "open_returncode": result.get("open_returncode"),
                "stdout_tail": stdout.strip()[-1000:],
                "stderr_tail": stderr.strip()[-1000:],
            }
        )
        print(
            f"{case_id}\tduration={duration:.2f}s\tinfer={infer_sec:.2f}s\t"
            f"rtf={(infer_sec / duration if duration else 0):.3f}\tlocale={locale}\terror={error}",
            flush=True,
        )

    metrics = {
        "backend": "Apple Speech framework",
        "mode": args.mode,
        "app": str(app_path),
        "audio_manifest": args.audio_manifest,
        "case_count": len(case_results),
        "total_audio_sec": total_audio_sec,
        "total_infer_sec": total_infer_sec,
        "overall_rtf": total_infer_sec / total_audio_sec if total_audio_sec else None,
        "peak_rss_gb": peak_rss / (1024**3),
        "model_disk_gb": None,
        "wall_sec": time.time() - run_started,
        "cases": case_results,
        "note": (
            "Apple Speech framework is a macOS system service. In allow-server mode it may use Apple services "
            "and should not be treated as a pure local/offline model. RSS is sampled from the launched helper app."
        ),
    }
    (out_dir / "run_metrics.json").write_text(json.dumps(metrics, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(f"pred_dir={pred_dir}")
    print(f"metrics={out_dir / 'run_metrics.json'}")


if __name__ == "__main__":
    main()
