#!/usr/bin/env python3
import argparse
import json
import os
import tempfile
import subprocess
import time
from collections import defaultdict
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


def dir_size_bytes(local: Path) -> int:
    if local.is_file():
        return local.stat().st_size
    total = 0
    for item in local.rglob("*"):
        if item.is_file():
            total += item.stat().st_size
    return total


def normalize_language(lang: str, mode: str) -> str:
    if mode == "auto":
        return "auto"
    if lang.startswith("zh"):
        return "zh"
    if lang.startswith("en"):
        return "en"
    if lang.startswith("ja"):
        return "ja"
    return "auto"


def run_cli(cmd: list[str]) -> tuple[int, str, str, int]:
    # whisper.cpp logs progress to stderr for every file. Large batched runs can
    # fill an OS pipe and block the child process if the parent only polls RSS.
    with tempfile.TemporaryFile(mode="w+", encoding="utf-8") as stdout_file, tempfile.TemporaryFile(
        mode="w+", encoding="utf-8"
    ) as stderr_file:
        proc = subprocess.Popen(cmd, stdout=stdout_file, stderr=stderr_file, text=True)
        peak_rss = 0
        ps_proc = psutil.Process(proc.pid)
        while proc.poll() is None:
            try:
                rss = ps_proc.memory_info().rss
                for child in ps_proc.children(recursive=True):
                    try:
                        rss += child.memory_info().rss
                    except psutil.Error:
                        pass
                peak_rss = max(peak_rss, rss)
            except psutil.Error:
                pass
            time.sleep(0.05)
        stdout_file.seek(0)
        stderr_file.seek(0)
        stdout = stdout_file.read()
        stderr = stderr_file.read()
        return proc.returncode, stdout, stderr, peak_rss


def main() -> None:
    parser = argparse.ArgumentParser(description="Run whisper.cpp over a local ASR audio manifest.")
    parser.add_argument("--binary", default="external/whisper.cpp/build/bin/whisper-cli")
    parser.add_argument("--model", required=True)
    parser.add_argument("--audio-manifest", default="data/audio/gold/audio_manifest.json")
    parser.add_argument("--manifest", default="data/gold_manifest.v1.json")
    parser.add_argument("--out-dir", required=True)
    parser.add_argument("--max-cases", type=int)
    parser.add_argument("--case", action="append")
    parser.add_argument("--threads", type=int, default=min(8, os.cpu_count() or 4))
    parser.add_argument("--processors", type=int, default=1)
    parser.add_argument("--language-mode", choices=["manifest", "auto"], default="manifest")
    parser.add_argument("--batch-by-language", action="store_true", help="Run one whisper-cli process per language group to avoid per-file model reload overhead.")
    parser.add_argument("--no-gpu", action="store_true")
    parser.add_argument("--beam-size", type=int)
    parser.add_argument("--best-of", type=int)
    parser.add_argument("--extra-arg", action="append", default=[])
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

    model_path = Path(args.model)
    binary_path = Path(args.binary)
    if not model_path.exists():
        raise FileNotFoundError(model_path)
    if not binary_path.exists():
        raise FileNotFoundError(binary_path)

    run_started = time.time()
    peak_rss = psutil.Process().memory_info().rss
    case_results = []
    total_audio_sec = 0.0
    total_infer_sec = 0.0

    def base_cmd(language: str) -> list[str]:
        cmd = [
            str(binary_path),
            "-m",
            str(model_path),
            "-l",
            language,
            "-nt",
            "-otxt",
            "-np",
            "-t",
            str(args.threads),
            "-p",
            str(args.processors),
        ]
        if args.no_gpu:
            cmd.append("--no-gpu")
        if args.beam_size is not None:
            cmd.extend(["-bs", str(args.beam_size)])
        if args.best_of is not None:
            cmd.extend(["-bo", str(args.best_of)])
        cmd.extend(args.extra_arg)
        return cmd

    if args.batch_by_language:
        groups: dict[str, list[dict]] = defaultdict(list)
        for item in audio_cases:
            meta = case_meta.get(item["case_id"], {})
            language = normalize_language(meta.get("language", ""), args.language_mode)
            groups[language].append(item)

        with tempfile.TemporaryDirectory(prefix="whisper_cpp_eval_") as tmp_name:
            tmp_dir = Path(tmp_name)
            for language, group_items in groups.items():
                link_paths = []
                group_audio_sec = 0.0
                for item in group_items:
                    link_path = tmp_dir / f"{item['case_id']}.wav"
                    link_path.unlink(missing_ok=True)
                    link_path.symlink_to(Path(item["audio_path"]).resolve())
                    link_paths.append(link_path)
                    group_audio_sec += float(item.get("duration_sec") or ffprobe_duration(item["audio_path"]))

                cmd = base_cmd(language) + [str(path) for path in link_paths]
                started = time.time()
                returncode, stdout, stderr, child_peak_rss = run_cli(cmd)
                group_infer_sec = time.time() - started
                peak_rss = max(peak_rss, child_peak_rss)
                total_audio_sec += group_audio_sec
                total_infer_sec += group_infer_sec
                group_error = None if returncode == 0 else f"returncode={returncode}: {stderr.strip()[-1000:]}"

                for item, link_path in zip(group_items, link_paths):
                    case_id = item["case_id"]
                    duration = float(item.get("duration_sec") or ffprobe_duration(item["audio_path"]))
                    out_txt = link_path.with_suffix(".wav.txt")
                    if out_txt.exists():
                        text = out_txt.read_text(encoding="utf-8").strip()
                    else:
                        text = ""
                    (pred_dir / f"{case_id}.txt").write_text(text + "\n", encoding="utf-8")
                    infer_sec = group_infer_sec * (duration / group_audio_sec) if group_audio_sec else 0.0
                    error = group_error
                    if not out_txt.exists() and error is None:
                        error = "missing_output_file"
                    case_results.append(
                        {
                            "case_id": case_id,
                            "audio_path": item["audio_path"],
                            "duration_sec": duration,
                            "infer_sec": infer_sec,
                            "rtf": infer_sec / duration if duration else None,
                            "timing_scope": "batch_by_language_apportioned",
                            "group_language": language,
                            "group_infer_sec": group_infer_sec,
                            "group_audio_sec": group_audio_sec,
                            "language_arg": language,
                            "threads": args.threads,
                            "processors": args.processors,
                            "no_gpu": args.no_gpu,
                            "returncode": returncode,
                            "error": error,
                            "prediction_chars": len(text),
                            "stderr_tail": stderr.strip()[-2000:],
                        }
                    )
                    print(
                        f"{case_id}\tduration={duration:.2f}s\tgroup_infer={group_infer_sec:.2f}s\t"
                        f"group_rtf={(group_infer_sec / group_audio_sec if group_audio_sec else 0):.3f}\t"
                        f"lang={language}\terror={error}",
                        flush=True,
                    )
    else:
        for item in audio_cases:
            case_id = item["case_id"]
            meta = case_meta.get(case_id, {})
            audio_path = item["audio_path"]
            duration = float(item.get("duration_sec") or ffprobe_duration(audio_path))
            language = normalize_language(meta.get("language", ""), args.language_mode)
            out_base = pred_dir / case_id
            out_txt = out_base.with_suffix(".txt")
            out_txt.unlink(missing_ok=True)

            cmd = base_cmd(language) + ["-f", audio_path, "-of", str(out_base)]

            started = time.time()
            returncode, stdout, stderr, child_peak_rss = run_cli(cmd)
            infer_sec = time.time() - started
            peak_rss = max(peak_rss, child_peak_rss)
            error = None
            if returncode != 0:
                error = f"returncode={returncode}: {stderr.strip()[-1000:]}"
                text = ""
                out_txt.write_text("", encoding="utf-8")
            elif out_txt.exists():
                text = out_txt.read_text(encoding="utf-8").strip()
                out_txt.write_text(text + "\n", encoding="utf-8")
            else:
                text = stdout.strip()
                out_txt.write_text(text + "\n", encoding="utf-8")

            total_audio_sec += duration
            total_infer_sec += infer_sec
            case_results.append(
                {
                    "case_id": case_id,
                    "audio_path": audio_path,
                    "duration_sec": duration,
                    "infer_sec": infer_sec,
                    "rtf": infer_sec / duration if duration else None,
                    "timing_scope": "per_file_process",
                    "language_arg": language,
                    "threads": args.threads,
                    "processors": args.processors,
                    "no_gpu": args.no_gpu,
                    "returncode": returncode,
                    "error": error,
                    "prediction_chars": len(text),
                    "stderr_tail": stderr.strip()[-2000:],
                }
            )
            print(
                f"{case_id}\tduration={duration:.2f}s\tinfer={infer_sec:.2f}s\t"
                f"rtf={(infer_sec / duration if duration else 0):.3f}\tlang={language}\terror={error}",
                flush=True,
            )

    metrics = {
        "backend": "whisper.cpp",
        "binary": str(binary_path),
        "model": str(model_path),
        "audio_manifest": args.audio_manifest,
        "case_count": len(case_results),
        "total_audio_sec": total_audio_sec,
        "total_infer_sec": total_infer_sec,
        "overall_rtf": total_infer_sec / total_audio_sec if total_audio_sec else None,
        "peak_rss_gb": peak_rss / (1024**3),
        "wall_sec": time.time() - run_started,
        "model_disk_gb": dir_size_bytes(model_path) / (1024**3),
        "language_mode": args.language_mode,
        "batch_by_language": args.batch_by_language,
        "threads": args.threads,
        "processors": args.processors,
        "no_gpu": args.no_gpu,
        "beam_size": args.beam_size,
        "best_of": args.best_of,
        "extra_args": args.extra_arg,
        "cases": case_results,
        "note": "RSS is sampled from the whisper-cli child process; Metal unified GPU allocations may still be approximate.",
    }
    (out_dir / "run_metrics.json").write_text(json.dumps(metrics, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(f"pred_dir={pred_dir}")
    print(f"metrics={out_dir / 'run_metrics.json'}")


if __name__ == "__main__":
    main()
