#!/usr/bin/env python3
import argparse
import json
import subprocess
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
DEFAULT_MANIFEST = "data/gold_manifest.v1.json"


def read_json(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def rel(path: Path) -> str:
    return str(path.resolve().relative_to(ROOT))


def duration_for(case: dict) -> float:
    segment = case.get("segment") or {}
    duration = segment.get("duration_sec")
    if isinstance(duration, (int, float)) and duration > 0:
        return float(duration)
    start = float(segment["start_sec"])
    end = float(segment["end_sec"])
    return end - start


def source_video_id(case: dict) -> str:
    source = case.get("source") or {}
    return str(source.get("video_id") or case["case_id"])


def download_source(case: dict, cache_dir: Path, args: argparse.Namespace) -> Path:
    source = case.get("source") or {}
    url = source.get("url")
    if not url:
        raise ValueError(f"{case['case_id']}: source.url is required")

    cache_dir.mkdir(parents=True, exist_ok=True)
    before = {p.resolve() for p in cache_dir.glob("*") if p.is_file()}
    output_template = str(cache_dir / "%(id)s.%(ext)s")
    cmd = [
        "yt-dlp",
        "-f",
        args.format,
        "--no-playlist",
        "-o",
        output_template,
        url,
    ]
    if args.cookies:
        cmd.extend(["--cookies", args.cookies])
    if args.cookies_from_browser:
        cmd.extend(["--cookies-from-browser", args.cookies_from_browser])
    subprocess.run(cmd, check=True)

    wanted_id = source_video_id(case)
    matches = sorted(cache_dir.glob(f"{wanted_id}.*"))
    if matches:
        return matches[0]
    after = sorted((p for p in cache_dir.glob("*") if p.is_file() and p.resolve() not in before), key=lambda p: p.stat().st_mtime)
    if after:
        return after[-1]
    raise FileNotFoundError(f"{case['case_id']}: yt-dlp completed but no source media was found in {cache_dir}")


def extract_segment(source_media: Path, out_audio: Path, case: dict, args: argparse.Namespace) -> None:
    segment = case.get("segment") or {}
    start = float(segment.get("start_sec") or 0)
    duration = duration_for(case)
    out_audio.parent.mkdir(parents=True, exist_ok=True)
    cmd = [
        "ffmpeg",
        "-hide_banner",
        "-loglevel",
        "error",
        "-y",
        "-ss",
        f"{start:.3f}",
        "-t",
        f"{duration:.3f}",
        "-i",
        str(source_media),
        "-vn",
        "-ac",
        str(args.channels),
        "-ar",
        str(args.sample_rate),
        str(out_audio),
    ]
    subprocess.run(cmd, check=True)


def should_include(case: dict, args: argparse.Namespace) -> bool:
    if args.case and case["case_id"] not in set(args.case):
        return False
    if args.short_only and case.get("case_type") == "real_long_gold":
        return False
    if args.long_only and case.get("case_type") != "real_long_gold":
        return False
    return True


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Rebuild local audio clips for the public Gold ASR dataset from source URLs."
    )
    parser.add_argument("--manifest", default=DEFAULT_MANIFEST)
    parser.add_argument("--out-dir", default="data/audio/gold")
    parser.add_argument("--cache-dir", default="data/audio/source_cache")
    parser.add_argument("--case", action="append", help="Materialize one case_id. Can be repeated.")
    parser.add_argument("--limit", type=int)
    parser.add_argument("--short-only", action="store_true")
    parser.add_argument("--long-only", action="store_true")
    parser.add_argument("--force", action="store_true", help="Overwrite existing extracted clips.")
    parser.add_argument("--format", default="ba/bestaudio/best")
    parser.add_argument("--sample-rate", type=int, default=16000)
    parser.add_argument("--channels", type=int, default=1)
    parser.add_argument("--audio-ext", default="wav", choices=["wav", "m4a", "mp3", "flac"])
    parser.add_argument("--cookies", help="Path to a yt-dlp cookies.txt file, if a source requires login.")
    parser.add_argument(
        "--cookies-from-browser",
        help="Browser name passed to yt-dlp, for example chrome or safari. Use only when the source requires it.",
    )
    args = parser.parse_args()

    if args.short_only and args.long_only:
        parser.error("--short-only and --long-only are mutually exclusive")

    manifest_path = (ROOT / args.manifest).resolve()
    manifest = read_json(manifest_path)
    out_dir = (ROOT / args.out_dir).resolve()
    cache_dir = (ROOT / args.cache_dir).resolve()
    selected = [case for case in manifest["cases"] if should_include(case, args)]
    if args.limit:
        selected = selected[: args.limit]
    if not selected:
        raise SystemExit("No cases selected.")

    materialized = []
    source_cache: dict[str, Path] = {}
    for index, case in enumerate(selected, 1):
        case_id = case["case_id"]
        out_audio = out_dir / f"{case_id}.{args.audio_ext}"
        if out_audio.exists() and not args.force:
            print(f"[{index}/{len(selected)}] skip existing {case_id}: {rel(out_audio)}")
        else:
            key = (case.get("source") or {}).get("url") or case_id
            source_media = source_cache.get(key)
            if source_media is None or not source_media.exists():
                print(f"[{index}/{len(selected)}] download source {case_id}")
                source_media = download_source(case, cache_dir, args)
                source_cache[key] = source_media
            print(f"[{index}/{len(selected)}] extract {case_id}: {rel(out_audio)}")
            extract_segment(source_media, out_audio, case, args)
        materialized.append(
            {
                "case_id": case_id,
                "audio_path": rel(out_audio),
                "duration_sec": duration_for(case),
                "language": case.get("language"),
                "hotwords": case.get("hotwords", []),
            }
        )

    audio_manifest = {
        "source_manifest": rel(manifest_path),
        "generated_by": "tools/materialize_gold_audio.py",
        "audio_root": rel(out_dir),
        "cases": materialized,
    }
    audio_manifest_path = out_dir / "audio_manifest.json"
    audio_manifest_path.write_text(json.dumps(audio_manifest, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(f"wrote {rel(audio_manifest_path)}")


if __name__ == "__main__":
    main()
