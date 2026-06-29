# Personal ASR Gold Benchmark

一个面向真实本地 ASR 使用场景的个人开源测评集。数据集只公开最终 Gold 集合、人工/严格筛选后的 reference、评分脚本和方法文档；不包含模型权重、音视频文件、下载缓存、候选集、草稿字幕或中间结果。

## What Is Included

- `data/gold_manifest.v1.json`: Gold-only manifest with source URL, segment range, scenario, language, metric, weight, hotwords, and reference metadata.
- `data/gold_references/`: final reference transcript for every Gold case.
- `tools/validate_manifest.py`: schema and reference integrity checks.
- `tools/summarize_dataset.py`: dataset coverage summary.
- `tools/materialize_gold_audio.py`: rebuild local audio clips from source URLs for reproducible model runs.
- `tools/score_transcript.py`: CER/WER/hybrid TER, keyword, hotword, number, punctuation, hallucination, script-normalized Cantonese, and Japanese orthographic diagnostics.
- `docs/`: benchmark principles, metric definitions, dataset design, long-audio notes, and model comparison notes.

## Dataset Snapshot

As of 2026-06-29:

- Gold cases: 202
- Short-form Gold: 197
- Long-form Gold: 5
- Languages: English 100, Mandarin Chinese 67, Japanese 25, Cantonese 10
- Main scenarios: student lectures, sales/customer calls, meetings/panels, online talks, interviews/podcasts, courses, demos, healthcare/consultation, news/narrative, and long-form content.
- Sources: public YouTube and Bilibili videos with platform subtitles/transcripts or user-reviewed subtitle drafts.

Gold confidence tiers:

- `user_confirmed_real_audio`: reviewed against real audio by the project owner or assistant-led QA with explicit acceptance.
- `auto_screened_public_subtitle`: reference comes from public subtitle/transcript text and passed strict automatic screening; local ASR output was not used as reference text.

## Quick Start

Install core scoring dependencies:

```bash
python3 -m venv .venv
. .venv/bin/activate
python -m pip install -r requirements.txt
```

Validate the public Gold dataset:

```bash
python tools/validate_manifest.py
python tools/summarize_dataset.py
```

Rebuild local audio clips for a smoke test:

```bash
python tools/materialize_gold_audio.py --limit 3
```

This writes ignored local files under `data/audio/gold/`, including:

```text
data/audio/gold/audio_manifest.json
data/audio/gold/<case_id>.wav
```

Some Bilibili or YouTube sources may require region access or login cookies. In that case use standard `yt-dlp` options through:

```bash
python tools/materialize_gold_audio.py --case <case_id> --cookies-from-browser chrome
```

## Running A Model

The repository does not require one specific ASR model. Any model can be evaluated if it emits one UTF-8 text file per case:

```text
predictions/my_model/<case_id>.txt
```

Score a prediction directory:

```bash
python tools/score_transcript.py \
  --pred-dir predictions/my_model \
  --out results/my_model.score.json
```

Score one case:

```bash
python tools/score_transcript.py \
  --case youtube_bbc_ai_vocab_001 \
  --prediction predictions/my_model/youtube_bbc_ai_vocab_001.txt
```

If your model consumes case hotwords or context terms, keep those results separate from zero-shot results:

```bash
python tools/score_transcript.py \
  --pred-dir predictions/my_model_with_context \
  --hotword-mode with_context_terms \
  --out results/my_model_with_context.score.json
```

Native decoder hotword or vocabulary-bias mechanisms should use:

```bash
python tools/score_transcript.py \
  --pred-dir predictions/my_model_native_hotwords \
  --hotword-mode native_hotwords \
  --out results/my_model_native_hotwords.score.json
```

## Reproducibility Contract

The public repo is designed so another user can reproduce the benchmark workflow without private intermediate files:

1. Validate `data/gold_manifest.v1.json`.
2. Rebuild local audio from source URLs and segment timestamps with `tools/materialize_gold_audio.py`.
3. Run any ASR model over `data/audio/gold/audio_manifest.json`.
4. Save one transcript per case.
5. Score with `tools/score_transcript.py`.

Audio/video media is intentionally excluded because it belongs to upstream publishers. Source URL and segment metadata are included for reproducibility.

## Scoring Notes

- `weighted_text_score_90` is this benchmark's weighted aggregate, not an industry standard universal score.
- CER/WER/hybrid TER remain visible so results can be compared with common ASR practice.
- Cantonese gets an additional script-normalized diagnostic score to separate content recognition from Traditional/Simplified script mismatch.
- Japanese gets diagnostic scores for common furigana/orthographic variants, while bad references should still be fixed or removed.
- ITN/math formatting can be inspected separately; semantic-equivalent number or formula formatting should not be over-interpreted as raw ASR failure.

See [docs/metrics.md](docs/metrics.md) and [docs/benchmark_principles.md](docs/benchmark_principles.md).

## What Is Not Published

- Model weights and model caches.
- Downloaded audio/video/subtitle files.
- Browser cookies or account state.
- Non-Gold candidate cases, backup cases, synthetic stress manifests, draft references, and local ASR drafts.
- Full per-model prediction directories and intermediate benchmark outputs.

## License

Code is released under the MIT License. Dataset metadata and project-authored reference text are published for research and evaluation use; upstream media and source subtitles remain governed by their original publishers. See [DATA_LICENSE.md](DATA_LICENSE.md).
