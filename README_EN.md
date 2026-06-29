# Personal ASR Benchmark

[中文 README](README.md)

This is a local ASR benchmark built around real user scenarios. It asks a practical question: if you want to transcribe class videos, sales calls, meeting recordings, public talks, or podcasts on a local machine, which ASR models actually work well, and which failure modes will affect summarization, search, and note-taking?

This project is not a replica of a public leaderboard, and it is not a single WER/CER-only test. It is a reproducible personal benchmark: samples come from public videos and public subtitles/transcripts, references are reviewed or strictly screened, and scoring looks at content accuracy, terminology, numbers, hallucination, punctuation, plus language-specific issues for Cantonese and Japanese.

## Who This Is For

- People comparing open-source ASR models on Mac or local machines.
- People who need benchmarks for classes, meetings, sales calls, interviews, and podcasts.
- People who care about multilingual ASR across Chinese, English, Japanese, and Cantonese.
- People who want to score their own ASR outputs with interpretable metrics.

## Dataset Coverage

As of 2026-06-29, the benchmark includes 202 samples:

- Short-form samples: 197
- Long-form samples: 5
- Languages: English 100, Mandarin Chinese 67, Japanese 25, Cantonese 10
- Scenarios: student lectures, sales/customer calls, meetings/panels, public talks, interviews/podcasts, online courses, tool demos, healthcare/consultation, news/narrative, and long-form audio.

Sources are mainly public YouTube and Bilibili videos. The repository stores source URLs, segment timestamps, scenario labels, language labels, hotwords, and final reference text so others can reproduce the same benchmark.

## Quick Start

Install dependencies:

```bash
python3 -m venv .venv
. .venv/bin/activate
python -m pip install -r requirements.txt
```

Validate the dataset:

```bash
python tools/validate_manifest.py
python tools/summarize_dataset.py
```

Rebuild a few local audio clips for a smoke test:

```bash
python tools/materialize_audio.py --limit 3
```

Audio is written to an ignored local directory:

```text
data/audio/benchmark/audio_manifest.json
data/audio/benchmark/<case_id>.wav
```

Some Bilibili or YouTube sources may require region access or login cookies:

```bash
python tools/materialize_audio.py --case <case_id> --cookies-from-browser chrome
```

## Evaluate Your Model

Any ASR model can be evaluated. Emit one UTF-8 text file per case:

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

If your model uses context terms or prompt-based hotwords, keep that result separate from zero-shot:

```bash
python tools/score_transcript.py \
  --pred-dir predictions/my_model_with_context \
  --hotword-mode with_context_terms \
  --out results/my_model_with_context.score.json
```

If your model has native hotword lists, decoder bias, or vocabulary bias:

```bash
python tools/score_transcript.py \
  --pred-dir predictions/my_model_native_hotwords \
  --hotword-mode native_hotwords \
  --out results/my_model_native_hotwords.score.json
```

## Reading The Scores

The main output is a 90-point text-side score:

- `weighted_text_score_90`: weighted aggregate text score for comparisons within this benchmark.
- `primary_error_rate`: per-case primary error rate; Chinese/Japanese/Cantonese usually use CER, English usually uses WER.
- `keyword_f1`: recall/precision for names, products, terms, and other key entities.
- `number_f1`: accuracy for numbers, money, time, versions, and similar fields.
- `hallucination_score`: penalties for overlong output, repetition, silence hallucination, and similar issues.
- `script_normalized_text_score_90`: Cantonese diagnostic score after Traditional/Simplified normalization.
- `japanese_orthographic_alias_text_score_90`: Japanese diagnostic score for acceptable writing variants.

`weighted_text_score_90` is specific to this project. It is not an industry-wide standard score. For serious comparisons, look at language and scenario breakdowns instead of only one aggregate number.

## Reproducibility

The repository does not redistribute third-party audio/video because the media belongs to the original publishers. The reproducible workflow is:

1. Read source URLs and segment timestamps from `data/benchmark_manifest.v1.json`.
2. Rebuild local audio clips with `tools/materialize_audio.py`.
3. Run any ASR model to produce prediction text.
4. Score predictions with `tools/score_transcript.py`.

This keeps large files, model weights, download caches, and account state out of the repository while preserving enough information to reproduce the same benchmark.

## Project Layout

- `data/benchmark_manifest.v1.json`: benchmark manifest.
- `data/benchmark_references/`: final reference transcripts.
- `tools/materialize_audio.py`: rebuild local audio from public sources.
- `tools/score_transcript.py`: scoring script.
- `tools/validate_manifest.py`: dataset validation.
- `tools/summarize_dataset.py`: dataset coverage summary.
- `docs/metrics.md`: metric definitions.
- `docs/benchmark_principles.md`: benchmark principles.
- `docs/dataset_card.md`: dataset card.

## License

Code is released under the MIT License. Dataset metadata and project-authored reference text are published for research and evaluation use; upstream media and source subtitles remain governed by their original publishers. See [DATA_LICENSE.md](DATA_LICENSE.md).
