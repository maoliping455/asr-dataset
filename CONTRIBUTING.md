# Contributing

Contributions should preserve the Gold-only public contract.

## Accepted Contributions

- Fix a reference typo with source evidence.
- Add a new Gold case from public media that has an explicit subtitle or transcript source.
- Improve scoring, validation, materialization, or documentation.
- Add reproducible model-run instructions without committing model weights or predictions.

## Gold Case Requirements

Every new Gold case must include:

- `case_id`
- source URL and platform metadata
- exact segment start/end/duration
- language and scenario
- primary error metric
- `hotwords`, using `[]` when none are useful
- final reference text under `data/gold_references/`
- `reference.status = "ready"`
- `reference.review_level = "user_confirmed_real_audio"` or `auto_screened_public_subtitle`
- `reference.punctuation = "none"` when the final reference intentionally has no sentence punctuation

Reference text must come from source subtitles/transcripts plus review. Do not use a local ASR model output as the reference.

## Do Not Commit

- audio/video files
- downloaded subtitle files
- model weights or model caches
- browser cookies
- candidate/back-up manifests
- draft references
- per-model prediction directories
- large intermediate benchmark outputs

Run before submitting:

```bash
python tools/validate_manifest.py
python tools/summarize_dataset.py
python -m py_compile tools/*.py
```
