# Dataset Card

## Summary

This is a Gold-only personal ASR benchmark for local model evaluation on Mac and other local environments. It targets practical use cases such as classroom videos, sales/customer calls, video meetings, public talks, podcasts/interviews, tool demos, and long-form audio.

The public export contains 202 Gold cases as of 2026-06-29:

| Split | Cases |
| --- | ---: |
| Short-form Gold | 197 |
| Long-form Gold | 5 |

| Language | Cases |
| --- | ---: |
| English | 100 |
| Mandarin Chinese | 67 |
| Japanese | 25 |
| Cantonese | 10 |

## Sources

Cases are selected from public YouTube and Bilibili videos. The repository stores source URLs and segment timestamps, but not the audio/video files. Local audio can be rebuilt with `tools/materialize_gold_audio.py`.

## Reference Policy

Reference text must come from public subtitles/transcripts or project-reviewed subtitle drafts. Local ASR model output may be used for screening or error analysis, but not as the reference source.

Gold review levels:

- `user_confirmed_real_audio`: accepted after direct review against real audio.
- `auto_screened_public_subtitle`: accepted from public subtitle/transcript text after strict automatic screening.

## Intended Use

- Compare local ASR models under realistic personal productivity scenarios.
- Track regressions as the dataset expands.
- Study hotword/context-term behavior separately from zero-shot recognition.
- Diagnose language-specific issues such as Japanese orthographic variants and Cantonese script normalization.

## Limitations

- It is a personal benchmark, not a universal public leaderboard.
- Subtitle-derived references can still contain upstream subtitle errors.
- Source media availability may change over time.
- Some sources may require login cookies or region access to rebuild local audio.
- Audio is not redistributed, so full end-to-end model reruns require network access and source availability.

## Excluded Data

The public repository excludes non-Gold candidates, backup cases, synthetic stress manifests, downloaded media, draft subtitles, model weights, local ASR drafts, and full prediction outputs.
