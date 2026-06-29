# Data License And Usage Notes

This repository publishes a curated ASR benchmark manifest and final reference transcripts for research, evaluation, and reproducibility.

The repository does not redistribute third-party audio, video, or full subtitle assets. Source media remains owned by the original publishers on YouTube, Bilibili, or other public platforms. Users who rebuild local audio clips with `tools/materialize_audio.py` are responsible for complying with the terms of the source platform and publisher.

Project-authored metadata, case selection, scoring scripts, and benchmark documentation are released under the MIT License unless otherwise noted.

Reference transcripts are included as benchmark labels for the selected short excerpts and long-form cases. They may derive from public subtitles/transcripts plus project review. Use them for ASR research and evaluation; do not treat this repository as a redistribution channel for the original media or subtitle corpus.

Do not commit regenerated media, browser cookies, model weights, downloaded subtitle files, or model prediction dumps to this repository.
