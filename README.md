---
title: Sonic Signatures
emoji: 🎵
colorFrom: blue
colorTo: green
sdk: gradio
app_file: gradio_app.py
pinned: false
---

# Sonic Signatures — Mini Music Identifier

EE200 Course Project, Q3A/Q3B — a small Shazam-style audio fingerprinting
system: spectrogram → constellation of peaks → pair-hash fingerprints →
offset-histogram matching.

- `fingerprint.py` — the engine (shared by the Kaggle notebook and both apps below).
- `app.py` — Streamlit version of the app.
- `gradio_app.py` — Gradio version of the app (used on Hugging Face Spaces).
- `songs/` — the indexed song library.

See `README_DEPLOY.md` for deployment instructions for both Streamlit
Community Cloud and Hugging Face Spaces.
