# Sonic Signatures — Mini Music Identifier

EE200 Course Project (Summer 2026) — Q3A/Q3B: a small Shazam-style audio
fingerprinting system. It turns each song into a spectrogram, keeps only
the strongest time-frequency peaks (the "constellation"), pairs nearby
peaks into hashes `(f1, f2, dt)`, and identifies a query clip by checking
which song's hashes line up at a single time offset.

## What's in this repo

- `fingerprint.py` — the fingerprinting engine: spectrogram, peak
  extraction, hashing, database build, and offset-histogram matching.
- `app.py` — the Streamlit app built on top of it.
- `build_database.py` — standalone script to pre-build the fingerprint
  database locally if needed.
- `songs/` — the course's song library (50 mp3 files, unrenamed — the
  filename without extension is the label the identifier outputs).
- `requirements.txt`, `apt.txt` — dependencies for deployment.

## The app

Two modes, per the project spec:

1. **Single-clip mode** — upload one query clip and see the spectrogram,
   the constellation of peaks, the offset histogram that decides the
   match, and the identified song.
2. **Batch mode** — upload several query clips at once and download
   `results.csv` with columns `filename, prediction`.

See `README_DEPLOY.md` for how to run it locally and deploy it on
Streamlit Community Cloud.
