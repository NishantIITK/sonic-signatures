# Deploying the Q3B app

Two deployable apps live in this repo, sharing the same `fingerprint.py`
engine and the same `songs/` library, so you can pick whichever host treats
you better:

- **`app.py`** (Streamlit) → deploy on Streamlit Community Cloud. Free tier
  caps memory around ~1 GB, which this project sits close to even after
  tuning — see the memory notes below if it crashes.
- **`gradio_app.py`** (Gradio) → deploy on **Hugging Face Spaces** instead.
  Its free CPU tier gives roughly 16 GB RAM for a Gradio Space, which this
  project is nowhere near, so it's the more robust option if Streamlit Cloud
  keeps crashing on batch mode.

Both implement the same two required modes (single-clip with spectrogram /
constellation / offset histogram / matched name, and batch with a
`results.csv` of `filename,prediction`). Pick one for your report link —
you don't need to deploy both.

## Files

- `fingerprint.py` — the fingerprinting engine (shared by everything below).
- `app.py` — the Streamlit app.
- `gradio_app.py` — the Gradio app (for Hugging Face Spaces).
- `build_database.py` — optional, only needed for the Git LFS path below.
- `requirements.txt`, `apt.txt` — dependencies (`apt.txt` installs `ffmpeg` so
  the host's minimal Linux image can decode `.mp3`).
- `songs/` — the 50 course mp3 files (unrenamed).
- `.gitignore` — keeps `fingerprints.pkl` and caches out of git.
- `README.md` — Hugging Face Space metadata (the YAML header at the top)
  plus a short project blurb; this is also what GitHub shows as the repo's
  front page.

## Option 2: Hugging Face Spaces (Gradio) — recommended if Streamlit keeps crashing

1. Go to [huggingface.co/new-space](https://huggingface.co/new-space),
   create a Space, choose **Gradio** as the SDK. Note the Space's git URL,
   e.g. `https://huggingface.co/spaces/<username>/<space-name>`.
2. On your Mac, open Terminal in `~/Documents/sonic-signatures` (this is
   already a git repo with everything committed) and run:
   ```
   git remote add space https://huggingface.co/spaces/<username>/<space-name>
   git push space main
   ```
   When prompted for a password, use a Hugging Face **access token**
   (Settings → Access Tokens → create one with write access), not your
   account password.
3. The Space builds automatically (it reads the YAML header in `README.md`
   to know `sdk: gradio` and `app_file: gradio_app.py`). First load indexes
   the 50 songs once (~1–3 min), same as the Streamlit version, then it's
   instant.
4. Test both tabs on the Space's URL, then put that URL (and the GitHub repo
   URL) in your report.

## Option 1: Streamlit Community Cloud

## Important: do NOT commit fingerprints.pkl

The fingerprint database for 50 songs comes out to ~150 MB. GitHub hard-
rejects any single file over 100 MB unless you set up Git LFS, so committing
the raw pickle will fail the push. There are two ways to handle this —
**Option A is simpler and what this folder is set up for by default.**

### Option A — let the app build its own index on first run (recommended)

`app.py` already does this: if it doesn't find `fingerprints.pkl` next to
it, it indexes everything in `songs/` once, caches the result in memory for
the life of the container, and (if the filesystem allows it) writes
`fingerprints.pkl` locally so a same-container restart is instant. Cold
starts (the very first request after a deploy or after Streamlit Cloud puts
the app to sleep) will take roughly 1–3 minutes to index 50 songs; every
request after that is instant.

Steps:
1. Push this whole folder — including `songs/` (each mp3 is well under
   GitHub's 100 MB file limit) but **not** `fingerprints.pkl` — to a public
   GitHub repo. The included `.gitignore` already excludes it.
2. Go to [share.streamlit.io](https://share.streamlit.io), sign in with
   GitHub, click **New app**, pick the repo/branch, set the main file to
   `app.py`, and deploy.
3. Open the app once after it deploys and wait out the one-time indexing —
   you'll see the "Loading / building the song database..." spinner.

### Option B — pre-build the pickle and ship it via Git LFS (faster cold start)

If you want the very first visitor to skip the 1–3 minute wait:
```
git lfs install
git lfs track "fingerprints.pkl"
python build_database.py        # creates fingerprints.pkl (needs librosa/scipy)
git add .gitattributes fingerprints.pkl
git add -A
git commit -m "Add LFS-tracked fingerprint index"
git push
```
Streamlit Community Cloud does pull LFS objects on deploy, but free GitHub
LFS accounts only include 1 GB storage + 1 GB/month bandwidth — fine for one
project, but avoid Option B if you expect heavy traffic.

## Steps (common to both options)

1. Once deployed, test both modes on the live URL:
   - **Single-clip mode**: upload a short clip cut from one of the songs
     (try it with noise/pitch-shift too) and confirm the spectrogram,
     constellation, offset histogram, and matched name all show up.
   - **Batch mode**: upload several clips and download `results.csv`; check
     it has exactly the two columns `filename,prediction`.
2. Copy the live app URL and the GitHub repo URL into your Q3 report PDF.

## Notes

- If you change any parameter in `fingerprint.py` (n_fft, threshold, etc.),
  delete any cached `fingerprints.pkl` and let the app (or
  `build_database.py`) rebuild it — the database must be built with the
  same settings the app uses to query it.
- `songs/` files (max ~17 MB each here) are all individually under GitHub's
  100 MB hard limit, so they push fine without LFS.
