# Deploying the Q3B app (Streamlit Community Cloud)

## Files

- `fingerprint.py` — the fingerprinting engine (shared with the Kaggle notebook).
- `app.py` — the Streamlit app (single-clip mode + batch mode).
- `build_database.py` — optional, only needed for the Git LFS path below.
- `requirements.txt`, `apt.txt` — dependencies (`apt.txt` installs `ffmpeg` so
  Streamlit Cloud's minimal Linux image can decode `.mp3`).
- `songs/` — the 50 course mp3 files (unrenamed).
- `.gitignore` — keeps `fingerprints.pkl` and caches out of git.

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
