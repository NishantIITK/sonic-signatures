# Deploying the app (Streamlit Community Cloud)

## Files

- `fingerprint.py` — the fingerprinting engine.
- `app.py` — the Streamlit app (single-clip mode + batch mode).
- `build_database.py` — optional, only needed for the Git LFS path below.
- `requirements.txt`, `apt.txt` — dependencies (`apt.txt` installs `ffmpeg`
  so Streamlit Cloud's minimal Linux image can decode `.mp3`).
- `songs/` — the 50 course mp3 files (unrenamed).
- `.gitignore` — keeps `fingerprints.pkl` and caches out of git.

## Running it locally first

```
pip install -r requirements.txt
streamlit run app.py
```

Make sure `songs/` is sitting next to `app.py` before you run it — the app
indexes whatever's in there on first launch.

## Important: do NOT commit fingerprints.pkl

The fingerprint database for 50 songs comes out to over 100 MB. GitHub
hard-rejects any single file over 100 MB unless you set up Git LFS, so
committing the raw pickle will fail the push. This repo is set up to avoid
that entirely:

`app.py` builds its own index on first run if it doesn't find
`fingerprints.pkl` next to it — it indexes everything in `songs/` once,
caches the result in memory for the life of the container, and writes
`fingerprints.pkl` locally so a same-container restart is instant. Cold
starts (the very first request after a deploy, or after Streamlit puts the
app to sleep from inactivity) take roughly 1–3 minutes to index 50 songs;
every request after that is instant. The `.gitignore` already excludes
`fingerprints.pkl` so it never gets pushed.

If you ever want to skip that first-load wait by shipping a pre-built index
instead, use Git LFS:
```
git lfs install
git lfs track "fingerprints.pkl"
python build_database.py        # creates fingerprints.pkl
git add .gitattributes fingerprints.pkl
git add -A
git commit -m "Add LFS-tracked fingerprint index"
git push
```
Free GitHub LFS accounts include 1 GB storage + 1 GB/month bandwidth,
which is plenty for a single project.

## Deploy steps

1. Push this repo to GitHub (already done — `songs/` is well under GitHub's
   100 MB per-file limit, so it pushes fine without LFS).
2. Go to [share.streamlit.io](https://share.streamlit.io), sign in with
   GitHub, click **New app**, pick the repo/branch, set the main file to
   `app.py`, and deploy.
3. Open the app once after it deploys and wait out the one-time indexing —
   you'll see a "Loading / building the song database..." spinner.
4. Test both modes on the live URL:
   - **Single-clip mode**: upload a short clip cut from one of the songs
     and confirm the spectrogram, constellation, offset histogram, and
     matched name all show up.
   - **Batch mode**: upload several clips and download `results.csv`;
     check it has exactly the two columns `filename,prediction`.
5. Put the live app URL and the GitHub repo URL in the report.

## Memory notes

Streamlit Community Cloud's free tier caps memory around ~1 GB, and this
project sits close to that with all 50 songs indexed. To stay under the
limit, `fingerprint.py` is tuned for a smaller hash database (`FAN_VALUE=4`,
`MAX_DT=60`), the app only builds the pair-hash table (no single-hash table,
which isn't needed by the app's UI), matplotlib figures are explicitly
closed after each render, and batch mode runs garbage collection every few
files. If you ever change `fingerprint.py`'s parameters (n_fft, threshold,
fan value, etc.), delete any cached `fingerprints.pkl` and let the app (or
`build_database.py`) rebuild it — the database must be built with the same
settings the app uses to query it.
