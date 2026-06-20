"""
app.py — Q3B "Zapptain America": Streamlit wrapper around the Sonic
Signatures fingerprinting engine (fingerprint.py).

Two modes, as required by the assignment:
  1. Single-clip mode: upload one query clip, see the spectrogram, the
     constellation of peaks, the offset histogram, and the matched song.
  2. Batch mode: upload several query clips, get a results.csv with
     exactly two columns: filename, prediction.

Run locally:    streamlit run app.py
Deploy:         see README_DEPLOY.md
"""

import gc
import os
import tempfile

import librosa
import librosa.display
import matplotlib.pyplot as plt
import pandas as pd
import streamlit as st

from fingerprint import (
    SR, N_FFT, HOP_LEN,
    build_database, load_database, save_database, list_audio_files,
    match_clip,
)

# Memory housekeeping: only the pair-hash database is used by this app (the
# single-peak comparison lives in the Q3A report/notebook, not the app), and
# we process one query clip fully before moving to the next so peak memory
# stays bounded to ~one song/clip at a time -- this is what keeps both modes
# inside Streamlit Community Cloud's free-tier memory limit.
BATCH_GC_EVERY = 5  # force a garbage-collection pass every N batch files

st.set_page_config(page_title="Sonic Signatures — Mini Shazam", layout="wide")
st.title("Sonic Signatures — Mini Music Identifier")
st.caption("EE200 Course Project — Q3B: Signals to Softwares")

DB_PATH = "fingerprints.pkl"
SONG_DIR = "songs"


@st.cache_resource(show_spinner="Loading / building the song database...")
def get_database():
    """Load a pre-built fingerprints.pkl if it ships with the app; otherwise
    build it once from the songs/ folder and cache it for next time.

    compute_single=False: the app only ever needs the pair-hash database for
    matching (the single-vs-pair comparison is a report/notebook thing, not
    an app feature), so we never even build the single-hash table here --
    that alone removes roughly a fifth of the resident memory.
    """
    if os.path.exists(DB_PATH):
        db_p, db_s, stats, song_files = load_database(DB_PATH)
    else:
        song_files = list_audio_files(SONG_DIR)
        if not song_files:
            return None
        db_p, db_s, stats = build_database(
            song_files=song_files, verbose=False, compute_single=False
        )
        save_database(DB_PATH, db_p, db_s, stats, song_files)
    gc.collect()
    return db_p, db_s, stats, song_files


db = get_database()
if db is None:
    st.error(
        f"No songs found. Put the song library in a '{SONG_DIR}/' folder "
        f"next to app.py (or ship a pre-built '{DB_PATH}') and rerun."
    )
    st.stop()

db_p, db_s, stats, song_files = db

with st.sidebar:
    st.header("Indexed database")
    st.write(f"**{len(song_files)}** songs indexed")
    st.dataframe(pd.DataFrame(stats).T, use_container_width=True)


def load_audio_from_upload(uploaded_file):
    suffix = os.path.splitext(uploaded_file.name)[1] or ".wav"
    with tempfile.NamedTemporaryFile(delete=False, suffix=suffix) as tmp:
        tmp.write(uploaded_file.read())
        path = tmp.name
    try:
        y, _ = librosa.load(path, sr=SR, mono=True)
    finally:
        os.unlink(path)
    return y


def fig_spectrogram(S_db, title="Spectrogram"):
    fig, ax = plt.subplots(figsize=(8, 3))
    img = librosa.display.specshow(
        S_db, sr=SR, hop_length=HOP_LEN, x_axis="time", y_axis="hz", ax=ax, cmap="magma"
    )
    ax.set_title(title)
    fig.colorbar(img, ax=ax, format="%+2.0f dB")
    fig.tight_layout()
    return fig


def fig_constellation(S_db, peaks, title="Constellation"):
    fig, ax = plt.subplots(figsize=(8, 3))
    librosa.display.specshow(
        S_db, sr=SR, hop_length=HOP_LEN, x_axis="time", y_axis="hz", ax=ax, cmap="gray_r"
    )
    t = [p[0] * HOP_LEN / SR for p in peaks]
    f = [p[1] * SR / N_FFT for p in peaks]
    ax.scatter(t, f, s=8, c="cyan", linewidths=0)
    ax.set_title(f"{title} ({len(peaks)} peaks)")
    fig.tight_layout()
    return fig


def fig_offsets(offsets, scores, title, top=4):
    fig, ax = plt.subplots(figsize=(8, 3))
    ranked = sorted(scores.items(), key=lambda x: -x[1])[:top]
    palette = ["gold", "steelblue", "salmon", "mediumpurple"]
    for i, (name, sc) in enumerate(ranked):
        ox = list(offsets[name].keys())
        oy = list(offsets[name].values())
        ax.vlines(ox, 0, oy, color=palette[i % len(palette)], alpha=0.7, label=f"{name} ({sc})")
    ax.set(title=title, xlabel="time offset (frames)", ylabel="hash matches")
    ax.legend(fontsize=7)
    fig.tight_layout()
    return fig


def show_and_close(fig):
    """st.pyplot keeps no reference, but matplotlib's global figure manager
    does -- close explicitly so repeated uploads in one session don't leak
    memory figure by figure."""
    st.pyplot(fig)
    plt.close(fig)


tab_single, tab_batch = st.tabs(["Single-clip mode", "Batch mode"])

with tab_single:
    st.subheader("Identify one query clip")
    uploaded = st.file_uploader(
        "Upload a query clip", type=["wav", "mp3", "flac", "m4a", "ogg"], key="single_uploader"
    )
    if uploaded is not None:
        y = load_audio_from_upload(uploaded)
        with st.spinner("Fingerprinting and matching..."):
            res = match_clip(y, SR, db_p, db_s)
        del y

        pair_name = res["pair_match"] or "No match found"
        pair_score = res["pair_scores"].get(res["pair_match"], 0)

        st.success(f"Identified song: **{pair_name}**")
        st.caption(f"Pair-hash peak score: {pair_score}")

        c1, c2 = st.columns(2)
        with c1:
            show_and_close(fig_spectrogram(res["S_db"], "Query spectrogram"))
        with c2:
            show_and_close(fig_constellation(res["S_db"], res["peaks"], "Query constellation"))

        if res["pair_offsets"]:
            show_and_close(
                fig_offsets(res["pair_offsets"], res["pair_scores"], "Offset histogram (pair hashes)")
            )
        else:
            st.info("No pair-hash matches found against the database.")
        del res
        gc.collect()

with tab_batch:
    st.subheader("Identify many query clips at once")
    files = st.file_uploader(
        "Upload query clips",
        type=["wav", "mp3", "flac", "m4a", "ogg"],
        accept_multiple_files=True,
        key="batch_uploader",
    )
    if files:
        st.caption(
            f"{len(files)} file(s) queued. Each clip is fully processed and "
            f"freed before the next one starts to keep memory bounded."
        )
        rows = []
        progress = st.progress(0.0)
        for i, f in enumerate(files):
            y = load_audio_from_upload(f)
            res = match_clip(y, SR, db_p, db_s)
            prediction = res["pair_match"] or ""
            rows.append({"filename": f.name, "prediction": prediction})
            progress.progress((i + 1) / len(files))
            del y, res
            if (i + 1) % BATCH_GC_EVERY == 0:
                gc.collect()
        gc.collect()

        results_df = pd.DataFrame(rows, columns=["filename", "prediction"])
        st.dataframe(results_df, use_container_width=True)

        csv_bytes = results_df.to_csv(index=False).encode("utf-8")
        st.download_button(
            "Download results.csv", data=csv_bytes, file_name="results.csv", mime="text/csv"
        )
