"""
gradio_app.py — Q3B "Zapptain America": Gradio wrapper around the Sonic
Signatures fingerprinting engine (fingerprint.py).

Why a second app file: Streamlit Community Cloud's free tier caps memory at
~1GB, which this project sits close to. Hugging Face Spaces' free CPU tier
gives ~16GB RAM for a Gradio app with zero extra optimisation -- same
fingerprint.py engine, same two required modes, much more headroom.

Two modes, as required by the assignment:
  1. Single-clip mode: upload one query clip, see the spectrogram, the
     constellation of peaks, the offset histogram, and the matched song.
  2. Batch mode: upload several query clips, get a results.csv with
     exactly two columns: filename, prediction.

Run locally:    python gradio_app.py
Deploy:         see README_DEPLOY.md (Hugging Face Spaces section)
"""

import gc
import os

import gradio as gr
import librosa
import librosa.display
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd

from fingerprint import (
    SR, N_FFT, HOP_LEN,
    build_database, load_database, save_database, list_audio_files,
    match_clip,
)

DB_PATH = "fingerprints.pkl"
SONG_DIR = "songs"

_db_cache = {}


def get_database():
    """Load fingerprints.pkl if present, else build once from songs/ and
    cache for the lifetime of the process (module-level dict, not per
    request) -- same one-time-index idea as the Streamlit app."""
    if _db_cache:
        return _db_cache["db_p"], _db_cache["db_s"], _db_cache["stats"], _db_cache["song_files"]

    if os.path.exists(DB_PATH):
        db_p, db_s, stats, song_files = load_database(DB_PATH)
    else:
        song_files = list_audio_files(SONG_DIR)
        if not song_files:
            raise FileNotFoundError(
                f"No songs found in '{SONG_DIR}/'. Add the song library and restart."
            )
        db_p, db_s, stats = build_database(
            song_files=song_files, verbose=True, compute_single=False
        )
        save_database(DB_PATH, db_p, db_s, stats, song_files)

    _db_cache.update(db_p=db_p, db_s=db_s, stats=stats, song_files=song_files)
    gc.collect()
    return db_p, db_s, stats, song_files


def fig_spectrogram(S_db, title="Spectrogram"):
    fig, ax = plt.subplots(figsize=(7, 3))
    img = librosa.display.specshow(
        S_db, sr=SR, hop_length=HOP_LEN, x_axis="time", y_axis="hz", ax=ax, cmap="magma"
    )
    ax.set_title(title)
    fig.colorbar(img, ax=ax, format="%+2.0f dB")
    fig.tight_layout()
    return fig


def fig_constellation(S_db, peaks, title="Constellation"):
    fig, ax = plt.subplots(figsize=(7, 3))
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
    fig, ax = plt.subplots(figsize=(7, 3))
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


def identify_clip(audio_path):
    if audio_path is None:
        return None, None, None, "Upload a clip first."

    db_p, db_s, _, _ = get_database()
    y, _ = librosa.load(audio_path, sr=SR, mono=True)
    res = match_clip(y, SR, db_p, db_s)
    del y

    name = res["pair_match"] or "No match found"
    score = res["pair_scores"].get(res["pair_match"], 0)

    spec_fig = fig_spectrogram(res["S_db"], "Query spectrogram")
    const_fig = fig_constellation(res["S_db"], res["peaks"], "Query constellation")
    if res["pair_offsets"]:
        off_fig = fig_offsets(res["pair_offsets"], res["pair_scores"], "Offset histogram (pair hashes)")
    else:
        off_fig, ax = plt.subplots(figsize=(7, 3))
        ax.set_title("No pair-hash matches found")

    result_md = f"### Identified song: **{name}**\nPair-hash peak score: {score}"
    del res
    gc.collect()
    return spec_fig, const_fig, off_fig, result_md


def run_batch(files):
    if not files:
        return None, None

    db_p, db_s, _, _ = get_database()
    rows = []
    for i, f in enumerate(files):
        path = f if isinstance(f, str) else f.name
        y, _ = librosa.load(path, sr=SR, mono=True)
        res = match_clip(y, SR, db_p, db_s)
        prediction = res["pair_match"] or ""
        rows.append({"filename": os.path.basename(path), "prediction": prediction})
        del y, res
        if (i + 1) % 5 == 0:
            gc.collect()

    df = pd.DataFrame(rows, columns=["filename", "prediction"])
    out_path = "results.csv"
    df.to_csv(out_path, index=False)
    gc.collect()
    return df, out_path


with gr.Blocks(title="Sonic Signatures — Mini Shazam") as demo:
    gr.Markdown(
        "# Sonic Signatures — Mini Music Identifier\n"
        "EE200 Course Project — Q3B: Signals to Softwares"
    )

    with gr.Tab("Single-clip mode"):
        audio_in = gr.Audio(type="filepath", label="Upload a query clip")
        identify_btn = gr.Button("Identify", variant="primary")
        result_md = gr.Markdown()
        with gr.Row():
            spec_plot = gr.Plot(label="Spectrogram")
            const_plot = gr.Plot(label="Constellation")
        offset_plot = gr.Plot(label="Offset histogram (pair hashes)")

        identify_btn.click(
            identify_clip,
            inputs=audio_in,
            outputs=[spec_plot, const_plot, offset_plot, result_md],
        )

    with gr.Tab("Batch mode"):
        gr.Markdown(
            "Upload a set of query clips. Each is identified against the indexed "
            "library and the results are written to `results.csv` with columns "
            "`filename, prediction`."
        )
        files_in = gr.File(file_count="multiple", label="Upload query clips")
        batch_btn = gr.Button("Run batch", variant="primary")
        results_table = gr.Dataframe(headers=["filename", "prediction"], label="Results")
        results_file = gr.File(label="Download results.csv")

        batch_btn.click(run_batch, inputs=files_in, outputs=[results_table, results_file])

if __name__ == "__main__":
    demo.launch()
