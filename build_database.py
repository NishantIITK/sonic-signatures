"""
build_database.py
==================
Run this ONCE locally, with the song-library folder named 'songs/' sitting
next to this script, to pre-build fingerprints.pkl:

    python build_database.py

Commit the resulting fingerprints.pkl together with app.py, fingerprint.py,
requirements.txt and apt.txt before deploying to Streamlit Community Cloud.
Shipping the pickle means the deployed app loads instantly instead of
re-indexing every song on first visit.
"""

from fingerprint import build_database, save_database, list_audio_files

SONG_DIR = "songs"
DB_PATH = "fingerprints.pkl"

if __name__ == "__main__":
    song_files = list_audio_files(SONG_DIR)
    print(f"Found {len(song_files)} song(s) in '{SONG_DIR}/'")
    if not song_files:
        raise SystemExit(f"No audio files found in '{SONG_DIR}/'. Add the song library there first.")

    db_p, db_s, stats = build_database(song_files=song_files, verbose=True)
    save_database(DB_PATH, db_p, db_s, stats, song_files)

    print(f"\nSaved {DB_PATH}")
    print(f"  pair-hash keys   : {len(db_p):,}")
    print(f"  single-hash keys : {len(db_s):,}")
    print(f"  songs indexed    : {len(song_files)}")
