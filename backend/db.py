"""SQLite storage: profiles (one per upload session) and try-on history."""
import sqlite3
from contextlib import contextmanager

from config import DB_PATH

SCHEMA = """
CREATE TABLE IF NOT EXISTS profiles (
    session_id TEXT PRIMARY KEY,
    created_at TEXT DEFAULT (datetime('now')),
    photo_path TEXT,
    photo_coverage TEXT,          -- full_body / upper_body
    face_blur INTEGER DEFAULT 1,  -- ON by default (privacy-first)
    height_cm REAL,
    weight_kg REAL,
    bust_band INTEGER,
    bust_cup TEXT,
    bust_input_method TEXT,       -- band_cup / chest_cm (lower precision)
    waist_cm REAL,
    hip_cm REAL,
    body_type TEXT
);
CREATE TABLE IF NOT EXISTS tryons (
    tryon_id TEXT PRIMARY KEY,
    session_id TEXT NOT NULL REFERENCES profiles(session_id),
    created_at TEXT DEFAULT (datetime('now')),
    item_id TEXT NOT NULL,
    brand TEXT,
    fabric TEXT,
    category TEXT,
    size TEXT,
    color TEXT,
    recommended_size TEXT,
    confidence INTEGER,
    state TEXT,                   -- blue / amber
    image_path TEXT,
    fit_feedback TEXT             -- small / fit / large (user-reported, optional)
);
"""


def init_db():
    with sqlite3.connect(DB_PATH) as conn:
        conn.executescript(SCHEMA)


@contextmanager
def get_db():
    conn = sqlite3.connect(DB_PATH)
    conn.row_factory = sqlite3.Row
    try:
        yield conn
        conn.commit()
    finally:
        conn.close()
