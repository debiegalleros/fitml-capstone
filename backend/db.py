"""SQLite storage: profiles (one per upload session) and try-on history."""
import sqlite3
from contextlib import contextmanager

from config import DB_PATH

SCHEMA = """
CREATE TABLE IF NOT EXISTS profiles (
    session_id TEXT PRIMARY KEY,
    created_at TEXT DEFAULT (datetime('now')),
    name TEXT,
    photo_path TEXT,
    photo_side_path TEXT,         -- optional, not pose-extracted (future multi-angle try-on)
    photo_back_path TEXT,         -- optional, not pose-extracted
    photo_coverage TEXT,          -- full_body / upper_body
    face_cropped INTEGER DEFAULT 0, -- opt-in checkbox; OFF by default
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
        _migrate(conn)


def _migrate(conn):
    """Add columns introduced after a dev DB was first created (SQLite's
    CREATE TABLE IF NOT EXISTS won't retrofit an existing table)."""
    cols = {row[1] for row in conn.execute("PRAGMA table_info(profiles)")}
    for col, coltype in (("name", "TEXT"), ("photo_side_path", "TEXT"), ("photo_back_path", "TEXT"),
                        ("face_cropped", "INTEGER DEFAULT 0")):
        if col not in cols:
            conn.execute(f"ALTER TABLE profiles ADD COLUMN {col} {coltype}")


@contextmanager
def get_db():
    conn = sqlite3.connect(DB_PATH)
    conn.row_factory = sqlite3.Row
    try:
        yield conn
        conn.commit()
    finally:
        conn.close()
