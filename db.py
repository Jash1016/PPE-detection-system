"""
Database layer.
===============
All SQLite access lives here. No other module runs SQL directly -- they call
these functions. If you ever switch to Postgres/MySQL, this is the only file
that changes.
"""
import sqlite3
from datetime import datetime

from config import DB_PATH


def init_db():
    """Create the violations table if it doesn't already exist."""
    conn = sqlite3.connect(DB_PATH)
    conn.execute("""
        CREATE TABLE IF NOT EXISTS violations (
            id          INTEGER PRIMARY KEY AUTOINCREMENT,
            feed_name   TEXT NOT NULL,
            track_id    INTEGER,             -- persistent tracker ID of the violator
            violation   TEXT NOT NULL,       -- 'no_helmet' or 'no_vest'
            confidence  REAL,
            date        TEXT NOT NULL,       -- YYYY-MM-DD
            time        TEXT NOT NULL,       -- HH:MM:SS
            timestamp   TEXT NOT NULL,       -- full ISO
            snapshot    TEXT                 -- relative path to jpg
        )
    """)
    conn.commit()
    conn.close()


def log_violation(feed_name, violation, confidence, snapshot_path, track_id=None):
    """Insert one violation row, stamping it with the current date/time."""
    now = datetime.now()
    conn = sqlite3.connect(DB_PATH)
    conn.execute(
        "INSERT INTO violations (feed_name, track_id, violation, confidence, date, time, timestamp, snapshot) "
        "VALUES (?,?,?,?,?,?,?,?)",
        (feed_name, track_id, violation, round(float(confidence), 3),
         now.strftime("%Y-%m-%d"), now.strftime("%H:%M:%S"),
         now.isoformat(timespec="seconds"), snapshot_path)
    )
    conn.commit()
    conn.close()


def fetch_violations(limit=200):
    """Most-recent-first list of violation rows as plain dicts."""
    conn = sqlite3.connect(DB_PATH)
    conn.row_factory = sqlite3.Row
    rows = conn.execute(
        "SELECT * FROM violations ORDER BY id DESC LIMIT ?", (limit,)
    ).fetchall()
    conn.close()
    return [dict(r) for r in rows]


def fetch_stats():
    """Aggregate counters for the dashboard header + charts."""
    conn = sqlite3.connect(DB_PATH)
    conn.row_factory = sqlite3.Row
    total = conn.execute("SELECT COUNT(*) c FROM violations").fetchone()["c"]
    today = conn.execute(
        "SELECT COUNT(*) c FROM violations WHERE date = ?",
        (datetime.now().strftime("%Y-%m-%d"),)
    ).fetchone()["c"]
    by_feed = conn.execute(
        "SELECT feed_name, COUNT(*) c FROM violations GROUP BY feed_name"
    ).fetchall()
    by_type = conn.execute(
        "SELECT violation, COUNT(*) c FROM violations GROUP BY violation"
    ).fetchall()
    conn.close()
    return {
        "total": total,
        "today": today,
        "by_feed": {r["feed_name"]: r["c"] for r in by_feed},
        "by_type": {r["violation"]: r["c"] for r in by_type},
    }
