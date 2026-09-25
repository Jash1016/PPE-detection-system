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


def _row(r):
    """sqlite Row -> dict, with the snapshot path normalized for use as a URL.

    Older rows were written with os.path.join, so on Windows they contain
    backslashes ('snapshots\\cam1_1_x.jpg'). A backslash inside a JavaScript
    string literal is an escape character, which silently mangled the path in
    the browser -- so always hand out forward slashes.
    """
    d = dict(r)
    if d.get("snapshot"):
        d["snapshot"] = d["snapshot"].replace("\\", "/")
    return d


def _feed_clause(feeds):
    """SQL fragment restricting rows to a set of feeds.

    feeds is None  -> no restriction (admin sees everything)
    feeds is empty -> match nothing (a user with no feeds assigned)
    """
    if feeds is None:
        return "", []
    if not feeds:
        return " AND 1=0", []
    feeds = list(feeds)
    marks = ",".join("?" * len(feeds))
    return f" AND feed_name IN ({marks})", feeds


def fetch_violations(limit=200, feeds=None):
    """Most-recent-first list of violation rows as plain dicts."""
    clause, params = _feed_clause(feeds)
    conn = sqlite3.connect(DB_PATH)
    conn.row_factory = sqlite3.Row
    rows = conn.execute(
        f"SELECT * FROM violations WHERE 1=1{clause} ORDER BY id DESC LIMIT ?",
        (*params, limit)
    ).fetchall()
    conn.close()
    return [_row(r) for r in rows]


def fetch_snapshots(feeds=None, feed=None, violation=None, date_from=None,
                    date_to=None, limit=120, offset=0):
    """Violation rows that have a snapshot image, with gallery filters.

    feeds     - permission restriction (None = unrestricted)
    feed      - filter to one feed chosen in the UI
    violation - 'no_helmet' | 'no_vest'
    date_from / date_to - inclusive YYYY-MM-DD bounds
    Returns (rows, total_matching).
    """
    clause, params = _feed_clause(feeds)
    where = f"WHERE snapshot IS NOT NULL AND snapshot != ''{clause}"
    if feed:
        where += " AND feed_name = ?"; params.append(feed)
    if violation:
        where += " AND violation = ?"; params.append(violation)
    if date_from:
        where += " AND date >= ?"; params.append(date_from)
    if date_to:
        where += " AND date <= ?"; params.append(date_to)

    conn = sqlite3.connect(DB_PATH)
    conn.row_factory = sqlite3.Row
    total = conn.execute(f"SELECT COUNT(*) c FROM violations {where}",
                         params).fetchone()["c"]
    rows = conn.execute(
        f"SELECT * FROM violations {where} ORDER BY id DESC LIMIT ? OFFSET ?",
        (*params, limit, offset)
    ).fetchall()
    conn.close()
    return [_row(r) for r in rows], total


def fetch_violations_by_feed(feeds=None, per_feed=12):
    """Recent violations grouped per feed, for the log under each feed tile.

    One round trip instead of one query per camera: a window function picks
    the newest `per_feed` rows within each feed, and a second query gets the
    running totals. Returns {feed_name: {items, total, today}}.
    """
    clause, params = _feed_clause(feeds)
    today = datetime.now().strftime("%Y-%m-%d")

    conn = sqlite3.connect(DB_PATH)
    conn.row_factory = sqlite3.Row
    rows = conn.execute(f"""
        SELECT * FROM (
            SELECT *, ROW_NUMBER() OVER (PARTITION BY feed_name ORDER BY id DESC) AS rn
            FROM violations WHERE 1=1{clause}
        ) WHERE rn <= ?
        ORDER BY id DESC
    """, (*params, per_feed)).fetchall()

    counts = conn.execute(f"""
        SELECT feed_name,
               COUNT(*) AS total,
               SUM(CASE WHEN date = ? THEN 1 ELSE 0 END) AS today
        FROM violations WHERE 1=1{clause}
        GROUP BY feed_name
    """, (today, *params)).fetchall()
    conn.close()

    out = {}
    for c in counts:
        out[c["feed_name"]] = {"total": c["total"], "today": c["today"] or 0,
                               "items": []}
    for r in rows:
        d = _row(r)
        d.pop("rn", None)
        out.setdefault(d["feed_name"],
                       {"total": 0, "today": 0, "items": []})["items"].append(d)
    return out


def snapshot_feed(filename):
    """Which feed a snapshot file belongs to (for permission checks).

    Matches on the basename because stored paths may use either slash style
    depending on the OS that wrote the row.
    """
    conn = sqlite3.connect(DB_PATH)
    conn.row_factory = sqlite3.Row
    row = conn.execute(
        "SELECT feed_name FROM violations WHERE snapshot LIKE ? ORDER BY id DESC LIMIT 1",
        ("%" + filename.replace("\\", "/").split("/")[-1],)
    ).fetchone()
    conn.close()
    return row["feed_name"] if row else None


def fetch_stats(feeds=None):
    """Aggregate counters for the dashboard header + charts."""
    clause, params = _feed_clause(feeds)
    conn = sqlite3.connect(DB_PATH)
    conn.row_factory = sqlite3.Row
    total = conn.execute(
        f"SELECT COUNT(*) c FROM violations WHERE 1=1{clause}", params
    ).fetchone()["c"]
    today = conn.execute(
        f"SELECT COUNT(*) c FROM violations WHERE date = ?{clause}",
        (datetime.now().strftime("%Y-%m-%d"), *params)
    ).fetchone()["c"]
    by_feed = conn.execute(
        f"SELECT feed_name, COUNT(*) c FROM violations WHERE 1=1{clause} GROUP BY feed_name",
        params
    ).fetchall()
    by_type = conn.execute(
        f"SELECT violation, COUNT(*) c FROM violations WHERE 1=1{clause} GROUP BY violation",
        params
    ).fetchall()
    conn.close()
    return {
        "total": total,
        "today": today,
        "by_feed": {r["feed_name"]: r["c"] for r in by_feed},
        "by_type": {r["violation"]: r["c"] for r in by_type},
    }
