"""
Accounts, roles and per-feed permissions.
=========================================
Two roles:
  admin - sees every feed, manages feeds and users, controls playback
  user  - sees ONLY the feeds an admin assigned, view-only

Passwords are never stored in plain text; werkzeug hashes them (scrypt).
Sessions are signed cookies keyed by the secret in config.SECRET_KEY_FILE.
"""
import os
import sqlite3
import secrets
from datetime import datetime
from functools import wraps

from flask import session, redirect, url_for, request, jsonify, g
from werkzeug.security import generate_password_hash, check_password_hash

from config import (DB_PATH, SECRET_KEY_FILE, DEFAULT_ADMIN_USER,
                    DEFAULT_ADMIN_PASS)

_HERE = os.path.dirname(os.path.abspath(__file__))


# ---------------------------------------------------------------------------
# SETUP
# ---------------------------------------------------------------------------
def get_secret_key():
    """Load the cookie-signing key, creating it on first run."""
    path = os.path.join(_HERE, SECRET_KEY_FILE)
    if os.path.exists(path):
        with open(path) as f:
            key = f.read().strip()
            if key:
                return key
    key = secrets.token_hex(32)
    with open(path, "w") as f:
        f.write(key)
    return key


def _conn():
    conn = sqlite3.connect(DB_PATH)
    conn.row_factory = sqlite3.Row
    return conn


def init_auth_db():
    """Create the user tables and bootstrap the first admin if none exist."""
    conn = _conn()
    conn.execute("""
        CREATE TABLE IF NOT EXISTS users (
            id            INTEGER PRIMARY KEY AUTOINCREMENT,
            username      TEXT NOT NULL UNIQUE,
            password_hash TEXT NOT NULL,
            role          TEXT NOT NULL DEFAULT 'user',   -- 'admin' | 'user'
            created_at    TEXT NOT NULL
        )
    """)
    conn.execute("""
        CREATE TABLE IF NOT EXISTS user_feeds (
            user_id   INTEGER NOT NULL,
            feed_name TEXT NOT NULL,
            PRIMARY KEY (user_id, feed_name)
        )
    """)
    conn.commit()

    n = conn.execute("SELECT COUNT(*) c FROM users").fetchone()["c"]
    conn.close()

    if n == 0:
        create_user(DEFAULT_ADMIN_USER, DEFAULT_ADMIN_PASS, "admin")
        print("=" * 68)
        print(f"  Created default admin account:  {DEFAULT_ADMIN_USER} / {DEFAULT_ADMIN_PASS}")
        print("  CHANGE THIS PASSWORD from the Users panel after logging in.")
        print("=" * 68)


# ---------------------------------------------------------------------------
# USER CRUD
# ---------------------------------------------------------------------------
def create_user(username, password, role="user", feeds=None):
    """Add a user. Returns (ok, error). `feeds` is a list of feed names."""
    username = (username or "").strip()
    if not username:
        return False, "username is required"
    if not password:
        return False, "password is required"
    if role not in ("admin", "user"):
        return False, "role must be admin or user"
    conn = _conn()
    try:
        cur = conn.execute(
            "INSERT INTO users (username, password_hash, role, created_at) VALUES (?,?,?,?)",
            (username, generate_password_hash(password), role,
             datetime.now().isoformat(timespec="seconds")))
        uid = cur.lastrowid
        for fn in (feeds or []):
            conn.execute("INSERT OR IGNORE INTO user_feeds (user_id, feed_name) VALUES (?,?)",
                         (uid, fn))
        conn.commit()
        return True, None
    except sqlite3.IntegrityError:
        return False, f"user '{username}' already exists"
    finally:
        conn.close()


def delete_user(uid):
    conn = _conn()
    row = conn.execute("SELECT role FROM users WHERE id=?", (uid,)).fetchone()
    if not row:
        conn.close()
        return False, "no such user"
    # never allow removing the last admin -- you'd lock yourself out
    if row["role"] == "admin":
        n = conn.execute("SELECT COUNT(*) c FROM users WHERE role='admin'").fetchone()["c"]
        if n <= 1:
            conn.close()
            return False, "cannot delete the only admin account"
    conn.execute("DELETE FROM user_feeds WHERE user_id=?", (uid,))
    conn.execute("DELETE FROM users WHERE id=?", (uid,))
    conn.commit()
    conn.close()
    return True, None


def set_password(uid, password):
    if not password:
        return False, "password is required"
    conn = _conn()
    conn.execute("UPDATE users SET password_hash=? WHERE id=?",
                 (generate_password_hash(password), uid))
    conn.commit()
    conn.close()
    return True, None


def set_user_feeds(uid, feeds):
    """Replace a user's feed permissions with the given list."""
    conn = _conn()
    conn.execute("DELETE FROM user_feeds WHERE user_id=?", (uid,))
    for fn in (feeds or []):
        conn.execute("INSERT OR IGNORE INTO user_feeds (user_id, feed_name) VALUES (?,?)",
                     (uid, fn))
    conn.commit()
    conn.close()
    return True, None


def get_user_by_id(uid):
    conn = _conn()
    row = conn.execute("SELECT id, username, role FROM users WHERE id=?", (uid,)).fetchone()
    conn.close()
    return dict(row) if row else None


def list_users():
    """All users plus the feeds each one is allowed to see."""
    conn = _conn()
    users = [dict(r) for r in conn.execute(
        "SELECT id, username, role, created_at FROM users ORDER BY id").fetchall()]
    perms = conn.execute("SELECT user_id, feed_name FROM user_feeds").fetchall()
    conn.close()
    by_uid = {}
    for p in perms:
        by_uid.setdefault(p["user_id"], []).append(p["feed_name"])
    for u in users:
        u["feeds"] = sorted(by_uid.get(u["id"], []))
    return users


def verify_login(username, password):
    """Return the user dict on success, None on bad credentials."""
    conn = _conn()
    row = conn.execute("SELECT * FROM users WHERE username=?",
                       ((username or "").strip(),)).fetchone()
    conn.close()
    if row and check_password_hash(row["password_hash"], password or ""):
        return {"id": row["id"], "username": row["username"], "role": row["role"]}
    return None


# ---------------------------------------------------------------------------
# PERMISSIONS
# ---------------------------------------------------------------------------
def current_user():
    """The logged-in user for this request, or None. Cached on `g`."""
    if "user" not in g:
        uid = session.get("uid")
        g.user = get_user_by_id(uid) if uid else None
    return g.user


def is_admin(user=None):
    u = user or current_user()
    return bool(u and u["role"] == "admin")


def allowed_feeds(user=None):
    """Feed names this user may see. None means 'no restriction' (admin)."""
    u = user or current_user()
    if u is None:
        return set()
    if u["role"] == "admin":
        return None
    conn = _conn()
    rows = conn.execute("SELECT feed_name FROM user_feeds WHERE user_id=?",
                        (u["id"],)).fetchall()
    conn.close()
    return {r["feed_name"] for r in rows}


def can_access_feed(name, user=None):
    allowed = allowed_feeds(user)
    return allowed is None or name in allowed


# ---------------------------------------------------------------------------
# DECORATORS
# ---------------------------------------------------------------------------
def _deny(msg, code):
    """API callers get JSON; browsers get bounced to the login page."""
    if request.path.startswith(("/api/", "/stream/", "/snapshots/")):
        return jsonify({"error": msg}), code
    if code == 401:
        return redirect(url_for("login", next=request.full_path))
    return msg, code


def login_required(f):
    @wraps(f)
    def wrapper(*a, **kw):
        if current_user() is None:
            return _deny("authentication required", 401)
        return f(*a, **kw)
    return wrapper


def admin_required(f):
    @wraps(f)
    def wrapper(*a, **kw):
        if current_user() is None:
            return _deny("authentication required", 401)
        if not is_admin():
            return _deny("admin privileges required", 403)
        return f(*a, **kw)
    return wrapper
