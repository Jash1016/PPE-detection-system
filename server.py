"""
Web layer.
==========
The Flask app: HTTP routes, the in-memory feed registry, and helpers to
add/remove/persist feeds. Knows about HTTP and FeedWorkers, but contains no
SQL or OpenCV code -- those live in db.py and detection.py.

Runtime state (MODEL, MODEL_PATH, DEVICE) is set by main.py at startup:
    import server
    server.MODEL = YOLO(...); server.MODEL_PATH = "my_model.pt"
    server.DEVICE = "0"
"""
import os
import csv
import io
import json
import uuid
import time
import threading

from datetime import timedelta

from flask import (Flask, Response, jsonify, render_template, redirect,
                   send_from_directory, request, session, url_for)
from werkzeug.utils import secure_filename

from config import (SNAPSHOT_DIR, UPLOAD_DIR, FEEDS_CONFIG_PATH,
                    ALLOWED_VIDEO_EXT, SESSION_HOURS, MODELS, DEFAULT_MODEL)
from detection import FeedWorker, build_model
from db import (fetch_violations, fetch_stats, fetch_snapshots, snapshot_feed,
                fetch_violations_by_feed)
import cctv
import auth
from auth import (login_required, admin_required, current_user, is_admin,
                  allowed_feeds, can_access_feed)

_HERE = os.path.dirname(os.path.abspath(__file__))
app = Flask(__name__, template_folder=os.path.join(_HERE, "templates"))
app.secret_key = auth.get_secret_key()
app.config.update(
    SESSION_COOKIE_HTTPONLY=True,      # JS can't read the cookie
    SESSION_COOKIE_SAMESITE="Lax",
    PERMANENT_SESSION_LIFETIME=timedelta(hours=SESSION_HOURS),
)

# ---- Shared runtime state (populated by main.py) ---------------------------
WORKERS = {}                 # name -> FeedWorker
FEEDS_LOCK = threading.Lock()
MODEL = None                 # the loaded YOLO model (startup probe / readiness)
MODEL_PATH = None            # weights path, so each feed can load its own copy
DEVICE = "cpu"               # inference device string


# ---------------------------------------------------------------------------
# FEED REGISTRY
# ---------------------------------------------------------------------------
def _feeds_snapshot():
    """Current feed configs (for persisting to feeds.json)."""
    out = []
    for name, w in WORKERS.items():
        entry = {"name": name, "source": w.source, "model": w.model_key}
        if w.resolution:
            entry["resolution"] = list(w.resolution)
        out.append(entry)
    return out


def model_path(key):
    """Absolute path to a registry model's weights (they sit beside the code)."""
    path = MODELS[key]["path"]
    return path if os.path.isabs(path) else os.path.join(_HERE, path)


def resolve_model(key):
    """Map a registry key to (weights path, label mode, the key it resolved to).

    An unknown key or a missing .pt falls back to the startup model instead of
    failing: a name left over in feeds.json, or a model file that hasn't been
    copied across yet, should not stop a camera from coming up.
    """
    if key in MODELS:
        path = model_path(key)
        if os.path.exists(path):
            return path, MODELS[key]["labels"], key
        print(f"WARNING: model '{key}' not found at {path}; using the default")
    # MODEL_PATH may be a --model the registry has never heard of, so read it
    # with the default entry's scheme.
    return MODEL_PATH, MODELS[DEFAULT_MODEL]["labels"], DEFAULT_MODEL


def save_feeds_config():
    try:
        with open(FEEDS_CONFIG_PATH, "w") as f:
            json.dump(_feeds_snapshot(), f, indent=2)
    except Exception as e:
        print("WARNING: could not save feeds.json:", e)


def add_feed(name, source, resolution=None, model_key=None):
    """Create + start a FeedWorker. Returns (ok, error_message)."""
    if MODEL is None or not MODEL_PATH:
        return False, "model not loaded"
    name = name.strip()
    if not name:
        return False, "feed name is required"
    path, label_mode, key = resolve_model(model_key or DEFAULT_MODEL)
    with FEEDS_LOCK:
        if name in WORKERS:
            return False, f"a feed named '{name}' already exists"
        # A private model per worker, NOT the shared server.MODEL: ultralytics
        # hangs tracker state off model.predictor, so sharing one object makes
        # every camera thread mutate the same track-ID pool. Costs one more
        # copy of the weights per feed (see build_model).
        feed_model, feed_device = build_model(path, DEVICE)
        w = FeedWorker(name, source, feed_model, device=feed_device,
                       resolution=resolution, label_mode=label_mode,
                       model_key=key)
        WORKERS[name] = w
        w.start()
        save_feeds_config()
    print(f"Started feed: {name}  <- {source}  [model: {key}]")
    return True, None


def switch_model(name, model_key):
    """Point a running feed at a different model. Returns (ok, error_message)."""
    if MODEL is None or not MODEL_PATH:
        return False, "model not loaded"
    if model_key not in MODELS:
        return False, f"unknown model: {model_key}"
    w = WORKERS.get(name)
    if not w:
        return False, "no such feed"
    if w.model_key == model_key:
        return True, None
    path, label_mode, key = resolve_model(model_key)
    # Built before touching the worker: loading weights takes long enough that
    # doing it while holding the worker's model lock would stall its frames.
    feed_model, feed_device = build_model(path, DEVICE)
    w.set_model(feed_model, label_mode, key, feed_device)
    save_feeds_config()
    print(f"Feed '{name}' switched to model: {key}")
    return True, None


def remove_feed(name):
    with FEEDS_LOCK:
        w = WORKERS.pop(name, None)
        if not w:
            return False, "no such feed"
        w.running = False
        save_feeds_config()
    return True, None


def mjpeg_generator(worker):
    """Yield the worker's latest JPEG as a multipart MJPEG stream (~25 fps)."""
    while True:
        jpeg = worker.get_jpeg()
        if jpeg is not None:
            yield (b"--frame\r\nContent-Type: image/jpeg\r\n\r\n" + jpeg + b"\r\n")
        time.sleep(0.04)


# ---------------------------------------------------------------------------
# AUTH ROUTES
# ---------------------------------------------------------------------------
@app.route("/login", methods=["GET", "POST"])
def login():
    if request.method == "GET":
        if current_user():
            return redirect(url_for("index"))
        return render_template("login.html", error=None)

    user = auth.verify_login(request.form.get("username"),
                             request.form.get("password"))
    if not user:
        return render_template("login.html",
                               error="Incorrect username or password."), 401
    session.clear()
    session["uid"] = user["id"]
    session.permanent = True
    nxt = request.args.get("next") or url_for("index")
    if not nxt.startswith("/"):       # never redirect off-site
        nxt = url_for("index")
    return redirect(nxt)


@app.route("/logout")
def logout():
    session.clear()
    return redirect(url_for("login"))


@app.route("/api/me")
@login_required
def api_me():
    u = current_user()
    return jsonify({"username": u["username"], "role": u["role"],
                    "is_admin": is_admin()})


# ---------------------------------------------------------------------------
# USER MANAGEMENT (admin only)
# ---------------------------------------------------------------------------
@app.route("/admin")
@admin_required
def admin_page():
    return render_template("admin.html")


@app.route("/api/users", methods=["GET"])
@admin_required
def api_users_list():
    return jsonify({"users": auth.list_users(),
                    "feeds": sorted(WORKERS.keys())})


@app.route("/api/users", methods=["POST"])
@admin_required
def api_users_add():
    d = request.get_json(silent=True) or {}
    ok, err = auth.create_user(d.get("username"), d.get("password"),
                               d.get("role", "user"), d.get("feeds") or [])
    if not ok:
        return jsonify({"error": err}), 400
    return jsonify({"ok": True}), 201


@app.route("/api/users/<int:uid>", methods=["DELETE"])
@admin_required
def api_users_delete(uid):
    if current_user()["id"] == uid:
        return jsonify({"error": "you cannot delete your own account"}), 400
    ok, err = auth.delete_user(uid)
    if not ok:
        return jsonify({"error": err}), 400
    return jsonify({"ok": True})


@app.route("/api/users/<int:uid>", methods=["PATCH"])
@admin_required
def api_users_update(uid):
    """Change a user's password and/or their allowed feeds."""
    d = request.get_json(silent=True) or {}
    if "password" in d:
        ok, err = auth.set_password(uid, d["password"])
        if not ok:
            return jsonify({"error": err}), 400
    if "feeds" in d:
        auth.set_user_feeds(uid, d["feeds"])
    return jsonify({"ok": True})


# ---------------------------------------------------------------------------
# ROUTES
# ---------------------------------------------------------------------------
@app.route("/")
@login_required
def index():
    return render_template("dashboard.html")


@app.route("/stream/<name>")
@login_required
def stream(name):
    if not can_access_feed(name):
        return "forbidden", 403
    w = WORKERS.get(name)
    if not w:
        return "no such feed", 404
    return Response(mjpeg_generator(w),
                    mimetype="multipart/x-mixed-replace; boundary=frame")


@app.route("/api/violations")
@login_required
def api_violations():
    return jsonify(fetch_violations(feeds=allowed_feeds()))


@app.route("/api/stats")
@login_required
def api_stats():
    return jsonify(fetch_stats(feeds=allowed_feeds()))


@app.route("/api/violations/by-feed")
@login_required
def api_violations_by_feed():
    """Recent violations grouped per feed, for the log under each tile."""
    try:
        per_feed = min(50, max(1, int(request.args.get("per_feed", 12))))
    except ValueError:
        return jsonify({"error": "per_feed must be an integer"}), 400
    return jsonify(fetch_violations_by_feed(feeds=allowed_feeds(),
                                            per_feed=per_feed))


@app.route("/api/snapshots")
@login_required
def api_snapshots():
    """Paged snapshot gallery, filtered by the caller's permissions."""
    try:
        limit = min(300, max(1, int(request.args.get("limit", 60))))
        offset = max(0, int(request.args.get("offset", 0)))
    except ValueError:
        return jsonify({"error": "limit/offset must be integers"}), 400

    rows, total = fetch_snapshots(
        feeds=allowed_feeds(),
        feed=request.args.get("feed") or None,
        violation=request.args.get("violation") or None,
        date_from=request.args.get("from") or None,
        date_to=request.args.get("to") or None,
        limit=limit, offset=offset)
    return jsonify({"items": rows, "total": total,
                    "limit": limit, "offset": offset})


@app.route("/api/feeds", methods=["GET"])
@login_required
def api_feeds_list():
    allowed = allowed_feeds()
    with FEEDS_LOCK:
        # mask_url hides the camera password -- the real URL never leaves
        # the server, so no logged-in user can read DVR credentials.
        out = [{"name": n, "source": cctv.mask_url(w.source), **w.state()}
               for n, w in WORKERS.items()
               if allowed is None or n in allowed]
    return jsonify(out)


@app.route("/api/models")
@login_required
def api_models():
    """The models a feed can be pointed at, for the pickers in the dashboard."""
    return jsonify([
        {"id": key, "label": m["label"], "hint": m.get("hint", ""),
         "default": key == DEFAULT_MODEL,
         # a registry entry whose .pt was never copied over is shown but not
         # offered, which beats a feed silently starting on the wrong model
         "available": os.path.exists(model_path(key))}
        for key, m in MODELS.items()
    ])


@app.route("/api/cctv/presets")
@admin_required
def api_cctv_presets():
    """Brand presets for the CCTV tab of the add-feed dialog."""
    return jsonify([
        {"id": k, "label": v["label"], "port": v["port"],
         "channels": v["channels"], "custom": v["path"] is None}
        for k, v in cctv.PRESETS.items()
    ])


@app.route("/api/feeds", methods=["POST"])
@admin_required
def api_feeds_add():
    """
    Accepts multipart/form-data (works whether or not a file is attached):
      name          - feed display name (required)
      source_type   - 'camera' | 'video' | 'rtsp'  (required)
      camera_index  - int, required if source_type == 'camera'
      rtsp_url      - string, required if source_type == 'rtsp'
      video         - file upload, required if source_type == 'video'
      width, height - optional resolution override
      model         - registry key from /api/models (optional; DEFAULT_MODEL)
    """
    name = (request.form.get("name") or "").strip()
    source_type = (request.form.get("source_type") or "").strip().lower()

    if not name:
        return jsonify({"error": "feed name is required"}), 400
    if source_type not in ("camera", "video", "rtsp", "cctv"):
        return jsonify({"error": "source_type must be camera, video, rtsp, or cctv"}), 400

    source = None
    if source_type == "cctv":
        # IP / DVR / NVR / phone camera -- we build the stream URL from parts
        f = request.form
        brand = (f.get("brand") or "").strip().lower()
        ip = (f.get("ip") or "").strip()
        port = (f.get("port") or "").strip() or None
        user = (f.get("username") or "").strip() or None
        pw = f.get("password") or None
        channel = (f.get("channel") or "1").strip()
        stream = (f.get("stream") or "sub").strip().lower()

        if brand == "onvif":
            source, err = cctv.onvif_stream_url(ip, port or 80, user, pw)
        else:
            source, err = cctv.build_url(
                brand, ip, port=port, username=user, password=pw,
                channel=channel, stream=stream,
                custom_path=f.get("custom_path"))
        if err:
            return jsonify({"error": err}), 400

    elif source_type == "camera":
        idx = request.form.get("camera_index", "0")
        if not idx.isdigit():
            return jsonify({"error": "camera_index must be a non-negative integer"}), 400
        source = f"usb{idx}"

    elif source_type == "rtsp":
        url = (request.form.get("rtsp_url") or "").strip()
        if not url.lower().startswith("rtsp://"):
            return jsonify({"error": "rtsp_url must start with rtsp://"}), 400
        source = url

    elif source_type == "video":
        file = request.files.get("video")
        if not file or not file.filename:
            return jsonify({"error": "a video file is required for source_type=video"}), 400
        ext = os.path.splitext(file.filename)[1].lower()
        if ext not in ALLOWED_VIDEO_EXT:
            return jsonify({"error": f"unsupported video extension: {ext}"}), 400
        os.makedirs(UPLOAD_DIR, exist_ok=True)
        safe_name = secure_filename(file.filename)
        stored_name = f"{uuid.uuid4().hex[:8]}_{safe_name}"
        path = os.path.join(UPLOAD_DIR, stored_name)
        file.save(path)
        source = path

    resolution = None
    w_str, h_str = request.form.get("width"), request.form.get("height")
    if w_str and h_str and w_str.isdigit() and h_str.isdigit():
        resolution = (int(w_str), int(h_str))

    model_key = (request.form.get("model") or "").strip() or None
    if model_key and model_key not in MODELS:
        return jsonify({"error": f"unknown model: {model_key}"}), 400

    ok, err = add_feed(name, source, resolution=resolution, model_key=model_key)
    if not ok:
        return jsonify({"error": err}), 400
    return jsonify({"ok": True, "name": name}), 201


@app.route("/api/feeds/<name>", methods=["DELETE"])
@admin_required
def api_feeds_delete(name):
    ok, err = remove_feed(name)
    if not ok:
        return jsonify({"error": err}), 404
    return jsonify({"ok": True})


@app.route("/api/feeds/<name>/control", methods=["POST"])
@admin_required
def api_feeds_control(name):
    """Playback control for a single feed. JSON body: {action, value?}.
      action: pause | play | toggle | step | restart | seek | speed | model
      value : seconds (seek), multiplier (speed) or a /api/models id (model)
    Returns the worker's new state.
    """
    w = WORKERS.get(name)
    if not w:
        return jsonify({"error": "no such feed"}), 404
    data = request.get_json(silent=True) or {}
    action = (data.get("action") or "").lower()

    if action == "pause":
        w.set_paused(True)
    elif action == "play":
        w.set_paused(False)
    elif action == "toggle":
        w.toggle_pause()
    elif action == "step":
        w.step()
    elif action == "restart":
        w.restart()
    elif action == "seek":
        w.seek(data.get("value", 0))
    elif action == "speed":
        w.set_speed(data.get("value", 1))
    elif action == "model":
        ok, err = switch_model(name, str(data.get("value") or ""))
        if not ok:
            return jsonify({"error": err}), 400
    else:
        return jsonify({"error": f"unknown action: {action}"}), 400

    return jsonify(w.state())


@app.route("/snapshots/<path:fn>")
@login_required
def snap(fn):
    # a non-admin may only open snapshots belonging to their own feeds
    if not is_admin():
        owner = snapshot_feed(fn)
        if owner is None or not can_access_feed(owner):
            return "forbidden", 403
    return send_from_directory(SNAPSHOT_DIR, fn)


@app.route("/export.csv")
@login_required
def export_csv():
    rows = fetch_violations(limit=100000, feeds=allowed_feeds())
    buf = io.StringIO()
    if rows:
        w = csv.DictWriter(buf, fieldnames=rows[0].keys())
        w.writeheader(); w.writerows(rows)
    return Response(buf.getvalue(), mimetype="text/csv",
                    headers={"Content-Disposition": "attachment; filename=violations.csv"})
