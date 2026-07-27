"""
Web layer.
==========
The Flask app: HTTP routes, the in-memory feed registry, and helpers to
add/remove/persist feeds. Knows about HTTP and FeedWorkers, but contains no
SQL or OpenCV code -- those live in db.py and detection.py.

Runtime state (MODEL, DEVICE) is set by main.py at startup:
    import server
    server.MODEL = YOLO(...); server.DEVICE = "0"
"""
import os
import csv
import io
import json
import uuid
import time
import threading

from flask import (Flask, Response, jsonify, render_template,
                   send_from_directory, request)
from werkzeug.utils import secure_filename

from config import (SNAPSHOT_DIR, UPLOAD_DIR, FEEDS_CONFIG_PATH, ALLOWED_VIDEO_EXT)
from detection import FeedWorker
from db import fetch_violations, fetch_stats

_HERE = os.path.dirname(os.path.abspath(__file__))
app = Flask(__name__, template_folder=os.path.join(_HERE, "templates"))

# ---- Shared runtime state (populated by main.py) ---------------------------
WORKERS = {}                 # name -> FeedWorker
FEEDS_LOCK = threading.Lock()
MODEL = None                 # the loaded YOLO model
DEVICE = "cpu"               # inference device string


# ---------------------------------------------------------------------------
# FEED REGISTRY
# ---------------------------------------------------------------------------
def _feeds_snapshot():
    """Current feed configs (for persisting to feeds.json)."""
    out = []
    for name, w in WORKERS.items():
        entry = {"name": name, "source": w.source}
        if w.resolution:
            entry["resolution"] = list(w.resolution)
        out.append(entry)
    return out


def save_feeds_config():
    try:
        with open(FEEDS_CONFIG_PATH, "w") as f:
            json.dump(_feeds_snapshot(), f, indent=2)
    except Exception as e:
        print("WARNING: could not save feeds.json:", e)


def add_feed(name, source, resolution=None):
    """Create + start a FeedWorker. Returns (ok, error_message)."""
    if MODEL is None:
        return False, "model not loaded"
    name = name.strip()
    if not name:
        return False, "feed name is required"
    with FEEDS_LOCK:
        if name in WORKERS:
            return False, f"a feed named '{name}' already exists"
        w = FeedWorker(name, source, MODEL, device=DEVICE, resolution=resolution)
        WORKERS[name] = w
        w.start()
        save_feeds_config()
    print(f"Started feed: {name}  <- {source}")
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
# ROUTES
# ---------------------------------------------------------------------------
@app.route("/")
def index():
    return render_template("dashboard.html")


@app.route("/stream/<name>")
def stream(name):
    w = WORKERS.get(name)
    if not w:
        return "no such feed", 404
    return Response(mjpeg_generator(w),
                    mimetype="multipart/x-mixed-replace; boundary=frame")


@app.route("/api/violations")
def api_violations():
    return jsonify(fetch_violations())


@app.route("/api/stats")
def api_stats():
    return jsonify(fetch_stats())


@app.route("/api/feeds", methods=["GET"])
def api_feeds_list():
    with FEEDS_LOCK:
        out = [{"name": n, "source": w.source, **w.state()}
               for n, w in WORKERS.items()]
    return jsonify(out)


@app.route("/api/feeds", methods=["POST"])
def api_feeds_add():
    """
    Accepts multipart/form-data (works whether or not a file is attached):
      name          - feed display name (required)
      source_type   - 'camera' | 'video' | 'rtsp'  (required)
      camera_index  - int, required if source_type == 'camera'
      rtsp_url      - string, required if source_type == 'rtsp'
      video         - file upload, required if source_type == 'video'
      width, height - optional resolution override
    """
    name = (request.form.get("name") or "").strip()
    source_type = (request.form.get("source_type") or "").strip().lower()

    if not name:
        return jsonify({"error": "feed name is required"}), 400
    if source_type not in ("camera", "video", "rtsp"):
        return jsonify({"error": "source_type must be camera, video, or rtsp"}), 400

    source = None
    if source_type == "camera":
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

    ok, err = add_feed(name, source, resolution=resolution)
    if not ok:
        return jsonify({"error": err}), 400
    return jsonify({"ok": True, "name": name}), 201


@app.route("/api/feeds/<name>", methods=["DELETE"])
def api_feeds_delete(name):
    ok, err = remove_feed(name)
    if not ok:
        return jsonify({"error": err}), 404
    return jsonify({"ok": True})


@app.route("/api/feeds/<name>/control", methods=["POST"])
def api_feeds_control(name):
    """Playback control for a single feed. JSON body: {action, value?}.
      action: pause | play | toggle | step | restart | seek | speed
      value : seconds (seek) or multiplier (speed)
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
    else:
        return jsonify({"error": f"unknown action: {action}"}), 400

    return jsonify(w.state())


@app.route("/snapshots/<path:fn>")
def snap(fn):
    return send_from_directory(SNAPSHOT_DIR, fn)


@app.route("/export.csv")
def export_csv():
    rows = fetch_violations(limit=100000)
    buf = io.StringIO()
    if rows:
        w = csv.DictWriter(buf, fieldnames=rows[0].keys())
        w.writeheader(); w.writerows(rows)
    return Response(buf.getvalue(), mimetype="text/csv",
                    headers={"Content-Disposition": "attachment; filename=violations.csv"})
