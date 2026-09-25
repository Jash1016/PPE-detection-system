"""
Detection + tracking.
=====================
The computer-vision half of the app: turning a video source into annotated
frames and violation records. Knows nothing about Flask or HTTP.
"""
import os
import time
import threading
from datetime import datetime

import cv2

# RTSP over TCP. UDP loses packets on Wi-Fi, which shows up as smeared or
# green frames and can kill the stream. Must be set before the first capture.
os.environ.setdefault("OPENCV_FFMPEG_CAPTURE_OPTIONS",
                      "rtsp_transport;tcp|stimeout;5000000")

from ultralytics import YOLO

from config import (SNAPSHOT_DIR, MIN_THRESH, LOG_THRESH, TRACK_RELOG_AFTER,
                    FEED_RELOG_COOLDOWN, TRACKER_CFG, HELMET_KEYS, VEST_KEYS,
                    NEG_KEYS, HEAD_KEYS, GEAR_OVERLAP_MIN, TORSO_WIDTH_HEADS,
                    TORSO_HEIGHT_HEADS, BBOX_COLORS)
from db import log_violation


def build_model(model_path, device="cpu"):
    """Construct a YOLO instance and move it to `device`.

    Returns (model, device) -- device comes back as "cpu" when the move to GPU
    fails, so the caller stops handing inference to a device that isn't there.

    Every FeedWorker gets its OWN instance from here. Ultralytics keeps tracker
    state on model.predictor, so one shared model means every camera thread
    writes into a single track-ID pool with no synchronisation: IDs collide,
    get reassigned, and climb into the thousands within minutes.
    """
    model = YOLO(model_path, task="detect")
    if device != "cpu":
        try:
            model.to(f"cuda:{device}" if device.isdigit() else device)
        except Exception as e:
            print("Could not move model to GPU, using CPU:", e)
            device = "cpu"
    return model, device


def _norm(s):
    """Lowercase and strip spaces/underscores/hyphens for robust matching."""
    return s.lower().replace(" ", "").replace("_", "").replace("-", "")


def classify_label(raw_name):
    """
    Return 'no_helmet', 'no_vest', or None for a given raw model label.
    Only the negative (violation) classes return a value.

    A label counts as a violation only if -- after normalizing -- it starts
    with a negative token (no/without/missing) AND mentions the item. So
    "no safty vests", "no_vest", "no-vest", "novest" all map to 'no_vest'.
    """
    n = _norm(raw_name)
    is_neg = any(n.startswith(k) for k in NEG_KEYS)
    if not is_neg:
        return None
    if any(k in n for k in HELMET_KEYS):
        return "no_helmet"
    if any(k in n for k in VEST_KEYS):
        return "no_vest"
    return None


# ---------------------------------------------------------------------------
# POSITIVE-LABEL MODELS (head / helmet / vest)
# ---------------------------------------------------------------------------
def _gear_kind(raw_name):
    """Bucket a positive-model label as 'head', 'helmet', 'vest', or None.

    Checked in this order because the buckets are not mutually exclusive in
    every label set -- "hardhat" must not be read as a head.
    """
    n = _norm(raw_name)
    if any(k in n for k in HELMET_KEYS):
        return "helmet"
    if any(k in n for k in VEST_KEYS):
        return "vest"
    if any(k in n for k in HEAD_KEYS):
        return "head"
    return None


def _overlap_frac(inner, outer):
    """Fraction of `inner`'s area that falls inside `outer`. See GEAR_OVERLAP_MIN."""
    ax0, ay0, ax1, ay1 = inner
    bx0, by0, bx1, by1 = outer
    iw = max(0, min(ax1, bx1) - max(ax0, bx0))
    ih = max(0, min(ay1, by1) - max(ay0, by0))
    area = max(1, (ax1 - ax0) * (ay1 - ay0))
    return (iw * ih) / area


def _torso_box(head, shape):
    """Estimate the torso hanging below a head box, clipped to the frame."""
    h, w = shape[:2]
    x0, y0, x1, y1 = head
    hw, hh = x1 - x0, y1 - y0
    cx = (x0 + x1) / 2.0
    half = hw * TORSO_WIDTH_HEADS / 2.0
    return (int(max(0, cx - half)), int(min(h, y1)),
            int(min(w, cx + half)), int(min(h, y1 + hh * TORSO_HEIGHT_HEADS)))


def derive_violations(dets, shape):
    """Infer violations from a model that only names gear that IS present.

    The close-range model knows 'head', 'helmet' and 'vest' -- an unprotected
    worker is the ABSENCE of a helmet box, not a class of its own, so it has to
    be worked out per frame. Each head is tested against every helmet in the
    frame, and the torso below it against every vest; whatever goes unmatched
    becomes a violation recorded against the HEAD's index, so the box drawn and
    logged is the person rather than a piece of gear.

    Note the confidence carried forward is the head detection's -- how sure the
    model is that a person is there, not how sure it is the helmet is missing.
    There is no score for an absent box, so LOG_THRESH reads as "only log this
    when we are confident there is really a person here".

    Returns {detection index: [vtype, ...]} -- one head can be missing both.
    """
    kinds = [_gear_kind(d["name"]) for d in dets]
    helmets = [d["box"] for d, k in zip(dets, kinds) if k == "helmet"]
    vests = [d["box"] for d, k in zip(dets, kinds) if k == "vest"]

    out = {}
    for i, (d, kind) in enumerate(zip(dets, kinds)):
        if kind != "head":
            continue
        missing = []
        if not any(_overlap_frac(b, d["box"]) >= GEAR_OVERLAP_MIN for b in helmets):
            missing.append("no_helmet")
        torso = _torso_box(d["box"], shape)
        if not any(_overlap_frac(b, torso) >= GEAR_OVERLAP_MIN for b in vests):
            missing.append("no_vest")
        if missing:
            out[i] = missing
    return out


class FeedWorker(threading.Thread):
    """One background thread per camera/video/RTSP source.

    Reads frames, runs YOLO tracking, draws boxes, logs violations, and keeps
    the latest annotated JPEG ready for the web layer to stream.
    """

    def __init__(self, name, source, model, device="cpu", resolution=None,
                 label_mode="negative", model_key=None):
        super().__init__(daemon=True)
        self.name = name
        self.source = source
        self.model = model
        self.device = device
        self.labels = model.names
        # label_mode says how to read this model's classes ("negative" =the
        # model names the violation, "positive" =derive it, see MODELS in
        # config). model_key is the registry name, for the UI and feeds.json.
        self.label_mode = label_mode
        self.model_key = model_key
        # Held only while swapping or reading the model+labels+mode triple, so
        # a frame can never be scored with one model's boxes and another's
        # class names. Separate from self.lock, which guards the JPEG.
        self.model_lock = threading.Lock()
        self.resolution = resolution
        self.latest_jpeg = None
        self.lock = threading.Lock()
        self.running = True
        self.status = "starting"
        # (track_id, violation) -> last time we SAW that id committing that
        # violation (for the re-log timeout). Keyed by the pair, not the id
        # alone, because one person can be missing a helmet AND a vest and
        # each deserves its own row.
        self.logged_tracks = {}
        # violation type -> last time we LOGGED that type on this feed. Guards
        # against ID churn, which makes logged_tracks think every flicker is a
        # new person (see FEED_RELOG_COOLDOWN).
        self.logged_types = {}

        # ---- playback controls (driven by the dashboard) -------------------
        # Seek/speed/step only apply to video FILES; live cams ignore them.
        self.is_file = isinstance(source, str) and os.path.isfile(source)
        self.is_network = isinstance(source, str) and source.startswith(
            ("rtsp://", "http://", "https://"))
        self.paused = False          # freeze the feed (keeps last frame)
        self.speed = 1.0             # playback multiplier for files
        self.fps = 25.0              # source fps, filled in once cap is open
        self._seek_frames = None     # pending relative seek, in frames
        self._step = False           # advance exactly one frame while paused

    def _open(self):
        """Open the capture. 'usbN' strings become integer camera indices."""
        src = self.source
        if isinstance(src, str) and src.startswith("usb"):
            src = int(src[3:])
        cap = cv2.VideoCapture(src)
        if self.is_network:
            # Keep only the newest frame. Without this the driver queues
            # frames while YOLO is busy and the "live" view drifts further
            # and further behind reality.
            try:
                cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)
            except Exception:
                pass
        return cap

    def run(self):
        cap = self._open()
        if not cap.isOpened():
            self.status = "error: cannot open source"
            print(f"[{self.name}] ERROR: cannot open {self.source}")
            return
        if self.resolution:
            w, h = self.resolution
            cap.set(3, w); cap.set(4, h)
        fps = cap.get(cv2.CAP_PROP_FPS)
        self.fps = fps if fps and fps > 1 else 25.0
        self.status = "running"

        while self.running:
            # apply a pending relative seek (files only)
            if self._seek_frames is not None and self.is_file:
                cur = cap.get(cv2.CAP_PROP_POS_FRAMES)
                cap.set(cv2.CAP_PROP_POS_FRAMES, max(0, cur + self._seek_frames))
                self._seek_frames = None

            # paused: don't advance. mjpeg_generator keeps pushing the last
            # frame, so the browser shows a frozen image. A single "step"
            # request lets exactly one frame through.
            if self.paused and not self._step:
                time.sleep(0.05)
                continue
            stepping = self._step
            self._step = False

            ret, frame = cap.read()
            if not ret:
                # loop video files; for live cams try to reconnect
                if self.is_file:
                    cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
                    continue
                time.sleep(0.5)
                cap.release()
                cap = self._open()
                continue

            self._process_frame(frame)

            if stepping:
                self.paused = True                 # one frame, then re-freeze
            elif self.is_file and self.speed > 0:
                # pace file playback to (roughly) real time, scaled by speed
                time.sleep((1.0 / self.fps) / self.speed)

        cap.release()
        self.status = "stopped"

    # ---- playback control API (called from the web layer) ------------------
    def set_paused(self, value):
        self.paused = bool(value)

    def toggle_pause(self):
        self.paused = not self.paused

    def set_speed(self, s):
        self.speed = max(0.1, min(8.0, float(s)))

    def seek(self, seconds):
        """Queue a relative seek by +/- seconds (files only)."""
        self._seek_frames = int(float(seconds) * self.fps)

    def step(self):
        """Advance exactly one frame, then stay paused."""
        self.paused = True
        self._step = True

    def restart(self):
        """Jump back to the start of a video file."""
        self._seek_frames = -10 ** 9   # clamped to frame 0 in run()
        self.paused = False

    def set_model(self, model, label_mode, model_key, device=None):
        """Point this feed at a different network while it keeps running.

        The tracker lives on the model, so the incoming instance starts with an
        empty ID pool. logged_tracks is cleared with it: those IDs describe
        people the old model was following, and the new model will hand the
        same low numbers to different people, silently suppressing their first
        violation for TRACK_RELOG_AFTER seconds.
        """
        with self.model_lock:
            self.model = model
            self.labels = model.names
            self.label_mode = label_mode
            self.model_key = model_key
            if device:
                self.device = device
            self.logged_tracks = {}

    def state(self):
        """Serializable snapshot for /api/feeds."""
        return {"status": self.status, "is_file": self.is_file,
                "paused": self.paused, "speed": self.speed,
                "model": self.model_key}

    def _process_frame(self, frame):
        """Run tracking on one frame, annotate it, and log new violations."""
        # Keep a pristine copy BEFORE any boxes/labels are drawn -- snapshots
        # are cropped from this so the saved image has no annotations covering
        # the person's face.
        clean = frame.copy()

        # Read the model and its label scheme together -- set_model may swap
        # them from the web thread between any two frames.
        with self.model_lock:
            model, labels, label_mode = self.model, self.labels, self.label_mode

        # persist=True keeps the tracker state between calls on this stream,
        # so each violator keeps a stable ID across frames.
        #
        # The NMS args are not defaults and matter a lot here:
        #   conf=MIN_THRESH  - filter inside NMS instead of after it. At the
        #                      ultralytics default (0.25) weak boxes survive
        #                      NMS, get handed to the tracker, and only then
        #                      get dropped by our own threshold -- too late to
        #                      merge them with the box they overlap.
        #   iou=0.5          - 0.7 (default) is loose enough to keep two boxes
        #                      on one person; 0.5 collapses them.
        #   agnostic_nms     - the model emits "Helmets" AND "No Helmets" on the
        #                      same head. Class-aware NMS never suppresses
        #                      across classes, so both boxes survive and the
        #                      tracker opens an ID for each.
        #   max_det=50       - a bounded number of people per frame; caps how
        #                      many tracks a bad frame can spawn.
        results = model.track(frame, persist=True, verbose=False,
                              tracker=TRACKER_CFG, device=self.device,
                              conf=MIN_THRESH, iou=0.5,
                              agnostic_nms=True, max_det=50)
        detections = results[0].boxes

        now = time.time()
        frame_viol_types = set()   # for the on-screen banner
        new_logs = []              # (track_id, violation, conf, xyxy)

        # Unpack the whole frame before judging any of it: a positive-label
        # model can't tell whether a head is bare until it has seen every
        # helmet box in the frame.
        dets = []
        for i in range(len(detections)):
            conf = detections[i].conf.item()
            if conf < MIN_THRESH:
                continue
            xyxy = detections[i].xyxy.cpu().numpy().squeeze().astype(int)
            cls = int(detections[i].cls.item())
            # track id may be None on the very first frames of a track
            tid = int(detections[i].id.item()) if detections[i].id is not None else None
            dets.append({"conf": conf, "box": tuple(int(v) for v in xyxy),
                         "cls": cls, "name": labels[cls], "tid": tid})

        # How a violation is spotted depends on what this model was taught to
        # name; both paths produce the same {index: [vtype, ...]} shape.
        if label_mode == "positive":
            viols = derive_violations(dets, frame.shape)
        else:
            viols = {}
            for i, d in enumerate(dets):
                vtype = classify_label(d["name"])
                if vtype is not None:
                    viols[i] = [vtype]

        for i, d in enumerate(dets):
            xmin, ymin, xmax, ymax = d["box"]
            tid, conf = d["tid"], d["conf"]
            vtypes = viols.get(i, [])

            self._draw_box(frame, xmin, ymin, xmax, ymax, d["cls"], d["name"],
                           tid, conf, bool(vtypes))
            if not vtypes:
                continue
            frame_viol_types.update(vtypes)
            if tid is None:
                continue

            for vtype in vtypes:
                key = (tid, vtype)
                seen_before = key in self.logged_tracks
                # log this id if never logged, or if it was gone long enough
                due = (not seen_before) or (now - self.logged_tracks[key] > TRACK_RELOG_AFTER)
                # ...but only above LOG_THRESH, and only if this feed hasn't
                # just logged the same violation type. The cooldown is what
                # absorbs duplicate boxes and recycled IDs on one person;
                # the per-track check above still handles the stable case.
                if due and conf >= LOG_THRESH and self._cooldown_ok(vtype, now):
                    new_logs.append((tid, vtype, conf, d["box"]))
                    self.logged_types[vtype] = now
                # refresh "last seen" every frame we see it
                self.logged_tracks[key] = now

        self._save_new_logs(clean, new_logs)
        self._prune_tracks(now)
        self._draw_banner(frame, frame_viol_types)
        self._encode(frame)

    def _draw_box(self, frame, xmin, ymin, xmax, ymax, cls, name, tid, conf, is_viol):
        """Draw one bounding box + label (red for violations)."""
        color = (0, 0, 255) if is_viol else BBOX_COLORS[cls % 10]
        cv2.rectangle(frame, (xmin, ymin), (xmax, ymax), color, 2)
        id_txt = f" #{tid}" if tid is not None else ""
        label = f"{name}{id_txt}: {int(conf*100)}%"
        ls, bl = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 1)
        ly = max(ymin, ls[1] + 10)
        cv2.rectangle(frame, (xmin, ly-ls[1]-10), (xmin+ls[0], ly+bl-10), color, cv2.FILLED)
        cv2.putText(frame, label, (xmin, ly-7), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)

    def _save_new_logs(self, clean, new_logs):
        """Write a snapshot + DB row per newly-seen violating track id.

        The snapshot is a CROP of the clean (un-annotated) frame around the
        violator -- padded for context and upscaled if the person is small --
        so their face stays visible and no label covers it.
        """
        for tid, vtype, conf, box in new_logs:
            now_dt = datetime.now()
            ts = now_dt.strftime("%Y%m%d_%H%M%S")
            # forward slashes: this string is also used as a URL by the browser
            safe = "".join(c if c.isalnum() or c in "-_" else "_" for c in self.name)
            snap_rel = f"{SNAPSHOT_DIR}/{safe}_{tid}_{ts}.jpg"
            crop = self._crop_violation(clean, box)
            vtxt = vtype.replace("_", " ").upper()   # 'no_helmet' -> 'NO HELMET'
            caption = f"{vtxt}  |  {self.name}  |  {now_dt.strftime('%Y-%m-%d %H:%M:%S')}"
            crop = self._add_caption(crop, caption)
            cv2.imwrite(snap_rel, crop)
            log_violation(self.name, vtype, conf, snap_rel, track_id=tid)
            print(f"[{self.name}] VIOLATION logged: id#{tid} {vtype}")

    def _crop_violation(self, clean, box, pad=0.5, target_min=400, max_zoom=4.0):
        """Crop the clean frame to the violator + padding, upscaling if small.

        pad         - fraction of the box size to add on each side for context
        target_min  - desired shortest side (px); smaller crops get enlarged
        max_zoom    - cap on how much a tiny crop may be scaled up
        Falls back to the full clean frame if the crop would be empty.
        """
        h, w = clean.shape[:2]
        xmin, ymin, xmax, ymax = box
        bw, bh = xmax - xmin, ymax - ymin

        # expand the box on all sides (a bit extra on top to include the head)
        x0 = max(0, int(xmin - bw * pad))
        x1 = min(w, int(xmax + bw * pad))
        y0 = max(0, int(ymin - bh * pad * 1.3))
        y1 = min(h, int(ymax + bh * pad))

        crop = clean[y0:y1, x0:x1]
        if crop.size == 0:
            return clean  # safety fallback

        # if the person is small, zoom in so the face is legible
        ch, cw = crop.shape[:2]
        scale = target_min / max(1, min(ch, cw))
        if scale > 1.0:
            scale = min(scale, max_zoom)
            crop = cv2.resize(crop, (int(cw * scale), int(ch * scale)),
                              interpolation=cv2.INTER_CUBIC)
        return crop

    def _add_caption(self, img, text):
        """Add a black bar UNDER the image with feed name + timestamp.

        The caption lives outside the picture area (added via a bottom border),
        so it never covers the person. Font shrinks to fit narrow crops.
        """
        h, w = img.shape[:2]
        font = cv2.FONT_HERSHEY_SIMPLEX
        thick = 1
        scale = 0.6
        while scale > 0.3:
            (tw, _), _ = cv2.getTextSize(text, font, scale, thick)
            if tw <= w - 16:
                break
            scale -= 0.05
        (tw, th), bl = cv2.getTextSize(text, font, scale, thick)
        bar = th + bl + 12
        out = cv2.copyMakeBorder(img, 0, bar, 0, 0, cv2.BORDER_CONSTANT, value=(0, 0, 0))
        cv2.putText(out, text, (8, h + th + 6), font, scale, (255, 255, 255),
                    thick, cv2.LINE_AA)
        return out

    def _cooldown_ok(self, vtype, now):
        """True if this feed may log `vtype` again yet.

        Deliberately per (feed, type) and not per track: the whole point is to
        hold when the track ID can't be trusted. The cost is that two people
        committing the same violation within FEED_RELOG_COOLDOWN produce one
        row -- cheaper than a row per duplicate box on a single head.
        """
        last = self.logged_types.get(vtype)
        return last is None or (now - last) >= FEED_RELOG_COOLDOWN

    def _prune_tracks(self, now):
        """Drop track ids we haven't seen in a long time to bound memory."""
        if self.logged_tracks:
            self.logged_tracks = {k: v for k, v in self.logged_tracks.items()
                                  if now - v < TRACK_RELOG_AFTER * 3}

    def _draw_banner(self, frame, frame_viol_types):
        """Feed name + (if any) a red violation banner in the corner."""
        cv2.putText(frame, self.name, (10, 25),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2)
        if frame_viol_types:
            cv2.putText(frame, "! " + ", ".join(sorted(frame_viol_types)).upper(), (10, 55),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)

    def _encode(self, frame):
        """Encode the annotated frame to JPEG for the MJPEG stream."""
        ok, jpeg = cv2.imencode(".jpg", frame, [cv2.IMWRITE_JPEG_QUALITY, 70])
        if ok:
            with self.lock:
                self.latest_jpeg = jpeg.tobytes()

    def get_jpeg(self):
        with self.lock:
            return self.latest_jpeg
