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

from config import (SNAPSHOT_DIR, MIN_THRESH, TRACK_RELOG_AFTER, TRACKER_CFG,
                    HELMET_KEYS, VEST_KEYS, NEG_KEYS, BBOX_COLORS)
from db import log_violation


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


class FeedWorker(threading.Thread):
    """One background thread per camera/video/RTSP source.

    Reads frames, runs YOLO tracking, draws boxes, logs violations, and keeps
    the latest annotated JPEG ready for the web layer to stream.
    """

    def __init__(self, name, source, model, device="cpu", resolution=None):
        super().__init__(daemon=True)
        self.name = name
        self.source = source
        self.model = model
        self.device = device
        self.labels = model.names
        self.resolution = resolution
        self.latest_jpeg = None
        self.lock = threading.Lock()
        self.running = True
        self.status = "starting"
        # track_id -> last time we SAW this violating id (for re-log timeout)
        self.logged_tracks = {}

        # ---- playback controls (driven by the dashboard) -------------------
        # Seek/speed/step only apply to video FILES; live cams ignore them.
        self.is_file = isinstance(source, str) and os.path.isfile(source)
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
        return cv2.VideoCapture(src)

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

    def state(self):
        """Serializable snapshot for /api/feeds."""
        return {"status": self.status, "is_file": self.is_file,
                "paused": self.paused, "speed": self.speed}

    def _process_frame(self, frame):
        """Run tracking on one frame, annotate it, and log new violations."""
        # Keep a pristine copy BEFORE any boxes/labels are drawn -- snapshots
        # are cropped from this so the saved image has no annotations covering
        # the person's face.
        clean = frame.copy()

        # persist=True keeps the tracker state between calls on this stream,
        # so each violator keeps a stable ID across frames.
        results = self.model.track(frame, persist=True, verbose=False,
                                   tracker=TRACKER_CFG, device=self.device)
        detections = results[0].boxes

        now = time.time()
        frame_viol_types = set()   # for the on-screen banner
        new_logs = []              # (track_id, violation, conf, xyxy)

        for i in range(len(detections)):
            conf = detections[i].conf.item()
            if conf < MIN_THRESH:
                continue
            xyxy = detections[i].xyxy.cpu().numpy().squeeze().astype(int)
            xmin, ymin, xmax, ymax = xyxy
            cls = int(detections[i].cls.item())
            name = self.labels[cls]

            # track id may be None on the very first frames of a track
            tid = int(detections[i].id.item()) if detections[i].id is not None else None

            vtype = classify_label(name)      # 'no_helmet' | 'no_vest' | None
            is_viol = vtype is not None

            self._draw_box(frame, xmin, ymin, xmax, ymax, cls, name, tid, conf, is_viol)

            if is_viol:
                frame_viol_types.add(vtype)
                if tid is not None:
                    seen_before = tid in self.logged_tracks
                    # log this id if never logged, or if it was gone long enough
                    if (not seen_before) or (now - self.logged_tracks[tid] > TRACK_RELOG_AFTER):
                        new_logs.append((tid, vtype, conf, (xmin, ymin, xmax, ymax)))
                    # refresh "last seen" every frame we see it
                    self.logged_tracks[tid] = now

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
            snap_rel = os.path.join(SNAPSHOT_DIR, f"{self.name}_{tid}_{ts}.jpg")
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
