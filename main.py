"""
Entrypoint / startup wiring.
============================
Parses CLI args, loads the YOLO model, decides the inference device, seeds any
feeds from --config/--source, and starts the Flask server.

Run:
  python main.py --model best.pt --config feeds.json
  # then open http://localhost:5000 in your browser
"""
import os
import sys
import json
import argparse

from ultralytics import YOLO

import config
import server
from db import init_db


def resolve_device(want):
    """Turn the requested device ('auto'/'cpu'/'0') into a concrete choice.

    Returns (device_string, cuda_available). Falls back to CPU with a loud
    warning if a GPU was requested but CUDA isn't available to torch.
    """
    try:
        import torch
        cuda_ok = torch.cuda.is_available()
    except Exception:
        cuda_ok = False

    want = want.lower()
    if want == "auto":
        device = "0" if cuda_ok else "cpu"
    elif want == "cpu":
        device = "cpu"
    else:
        device = want  # e.g. "0", "1"

    if device != "cpu" and not cuda_ok:
        print("=" * 68)
        print("WARNING: GPU requested but CUDA is NOT available to PyTorch.")
        print("You likely have the CPU-only build of torch installed.")
        print("Install the CUDA build (example for CUDA 12.1):")
        print("  pip uninstall -y torch torchvision")
        print("  pip install torch torchvision --index-url https://download.pytorch.org/whl/cu121")
        print("Falling back to CPU for now.")
        print("=" * 68)
        device = "cpu"

    return device, cuda_ok


def load_model(model_path, device):
    """Load YOLO and move it to the chosen device. Returns (model, device)."""
    model = YOLO(model_path, task="detect")
    if device != "cpu":
        try:
            model.to(f"cuda:{device}" if device.isdigit() else device)
        except Exception as e:
            print("Could not move model to GPU, using CPU:", e)
            device = "cpu"
    print("Model label map:", model.names)
    print(">>> Violation matching is automatic (see classify_label). If a class")
    print("    that should count as a violation is missed, check these names.\n")
    return model, device


def seed_feeds(args, res):
    """Start any feeds listed in --config, or the single --source feed."""
    feeds = []
    if args.config and os.path.exists(args.config):
        with open(args.config) as f:
            feeds = json.load(f)          # [{"name":"Gate A","source":"usb0"}, ...]
    elif args.source:
        feeds = [{"name": "Feed 1", "source": args.source}]

    for f in feeds:
        ok, err = server.add_feed(
            f["name"], f["source"],
            resolution=tuple(f["resolution"]) if f.get("resolution") else res)
        if not ok:
            print(f"WARNING: could not start feed '{f.get('name')}': {err}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True, help='path to best.pt')
    ap.add_argument("--config", help='JSON file listing feeds (see feeds.json)')
    ap.add_argument("--source", help='single source if not using --config (e.g. usb0 or vid.mp4)')
    ap.add_argument("--resolution", default=None, help='WxH e.g. 640x480')
    ap.add_argument("--device", default="auto",
                    help='inference device: "auto", "0" (first GPU), or "cpu"')
    ap.add_argument("--host", default="0.0.0.0")
    ap.add_argument("--port", type=int, default=5000)
    args = ap.parse_args()

    os.makedirs(config.SNAPSHOT_DIR, exist_ok=True)
    os.makedirs(config.UPLOAD_DIR, exist_ok=True)
    init_db()

    if not os.path.exists(args.model):
        print("ERROR: model not found:", args.model); sys.exit(1)

    # ---- decide device + load model ----------------------------------------
    device, cuda_ok = resolve_device(args.device)
    print(f">>> Inference device: {device}  (CUDA available: {cuda_ok})")
    model, device = load_model(args.model, device)

    # publish runtime state to the web layer
    config.DEVICE = device
    server.MODEL = model
    server.DEVICE = device

    res = None
    if args.resolution:
        res = tuple(int(x) for x in args.resolution.lower().split("x"))

    seed_feeds(args, res)

    print(f"\nDashboard: http://localhost:{args.port}")
    print("Add more feeds any time from the dashboard's + button.\n")
    server.app.run(host=args.host, port=args.port, threaded=True)


if __name__ == "__main__":
    main()
