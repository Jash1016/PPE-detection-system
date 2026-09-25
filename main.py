"""
Entrypoint / startup wiring.
============================
Parses CLI args, loads the YOLO model, decides the inference device, seeds any
feeds from --config/--source, and starts the Flask server.

Run:
  python main.py                       # auto-finds the model next to this file
  python main.py --model best.pt       # or point at a specific one
  # then open the link it prints (http://localhost:5000)
"""
import os
import sys
import glob
import json
import socket
import argparse

import config
import server
import auth
from detection import build_model
from db import init_db

HERE = os.path.dirname(os.path.abspath(__file__))


def find_model(explicit):
    """Decide which model file to load.

    1) --model if the user passed one
    2) config.MODEL_PATH (e.g. my_model.pt) sitting next to this script
    3) any *.pt file in the script folder
    Returns a path (which may not exist -- caller checks and errors cleanly).
    """
    if explicit:
        return explicit
    default = os.path.join(HERE, config.MODEL_PATH)
    if os.path.exists(default):
        return default
    candidates = sorted(glob.glob(os.path.join(HERE, "*.pt")))
    if candidates:
        print(f">>> No --model given; auto-selected {os.path.basename(candidates[0])}")
        return candidates[0]
    return default


def local_ip():
    """Best-effort LAN IP so you can open the dashboard from other devices."""
    try:
        s = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        s.connect(("8.8.8.8", 80))     # no packets sent; just picks the route
        ip = s.getsockname()[0]
        s.close()
        return ip
    except Exception:
        return "127.0.0.1"


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
    """Load YOLO and move it to the chosen device. Returns (model, device).

    Construction lives in detection.build_model so the per-feed models made in
    server.add_feed land on the same device by the same path -- a second copy
    of this logic here would drift from it.
    """
    model, device = build_model(model_path, device)
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
            resolution=tuple(f["resolution"]) if f.get("resolution") else res,
            model_key=f.get("model"))     # absent in pre-registry feeds.json
        if not ok:
            print(f"WARNING: could not start feed '{f.get('name')}': {err}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default=None,
                    help='path to model .pt (optional; auto-found next to this script)')
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
    auth.init_auth_db()     # creates the users tables + first admin if needed

    model_path = find_model(args.model)
    if not os.path.exists(model_path):
        print("ERROR: model not found:", model_path)
        print(f"       Put a .pt file (e.g. {config.MODEL_PATH}) next to main.py,")
        print("       or pass one with --model path/to/model.pt")
        sys.exit(1)

    # ---- decide device + load model ----------------------------------------
    device, cuda_ok = resolve_device(args.device)
    print(f">>> Inference device: {device}  (CUDA available: {cuda_ok})")
    model, device = load_model(model_path, device)

    # publish runtime state to the web layer
    config.DEVICE = device
    server.MODEL = model
    server.MODEL_PATH = model_path   # add_feed loads a private copy per feed
    server.DEVICE = device

    res = None
    if args.resolution:
        res = tuple(int(x) for x in args.resolution.lower().split("x"))

    seed_feeds(args, res)

    # ---- show the links the server will be reachable at --------------------
    lan = local_ip()
    print("\n" + "=" * 52)
    print("  Dashboard is running! Open one of these:")
    print(f"    Local:    http://localhost:{args.port}")
    print(f"    Local:    http://127.0.0.1:{args.port}")
    if lan != "127.0.0.1":
        print(f"    Network:  http://{lan}:{args.port}   (other devices on your Wi-Fi/LAN)")
    print("=" * 52)
    print("Sign in with your account. Admins manage users at /admin.\n")
    server.app.run(host=args.host, port=args.port, threaded=True)


if __name__ == "__main__":
    main()
