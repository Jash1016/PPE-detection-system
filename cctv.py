"""
CCTV / DVR / NVR connection helper.
===================================
Turns the details printed on a camera or DVR -- IP, port, username, password,
channel -- into the stream URL that OpenCV can open.

Every device here ends up speaking RTSP (or HTTP-MJPEG for phone apps and
older cameras). The only thing that differs between brands is the PATH part
of the URL, which is not standardised -- hence the presets below.

Analog cameras have no IP of their own: they plug into a DVR over coax, and
you connect to the DVR's IP using the CHANNEL number of the camera you want.
"""
from urllib.parse import quote

# ---------------------------------------------------------------------------
# BRAND PRESETS
# ---------------------------------------------------------------------------
# scheme    - rtsp or http
# port      - default port if the user doesn't override it
# path      - template; {ch} = channel, {main} / {sub} filled per stream choice
# channels  - does this device have selectable channels? (DVRs do, phones don't)
PRESETS = {
    "hikvision": {
        "label": "Hikvision (also most DVR/NVR clones)",
        "scheme": "rtsp", "port": 554, "channels": True,
        # channel id is <channel><2-digit stream>:
        # 101 = ch1 main, 102 = ch1 sub, 202 = ch2 sub, 1001 = ch10 main
        "path": "/Streaming/Channels/{ch}{stream_pad}",
    },
    "dahua": {
        "label": "Dahua / CP Plus",
        "scheme": "rtsp", "port": 554, "channels": True,
        "path": "/cam/realmonitor?channel={ch}&subtype={stream01}",
    },
    "uniview": {
        "label": "Uniview / generic ONVIF",
        "scheme": "rtsp", "port": 554, "channels": True,
        "path": "/media/video{ch}",
    },
    "ipwebcam": {
        "label": "Android phone — IP Webcam app",
        "scheme": "rtsp", "port": 8080, "channels": False,
        "path": "/h264_ulaw.sdp",
    },
    "ipwebcam_mjpeg": {
        "label": "Android phone — IP Webcam (MJPEG fallback)",
        "scheme": "http", "port": 8080, "channels": False,
        "path": "/video",
    },
    "custom": {
        "label": "Other — I'll type the path myself",
        "scheme": "rtsp", "port": 554, "channels": False,
        "path": None,          # supplied by the user
    },
}


def build_url(brand, ip, port=None, username=None, password=None,
              channel=1, stream="sub", custom_path=None, scheme=None):
    """Assemble a stream URL. Returns (url, error).

    stream - 'main' (full resolution) or 'sub' (smaller, much lighter on CPU)
    Credentials are percent-encoded so passwords containing @ : / # still work.
    """
    p = PRESETS.get(brand)
    if not p:
        return None, f"unknown brand preset: {brand}"

    ip = (ip or "").strip()
    if not ip:
        return None, "IP address is required"

    port = port or p["port"]
    scheme = scheme or p["scheme"]

    # credentials -- optional (some devices allow anonymous viewing)
    cred = ""
    if username:
        cred = quote(str(username), safe="")
        if password:
            cred += ":" + quote(str(password), safe="")
        cred += "@"

    # path
    if p["path"] is None:
        path = (custom_path or "").strip()
        if not path:
            return None, "a stream path is required for the 'Other' preset"
        if not path.startswith("/"):
            path = "/" + path
    else:
        try:
            ch = int(channel or 1)
        except (TypeError, ValueError):
            return None, "channel must be a number"
        if ch < 1:
            return None, "channel must be 1 or higher"
        path = p["path"].format(
            ch=ch,
            stream_pad="01" if stream == "main" else "02",  # Hikvision style
            stream01=0 if stream == "main" else 1,          # Dahua style
        )

    return f"{scheme}://{cred}{ip}:{port}{path}", None


def mask_url(url):
    """Hide the password in a stream URL before showing or logging it.

    rtsp://admin:secret@10.0.0.5:554/x  ->  rtsp://admin:****@10.0.0.5:554/x
    """
    if not isinstance(url, str) or "://" not in url:
        return url
    scheme, rest = url.split("://", 1)
    if "@" not in rest:
        return url
    cred, host = rest.split("@", 1)
    if ":" in cred:
        user = cred.split(":", 1)[0]
        return f"{scheme}://{user}:****@{host}"
    return f"{scheme}://{cred}@{host}"


def onvif_stream_url(ip, port, username, password):
    """Ask an ONVIF device for its own stream URI. Returns (url, error).

    Optional: needs `pip install onvif-zeep`. Preferred over the brand
    presets when available, because the device tells you the exact path
    instead of you guessing it.
    """
    try:
        from onvif import ONVIFCamera
    except ImportError:
        return None, ("ONVIF support needs an extra package. Run "
                      "'pip install onvif-zeep', or pick your camera's brand "
                      "from the list instead.")
    try:
        cam = ONVIFCamera(ip, int(port or 80), username or "", password or "")
        media = cam.create_media_service()
        profiles = media.GetProfiles()
        if not profiles:
            return None, "the device reported no video profiles"
        # last profile is usually the lowest-resolution sub-stream
        req = media.create_type("GetStreamUri")
        req.ProfileToken = profiles[-1].token
        req.StreamSetup = {"Stream": "RTP-Unicast",
                           "Transport": {"Protocol": "RTSP"}}
        uri = media.GetStreamUri(req).Uri
        # the device returns a URL without credentials -- add them back
        if username and "://" in uri and "@" not in uri.split("://", 1)[1]:
            scheme, rest = uri.split("://", 1)
            cred = quote(str(username), safe="")
            if password:
                cred += ":" + quote(str(password), safe="")
            uri = f"{scheme}://{cred}@{rest}"
        return uri, None
    except Exception as e:
        return None, f"could not reach the ONVIF device: {e}"
