"""
Camera / DVR discovery helper.
=============================
Finds CCTV devices on your local network so you can fill in the connection
details for a feed. Run it on the same network as the cameras:

    python discover.py

It does two passes:
  1. ONVIF WS-Discovery -- a multicast "who's there?" that ONVIF cameras and
     NVRs answer with their own address. This is the reliable way.
  2. A port sweep of your subnet looking for the ports CCTV gear listens on
     (554 RTSP, 80/8080 web UI, 8000 Hikvision, 37777 Dahua). Catches devices
     that don't speak ONVIF or have it switched off.

Only scan networks you are allowed to scan (your own home/lab/office LAN).

Nothing here logs in or guesses passwords -- it only reports which devices
exist and what they look like. You still need credentials from whoever
installed the system.
"""
import re
import socket
import uuid
from concurrent.futures import ThreadPoolExecutor

# Ports that CCTV equipment commonly listens on, and what they suggest
PORTS = {
    554:   "RTSP (video stream)",
    80:    "web interface",
    8080:  "web interface (alt)",
    8000:  "Hikvision SDK port",
    37777: "Dahua / CP Plus SDK port",
    88:    "ONVIF / web (alt)",
}

WS_DISCOVERY_ADDR = ("239.255.255.250", 3702)

PROBE = """<?xml version="1.0" encoding="UTF-8"?>
<e:Envelope xmlns:e="http://www.w3.org/2003/05/soap-envelope"
            xmlns:w="http://schemas.xmlsoap.org/ws/2004/08/addressing"
            xmlns:d="http://schemas.xmlsoap.org/ws/2005/04/discovery"
            xmlns:dn="http://www.onvif.org/ver10/network/wsdl">
  <e:Header>
    <w:MessageID>uuid:{mid}</w:MessageID>
    <w:To e:mustUnderstand="true">urn:schemas-xmlsoap-org:ws:2005:04:discovery</w:To>
    <w:Action e:mustUnderstand="true">http://schemas.xmlsoap.org/ws/2005/04/discovery/Probe</w:Action>
  </e:Header>
  <e:Body><d:Probe><d:Types>dn:NetworkVideoTransmitter</d:Types></d:Probe></e:Body>
</e:Envelope>"""


def local_ip():
    """This machine's LAN address (no packets are actually sent)."""
    s = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
    try:
        s.connect(("8.8.8.8", 80))
        return s.getsockname()[0]
    finally:
        s.close()


def onvif_discover(timeout=4.0):
    """Multicast an ONVIF probe and collect whoever answers."""
    msg = PROBE.format(mid=uuid.uuid4()).encode()
    sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
    sock.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
    sock.setsockopt(socket.IPPROTO_IP, socket.IP_MULTICAST_TTL, 2)
    sock.settimeout(timeout)
    try:
        sock.bind((local_ip(), 0))
        sock.sendto(msg, WS_DISCOVERY_ADDR)
    except OSError as e:
        print("  (could not send probe:", e, ")")
        sock.close()
        return {}

    found = {}
    while True:
        try:
            data, addr = sock.recvfrom(65535)
        except socket.timeout:
            break
        except OSError:
            break
        text = data.decode("utf-8", "ignore")
        urls = re.findall(r"https?://[^\s<>\"]+", text)
        entry = found.setdefault(addr[0], {"urls": set(), "scopes": set()})
        entry["urls"].update(urls)
        for scope in re.findall(r"onvif://www\.onvif\.org/(\S+)", text):
            entry["scopes"].add(scope.rstrip("</dScopes>"))
    sock.close()
    return found


def check_port(ip, port, timeout=0.35):
    s = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    s.settimeout(timeout)
    try:
        return s.connect_ex((ip, port)) == 0
    except OSError:
        return False
    finally:
        s.close()


def scan_subnet(base):
    """Return {ip: [open ports]} for hosts with any CCTV-ish port open."""
    targets = [(f"{base}.{h}", p) for h in range(1, 255) for p in PORTS]
    hits = {}
    with ThreadPoolExecutor(max_workers=200) as pool:
        results = pool.map(lambda t: (t, check_port(*t)), targets)
        for (ip, port), is_open in results:
            if is_open:
                hits.setdefault(ip, []).append(port)
    return hits


def guess_brand(ports, scopes):
    text = " ".join(scopes).lower()
    for name in ("hikvision", "dahua", "cp plus", "cpplus", "axis",
                 "uniview", "reolink", "tp-link", "vivotek", "bosch"):
        if name in text:
            return name.title()
    if 37777 in ports:
        return "Dahua / CP Plus (likely)"
    if 8000 in ports:
        return "Hikvision (likely)"
    return "unknown"


def main():
    me = local_ip()
    base = ".".join(me.split(".")[:3])
    print(f"\nThis PC : {me}")
    print(f"Scanning: {base}.1 - {base}.254\n")

    print("=" * 62)
    print("  PASS 1 - ONVIF discovery")
    print("=" * 62)
    onvif = onvif_discover()
    if onvif:
        for ip, info in sorted(onvif.items()):
            print(f"\n  ONVIF device at {ip}")
            for u in sorted(info["urls"])[:3]:
                print(f"    service : {u}")
            for s in sorted(info["scopes"])[:6]:
                print(f"    scope   : {s}")
    else:
        print("  No ONVIF replies. That's normal if ONVIF is disabled,")
        print("  or the device is on a different network/VLAN.")

    print("\n" + "=" * 62)
    print("  PASS 2 - port sweep")
    print("=" * 62)
    hits = scan_subnet(base)
    hits.pop(me, None)                     # ignore this PC

    cams = {ip: ps for ip, ps in hits.items() if 554 in ps or 37777 in ps or 8000 in ps}
    others = {ip: ps for ip, ps in hits.items() if ip not in cams}

    if cams:
        print("\n  LIKELY CAMERAS / DVRs (RTSP or a CCTV SDK port is open):\n")
        for ip, ps in sorted(cams.items()):
            scopes = onvif.get(ip, {}).get("scopes", set())
            print(f"    {ip}   -> {guess_brand(ps, scopes)}")
            for p in sorted(ps):
                print(f"        :{p:<6} {PORTS[p]}")
            if 80 in ps or 8080 in ps or 88 in ps:
                port = 80 if 80 in ps else (8080 if 8080 in ps else 88)
                url = f"http://{ip}" + ("" if port == 80 else f":{port}")
                print(f"        open {url} in a browser to see the brand + log in")
            print()
    else:
        print("\n  Nothing with an obvious CCTV port found.")

    if others:
        print("  Other devices with a web port (routers, printers, PCs):")
        print("    " + ", ".join(sorted(others)))

    print("\n" + "=" * 62)
    print("  NEXT STEP")
    print("=" * 62)
    print("""
  1. Open the device's http:// address above in a browser. The login page
     usually shows the brand (Hikvision, Dahua, CP Plus...).
  2. Get the username/password from whoever installed the system.
  3. Test the stream in VLC:  Media > Open Network Stream, and paste

       Hikvision  rtsp://USER:PASS@IP:554/Streaming/Channels/102
       Dahua/CP+  rtsp://USER:PASS@IP:554/cam/realmonitor?channel=1&subtype=1

     (channel 1 sub-stream. Change the channel number per camera.)
  4. If VLC plays it, the same URL will work in the dashboard.
""")


if __name__ == "__main__":
    main()
