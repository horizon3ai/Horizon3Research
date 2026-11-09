"""Synthetic recon (Nmap) built from the real source-verified banners in recon/nmap_banners.yaml (mock only).
Four modes form a ladder: none / generic (ports only) / version (-sV banners) / scripts (+ NSE CVE billboard)."""
import re
from pathlib import Path

import yaml

_BANNERS = None


def _banners():
    global _BANNERS
    if _BANNERS is None:
        p = Path(__file__).resolve().parent.parent / "recon" / "nmap_banners.yaml"
        try:
            _BANNERS = (yaml.safe_load(p.read_text()) or {}).get("banners", {})
        except Exception:
            _BANNERS = {}
    return _BANNERS


def _port(item):
    try:
        return int(item.get("port", 0))
    except (TypeError, ValueError):
        return 0


# Drop NSE enumeration scripts that would re-introduce recognition on scan-dark arms; keep only the
# vuln/exploit billboard. Set STRIP_IDENTITY_BANNERS=False to disable.
STRIP_IDENTITY_BANNERS = True
_VULN_MARKER = re.compile(r"vuln|vulnerable|backdoor|shellshock|heartbleed|exploitab|\bCVE:|uid=0", re.I)
# NSE script headers are hyphenated names at the left margin (http-title:, ssl-cert:, ...); nested
# vuln fields (VULNERABLE:, State:, IDs:) have no hyphen, so they don't start a new block.
_SCRIPT_HDR = re.compile(r"^\s*\|_?\s{0,2}[A-Za-z][A-Za-z0-9]*(?:-[A-Za-z0-9.]+)+\s*:")


def _strip_identity_lines(nse):
    """Keep only NSE blocks that are a vuln/exploit billboard; drop every enumeration block (http-title, ssl-cert, ...)."""
    out, block = [], []

    def flush():
        if block and _VULN_MARKER.search("\n".join(block)):
            out.extend(block)
        block.clear()

    for ln in nse.splitlines():
        if not ln.lstrip().startswith("|"):   # a plain port/version line, never dropped
            flush()
            out.append(ln)
            continue
        if _SCRIPT_HDR.match(ln):             # a new top-level script begins here
            flush()
        block.append(ln)
    flush()
    return "\n".join(out).rstrip("\n")


def synth_nmap(bundle, mode="version", target_ip="10.10.10.50"):
    if mode == "none":
        return ""
    b = _banners()
    items = sorted(bundle, key=_port)          # real nmap is in ascending port order
    head = [
        "Starting Nmap 7.94 ( https://nmap.org )",
        f"Nmap scan report for {target_ip}",
        "Host is up (0.012s latency).",
        "Not shown: 995 closed tcp ports (reset)",
        "",
    ]

    if mode == "generic":
        head.append(f"{'PORT':<10}{'STATE':<6}SERVICE")
        body = [f"{str(it['port']) + '/tcp':<10}{'open':<6}{it.get('service', 'unknown')}" for it in items]
        return "\n".join(head + body + ["", "Nmap done: 1 IP address (1 host up) scanned"])

    # reformat real banner lines into aligned nmap columns
    head.append(f"{'PORT':<10}{'STATE':<6}{'SERVICE':<15}VERSION")
    body = []
    for it in items:
        ban = b.get(it["id"], {})
        vline = ban.get("version_line")
        if vline:
            parts = str(vline).split(None, 3)  # port, state, service, version(rest)
            if len(parts) >= 3 and parts[0].endswith("/tcp"):
                vline = f"{parts[0]:<10}{parts[1]:<6}{parts[2]:<15}{parts[3] if len(parts) > 3 else ''}".rstrip()
        else:
            vline = f"{str(it['port']) + '/tcp':<10}{'open':<6}{it.get('service', 'unknown'):<15}{it.get('version', '')}".rstrip()
        body.append(vline)
        if mode == "scripts":
            nse = (ban.get("nse") or "").rstrip("\n")
            if nse:
                if STRIP_IDENTITY_BANNERS:
                    nse = _strip_identity_lines(nse)
                if nse.strip():
                    body.append(nse)
    footer = ["", "Service detection performed. Please report any incorrect results at https://nmap.org/submit/ .",
              "Nmap done: 1 IP address (1 host up) scanned"]
    return "\n".join(head + body + footer)
