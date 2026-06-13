import ipaddress
import socket
from urllib.parse import urlparse
from urllib.request import Request, urlopen

from toolbench.tools import tool

_MAX_CHARS = 100_000


def _is_safe_url(url: str) -> bool:
    p = urlparse(url)
    if p.scheme not in ("http", "https") or not p.hostname:
        return False
    try:
        infos = socket.getaddrinfo(p.hostname, None)
    except socket.gaierror:
        return False
    for info in infos:
        ip = ipaddress.ip_address(info[4][0])
        if (
            ip.is_private
            or ip.is_loopback
            or ip.is_link_local
            or ip.is_reserved
            or ip.is_multicast
        ):
            return False
    return True


@tool
def http_fetch(url: str) -> str:
    """Fetch a web page over HTTP GET and return up to 100k characters of text.

    Args:
        url: An http(s) URL to GET.
    """
    if not _is_safe_url(url):
        return "ERROR: refused unsafe or non-public URL"
    req = Request(url, headers={"User-Agent": "toolbench/0.1"})
    with urlopen(req, timeout=15) as r:  # noqa: S310 (scheme validated above)
        return r.read(_MAX_CHARS).decode("utf-8", errors="replace")
