"""Guard against Server-Side Request Forgery (SSRF) for ``http`` tools.

Only public, non-private target hosts are allowed. This prevents a registered
tool from pointing the server at loopback, internal subnets, cloud metadata
endpoints, or other internal services.
"""

import ipaddress
import socket
from urllib.parse import urlparse


class SSRFError(Exception):
    pass


# RFC 1918 + loopback + link-local + CGNAT + reserved + IPv6 equivalents.
_PRIVATE_NETWORKS = [
    "0.0.0.0/8",
    "10.0.0.0/8",
    "100.64.0.0/10",
    "127.0.0.0/8",
    "169.254.0.0/16",
    "172.16.0.0/12",
    "192.0.0.0/24",
    "192.168.0.0/16",
    "198.18.0.0/15",
    "198.51.100.0/24",
    "203.0.113.0/24",
    "224.0.0.0/4",
    "240.0.0.0/4",
    "255.255.255.255/32",
    "::1/128",
    "::/128",
    "fc00::/7",
    "fe80::/10",
]

_BLOCKED_HOSTNAMES = {
    "localhost",
    "host.docker.internal",
    "metadata.google.internal",
    "kubernetes.default.svc",
}

_BLOCKED_HOST_SUFFIXES = (".local", ".internal", ".localhost", ".localdomain")


def _is_private_ip(ip_str: str) -> bool:
    try:
        ip = ipaddress.ip_address(ip_str)
    except ValueError:
        return False
    return any(ip in ipaddress.ip_network(net) for net in _PRIVATE_NETWORKS)


def validate_http_url(url: str) -> None:
    """Validate that ``url`` is a public http/https URL.

    Raises :class:`SSRFError` when the scheme is not http(s) or when the target
    is a private/loopback/link-local address (including via DNS resolution).
    """
    parsed = urlparse(url)
    if parsed.scheme not in ("http", "https"):
        raise SSRFError(f"Only http/https URLs are allowed (got scheme '{parsed.scheme}')")

    host = parsed.hostname
    if not host:
        raise SSRFError("URL has no host")

    host_lower = host.lower()
    if host_lower in _BLOCKED_HOSTNAMES or any(
        host_lower.endswith(s) for s in _BLOCKED_HOST_SUFFIXES
    ):
        raise SSRFError(f"URL host '{host}' is not allowed")

    # Resolve the host and reject if any address is private. This closes the
    # obvious DNS-rebinding hole at call time.
    port = parsed.port or (443 if parsed.scheme == "https" else 80)
    try:
        infos = socket.getaddrinfo(host, port)
    except OSError:
        # Unresolvable — the caller will fail naturally on connect.
        return

    for info in infos:
        addr = info[4][0]
        if _is_private_ip(addr):
            raise SSRFError(
                f"URL host '{host}' resolves to a private/internal address ({addr})"
            )
