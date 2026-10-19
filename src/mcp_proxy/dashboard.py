"""A tiny read-only dashboard served from the standard library.

It runs in a daemon thread and only serves bytes the event loop has already prepared,
so it adds no dependency and no work on the proxy's event loop.
"""

import logging
import threading
from http import HTTPStatus
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from importlib import resources

logger = logging.getLogger(__name__)

LOOPBACK_NAMES = ("127.0.0.1", "localhost", "[::1]")


class DashboardState:
    """The latest metrics snapshot, replaced atomically by the proxy's sampler task."""

    def __init__(self) -> None:
        """Start with an empty snapshot."""
        self.snapshot_json = b"{}"


def _page() -> bytes:
    return resources.files("mcp_proxy").joinpath("dashboard.html").read_bytes()


def make_handler(state: DashboardState, port: int) -> type[BaseHTTPRequestHandler]:
    """Build the request handler class bound to ``state``."""
    allowed_hosts = {f"{name}:{port}" for name in LOOPBACK_NAMES}

    class Handler(BaseHTTPRequestHandler):
        def log_message(self, format: str, *args: object) -> None:  # noqa: A002
            logger.debug("dashboard: " + format, *args)  # noqa: G003

        def _send(self, status: HTTPStatus, body: bytes, content_type: str) -> None:
            self.send_response(status)
            self.send_header("Content-Type", content_type)
            self.send_header("Content-Length", str(len(body)))
            self.send_header("Cache-Control", "no-store")
            self.send_header("X-Content-Type-Options", "nosniff")
            self.end_headers()
            if self.command != "HEAD":
                self.wfile.write(body)

        def do_GET(self) -> None:
            # A web page that rebinds its hostname to 127.0.0.1 must not read the metrics.
            if self.headers.get("Host") not in allowed_hosts:
                self._send(HTTPStatus.MISDIRECTED_REQUEST, b"unexpected Host", "text/plain")
            elif self.path in ("/", "/index.html"):
                self._send(HTTPStatus.OK, _page(), "text/html; charset=utf-8")
            elif self.path == "/api/metrics":
                self._send(HTTPStatus.OK, state.snapshot_json, "application/json")
            else:
                self._send(HTTPStatus.NOT_FOUND, b"not found", "text/plain")

        do_HEAD = do_GET  # noqa: N815

    return Handler


def start_dashboard(host: str, port: int, state: DashboardState) -> ThreadingHTTPServer:
    """Start serving in a daemon thread. Call ``shutdown`` on the result to stop."""
    server = ThreadingHTTPServer((host, port), make_handler(state, port))
    server.daemon_threads = True
    threading.Thread(target=server.serve_forever, name="dashboard", daemon=True).start()
    logger.info("Dashboard at http://%s:%s/", host, port)
    return server
