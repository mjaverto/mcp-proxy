# ruff: noqa: D103
"""Dashboard metrics and the standard-library server that publishes them."""

import json
import socket
import time
from http import HTTPStatus
from http.client import HTTPConnection

import pytest

from mcp_proxy.dashboard import DashboardState, start_dashboard
from mcp_proxy.metrics import BackendStats, build_snapshot, match_token


class FakeBackend:  # noqa: D101
    def __init__(self, name: str, memory_mb: float | None, clients: int) -> None:  # noqa: D107
        self.name = name
        self.state = "up"
        self.restarts = 0
        self.stats = BackendStats(up_since=time.time() - 60, memory_mb=memory_mb)
        for i in range(clients):
            self.stats.note_client(f"s{i}", "omp")


def test_memory_avoided_counts_only_the_extra_clients() -> None:
    snap = build_snapshot(
        [FakeBackend("a", 100.0, 3), FakeBackend("b", 50.0, 3), FakeBackend("c", 10.0, 1)],
        started_at=time.time() - 10,
        proxy_mb=40.0,
    )
    assert snap["totals"]["memory_mb"] == 160.0  # noqa: PLR2004
    assert snap["totals"]["clients_active"] == 3  # noqa: PLR2004
    assert snap["totals"]["memory_avoided_mb"] == 300.0  # noqa: PLR2004  (3 clients: 100+50 x 2)


def test_one_backend_with_reconnect_noise_does_not_drive_the_estimate() -> None:
    snap = build_snapshot(
        [FakeBackend("a", 100.0, 26), FakeBackend("b", 50.0, 4), FakeBackend("c", 10.0, 4)],
        started_at=time.time() - 10,
        proxy_mb=40.0,
    )
    by_name = {b["name"]: b for b in snap["backends"]}
    assert by_name["a"]["clients_active"] == 26  # noqa: PLR2004  (table shows the raw count)
    assert by_name["a"]["clients_effective"] == 4  # noqa: PLR2004
    assert snap["totals"]["memory_avoided_mb"] == 480.0  # noqa: PLR2004  (160 MB x 3 extra)


def test_curl_probes_are_not_counted_as_clients() -> None:
    stats = BackendStats()
    stats.note_client("probe", "curl/8.7.1")
    stats.note_client("real", "node")
    assert stats.active_clients() == 1


def test_idle_clients_age_out() -> None:
    stats = BackendStats()
    stats.note_client("old", "node")
    stats.clients["old"] -= 3600
    assert stats.active_clients() == 0
    assert stats.clients_total == 1


def test_latency_percentiles_and_error_count() -> None:
    stats = BackendStats()
    for ms in range(1, 101):
        stats.record_call("call_tool", ms / 1000, failed=ms == 100)  # noqa: PLR2004
    assert stats.tool_calls == 100  # noqa: PLR2004
    assert stats.errors == 1
    assert 49 <= (stats.percentile_ms(0.5) or 0) <= 52  # noqa: PLR2004
    assert (stats.percentile_ms(0.95) or 0) >= 95  # noqa: PLR2004


@pytest.mark.parametrize(
    ("command", "args", "token"),
    [
        ("npx", ["-y", "figma-console-mcp@latest"], "figma-console-mcp@latest"),
        (
            "npx",
            ["-y", "--@scope:registry=https://x", "@shopify-internal/slack-mcp@latest"],
            "@shopify-internal/slack-mcp@latest",
        ),
        (
            "npx",
            ["@playwright/mcp@latest", "--cdp-endpoint", "http://127.0.0.1:9222"],
            "@playwright/mcp@latest",
        ),
        ("/opt/dev/bin/devx", [], "devx"),
    ],
)
def test_match_token_picks_the_package_not_the_flags(
    command: str, args: list[str], token: str
) -> None:
    assert match_token(command, args) == token


@pytest.fixture
def dashboard() -> tuple[int, DashboardState]:
    with socket.socket() as sock:
        sock.bind(("127.0.0.1", 0))
        port = sock.getsockname()[1]
    state = DashboardState()
    state.snapshot_json = json.dumps({"totals": {"backends": 1}}).encode()
    server = start_dashboard("127.0.0.1", port, state)
    yield port, state
    server.shutdown()
    server.server_close()


def _get(port: int, path: str, host: str | None = None) -> tuple[int, bytes, str]:
    conn = HTTPConnection("127.0.0.1", port, timeout=5)
    conn.putrequest("GET", path, skip_host=True)
    conn.putheader("Host", host or f"127.0.0.1:{port}")
    conn.endheaders()
    resp = conn.getresponse()
    body = resp.read()
    conn.close()
    return resp.status, body, resp.getheader("Content-Type") or ""


def test_page_and_metrics_are_served(dashboard: tuple[int, DashboardState]) -> None:
    port, _ = dashboard
    status, body, ctype = _get(port, "/")
    assert status == HTTPStatus.OK
    assert ctype.startswith("text/html")
    assert b"mcp-proxy" in body
    status, body, ctype = _get(port, "/api/metrics")
    assert status == HTTPStatus.OK
    assert json.loads(body) == {"totals": {"backends": 1}}
    assert ctype == "application/json"


def test_foreign_host_cannot_read_metrics(dashboard: tuple[int, DashboardState]) -> None:
    port, _ = dashboard
    status, body, _ = _get(port, "/api/metrics", host="evil.example.com")
    assert status == HTTPStatus.MISDIRECTED_REQUEST
    assert b"totals" not in body


def test_unknown_path_is_404(dashboard: tuple[int, DashboardState]) -> None:
    port, _ = dashboard
    assert _get(port, "/nope")[0] == HTTPStatus.NOT_FOUND
