"""Per-backend counters and process memory sampling for the dashboard.

Everything here is in memory and reset when the proxy restarts. Command lines and
arguments are never exposed: they can hold registry URLs and tokens.
"""

import os
import re
import statistics
import subprocess
import time
import typing as t
from collections import deque
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path

CLIENT_ACTIVE_WINDOW_S = 600.0
MEMORY_HISTORY_POINTS = 60
LATENCY_SAMPLES = 500
# curl probes (the healthcheck) are not real clients of a backend.
IGNORED_USER_AGENT_PREFIXES = ("curl/",)


@dataclass
class BackendStats:
    """Counters for one backend."""

    calls: int = 0
    tool_calls: int = 0
    errors: int = 0
    last_call_at: float | None = None
    up_since: float | None = None
    durations_ms: deque[float] = field(default_factory=lambda: deque(maxlen=LATENCY_SAMPLES))
    clients: dict[str, float] = field(default_factory=dict)  # session id -> last seen
    clients_total: int = 0
    memory_mb: float | None = None
    memory_history: deque[float] = field(
        default_factory=lambda: deque(maxlen=MEMORY_HISTORY_POINTS)
    )

    def record_call(self, method: str, duration_s: float, *, failed: bool) -> None:
        """Count one request forwarded to the backend."""
        self.calls += 1
        if method == "call_tool":
            self.tool_calls += 1
        if failed:
            self.errors += 1
        self.last_call_at = time.time()
        self.durations_ms.append(duration_s * 1000)

    def note_client(self, session_id: str, user_agent: str | None) -> None:
        """Remember an MCP session id seen on a request to this backend."""
        if user_agent and user_agent.startswith(IGNORED_USER_AGENT_PREFIXES):
            return
        if session_id not in self.clients:
            self.clients_total += 1
        self.clients[session_id] = time.time()

    def active_clients(self) -> int:
        """Sessions seen within the active window; prunes older ones."""
        cutoff = time.time() - CLIENT_ACTIVE_WINDOW_S
        for sid in [s for s, seen in self.clients.items() if seen < cutoff]:
            del self.clients[sid]
        return len(self.clients)

    def percentile_ms(self, pct: float) -> float | None:
        """Latency percentile over the most recent calls."""
        if not self.durations_ms:
            return None
        ordered = sorted(self.durations_ms)
        return ordered[min(len(ordered) - 1, int(len(ordered) * pct))]


def _iso(ts: float | None) -> str | None:
    return datetime.fromtimestamp(ts, timezone.utc).isoformat() if ts else None


def match_token(command: str, args: list[str]) -> str:
    """A string that appears in the process command line of the backend's wrapper."""
    for arg in args:
        if not arg.startswith("-"):
            return arg
    return Path(command).name


def _ps_rows() -> list[tuple[int, int, int, str]]:
    out = subprocess.run(
        ["/bin/ps", "-axo", "pid=,ppid=,rss=,command="],
        capture_output=True,
        text=True,
        check=False,
        timeout=10,
    ).stdout
    rows = []
    for line in out.splitlines():
        parts = line.split(None, 3)
        if len(parts) == 4:  # noqa: PLR2004
            rows.append((int(parts[0]), int(parts[1]), int(parts[2]), parts[3]))
    return rows


_FOOTPRINT_RE = re.compile(r"\[(\d+)\]: .*?Footprint: ([\d.]+) (KB|MB|GB)")
_TO_MB = {"KB": 1 / 1024, "MB": 1.0, "GB": 1024.0}


def _footprints_mb(pids: list[int]) -> dict[int, float]:
    """Memory footprint per pid on macOS (counts compressed memory, unlike RSS); empty elsewhere."""  # noqa: D403
    if not pids or not Path("/usr/bin/footprint").exists():
        return {}
    cmd = ["/usr/bin/footprint"]
    for pid in pids:
        cmd += ["-p", str(pid)]
    out = subprocess.run(cmd, capture_output=True, text=True, check=False, timeout=30).stdout  # noqa: S603
    return {int(m[1]): float(m[2]) * _TO_MB[m[3]] for m in _FOOTPRINT_RE.finditer(out)}


def sample_memory_mb(tokens: dict[str, str]) -> dict[str, float]:
    """Memory (MB) of each backend's process tree, matched by wrapper command line.

    Uses macOS ``footprint`` when available, falling back to resident set size.
    """
    rows = _ps_rows()
    children: dict[int, list[int]] = {}
    for pid, ppid, _, _ in rows:
        children.setdefault(ppid, []).append(pid)
    rss = {pid: kb / 1024 for pid, _, kb, _ in rows}
    me = os.getpid()
    trees: dict[str, list[int]] = {}
    for name, token in tokens.items():
        for pid, ppid, _, command in rows:
            if ppid != me or token not in command:
                continue
            stack = [pid]
            while stack:
                cur = stack.pop()
                trees.setdefault(name, []).append(cur)
                stack.extend(children.get(cur, []))
    footprint = _footprints_mb([p for pids in trees.values() for p in pids])
    return {
        name: sum(footprint.get(p, rss.get(p, 0.0)) for p in pids) for name, pids in trees.items()
    }


def proxy_rss_mb() -> float:
    """Resident memory of the proxy process itself."""
    out = subprocess.run(  # noqa: S603
        ["/bin/ps", "-o", "rss=", "-p", str(os.getpid())],
        capture_output=True,
        text=True,
        check=False,
        timeout=10,
    ).stdout.strip()
    return int(out) / 1024 if out else 0.0


class Backend(t.Protocol):
    """What the snapshot needs from a supervised backend."""

    name: str
    state: str
    restarts: int
    stats: BackendStats


def build_snapshot(
    backends: t.Iterable[Backend], started_at: float, proxy_mb: float
) -> dict[str, t.Any]:
    """The JSON served to the dashboard."""
    now = time.time()
    rows: list[dict[str, t.Any]] = []
    for b in backends:
        s = b.stats
        rows.append(
            {
                "name": b.name,
                "state": b.state,
                "restarts": b.restarts,
                "uptime_s": round(now - s.up_since) if s.up_since and b.state == "up" else None,
                "memory_mb": round(s.memory_mb, 1) if s.memory_mb is not None else None,
                "memory_history": [round(x, 1) for x in s.memory_history],
                "clients_active": s.active_clients(),
                "clients_total": s.clients_total,
                "calls": s.calls,
                "tool_calls": s.tool_calls,
                "errors": s.errors,
                "p50_ms": round(p, 1) if (p := s.percentile_ms(0.5)) is not None else None,
                "p95_ms": round(p, 1) if (p := s.percentile_ms(0.95)) is not None else None,
                "last_call_at": _iso(s.last_call_at),
            }
        )
    held = sum(r["memory_mb"] or 0 for r in rows)
    # A client that reconnects leaves its old session behind for a while, so one backend can
    # show far more sessions than the others. Cap each at the typical count among backends
    # that have any, so the estimate is not driven by reconnect noise.
    seen = [r["clients_active"] for r in rows if r["clients_active"] > 0]
    cap = int(statistics.median(seen)) if seen else 0
    for r in rows:
        r["clients_effective"] = min(r["clients_active"], cap)
    # If each extra client had started its own copy, they would hold this much more.
    avoided = sum((r["memory_mb"] or 0) * max(r["clients_effective"] - 1, 0) for r in rows)
    return {
        "now": _iso(now),
        "proxy": {
            "started_at": _iso(started_at),
            "uptime_s": round(now - started_at),
            "memory_mb": round(proxy_mb, 1),
        },
        "totals": {
            "backends": len(rows),
            "up": sum(1 for r in rows if r["state"] == "up"),
            "restarts": sum(r["restarts"] for r in rows),
            "calls": sum(r["calls"] for r in rows),
            "tool_calls": sum(r["tool_calls"] for r in rows),
            "errors": sum(r["errors"] for r in rows),
            "memory_mb": round(held, 1),
            "clients_active": cap,
            "memory_avoided_mb": round(avoided, 1),
        },
        "backends": rows,
    }
