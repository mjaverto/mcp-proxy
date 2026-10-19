"""SupervisedBackend against a real stdio MCP server process."""
# ruff: noqa: D103, ASYNC110

import contextlib
import os
import signal
import socket
import sys
import textwrap
from collections.abc import AsyncIterator
from pathlib import Path

import anyio
import httpx
import pytest
import uvicorn
from mcp import McpError, types
from mcp.client.stdio import StdioServerParameters

from mcp_proxy import mcp_server, supervised_backend
from mcp_proxy.mcp_server import MCPServerSettings, run_mcp_server
from mcp_proxy.supervised_backend import SupervisedBackend

FAKE_SERVER = textwrap.dedent(
    """
    import os
    import anyio
    from mcp import McpError, types
    from mcp.server.lowlevel import Server
    from mcp.server.stdio import stdio_server

    app = Server("fake")

    @app.list_tools()
    async def list_tools():
        return [types.Tool(name="pid", inputSchema={"type": "object"})]

    @app.call_tool()
    async def call_tool(name, arguments):
        return [types.TextContent(type="text", text=str(os.getpid()))]

    @app.list_resources()  # advertises resources, then refuses, like figma-console
    async def list_resources():
        raise McpError(types.ErrorData(code=types.METHOD_NOT_FOUND, message="Method not found"))

    async def main():
        async with stdio_server() as (r, w):
            await app.run(r, w, app.create_initialization_options())

    anyio.run(main)
    """,
)


@pytest.fixture
def params(tmp_path: Path) -> StdioServerParameters:
    script = tmp_path / "fake_server.py"
    script.write_text(FAKE_SERVER)
    return StdioServerParameters(command=sys.executable, args=[str(script)])


@contextlib.asynccontextmanager
async def two_backends(
    params: StdioServerParameters,
) -> AsyncIterator[tuple[SupervisedBackend, SupervisedBackend]]:
    a, b = SupervisedBackend("a", params), SupervisedBackend("b", params)
    async with anyio.create_task_group() as tg:
        await tg.start(a.run)
        await tg.start(b.run)
        await a.initialize()
        await b.initialize()
        yield a, b
        tg.cancel_scope.cancel()


async def _pid(backend: SupervisedBackend) -> int:
    result = await backend.call_tool("pid", {})
    return int(result.content[0].text)


async def _wait_for_restart(backend: SupervisedBackend) -> None:
    with anyio.fail_after(10):
        while backend.restarts == 0:
            await anyio.sleep(0.05)


async def test_killed_backend_restarts_and_next_call_works(params: StdioServerParameters) -> None:
    async with two_backends(params) as (a, _):
        old_pid = await _pid(a)
        os.kill(old_pid, signal.SIGKILL)
        await _wait_for_restart(a)
        assert await _pid(a) != old_pid
        assert a.status() == {"state": "up", "restarts": 1}


async def test_jsonrpc_error_is_returned_without_restart(params: StdioServerParameters) -> None:
    async with two_backends(params) as (a, _):
        old_pid = await _pid(a)
        with pytest.raises(McpError) as exc:
            await a.list_resources()
        assert exc.value.error.code == types.METHOD_NOT_FOUND
        assert await _pid(a) == old_pid
        assert a.restarts == 0


async def test_crashing_backend_does_not_affect_another(params: StdioServerParameters) -> None:
    async with two_backends(params) as (a, b):
        b_pid = await _pid(b)
        os.kill(await _pid(a), signal.SIGKILL)
        await _wait_for_restart(a)
        await _pid(a)
        assert await _pid(b) == b_pid
        assert b.restarts == 0


async def test_stop_kills_backend_that_outlives_stdin_eof(tmp_path: Path) -> None:
    # Like `npx figma-console-mcp`: a wrapper process runs the real server as a child, and
    # the server keeps running after stdin closes. stop() must kill the server, not just
    # the wrapper, or it is orphaned holding its ports.
    script = tmp_path / "lingering_server.py"
    script.write_text(FAKE_SERVER + "import time\ntime.sleep(600)\n")
    wrapped = StdioServerParameters(
        command="/bin/sh", args=["-c", f"'{sys.executable}' '{script}'; true"]
    )
    backend = SupervisedBackend("a", wrapped)
    with anyio.fail_after(15):
        async with anyio.create_task_group() as tg:
            await tg.start(backend.run)
            await backend.initialize()
            pid = await _pid(backend)
            backend.stop()
    with pytest.raises(ProcessLookupError):
        os.kill(pid, 0)


async def _initialize(client: httpx.AsyncClient, name: str) -> int:
    response = await client.post(
        f"/servers/{name}/mcp",
        json={
            "jsonrpc": "2.0",
            "id": 1,
            "method": "initialize",
            "params": {
                "protocolVersion": "2025-06-18",
                "capabilities": {},
                "clientInfo": {"name": "test", "version": "0"},
            },
        },
        headers={"Accept": "application/json, text/event-stream"},
    )
    return response.status_code


async def test_slow_backend_does_not_delay_serving_others(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    # A slow `npx` start once kept the whole proxy from serving for hours. The proxy must
    # serve the fast backend right away and mount the slow one once it comes up.
    fast = tmp_path / "fast_server.py"
    fast.write_text(FAKE_SERVER)
    slow = tmp_path / "slow_server.py"
    slow.write_text("import time\ntime.sleep(5)\n" + FAKE_SERVER)
    monkeypatch.setattr(mcp_server, "STARTUP_GRACE_S", 1.0)
    monkeypatch.setattr(supervised_backend, "READY_TIMEOUT_S", 2.0)  # slow outlasts one wait
    monkeypatch.setattr(mcp_server, "_backends", {})

    servers: list[uvicorn.Server] = []

    class CapturedServer(uvicorn.Server):
        def __init__(self, config: uvicorn.Config) -> None:
            super().__init__(config)
            servers.append(self)

    monkeypatch.setattr(uvicorn, "Server", CapturedServer)
    with socket.socket() as sock:
        sock.bind(("127.0.0.1", 0))
        port = sock.getsockname()[1]
    named = {
        "fast": StdioServerParameters(command=sys.executable, args=[str(fast)]),
        "slow": StdioServerParameters(command=sys.executable, args=[str(slow)]),
    }

    with anyio.fail_after(40):
        async with anyio.create_task_group() as tg:
            tg.start_soon(
                run_mcp_server, MCPServerSettings(bind_host="127.0.0.1", port=port), None, named
            )
            while not (servers and servers[0].started):
                await anyio.sleep(0.05)
            async with httpx.AsyncClient(base_url=f"http://127.0.0.1:{port}") as client:
                while await _initialize(client, "fast") != httpx.codes.OK:
                    await anyio.sleep(0.1)
                assert mcp_server._backends["slow"].state != "up"  # noqa: SLF001
                assert await _initialize(client, "slow") == httpx.codes.NOT_FOUND
                while await _initialize(client, "slow") != httpx.codes.OK:
                    await anyio.sleep(0.2)
            servers[0].should_exit = True
