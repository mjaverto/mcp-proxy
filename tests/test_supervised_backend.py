"""SupervisedBackend against a real stdio MCP server process."""
# ruff: noqa: D103, ASYNC110

import contextlib
import os
import signal
import sys
import textwrap
from collections.abc import AsyncIterator
from pathlib import Path

import anyio
import pytest
from mcp import McpError, types
from mcp.client.stdio import StdioServerParameters

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
    wrapped = StdioServerParameters(command="/bin/sh", args=["-c", f"'{sys.executable}' '{script}'; true"])
    backend = SupervisedBackend("a", wrapped)
    with anyio.fail_after(15):
        async with anyio.create_task_group() as tg:
            await tg.start(backend.run)
            await backend.initialize()
            pid = await _pid(backend)
            backend.stop()
    with pytest.raises(ProcessLookupError):
        os.kill(pid, 0)
