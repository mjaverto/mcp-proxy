"""Keep a stdio MCP backend running and restart it when its process dies.

One supervisor task owns the backend's ``stdio_client`` and ``ClientSession``: it
enters and exits them itself, so anyio cancel scopes never cross tasks. Callers
only send requests on the live session. JSON-RPC errors from the backend go
straight back to the caller; only a dead process or closed stream triggers a
restart.
"""

import logging
import typing as t

import anyio
from anyio.abc import TaskStatus
from mcp import McpError, types
from mcp.client.session import ClientSession
from mcp.client.stdio import StdioServerParameters, stdio_client
from mcp.shared.message import SessionMessage

logger = logging.getLogger(__name__)

TRANSPORT_ERRORS = (
    anyio.ClosedResourceError,
    anyio.BrokenResourceError,
    anyio.EndOfStream,
    BrokenPipeError,
)
READY_TIMEOUT_S = 60.0
BACKOFF_MAX_S = 30.0


def _is_transport_error(exc: BaseException) -> bool:
    if isinstance(exc, McpError):
        return exc.error.code == types.CONNECTION_CLOSED
    return isinstance(exc, TRANSPORT_ERRORS)


class SupervisedBackend:
    """Duck-types the ``ClientSession`` methods ``create_proxy_server`` uses."""

    def __init__(self, name: str, params: StdioServerParameters) -> None:
        """Create a supervisor for one stdio backend; call ``run`` to start it."""
        self.name = name
        self.params = params
        self.init_result: types.InitializeResult | None = None
        self.restarts = 0
        self.state = "starting"
        self._session: ClientSession | None = None
        self._ready = anyio.Event()
        self._dead = anyio.Event()
        self._stopping = False
        self._finished = anyio.Event()
        self._scope = anyio.CancelScope()
        self._starting: anyio.CancelScope | None = None

    def status(self) -> dict[str, t.Any]:
        """Return state and restart count for ``/status``."""
        return {"state": self.state, "restarts": self.restarts}

    def stop(self) -> None:
        """Stop supervising: shut the backend down and never restart it.

        A live backend is ended by releasing its session wait, and one still starting by
        cancelling only its ``initialize``, never by cancelling ``stdio_client``: its exit
        closes stdin, waits, then kills the process group, and under cancellation that
        wait aborts before the kill. A backend that ignores stdin EOF (figma-console keeps
        its WebSocket server up, npx keeps downloading) would be orphaned. Between
        attempts there is no process, so the supervisor itself is cancelled.
        """
        self.state = "stopped"
        self._stopping = True
        if self._session is not None:
            self._dead.set()
        elif self._starting is not None:
            self._starting.cancel()
        else:
            self._scope.cancel()

    async def aclose(self) -> None:
        """``stop`` and wait until the backend process is gone."""
        self.stop()
        await self._finished.wait()

    async def run(self, *, task_status: TaskStatus[None] = anyio.TASK_STATUS_IGNORED) -> None:
        """Supervise until ``stop``. Signals ``task_status`` once the first attempt starts."""
        task_status.started()
        backoff = 1.0
        try:
            with self._scope:
                while True:
                    try:
                        await self._serve_once()
                        backoff = 1.0  # served until the process died: restart promptly
                    except Exception as exc:  # noqa: BLE001 - any failure to start is a transport failure
                        logger.warning("Backend '%s' failed: %r", self.name, exc)
                    if self._stopping:
                        return
                    self._session = None
                    # Replace only a fired event; waiters on an unset one stay valid.
                    if self._ready.is_set():
                        self._ready = anyio.Event()
                    self.state = "restarting"
                    self.restarts += 1
                    logger.warning("Restarting backend '%s' in %.0fs", self.name, backoff)
                    await anyio.sleep(backoff)
                    backoff = min(backoff * 2, BACKOFF_MAX_S)
        finally:
            self._finished.set()

    async def _serve_once(self) -> None:
        self._dead = anyio.Event()
        starting = self._starting = anyio.CancelScope()
        try:
            async with (
                stdio_client(self.params) as (read, write),
                anyio.create_task_group() as tg,
            ):
                # Relay the backend's output so we see EOF the moment the process exits.
                relay_send, relay_recv = anyio.create_memory_object_stream[
                    SessionMessage | Exception
                ](0)

                async def relay() -> None:
                    async with relay_send:
                        async for msg in read:
                            await relay_send.send(msg)
                    self._dead.set()

                tg.start_soon(relay)
                async with ClientSession(relay_recv, write) as session:
                    with starting:
                        result = await session.initialize()
                    self._starting = None
                    if not starting.cancelled_caught:
                        if self.init_result is None:
                            self.init_result = result
                        self._session = session
                        self.state = "up"
                        self._ready.set()
                        logger.info("Backend '%s' is up", self.name)
                        await self._dead.wait()
                        if not self._stopping:
                            logger.warning("Backend '%s' connection lost", self.name)
                        self._session = None
                        self._ready = anyio.Event()
                tg.cancel_scope.cancel()
        finally:
            self._starting = None

    async def _live_session(self) -> ClientSession:
        with anyio.fail_after(READY_TIMEOUT_S):
            while self._session is None:
                await self._ready.wait()
        return self._session

    async def initialize(self) -> types.InitializeResult:
        """Wait for the backend and return its first ``initialize`` result."""
        await self._live_session()
        assert self.init_result is not None  # noqa: S101
        return self.init_result

    def __getattr__(self, method: str) -> t.Callable[..., t.Awaitable[t.Any]]:
        """Forward any ``ClientSession`` request method to the live session."""

        async def forward(*args: t.Any, **kwargs: t.Any) -> t.Any:  # noqa: ANN401
            session = await self._live_session()
            try:
                return await getattr(session, method)(*args, **kwargs)
            except Exception as exc:
                if _is_transport_error(exc):
                    self._dead.set()
                raise

        return forward
