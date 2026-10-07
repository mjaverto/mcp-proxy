"""Resilient backend that auto-reconnects a stdio MCP server on failure."""

import asyncio
import contextlib
import logging
import typing as t

from mcp import types
from mcp.client.session import ClientSession
from mcp.client.stdio import StdioServerParameters, stdio_client

logger = logging.getLogger(__name__)

MAX_RECONNECT_ATTEMPTS = 3
RECONNECT_DELAY_SECS = 5


class ResilientBackend:
    """Wraps a stdio MCP backend with auto-reconnect on failure.

    Each request goes through _call() which catches connection errors,
    respawns the stdio process, and retries once.
    """

    def __init__(self, name: str, params: StdioServerParameters) -> None:
        self.name = name
        self.params = params
        self._session: ClientSession | None = None
        self._lock = asyncio.Lock()
        self._stdio_ctx: t.Any = None
        self._session_ctx: t.Any = None
        self._healthy = False

    async def start(self) -> None:
        """Start the backend stdio process and initialize the session."""
        await self._connect()

    async def _connect(self) -> None:
        """Spawn stdio process and create ClientSession."""
        # Clean up old connection if any
        await self._disconnect()

        logger.info("Connecting backend '%s': %s %s", self.name, self.params.command, " ".join(self.params.args))
        try:
            self._stdio_ctx = stdio_client(self.params)
            streams = await self._stdio_ctx.__aenter__()
            self._session_ctx = ClientSession(*streams)
            self._session = await self._session_ctx.__aenter__()
            await self._session.initialize()
            self._healthy = True
            logger.info("Backend '%s' connected successfully.", self.name)
        except Exception:
            logger.exception("Backend '%s' failed to connect.", self.name)
            self._healthy = False
            await self._disconnect()
            raise

    async def _disconnect(self) -> None:
        """Tear down existing connection."""
        self._healthy = False
        self._session = None
        if self._session_ctx is not None:
            with contextlib.suppress(Exception):
                await self._session_ctx.__aexit__(None, None, None)
            self._session_ctx = None
        if self._stdio_ctx is not None:
            with contextlib.suppress(Exception):
                await self._stdio_ctx.__aexit__(None, None, None)
            self._stdio_ctx = None

    async def _reconnect(self) -> bool:
        """Try to reconnect. Returns True on success."""
        for attempt in range(1, MAX_RECONNECT_ATTEMPTS + 1):
            logger.warning(
                "Backend '%s' reconnect attempt %d/%d",
                self.name, attempt, MAX_RECONNECT_ATTEMPTS,
            )
            try:
                await self._connect()
                return True
            except Exception:
                if attempt < MAX_RECONNECT_ATTEMPTS:
                    await asyncio.sleep(RECONNECT_DELAY_SECS)
        logger.error("Backend '%s' failed to reconnect after %d attempts.", self.name, MAX_RECONNECT_ATTEMPTS)
        return False

    def _is_stale_result(self, fn_name: str, result: t.Any) -> bool:
        """Detect stale sessions that return empty data instead of raising."""
        if fn_name == "list_tools":
            tools = getattr(result, "tools", None)
            if tools is not None and len(tools) == 0:
                return True
        return False

    async def call(self, fn_name: str, *args: t.Any, **kwargs: t.Any) -> t.Any:
        """Call a method on the session, reconnecting on failure."""
        async with self._lock:
            if not self._healthy or self._session is None:
                if not await self._reconnect():
                    raise ConnectionError(f"Backend '{self.name}' is unavailable")

            try:
                method = getattr(self._session, fn_name)
                result = await method(*args, **kwargs)
                # Detect stale sessions that return empty instead of erroring
                if self._is_stale_result(fn_name, result):
                    logger.warning(
                        "Backend '%s' returned stale result for %s (0 tools). Reconnecting.",
                        self.name, fn_name,
                    )
                    self._healthy = False
                    if await self._reconnect():
                        method = getattr(self._session, fn_name)
                        return await method(*args, **kwargs)
                    raise ConnectionError(f"Backend '{self.name}' stale after reconnect")
                return result
            except Exception as first_error:
                logger.warning(
                    "Backend '%s' call to %s failed: %s. Attempting reconnect.",
                    self.name, fn_name, first_error,
                )
                self._healthy = False
                if await self._reconnect():
                    try:
                        method = getattr(self._session, fn_name)
                        return await method(*args, **kwargs)
                    except Exception:
                        logger.exception("Backend '%s' retry of %s failed after reconnect.", self.name, fn_name)
                        self._healthy = False
                        raise
                raise

    async def close(self) -> None:
        """Shut down the backend."""
        await self._disconnect()

    @property
    def is_healthy(self) -> bool:
        return self._healthy
