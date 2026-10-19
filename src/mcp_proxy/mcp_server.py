"""Create a local SSE server that proxies requests to a stdio MCP server."""

import contextlib
import logging
from collections.abc import AsyncIterator, Awaitable, Callable
from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Any, Final, Literal

import anyio
import uvicorn
from mcp.client.stdio import StdioServerParameters
from mcp.server import Server as MCPServerSDK  # Renamed to avoid conflict
from mcp.server.sse import SseServerTransport
from mcp.server.streamable_http_manager import StreamableHTTPSessionManager
from mcp.server.transport_security import TransportSecuritySettings
from starlette.applications import Starlette
from starlette.middleware import Middleware
from starlette.middleware.cors import CORSMiddleware
from starlette.requests import Request
from starlette.responses import JSONResponse, Response
from starlette.routing import BaseRoute, Mount, Route
from starlette.types import Receive, Scope, Send

from .proxy_server import create_proxy_server
from .supervised_backend import SupervisedBackend

logger = logging.getLogger(__name__)

DEFAULT_EXPOSE_HEADERS: Final[tuple[str, ...]] = ("mcp-session-id",)
# stdio_client waits 2s for exit after closing stdin, then up to 2s more for SIGTERM/SIGKILL.
BACKEND_SHUTDOWN_TIMEOUT_S: Final = 10.0
# How long to wait for backends before serving; slower ones are mounted when they come up.
STARTUP_GRACE_S: Final = 15.0


def _default_expose_headers() -> list[str]:
    return list(DEFAULT_EXPOSE_HEADERS)


@dataclass
class MCPServerSettings:
    """Settings for the MCP server."""

    bind_host: str
    port: int
    stateless: bool = False
    allow_origins: list[str] | None = None
    expose_headers: list[str] = field(default_factory=_default_expose_headers)
    log_level: Literal["DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL"] = "INFO"


# To store last activity for multiple servers if needed, though status endpoint is global for now.
_global_status: dict[str, Any] = {
    "api_last_activity": datetime.now(timezone.utc).isoformat(),
    "active_sse_connections": 0,
    "server_instances": {},  # Could be used to store per-instance status later
}
_backends: dict[str, SupervisedBackend] = {}


def _update_global_activity() -> None:
    _global_status["api_last_activity"] = datetime.now(timezone.utc).isoformat()


class _ASGIEndpointAdapter:
    """Wrap a coroutine function into an ASGI application."""

    def __init__(self, endpoint: Callable[[Scope, Receive, Send], Awaitable[None]]) -> None:
        self._endpoint = endpoint

    async def __call__(self, scope: Scope, receive: Receive, send: Send) -> None:
        await self._endpoint(scope, receive, send)


HTTP_METHODS = ["DELETE", "GET", "HEAD", "OPTIONS", "PATCH", "POST", "PUT", "TRACE"]


async def _handle_status(_: Request) -> Response:
    """Global health check and service usage monitoring endpoint."""
    instances = {name: backend.status() for name, backend in _backends.items()}
    return JSONResponse({**_global_status, "server_instances": instances})


async def _handle_healthz(_: Request) -> Response:
    """Respond to a transport liveness probe without allocating an MCP session."""
    return Response(status_code=204)


LOOPBACK_HOSTS: Final[tuple[str, ...]] = ("127.0.0.1", "localhost", "::1")


def loopback_security_settings(
    bind_host: str, port: int, allow_origins: list[str] | None = None
) -> TransportSecuritySettings | None:
    """Reject requests whose Host or Origin isn't this loopback listener (DNS rebinding).

    Without it, a web page can rebind its own hostname to 127.0.0.1 and call every backend.
    Origins from ``--allow-origin`` are allowed too; ``*`` opts out of the check, as the SDK
    cannot match a bare wildcard. Returns ``None`` (no check, the SDK default) when bound to a
    non-loopback address, where the expected Host is unknown.
    """
    if bind_host not in LOOPBACK_HOSTS or "*" in (allow_origins or []):
        return None
    hosts = [f"127.0.0.1:{port}", f"localhost:{port}", f"[::1]:{port}"]
    return TransportSecuritySettings(
        enable_dns_rebinding_protection=True,
        allowed_hosts=hosts,
        allowed_origins=[f"http://{h}" for h in hosts] + list(allow_origins or []),
    )


def create_single_instance_routes(
    mcp_server_instance: MCPServerSDK[object],
    *,
    stateless_instance: bool,
    security_settings: TransportSecuritySettings | None = None,
) -> tuple[list[BaseRoute], StreamableHTTPSessionManager]:  # Return the manager itself
    """Create Starlette routes and the HTTP session manager for a single MCP server instance."""
    logger.debug(
        "Creating routes for a single MCP server instance (stateless: %s)",
        stateless_instance,
    )

    sse_transport = SseServerTransport("/messages/", security_settings=security_settings)
    http_session_manager = StreamableHTTPSessionManager(
        app=mcp_server_instance,
        event_store=None,
        json_response=True,
        stateless=stateless_instance,
        security_settings=security_settings,
    )

    async def handle_sse_instance(request: Request) -> Response:
        try:
            async with sse_transport.connect_sse(
                request.scope,
                request.receive,
                request._send,  # noqa: SLF001
            ) as (read_stream, write_stream):
                _update_global_activity()
                _global_status["active_sse_connections"] += 1
                try:
                    await mcp_server_instance.run(
                        read_stream,
                        write_stream,
                        mcp_server_instance.create_initialization_options(),
                        stateless=stateless_instance,
                    )
                finally:
                    _global_status["active_sse_connections"] -= 1
        except Exception:  # noqa: BLE001
            # SSE response already started — cannot send HTTP error response.
            # Log and let the connection close gracefully.
            logger.warning(
                "SSE connection for '%s' terminated with error (upstream disconnect?)",
                mcp_server_instance.name,
                exc_info=True,
            )
        return Response()

    async def handle_streamable_http_instance(scope: Scope, receive: Receive, send: Send) -> None:
        _update_global_activity()
        updated_scope = scope
        if scope.get("type") == "http":
            path = scope.get("path", "")
            if path and path.rstrip("/") == "/mcp" and not path.endswith("/"):
                updated_scope = dict(scope)
                normalized_path = path + "/"
                logger.debug(
                    "Normalized request path from '%s' to '%s' without redirect",
                    path,
                    normalized_path,
                )
                updated_scope["path"] = normalized_path

                raw_path = scope.get("raw_path")
                if raw_path:
                    if b"?" in raw_path:
                        path_part, query_part = raw_path.split(b"?", 1)
                        updated_scope["raw_path"] = path_part.rstrip(b"/") + b"/?" + query_part
                    else:
                        updated_scope["raw_path"] = raw_path.rstrip(b"/") + b"/"

        await http_session_manager.handle_request(updated_scope, receive, send)

    routes = [
        Route("/healthz", endpoint=_handle_healthz, methods=["GET", "HEAD"]),
        Route(
            "/mcp",
            endpoint=_ASGIEndpointAdapter(handle_streamable_http_instance),
            methods=HTTP_METHODS,
            include_in_schema=False,
        ),
        Mount("/mcp", app=handle_streamable_http_instance),
        Route("/sse", endpoint=handle_sse_instance),
        Mount("/messages/", app=sse_transport.handle_post_message),
    ]
    return routes, http_session_manager


async def run_mcp_server(  # noqa: C901, PLR0915
    mcp_settings: MCPServerSettings,
    default_server_params: StdioServerParameters | None = None,
    named_server_params: dict[str, StdioServerParameters] | None = None,
) -> None:
    """Run stdio client(s) and expose an MCP server with multiple possible backends."""
    if named_server_params is None:
        named_server_params = {}
    if not default_server_params and not named_server_params:
        logger.error("No servers configured to run.")
        return

    all_routes: list[BaseRoute] = [
        Route("/status", endpoint=_handle_status),  # Global status endpoint
    ]
    # Use AsyncExitStack to manage lifecycles of multiple components
    async with contextlib.AsyncExitStack() as stack:
        # Manage lifespans of all StreamableHTTPSessionManagers
        @contextlib.asynccontextmanager
        async def combined_lifespan(_app: Starlette) -> AsyncIterator[None]:
            logger.info("Main application lifespan starting...")
            # Each backend's session manager runs inside its serve_backend task.
            yield
            logger.info("Main application lifespan shutting down...")
            # Kill backends here, not after serve() returns: uvicorn re-raises SIGTERM as
            # soon as it returns, so the process dies before the exit stack unwinds.
            with anyio.move_on_after(BACKEND_SHUTDOWN_TIMEOUT_S):
                async with anyio.create_task_group() as closing:
                    for backend in _backends.values():
                        closing.start_soon(backend.aclose)

        # Each stdio backend runs under its own supervisor task, which restarts it if it dies.
        # Stop them rather than cancel the group, so each backend's process is actually
        # killed (see SupervisedBackend.stop); the group then exits on its own.
        supervisors = await stack.enter_async_context(anyio.create_task_group())

        def stop_backends() -> None:
            for backend in _backends.values():
                backend.stop()

        stack.callback(stop_backends)

        middleware: list[Middleware] = []
        if mcp_settings.allow_origins:
            middleware.append(
                Middleware(
                    CORSMiddleware,
                    allow_origins=mcp_settings.allow_origins,
                    allow_methods=["*"],
                    allow_headers=["*"],
                    expose_headers=mcp_settings.expose_headers,
                ),
            )

        starlette_app = Starlette(
            debug=(mcp_settings.log_level == "DEBUG"),
            routes=all_routes,
            middleware=middleware,
            lifespan=combined_lifespan,
        )

        starlette_app.router.redirect_slashes = False

        # Mount tasks run until shutdown; cancel them before the backends stop.
        mounts = await stack.enter_async_context(anyio.create_task_group())
        stack.callback(mounts.cancel_scope.cancel)

        async def serve_backend(name: str, mounted: anyio.Event) -> None:
            """Mount a backend once it first comes up. A slow one never blocks the rest."""
            while True:
                try:
                    proxy = await create_proxy_server(_backends[name])  # type: ignore[arg-type]
                    break
                except TimeoutError:
                    logger.warning("Server '%s' not up yet; serving the others meanwhile", name)
            routes, http_manager = create_single_instance_routes(
                proxy,
                stateless_instance=mcp_settings.stateless,
                security_settings=loopback_security_settings(
                    mcp_settings.bind_host, mcp_settings.port, mcp_settings.allow_origins
                ),
            )
            async with http_manager.run():
                if name == "default" and default_server_params:
                    starlette_app.router.routes.extend(routes)
                else:
                    starlette_app.router.routes.append(Mount(f"/servers/{name}", routes=routes))
                mounted.set()
                await anyio.sleep_forever()

        backend_params = dict(named_server_params)
        if default_server_params:
            backend_params = {"default": default_server_params, **backend_params}

        mounted: list[anyio.Event] = []
        for name, params in backend_params.items():
            logger.info(
                "Setting up server '%s': %s %s", name, params.command, " ".join(params.args)
            )
            _backends[name] = SupervisedBackend(name, params)
            await supervisors.start(_backends[name].run)
            mounted.append(anyio.Event())
            mounts.start_soon(serve_backend, name, mounted[-1])

        # Give fast backends a moment so the first requests after a restart find them.
        with anyio.move_on_after(STARTUP_GRACE_S):
            for event in mounted:
                await event.wait()

        config = uvicorn.Config(
            starlette_app,
            host=mcp_settings.bind_host,
            port=mcp_settings.port,
            log_level=mcp_settings.log_level.lower(),
        )
        http_server = uvicorn.Server(config)

        # Print out the SSE URLs for all configured servers
        base_url = f"http://{mcp_settings.bind_host}:{mcp_settings.port}"
        sse_urls = []

        # Add default server if configured
        if default_server_params:
            sse_urls.append(f"{base_url}/sse")

        # Add named servers
        sse_urls.extend([f"{base_url}/servers/{name}/sse" for name in named_server_params])

        # Display the SSE URLs prominently
        if sse_urls:
            # Using print directly for user visibility, with noqa to ignore linter warnings
            logger.info("Serving MCP Servers via SSE:")
            for url in sse_urls:
                logger.info("  - %s", url)

        logger.debug(
            "Serving incoming MCP requests on %s:%s",
            mcp_settings.bind_host,
            mcp_settings.port,
        )
        await http_server.serve()
