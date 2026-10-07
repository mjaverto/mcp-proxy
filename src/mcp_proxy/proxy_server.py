"""Create an MCP server that proxies requests through an MCP client.

This server is created independent of any transport mechanism.
Supports both raw ClientSession (original) and ResilientBackend (auto-reconnect).
"""

import logging
import typing as t

from mcp import server, types
from mcp.client.session import ClientSession

logger = logging.getLogger(__name__)


async def create_proxy_server(
    remote_app: t.Any,  # ClientSession or ResilientBackend
) -> server.Server[object]:  # noqa: C901, PLR0915
    """Create a server instance from a remote app.

    remote_app can be a ClientSession (original behavior) or a
    ResilientBackend (auto-reconnect on failure).
    """
    from .resilient_backend import ResilientBackend

    is_resilient = isinstance(remote_app, ResilientBackend)

    if is_resilient:
        # ResilientBackend is already initialized
        logger.debug("Creating proxy server with resilient backend '%s'", remote_app.name)
        # We need to get capabilities by calling list methods and seeing what works
        # For resilient backends, we register all handlers and let errors propagate
        app: server.Server[object] = server.Server(name=remote_app.name)
        _register_all_handlers(app, remote_app, is_resilient=True)
        return app

    # Original path: raw ClientSession
    logger.debug("Sending initialization request to remote MCP server...")
    response = await remote_app.initialize()
    capabilities = response.capabilities

    logger.debug("Configuring proxied MCP server...")
    app = server.Server(name=response.serverInfo.name)
    _register_all_handlers(app, remote_app, is_resilient=False, capabilities=capabilities)
    return app


def _register_all_handlers(  # noqa: C901, PLR0915
    app: server.Server[object],
    remote_app: t.Any,
    *,
    is_resilient: bool,
    capabilities: t.Any = None,
) -> None:
    """Register request handlers on the server."""

    def _call(method: str, *args: t.Any, **kwargs: t.Any) -> t.Any:
        """Helper to call either ClientSession method or ResilientBackend.call()."""
        if is_resilient:
            return remote_app.call(method, *args, **kwargs)
        return getattr(remote_app, method)(*args, **kwargs)

    if is_resilient or (capabilities and capabilities.prompts):
        logger.debug("Capabilities: adding Prompts...")

        async def _list_prompts(_: t.Any) -> types.ServerResult:  # noqa: ANN401
            result = await _call("list_prompts")
            return types.ServerResult(result)

        app.request_handlers[types.ListPromptsRequest] = _list_prompts

        async def _get_prompt(req: types.GetPromptRequest) -> types.ServerResult:
            result = await _call("get_prompt", req.params.name, req.params.arguments)
            return types.ServerResult(result)

        app.request_handlers[types.GetPromptRequest] = _get_prompt

    if is_resilient or (capabilities and capabilities.resources):
        logger.debug("Capabilities: adding Resources...")

        async def _list_resources(_: t.Any) -> types.ServerResult:  # noqa: ANN401
            result = await _call("list_resources")
            return types.ServerResult(result)

        app.request_handlers[types.ListResourcesRequest] = _list_resources

        async def _list_resource_templates(_: t.Any) -> types.ServerResult:  # noqa: ANN401
            result = await _call("list_resource_templates")
            return types.ServerResult(result)

        app.request_handlers[types.ListResourceTemplatesRequest] = _list_resource_templates

        async def _read_resource(req: types.ReadResourceRequest) -> types.ServerResult:
            result = await _call("read_resource", req.params.uri)
            return types.ServerResult(result)

        app.request_handlers[types.ReadResourceRequest] = _read_resource

        async def _subscribe_resource(req: types.SubscribeRequest) -> types.ServerResult:
            await _call("subscribe_resource", req.params.uri)
            return types.ServerResult(types.EmptyResult())

        app.request_handlers[types.SubscribeRequest] = _subscribe_resource

        async def _unsubscribe_resource(req: types.UnsubscribeRequest) -> types.ServerResult:
            await _call("unsubscribe_resource", req.params.uri)
            return types.ServerResult(types.EmptyResult())

        app.request_handlers[types.UnsubscribeRequest] = _unsubscribe_resource

    if is_resilient or (capabilities and capabilities.logging):
        logger.debug("Capabilities: adding Logging...")

        async def _set_logging_level(req: types.SetLevelRequest) -> types.ServerResult:
            await _call("set_logging_level", req.params.level)
            return types.ServerResult(types.EmptyResult())

        app.request_handlers[types.SetLevelRequest] = _set_logging_level

    if is_resilient or (capabilities and capabilities.tools):
        logger.debug("Capabilities: adding Tools...")

        async def _list_tools(_: t.Any) -> types.ServerResult:  # noqa: ANN401
            tools = await _call("list_tools")
            return types.ServerResult(tools)

        app.request_handlers[types.ListToolsRequest] = _list_tools

        async def _call_tool(req: types.CallToolRequest) -> types.ServerResult:
            try:
                result = await _call(
                    "call_tool",
                    req.params.name,
                    (req.params.arguments or {}),
                )
                return types.ServerResult(result)
            except Exception as e:  # noqa: BLE001
                return types.ServerResult(
                    types.CallToolResult(
                        content=[types.TextContent(type="text", text=str(e))],
                        isError=True,
                    ),
                )

        app.request_handlers[types.CallToolRequest] = _call_tool

    async def _send_progress_notification(req: types.ProgressNotification) -> None:
        await _call(
            "send_progress_notification",
            req.params.progressToken,
            req.params.progress,
            req.params.total,
        )

    app.notification_handlers[types.ProgressNotification] = _send_progress_notification

    async def _complete(req: types.CompleteRequest) -> types.ServerResult:
        result = await _call(
            "complete",
            req.params.ref,
            req.params.argument.model_dump(),
        )
        return types.ServerResult(result)

    app.request_handlers[types.CompleteRequest] = _complete
