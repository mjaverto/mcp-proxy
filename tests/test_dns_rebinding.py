"""A web page that rebinds its hostname to 127.0.0.1 must not reach the backends."""
# ruff: noqa: D103

import contextlib
from collections.abc import AsyncIterator
from http import HTTPStatus

import pytest
from httpx import ASGITransport, AsyncClient
from mcp.server import Server
from starlette.applications import Starlette

from mcp_proxy.mcp_server import create_single_instance_routes, loopback_security_settings

PORT = 9700
INITIALIZE = {
    "jsonrpc": "2.0",
    "id": 1,
    "method": "initialize",
    "params": {
        "protocolVersion": "2025-06-18",
        "capabilities": {},
        "clientInfo": {"name": "test", "version": "0"},
    },
}
HEADERS = {"Accept": "application/json, text/event-stream"}


@contextlib.asynccontextmanager
async def client_for(bind_host: str) -> AsyncIterator[AsyncClient]:
    routes, manager = create_single_instance_routes(
        Server("rebind-test"),
        stateless_instance=True,
        security_settings=loopback_security_settings(bind_host, PORT),
    )

    @contextlib.asynccontextmanager
    async def lifespan(_app: Starlette) -> AsyncIterator[None]:
        async with manager.run():
            yield

    app = Starlette(routes=routes, lifespan=lifespan)
    async with (
        lifespan(app),
        AsyncClient(transport=ASGITransport(app=app), base_url="http://test") as client,
    ):
        yield client


@pytest.mark.asyncio
@pytest.mark.parametrize("host", [f"127.0.0.1:{PORT}", f"localhost:{PORT}"])
async def test_loopback_host_is_served(host: str) -> None:
    async with client_for("127.0.0.1") as client:
        response = await client.post("/mcp", json=INITIALIZE, headers={**HEADERS, "Host": host})
    assert response.status_code == HTTPStatus.OK


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "headers",
    [
        {"Host": "evil.example.com"},
        {"Host": f"127.0.0.1:{PORT}", "Origin": "http://evil.example.com"},
    ],
)
async def test_foreign_host_or_origin_is_rejected(headers: dict[str, str]) -> None:
    async with client_for("127.0.0.1") as client:
        response = await client.post("/mcp", json=INITIALIZE, headers={**HEADERS, **headers})
    assert HTTPStatus.BAD_REQUEST <= response.status_code < HTTPStatus.INTERNAL_SERVER_ERROR


@pytest.mark.asyncio
async def test_non_loopback_bind_keeps_previous_behaviour() -> None:
    async with client_for("0.0.0.0") as client:  # noqa: S104
        response = await client.post(
            "/mcp", json=INITIALIZE, headers={**HEADERS, "Host": "proxy.internal"}
        )
    assert response.status_code == HTTPStatus.OK
