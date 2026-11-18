"""Tests for the MCP bridge: schema mapping, execute() failure modes,
projection of results, allow-filtering, auth headers.

Everything is offline: the MCP server runs in-process via FastMCP over
memory streams (docs/mcp-tools-spec.md §11) — no network anywhere. The
session is opened inside each test (`async with`), one task end-to-end:
an async-generator fixture would tear the anyio cancel scopes apart.
"""

import json
from contextlib import AsyncExitStack, asynccontextmanager
from typing import Any, AsyncIterator, cast

import pytest
from mcp import ClientSession
from mcp.server.fastmcp import FastMCP
from mcp.shared.memory import create_connected_server_and_client_session
from mcp.types import CallToolResult, ImageContent, TextContent, Tool

from grase_tg.config import McpSettings
from grase_tg.mcp_bridge import (
    McpToolRegistry,
    _auth_headers,
    _filter_allowed,
    _host,
    _project_result,
    connect_mcp,
)


def _tool(name: str, description: str = "a tool") -> Tool:
    return Tool(name=name, description=description, inputSchema={"type": "object", "properties": {}})


def _build_server() -> FastMCP:
    server = FastMCP("test-server")

    @server.tool(description="Weather forecast for a city.")
    def get_forecast(city: str, days: int = 2) -> str:
        return f"{city}: sunny, +20C, {days} day(s)"

    @server.tool(description="Always fails.")
    def broken_tool() -> str:
        raise RuntimeError("upstream is down")

    return server


@asynccontextmanager
async def _open_registry() -> AsyncIterator[McpToolRegistry]:
    async with create_connected_server_and_client_session(_build_server()) as session:
        discovered = (await session.list_tools()).tools
        yield McpToolRegistry(session=session, tools=tuple(discovered))


# --- openai_schema ---


async def test_openai_schema_shape():
    async with _open_registry() as registry:
        schema = registry.openai_schema()
    assert [entry["type"] for entry in schema] == ["function", "function"]
    by_name = {entry["function"]["name"]: entry["function"] for entry in schema}
    forecast = by_name["get_forecast"]
    assert forecast["description"] == "Weather forecast for a city."
    properties = forecast["parameters"]["properties"]
    assert "city" in properties and "days" in properties
    assert "city" in forecast["parameters"].get("required", [])


async def test_empty_registry_schema_is_empty():
    # A registry with no tools (e.g. everything filtered out) yields an empty
    # schema — the loop then omits the tools field entirely (tools-off).
    async with create_connected_server_and_client_session(_build_server()) as session:
        assert McpToolRegistry(session=session, tools=()).openai_schema() == []


# --- execute(): the happy path and the never-raises contract ---


async def test_execute_roundtrip():
    async with _open_registry() as registry:
        result = await registry.execute("get_forecast", json.dumps({"city": "Moscow", "days": 1}))
    assert result == "Moscow: sunny, +20C, 1 day(s)"


async def test_server_side_validation_error_becomes_text():
    # Argument validation lives on the MCP server: a required field missing
    # comes back as an isError result whose text flows to the model —
    # execute() must not raise (docs/mcp-tools-spec.md §6.3).
    async with _open_registry() as registry:
        result = await registry.execute("get_forecast", "")
    assert "validation error" in result
    assert "city" in result


async def test_unknown_tool_returns_error_text():
    # A hallucinated (or allow-filtered) name must degrade into text the
    # model can react to — never an exception.
    async with _open_registry() as registry:
        result = await registry.execute("get_moon_phase", "{}")
    assert result == "Error: no such tool 'get_moon_phase'."


async def test_malformed_arguments_json_returns_error_text():
    async with _open_registry() as registry:
        result = await registry.execute("get_forecast", "{broken")
    assert result.startswith("Error: arguments are not valid JSON:")


async def test_non_object_arguments_return_error_text():
    async with _open_registry() as registry:
        result = await registry.execute("get_forecast", "[1, 2]")
    assert result == "Error: arguments must be a JSON object."


async def test_tool_failure_becomes_text_not_exception():
    # The server reports isError=True: the message travels to the model as a
    # normal tool result — execute() must not raise.
    async with _open_registry() as registry:
        result = await registry.execute("broken_tool", "{}")
    assert "upstream is down" in result


async def test_transport_error_becomes_text():
    # call_tool raising (e.g. the server died) must also degrade into text.
    class DyingSession:
        async def call_tool(self, name: str, arguments: dict):
            raise ConnectionError("server is gone")

    registry = McpToolRegistry(session=cast(Any, DyingSession()), tools=(_tool("get_forecast"),))
    result = await registry.execute("get_forecast", "{}")
    assert "server is gone" in result


# --- result projection ---


def test_projection_joins_text_blocks():
    result = CallToolResult(
        content=[TextContent(type="text", text="line one"), TextContent(type="text", text="line two")],
        isError=False,
    )
    assert _project_result(result) == "line one\nline two"


def test_projection_marks_binary_blocks():
    result = CallToolResult(
        content=[
            TextContent(type="text", text="a chart follows"),
            ImageContent(type="image", data="aGVsbG8=", mimeType="image/png"),
        ],
        isError=False,
    )
    assert _project_result(result) == "a chart follows\n(binary content omitted)"


def test_projection_falls_back_to_structured_content():
    result = CallToolResult(content=[], structuredContent={"temp_c": 21}, isError=False)
    assert _project_result(result) == '{"temp_c": 21}'


def test_projection_of_empty_result():
    result = CallToolResult(content=[], isError=False)
    assert _project_result(result) == "(no content)"


# --- allow-filtering ---


def test_filter_allowed_empty_allow_keeps_everything():
    tools = (_tool("a"), _tool("b"))
    assert _filter_allowed(tools, []) == tools


def test_filter_allowed_keeps_only_whitelisted_in_order():
    tools = (_tool("a"), _tool("b"), _tool("c"))
    assert _filter_allowed(tools, ["c", "a"]) == (tools[0], tools[2])


def test_filter_allowed_warns_on_unknown_names(caplog):
    import logging

    with caplog.at_level(logging.WARNING, logger="grase_tg.mcp_bridge"):
        filtered = _filter_allowed((_tool("a"),), ["a", "stale"])
    assert filtered == (_tool("a"),)
    assert "stale" in caplog.text


# --- connection: config errors vs network ---


async def test_malformed_url_fails_as_config_error():
    # The transport validates the url lazily and httpx.InvalidURL is not a
    # ValueError: the bridge must translate it, so a typo like a non-numeric
    # port fails the startup as a config error instead of degrading to
    # "server unreachable" (docs/mcp-tools-spec.md §6.1).
    settings = McpSettings(url="http://mcp.test:notaport/mcp")
    stack = AsyncExitStack()
    try:
        with pytest.raises(ValueError, match="tools.mcp.url"):
            await connect_mcp(settings, stack)
        # The failed transport must never be handed to the caller's stack:
        # unwinding it has to be silent, or the poisoned task group would
        # mask the clean SystemExit during a real startup.
        await stack.aclose()
    finally:
        await stack.aclose()


async def test_unreachable_server_raises_network_error_and_leaves_clean_stack():
    # Connection refused (nothing listens on port 1): the failure class is
    # "unreachable" (NOT ValueError -> the caller degrades instead of
    # exiting), and the caller's stack stays clean for the same reason as
    # above.
    settings = McpSettings(url="http://127.0.0.1:1/mcp", timeout_seconds=2)
    stack = AsyncExitStack()
    try:
        with pytest.raises(Exception) as exc_info:
            await connect_mcp(settings, stack)
        assert not isinstance(exc_info.value, ValueError)
        await stack.aclose()
    finally:
        await stack.aclose()


# --- auth headers ---


def test_auth_headers_bearer_from_env(env_isolated):
    env_isolated.setenv("MCP_TEST_TOKEN", "sekrit")
    settings = McpSettings(url="http://mcp.test/mcp", auth_env="MCP_TEST_TOKEN")
    assert _auth_headers(settings) == {"Authorization": "Bearer sekrit"}


def test_auth_headers_missing_env_var_is_value_error(env_isolated):
    settings = McpSettings(url="http://mcp.test/mcp", auth_env="MCP_ABSENT_TOKEN")
    with pytest.raises(ValueError, match="MCP_ABSENT_TOKEN"):
        _auth_headers(settings)


def test_auth_headers_without_auth_env_are_empty():
    settings = McpSettings(url="http://mcp.test/mcp")
    assert _auth_headers(settings) == {}


# --- misc ---


def test_host_for_logs_strips_credentials():
    assert _host("http://mcp.test:8080/mcp") == "mcp.test:8080"
    assert _host("http://user:pass@mcp.test/mcp") == "mcp.test"
    assert _host("not a url") == "not a url"
