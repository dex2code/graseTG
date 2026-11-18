"""Bridge: one MCP server (streamable HTTP) -> OpenAI tool calling.

Architectural constraint (docs/mcp-tools-spec.md §1.1): exactly ONE MCP
server. The config schema has no list of servers and this module holds one
ClientSession — there is no aggregation, no tool-name routing and no
cross-server collision handling. Tool names are passed to the model as the
server publishes them (uniqueness inside one server is the server's own
contract).

execute() never raises: every failure — unknown tool, malformed arguments
JSON, transport error, an isError result — is projected into text for the
model's role="tool" message, so the model can answer honestly instead of
crashing the handler.

Privacy: tool names and argument COUNTS may be logged; argument VALUES may
be derived from the user's question text and must never reach the logs
(the same rule as question texts).
"""

import asyncio
import json
import logging
from collections.abc import Sequence
from contextlib import AsyncExitStack
from dataclasses import dataclass
from datetime import timedelta
from pathlib import Path
from urllib.parse import urlsplit

import httpx
from mcp import ClientSession
from mcp.client.streamable_http import streamable_http_client
from mcp.shared._httpx_utils import create_mcp_http_client
from mcp.types import CallToolResult, TextContent, Tool

from .config import McpSettings, default_config_dir, env_value
from .tools import ToolRegistry

logger = logging.getLogger(__name__)

# Markers sent to the model in place of content it cannot read. Explicit
# omission beats silence: the model must know that something was there.
_BINARY_MARKER = "(binary content omitted)"
_EMPTY_RESULT = "(no content)"


def _auth_headers(settings: McpSettings, config_dir: Path | None = None) -> dict[str, str]:
    """Bearer headers from the variable NAMED by auth_env.

    The value is resolved exactly like the Secrets fields (env_value): the OS
    environment wins, the config dir's .env file is the fallback. The error
    text and the docs both promise "add it to .env or the environment", so a
    token defined only in .env MUST work. A missing variable is a
    configuration error and must fail the startup loudly — connecting
    unauthenticated by accident is worse than not connecting at all.
    """
    if not settings.auth_env:
        return {}
    token = env_value(settings.auth_env, config_dir)
    if not token:
        env_path = (config_dir or default_config_dir()) / ".env"
        raise ValueError(
            f"tools.mcp.auth_env: environment variable {settings.auth_env!r} is not set "
            f"in the environment or {env_path} (add it to one of them)"
        )
    return {"Authorization": f"Bearer {token}"}


def _host(url: str) -> str:
    """The server host for logs — never the full url (it may embed credentials)."""
    parts = urlsplit(url)
    host = parts.hostname or ""
    if parts.port:
        host = f"{host}:{parts.port}"
    return host or url


def _filter_allowed(tools: Sequence[Tool], allow: list[str]) -> tuple[Tool, ...]:
    """Apply the optional whitelist (tools.mcp.allow).

    Every tool description goes into EVERY request, so filtering is a token
    budget and a focus knob for a small local model, not a security boundary.
    Unknown names in `allow` are a warning, not a failure: the server
    legitimately evolves, and a stale entry must not take the bot down.
    """
    if not allow:
        return tuple(tools)
    wanted = set(allow)
    known = {tool.name for tool in tools}
    unknown = sorted(wanted - known)
    if unknown:
        logger.warning(
            "tools.mcp.allow lists unknown tool(s) — skipped: %s", ", ".join(unknown)
        )
    return tuple(tool for tool in tools if tool.name in wanted)


def _project_result(result: CallToolResult) -> str:
    """CallToolResult -> one text string for the model's role="tool" message.

    - text blocks are joined with newlines;
    - with no text at all, structuredContent becomes compact JSON;
    - binary blocks (images, audio, embedded resources) become an explicit
      omission marker — the model must know something was there;
    - isError is deliberately NOT raised: the content (usually the error
      message) travels to the model as a normal tool result, and the model
      words the apology (docs/mcp-tools-spec.md §6.3).
    """
    texts = [block.text for block in result.content if isinstance(block, TextContent)]
    binary_count = sum(1 for block in result.content if not isinstance(block, TextContent))
    parts = [*texts, *([_BINARY_MARKER] * binary_count)]
    if not parts and result.structuredContent is not None:
        parts.append(json.dumps(result.structuredContent, ensure_ascii=False))
    if not parts:
        parts.append(_EMPTY_RESULT)
    return "\n".join(parts)


@dataclass(frozen=True)
class McpToolRegistry:
    """One MCP session's tools, dressed as OpenAI function definitions.

    Implements tools.ToolRegistry. Built by connect_mcp / _filter_allowed;
    tests may also construct it directly around an in-memory session.
    """

    session: ClientSession
    tools: tuple[Tool, ...]

    def openai_schema(self) -> list[dict]:
        """MCP's name/description/inputSchema map 1:1 onto the OpenAI function
        definition — inputSchema is already a JSON Schema object."""
        return [
            {
                "type": "function",
                "function": {
                    "name": tool.name,
                    "description": tool.description or "",
                    "parameters": tool.inputSchema,
                },
            }
            for tool in self.tools
        ]

    async def execute(self, name: str, arguments_json: str) -> str:
        """Run one tool call against the MCP server. Never raises (see module
        docstring): every failure mode degrades into error text for the model."""
        if name not in {tool.name for tool in self.tools}:
            # A filtered-out (allow) or hallucinated tool name: say so, and
            # the model usually corrects itself or answers without the tool.
            logger.info("Model called tool '%s' — no such tool in the registry", name)
            return f"Error: no such tool '{name}'."
        try:
            arguments = json.loads(arguments_json or "{}")
        except json.JSONDecodeError as error:
            logger.info("Tool '%s' called with malformed arguments JSON", name)
            return f"Error: arguments are not valid JSON: {error}"
        if not isinstance(arguments, dict):
            logger.info("Tool '%s' called with non-object arguments", name)
            return "Error: arguments must be a JSON object."
        # Privacy: log the tool name and the argument count, never the values.
        logger.info("Tool '%s' called with %d argument(s)", name, len(arguments))
        try:
            result = await self.session.call_tool(name, arguments)
        except Exception as error:
            # Deliberately NOT BaseException: a CancelledError (bot shutdown
            # while the call is in flight) must propagate and stop the loop,
            # not be buried into a tool result.
            logger.warning("Tool '%s' call failed: %s", name, error)
            return f"Error: tool call failed: {error}"
        return _project_result(result)


def _first_leaf(error: BaseException) -> BaseException:
    """Leftmost leaf of a (possibly nested) ExceptionGroup — the transport's
    task group wraps the single real failure in a group of one."""
    while isinstance(error, BaseExceptionGroup):
        error = error.exceptions[0]
    return error


async def connect_mcp(
    settings: McpSettings,
    stack: AsyncExitStack,
    config_dir: Path | None = None,
) -> McpToolRegistry:
    """Connect to the single configured MCP server and discover its tools.

    `config_dir` locates the .env fallback for the auth_env token (None = the
    bot's default config dir).

    Ownership: transport, HTTP client and session are opened on a LOCAL stack
    and handed to the caller's `stack` only after a fully successful connect.
    On failure they are unwound right here and exactly ONE classified error
    escapes — a failed transport has a poisoned anyio task group whose
    teardown re-raises the same error, and that must never resurface on the
    caller's shutdown path (it would mask either the config ValueError or
    the "unreachable" degrade of docs/mcp-tools-spec.md §6.1).
    """
    headers = _auth_headers(settings, config_dir)
    local = AsyncExitStack()
    try:
        # Eager url validation: httpx fails on a bad url only when the request
        # is BUILT, deep inside the transport's task group (where it surfaces
        # as a bare CancelledError plus a poisoned teardown). Validating here
        # fails the startup synchronously, before any task group exists.
        try:
            httpx.URL(settings.url)
        except httpx.InvalidURL as error:
            raise ValueError(f"tools.mcp.url: {error}") from error

        # The modern transport takes a pre-configured httpx client: that is
        # where the auth headers go. We own the client — the local stack
        # closes it (httpx close is idempotent; the transport exits first,
        # LIFO, and may still use the client to terminate the session).
        # create_mcp_http_client lives in a private module by name, but it is
        # the SDK's own builder (recommended timeouts, origin-safe redirects)
        # and the only import location pyright accepts for it.
        client = await local.enter_async_context(create_mcp_http_client(headers=headers or None))
        read, write, _ = await local.enter_async_context(
            streamable_http_client(settings.url, http_client=client)
        )
        session = await local.enter_async_context(
            ClientSession(read, write, read_timeout_seconds=timedelta(seconds=settings.timeout_seconds))
        )
        await session.initialize()
        tools = _filter_allowed((await session.list_tools()).tools, settings.allow)
    except BaseException as connect_error:
        # The transport's task group dies together with the failed request:
        # initialize() surfaces only a symptom (a bare CancelledError), the
        # REAL error comes out of the poisoned teardown. Unwind locally, dig
        # the real error out, and raise one clean, correctly classified
        # failure instead of the group.
        real_error: BaseException = connect_error
        try:
            await local.aclose()
        except BaseException as teardown_error:
            real_error = _first_leaf(teardown_error)
        if isinstance(real_error, httpx.InvalidURL):
            raise ValueError(f"tools.mcp.url: {real_error}") from real_error
        if isinstance(real_error, asyncio.CancelledError):
            raise  # our own task was cancelled (bot shutdown) — keep it a cancellation
        raise real_error from connect_error
    else:
        # Fully connected: transfer the live contexts to the caller's stack,
        # where they live until bot shutdown.
        stack.push_async_callback(local.pop_all().aclose)
    names = ", ".join(tool.name for tool in tools)
    logger.info(
        "MCP: %d tool(s) on %s%s", len(tools), _host(settings.url), f" ({names})" if names else ""
    )
    return McpToolRegistry(session=session, tools=tools)
