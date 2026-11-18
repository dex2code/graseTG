"""Tool registry interface for the LLM tool loop.

The loop in llm.py knows tools ONLY through this protocol: a flat OpenAI
"tools" schema plus an execute() that never raises. A failing tool returns
its error as text, and the model formulates the honest apology — a tool
failure must never crash the handler or lose the ack message.

The only v1 implementation is the MCP bridge (mcp_bridge.py); tests provide
stubs behind the same interface.
"""

from typing import Protocol


class ToolRegistry(Protocol):
    """A set of tools the model may call, bridged to some backend."""

    def openai_schema(self) -> list[dict]:
        # Definitions for the chat/completions "tools" parameter:
        #   [{"type": "function",
        #     "function": {"name": ..., "description": ..., "parameters": {...}}}]
        # An empty list means "no tools available" — the loop must then omit
        # the field from the request entirely (an empty tools:[] confuses
        # some OpenAI-compatible backends).
        ...

    async def execute(self, name: str, arguments_json: str) -> str:
        # Run one tool call; `arguments_json` is the raw JSON string produced
        # by the model.
        #
        # NEVER raises: unknown tool, malformed arguments, transport error or
        # an isError result from the backend all come back as error text, so
        # the model can react (retry, apologize) instead of the bot crashing.
        ...
