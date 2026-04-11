from __future__ import annotations

CALL_AGENT = "call_agent"
FINISH = "finish"

BUILTIN_TOOL_NAMES: frozenset[str] = frozenset(
    {
        CALL_AGENT,
        "spawn_background_agent",
        "send_message",
        "get_messages",
        FINISH,
    }
)
