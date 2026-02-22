from __future__ import annotations

import re
from abc import ABC, abstractmethod
from collections.abc import AsyncIterator
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from agentouto.agent import Agent
    from agentouto.context import Context, ToolCall
    from agentouto.provider import Provider
    from agentouto.tool import Tool

_REASONING_TAG_RE = re.compile(
    r"<(think|thinking|reason|reasoning)>.*?(?:</\1>|$)",
    re.DOTALL,
)


def _content_outside_reasoning(content: str) -> str:
    """Return *content* with all reasoning-tag blocks removed.

    Use this **before** parsing text for tool-call patterns so that anything
    inside ``<think>``, ``<thinking>``, ``<reason>``, or ``<reasoning>``
    blocks is excluded from detection.  The original content stored in the
    context is never modified.
    """
    return _REASONING_TAG_RE.sub("", content).strip()


class LLMResponse:
    __slots__ = ("content", "tool_calls")

    def __init__(
        self,
        content: str | None = None,
        tool_calls: list[ToolCall] | None = None,
    ) -> None:
        self.content = content
        self.tool_calls = tool_calls or []

    @property
    def content_without_reasoning(self) -> str | None:
        """Content with reasoning tag blocks stripped. ``None`` when empty."""
        if self.content is None:
            return None
        result = _content_outside_reasoning(self.content)
        return result or None


class ProviderBackend(ABC):
    @abstractmethod
    async def call(
        self,
        context: Context,
        tools: list[dict[str, Any]],
        agent: Agent,
        provider: Provider,
    ) -> LLMResponse: ...

    async def stream(
        self,
        context: Context,
        tools: list[dict[str, Any]],
        agent: Agent,
        provider: Provider,
    ) -> AsyncIterator[str | LLMResponse]:
        """Stream LLM response. Yields str for text chunks, then a final LLMResponse."""
        response = await self.call(context, tools, agent, provider)
        if response.content:
            yield response.content
        yield response


def get_backend(kind: str) -> ProviderBackend:
    if kind == "openai":
        from agentouto.providers.openai import OpenAIBackend
        return OpenAIBackend()
    elif kind == "anthropic":
        from agentouto.providers.anthropic import AnthropicBackend
        return AnthropicBackend()
    elif kind == "google":
        from agentouto.providers.google import GoogleBackend
        return GoogleBackend()
    else:
        raise ValueError(f"Unknown provider kind: {kind}")
