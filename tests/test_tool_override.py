from __future__ import annotations

from typing import Any
from unittest.mock import patch

import pytest

from agentouto.agent import Agent
from agentouto.context import Context, ToolCall
from agentouto.provider import Provider
from agentouto.providers import LLMResponse, ProviderBackend
from agentouto.runtime import RunResult, async_run
from agentouto.tool import Tool, ToolResult


class MockBackend(ProviderBackend):
    def __init__(self, responses: list[LLMResponse]) -> None:
        self._responses = list(responses)
        self._call_count = 0

    async def call(
        self,
        context: Context,
        tools: list[dict[str, Any]],
        agent: Agent,
        provider: Provider,
    ) -> LLMResponse:
        assert self._call_count < len(self._responses)
        response = self._responses[self._call_count]
        self._call_count += 1
        return response


def _finish(message: str) -> LLMResponse:
    return LLMResponse(
        content=None,
        tool_calls=[
            ToolCall(id="fin_1", name="finish", arguments={"message": message})
        ],
    )


def _tool_call(tool_name: str, tool_id: str, **kwargs: str) -> LLMResponse:
    return LLMResponse(
        content=None,
        tool_calls=[ToolCall(id=tool_id, name=tool_name, arguments=dict(kwargs))],
    )


@pytest.fixture
def provider() -> Provider:
    return Provider(name="openai", kind="openai", api_key="sk-test")


@pytest.fixture
def agent_a() -> Agent:
    return Agent(
        name="agent_a", instructions="Agent A.", model="gpt-4o", provider="openai"
    )


class TestDisabledTools:
    @pytest.mark.asyncio
    async def test_disabled_tool_not_in_schema(
        self, agent_a: Agent, provider: Provider
    ) -> None:
        schemas_seen: list[list[dict[str, Any]]] = []

        class CapturingBackend(ProviderBackend):
            async def call(
                self,
                context: Context,
                tools: list[dict[str, Any]],
                agent: Agent,
                provider: Provider,
            ) -> LLMResponse:
                schemas_seen.append(tools)
                return _finish("done")

        mock = CapturingBackend()
        with patch("agentouto.router.get_backend", return_value=mock):
            await async_run(
                starting_agents=[agent_a],
                message="Hello",
                tools=[],
                providers=[provider],
                disabled_tools={"spawn_background_agent", "get_messages"},
            )
        tool_names = [t["name"] for t in schemas_seen[0]]
        assert "spawn_background_agent" not in tool_names
        assert "get_messages" not in tool_names
        assert "call_agent" in tool_names
        assert "finish" in tool_names

    @pytest.mark.asyncio
    async def test_disabled_tool_returns_error(
        self, agent_a: Agent, provider: Provider
    ) -> None:
        contexts_seen: list[Context] = []

        class CapturingBackend(ProviderBackend):
            def __init__(self) -> None:
                self._call_count = 0

            async def call(
                self,
                context: Context,
                tools: list[dict[str, Any]],
                agent: Agent,
                provider: Provider,
            ) -> LLMResponse:
                self._call_count += 1
                if self._call_count == 1:
                    return _tool_call(
                        "spawn_background_agent", "tc1", agent_name="x", message="y"
                    )
                contexts_seen.append(context)
                return _finish("ok")

        mock = CapturingBackend()
        with patch("agentouto.router.get_backend", return_value=mock):
            result = await async_run(
                starting_agents=[agent_a],
                message="Try spawn",
                tools=[],
                providers=[provider],
                disabled_tools={"spawn_background_agent"},
            )
        assert result.output == "ok"
        error_msg = contexts_seen[0].messages[-1].content or ""
        assert "disabled" in error_msg.lower()

    def test_disable_finish_raises(self) -> None:
        from agentouto.router import Router

        with pytest.raises(ValueError, match="finish"):
            Router(
                [Agent(name="a", instructions="A.", model="gpt-4o", provider="openai")],
                [],
                [Provider(name="openai", kind="openai", api_key="sk-test")],
                disabled_tools={"finish"},
            )


class TestFinishOverride:
    @pytest.mark.asyncio
    async def test_finish_override_runs_custom_and_exits(
        self, agent_a: Agent, provider: Provider
    ) -> None:
        finish_calls: list[str] = []

        @Tool
        def finish(message: str) -> str:
            """Custom finish with logging."""
            finish_calls.append(message)
            return f"PROCESSED: {message}"

        mock = MockBackend(
            [
                LLMResponse(
                    content=None,
                    tool_calls=[
                        ToolCall(
                            id="fin_1",
                            name="finish",
                            arguments={"message": "raw result"},
                        )
                    ],
                ),
            ]
        )
        with patch("agentouto.router.get_backend", return_value=mock):
            result = await async_run(
                starting_agents=[agent_a],
                message="Do work",
                tools=[finish],
                providers=[provider],
            )
        assert finish_calls == ["raw result"]
        assert result.output == "PROCESSED: raw result"

    @pytest.mark.asyncio
    async def test_finish_override_with_extra_params(
        self, agent_a: Agent, provider: Provider
    ) -> None:
        @Tool
        def finish(message: str, confidence: float = 1.0) -> str:
            """Finish with confidence score."""
            return f"[{confidence}] {message}"

        mock = MockBackend(
            [
                LLMResponse(
                    content=None,
                    tool_calls=[
                        ToolCall(
                            id="fin_1",
                            name="finish",
                            arguments={"message": "result", "confidence": 0.95},
                        )
                    ],
                ),
            ]
        )
        with patch("agentouto.router.get_backend", return_value=mock):
            result = await async_run(
                starting_agents=[agent_a],
                message="Do work",
                tools=[finish],
                providers=[provider],
            )
        assert result.output == "[0.95] result"


class TestOverrideBuiltinTool:
    @pytest.mark.asyncio
    async def test_override_send_message_executes_custom(
        self, agent_a: Agent, provider: Provider
    ) -> None:
        calls: list[dict[str, str]] = []

        @Tool
        def send_message(task_id: str, message: str) -> str:
            """Custom send."""
            calls.append({"task_id": task_id, "message": message})
            return "custom: sent"

        mock = MockBackend(
            [
                _tool_call("send_message", "tc1", task_id="bg_123", message="hello"),
                _finish("done"),
            ]
        )
        with patch("agentouto.router.get_backend", return_value=mock):
            result = await async_run(
                starting_agents=[agent_a],
                message="Send a message",
                tools=[send_message],
                providers=[provider],
            )
        assert result.output == "done"
        assert len(calls) == 1
        assert calls[0]["message"] == "hello"
