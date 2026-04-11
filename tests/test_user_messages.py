from __future__ import annotations

import re
from typing import Any
from unittest.mock import MagicMock, patch

import pytest

from agentouto.agent import Agent
from agentouto.context import Context, ToolCall
from agentouto.loop_manager import RegisteredAgentLoop
from agentouto.message import Message
from agentouto.provider import Provider
from agentouto.providers import LLMResponse, ProviderBackend
from agentouto.runtime import async_run
from agentouto.tool import Tool


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


def _tool_call(tool_name: str, tool_id: str, **kwargs: Any) -> LLMResponse:
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


class TestRegisteredAgentLoopCallback:
    @pytest.mark.asyncio
    async def test_on_message_fires_on_inject(self) -> None:
        callback = MagicMock()
        agent = Agent(name="test", instructions="T.", model="gpt-4o", provider="openai")
        loop = RegisteredAgentLoop(agent=agent, task_id="loop_1", on_message=callback)
        msg = Message(
            type="forward", sender="agent_a", receiver="test", content="hello"
        )
        await loop.inject_message(msg)
        callback.assert_called_once_with(msg)

    @pytest.mark.asyncio
    async def test_on_message_none_no_error(self) -> None:
        agent = Agent(name="test", instructions="T.", model="gpt-4o", provider="openai")
        loop = RegisteredAgentLoop(agent=agent, task_id="loop_1")
        msg = Message(
            type="forward", sender="agent_a", receiver="test", content="hello"
        )
        await loop.inject_message(msg)
        assert len(loop.messages) == 1

    def test_caller_loop_id_stored(self) -> None:
        agent = Agent(name="test", instructions="T.", model="gpt-4o", provider="openai")
        loop = RegisteredAgentLoop(
            agent=agent, task_id="loop_1", caller_loop_id="caller_xyz"
        )
        assert loop.caller_loop_id == "caller_xyz"

    def test_caller_loop_id_default_none(self) -> None:
        agent = Agent(name="test", instructions="T.", model="gpt-4o", provider="openai")
        loop = RegisteredAgentLoop(agent=agent, task_id="loop_1")
        assert loop.caller_loop_id is None


class TestUserReceivesIntermediateMessages:
    @pytest.mark.asyncio
    async def test_on_message_receives_send_message(
        self, agent_a: Agent, provider: Provider
    ) -> None:
        received: list[Message] = []

        class SmartMock(ProviderBackend):
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
                    match = re.search(r"task_id is '([^']+)'", context.system_prompt)
                    if match:
                        return _tool_call(
                            "send_message",
                            "tc1",
                            task_id=match.group(1),
                            message="progress: 50%",
                        )
                    return _finish("no loop id found")
                return _finish("done")

        mock = SmartMock()
        with patch("agentouto.router.get_backend", return_value=mock):
            result = await async_run(
                starting_agents=[agent_a],
                message="Do work",
                tools=[],
                providers=[provider],
                on_message=lambda msg, send: received.append(msg),
            )
        assert result.output == "done"
        assert len(received) == 1
        assert received[0].content == "progress: 50%"
        assert received[0].sender == "agent_a"

    @pytest.mark.asyncio
    async def test_on_message_none_backward_compat(
        self, agent_a: Agent, provider: Provider
    ) -> None:
        mock = MockBackend([_finish("result")])
        with patch("agentouto.router.get_backend", return_value=mock):
            result = await async_run(
                starting_agents=[agent_a],
                message="Hello",
                tools=[],
                providers=[provider],
            )
        assert result.output == "result"

    @pytest.mark.asyncio
    async def test_system_prompt_includes_caller_loop_id(
        self, agent_a: Agent, provider: Provider
    ) -> None:
        prompts_seen: list[str] = []

        class CapturingBackend(ProviderBackend):
            async def call(
                self,
                context: Context,
                tools: list[dict[str, Any]],
                agent: Agent,
                provider: Provider,
            ) -> LLMResponse:
                prompts_seen.append(context.system_prompt)
                return _finish("done")

        with patch("agentouto.router.get_backend", return_value=CapturingBackend()):
            await async_run(
                starting_agents=[agent_a],
                message="Hello",
                tools=[],
                providers=[provider],
                on_message=lambda msg, send: None,
            )
        assert any("task_id" in p for p in prompts_seen)

    @pytest.mark.asyncio
    async def test_intermediate_messages_in_result_messages(
        self, agent_a: Agent, provider: Provider
    ) -> None:
        class SmartMock(ProviderBackend):
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
                    match = re.search(r"task_id is '([^']+)'", context.system_prompt)
                    if match:
                        return _tool_call(
                            "send_message",
                            "tc1",
                            task_id=match.group(1),
                            message="update",
                        )
                    return _finish("no id")
                return _finish("done")

        with patch("agentouto.router.get_backend", return_value=SmartMock()):
            result = await async_run(
                starting_agents=[agent_a],
                message="Work",
                tools=[],
                providers=[provider],
                on_message=lambda msg, send: None,
            )
        intermediate = [
            m
            for m in result.messages
            if m.sender == "agent_a"
            and m.receiver == "user"
            and m.type == "forward"
            and m.content == "update"
        ]
        assert len(intermediate) == 1


class TestUserToAgentMessages:
    @pytest.mark.asyncio
    async def test_user_sends_message_to_agent(self, provider: Provider) -> None:
        """User calls send() in on_message callback → agent receives message next iteration."""
        agent_responses: list[str] = []

        class SmartMock(ProviderBackend):
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
                    agent_responses.append(
                        "first: " + (context.messages[-1].content or "")
                    )
                    match = re.search(r"task_id is '([^']+)'", context.system_prompt)
                    if match:
                        return _tool_call(
                            "send_message",
                            "tc1",
                            task_id=match.group(1),
                            message="request input",
                        )
                    return _finish("done")
                agent_responses.append(
                    "second: " + (context.messages[-1].content or "")
                )
                return _finish("received user input")

        agent_a = Agent(
            name="agent_a", instructions="Agent A.", model="gpt-4o", provider="openai"
        )
        send_fn_ref: list[Callable[[str], None]] = []

        def on_msg(msg, send):
            send_fn_ref.append(send)
            if msg.content == "request input":
                send("user reply here")

        mock = SmartMock()
        with patch("agentouto.router.get_backend", return_value=mock):
            result = await async_run(
                starting_agents=[agent_a],
                message="Hello",
                tools=[],
                providers=[provider],
                on_message=on_msg,
            )
        assert result.output == "received user input"
        assert "second: user reply here" in agent_responses
        assert len(send_fn_ref) == 1

    @pytest.mark.asyncio
    async def test_user_sends_multiple_messages(self, provider: Provider) -> None:
        """User can send multiple messages in sequence."""
        agent_a = Agent(
            name="agent_a", instructions="Agent A.", model="gpt-4o", provider="openai"
        )
        contexts_seen: list[Context] = []

        class SmartMock(ProviderBackend):
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
                contexts_seen.append(context)
                if self._call_count == 1:
                    match = re.search(r"task_id is '([^']+)'", context.system_prompt)
                    if match:
                        return _tool_call(
                            "send_message",
                            "tc1",
                            task_id=match.group(1),
                            message="ask for details",
                        )
                    return _finish("done")
                return _finish("all done")

        def on_msg(msg, send):
            if msg.content == "ask for details":
                send("detail 1")
                send("detail 2")

        mock = SmartMock()
        with patch("agentouto.router.get_backend", return_value=mock):
            result = await async_run(
                starting_agents=[agent_a],
                message="Hello",
                tools=[],
                providers=[provider],
                on_message=on_msg,
            )
        assert result.output == "all done"
        last_user_messages = [
            m.content for m in contexts_seen[-1].messages if m.role == "user"
        ]
        assert "detail 1" in last_user_messages
        assert "detail 2" in last_user_messages
