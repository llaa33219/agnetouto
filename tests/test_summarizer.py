from __future__ import annotations

from typing import Any
from unittest.mock import patch

import pytest

from agentouto.agent import Agent
from agentouto.context import Context, ContextMessage, ToolCall
from agentouto.provider import Provider
from agentouto.providers import LLMResponse, ProviderBackend
from agentouto.runtime import async_run
from agentouto.summarizer import (
    _estimate_message_tokens,
    build_summary_prompt,
    estimate_context_tokens,
    find_summarization_boundary,
    parse_summary_response,
)
from agentouto.tool import Tool


# --- Token Estimation ---


class TestEstimateMessageTokens:
    def test_text_only(self) -> None:
        msg = ContextMessage(role="user", content="a" * 100)
        assert _estimate_message_tokens(msg) == 25

    def test_empty_content(self) -> None:
        msg = ContextMessage(role="user", content="")
        assert _estimate_message_tokens(msg) == 1

    def test_none_content(self) -> None:
        msg = ContextMessage(role="user", content=None)
        assert _estimate_message_tokens(msg) == 1

    def test_tool_call(self) -> None:
        tc = ToolCall(id="tc1", name="search", arguments={"query": "AI"})
        msg = ContextMessage(role="assistant", tool_calls=[tc])
        assert _estimate_message_tokens(msg) >= 1

    def test_tool_result(self) -> None:
        msg = ContextMessage(
            role="tool", content="Result data", tool_call_id="tc1", tool_name="search"
        )
        tokens = _estimate_message_tokens(msg)
        assert tokens >= 4


class TestEstimateContextTokens:
    def test_basic(self) -> None:
        ctx = Context("System prompt for the agent")
        ctx.add_user("Hello")
        ctx.add_assistant_text("Hi there")
        tokens = estimate_context_tokens(ctx)
        assert tokens > 0

    def test_empty_context(self) -> None:
        ctx = Context("sys")
        assert estimate_context_tokens(ctx) >= 0


# --- Boundary Finding ---


class TestFindSummarizationBoundary:
    def test_too_few_messages(self) -> None:
        messages = [ContextMessage(role="user", content="hi")]
        assert find_summarization_boundary(messages, 100) is None

    def test_two_messages(self) -> None:
        messages = [
            ContextMessage(role="user", content="hi"),
            ContextMessage(role="assistant", content="hello"),
        ]
        assert find_summarization_boundary(messages, 100) is None

    def test_finds_boundary(self) -> None:
        messages = [
            ContextMessage(role="user", content="a" * 400),
            ContextMessage(role="assistant", content="b" * 400),
            ContextMessage(role="user", content="c" * 400),
            ContextMessage(role="assistant", content="d" * 400),
            ContextMessage(role="user", content="e" * 400),
            ContextMessage(role="assistant", content="f" * 400),
        ]
        split = find_summarization_boundary(messages, 200)
        assert split is not None
        assert 0 < split < len(messages)

    def test_never_splits_at_tool_message(self) -> None:
        messages = [
            ContextMessage(role="user", content="a" * 200),
            ContextMessage(
                role="assistant",
                content=None,
                tool_calls=[ToolCall(id="tc1", name="search", arguments={"q": "x"})],
            ),
            ContextMessage(
                role="tool",
                content="result " * 50,
                tool_call_id="tc1",
                tool_name="search",
            ),
            ContextMessage(role="user", content="c" * 200),
            ContextMessage(role="assistant", content="d" * 200),
        ]
        split = find_summarization_boundary(messages, 200)
        if split is not None:
            assert messages[split].role != "tool"

    def test_snaps_backward_past_tool_messages(self) -> None:
        messages = [
            ContextMessage(role="user", content="start"),
            ContextMessage(
                role="assistant",
                content=None,
                tool_calls=[ToolCall(id="tc1", name="s", arguments={})],
            ),
            ContextMessage(
                role="tool", content="r1", tool_call_id="tc1", tool_name="s"
            ),
            ContextMessage(
                role="tool", content="r2", tool_call_id="tc2", tool_name="s"
            ),
            ContextMessage(role="assistant", content="final" * 100),
        ]
        split = find_summarization_boundary(messages, 50)
        if split is not None:
            assert messages[split].role != "tool"

    def test_returns_none_if_all_tool_messages(self) -> None:
        messages = [
            ContextMessage(role="user", content="x"),
            ContextMessage(
                role="tool", content="r1", tool_call_id="tc1", tool_name="s"
            ),
            ContextMessage(
                role="tool", content="r2", tool_call_id="tc2", tool_name="s"
            ),
        ]
        split = find_summarization_boundary(messages, 100)
        if split is not None:
            assert messages[split].role != "tool"

    def test_keeps_minimum_two_messages(self) -> None:
        messages = [
            ContextMessage(role="user", content="a" * 1000),
            ContextMessage(role="assistant", content="b" * 10),
            ContextMessage(role="user", content="c" * 10),
        ]
        split = find_summarization_boundary(messages, 50)
        if split is not None:
            assert len(messages) - split >= 2


# --- Summary Prompt Building ---


class TestBuildSummaryPrompt:
    def test_user_message(self) -> None:
        messages = [ContextMessage(role="user", content="Hello")]
        result = build_summary_prompt(messages)
        assert "User: Hello" in result

    def test_assistant_text(self) -> None:
        messages = [ContextMessage(role="assistant", content="Hi there")]
        result = build_summary_prompt(messages)
        assert "Assistant: Hi there" in result

    def test_assistant_tool_calls(self) -> None:
        tc = ToolCall(id="tc1", name="search", arguments={"query": "AI"})
        messages = [
            ContextMessage(role="assistant", tool_calls=[tc], content="Searching...")
        ]
        result = build_summary_prompt(messages)
        assert "Called tools:" in result
        assert "search" in result
        assert "Searching..." in result

    def test_tool_result(self) -> None:
        messages = [
            ContextMessage(
                role="tool",
                content="Found 10 results",
                tool_call_id="tc1",
                tool_name="search",
            ),
        ]
        result = build_summary_prompt(messages)
        assert "Tool result (search):" in result
        assert "Found 10 results" in result

    def test_full_conversation(self) -> None:
        tc = ToolCall(id="tc1", name="search", arguments={"query": "AI"})
        messages = [
            ContextMessage(role="user", content="Search for AI"),
            ContextMessage(role="assistant", tool_calls=[tc]),
            ContextMessage(
                role="tool",
                content="Results: ...",
                tool_call_id="tc1",
                tool_name="search",
            ),
            ContextMessage(role="assistant", content="Based on the search..."),
        ]
        result = build_summary_prompt(messages)
        assert "User:" in result
        assert "Assistant:" in result
        assert "Tool result" in result


# --- Summary Response Parsing ---


class TestParseSummaryResponse:
    def test_full_format(self) -> None:
        content = """<summary>
The user asked about AI trends. We searched and found GPT-5 info.
</summary>

<next_steps>
1. Research Claude 4 updates
2. Write comparison report
</next_steps>"""
        result = parse_summary_response(content)
        assert "AI trends" in result.summary
        assert "GPT-5" in result.summary
        assert "Claude 4" in (result.next_steps or "")
        assert "comparison report" in (result.next_steps or "")

    def test_next_steps_none(self) -> None:
        content = """<summary>
Task completed successfully.
</summary>

<next_steps>
None
</next_steps>"""
        result = parse_summary_response(content)
        assert "Task completed" in result.summary
        assert result.next_steps is None

    def test_fallback_no_tags(self) -> None:
        content = "Just a plain summary without any tags."
        result = parse_summary_response(content)
        assert result.summary == content.strip()
        assert result.next_steps is None

    def test_fallback_partial_summary_tag(self) -> None:
        content = """<summary>
Only summary tag exists.
</summary>"""
        result = parse_summary_response(content)
        assert "Only summary tag" in result.summary
        assert result.next_steps is None

    def test_case_insensitive_tags(self) -> None:
        content = """<SUMMARY>
Mixed case tags.
</Summary>

<Next_Steps>
Investigate further.
</NEXT_STEPS>"""
        result = parse_summary_response(content)
        assert "Mixed case" in result.summary
        assert "Investigate" in (result.next_steps or "")

    def test_empty_next_steps(self) -> None:
        content = """<summary>
Done.
</summary>

<next_steps>

</next_steps>"""
        result = parse_summary_response(content)
        assert result.summary == "Done."
        assert result.next_steps is None

    def test_na_next_steps(self) -> None:
        content = """<summary>
Summary here.
</summary>

<next_steps>
N/A
</next_steps>"""
        result = parse_summary_response(content)
        assert result.next_steps is None


# --- Context.replace_with_summary ---


class TestReplaceWithSummary:
    def test_basic_replacement(self) -> None:
        ctx = Context("sys")
        ctx.add_user("first")
        ctx.add_assistant_text("response1")
        ctx.add_user("second")
        ctx.add_assistant_text("response2")

        ctx.replace_with_summary("Summary of first exchange", keep_from=2)

        msgs = ctx.messages
        assert len(msgs) == 2
        assert msgs[0].role == "user"
        assert "[Previous conversation summary]" in (msgs[0].content or "")
        assert "Summary of first exchange" in (msgs[0].content or "")
        assert "second" in (msgs[0].content or "")
        assert msgs[1].role == "assistant"
        assert msgs[1].content == "response2"

    def test_merges_when_kept_starts_with_user(self) -> None:
        ctx = Context("sys")
        ctx.add_user("first")
        ctx.add_assistant_text("response1")
        ctx.add_user("second question")

        ctx.replace_with_summary("Summary", keep_from=2)

        msgs = ctx.messages
        assert len(msgs) == 1
        assert msgs[0].role == "user"
        assert "[Previous conversation summary]" in (msgs[0].content or "")
        assert "second question" in (msgs[0].content or "")

    def test_separate_when_kept_starts_with_assistant(self) -> None:
        ctx = Context("sys")
        ctx.add_user("first")
        ctx.add_assistant_text("response1")
        tc = ToolCall(id="tc1", name="search", arguments={"q": "test"})
        ctx.add_assistant_tool_calls([tc])
        ctx.add_tool_result("tc1", "search", "result")

        ctx.replace_with_summary("Summary", keep_from=2)

        msgs = ctx.messages
        assert msgs[0].role == "user"
        assert "[Previous conversation summary]" in (msgs[0].content or "")
        assert msgs[1].role == "assistant"
        assert msgs[1].tool_calls is not None


# --- Runtime Integration ---


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
        assert self._call_count < len(self._responses), (
            f"MockBackend exhausted: {self._call_count} calls made, "
            f"only {len(self._responses)} responses configured"
        )
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
def search_tool() -> Tool:
    @Tool
    def search(query: str) -> str:
        """Search the web."""
        return f"Results for: {query}" + " data" * 100

    return search


class TestRuntimeSummarization:
    @pytest.mark.asyncio
    async def test_no_summarization_without_context_window(
        self,
        provider: Provider,
        search_tool: Tool,
    ) -> None:
        agent = Agent(
            name="agent_a",
            instructions="Agent A.",
            model="gpt-4o",
            provider="openai",
        )
        mock = MockBackend([_finish("done")])
        with patch("agentouto.router.get_backend", return_value=mock):
            result = await async_run(
                starting_agents=[agent],
                message="Hello",
                tools=[search_tool],
                providers=[provider],
            )
        assert result.output == "done"
        assert mock._call_count == 1

    @pytest.mark.asyncio
    async def test_summarization_triggers_on_large_context(
        self,
        provider: Provider,
        search_tool: Tool,
    ) -> None:
        agent = Agent(
            name="agent_a",
            instructions="A.",
            model="gpt-4o",
            provider="openai",
            context_window=200,
        )

        contexts_seen: list[Context] = []
        regular_call_count = 0

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
                nonlocal regular_call_count
                self._call_count += 1
                if not tools:
                    return LLMResponse(content="concise summary of prior conversation")
                regular_call_count += 1
                if regular_call_count == 1:
                    return _tool_call("search", "tc1", query="first")
                if regular_call_count == 2:
                    contexts_seen.append(context)
                    return _finish("final result")
                return _finish("done")

        mock = CapturingBackend()
        with patch("agentouto.router.get_backend", return_value=mock):
            result = await async_run(
                starting_agents=[agent],
                message="Do something long " * 20,
                tools=[search_tool],
                providers=[provider],
            )
        assert result.output == "final result"
        if contexts_seen:
            ctx = contexts_seen[0]
            has_summary = any(
                "[Previous conversation summary]" in (m.content or "")
                for m in ctx.messages
            )
            assert has_summary

    @pytest.mark.asyncio
    async def test_summarization_failure_does_not_crash(
        self,
        provider: Provider,
        search_tool: Tool,
    ) -> None:
        agent = Agent(
            name="agent_a",
            instructions="A.",
            model="gpt-4o",
            provider="openai",
            context_window=50,
        )

        class FailingSummaryBackend(ProviderBackend):
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
                if not tools:
                    raise RuntimeError("LLM unavailable for summarization")
                if self._call_count == 1:
                    return _tool_call("search", "tc1", query="test")
                return _finish("done despite failure")

        mock = FailingSummaryBackend()
        with patch("agentouto.router.get_backend", return_value=mock):
            result = await async_run(
                starting_agents=[agent],
                message="Long message " * 50,
                tools=[search_tool],
                providers=[provider],
            )
        assert result.output == "done despite failure"

    @pytest.mark.asyncio
    async def test_context_window_none_skips_summarization(
        self,
        provider: Provider,
        search_tool: Tool,
    ) -> None:
        agent = Agent(
            name="agent_a",
            instructions="A.",
            model="gpt-4o",
            provider="openai",
            context_window=None,
        )
        mock = MockBackend([_finish("done")])
        with patch("agentouto.router.get_backend", return_value=mock):
            result = await async_run(
                starting_agents=[agent],
                message="Hello " * 1000,
                tools=[search_tool],
                providers=[provider],
            )
        assert result.output == "done"
        assert mock._call_count == 1


class TestOnSummarizeCallback:
    @pytest.mark.asyncio
    async def test_on_summarize_receives_info(self, provider: Provider, search_tool: Tool) -> None:
        agent = Agent(
            name="agent_a",
            instructions="A.",
            model="gpt-4o",
            provider="openai",
            context_window=200,
        )

        captured_info = None

        def on_summarize(info):
            nonlocal captured_info
            captured_info = info
            return None

        call_count = 0

        class CapturingBackend(ProviderBackend):
            async def call(self, context, tools, agent, provider):
                nonlocal call_count
                call_count += 1
                if not tools:
                    return LLMResponse(content="<summary>Short</summary>\n<next_steps>1. Done</next_steps>")
                if call_count == 1:
                    return _tool_call("search", "tc1", query="x")
                return _finish("final")

        mock = CapturingBackend()
        with patch("agentouto.router.get_backend", return_value=mock):
            result = await async_run(
                starting_agents=[agent],
                message="Do something long " * 20,
                tools=[search_tool],
                providers=[provider],
                on_summarize=on_summarize,
            )
        assert result.output == "final"
        assert captured_info is not None
        assert captured_info.agent_name == "agent_a"
        assert captured_info.summary == "Short"
        assert "Done" in (captured_info.next_steps or "")

    @pytest.mark.asyncio
    async def test_on_summarize_can_override_summary(self, provider: Provider, search_tool: Tool) -> None:
        agent = Agent(
            name="agent_a",
            instructions="A.",
            model="gpt-4o",
            provider="openai",
            context_window=200,
        )

        def on_summarize(info):
            return "OVERRIDDEN SUMMARY"

        call_count = 0
        last_context = None

        class OverridingBackend(ProviderBackend):
            async def call(self, context, tools, agent, provider):
                nonlocal call_count, last_context
                call_count += 1
                if not tools:
                    return LLMResponse(content="<summary>Original</summary>\n<next_steps>None</next_steps>")
                last_context = context
                if call_count == 1:
                    return _tool_call("search", "tc1", query="x")
                return _finish("final")

        mock = OverridingBackend()
        with patch("agentouto.router.get_backend", return_value=mock):
            result = await async_run(
                starting_agents=[agent],
                message="Long message " * 20,
                tools=[search_tool],
                providers=[provider],
                on_summarize=on_summarize,
            )
        assert result.output == "final"
        assert last_context is not None
        assert any("OVERRIDDEN SUMMARY" in (m.content or "") for m in last_context.messages)

    @pytest.mark.asyncio
    async def test_on_summarize_error_is_ignored(self, provider: Provider, search_tool: Tool) -> None:
        agent = Agent(
            name="agent_a",
            instructions="A.",
            model="gpt-4o",
            provider="openai",
            context_window=200,
        )

        def on_summarize(info):
            raise RuntimeError("callback error")

        call_count = 0

        class ErrorBackend(ProviderBackend):
            async def call(self, context, tools, agent, provider):
                nonlocal call_count
                call_count += 1
                if not tools:
                    return LLMResponse(content="<summary>Short</summary>\n<next_steps>None</next_steps>")
                if call_count == 1:
                    return _tool_call("search", "tc1", query="x")
                return _finish("final")

        mock = ErrorBackend()
        with patch("agentouto.router.get_backend", return_value=mock):
            result = await async_run(
                starting_agents=[agent],
                message="Long message " * 20,
                tools=[search_tool],
                providers=[provider],
                on_summarize=on_summarize,
            )
        assert result.output == "final"
