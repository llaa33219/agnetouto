from __future__ import annotations

from collections.abc import AsyncIterator
from dataclasses import dataclass, field
from typing import Any, Literal

from agentouto.agent import Agent
from agentouto.context import Attachment
from agentouto.message import Message
from agentouto.provider import Provider
from agentouto.tool import Tool


@dataclass
class StreamEvent:
    type: Literal[
        "token",
        "tool_call",
        "tool_result",
        "agent_call",
        "agent_return",
        "finish",
        "error",
    ]
    agent_name: str
    call_id: str
    parent_call_id: str | None
    data: dict[str, Any] = field(default_factory=dict)


async def async_run_stream(
    message: str,
    starting_agents: list[Agent] | None = None,
    tools: list[Tool] | None = None,
    providers: list[Provider] | None = None,
    *,
    attachments: list[Attachment] | None = None,
    history: list[Message] | None = None,
    extra_instructions: str | None = None,
    extra_instructions_scope: Literal["entry", "all"] = "entry",
    run_agents: list[Agent] | None = None,
) -> AsyncIterator[StreamEvent]:
    from agentouto.router import Router
    from agentouto.runtime import Runtime

    if starting_agents is None or len(starting_agents) == 0:
        raise ValueError(
            "starting_agents must be provided (list of agents to start in parallel)"
        )

    run_agents_list = run_agents if run_agents is not None else starting_agents

    if run_agents is not None:
        starting_names = {a.name for a in starting_agents}
        run_names = {a.name for a in run_agents}
        missing = starting_names - run_names
        if missing:
            import warnings

            warnings.warn(
                f"Agents in starting_agents but not in run_agents: {missing}. "
                f"These agents will execute but cannot call or perceive other agents.",
                UserWarning,
                stacklevel=2,
            )

    router = Router(
        run_agents_list, tools or [], providers or [], run_agents=run_agents_list
    )
    runtime = Runtime(
        router,
        extra_instructions=extra_instructions,
        extra_instructions_scope=extra_instructions_scope,
    )
    async for event in runtime.execute_stream(
        starting_agents[0], message, attachments=attachments, history=history
    ):
        yield event
