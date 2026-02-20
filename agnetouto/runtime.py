from __future__ import annotations

import asyncio
from dataclasses import dataclass

from agnetouto._constants import CALL_AGENT, FINISH
from agnetouto.agent import Agent
from agnetouto.context import Context, ToolCall
from agnetouto.exceptions import ToolError
from agnetouto.provider import Provider
from agnetouto.router import Router
from agnetouto.tool import Tool


@dataclass
class RunResult:
    output: str


class Runtime:
    def __init__(self, router: Router) -> None:
        self._router = router

    async def execute(self, agent: Agent, forward_message: str) -> str:
        return await self._run_agent_loop(agent, forward_message)

    async def _run_agent_loop(self, agent: Agent, forward_message: str) -> str:
        system_prompt = self._router.build_system_prompt(agent)
        context = Context(system_prompt)
        context.add_user(forward_message)
        tool_schemas = self._router.build_tool_schemas(agent.name)

        while True:
            response = await self._router.call_llm(agent, context, tool_schemas)

            if not response.tool_calls:
                return response.content or ""

            finish_call = _find_finish(response.tool_calls)
            if finish_call is not None:
                return finish_call.arguments.get("message", "")

            context.add_assistant_tool_calls(response.tool_calls, response.content)

            tasks = [
                self._execute_tool_call(tc)
                for tc in response.tool_calls
            ]
            results = await asyncio.gather(*tasks, return_exceptions=True)

            for tc, result in zip(response.tool_calls, results):
                if isinstance(result, BaseException):
                    context.add_tool_result(tc.id, tc.name, f"Error: {result}")
                else:
                    context.add_tool_result(tc.id, tc.name, result)

    async def _execute_tool_call(self, tc: ToolCall) -> str:
        if tc.name == CALL_AGENT:
            agent_name = tc.arguments.get("agent_name", "")
            message = tc.arguments.get("message", "")
            target = self._router.get_agent(agent_name)
            return await self._run_agent_loop(target, message)

        tool = self._router.get_tool(tc.name)
        try:
            return await tool.execute(**tc.arguments)
        except Exception as exc:
            raise ToolError(tc.name, str(exc)) from exc


def _find_finish(tool_calls: list[ToolCall]) -> ToolCall | None:
    for tc in tool_calls:
        if tc.name == FINISH:
            return tc
    return None


async def async_run(
    entry: Agent,
    message: str,
    agents: list[Agent],
    tools: list[Tool],
    providers: list[Provider],
) -> RunResult:
    router = Router(agents, tools, providers)
    runtime = Runtime(router)
    output = await runtime.execute(entry, message)
    return RunResult(output=output)


def run(
    entry: Agent,
    message: str,
    agents: list[Agent],
    tools: list[Tool],
    providers: list[Provider],
) -> RunResult:
    return asyncio.run(async_run(entry, message, agents, tools, providers))
