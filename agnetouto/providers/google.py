from __future__ import annotations

import uuid
from typing import Any

import google.generativeai as genai

from agnetouto.agent import Agent
from agnetouto.context import Context, ToolCall
from agnetouto.exceptions import ProviderError
from agnetouto.provider import Provider
from agnetouto.providers import LLMResponse, ProviderBackend

_JSON_TYPE_MAP: dict[str, int] = {
    "string": 1,
    "number": 2,
    "integer": 3,
    "boolean": 4,
    "array": 5,
    "object": 6,
}


class GoogleBackend(ProviderBackend):
    def __init__(self) -> None:
        self._configured: set[str] = set()

    async def call(
        self,
        context: Context,
        tools: list[dict[str, Any]],
        agent: Agent,
        provider: Provider,
    ) -> LLMResponse:
        if provider.name not in self._configured:
            genai.configure(api_key=provider.api_key)
            self._configured.add(provider.name)

        contents = _build_contents(context)
        google_tools = _build_tools(tools)

        gen_config: dict[str, Any] = {
            "temperature": agent.temperature,
            "max_output_tokens": agent.max_output_tokens,
        }
        if agent.reasoning:
            gen_config["thinking_config"] = {
                "thinking_budget": agent.reasoning_budget or 4096,
            }

        model = genai.GenerativeModel(
            agent.model,
            system_instruction=context.system_prompt,
        )

        params: dict[str, Any] = {
            "contents": contents,
            "generation_config": gen_config,
            **agent.extra,
        }
        if google_tools:
            params["tools"] = google_tools

        try:
            response = await model.generate_content_async(**params)
        except Exception as exc:
            raise ProviderError(provider.name, str(exc)) from exc

        if not response.candidates:
            raise ProviderError(provider.name, "Empty response: no candidates returned")

        content_text: str | None = None
        parsed_calls: list[ToolCall] = []

        for part in response.candidates[0].content.parts:
            fn = part.function_call
            if fn and fn.name:
                parsed_calls.append(
                    ToolCall(
                        id=uuid.uuid4().hex,
                        name=fn.name,
                        arguments=dict(fn.args) if fn.args else {},
                    )
                )
            elif part.text:
                content_text = (content_text or "") + part.text

        return LLMResponse(content=content_text, tool_calls=parsed_calls)


def _build_contents(context: Context) -> list[Any]:
    contents: list[Any] = []
    ctx_messages = context.messages
    i = 0

    while i < len(ctx_messages):
        msg = ctx_messages[i]

        if msg.role == "user":
            contents.append(
                genai.protos.Content(
                    role="user",
                    parts=[genai.protos.Part(text=msg.content or "")],
                )
            )
            i += 1

        elif msg.role == "assistant":
            parts: list[Any] = []
            if msg.content:
                parts.append(genai.protos.Part(text=msg.content))
            if msg.tool_calls:
                for tc in msg.tool_calls:
                    parts.append(
                        genai.protos.Part(
                            function_call=genai.protos.FunctionCall(
                                name=tc.name, args=tc.arguments
                            )
                        )
                    )
            contents.append(genai.protos.Content(role="model", parts=parts))
            i += 1

        elif msg.role == "tool":
            parts = []
            while i < len(ctx_messages) and ctx_messages[i].role == "tool":
                parts.append(
                    genai.protos.Part(
                        function_response=genai.protos.FunctionResponse(
                            name=ctx_messages[i].tool_name or "",
                            response={"result": ctx_messages[i].content or ""},
                        )
                    )
                )
                i += 1
            contents.append(genai.protos.Content(role="function", parts=parts))

        else:
            i += 1

    return contents


def _json_schema_to_google(schema: dict[str, Any]) -> genai.protos.Schema:
    raw_type = schema.get("type", "string")
    type_enum = _JSON_TYPE_MAP.get(raw_type, 1)

    props: dict[str, Any] | None = None
    if "properties" in schema:
        props = {
            k: _json_schema_to_google(v) for k, v in schema["properties"].items()
        }

    return genai.protos.Schema(
        type=type_enum,
        properties=props,
        required=schema.get("required"),
    )


def _build_tools(tools: list[dict[str, Any]]) -> list[Any] | None:
    if not tools:
        return None

    func_decls: list[Any] = []
    for t in tools:
        params = t.get("parameters")
        google_params = _json_schema_to_google(params) if params else None
        func_decls.append(
            genai.protos.FunctionDeclaration(
                name=t["name"],
                description=t["description"],
                parameters=google_params,
            )
        )

    return [genai.protos.Tool(function_declarations=func_decls)]
