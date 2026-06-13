from __future__ import annotations

import json
import os
import time
from dataclasses import dataclass


@dataclass
class ToolCallRequest:
    id: str
    name: str
    args: dict


@dataclass
class ModelResponse:
    content: str | None
    tool_calls: list[ToolCallRequest]
    usage: dict  # {"prompt": int, "completion": int}
    assistant_message: dict  # ready to append to the message list


def make_response(content=None, tool_calls=None, usage=None) -> ModelResponse:
    tool_calls = tool_calls or []
    am: dict = {"role": "assistant", "content": content}
    if tool_calls:
        am["tool_calls"] = [
            {
                "id": c.id,
                "type": "function",
                "function": {"name": c.name, "arguments": json.dumps(c.args)},
            }
            for c in tool_calls
        ]
    return ModelResponse(
        content=content,
        tool_calls=tool_calls,
        usage=usage or {"prompt": 0, "completion": 0},
        assistant_message=am,
    )


class FakeClient:
    """Returns scripted ModelResponses in order. For offline tests."""

    def __init__(self, responses):
        self._responses = list(responses)
        self.calls: list[dict] = []

    def chat(self, model, messages, tools) -> ModelResponse:
        self.calls.append({"model": model, "messages": list(messages), "tools": tools})
        return self._responses.pop(0)


class OpenRouterClient:
    def __init__(self, api_key=None, base_url="https://openrouter.ai/api/v1", max_retries=3):
        from openai import OpenAI

        self._client = OpenAI(
            api_key=api_key or os.environ["OPENROUTER_API_KEY"], base_url=base_url
        )
        self._max_retries = max_retries

    def chat(self, model, messages, tools) -> ModelResponse:
        last_err = None
        for attempt in range(self._max_retries):
            try:
                resp = self._client.chat.completions.create(
                    model=model, messages=messages, tools=tools or None
                )
                break
            except Exception as e:  # rate limit / transient network
                last_err = e
                time.sleep(2**attempt)
        else:
            raise last_err
        msg = resp.choices[0].message
        calls = []
        for tc in msg.tool_calls or []:
            try:
                args = json.loads(tc.function.arguments or "{}")
            except json.JSONDecodeError:
                args = {}
            calls.append(ToolCallRequest(id=tc.id, name=tc.function.name, args=args))
        usage = {
            "prompt": getattr(resp.usage, "prompt_tokens", 0),
            "completion": getattr(resp.usage, "completion_tokens", 0),
        }
        return ModelResponse(
            content=msg.content,
            tool_calls=calls,
            usage=usage,
            assistant_message=msg.model_dump(exclude_none=True),
        )
