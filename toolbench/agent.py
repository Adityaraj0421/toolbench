from __future__ import annotations

import time

from .tools import ToolRegistry
from .trace import Trace


def _system_msg(system):
    return {"role": "system", "content": system or "You are a helpful assistant."}


def _user_msg(task):
    return {"role": "user", "content": task}


def _tool_msg(call_id, result):
    return {"role": "tool", "tool_call_id": call_id, "content": result}


def run_agent(task, tools, model, client, *, system=None, max_turns=12, meta=None) -> Trace:
    registry = tools if isinstance(tools, ToolRegistry) else ToolRegistry(tools)
    messages = [_system_msg(system), _user_msg(task)]
    trace = Trace(model=model, task=task, meta=meta or {})

    for i in range(max_turns):
        t0 = time.monotonic()
        resp = client.chat(model, messages, registry.schemas())
        ms = int((time.monotonic() - t0) * 1000)
        messages.append(resp.assistant_message)

        if not resp.tool_calls:
            trace.record_turn(i, resp.content, [], resp.usage, ms)
            return trace.finish(answer=resp.content)

        records = []
        for call in resp.tool_calls:
            tt0 = time.monotonic()
            result = registry.execute(call.name, call.args)
            tms = int((time.monotonic() - tt0) * 1000)
            records.append(
                {
                    "name": call.name,
                    "args": call.args,
                    "result": result,
                    "ok": not result.startswith("ERROR:"),
                    "ms": tms,
                }
            )
            messages.append(_tool_msg(call.id, result))
        trace.record_turn(i, resp.content, records, resp.usage, ms)

    return trace.finish(hit_max_turns=True)
