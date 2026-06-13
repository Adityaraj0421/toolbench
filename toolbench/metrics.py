from __future__ import annotations

from collections import Counter


def summarize(trace) -> dict:
    by_tool: Counter = Counter()
    failures = 0
    total_calls = 0
    latency = 0
    for t in trace.turns:
        latency += t.get("ms", 0)
        for c in t["tool_calls"]:
            by_tool[c["name"]] += 1
            total_calls += 1
            if not c["ok"]:
                failures += 1
    return {
        "model": trace.model,
        "turns": len(trace.turns),
        "total_tokens": trace.total_tokens,
        "tool_calls": total_calls,
        "failures": failures,
        "by_tool": dict(by_tool),
        "latency_ms": latency,
        "completed": not trace.hit_max_turns,
    }
