from toolbench.trace import Trace
from toolbench.metrics import summarize


def test_summarize():
    tr = Trace(model="m", task="t")
    tr.record_turn(
        0,
        None,
        [
            {"name": "calculator", "args": {}, "result": "5", "ok": True, "ms": 2},
            {"name": "calculator", "args": {}, "result": "ERROR: x", "ok": False, "ms": 1},
        ],
        {"prompt": 10, "completion": 5},
        30,
    )
    tr.finish(answer="ok")
    s = summarize(tr)
    assert s["turns"] == 1
    assert s["total_tokens"] == 15
    assert s["tool_calls"] == 2
    assert s["failures"] == 1
    assert s["by_tool"] == {"calculator": 2}
    assert s["completed"] is True
    assert s["latency_ms"] == 30
