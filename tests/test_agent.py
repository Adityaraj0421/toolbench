from toolbench.agent import run_agent
from toolbench.client import FakeClient, make_response, ToolCallRequest
from toolbench.tools import tool


@tool
def adder(a: int, b: int) -> str:
    """Add two integers.

    Args:
        a: first
        b: second
    """
    return str(a + b)


def test_agent_calls_tool_then_finishes():
    responses = [
        make_response(
            tool_calls=[ToolCallRequest(id="c1", name="adder", args={"a": 2, "b": 3})],
            usage={"prompt": 10, "completion": 5},
        ),
        make_response(content="The answer is 5.", usage={"prompt": 20, "completion": 4}),
    ]
    trace = run_agent("add 2 and 3", [adder], "fake/model", FakeClient(responses))
    assert trace.answer == "The answer is 5."
    assert len(trace.turns) == 2
    assert trace.turns[0]["tool_calls"][0]["result"] == "5"
    assert trace.turns[0]["tool_calls"][0]["ok"] is True
    assert trace.total_tokens == 39


def test_agent_recovers_from_tool_error():
    responses = [
        make_response(tool_calls=[ToolCallRequest(id="c1", name="adder", args={"a": 2})]),
        make_response(content="recovered"),
    ]
    trace = run_agent("x", [adder], "fake/model", FakeClient(responses))
    assert trace.turns[0]["tool_calls"][0]["ok"] is False
    assert trace.answer == "recovered"


def test_agent_max_turns_guard():
    looping = [
        make_response(tool_calls=[ToolCallRequest(id="c", name="adder", args={"a": 1, "b": 1})])
        for _ in range(10)
    ]
    trace = run_agent("x", [adder], "fake/model", FakeClient(looping), max_turns=3)
    assert trace.hit_max_turns is True
    assert len(trace.turns) == 3
