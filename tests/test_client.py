import json
from toolbench.client import FakeClient, make_response, ToolCallRequest


def test_make_response_tool_call():
    r = make_response(tool_calls=[ToolCallRequest(id="c1", name="t", args={"x": 1})])
    am = r.assistant_message
    assert am["role"] == "assistant"
    assert am["tool_calls"][0]["function"]["name"] == "t"
    assert json.loads(am["tool_calls"][0]["function"]["arguments"]) == {"x": 1}


def test_make_response_content():
    r = make_response(content="hi", usage={"prompt": 3, "completion": 2})
    assert r.content == "hi"
    assert r.tool_calls == []
    assert r.usage == {"prompt": 3, "completion": 2}


def test_fake_client_scripts_in_order():
    c = FakeClient([make_response(content="a"), make_response(content="b")])
    assert c.chat("m", [], []).content == "a"
    assert c.chat("m", [], []).content == "b"
    assert len(c.calls) == 2
