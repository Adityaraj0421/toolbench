# toolbench Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Build a minimal, transparent, pluggable tool-calling agent harness in Python (OpenRouter) for experimenting with tool design across models.

**Architecture:** A hand-rolled agent loop drives an OpenAI-compatible client (OpenRouter). Tools are plain functions decorated with `@tool`, which derives the JSON schema from type hints + docstring. An experiment runner expands a `models × variants × repeats` matrix, writes one JSONL trace per cell, and prints aggregated metrics. A `FakeClient` makes the whole loop testable offline.

**Tech Stack:** Python 3.11+, `openai` SDK (pointed at OpenRouter), `pyyaml`, `pytest`. No agent framework.

---

## File Structure

| File | Responsibility |
|------|----------------|
| `toolbench/tools.py` | `@tool` decorator, schema derivation, `ToolRegistry` |
| `toolbench/client.py` | `OpenRouterClient`, `FakeClient`, `ModelResponse`, `ToolCallRequest`, `make_response` |
| `toolbench/trace.py` | `Trace` dataclass + JSONL writer |
| `toolbench/agent.py` | `run_agent()` loop |
| `toolbench/metrics.py` | `summarize()` per-trace metrics |
| `toolbench/experiment.py` | config load, matrix expansion, runner, table formatter |
| `toolbench/cli.py` + `__main__.py` | argparse entrypoint |
| `toolbench/builtins/*` | calculator, http_fetch, files (safe); shell (opt-in) |
| `experiments/example.yaml` | sample experiment |
| `tests/*` | pytest, fully offline via FakeClient |

---

## Task 1: Project scaffold

**Files:**
- Create: `pyproject.toml`, `toolbench/__init__.py`, `toolbench/builtins/__init__.py`, `tests/__init__.py`, `.env.example`

- [ ] **Step 1: Create `pyproject.toml`**

```toml
[project]
name = "toolbench"
version = "0.1.0"
description = "A pluggable tool-calling agent experiment bench"
requires-python = ">=3.11"
dependencies = ["openai>=1.40", "pyyaml>=6.0"]

[project.optional-dependencies]
dev = ["pytest>=8.0"]

[build-system]
requires = ["setuptools>=68"]
build-backend = "setuptools.build_meta"

[tool.setuptools.packages.find]
include = ["toolbench*"]
```

- [ ] **Step 2: Create package init files**

`toolbench/__init__.py`:
```python
"""toolbench — a pluggable tool-calling agent experiment bench."""
__version__ = "0.1.0"
```

`tests/__init__.py`: empty file. `toolbench/builtins/__init__.py`: empty for now (filled in Task 6).

- [ ] **Step 3: Create `.env.example`**

```
OPENROUTER_API_KEY=
```

- [ ] **Step 4: Create and activate a venv, install editable**

Run:
```bash
python3 -m venv .venv && . .venv/bin/activate && pip install -e ".[dev]"
```
Expected: installs openai, pyyaml, pytest; `Successfully installed toolbench-0.1.0`.

- [ ] **Step 5: Verify import**

Run: `python -c "import toolbench; print(toolbench.__version__)"`
Expected: `0.1.0`

- [ ] **Step 6: Commit**

```bash
git add pyproject.toml toolbench tests .env.example
git commit -m "chore: scaffold toolbench package"
```

---

## Task 2: Tool decorator + registry (`tools.py`)

**Files:**
- Create: `toolbench/tools.py`
- Test: `tests/test_tools.py`

- [ ] **Step 1: Write the failing test**

`tests/test_tools.py`:
```python
import pytest
from toolbench.tools import tool, ToolRegistry


@tool
def sample(city: str, days: int = 3, fast: bool = False) -> str:
    """Get a forecast.

    Args:
        city: City name.
        days: Number of days.
    """
    return f"{city}:{days}:{fast}"


def test_schema_derivation():
    fn = sample.tool.schema()["function"]
    assert fn["name"] == "sample"
    assert fn["description"] == "Get a forecast."
    props = fn["parameters"]["properties"]
    assert props["city"] == {"type": "string", "description": "City name."}
    assert props["days"]["type"] == "integer"
    assert props["fast"]["type"] == "boolean"
    assert fn["parameters"]["required"] == ["city"]


def test_registry_execute_and_error():
    reg = ToolRegistry([sample])
    assert reg.execute("sample", {"city": "NYC"}) == "NYC:3:False"
    assert reg.execute("nope", {}).startswith("ERROR: unknown tool")
    assert reg.execute("sample", {}).startswith("ERROR:")


def test_variant_override_keeps_function():
    reg = ToolRegistry([sample], overrides={"sample": {"description": "Math."}})
    assert reg.schemas()[0]["function"]["description"] == "Math."
    assert reg.execute("sample", {"city": "LA"}) == "LA:3:False"


def test_unsupported_type_fails():
    with pytest.raises(TypeError):
        @tool
        def bad(x: dict) -> str:
            """Bad.

            Args:
                x: a dict.
            """
            return "x"
```

- [ ] **Step 2: Run test to verify it fails**

Run: `pytest tests/test_tools.py -v`
Expected: FAIL — `ModuleNotFoundError: No module named 'toolbench.tools'`

- [ ] **Step 3: Write `toolbench/tools.py`**

```python
from __future__ import annotations

import inspect
import json
import re
from dataclasses import dataclass
from typing import Any, Callable, Union, get_args, get_origin

_PYTYPE_TO_JSON = {str: "string", int: "integer", float: "number", bool: "boolean", list: "array"}


def _json_type(annotation) -> str:
    origin = get_origin(annotation)
    if origin is Union:  # Optional[X] == Union[X, None]
        args = [a for a in get_args(annotation) if a is not type(None)]
        if len(args) == 1:
            return _json_type(args[0])
    if annotation in _PYTYPE_TO_JSON:
        return _PYTYPE_TO_JSON[annotation]
    if origin in _PYTYPE_TO_JSON:  # e.g. list[str]
        return _PYTYPE_TO_JSON[origin]
    raise TypeError(f"Unsupported tool parameter type: {annotation!r}")


def _parse_docstring(doc: str | None) -> tuple[str, dict[str, str]]:
    doc = doc or ""
    parts = re.split(r"\n\s*Args:\s*\n", doc, maxsplit=1)
    summary = " ".join(parts[0].split()).strip()
    arg_desc: dict[str, str] = {}
    if len(parts) == 2:
        for line in parts[1].splitlines():
            m = re.match(r"\s*(\w+)\s*:\s*(.+)", line)
            if m:
                arg_desc[m.group(1)] = m.group(2).strip()
    return summary, arg_desc


@dataclass
class Tool:
    name: str
    description: str
    parameters: dict
    func: Callable

    def schema(self) -> dict:
        return {
            "type": "function",
            "function": {
                "name": self.name,
                "description": self.description,
                "parameters": self.parameters,
            },
        }


def tool(func: Callable) -> Callable:
    sig = inspect.signature(func)
    hints = getattr(func, "__annotations__", {})
    summary, arg_desc = _parse_docstring(func.__doc__)
    props: dict[str, Any] = {}
    required: list[str] = []
    for pname, param in sig.parameters.items():
        annotation = hints.get(pname, str)
        prop: dict[str, Any] = {"type": _json_type(annotation)}
        if pname in arg_desc:
            prop["description"] = arg_desc[pname]
        props[pname] = prop
        if param.default is inspect.Parameter.empty:
            required.append(pname)
    func.tool = Tool(
        name=func.__name__,
        description=summary,
        parameters={"type": "object", "properties": props, "required": required},
        func=func,
    )
    return func


class ToolRegistry:
    def __init__(self, tools, overrides: dict | None = None):
        overrides = overrides or {}
        self._tools: dict[str, Tool] = {}
        for t in tools:
            base = t.tool if hasattr(t, "tool") else t
            ov = overrides.get(base.name, {})
            self._tools[ov.get("name", base.name)] = Tool(
                name=ov.get("name", base.name),
                description=ov.get("description", base.description),
                parameters=ov.get("parameters", base.parameters),
                func=base.func,
            )

    def schemas(self) -> list[dict]:
        return [t.schema() for t in self._tools.values()]

    def execute(self, name: str, args: dict) -> str:
        if name not in self._tools:
            return f"ERROR: unknown tool '{name}'"
        try:
            result = self._tools[name].func(**args)
            return result if isinstance(result, str) else json.dumps(result)
        except Exception as e:  # fed back to the model so it can recover
            return f"ERROR: {type(e).__name__}: {e}"
```

- [ ] **Step 4: Run test to verify it passes**

Run: `pytest tests/test_tools.py -v`
Expected: PASS (4 passed)

- [ ] **Step 5: Commit**

```bash
git add toolbench/tools.py tests/test_tools.py
git commit -m "feat: @tool decorator, schema derivation, ToolRegistry"
```

---

## Task 3: Client + fakes (`client.py`)

**Files:**
- Create: `toolbench/client.py`
- Test: `tests/test_client.py`

- [ ] **Step 1: Write the failing test**

`tests/test_client.py`:
```python
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
```

- [ ] **Step 2: Run test to verify it fails**

Run: `pytest tests/test_client.py -v`
Expected: FAIL — `ModuleNotFoundError: No module named 'toolbench.client'`

- [ ] **Step 3: Write `toolbench/client.py`**

```python
from __future__ import annotations

import json
import os
import time
from dataclasses import dataclass, field


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
```

- [ ] **Step 4: Run test to verify it passes**

Run: `pytest tests/test_client.py -v`
Expected: PASS (3 passed)

- [ ] **Step 5: Commit**

```bash
git add toolbench/client.py tests/test_client.py
git commit -m "feat: OpenRouter client, FakeClient, ModelResponse"
```

---

## Task 4: Trace (`trace.py`)

**Files:**
- Create: `toolbench/trace.py`
- Test: `tests/test_trace.py`

- [ ] **Step 1: Write the failing test**

`tests/test_trace.py`:
```python
import json
from toolbench.trace import Trace


def test_total_tokens_and_records(tmp_path):
    tr = Trace(model="m", task="t", meta={"experiment": "e", "variant": "baseline"})
    tr.record_turn(
        0,
        "thinking",
        [{"name": "calc", "args": {}, "result": "5", "ok": True, "ms": 2}],
        {"prompt": 10, "completion": 5},
        30,
    )
    tr.finish(answer="done")
    assert tr.total_tokens == 15

    recs = tr.to_records()
    assert recs[0]["type"] == "meta"
    assert recs[0]["variant"] == "baseline"
    assert recs[1]["type"] == "turn"
    assert recs[-1]["type"] == "final"
    assert recs[-1]["answer"] == "done"
    assert recs[-1]["turns"] == 1

    path = tmp_path / "sub" / "t.jsonl"
    tr.write(path)
    lines = path.read_text().strip().splitlines()
    assert len(lines) == 3
    assert json.loads(lines[0])["type"] == "meta"
```

- [ ] **Step 2: Run test to verify it fails**

Run: `pytest tests/test_trace.py -v`
Expected: FAIL — `ModuleNotFoundError: No module named 'toolbench.trace'`

- [ ] **Step 3: Write `toolbench/trace.py`**

```python
from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path


@dataclass
class Trace:
    model: str
    task: str
    meta: dict = field(default_factory=dict)
    turns: list = field(default_factory=list)
    answer: str | None = None
    hit_max_turns: bool = False
    ok: bool = True

    def record_turn(self, i, content, tool_calls, usage, ms):
        self.turns.append(
            {"i": i, "content": content, "tool_calls": tool_calls, "usage": usage, "ms": ms}
        )

    def finish(self, answer=None, hit_max_turns=False):
        self.answer = answer
        self.hit_max_turns = hit_max_turns
        return self

    @property
    def total_tokens(self) -> int:
        return sum(
            t["usage"].get("prompt", 0) + t["usage"].get("completion", 0) for t in self.turns
        )

    def to_records(self) -> list[dict]:
        recs = [{"type": "meta", "model": self.model, "task": self.task, **self.meta}]
        for t in self.turns:
            recs.append({"type": "turn", **t})
        recs.append(
            {
                "type": "final",
                "answer": self.answer,
                "turns": len(self.turns),
                "total_tokens": self.total_tokens,
                "ok": self.ok,
                "hit_max_turns": self.hit_max_turns,
            }
        )
        return recs

    def write(self, path):
        p = Path(path)
        p.parent.mkdir(parents=True, exist_ok=True)
        with p.open("w") as f:
            for rec in self.to_records():
                f.write(json.dumps(rec) + "\n")
```

- [ ] **Step 4: Run test to verify it passes**

Run: `pytest tests/test_trace.py -v`
Expected: PASS (1 passed)

- [ ] **Step 5: Commit**

```bash
git add toolbench/trace.py tests/test_trace.py
git commit -m "feat: Trace dataclass + JSONL writer"
```

---

## Task 5: Agent loop (`agent.py`)

**Files:**
- Create: `toolbench/agent.py`
- Test: `tests/test_agent.py`

- [ ] **Step 1: Write the failing test**

`tests/test_agent.py`:
```python
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
```

- [ ] **Step 2: Run test to verify it fails**

Run: `pytest tests/test_agent.py -v`
Expected: FAIL — `ModuleNotFoundError: No module named 'toolbench.agent'`

- [ ] **Step 3: Write `toolbench/agent.py`**

```python
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
```

- [ ] **Step 4: Run test to verify it passes**

Run: `pytest tests/test_agent.py -v`
Expected: PASS (3 passed)

- [ ] **Step 5: Commit**

```bash
git add toolbench/agent.py tests/test_agent.py
git commit -m "feat: run_agent tool-calling loop"
```

---

## Task 6: Built-in tools + safety (`builtins/`)

**Files:**
- Create: `toolbench/builtins/calculator.py`, `toolbench/builtins/http_fetch.py`, `toolbench/builtins/files.py`, `toolbench/builtins/shell.py`
- Modify: `toolbench/builtins/__init__.py`
- Test: `tests/test_safety.py`

- [ ] **Step 1: Write the failing test**

`tests/test_safety.py`:
```python
import pytest
from toolbench.builtins.calculator import calculator
from toolbench.builtins.http_fetch import _is_safe_url
import toolbench.builtins.files as F
from toolbench.builtins import resolve_tools, BUILTINS


def test_calculator_basic():
    assert calculator("2 * (3 + 4)") == "14"


def test_calculator_rejects_code():
    with pytest.raises(Exception):
        calculator("__import__('os').system('echo hi')")


def test_ssrf_guard_blocks_private_and_nonhttp():
    assert _is_safe_url("http://127.0.0.1") is False
    assert _is_safe_url("http://169.254.169.254/latest/meta-data") is False
    assert _is_safe_url("ftp://example.com") is False


def test_files_confined_to_workspace(tmp_path, monkeypatch):
    monkeypatch.setattr(F, "WORKSPACE", tmp_path)
    assert "wrote" in F.write_file("a.txt", "hello")
    assert F.read_file("a.txt") == "hello"
    with pytest.raises(ValueError):
        F.write_file("../escape.txt", "x")


def test_resolve_tools_and_shell_not_registered():
    assert resolve_tools(["calculator"])[0] is calculator
    assert "run_shell" not in BUILTINS
    with pytest.raises(KeyError):
        resolve_tools(["does_not_exist"])
```

- [ ] **Step 2: Run test to verify it fails**

Run: `pytest tests/test_safety.py -v`
Expected: FAIL — `ModuleNotFoundError: No module named 'toolbench.builtins.calculator'`

- [ ] **Step 3a: Write `toolbench/builtins/calculator.py`**

```python
import ast
import operator

from toolbench.tools import tool

_OPS = {
    ast.Add: operator.add,
    ast.Sub: operator.sub,
    ast.Mult: operator.mul,
    ast.Div: operator.truediv,
    ast.Pow: operator.pow,
    ast.Mod: operator.mod,
    ast.USub: operator.neg,
}


def _eval(node):
    if isinstance(node, ast.Constant) and isinstance(node.value, (int, float)):
        return node.value
    if isinstance(node, ast.BinOp) and type(node.op) in _OPS:
        return _OPS[type(node.op)](_eval(node.left), _eval(node.right))
    if isinstance(node, ast.UnaryOp) and type(node.op) in _OPS:
        return _OPS[type(node.op)](_eval(node.operand))
    raise ValueError("unsupported expression")


@tool
def calculator(expression: str) -> str:
    """Evaluate a basic arithmetic expression and return the numeric result.

    Args:
        expression: A math expression like "2 * (3 + 4)".
    """
    result = _eval(ast.parse(expression, mode="eval").body)
    return str(int(result) if result == int(result) else result)
```

- [ ] **Step 3b: Write `toolbench/builtins/http_fetch.py`**

```python
import ipaddress
import socket
from urllib.parse import urlparse
from urllib.request import Request, urlopen

from toolbench.tools import tool

_MAX_CHARS = 100_000


def _is_safe_url(url: str) -> bool:
    p = urlparse(url)
    if p.scheme not in ("http", "https") or not p.hostname:
        return False
    try:
        infos = socket.getaddrinfo(p.hostname, None)
    except socket.gaierror:
        return False
    for info in infos:
        ip = ipaddress.ip_address(info[4][0])
        if (
            ip.is_private
            or ip.is_loopback
            or ip.is_link_local
            or ip.is_reserved
            or ip.is_multicast
        ):
            return False
    return True


@tool
def http_fetch(url: str) -> str:
    """Fetch a web page over HTTP GET and return up to 100k characters of text.

    Args:
        url: An http(s) URL to GET.
    """
    if not _is_safe_url(url):
        return "ERROR: refused unsafe or non-public URL"
    req = Request(url, headers={"User-Agent": "toolbench/0.1"})
    with urlopen(req, timeout=15) as r:  # noqa: S310 (scheme validated above)
        return r.read(_MAX_CHARS).decode("utf-8", errors="replace")
```

- [ ] **Step 3c: Write `toolbench/builtins/files.py`**

```python
from pathlib import Path

from toolbench.tools import tool

WORKSPACE = Path("workspace").resolve()


def _safe_path(path: str) -> Path:
    WORKSPACE.mkdir(parents=True, exist_ok=True)
    p = (WORKSPACE / path).resolve()
    if p != WORKSPACE and not str(p).startswith(str(WORKSPACE) + "/"):
        raise ValueError("path escapes workspace")
    return p


@tool
def read_file(path: str) -> str:
    """Read a UTF-8 text file from the workspace.

    Args:
        path: Path relative to the workspace directory.
    """
    return _safe_path(path).read_text()


@tool
def write_file(path: str, content: str) -> str:
    """Write a UTF-8 text file into the workspace, creating parent dirs.

    Args:
        path: Path relative to the workspace directory.
        content: Text to write.
    """
    p = _safe_path(path)
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text(content)
    return f"wrote {len(content)} chars to {path}"
```

- [ ] **Step 3d: Write `toolbench/builtins/shell.py`**

```python
"""OPT-IN ONLY. Runs arbitrary shell commands on the real machine.

This tool is intentionally NOT registered in builtins/__init__.py. To use it,
import it explicitly and pass it to run_agent yourself, accepting the risk.
"""

import subprocess

from toolbench.tools import tool


@tool
def run_shell(command: str) -> str:
    """Run a shell command and return combined stdout/stderr. DANGEROUS: opt-in.

    Args:
        command: The shell command to execute.
    """
    proc = subprocess.run(
        command, shell=True, capture_output=True, text=True, timeout=30
    )
    return (proc.stdout + proc.stderr)[:10_000]
```

- [ ] **Step 3e: Write `toolbench/builtins/__init__.py`**

```python
from .calculator import calculator
from .files import read_file, write_file
from .http_fetch import http_fetch

# NOTE: run_shell is deliberately excluded — opt-in only.
_ALL = [calculator, http_fetch, read_file, write_file]
BUILTINS = {f.tool.name: f for f in _ALL}


def resolve_tools(names):
    missing = [n for n in names if n not in BUILTINS]
    if missing:
        raise KeyError(f"Unknown tool(s): {missing}. Available: {sorted(BUILTINS)}")
    return [BUILTINS[n] for n in names]
```

- [ ] **Step 4: Run test to verify it passes**

Run: `pytest tests/test_safety.py -v`
Expected: PASS (5 passed)

- [ ] **Step 5: Commit**

```bash
git add toolbench/builtins tests/test_safety.py
git commit -m "feat: builtin tools (calculator, http_fetch, files) + opt-in shell"
```

---

## Task 7: Metrics (`metrics.py`)

**Files:**
- Create: `toolbench/metrics.py`
- Test: `tests/test_metrics.py`

- [ ] **Step 1: Write the failing test**

`tests/test_metrics.py`:
```python
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
```

- [ ] **Step 2: Run test to verify it fails**

Run: `pytest tests/test_metrics.py -v`
Expected: FAIL — `ModuleNotFoundError: No module named 'toolbench.metrics'`

- [ ] **Step 3: Write `toolbench/metrics.py`**

```python
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
```

- [ ] **Step 4: Run test to verify it passes**

Run: `pytest tests/test_metrics.py -v`
Expected: PASS (1 passed)

- [ ] **Step 5: Commit**

```bash
git add toolbench/metrics.py tests/test_metrics.py
git commit -m "feat: per-trace metrics summary"
```

---

## Task 8: Experiment runner (`experiment.py`)

**Files:**
- Create: `toolbench/experiment.py`
- Test: `tests/test_experiment.py`

- [ ] **Step 1: Write the failing test**

`tests/test_experiment.py`:
```python
from pathlib import Path

from toolbench.experiment import (
    ExperimentConfig,
    load_experiment,
    expand_matrix,
    run_experiment,
    format_table,
)
from toolbench.client import FakeClient, make_response


def _cfg():
    return ExperimentConfig(
        name="t",
        task="add",
        system=None,
        max_turns=4,
        models=["m1", "m2"],
        tools=["calculator"],
        variants=[
            {"name": "baseline"},
            {"name": "terse", "overrides": {"calculator": {"description": "Math."}}},
        ],
        repeats=1,
    )


def test_expand_matrix():
    cells = expand_matrix(_cfg())
    assert len(cells) == 4  # 2 models x 2 variants x 1 repeat
    assert {c.model for c in cells} == {"m1", "m2"}
    assert {c.variant for c in cells} == {"baseline", "terse"}


def test_load_experiment_defaults(tmp_path):
    p = tmp_path / "e.yaml"
    p.write_text("name: t\ntask: hi\nmodels: [m1]\ntools: [calculator]\n")
    cfg = load_experiment(p)
    assert cfg.models == ["m1"]
    assert cfg.variants == [{"name": "baseline"}]
    assert cfg.repeats == 1


def test_run_experiment_writes_traces(tmp_path):
    responses = [make_response(content="done") for _ in range(4)]
    summaries = run_experiment(_cfg(), FakeClient(responses), out_dir=tmp_path)
    assert len(summaries) == 4
    out = Path(tmp_path) / "t"
    assert len(list(out.glob("*.jsonl"))) == 4
    assert (out / "summary.json").exists()


def test_format_table_has_headers():
    table = format_table([
        {"model": "m1", "variant": "baseline", "turns": 2, "total_tokens": 100,
         "tool_calls": 1, "failures": 0, "latency_ms": 5, "completed": True}
    ])
    assert "model" in table and "m1" in table
```

- [ ] **Step 2: Run test to verify it fails**

Run: `pytest tests/test_experiment.py -v`
Expected: FAIL — `ModuleNotFoundError: No module named 'toolbench.experiment'`

- [ ] **Step 3: Write `toolbench/experiment.py`**

```python
from __future__ import annotations

import json
import re
from dataclasses import dataclass
from pathlib import Path

import yaml

from .agent import run_agent
from .builtins import resolve_tools
from .metrics import summarize
from .tools import ToolRegistry


@dataclass
class Cell:
    model: str
    variant: str
    overrides: dict
    repeat: int


@dataclass
class ExperimentConfig:
    name: str
    task: str
    system: str | None
    max_turns: int
    models: list
    tools: list
    variants: list
    repeats: int


def load_experiment(path) -> ExperimentConfig:
    data = yaml.safe_load(Path(path).read_text())
    return ExperimentConfig(
        name=data["name"],
        task=data["task"],
        system=data.get("system"),
        max_turns=data.get("max_turns", 12),
        models=data["models"],
        tools=data["tools"],
        variants=data.get("variants") or [{"name": "baseline"}],
        repeats=data.get("repeats", 1),
    )


def expand_matrix(config: ExperimentConfig) -> list[Cell]:
    cells = []
    for model in config.models:
        for variant in config.variants:
            for r in range(config.repeats):
                cells.append(
                    Cell(
                        model=model,
                        variant=variant["name"],
                        overrides=variant.get("overrides", {}),
                        repeat=r,
                    )
                )
    return cells


def _slug(s: str) -> str:
    return re.sub(r"[^A-Za-z0-9._-]+", "-", s)


def run_experiment(config: ExperimentConfig, client, out_dir="runs") -> list[dict]:
    out = Path(out_dir) / config.name
    funcs = resolve_tools(config.tools)
    summaries = []
    for cell in expand_matrix(config):
        registry = ToolRegistry(funcs, overrides=cell.overrides)
        meta = {"experiment": config.name, "variant": cell.variant, "repeat": cell.repeat}
        try:
            trace = run_agent(
                config.task,
                registry,
                cell.model,
                client,
                system=config.system,
                max_turns=config.max_turns,
                meta=meta,
            )
            fname = f"{_slug(cell.model)}__{cell.variant}__{cell.repeat}.jsonl"
            trace.write(out / fname)
            s = summarize(trace)
        except Exception as e:  # one bad cell never kills the matrix
            s = {
                "model": cell.model,
                "variant": cell.variant,
                "turns": 0,
                "total_tokens": 0,
                "tool_calls": 0,
                "failures": 0,
                "by_tool": {},
                "latency_ms": 0,
                "completed": False,
                "error": f"{type(e).__name__}: {e}",
            }
        s["variant"] = cell.variant
        s["repeat"] = cell.repeat
        summaries.append(s)
    out.mkdir(parents=True, exist_ok=True)
    (out / "summary.json").write_text(json.dumps(summaries, indent=2))
    return summaries


def format_table(summaries: list[dict]) -> str:
    headers = ["model", "variant", "turns", "tokens", "calls", "fail", "ms", "done"]
    rows = [
        [
            s["model"],
            s["variant"],
            s["turns"],
            s["total_tokens"],
            s["tool_calls"],
            s["failures"],
            s["latency_ms"],
            "y" if s["completed"] else "n",
        ]
        for s in summaries
    ]
    cols = [headers] + [[str(c) for c in r] for r in rows]
    widths = [max(len(cols[r][i]) for r in range(len(cols))) for i in range(len(headers))]
    fmt = lambda row: "  ".join(str(c).ljust(widths[i]) for i, c in enumerate(row))
    return "\n".join([fmt(headers)] + [fmt(r) for r in rows])
```

- [ ] **Step 4: Run test to verify it passes**

Run: `pytest tests/test_experiment.py -v`
Expected: PASS (4 passed)

- [ ] **Step 5: Commit**

```bash
git add toolbench/experiment.py tests/test_experiment.py
git commit -m "feat: experiment matrix runner + table formatter"
```

---

## Task 9: CLI + sample config + README

**Files:**
- Create: `toolbench/cli.py`, `toolbench/__main__.py`, `experiments/example.yaml`, `README.md`
- Test: `tests/test_cli.py`

- [ ] **Step 1: Write the failing test**

`tests/test_cli.py`:
```python
import pytest
from toolbench.cli import build_parser


def test_parser_run_subcommand():
    parser = build_parser()
    args = parser.parse_args(["run", "experiments/example.yaml", "--out", "runs"])
    assert args.cmd == "run"
    assert args.config == "experiments/example.yaml"
    assert args.out == "runs"


def test_parser_requires_subcommand():
    parser = build_parser()
    with pytest.raises(SystemExit):
        parser.parse_args([])
```

- [ ] **Step 2: Run test to verify it fails**

Run: `pytest tests/test_cli.py -v`
Expected: FAIL — `ModuleNotFoundError: No module named 'toolbench.cli'`

- [ ] **Step 3a: Write `toolbench/cli.py`**

```python
from __future__ import annotations

import argparse
import sys

from .client import OpenRouterClient
from .experiment import format_table, load_experiment, run_experiment


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="toolbench")
    sub = parser.add_subparsers(dest="cmd", required=True)
    run_p = sub.add_parser("run", help="run an experiment config")
    run_p.add_argument("config", help="path to a YAML experiment config")
    run_p.add_argument("--out", default="runs", help="output dir for traces")
    return parser


def main(argv=None) -> int:
    args = build_parser().parse_args(argv)
    if args.cmd == "run":
        config = load_experiment(args.config)
        summaries = run_experiment(config, OpenRouterClient(), out_dir=args.out)
        print(format_table(summaries))
    return 0


if __name__ == "__main__":
    sys.exit(main())
```

- [ ] **Step 3b: Write `toolbench/__main__.py`**

```python
import sys

from .cli import main

sys.exit(main())
```

- [ ] **Step 3c: Write `experiments/example.yaml`**

```yaml
name: calc-vs-fetch
task: "What is 17% of 2,340? Use a tool. Then state just the number."
system: "You are precise. Use the available tools when they help."
max_turns: 6
models:
  - openai/gpt-4o-mini
  - google/gemini-2.0-flash-001
tools: [calculator, http_fetch]
variants:
  - name: baseline
  - name: terse-desc
    overrides:
      calculator: { description: "Math." }
repeats: 1
```

- [ ] **Step 3d: Write `README.md`**

````markdown
# toolbench

A minimal, pluggable tool-calling agent harness for experimenting with tool
design across models (via OpenRouter).

## Setup

```bash
python3 -m venv .venv && . .venv/bin/activate
pip install -e ".[dev]"
cp .env.example .env   # then add your OPENROUTER_API_KEY
```

## Run an experiment

```bash
export OPENROUTER_API_KEY=sk-or-...
python -m toolbench run experiments/example.yaml
```

Traces land in `runs/<name>/*.jsonl`; a summary table prints to stdout and
`runs/<name>/summary.json` is written.

## Add a tool

```python
from toolbench.tools import tool

@tool
def reverse(text: str) -> str:
    """Reverse a string.

    Args:
        text: The string to reverse.
    """
    return text[::-1]
```

Register it by name in `toolbench/builtins/__init__.py` (`_ALL`), then list it
under `tools:` in an experiment config.

## Experiment axes

- `models:` — multi-model A/B (any OpenRouter model string)
- `variants:` — tweak a tool's `name`/`description`/`parameters` per variant and rerun
- `repeats:` — run each cell N times to see noise

## Tests

```bash
pytest        # fully offline, no API key needed
```
````

- [ ] **Step 4: Run test to verify it passes**

Run: `pytest tests/test_cli.py -v`
Expected: PASS (2 passed)

- [ ] **Step 5: Commit**

```bash
git add toolbench/cli.py toolbench/__main__.py experiments/example.yaml README.md tests/test_cli.py
git commit -m "feat: CLI entrypoint, sample experiment, README"
```

---

## Task 10: Full suite + live smoke

**Files:** none (verification only)

- [ ] **Step 1: Run the whole test suite offline**

Run: `pytest -v`
Expected: ALL PASS (test_tools 4, test_client 3, test_trace 1, test_agent 3, test_safety 5, test_metrics 1, test_experiment 4, test_cli 2).

- [ ] **Step 2: Live smoke (requires a real key)**

Run:
```bash
export OPENROUTER_API_KEY=sk-or-...   # real key
python -m toolbench run experiments/example.yaml
```
Expected: a printed table with rows for each `model × variant` cell, non-zero
`tokens`, and `done = y`; `runs/calc-vs-fetch/summary.json` written; four
`.jsonl` traces in `runs/calc-vs-fetch/`.

- [ ] **Step 3: Inspect a trace**

Run: `cat runs/calc-vs-fetch/openai-gpt-4o-mini__baseline__0.jsonl`
Expected: a `meta` line, one or more `turn` lines (at least one with a
`calculator` tool call computing `0.17*2340`), and a `final` line with the
answer `397.8`.

- [ ] **Step 4: Final commit**

```bash
git add -A
git commit -m "test: full offline suite green; live smoke verified"
```

---

## Self-Review

**Spec coverage:**
- Provider/runtime (OpenRouter + Python + openai SDK) → Task 1, Task 3 (`OpenRouterClient`). ✓
- `@tool` from signature → Task 2. ✓
- Agent loop + data flow → Task 5. ✓
- Full call tracing → Task 4 (`Trace`) + Task 5 records turns/usage/latency. ✓
- Tweak-and-rerun variants → Task 2 (`overrides`) + Task 8 (per-cell registry). ✓
- Multi-model A/B → Task 8 (`expand_matrix` over `models`). ✓
- Usage metrics → Task 7 + Task 8 (`summary.json`, table). ✓
- Error handling: tool errors fed back (Task 5), API retry/backoff (Task 3), per-cell isolation (Task 8), max_turns (Task 5). ✓
- Safety: safe calculator, SSRF guard, workspace sandbox, opt-in shell → Task 6. ✓
- Offline testing via FakeClient → Tasks 3,5,8. ✓
- CLI `python -m toolbench run` + `__main__.py` → Task 9. ✓
- Success criteria (run command, pytest green, 1-function tool add, variant rerun) → Tasks 9,10 + README. ✓

**Placeholder scan:** No TBD/TODO; every code step is complete. ✓

**Type consistency:** `ModelResponse`/`ToolCallRequest`/`make_response` (Task 3) used identically in Tasks 5,8. `ToolRegistry(tools, overrides)` signature (Task 2) matches calls in Task 8. `Trace.record_turn/finish/total_tokens/write` (Task 4) match usage in Tasks 5,7,8. `summarize()` keys (Task 7) match `format_table()` reads (Task 8). `resolve_tools` (Task 6) matches Task 8 import. ✓
