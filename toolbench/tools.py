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
