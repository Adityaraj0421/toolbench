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
    task: str
    task_prompt: str
    variant: str
    overrides: dict
    repeat: int


@dataclass
class ExperimentConfig:
    name: str
    tasks: list  # list of {"name": str, "prompt": str}
    system: str | None
    max_turns: int
    models: list
    tools: list
    variants: list
    repeats: int
    max_output_tokens: int = 1024


def _normalize_tasks(data: dict) -> list:
    """Build the task list. Accept a `tasks:` list (dicts or bare strings) or a
    single `task:` string (wrapped as one task named "default")."""
    if data.get("tasks"):
        out = []
        for i, t in enumerate(data["tasks"]):
            if isinstance(t, str):
                out.append({"name": f"t{i + 1}", "prompt": t})
            else:
                out.append({"name": t["name"], "prompt": t["prompt"]})
        return out
    return [{"name": "default", "prompt": data["task"]}]


def load_experiment(path) -> ExperimentConfig:
    data = yaml.safe_load(Path(path).read_text())
    return ExperimentConfig(
        name=data["name"],
        tasks=_normalize_tasks(data),
        system=data.get("system"),
        max_turns=data.get("max_turns", 12),
        models=data["models"],
        tools=data["tools"],
        variants=data.get("variants") or [{"name": "baseline"}],
        repeats=data.get("repeats", 1),
        max_output_tokens=data.get("max_output_tokens", 1024),
    )


def expand_matrix(config: ExperimentConfig) -> list[Cell]:
    cells = []
    for task in config.tasks:
        for model in config.models:
            for variant in config.variants:
                for r in range(config.repeats):
                    cells.append(
                        Cell(
                            model=model,
                            task=task["name"],
                            task_prompt=task["prompt"],
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
        meta = {
            "experiment": config.name,
            "task": cell.task,
            "variant": cell.variant,
            "repeat": cell.repeat,
        }
        try:
            trace = run_agent(
                cell.task_prompt,
                registry,
                cell.model,
                client,
                system=config.system,
                max_turns=config.max_turns,
                meta=meta,
            )
            fname = f"{_slug(cell.model)}__{_slug(cell.task)}__{cell.variant}__{cell.repeat}.jsonl"
            trace.write(out / fname)
            s = summarize(trace)
        except Exception as e:  # one bad cell never kills the matrix
            s = {
                "model": cell.model,
                "turns": 0,
                "total_tokens": 0,
                "tool_calls": 0,
                "failures": 0,
                "by_tool": {},
                "latency_ms": 0,
                "completed": False,
                "error": f"{type(e).__name__}: {e}",
            }
        s["task"] = cell.task
        s["variant"] = cell.variant
        s["repeat"] = cell.repeat
        summaries.append(s)
    out.mkdir(parents=True, exist_ok=True)
    (out / "summary.json").write_text(json.dumps(summaries, indent=2))
    return summaries


def format_table(summaries: list[dict]) -> str:
    headers = ["model", "task", "variant", "turns", "tokens", "calls", "fail", "ms", "done"]
    rows = [
        [
            s["model"],
            s.get("task", ""),
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
