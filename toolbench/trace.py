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
