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
