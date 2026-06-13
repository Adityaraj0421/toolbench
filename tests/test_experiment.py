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
