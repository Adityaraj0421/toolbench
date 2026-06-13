# toolbench — a tool-design experiment bench

**Date:** 2026-06-14
**Status:** Approved design, pending implementation plan

## 1. Purpose

A minimal, transparent, pluggable tool-calling agent harness in Python whose
*point* is experimenting with tool design: how a tool's name, description, and
parameter schema change the way a model decides to call it, and how different
models wield the same tools.

It is deliberately **not** a framework wrapper. The agent loop is hand-rolled
and short enough to read top to bottom, because the mechanics of tool-calling
are the subject of study, not an implementation detail to hide.

Origin: inspired by browsing the leaked agent tool schemas in `CL4R1T4S/`
(Cursor, Devin, Windsurf, Manus, Replit). Those are large hand-authored JSON
blobs; this project takes the opposite approach and derives schemas from code.

## 2. Goals / non-goals

**Goals**
- Adding a new tool costs ~10 lines (a decorated Python function).
- The agent loop is readable and fully testable offline.
- Four experiment features, all driven by a config file:
  1. **Full call tracing** — every turn, tool call, args, result, tokens, latency.
  2. **Tweak-and-rerun tool variants** — patch a tool's name/description/schema
     in config and rerun the same task.
  3. **Multi-model A/B** — run the same task+tools across several models.
  4. **Usage metrics** — aggregate stats across the run.
- Safe by default: no arbitrary shell on the real machine unless opted in.

**Non-goals**
- Not a production agent, not a chatbot UI, not a framework.
- No streaming UI, no persistence beyond trace files, no multi-user.
- No web browser automation in v1 (HTTP GET only).

## 3. Provider & runtime

- **Language:** Python (3.11+).
- **Provider:** OpenRouter, via the `openai` SDK pointed at
  `base_url="https://openrouter.ai/api/v1"` with `OPENROUTER_API_KEY`.
  OpenRouter is OpenAI-compatible, so swapping models is a string change —
  this is what makes multi-model A/B trivial.
- Dependencies kept minimal: `openai`, `pyyaml`, `pytest` (dev). No agent
  framework.

## 4. File layout

```
toolbench/
  __init__.py
  agent.py          # run_agent() — the agent loop (~60 lines)
  tools.py          # @tool decorator, schema derivation, ToolRegistry
  client.py         # thin OpenRouter wrapper around the openai SDK
  trace.py          # Trace / Turn / ToolCall dataclasses + JSONL writer
  metrics.py        # aggregate metrics from a set of traces
  experiment.py     # load YAML, expand model x variant x repeat matrix, run, compare
  cli.py            # argparse entrypoint: `run <config>` subcommand
  __main__.py       # enables `python -m toolbench ...` -> cli.main()
  builtins/
    __init__.py
    calculator.py   # safe arithmetic eval
    http_fetch.py   # HTTP GET only, SSRF-guarded
    files.py        # read/write confined to workspace/
    shell.py        # opt-in, guarded, OFF by default
experiments/
  example.yaml      # a runnable sample experiment
runs/               # JSONL traces (gitignored)
workspace/          # sandbox for file tools (gitignored)
tests/
  test_tools.py
  test_agent.py
  test_experiment.py
  test_metrics.py
  test_safety.py
pyproject.toml
README.md
.env.example        # OPENROUTER_API_KEY=
.gitignore
```

## 5. Tool interface

A tool is a plain function with type hints and a docstring, wrapped in `@tool`:

```python
from toolbench.tools import tool

@tool
def calculator(expression: str) -> str:
    """Evaluate a basic arithmetic expression.

    Args:
        expression: A math expression like "2 * (3 + 4)".
    """
    return str(_safe_eval(expression))
```

`@tool` derives the OpenAI/OpenRouter tool schema from the signature:
- function name <- function `__name__`
- `description` <- docstring summary (text before `Args:`)
- `parameters` <- JSON Schema from type hints; per-arg `description` parsed
  from the `Args:` section; `required` = args without a default.
- Supported param types: `str`, `int`, `float`, `bool`, `list`, and
  `Optional[...]`. Unsupported annotations raise at decoration time (fail fast).

The return value (string, or any JSON-serializable value -> stringified) is fed
back to the model as the tool result.

`ToolRegistry` holds tools by name and executes a tool call, catching
exceptions and returning `ERROR: <msg>`. You pass a list of tool functions to
`run_agent`, or name them in an experiment config (resolved against the
builtins plus any registered tools).

**Variant overrides:** at experiment time a tool's `name` / `description` /
parameter descriptions can be patched without touching code. The registry
produces a per-variant *view* of the schema; the underlying function is
unchanged. This is the tweak-and-rerun mechanism.

## 6. Agent loop (data flow)

```python
def run_agent(task, tools, model, *, system=None, max_turns=12, client=None) -> Trace:
    registry = ToolRegistry(tools)
    messages = [system_msg(system), user_msg(task)]
    trace = Trace(model=model, task=task)
    for i in range(max_turns):
        resp = client.chat(model, messages, tools=registry.schemas())   # timed
        trace.record_turn(resp)            # content, tool_calls, usage, latency
        messages.append(resp.assistant_message())
        if not resp.tool_calls:
            return trace.finish(answer=resp.content)
        for call in resp.tool_calls:
            result = registry.execute(call)            # exception -> "ERROR: ..."
            messages.append(tool_result_msg(call.id, result))
    return trace.finish(hit_max_turns=True)
```

Flow: `task -> messages -> OpenRouter -> tool_calls -> execute -> results ->
back to model -> ... -> final answer`. `client` is injectable so tests use a
fake.

## 7. Experiment config

```yaml
name: calc-vs-fetch
task: "What is 17% of 2,340, and what is the <title> of example.com?"
system: "You are precise. Use tools when needed."
max_turns: 8
models:
  - openai/gpt-4o-mini
  - google/gemini-2.0-flash
  - x-ai/grok-2
tools: [calculator, http_fetch]
variants:
  - name: baseline
  - name: terse-desc
    overrides:
      calculator: { description: "Math." }
repeats: 1
```

The runner expands the matrix = `models x variants x repeats`. Each cell runs
`run_agent` once and writes one JSONL trace to
`runs/<exp>/<model>__<variant>__<i>.jsonl` (model slug sanitized for the
filename). After all cells, it writes `runs/<exp>/summary.json` and prints a
comparison table (rows = cells; columns = turns, tokens, est. cost, tool calls,
errors, latency).

## 8. Trace format

JSONL, one record per line:

```json
{"type":"meta","experiment":"calc-vs-fetch","model":"openai/gpt-4o-mini","variant":"baseline","repeat":0,"task":"..."}
{"type":"turn","i":0,"content":"<model text/reasoning>","tool_calls":[{"name":"calculator","args":{"expression":"0.17*2340"},"result":"397.8","ok":true,"ms":2}],"usage":{"prompt":812,"completion":40}}
{"type":"final","answer":"...","turns":2,"total_tokens":1690,"ok":true,"hit_max_turns":false}
```

## 9. Metrics

Per cell, aggregated into `summary.json` and the printed table:
- turns to completion
- total tokens (prompt + completion); estimated cost if OpenRouter returns
  pricing in `usage` (optional, omitted if absent)
- tool-call counts per tool; success vs failure counts
- wall-clock latency (sum of per-turn `ms`)
- completed vs hit `max_turns`

Across repeats, metrics are reported with mean (and min/max when `repeats > 1`).

## 10. Error handling

- **Tool exception** -> caught by the registry, returned to the model as
  `ERROR: <msg>`, logged `ok:false`. The agent can recover; recovery behavior
  is itself measurable.
- **API error** (rate limit, invalid model id, network) -> retry with
  exponential backoff (default 3 attempts). On final failure, mark the *cell*
  failed in `summary.json` and continue the matrix. One bad cell never kills
  the run.
- **max_turns guard** prevents infinite tool loops; a cell that hits it is
  recorded `hit_max_turns:true`.

## 11. Safety (built-in tools)

- `calculator`: `_safe_eval` over an AST allowlist (numbers + `+ - * / ** ()`
  and unary minus). No `eval()` of arbitrary Python.
- `http_fetch`: GET only; timeout; response size cap; **SSRF guard** — resolve
  host and refuse localhost / link-local / private (RFC1918) / reserved ranges.
- `files`: `read_file` / `write_file` confined to `workspace/` via `realpath`
  prefix check; refuse path traversal.
- `shell`: a `run_shell` tool exists but is **commented out / not registered by
  default**, with a one-line risk note. Opt-in only.

## 12. Testing (pytest, offline)

A **fake client** returns scripted `tool_calls` then a final answer, so the
loop is testable with no API key and no network.

- `test_tools.py` — schema derivation from signatures; unsupported-type failure;
  variant override view.
- `test_agent.py` — loop runs tools and finishes; tool error is fed back and the
  agent recovers; max_turns guard.
- `test_experiment.py` — matrix expansion (models x variants x repeats); trace
  filenames.
- `test_metrics.py` — aggregation from sample traces.
- `test_safety.py` — calculator rejects non-arithmetic; file tool rejects
  traversal; http_fetch rejects private IPs.

## 13. Open questions / deferred

- Cost estimation depends on OpenRouter returning pricing; if not present in
  `usage`, cost is omitted (not hardcoded). Revisit if needed.
- A web UI for browsing traces is out of scope for v1 (traces are plain JSONL,
  greppable).
- Concurrency: v1 runs cells sequentially for simplicity and rate-limit
  friendliness. Parallel cells are a possible later optimization.

## 14. Success criteria

1. `python -m toolbench run experiments/example.yaml` runs the matrix and
   prints a comparison table (given a valid `OPENROUTER_API_KEY`).
2. `pytest` passes fully offline.
3. Adding a new tool is a single decorated function, no other wiring.
4. Changing a tool's description in the config and rerunning produces a visibly
   different trace/metric without any code change.
