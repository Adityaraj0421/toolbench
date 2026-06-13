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
