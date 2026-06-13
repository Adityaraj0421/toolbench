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
