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
