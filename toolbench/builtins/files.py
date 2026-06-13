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
