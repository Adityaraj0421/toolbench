import pytest
from toolbench.builtins.calculator import calculator
from toolbench.builtins.http_fetch import _is_safe_url
import toolbench.builtins.files as F
from toolbench.builtins import resolve_tools, BUILTINS


def test_calculator_basic():
    assert calculator("2 * (3 + 4)") == "14"


def test_calculator_rejects_code():
    with pytest.raises(Exception):
        calculator("__import__('os').system('echo hi')")


def test_ssrf_guard_blocks_private_and_nonhttp():
    assert _is_safe_url("http://127.0.0.1") is False
    assert _is_safe_url("http://169.254.169.254/latest/meta-data") is False
    assert _is_safe_url("ftp://example.com") is False


def test_files_confined_to_workspace(tmp_path, monkeypatch):
    monkeypatch.setattr(F, "WORKSPACE", tmp_path)
    assert "wrote" in F.write_file("a.txt", "hello")
    assert F.read_file("a.txt") == "hello"
    with pytest.raises(ValueError):
        F.write_file("../escape.txt", "x")


def test_resolve_tools_and_shell_not_registered():
    assert resolve_tools(["calculator"])[0] is calculator
    assert "run_shell" not in BUILTINS
    with pytest.raises(KeyError):
        resolve_tools(["does_not_exist"])
