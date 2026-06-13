import pytest
from toolbench.cli import build_parser


def test_parser_run_subcommand():
    parser = build_parser()
    args = parser.parse_args(["run", "experiments/example.yaml", "--out", "runs"])
    assert args.cmd == "run"
    assert args.config == "experiments/example.yaml"
    assert args.out == "runs"


def test_parser_requires_subcommand():
    parser = build_parser()
    with pytest.raises(SystemExit):
        parser.parse_args([])
