from __future__ import annotations

import argparse
import sys

from .client import OpenRouterClient
from .experiment import format_table, load_experiment, run_experiment


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="toolbench")
    sub = parser.add_subparsers(dest="cmd", required=True)
    run_p = sub.add_parser("run", help="run an experiment config")
    run_p.add_argument("config", help="path to a YAML experiment config")
    run_p.add_argument("--out", default="runs", help="output dir for traces")
    return parser


def main(argv=None) -> int:
    args = build_parser().parse_args(argv)
    if args.cmd == "run":
        config = load_experiment(args.config)
        client = OpenRouterClient(max_output_tokens=config.max_output_tokens)
        summaries = run_experiment(config, client, out_dir=args.out)
        print(format_table(summaries))
    return 0


if __name__ == "__main__":
    sys.exit(main())
