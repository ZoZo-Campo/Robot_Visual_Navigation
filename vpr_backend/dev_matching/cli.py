from __future__ import annotations

import argparse

from dev_matching.config import load_config
from dev_matching.pipeline import VPRExperiment
from dev_matching.utils.logging import configure_logging


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="dev-matching", description="Visual place recognition benchmark engine")
    parser.add_argument("config", help="Path to YAML configuration")
    parser.add_argument("--log-level", default="INFO", choices=("DEBUG", "INFO", "WARNING", "ERROR"))
    return parser


def main() -> int:
    args = build_parser().parse_args()
    configure_logging(args.log_level)
    results = VPRExperiment(load_config(args.config)).run_all()
    for result in results:
        print(f"{result.pair.robot_name} x {result.pair.streetview_name}: {result.scores.shape} -> {result.output_dir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
