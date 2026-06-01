"""Render an HTML report from the latest LSL benchmark results."""
from __future__ import annotations

import argparse

from lsl.report import write_html_report


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--results-root", type=str, default="results")
    parser.add_argument("--output", type=str, default="results/lsl_report.html")
    parser.add_argument("--title", type=str, default="LSL Benchmark Report")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    output = write_html_report(args.output, results_root=args.results_root, title=args.title)
    print(f"Wrote HTML report: {output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

