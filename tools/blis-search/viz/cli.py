#!/usr/bin/env python3
"""Turn one blis-search results JSON into a report directory (design §8).

    .venv/bin/python tools/blis-search/viz/cli.py results.json [options]

Orchestration only: load, analyse, render, write. Every choice the outputs make
was made in analysis.py; the only decisions here are flag defaults and where the
files land.
"""
import argparse
import json
import os
import sys
from typing import List, Optional

import analysis
import commands
import data
import render_html
import render_md
import render_mpl

def default_out_dir(results_path: str) -> str:
    """`search_output.json` -> `search_output-viz/`, beside the results file."""
    base = os.path.basename(results_path)
    stem = base[: -len(".json")] if base.endswith(".json") else base
    return os.path.join(os.path.dirname(os.path.abspath(results_path)),
                        stem + "-viz")


def positive_int(text: str) -> int:
    """A --max-points of 0 or less is meaningless, and silently ignoring the flag
    the user typed would be a silent failure."""
    value = int(text)
    if value < 1:
        raise argparse.ArgumentTypeError(
            "must be 1 or more (a cap of %s would drop the whole cloud)" % value)
    return value


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="blis-search-viz",
        description="Visualise a blis-search results JSON (§8).")
    parser.add_argument("results", help="results JSON from search.py")
    parser.add_argument("--out-dir", default=None,
                        help="output directory (default: <results-basename>-viz/)")
    parser.add_argument("--space", default=None,
                        help="space YAML, for results files predating param_flags")
    parser.add_argument("--x", default=None, help="override the x axis metric")
    parser.add_argument("--y", default=None, help="override the y axis metric")
    parser.add_argument("--color", default=None,
                        help="override the color dimension (metric or knob)")
    parser.add_argument("--max-points", type=positive_int, default=None,
                        help="grid-decimate the dominated cloud; the front is never "
                             "sampled")
    group = parser.add_mutually_exclusive_group()
    group.add_argument("--png-only", action="store_true", help="skip index.html")
    group.add_argument("--html-only", action="store_true", help="skip the PNGs")
    return parser


def prepare_out_dir(path: str) -> None:
    """An existing file is a hard error; an existing directory is reused (§9).

    Raises ValueError so main() reports it the same way as every other error:
    one line on stderr, exit 2.
    """
    if os.path.isfile(path):
        raise ValueError(f"--out-dir {path} is an existing file, not a directory")
    os.makedirs(path, exist_ok=True)


def generate(args) -> List[str]:
    """Write every requested output and return the paths, in written order."""
    try:
        dataset = data.load(args.results)
    except json.JSONDecodeError as exc:
        # Name the file: the bare decoder message gives a line and column with no
        # indication of what was being read.
        raise ValueError(f"{args.results} is not valid JSON: {exc}") from exc
    flags = commands.resolve_param_flags(dataset, args.space)
    if flags.warning:
        print(f"warning: {flags.warning}", file=sys.stderr)
    if flags.note:
        print(f"warning: no deploy commands — {flags.note}", file=sys.stderr)

    an = analysis.analyse(dataset, args.x, args.y, args.color, args.max_points)
    for banner in an.banners:
        print(f"warning: {banner}", file=sys.stderr)

    out_dir = args.out_dir or default_out_dir(args.results)
    prepare_out_dir(out_dir)
    written: List[str] = []

    if not args.html_only:
        pareto = os.path.join(out_dir, "pareto.png")
        render_mpl.pareto(an, pareto)
        written.append(pareto)
        knobs = os.path.join(out_dir, "knobs.png")
        render_mpl.knobs(an, knobs)
        written.append(knobs)
        # §3: parallel coordinates are emitted at 3+ objectives — the hero form at
        # 4+, a supporting figure at 3. Below that there is nothing to draw.
        if len(dataset.objectives) >= 3:
            parallel = os.path.join(out_dir, "parallel.png")
            render_mpl.parallel(an, parallel)
            written.append(parallel)

    picks = os.path.join(out_dir, "picks.md")
    with open(picks, "w") as handle:
        handle.write(render_md.document(an, flags))
    written.append(picks)

    if not args.png_only:
        page = os.path.join(out_dir, "index.html")
        with open(page, "w") as handle:
            handle.write(render_html.document(an, flags))
        written.append(page)

    return written


def main(argv: Optional[List[str]] = None) -> int:
    args = build_parser().parse_args(argv)
    try:
        written = generate(args)
    except (data.SchemaError, commands.CommandError, ValueError, OSError) as exc:
        # OSError covers the whole family a user can trigger from the command
        # line: a missing results file, an unreadable --space, an out-dir on a
        # read-only filesystem, a full disk. ValueError covers json.JSONDecodeError
        # (its subclass) and the axis/colour override errors.
        print(f"error: {exc}", file=sys.stderr)
        return 2
    for path in written:
        print(f"wrote {path}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
