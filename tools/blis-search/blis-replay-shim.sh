#!/usr/bin/env bash
# Lets tools/blis-search drive `blis replay` without patching evaluator.py.
#
# evaluator.py:30 hardcodes `[blis_binary, "run"]`, so point search.py at this
# script with --blis-binary. It does exactly two things:
#   1. rewrites the subcommand `run` -> `replay`
#   2. drops the four flags that exist on `run` but not on `replay`
#
# Everything else passes through, including --metrics-path (symmetric across
# run/replay, #1583) and the deprecated --max-num-running-reqs /
# --max-num-scheduled-tokens aliases, which replay still accepts.
#
# The trace, --max-model-len, and the closed-loop session settings are supplied
# by the search space YAML as single-valued parameters, NOT here — keeping one
# source of truth. See agentic-throughput-8.yaml `parameters`.
#
# Run from the repo root: `blis replay` resolves --defaults-filepath from cwd,
# so ./defaults.yaml must exist.
#
#   BLIS_BIN   path to the blis binary (default ./blis)
set -euo pipefail

BLIS_BIN="${BLIS_BIN:-./blis}"

[[ "${1:-}" == "run" ]] && shift   # swallow the hardcoded subcommand

# Strip run-only flags. Each takes exactly one value.
args=()
while (( $# )); do
  case "$1" in
    --workload|--rate|--num-requests|--workload-spec) shift 2 ;;
    *) args+=("$1"); shift ;;
  esac
done

exec "$BLIS_BIN" replay "${args[@]}"
