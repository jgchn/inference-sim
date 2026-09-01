"""search.py must emit param_flags and space_file — tier 1 of design §7.

These two keys are what make a results file self-describing, so this is the
visualizer's contract test on its own input, not a search test.
"""
import json
import os
import subprocess
import sys

import commands
import data
from conftest import TESTS_DIR, VIZ_DIR

SEARCH_DIR = os.path.dirname(VIZ_DIR)
SEARCH_PY = os.path.join(SEARCH_DIR, "search.py")
STUB = os.path.join(TESTS_DIR, "stub_blis")
SPACE = os.path.join(SEARCH_DIR, "full-stack.yaml")


def run_search(tmp_path):
    out = tmp_path / "out.json"
    subprocess.run(
        [sys.executable, SEARCH_PY,
         "--model", "Qwen/Qwen3-32B", "--hardware", "H100",
         "--workload", "chatbot", "--rate", "50",
         "--budget", "6", "--workers", "2",
         "--blis-binary", STUB, "--search-space", SPACE,
         "--objectives", "responses_per_sec:maximize,ttft_p99_ms:minimize",
         "--output-all", "--output", str(out)],
        cwd=SEARCH_DIR, check=True, capture_output=True, text=True)
    with open(out) as f:
        return json.load(f)


def test_search_output_carries_param_flags_and_space_file(tmp_path):
    raw = run_search(tmp_path)
    assert raw["space_file"] == "full-stack.yaml"      # basename, not a full path
    assert raw["param_flags"]["replicas"] == "--num-instances"
    assert raw["param_flags"]["scheduler"] == "--scheduler"


def test_a_fresh_search_output_yields_a_command_with_no_extra_arguments(tmp_path):
    """The whole point of the change: tier 1 needs no --space."""
    raw = run_search(tmp_path)
    path = tmp_path / "out.json"
    ds = data.load(str(path))
    src = commands.resolve_param_flags(ds)
    assert src.origin == "results-file"
    cmd = commands.deploy_command(ds, ds.front_indices()[0], src.flags)
    assert cmd[:2] == ["./blis", "run"]
    assert "--num-instances" in cmd


def test_the_new_keys_do_not_disturb_the_existing_ones(tmp_path):
    raw = run_search(tmp_path)
    for key in ("pareto_front", "objectives", "spec", "search_stats", "all_results"):
        assert key in raw, key
