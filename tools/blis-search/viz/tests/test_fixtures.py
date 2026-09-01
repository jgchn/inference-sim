"""The committed fixtures must keep the properties later tests rely on."""
import json
import os
import subprocess
import sys

import pytest

from conftest import FIXTURES, TESTS_DIR, VALID_FIXTURES, fixture_path, load_raw

EXPECTED = {
    # name: (front_size, all_results_size, n_objectives, has_param_flags)
    "two_obj_slo.json": (6, 60, 2, True),
    "front_only.json": (6, 0, 2, False),
    "three_obj.json": (17, 60, 3, True),
    "four_obj.json": (19, 60, 4, True),
    "single_objective.json": (1, 14, 1, True),
    "single_member_front.json": (1, 14, 2, True),
    "empty_slo_feasible.json": (4, 14, 2, True),
    "dirty_metrics.json": (4, 14, 2, True),
    "constant_objective.json": (1, 14, 2, True),
    "single_gpu_count.json": (5, 12, 2, True),
    "no_gpu_dimension.json": (4, 14, 2, True),
}


@pytest.mark.parametrize("name", VALID_FIXTURES)
def test_fixture_shape(name):
    raw = load_raw(name)
    front, allr, nobj, flags = EXPECTED[name]
    assert len(raw["pareto_front"]) == front
    assert len(raw.get("all_results", [])) == allr
    assert len(raw["objectives"]) == nobj
    assert ("param_flags" in raw) is flags
    for key in ("spec", "search_stats", "objectives", "pareto_front"):
        assert key in raw, "%s missing required key %s" % (name, key)


def test_two_obj_slo_has_multiple_gpu_tiers():
    """The knee, color legend, and knob-attribution tests all need a varied cloud."""
    raw = load_raw("two_obj_slo.json")
    tiers = {e["metrics"]["gpus_used"] for e in raw["all_results"]}
    assert len(tiers) >= 4
    assert len(raw["slo_feasible"]) > 0


def test_empty_slo_feasible_is_empty():
    assert load_raw("empty_slo_feasible.json")["slo_feasible"] == []


def test_single_gpu_count_has_exactly_one_tier():
    raw = load_raw("single_gpu_count.json")
    tiers = {e["metrics"]["gpus_used"] for e in raw["all_results"]}
    assert tiers == {8}


def test_no_gpu_dimension_derivable():
    raw = load_raw("no_gpu_dimension.json")
    for entry in raw["all_results"]:
        assert "gpus_used" not in entry["metrics"]
        assert "tp" not in entry["config"]
        assert "replicas" not in entry["config"]


def test_dirty_metrics_carries_missing_nan_and_inf():
    # json.load turns the JSON literals NaN / Infinity back into floats.
    raw = load_raw("dirty_metrics.json")
    allr = raw["all_results"]
    missing = [e for e in allr if "ttft_p99_ms" not in e["metrics"]]
    nans = [e for e in allr if e["metrics"].get("ttft_p99_ms") != e["metrics"].get("ttft_p99_ms")]
    infs = [e for e in allr if e["metrics"].get("responses_per_sec") == float("inf")]
    assert len(missing) == 1 and len(nans) == 1 and len(infs) == 1


def test_constant_objective_has_zero_spread():
    raw = load_raw("constant_objective.json")
    assert {e["metrics"]["responses_per_sec"] for e in raw["all_results"]} == {42.0}


def test_foreign_schema_lacks_our_required_keys():
    raw = load_raw("foreign_schema.json")
    assert "objectives" not in raw
    assert "search_stats" not in raw


def test_generator_is_deterministic(tmp_path):
    """Regenerating into a scratch dir must reproduce the committed bytes."""
    script = os.path.join(TESTS_DIR, "make_fixtures.py")
    scratch = tmp_path / "tests"
    scratch.mkdir()
    (scratch / "make_fixtures.py").write_bytes(open(script, "rb").read())
    subprocess.run([sys.executable, str(scratch / "make_fixtures.py")], check=True,
                   capture_output=True)
    for name in VALID_FIXTURES:
        regenerated = (scratch / "fixtures" / name).read_bytes()
        committed = open(fixture_path(name), "rb").read()
        assert regenerated == committed, "%s drifted; rerun make_fixtures.py" % name
