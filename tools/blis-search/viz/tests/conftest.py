"""Shared fixtures and path setup for the blis-search viz tests."""
import json
import os
import sys

import pytest

TESTS_DIR = os.path.dirname(os.path.abspath(__file__))
VIZ_DIR = os.path.dirname(TESTS_DIR)
FIXTURES = os.path.join(TESTS_DIR, "fixtures")

# viz modules use flat imports, exactly as cli.py does when run as a script.
if VIZ_DIR not in sys.path:
    sys.path.insert(0, VIZ_DIR)


def fixture_path(name):
    """Absolute path to a committed fixture."""
    return os.path.join(FIXTURES, name)


def load_raw(name):
    """Parsed JSON for a committed fixture, straight from disk."""
    with open(fixture_path(name)) as f:
        return json.load(f)


#: Every fixture that data.load() must accept, for the "renders without exception" sweeps.
VALID_FIXTURES = [
    "two_obj_slo.json",
    "front_only.json",
    "three_obj.json",
    "four_obj.json",
    "single_objective.json",
    "single_member_front.json",
    "empty_slo_feasible.json",
    "dirty_metrics.json",
    "constant_objective.json",
    "single_gpu_count.json",
    "no_gpu_dimension.json",
]

#: real_subsample.json is only present when make_fixtures.py ran with --from-real.
REAL_FIXTURE = "real_subsample.json"
has_real = pytest.mark.skipif(
    not os.path.exists(os.path.join(FIXTURES, REAL_FIXTURE)),
    reason="real_subsample.json not generated (run make_fixtures.py --from-real)",
)


@pytest.fixture(params=VALID_FIXTURES)
def any_fixture(request):
    """Parametrised over every fixture the tool must accept."""
    return fixture_path(request.param)
