"""commands.py: the three-tier flag resolution and the command it produces."""
import json

import pytest

import commands
import data
from conftest import fixture_path, load_raw

#: The space file that produced the synthetic fixtures' param_flags map.
SPACE = "full-stack.yaml"


def space_path():
    """tools/blis-search/full-stack.yaml, found from the tests directory."""
    import os
    from conftest import VIZ_DIR
    return os.path.join(os.path.dirname(VIZ_DIR), SPACE)


def test_tier_1_uses_the_flag_map_the_results_file_carries():
    ds = data.load(fixture_path("two_obj_slo.json"))
    src = commands.resolve_param_flags(ds)
    assert src.origin == "results-file"
    assert src.note == ""
    assert src.warning == ""
    assert src.flags["replicas"] == "--num-instances"


def test_tier_2_reconstructs_the_map_from_a_space_file():
    """front_only.json predates the search.py change, like every real old file."""
    ds = data.load(fixture_path("front_only.json"))
    assert ds.param_flags is None
    src = commands.resolve_param_flags(ds, space_path())
    assert src.origin == "space-file"
    assert src.flags["replicas"] == "--num-instances"


def test_tier_3_refuses_to_guess_and_says_why():
    ds = data.load(fixture_path("front_only.json"))
    src = commands.resolve_param_flags(ds)
    assert src.origin == "unavailable"
    assert src.flags is None
    assert src.note == commands.NO_FLAGS_NOTE
    assert commands.command_text(commands.deploy_command(ds, 0, src.flags)) \
        == commands.NO_FLAGS_NOTE


def test_an_embedded_map_wins_over_space_but_says_it_ignored_it():
    """D14: precedence is not a licence to drop a user flag silently."""
    ds = data.load(fixture_path("two_obj_slo.json"))
    src = commands.resolve_param_flags(ds, space_path())
    assert src.origin == "results-file"
    assert "ignored" in src.warning
    assert "two_obj_slo.json" in src.warning


def test_an_unreadable_space_is_a_named_error_not_a_traceback():
    ds = data.load(fixture_path("front_only.json"))
    with pytest.raises(commands.CommandError) as excinfo:
        commands.resolve_param_flags(ds, "/nonexistent/space.yaml")
    assert "/nonexistent/space.yaml" in str(excinfo.value)


def test_fixed_params_round_trip_from_spec():
    """search.py stored fixed_params with the leading dashes stripped."""
    ds = data.load(fixture_path("two_obj_slo.json"))
    fp = commands.fixed_params(ds)
    assert fp["--model"] == "Qwen/Qwen3-32B"
    assert fp["--num-requests"] == "2000"
    assert all(k.startswith("--") for k in fp)


def test_the_command_is_runnable_and_writes_no_metrics_file():
    ds = data.load(fixture_path("two_obj_slo.json"))
    src = commands.resolve_param_flags(ds)
    idx = ds.front_indices()[0]
    cmd = commands.deploy_command(ds, idx, src.flags)
    assert cmd[:2] == ["./blis", "run"]
    assert "--metrics-path" not in cmd
    # Every knob the config sets that has a flag and a non-false value appears.
    assert "--num-instances" in cmd
    text = commands.command_text(cmd)
    assert text.startswith("./blis run ")


def test_the_command_does_not_depend_on_flag_map_ordering():
    """D15: a reordered space YAML must not change the printed command."""
    ds = data.load(fixture_path("two_obj_slo.json"))
    flags = commands.resolve_param_flags(ds).flags
    idx = ds.front_indices()[0]
    forward = commands.deploy_command(ds, idx, flags)
    backward = commands.deploy_command(ds, idx, dict(reversed(list(flags.items()))))
    assert forward == backward


def test_flow_control_off_suppresses_its_sub_params():
    """The reason guessing flags is unacceptable: emission is conditional."""
    ds = data.load(fixture_path("two_obj_slo.json"))
    flags = commands.resolve_param_flags(ds).flags
    for idx in range(len(ds.points)):
        cfg = ds.points[idx].config
        if cfg.get("flow_control") is True:
            continue
        cmd = commands.deploy_command(ds, idx, flags)
        assert "--flow-control" not in cmd
        assert "--queue-depth-threshold" not in cmd
        break
    else:
        pytest.skip("fixture has no flow-control-off config")


def test_a_changed_build_command_contract_raises_instead_of_slipping_through(
        monkeypatch):
    """The guard must survive `python -O`, which strips asserts. If the upstream
    command builder stops ending with the metrics path, refusing loudly is the
    only safe behaviour: silently emitting the tail would hand the user a command
    that writes a metrics file they never asked for."""
    ds = data.load(fixture_path("two_obj_slo.json"))
    flags = commands.resolve_param_flags(ds).flags

    monkeypatch.setattr(commands, "build_command",
                        lambda *a, **k: ["./blis", "run", "--tp", "8"])
    with pytest.raises(commands.CommandError) as excinfo:
        commands.deploy_command(ds, ds.front_indices()[0], flags)
    assert "no longer ends with the metrics path" in str(excinfo.value)
