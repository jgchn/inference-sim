"""data.py: schema validation, normalisation, and the columnar encoding."""
import json
import pytest

import data
from conftest import VALID_FIXTURES, fixture_path, load_raw


@pytest.mark.parametrize("name", VALID_FIXTURES)
def test_every_valid_fixture_passes_validation(name):
    data.validate_schema(load_raw(name))   # must not raise


def test_foreign_schema_is_rejected_by_name():
    raw = load_raw("foreign_schema.json")
    with pytest.raises(data.SchemaError) as exc:
        data.validate_schema(raw)
    msg = str(exc.value)
    # The error must name what was expected and what was actually found (§9).
    assert "objectives" in msg and "search_stats" in msg
    assert "recommended_config" in msg or "best_objectives" in msg


def test_non_object_input_is_rejected():
    with pytest.raises(data.SchemaError):
        data.validate_schema([1, 2, 3])


def test_bad_objective_direction_is_rejected():
    raw = load_raw("two_obj_slo.json")
    raw["objectives"][0]["direction"] = "sideways"
    with pytest.raises(data.SchemaError) as exc:
        data.validate_schema(raw)
    assert "sideways" in str(exc.value)


def test_objectives_must_be_non_empty():
    raw = load_raw("two_obj_slo.json")
    raw["objectives"] = []
    with pytest.raises(data.SchemaError):
        data.validate_schema(raw)


def test_front_entry_missing_config_is_rejected():
    raw = load_raw("two_obj_slo.json")
    del raw["pareto_front"][0]["config"]
    with pytest.raises(data.SchemaError) as exc:
        data.validate_schema(raw)
    assert "config" in str(exc.value)


def test_optional_keys_are_optional():
    """front_only.json has no all_results, slo_*, param_flags, or space_file."""
    data.validate_schema(load_raw("front_only.json"))


# ---------------------------------------------------------------- Dataset


def test_load_two_obj_slo_shape():
    ds = data.load(fixture_path("two_obj_slo.json"))
    assert ds.has_cloud is True
    assert len(ds.points) == 60
    assert [o.key for o in ds.objectives] == ["responses_per_sec", "ttft_p99_ms"]
    assert [o.direction for o in ds.objectives] == ["maximize", "minimize"]
    assert len(ds.front_indices()) == 6
    assert len(ds.cloud_indices()) == 54
    assert ds.context["model"] == "Qwen/Qwen3-32B"
    assert ds.stats["convergence_eval"] == 18
    assert len(ds.source_sha256) == 64
    assert ds.source_name == "two_obj_slo.json"


def test_point_indices_are_dense_and_source_ordered():
    ds = data.load(fixture_path("two_obj_slo.json"))
    assert [p.idx for p in ds.points] == list(range(len(ds.points)))


def test_front_only_mode_has_no_cloud():
    ds = data.load(fixture_path("front_only.json"))
    assert ds.has_cloud is False
    assert len(ds.points) == 6
    assert all(p.on_front for p in ds.points)
    assert ds.cloud_indices() == []
    assert ds.param_flags is None
    assert ds.space_file is None


def test_param_flags_and_space_file_are_read_when_present():
    ds = data.load(fixture_path("two_obj_slo.json"))
    assert ds.param_flags["replicas"] == "--num-instances"
    assert ds.space_file == "full-stack.yaml"


def test_param_flags_override_wins_over_file():
    ds = data.load(fixture_path("two_obj_slo.json"),
                   param_flags={"tp": "--tp"}, space_file="override.yaml")
    assert ds.param_flags == {"tp": "--tp"}
    assert ds.space_file == "override.yaml"


def test_front_members_are_marked_exactly_once():
    """Front matching by (config, metrics) identity must not double-mark (D7)."""
    raw = load_raw("two_obj_slo.json")
    # Duplicate one cloud entry so two all_results rows share a key.
    raw["all_results"].append(dict(raw["all_results"][0]))
    ds = data.build_dataset(raw, source_sha256="0" * 64, source_name="dup.json")
    assert len(ds.points) == 61
    assert len(ds.front_indices()) == len(raw["pareto_front"])


def test_duplicate_front_entry_marks_two_distinct_points():
    raw = load_raw("two_obj_slo.json")
    first = raw["pareto_front"][0]
    raw["all_results"].append({"config": dict(first["config"]),
                               "metrics": dict(first["metrics"])})
    raw["pareto_front"].append({"config": dict(first["config"]),
                                "metrics": dict(first["metrics"])})
    ds = data.build_dataset(raw, source_sha256="0" * 64, source_name="dup2.json")
    assert len(ds.front_indices()) == 7


def test_dirty_metrics_points_are_excluded_and_counted():
    """Missing / NaN / inf on an OBJECTIVE metric excludes the point (D2, §9)."""
    ds = data.load(fixture_path("dirty_metrics.json"))
    assert len(ds.points) == 11          # 14 minus the missing, the NaN, and the inf
    assert len(ds.excluded) == 3
    reasons = sorted(e["reason"] for e in ds.excluded)
    assert reasons == ["inf", "missing", "nan"]
    assert all(math_isfinite(ds.value(p, o)) for p in ds.points for o in ds.objectives)


def math_isfinite(x):
    import math
    return math.isfinite(x)


def test_non_objective_metric_may_be_missing():
    raw = load_raw("two_obj_slo.json")
    del raw["all_results"][0]["metrics"]["itl_p99_ms"]
    ds = data.build_dataset(raw, source_sha256="0" * 64, source_name="x.json")
    assert len(ds.points) == 60           # not excluded — itl_p99_ms is not an objective
    assert ds.points[0].metrics.get("itl_p99_ms") is None


def test_slo_ok_is_computed_for_cloud_points_too():
    """slo_feasible only covers the front; the band and filter need the cloud (D4)."""
    ds = data.load(fixture_path("two_obj_slo.json"))
    assert all(p.slo_ok is not None for p in ds.points)
    cloud_ok = [p for p in ds.cloud_indices() if ds.points[p].slo_ok]
    assert cloud_ok, "expected at least one SLO-feasible dominated point"
    for p in ds.points:
        assert p.slo_ok == data.satisfies_slo(p.metrics, ds.slo_constraints)


def test_slo_ok_is_none_without_constraints():
    ds = data.load(fixture_path("front_only.json"))
    assert ds.slo_constraints == []
    assert all(p.slo_ok is None for p in ds.points)


def test_slo_mismatch_is_reported_not_swallowed():
    raw = load_raw("two_obj_slo.json")
    raw["slo_feasible"] = []          # the searcher claims nothing is feasible
    ds = data.build_dataset(raw, source_sha256="0" * 64, source_name="x.json")
    assert ds.slo_mismatch, "a disagreement with the reported slo_feasible must surface"


def test_empty_slo_feasible_agrees_when_truly_empty():
    ds = data.load(fixture_path("empty_slo_feasible.json"))
    assert ds.slo_mismatch == []
    assert all(p.slo_ok is False for p in ds.points)


def test_knob_and_metric_names_are_sorted_and_complete():
    ds = data.load(fixture_path("two_obj_slo.json"))
    knobs = ds.knob_names()
    metrics = ds.metric_names()
    assert knobs == sorted(knobs) and metrics == sorted(metrics)
    assert "scorer_profile" in knobs        # present even though often null
    assert "gpus_used" in knobs and "gpus_used" in metrics   # the real collision (D1)
    assert "ttft_p99_ms" in metrics


def test_objective_arrow_is_empty_for_a_non_objective_axis():
    """--x / --y may name any metric; its "better" direction is unknown."""
    assert data.Objective("responses_per_sec", "maximize").arrow == "↑"
    assert data.Objective("ttft_p99_ms", "minimize").arrow == "↓"
    assert data.Objective("itl_p95_ms", "").arrow == ""


def test_value_orients_nothing_it_returns_raw_metric():
    ds = data.load(fixture_path("two_obj_slo.json"))
    p = ds.points[0]
    assert ds.value(p, ds.objectives[0]) == p.metrics["responses_per_sec"]


@pytest.mark.parametrize("name", VALID_FIXTURES)
def test_every_valid_fixture_loads(name):
    ds = data.load(fixture_path(name))
    assert ds.points, "%s produced no usable points" % name
    assert ds.objectives


# ------------------------------------------------------- columnar encoding


def flat_rows(ds, indices=None):
    """The reference flattening the encoding must reproduce exactly."""
    idxs = indices if indices is not None else [p.idx for p in ds.points]
    rows = []
    for i in idxs:
        p = ds.points[i]
        row = {}
        for k in ds.knob_names():
            row["c:" + k] = p.config.get(k)
        for m in ds.metric_names():
            row["m:" + m] = p.metrics.get(m)
        rows.append(row)
    return rows


@pytest.mark.parametrize("name", VALID_FIXTURES)
def test_columnar_round_trips_losslessly(name):
    ds = data.load(fixture_path(name))
    enc = data.encode_columnar(ds)
    assert data.decode_columnar(enc) == flat_rows(ds)


def test_columnar_respects_an_index_subset():
    ds = data.load(fixture_path("two_obj_slo.json"))
    subset = [0, 5, 9, 30]
    enc = data.encode_columnar(ds, indices=subset)
    assert enc["n"] == 4
    assert data.decode_columnar(enc) == flat_rows(ds, subset)


def test_field_names_are_namespaced_so_gpus_used_does_not_collide():
    ds = data.load(fixture_path("two_obj_slo.json"))
    enc = data.encode_columnar(ds)
    assert "c:gpus_used" in enc["f"] and "m:gpus_used" in enc["f"]


def test_categorical_values_are_interned_once():
    ds = data.load(fixture_path("two_obj_slo.json"))
    enc = data.encode_columnar(ds)
    schedulers = enc["s"]["c:scheduler"]
    assert sorted(v for v in schedulers if v is not None) == \
        sorted({p.config["scheduler"] for p in ds.points})
    # A string never appears in the row payload — only its index does.
    for row in enc["r"]:
        for cell in row:
            assert not isinstance(cell, str)


def test_null_is_an_interned_value_not_missing_data():
    ds = data.load(fixture_path("two_obj_slo.json"))
    enc = data.encode_columnar(ds)
    assert None in enc["s"]["c:scorer_profile"]


def test_front_and_slo_membership_travel_with_the_encoding():
    ds = data.load(fixture_path("two_obj_slo.json"))
    enc = data.encode_columnar(ds)
    assert enc["front"] == [i for i, p in enumerate(ds.points) if p.on_front]
    assert enc["slo"] == [i for i, p in enumerate(ds.points) if p.slo_ok]


def test_slo_membership_is_null_without_constraints():
    ds = data.load(fixture_path("front_only.json"))
    assert data.encode_columnar(ds)["slo"] is None


def test_encoding_is_key_sorted_for_byte_stable_output():
    ds = data.load(fixture_path("two_obj_slo.json"))
    enc = data.encode_columnar(ds)
    assert enc["f"] == sorted(enc["f"])
    assert list(enc["s"]) == sorted(enc["s"])
    dumped = json.dumps(enc, sort_keys=True)
    assert json.dumps(data.encode_columnar(ds), sort_keys=True) == dumped


def test_encoding_is_smaller_than_verbatim_embedding():
    ds = data.load(fixture_path("two_obj_slo.json"))
    verbatim = json.dumps([{"config": p.config, "metrics": p.metrics} for p in ds.points])
    columnar = json.dumps(data.encode_columnar(ds), sort_keys=True)
    assert len(columnar) < len(verbatim) * 0.6
