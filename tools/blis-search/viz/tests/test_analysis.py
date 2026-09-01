"""analysis.py: every decision the renderers consume, tested as a law."""
import copy
import itertools
import math

import pytest

import analysis
import data
from conftest import VALID_FIXTURES, fixture_path, load_raw


def brute_force_front(ds):
    """O(n^2) reference: a point is on the front iff nothing dominates it."""
    vals = [[ds.value(p, o) for o in ds.objectives] for p in ds.points]
    front = []
    for i, vi in enumerate(vals):
        if not any(analysis.dominates(vj, vi, ds.objectives)
                   for j, vj in enumerate(vals) if j != i):
            front.append(i)
    return front


@pytest.mark.parametrize("name", VALID_FIXTURES)
def test_recomputed_front_matches_brute_force(name):
    ds = data.load(fixture_path(name))
    assert analysis.recompute_front(ds) == brute_force_front(ds)


@pytest.mark.parametrize("name", VALID_FIXTURES)
def test_no_front_member_is_dominated_by_any_point(name):
    """The §11 law: front membership means nothing beats you everywhere."""
    ds = data.load(fixture_path(name))
    vals = [[ds.value(p, o) for o in ds.objectives] for p in ds.points]
    for i in analysis.recompute_front(ds):
        for j in range(len(vals)):
            if j == i:
                continue
            assert not analysis.dominates(vals[j], vals[i], ds.objectives)


def test_dominance_needs_strict_improvement_somewhere():
    objs = [data.Objective("a", "maximize"), data.Objective("b", "minimize")]
    assert analysis.dominates([2.0, 1.0], [1.0, 2.0], objs)
    assert not analysis.dominates([1.0, 1.0], [1.0, 1.0], objs)   # identical: no
    assert not analysis.dominates([2.0, 3.0], [1.0, 2.0], objs)   # better a, worse b


def test_front_check_agrees_on_a_healthy_file():
    ds = data.load(fixture_path("two_obj_slo.json"))
    check = analysis.check_front(ds)
    assert check.verifiable is True
    assert check.agrees is True
    assert check.reported_only == [] and check.recomputed_only == []
    assert check.message is None


def test_front_check_is_unverifiable_in_front_only_mode():
    ds = data.load(fixture_path("front_only.json"))
    check = analysis.check_front(ds)
    assert check.verifiable is False
    assert check.message is not None
    assert "cannot be verified" in check.message


def test_front_check_reports_a_dominated_reported_member():
    """A searcher bug: a reported front member that a cloud point dominates."""
    raw = load_raw("two_obj_slo.json")
    # Promote a plainly dominated cloud entry into the reported front.
    worst = min(raw["all_results"],
                key=lambda e: e["metrics"]["responses_per_sec"] - e["metrics"]["ttft_p99_ms"])
    raw["pareto_front"].append({"config": worst["config"], "metrics": worst["metrics"]})
    ds = data.build_dataset(raw, source_sha256="0" * 64, source_name="bug.json")
    check = analysis.check_front(ds)
    assert check.agrees is False
    assert check.reported_only, "the wrongly-reported member must be named"
    assert check.message and "reported" in check.message


def test_front_check_reports_a_missed_member():
    raw = load_raw("two_obj_slo.json")
    dropped = raw["pareto_front"].pop(0)
    ds = data.build_dataset(raw, source_sha256="0" * 64, source_name="bug2.json")
    check = analysis.check_front(ds)
    assert check.agrees is False
    assert check.recomputed_only
    assert check.reported_only == []
    assert dropped["config"]["tp"] is not None      # sanity: we dropped a real member


def test_front_check_never_overwrites_reported_membership():
    """Disagreement is surfaced, not applied: on_front stays as the file said."""
    raw = load_raw("two_obj_slo.json")
    raw["pareto_front"].pop(0)
    ds = data.build_dataset(raw, source_sha256="0" * 64, source_name="bug3.json")
    before = ds.front_indices()
    analysis.check_front(ds)
    assert ds.front_indices() == before


def test_describe_config_is_short_and_stable():
    ds = data.load(fixture_path("two_obj_slo.json"))
    label = analysis.describe_config(ds, ds.front_indices()[0])
    assert "tp" in label
    assert len(label) < 40
    assert analysis.describe_config(ds, ds.front_indices()[0]) == label


# ------------------------------------------------------------------- axes


def test_two_objectives_map_in_declaration_order():
    ds = data.load(fixture_path("two_obj_slo.json"))
    axes = analysis.choose_axes(ds)
    assert axes.x.key == "responses_per_sec"
    assert axes.y.key == "ttft_p99_ms"
    assert axes.auto is True


def test_axis_labels_print_the_direction():
    ds = data.load(fixture_path("two_obj_slo.json"))
    axes = analysis.choose_axes(ds)
    assert axes.x_label == "responses_per_sec ↑ better"
    assert axes.y_label == "ttft_p99_ms ↓ better"


def test_orientation_puts_better_toward_the_lower_right():
    """The §11 law: 'better' is the same visual direction for both directions."""
    ds = data.load(fixture_path("two_obj_slo.json"))
    axes = analysis.choose_axes(ds)
    idxs = [p.idx for p in ds.points]
    fr = analysis.screen_fractions(ds, axes, idxs)

    best_x = max(idxs, key=lambda i: ds.points[i].metrics["responses_per_sec"])  # maximize
    best_y = min(idxs, key=lambda i: ds.points[i].metrics["ttft_p99_ms"])        # minimize
    assert fr[idxs.index(best_x)][0] == pytest.approx(1.0)   # rightmost
    assert fr[idxs.index(best_y)][1] == pytest.approx(1.0)   # lowest


def test_orientation_holds_when_the_axes_are_swapped_by_override():
    """Swapping which objective is x must not change which end means 'better'."""
    ds = data.load(fixture_path("two_obj_slo.json"))
    axes = analysis.choose_axes(ds, x_override="ttft_p99_ms", y_override="responses_per_sec")
    idxs = [p.idx for p in ds.points]
    fr = analysis.screen_fractions(ds, axes, idxs)
    best_x = min(idxs, key=lambda i: ds.points[i].metrics["ttft_p99_ms"])
    best_y = max(idxs, key=lambda i: ds.points[i].metrics["responses_per_sec"])
    assert fr[idxs.index(best_x)][0] == pytest.approx(1.0)
    assert fr[idxs.index(best_y)][1] == pytest.approx(1.0)


def test_invert_flags_match_the_orientation_rule():
    ds = data.load(fixture_path("two_obj_slo.json"))
    axes = analysis.choose_axes(ds)
    assert axes.x_invert is False    # maximize on x: natural
    assert axes.y_invert is False    # minimize on y: natural
    swapped = analysis.choose_axes(ds, x_override="ttft_p99_ms",
                                   y_override="responses_per_sec")
    assert swapped.x_invert is True  # minimize on x: invert so better is right
    assert swapped.y_invert is True  # maximize on y: invert so better is down


def test_three_objectives_pick_the_widest_spread_pair_and_say_so():
    ds = data.load(fixture_path("three_obj.json"))
    axes = analysis.choose_axes(ds)
    keys = {axes.x.key, axes.y.key}
    assert len(keys) == 2
    assert keys <= {o.key for o in ds.objectives}
    assert axes.x.key in axes.caption and axes.y.key in axes.caption
    # Declaration order within the chosen pair.
    order = [o.key for o in ds.objectives]
    assert order.index(axes.x.key) < order.index(axes.y.key)


def test_pair_selection_maximises_distinct_positions():
    ds = data.load(fixture_path("four_obj.json"))
    axes = analysis.choose_axes(ds)
    idxs = [p.idx for p in ds.points]

    def distinct(a, b):
        xs = analysis.normalise([ds.points[i].metrics[a.key] for i in idxs])
        ys = analysis.normalise([ds.points[i].metrics[b.key] for i in idxs])
        return len({(round(x, 9), round(y, 9)) for x, y in zip(xs, ys)})

    chosen = distinct(axes.x, axes.y)
    for a, b in itertools.combinations(ds.objectives, 2):
        assert distinct(a, b) <= chosen


def test_overrides_are_marked_not_auto_and_may_name_any_metric():
    ds = data.load(fixture_path("two_obj_slo.json"))
    axes = analysis.choose_axes(ds, x_override="tokens_per_sec", y_override="itl_p99_ms")
    assert axes.auto is False
    assert axes.x.key == "tokens_per_sec"
    # A non-objective metric has no known "better" direction, so no arrow and no invert.
    assert axes.x_label == "tokens_per_sec"
    assert axes.x_invert is False
    assert "not an objective" in axes.caption


def test_unknown_override_metric_is_a_hard_error_naming_the_metric():
    ds = data.load(fixture_path("two_obj_slo.json"))
    with pytest.raises(ValueError) as exc:
        analysis.choose_axes(ds, x_override="no_such_metric")
    assert "no_such_metric" in str(exc.value)


def test_constant_objective_normalises_without_dividing_by_zero():
    ds = data.load(fixture_path("constant_objective.json"))
    axes = analysis.choose_axes(ds)
    fr = analysis.screen_fractions(ds, axes, [p.idx for p in ds.points])
    assert all(math.isfinite(x) and math.isfinite(y) for x, y in fr)
    assert "zero spread" in axes.caption


def test_normalise_maps_all_equal_input_to_the_midpoint():
    assert analysis.normalise([3.0, 3.0, 3.0]) == [0.5, 0.5, 0.5]
    assert analysis.normalise([0.0, 5.0, 10.0]) == [0.0, 0.5, 1.0]


def test_single_objective_axes_use_the_objective_on_x():
    ds = data.load(fixture_path("single_objective.json"))
    axes = analysis.choose_axes(ds)
    assert axes.x.key == "responses_per_sec"
    assert axes.y is None
    assert "single objective" in axes.caption


@pytest.mark.parametrize("name", VALID_FIXTURES)
def test_choose_axes_never_raises_on_a_valid_fixture(name):
    ds = data.load(fixture_path(name))
    analysis.choose_axes(ds)


def _axis_synth(objectives, rows):
    """Minimal Dataset for axis-selection tests: every point is on the front, so
    pair selection sees all of them (D5 says selection runs over all valid points)."""
    entries = [{"config": {"tp": i + 1}, "metrics": m} for i, m in enumerate(rows)]
    raw = {"pareto_front": entries, "all_results": entries, "objectives": objectives,
           "spec": {"model": "synthetic"},
           "search_stats": {"total_evaluated": len(entries)}}
    return data.build_dataset(raw, source_sha256="0" * 64, source_name="synth.json")


THREE_MIN = [{"metric": "a", "direction": "minimize"},
             {"metric": "b", "direction": "minimize"},
             {"metric": "c", "direction": "minimize"}]


def test_pair_selection_breaks_a_position_tie_on_summed_spread():
    """All three pairs place 4 distinct positions, so the count cannot decide.
    Normalised spread then separates them: a and b are evenly spaced (stdev sum
    0.7454) while c bunches toward the extremes (0.7976 paired with either)."""
    ds = _axis_synth(THREE_MIN, [
        {"a": 0.0, "b": 0.0, "c": 0.0},
        {"a": 1.0, "b": 1.0, "c": 0.5},
        {"a": 2.0, "b": 2.0, "c": 2.5},
        {"a": 3.0, "b": 3.0, "c": 3.0}])
    by_key = {o.key: o for o in ds.objectives}
    counts = {p: analysis._distinct_positions(ds, by_key[p[0]], by_key[p[1]])
              for p in (("a", "b"), ("a", "c"), ("b", "c"))}
    assert set(counts.values()) == {4}, counts          # the tie this test needs
    spreads = {p: analysis._spread_sum(ds, by_key[p[0]], by_key[p[1]])
               for p in (("a", "b"), ("a", "c"), ("b", "c"))}
    assert spreads[("a", "c")] > spreads[("a", "b")]

    axes = analysis.choose_axes(ds)
    assert (axes.x.key, axes.y.key) == ("a", "c")       # widest spread wins
    assert "widest-spread pair" in axes.caption


def test_pair_selection_falls_back_to_declaration_order():
    """Here c is a permutation of a and b, so every pair ties on both distinct
    positions (4) and summed spread (0.7454). Declaration order must decide, and
    a/b are declared first."""
    ds = _axis_synth(THREE_MIN, [
        {"a": 0.0, "b": 0.0, "c": 0.0},
        {"a": 1.0, "b": 1.0, "c": 3.0},
        {"a": 2.0, "b": 2.0, "c": 1.0},
        {"a": 3.0, "b": 3.0, "c": 2.0}])
    by_key = {o.key: o for o in ds.objectives}
    pairs = (("a", "b"), ("a", "c"), ("b", "c"))
    assert len({analysis._distinct_positions(ds, by_key[p[0]], by_key[p[1]])
                for p in pairs}) == 1
    assert len({round(analysis._spread_sum(ds, by_key[p[0]], by_key[p[1]]), 9)
                for p in pairs}) == 1
    assert (analysis.choose_axes(ds).x.key, analysis.choose_axes(ds).y.key) == ("a", "b")


def test_a_constant_axis_lands_every_point_at_the_midpoint():
    """The renderer path, not just normalise(): a zero-spread axis must place all
    points at 0.5 rather than dividing by zero or collapsing to an edge."""
    ds = data.load(fixture_path("constant_objective.json"))
    axes = analysis.choose_axes(ds)
    xs = {p.metrics[axes.x.key] for p in ds.points}
    assert len(xs) == 1, "this fixture's x objective is the constant one"
    fractions = analysis.screen_fractions(ds, axes, list(range(len(ds.points))))
    assert all(fx == 0.5 for fx, _ in fractions)
    assert all(0.0 <= fy <= 1.0 for _, fy in fractions)
    assert "zero spread" in axes.caption


# ------------------------------------------------------------------- knee

MIN_MAX = [{"metric": "lat", "direction": "minimize"},
           {"metric": "rps", "direction": "maximize"}]


def synth(objectives, rows, front):
    """Minimal results object -> Dataset, for hand-built geometries.

    rows: [(config_dict, metrics_dict)]; front: indices into rows that the
    searcher reported as the front. Used by the knee, knob, picks, and
    decimation tests where a real fixture would obscure the point being made.
    """
    entries = [{"config": dict(c), "metrics": dict(m)} for c, m in rows]
    raw = {
        "pareto_front": [entries[i] for i in front],
        "all_results": entries,
        "objectives": objectives,
        "spec": {"model": "synthetic"},
        "search_stats": {"total_evaluated": len(entries)},
    }
    return data.build_dataset(raw, source_sha256="0" * 64, source_name="synth.json")


def test_knee_is_the_corner_of_an_L_shaped_front():
    # In cost space (both minimised) the front is (0,1), (0.1,0.1), (1,0): the
    # middle point is far off the chord joining the two extremes.
    rows = [
        ({"tp": 1}, {"lat": 10.0, "rps": 10.0}),     # cheapest latency, worst rps
        ({"tp": 2}, {"lat": 19.0, "rps": 91.0}),     # the corner
        ({"tp": 4}, {"lat": 100.0, "rps": 100.0}),   # best rps, worst latency
    ]
    ds = synth(MIN_MAX, rows, front=[0, 1, 2])
    k = analysis.knee(ds, analysis.choose_axes(ds))
    assert k.index == 1
    assert k.note == ""
    assert set(k.extremes) == {0, 2}


def test_a_collinear_front_falls_back_to_the_lowest_index_interior_member():
    # Every interior member is equally balanced on a straight front. There is no
    # distinguished knee, so the lowest-index interior member wins: deterministic,
    # still a front member, still not an extreme.
    rows = [
        ({"tp": 1}, {"lat": 10.0, "rps": 10.0}),
        ({"tp": 2}, {"lat": 55.0, "rps": 55.0}),
        ({"tp": 3}, {"lat": 70.0, "rps": 70.0}),
        ({"tp": 4}, {"lat": 100.0, "rps": 100.0}),
    ]
    ds = synth(MIN_MAX, rows, front=[0, 1, 2, 3])
    k = analysis.knee(ds, analysis.choose_axes(ds))
    assert k.index == 1
    assert k.index not in k.extremes


@pytest.mark.parametrize("name", VALID_FIXTURES)
def test_knee_is_a_front_member_and_never_an_extreme(name):
    """§11 law. Suppression is allowed; a knee outside the front is not."""
    ds = data.load(fixture_path(name))
    k = analysis.knee(ds, analysis.choose_axes(ds))
    if k.index is None:
        assert k.note, "a suppressed knee must say why"
        return
    front = ds.front_indices()
    assert len(front) >= 3
    assert k.index in front
    assert k.index not in k.extremes


def test_knee_is_suppressed_for_a_single_member_front():
    ds = data.load(fixture_path("single_member_front.json"))
    k = analysis.knee(ds, analysis.choose_axes(ds))
    assert k.index is None
    assert "a knee needs 3" in k.note


def test_knee_is_suppressed_with_one_objective():
    ds = data.load(fixture_path("single_objective.json"))
    k = analysis.knee(ds, analysis.choose_axes(ds))
    assert k.index is None
    assert "one objective" in k.note


def test_knee_is_suppressed_when_the_front_is_flat_on_an_axis():
    rows = [
        ({"tp": 1}, {"lat": 10.0, "rps": 40.0}),
        ({"tp": 2}, {"lat": 10.0, "rps": 50.0}),
        ({"tp": 4}, {"lat": 10.0, "rps": 60.0}),
    ]
    ds = synth(MIN_MAX, rows, front=[0, 1, 2])
    k = analysis.knee(ds, analysis.choose_axes(ds))
    assert k.index is None
    assert "flat in lat" in k.note


def test_knee_does_not_move_when_a_dominated_point_is_added():
    """D12's law: the knee is a property of the front, not of the sampling."""
    raw = load_raw("two_obj_slo.json")
    ds = data.build_dataset(raw, source_sha256="a" * 64, source_name="a.json")
    k = analysis.knee(ds, analysis.choose_axes(ds))
    assert k.index is not None
    before = data.entry_key({"config": ds.points[k.index].config,
                             "metrics": ds.points[k.index].metrics})

    raw2 = copy.deepcopy(raw)
    junk = copy.deepcopy(raw2["all_results"][0])
    junk["config"]["block_size"] = 999          # a config the front never used
    junk["metrics"]["responses_per_sec"] = min(
        r["metrics"]["responses_per_sec"] for r in raw2["all_results"]) / 10
    junk["metrics"]["ttft_p99_ms"] = max(
        r["metrics"]["ttft_p99_ms"] for r in raw2["all_results"]) * 10
    raw2["all_results"].append(junk)

    ds2 = data.build_dataset(raw2, source_sha256="b" * 64, source_name="b.json")
    k2 = analysis.knee(ds2, analysis.choose_axes(ds2))
    assert len(ds2.points) == len(ds.points) + 1
    after = data.entry_key({"config": ds2.points[k2.index].config,
                            "metrics": ds2.points[k2.index].metrics})
    assert after == before


# ------------------------------------------------------- knob attribution


def strips_by_name(ds):
    return {s.name: s for s in analysis.knob_strips(ds)}


@pytest.mark.parametrize("name", VALID_FIXTURES)
def test_every_knob_gets_one_bounded_ranked_strip(name):
    """§11 law: scores live in [0,1], every knob appears exactly once, and the
    ranking is exactly (-score, name) — no hidden reordering by the renderer."""
    ds = data.load(fixture_path(name))
    strips = analysis.knob_strips(ds)
    assert sorted(s.name for s in strips) == ds.knob_names()
    assert [(s.name) for s in strips] == [
        s.name for s in sorted(strips, key=lambda s: (-s.score, s.name))]
    for s in strips:
        assert 0.0 <= s.score <= 1.0, (s.name, s.score)
        cells = analysis.strip_cells(s)
        assert all(0.0 <= c <= 1.0 for c in cells)
        expected = len(s.values) if s.kind == "categorical" else 4
        assert len(cells) == expected


def test_total_variation_distance_is_exact():
    rows = [({"pol": "a"}, {"lat": 1.0, "rps": 9.0}),
            ({"pol": "a"}, {"lat": 2.0, "rps": 8.0}),
            ({"pol": "b"}, {"lat": 8.0, "rps": 2.0}),
            ({"pol": "b"}, {"lat": 9.0, "rps": 1.0})]
    ds = synth(MIN_MAX, rows, front=[0, 1])       # front all "a", cloud all "b"
    s = strips_by_name(ds)["pol"]
    assert s.kind == "categorical"
    assert s.score == 1.0                          # 0.5 * (|1-0| + |0-1|)
    assert s.values == ["a", "b"]
    assert s.front_fracs == [1.0, 0.0]
    assert s.cloud_fracs == [0.0, 1.0]
    assert s.headline == "front is 100% a (cloud 0%)"


def test_numeric_score_is_the_mean_gap_over_the_full_range():
    rows = [({"tp": 1}, {"lat": 1.0, "rps": 9.0}),
            ({"tp": 3}, {"lat": 2.0, "rps": 8.0}),
            ({"tp": 5}, {"lat": 8.0, "rps": 2.0}),
            ({"tp": 9}, {"lat": 9.0, "rps": 1.0})]
    ds = synth(MIN_MAX, rows, front=[0, 1])
    s = strips_by_name(ds)["tp"]
    assert s.kind == "numeric"
    # front mean 2, cloud mean 7, range 9 - 1 = 8
    assert s.score == pytest.approx(5.0 / 8.0)
    assert s.headline == "narrows to 1–3 of 1–9"
    assert s.narrows is True


def test_a_knob_the_front_spans_fully_says_so():
    rows = [({"bs": 16}, {"lat": 1.0, "rps": 9.0}),
            ({"bs": 32}, {"lat": 2.0, "rps": 8.0}),
            ({"bs": 24}, {"lat": 8.0, "rps": 2.0})]
    ds = synth(MIN_MAX, rows, front=[0, 1])
    s = strips_by_name(ds)["bs"]
    assert s.narrows is False
    assert s.headline == "spans the full searched range 16–32"


def test_a_pinned_numeric_front_lights_exactly_one_cell():
    """A zero-width front range overlaps no cell, so it is special-cased: the
    strip must still show where the front sits."""
    rows = [({"tp": 8}, {"lat": 1.0, "rps": 9.0}),
            ({"tp": 8}, {"lat": 2.0, "rps": 8.0}),
            ({"tp": 1}, {"lat": 8.0, "rps": 2.0}),
            ({"tp": 4}, {"lat": 9.0, "rps": 1.0})]
    ds = synth(MIN_MAX, rows, front=[0, 1])
    s = strips_by_name(ds)["tp"]
    assert s.headline == "front pins 8 of 1–8"
    assert analysis.strip_cells(s) == [0.0, 0.0, 0.0, 1.0]


def test_a_constant_knob_scores_zero():
    rows = [({"bs": 16}, {"lat": 1.0, "rps": 9.0}),
            ({"bs": 16}, {"lat": 8.0, "rps": 2.0}),
            ({"bs": 16}, {"lat": 9.0, "rps": 1.0})]
    ds = synth(MIN_MAX, rows, front=[0])
    s = strips_by_name(ds)["bs"]
    assert s.score == 0.0
    assert s.headline == "constant 16 across the search"


def test_a_missing_key_reads_as_unset_and_forces_categorical():
    """D13: real configs omit P/D knobs entirely when P/D is off."""
    rows = [({"tp": 1, "pd": "always"}, {"lat": 1.0, "rps": 9.0}),
            ({"tp": 3}, {"lat": 2.0, "rps": 8.0}),
            ({"tp": 5}, {"lat": 8.0, "rps": 2.0})]
    ds = synth(MIN_MAX, rows, front=[0])
    s = strips_by_name(ds)["pd"]
    assert s.kind == "categorical"
    assert s.values == [None, "always"]
    assert s.headline == "front is 100% always (cloud 0%)"


def test_a_null_makes_an_otherwise_numeric_knob_categorical():
    """D3: null is a value, and a value that is not a number cannot be averaged."""
    rows = [({"ttl": None}, {"lat": 1.0, "rps": 9.0}),
            ({"ttl": 5}, {"lat": 2.0, "rps": 8.0}),
            ({"ttl": 9}, {"lat": 8.0, "rps": 2.0})]
    ds = synth(MIN_MAX, rows, front=[0, 1])
    s = strips_by_name(ds)["ttl"]
    assert s.kind == "categorical"
    assert s.values == sorted(s.values, key=analysis._value_sort_key)
    assert s.values[0] is None                      # None sorts ahead of numbers
    assert s.values[1:] == [5, 9]
    assert "unset" in s.headline


def test_a_bool_knob_is_categorical_with_stable_value_order():
    rows = [({"fc": True}, {"lat": 1.0, "rps": 9.0}),
            ({"fc": False}, {"lat": 8.0, "rps": 2.0}),
            ({"fc": False}, {"lat": 9.0, "rps": 1.0})]
    ds = synth(MIN_MAX, rows, front=[0])
    s = strips_by_name(ds)["fc"]
    assert s.kind == "categorical"
    assert s.values == [False, True]
    assert s.headline == "front is 100% true (cloud 0%)"


def test_front_only_input_degrades_and_admits_it():
    """§5's degradation clause: no cloud means no attribution, and the headline
    must say so rather than implying one."""
    ds = data.load(fixture_path("front_only.json"))
    strips = analysis.knob_strips(ds)
    assert [s.name for s in strips] == sorted(s.name for s in strips)  # score ties
    for s in strips:
        assert s.score == 0.0
        assert s.comparable is False
        assert ("no cloud to compare" in s.headline
                or s.headline.startswith("constant "))


def test_an_empty_front_attributes_nothing_instead_of_crashing():
    """A results file may report pareto_front: [] when every evaluation failed.
    Numeric knobs would divide by zero; categorical knobs would score a
    meaningless 0.5 against an all-zero front distribution. Both must score 0
    and say there is nothing to compare."""
    entries = [{"config": {"tp": 1, "pol": "a"}, "metrics": {"lat": 1.0, "rps": 9.0}},
               {"config": {"tp": 4, "pol": "b"}, "metrics": {"lat": 8.0, "rps": 2.0}}]
    raw = {"pareto_front": [], "all_results": entries, "objectives": MIN_MAX,
           "spec": {"model": "synthetic"}, "search_stats": {"total_evaluated": 2}}
    ds = data.build_dataset(raw, source_sha256="0" * 64, source_name="empty.json")
    assert ds.front_indices() == []

    strips = analysis.knob_strips(ds)
    assert {s.name for s in strips} == {"tp", "pol"}
    for strip in strips:
        assert strip.score == 0.0, strip.name
        assert strip.comparable is False, strip.name
        assert "no front to compare" in strip.headline
        cells = analysis.strip_cells(strip)
        assert all(0.0 <= c <= 1.0 for c in cells)

# ------------------------------------------------------------------ picks


def titles_of(picks_obj):
    return [t for row in picks_obj.rows for t in row.titles]


def test_gpu_count_prefers_gpus_used():
    ds = data.load(fixture_path("two_obj_slo.json"))
    assert analysis.gpu_source(ds) == "gpus_used"
    idx = ds.front_indices()[0]
    assert analysis.gpu_count(ds, idx) == int(ds.points[idx].config["gpus_used"])


def test_gpu_count_falls_back_to_tp_times_replicas():
    rows = [({"tp": 2, "replicas": 3}, {"lat": 1.0, "rps": 9.0}),
            ({"tp": 4, "replicas": 1}, {"lat": 8.0, "rps": 2.0})]
    ds = synth(MIN_MAX, rows, front=[0])
    assert analysis.gpu_source(ds) == "tp x replicas"
    assert analysis.gpu_count(ds, 0) == 6


def test_gpu_count_fallback_includes_dp_when_present():
    """An instance occupies tp x dp GPUs, so the no-gpus_used fallback must
    multiply by dp. Omitting it under-reports a wide-EP topology's GPU cost —
    tp8 dp2 is 16 GPUs, not 8 — and silently merges distinct GPU tiers."""
    rows = [({"tp": 8, "dp": 2, "replicas": 1}, {"lat": 1.0, "rps": 9.0}),
            ({"tp": 4, "dp": 4, "replicas": 1}, {"lat": 8.0, "rps": 2.0})]
    ds = synth(MIN_MAX, rows, front=[0])
    assert analysis.gpu_source(ds) == "tp x dp x replicas"
    assert analysis.gpu_count(ds, 0) == 16
    assert analysis.gpu_count(ds, 1) == 16


def test_no_gpu_dimension_is_reported_not_invented():
    """§9: color channel dropped, stated — and no Fewest GPUs row."""
    ds = data.load(fixture_path("no_gpu_dimension.json"))
    assert analysis.gpu_source(ds) == ""
    assert analysis.gpu_count(ds, 0) is None
    chosen = analysis.picks(ds, analysis.choose_axes(ds))
    assert "Fewest GPUs" not in titles_of(chosen)
    assert any("no GPU dimension" in n for n in chosen.notes)


def test_picks_have_one_row_per_objective_plus_the_knee():
    ds = data.load(fixture_path("three_obj.json"))
    chosen = analysis.picks(ds, analysis.choose_axes(ds))
    for obj in ds.objectives:
        verb = "Max" if obj.direction == "maximize" else "Min"
        assert f"{verb} {obj.key}" in titles_of(chosen)
    assert "★ Best balance" in titles_of(chosen)
    # Every pick is drawn from the pool, and no config appears twice.
    assert all(row.index in chosen.pool for row in chosen.rows)
    assert len({row.index for row in chosen.rows}) == len(chosen.rows)


def test_a_config_winning_several_categories_appears_once():
    rows = [({"tp": 1, "replicas": 1}, {"lat": 1.0, "rps": 9.0}),   # best at everything
            ({"tp": 8, "replicas": 8}, {"lat": 8.0, "rps": 2.0})]
    ds = synth(MIN_MAX, rows, front=[0, 1])
    chosen = analysis.picks(ds, analysis.choose_axes(ds))
    assert len(chosen.rows) == 1
    assert chosen.rows[0].titles == ["Min lat", "Max rps", "Fewest GPUs"]


def test_the_pool_is_the_slo_feasible_set_when_constraints_exist():
    ds = data.load(fixture_path("two_obj_slo.json"))
    chosen = analysis.picks(ds, analysis.choose_axes(ds))
    assert "SLO-feasible" in chosen.header
    assert all(ds.points[i].slo_ok for i in chosen.pool)
    assert chosen.banner == ""


def test_an_unmeetable_slo_falls_back_to_the_front_with_a_banner():
    """§9: banner, picks fall back to the front, labelled SLO-violating."""
    ds = data.load(fixture_path("empty_slo_feasible.json"))
    chosen = analysis.picks(ds, analysis.choose_axes(ds))
    assert chosen.pool == ds.front_indices()
    assert "none met the SLO" in chosen.header
    assert "violate" in chosen.banner


def test_a_uniform_gpu_count_omits_the_fewest_gpus_row_and_says_so():
    ds = data.load(fixture_path("single_gpu_count.json"))
    chosen = analysis.picks(ds, analysis.choose_axes(ds))
    assert "Fewest GPUs" not in titles_of(chosen)
    assert any("uses 8 GPUs" in n for n in chosen.notes)


def test_a_suppressed_knee_is_listed_as_an_omission():
    ds = data.load(fixture_path("single_member_front.json"))
    chosen = analysis.picks(ds, analysis.choose_axes(ds))
    assert "★ Best balance" not in titles_of(chosen)
    assert any("best-balance row omitted" in n for n in chosen.notes)


def test_the_header_is_singular_for_one_config():
    ds = data.load(fixture_path("single_member_front.json"))
    chosen = analysis.picks(ds, analysis.choose_axes(ds))
    assert chosen.header == "from 1 front config"


def test_describe_config_disambiguates_colliding_labels():
    rows = [({"tp": 8, "replicas": 1}, {"lat": 1.0, "rps": 9.0}),
            ({"tp": 8, "replicas": 1}, {"lat": 2.0, "rps": 8.0}),
            ({"tp": 2, "replicas": 4}, {"lat": 8.0, "rps": 2.0})]
    ds = synth(MIN_MAX, rows, front=[0, 1])
    assert analysis.describe_config(ds, 0) == "tp8 ×1 #0"
    assert analysis.describe_config(ds, 1) == "tp8 ×1 #1"
    assert analysis.describe_config(ds, 2) == "tp2 ×4"       # unique: stays short


@pytest.mark.parametrize("name", VALID_FIXTURES)
def test_analyse_produces_one_bundle_per_fixture(name):
    """The bundle is what makes 'renderers cannot disagree' structural."""
    ds = data.load(fixture_path(name))
    an = analysis.analyse(ds)
    assert an.ds is ds
    assert an.axes.x is not None
    assert sorted(s.name for s in an.strips) == ds.knob_names()
    assert isinstance(an.banners, list)


def test_the_shared_furniture_is_formatted_once():
    """fmt, context_line, summary_bits and provenance_bits are what stop picks.md
    and index.html from printing the same fact two different ways."""
    assert analysis.fmt(8.0) == "8"
    assert analysis.fmt(169.34) == "169"
    assert analysis.fmt(42.17) == "42.2"
    assert analysis.fmt(0.8512) == "0.851"

    ds = data.load(fixture_path("two_obj_slo.json"))
    an = analysis.analyse(ds)
    assert analysis.context_line(ds).startswith("Qwen/Qwen3-32B · H100 · chatbot")
    bits = analysis.summary_bits(ds)
    assert bits[0].endswith("evaluated")
    assert any(b.startswith("SLO: ") for b in bits)
    prov = analysis.provenance_bits(an)
    assert prov[0].startswith(f"generated from {ds.source_name} (sha ")
    assert ds.source_sha256[:8] in prov[0]
    assert not any("202" in b for b in prov)          # no date, ever

    thinned = analysis.analyse(ds, max_points=8)
    assert analysis.provenance_bits(thinned)[-1] == thinned.plot_notes[0]


def test_page_banners_add_the_flag_note_to_the_data_banners():
    class Source:                                     # duck-typed FlagSource
        flags = None
        note = "re-run search or pass --space to generate commands"
        warning = ""

    ds = data.load(fixture_path("front_only.json"))
    an = analysis.analyse(ds)
    banners = analysis.page_banners(an, Source())
    assert banners[:len(an.banners)] == an.banners
    assert banners[-1].startswith("No deploy commands: ")


def test_banners_state_every_thing_the_data_forces():
    ds = data.load(fixture_path("front_only.json"))
    an = analysis.analyse(ds)
    assert any("cannot be verified" in b for b in an.banners)

    ds = data.load(fixture_path("dirty_metrics.json"))
    an = analysis.analyse(ds)
    assert any("excluded" in b for b in an.banners)

    ds = data.load(fixture_path("single_member_front.json"))
    an = analysis.analyse(ds)
    assert any("single non-dominated config" in b for b in an.banners)

    ds = data.load(fixture_path("empty_slo_feasible.json"))
    an = analysis.analyse(ds)
    assert any("violate" in b for b in an.banners)


# ------------------------------------------- hero form, color, decimation

import math                                       # noqa: E402 — used below


@pytest.mark.parametrize("name,form", [
    ("single_objective.json", analysis.HERO_CARD),
    ("two_obj_slo.json", analysis.HERO_SCATTER),
    ("three_obj.json", analysis.HERO_SCATTER),
    ("four_obj.json", analysis.HERO_PARALLEL),
])
def test_hero_form_follows_the_objective_count(name, form):
    ds = data.load(fixture_path(name))
    hero = analysis.choose_hero(ds, analysis.choose_axes(ds))
    assert hero.form == form
    assert hero.caption


def test_a_parallel_hero_also_asks_for_the_pair_scatter():
    ds = data.load(fixture_path("four_obj.json"))
    hero = analysis.choose_hero(ds, analysis.choose_axes(ds))
    assert hero.with_scatter is True


def test_a_single_objective_card_asks_for_the_cloud_distribution():
    ds = data.load(fixture_path("single_objective.json"))
    hero = analysis.choose_hero(ds, analysis.choose_axes(ds))
    assert hero.with_distribution is ds.has_cloud


def test_the_third_objective_is_the_one_not_on_an_axis():
    """With 3 objectives the axes are the widest-spread pair, so the colored
    objective is not objectives[2]. The caption and the channel must agree."""
    ds = data.load(fixture_path("three_obj.json"))
    axes = analysis.choose_axes(ds)
    third = analysis.third_objective(ds, axes)
    assert third.key not in (axes.x.key, axes.y.key)
    colour = analysis.choose_color(ds, axes, list(range(len(ds.points))))
    assert colour.key == third.key
    assert colour.kind == "numeric"
    assert third.key in analysis.choose_hero(ds, axes).caption


def test_color_defaults_to_gpu_count():
    ds = data.load(fixture_path("two_obj_slo.json"))
    axes = analysis.choose_axes(ds)
    colour = analysis.choose_color(ds, axes, list(range(len(ds.points))))
    assert colour.key == "gpus"
    assert colour.kind == "categorical"
    assert colour.values == sorted(colour.values)
    assert "GPU count" in colour.caption
    got = analysis.color_values(ds, colour, [0, 1])
    assert got == [analysis.gpu_count(ds, 0), analysis.gpu_count(ds, 1)]


def test_a_single_gpu_count_drops_the_channel_and_says_so():
    """§5: a one-entry legend is worse than none."""
    ds = data.load(fixture_path("single_gpu_count.json"))
    axes = analysis.choose_axes(ds)
    colour = analysis.choose_color(ds, axes, list(range(len(ds.points))))
    assert colour.key is None
    assert colour.kind == "none"
    assert "carries nothing" in colour.caption


def test_no_gpu_dimension_drops_the_channel_and_says_so():
    ds = data.load(fixture_path("no_gpu_dimension.json"))
    axes = analysis.choose_axes(ds)
    colour = analysis.choose_color(ds, axes, list(range(len(ds.points))))
    assert colour.key is None
    assert "no GPU count is derivable" in colour.caption


def test_a_color_override_takes_a_knob_or_a_metric():
    ds = data.load(fixture_path("two_obj_slo.json"))
    axes = analysis.choose_axes(ds)
    idx = list(range(len(ds.points)))

    cat = analysis.choose_color(ds, axes, idx, "scheduler")
    assert cat.kind == "categorical"
    assert cat.values == sorted(cat.values, key=str)
    assert set(analysis.color_values(ds, cat, idx)) <= set(cat.values)

    num = analysis.choose_color(ds, axes, idx, "e2e_p99_ms")
    assert num.kind == "numeric"
    assert num.values == []


def test_an_unknown_color_field_is_an_error_that_lists_the_options():
    ds = data.load(fixture_path("two_obj_slo.json"))
    axes = analysis.choose_axes(ds)
    with pytest.raises(ValueError) as excinfo:
        analysis.choose_color(ds, axes, [0], "does_not_exist")
    message = str(excinfo.value)
    assert "does_not_exist" in message
    assert "available metrics" in message and "available knobs" in message


@pytest.mark.parametrize("name", VALID_FIXTURES)
def test_decimation_never_samples_the_front(name):
    """§4.3's headline promise, as a law over every fixture."""
    ds = data.load(fixture_path(name))
    an = analysis.analyse(ds, max_points=5)
    assert set(ds.front_indices()) <= set(an.plotted)
    assert set(an.cloud_plotted) <= set(ds.cloud_indices())
    assert an.plotted == sorted(set(an.plotted))


@pytest.mark.parametrize("name", VALID_FIXTURES)
def test_decimation_is_a_pure_function_of_the_input(name):
    ds = data.load(fixture_path(name))
    first = analysis.analyse(ds, max_points=7)
    second = analysis.analyse(ds, max_points=7)
    assert first.cloud_plotted == second.cloud_plotted
    assert first.plot_notes == second.plot_notes


def test_decimation_keeps_at_most_one_point_per_grid_cell():
    """The pinned rule's invariant: one survivor per occupied bin, and the
    envelope survives because the corner bins are still occupied."""
    ds = data.load(fixture_path("two_obj_slo.json"))
    axes = analysis.choose_axes(ds)
    kept, note = analysis.decimate(ds, axes, 10)
    assert len(kept) <= 10
    assert note and "the front is never sampled" in note
    k = math.ceil(math.sqrt(10))
    cells = [(min(k - 1, int(fx * k)), min(k - 1, int(fy * k)))
             for fx, fy in analysis.screen_fractions(ds, axes, kept)]
    assert len(set(cells)) == len(cells)


def test_no_max_points_keeps_the_whole_cloud_and_says_nothing():
    ds = data.load(fixture_path("two_obj_slo.json"))
    kept, note = analysis.decimate(ds, analysis.choose_axes(ds), None)
    assert kept == ds.cloud_indices()
    assert note == ""
    assert analysis.analyse(ds).plot_notes == []


def test_a_cloud_under_the_cap_is_untouched():
    ds = data.load(fixture_path("two_obj_slo.json"))
    kept, note = analysis.decimate(ds, analysis.choose_axes(ds), 10_000)
    assert kept == ds.cloud_indices()
    assert note == ""
