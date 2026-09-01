"""The figures: written, deterministic, and drawn from the bundle alone.

MPLCONFIGDIR is set before matplotlib is imported: without a writable config dir
matplotlib warns on every import, which pollutes test output.
"""
import hashlib
import os

os.environ.setdefault("MPLCONFIGDIR", os.path.join(
    os.environ.get("TMPDIR", "/tmp"), "mplcfg"))

import pytest

import analysis
import data
import render_mpl
from conftest import VALID_FIXTURES, fixture_path


def sha(path):
    with open(path, "rb") as f:
        return hashlib.sha256(f.read()).hexdigest()


@pytest.mark.parametrize("name", VALID_FIXTURES)
def test_every_fixture_produces_a_pareto_png(name, tmp_path):
    """§11: every fixture renders without exception — including the degenerate
    ones, which is where a chart library usually throws."""
    an = analysis.analyse(data.load(fixture_path(name)))
    out = str(tmp_path / "pareto.png")
    render_mpl.pareto(an, out)
    assert os.path.getsize(out) > 5000
    with open(out, "rb") as f:
        assert f.read(8) == b"\x89PNG\r\n\x1a\n"


@pytest.mark.parametrize("name", VALID_FIXTURES)
def test_two_renders_are_byte_identical(name, tmp_path):
    """§10's law. Not a golden-file comparison: font stacks differ across
    machines, so this guards accidental nondeterminism, not appearance."""
    an = analysis.analyse(data.load(fixture_path(name)))
    first, second = str(tmp_path / "a.png"), str(tmp_path / "b.png")
    render_mpl.pareto(an, first)
    render_mpl.pareto(an, second)
    assert sha(first) == sha(second)


def test_decimation_changes_the_figure_but_not_the_front(tmp_path):
    ds = data.load(fixture_path("two_obj_slo.json"))
    full = analysis.analyse(ds)
    thin = analysis.analyse(ds, max_points=8)
    a, b = str(tmp_path / "full.png"), str(tmp_path / "thin.png")
    render_mpl.pareto(full, a)
    render_mpl.pareto(thin, b)
    assert sha(a) != sha(b)                       # fewer cloud points is visible
    assert full.ds.front_indices() == thin.ds.front_indices()


def test_the_rcparams_are_pinned_not_inherited():
    """A figure must be a function of the data, not of the user's matplotlibrc."""
    for key in ("figure.figsize", "figure.dpi", "savefig.dpi", "font.size"):
        assert key in render_mpl.RCPARAMS
    assert render_mpl.PNG_METADATA == {"Software": None}


def test_metadata_is_stripped_from_the_written_file(tmp_path):
    """matplotlib stamps its version into the PNG unless told not to; that would
    make byte-identity depend on the installed version."""
    ds = data.load(fixture_path("two_obj_slo.json"))
    out = str(tmp_path / "meta.png")
    render_mpl.pareto(analysis.analyse(ds), out)
    with open(out, "rb") as f:
        blob = f.read()
    assert b"matplotlib version" not in blob


@pytest.mark.parametrize("name", ["single_objective.json", "single_member_front.json",
                                  "constant_objective.json", "no_gpu_dimension.json",
                                  "front_only.json"])
def test_the_degenerate_cases_still_carry_their_caption(name, tmp_path):
    an = analysis.analyse(data.load(fixture_path(name)))
    out = str(tmp_path / "deg.png")
    render_mpl.pareto(an, out)
    assert os.path.getsize(out) > 5000
    assert an.hero.caption                        # the page always says why


# ---------------------------------------------------- knobs and parallel


@pytest.mark.parametrize("name", VALID_FIXTURES)
def test_every_fixture_produces_a_deterministic_knobs_png(name, tmp_path):
    an = analysis.analyse(data.load(fixture_path(name)))
    first, second = str(tmp_path / "k1.png"), str(tmp_path / "k2.png")
    render_mpl.knobs(an, first)
    render_mpl.knobs(an, second)
    assert os.path.getsize(first) > 5000
    assert sha(first) == sha(second)


@pytest.mark.parametrize("name", [n for n in VALID_FIXTURES
                                  if n != "single_objective.json"])
def test_every_multi_objective_fixture_produces_a_deterministic_parallel_png(
        name, tmp_path):
    an = analysis.analyse(data.load(fixture_path(name)))
    first, second = str(tmp_path / "p1.png"), str(tmp_path / "p2.png")
    render_mpl.parallel(an, first)
    render_mpl.parallel(an, second)
    assert os.path.getsize(first) > 5000
    assert sha(first) == sha(second)


def test_the_knobs_figure_grows_with_the_knob_count(tmp_path):
    """25 knobs in one fixed-height figure would be illegible."""
    many = analysis.analyse(data.load(fixture_path("two_obj_slo.json")))
    few = analysis.analyse(data.load(fixture_path("no_gpu_dimension.json")))
    assert len(many.strips) > len(few.strips)
    a, b = str(tmp_path / "many.png"), str(tmp_path / "few.png")
    render_mpl.knobs(many, a)
    render_mpl.knobs(few, b)
    assert os.path.getsize(a) > os.path.getsize(b)


def test_knobs_and_picks_md_agree_on_every_strip():
    """Both read strip_cells(); this is the law that keeps them in step."""
    import commands
    import render_md
    ds = data.load(fixture_path("two_obj_slo.json"))
    an = analysis.analyse(ds)
    text = render_md.document(an, commands.resolve_param_flags(ds))
    rows = {ln.split("|")[1].strip(): ln.split("|")[2].strip().strip("`")
            for ln in text.splitlines() if ln.startswith("| ")}
    for strip in an.strips:
        assert strip.headline in text            # same sentence in both outputs
        # One glyph per cell: the markdown strip and the drawn strip are the
        # same list of numbers, rendered two ways.
        assert len(rows[strip.name]) == len(analysis.strip_cells(strip))


def test_parallel_axes_are_labelled_with_their_range(tmp_path):
    """The tick label carries the range; there is no floating text to collide."""
    an = analysis.analyse(data.load(fixture_path("four_obj.json")))
    out = str(tmp_path / "p.png")
    render_mpl.parallel(an, out)
    assert os.path.getsize(out) > 5000
    assert len(an.ds.objectives) == 4
