"""The command surface, end to end over every fixture."""
import hashlib
import os
import subprocess
import sys

os.environ.setdefault("MPLCONFIGDIR", os.path.join(
    os.environ.get("TMPDIR", "/tmp"), "mplcfg"))

import pytest

import cli
import data
from conftest import VALID_FIXTURES, VIZ_DIR, fixture_path

CLI_PY = os.path.join(VIZ_DIR, "cli.py")
SEARCH_DIR = os.path.dirname(VIZ_DIR)


def run(args, expect=0):
    """Run the CLI in a subprocess, which is how a user runs it."""
    done = subprocess.run([sys.executable, CLI_PY] + args, capture_output=True,
                          text=True, cwd=SEARCH_DIR)
    assert done.returncode == expect, done.stderr
    return done


def sha(path):
    with open(path, "rb") as f:
        return hashlib.sha256(f.read()).hexdigest()


def test_default_out_dir_sits_beside_the_results_file(tmp_path):
    results = tmp_path / "search_output.json"
    results.write_text("{}")
    assert cli.default_out_dir(str(results)) == str(tmp_path / "search_output-viz")


@pytest.mark.parametrize("name", VALID_FIXTURES)
def test_every_fixture_produces_a_full_report(name, tmp_path):
    out = tmp_path / "report"
    done = run([fixture_path(name), "--out-dir", str(out)])
    objectives = len(data.load(fixture_path(name)).objectives)

    expected = ["pareto.png", "knobs.png"]
    if objectives >= 3:
        expected.append("parallel.png")
    expected += ["picks.md", "index.html"]

    assert [line.split("wrote ")[1] for line in done.stdout.strip().splitlines()] \
        == [str(out / f) for f in expected]
    for f in expected:
        assert (out / f).stat().st_size > 0


@pytest.mark.parametrize("name", VALID_FIXTURES)
def test_two_subprocess_runs_produce_identical_files(name, tmp_path):
    """§11's law, on the real command: identical input, identical bytes, in two
    separate processes with two separate interpreters' hash seeds."""
    first, second = tmp_path / "a", tmp_path / "b"
    one = run([fixture_path(name), "--out-dir", str(first)])
    two = run([fixture_path(name), "--out-dir", str(second)])
    names = sorted(os.listdir(first))
    # Exactly the reported files, so a newly added output cannot slip through
    # unchecked while this test still claims the runs match.
    reported = {line.split("wrote ")[1].rsplit("/", 1)[1]
                for line in one.stdout.strip().splitlines()}
    assert set(names) == reported
    assert names == sorted(os.listdir(second))
    for f in names:
        assert sha(first / f) == sha(second / f), f
    assert one.stdout.replace(str(first), "") == two.stdout.replace(str(second), "")


def test_parallel_png_is_absent_below_three_objectives(tmp_path):
    out = tmp_path / "two"
    run([fixture_path("two_obj_slo.json"), "--out-dir", str(out)])
    assert not (out / "parallel.png").exists()

    out3 = tmp_path / "three"
    run([fixture_path("three_obj.json"), "--out-dir", str(out3)])
    assert (out3 / "parallel.png").exists()


def test_png_only_and_html_only_restrict_the_outputs(tmp_path):
    pngs = tmp_path / "pngs"
    run([fixture_path("two_obj_slo.json"), "--out-dir", str(pngs), "--png-only"])
    assert not (pngs / "index.html").exists()
    assert (pngs / "pareto.png").exists()
    assert (pngs / "picks.md").exists()          # picks.md is not a PNG, but it is
                                                 # the point of the tool

    html = tmp_path / "html"
    run([fixture_path("two_obj_slo.json"), "--out-dir", str(html), "--html-only"])
    assert (html / "index.html").exists()
    assert not (html / "pareto.png").exists()


def test_png_only_and_html_only_are_mutually_exclusive(tmp_path):
    done = subprocess.run(
        [sys.executable, CLI_PY, fixture_path("two_obj_slo.json"),
         "--out-dir", str(tmp_path / "x"), "--png-only", "--html-only"],
        capture_output=True, text=True, cwd=SEARCH_DIR)
    assert done.returncode == 2
    assert "not allowed with" in done.stderr


def test_an_existing_directory_is_reused_and_overwritten(tmp_path):
    out = tmp_path / "again"
    run([fixture_path("two_obj_slo.json"), "--out-dir", str(out)])
    before = sha(out / "pareto.png")
    stale = out / "stale.txt"
    stale.write_text("kept")
    run([fixture_path("two_obj_slo.json"), "--out-dir", str(out)])
    assert sha(out / "pareto.png") == before
    assert stale.read_text() == "kept"           # we overwrite ours, not theirs


def test_an_out_dir_that_is_a_file_is_a_named_error(tmp_path):
    clash = tmp_path / "not-a-dir"
    clash.write_text("x")
    done = run([fixture_path("two_obj_slo.json"), "--out-dir", str(clash)], expect=2)
    assert "existing file" in done.stderr


def test_a_foreign_schema_is_rejected_with_the_expected_keys(tmp_path):
    done = run([fixture_path("foreign_schema.json"), "--out-dir", str(tmp_path / "o")],
               expect=2)
    assert "not a blis-search results file" in done.stderr
    assert "expected:" in done.stderr and "found:" in done.stderr


def test_a_missing_results_file_is_a_named_error(tmp_path):
    done = run(["does-not-exist.json", "--out-dir", str(tmp_path / "o")], expect=2)
    assert "does-not-exist.json" in done.stderr


def test_an_unknown_axis_or_color_names_the_available_fields(tmp_path):
    done = run([fixture_path("two_obj_slo.json"), "--out-dir", str(tmp_path / "o"),
                "--x", "nope"], expect=2)
    assert "unknown metric" in done.stderr and "available metrics" in done.stderr

    done = run([fixture_path("two_obj_slo.json"), "--out-dir", str(tmp_path / "o"),
                "--color", "nope"], expect=2)
    assert "available knobs" in done.stderr


def test_axis_overrides_change_the_figure(tmp_path):
    auto, forced = tmp_path / "auto", tmp_path / "forced"
    run([fixture_path("two_obj_slo.json"), "--out-dir", str(auto)])
    run([fixture_path("two_obj_slo.json"), "--out-dir", str(forced),
         "--y", "e2e_p99_ms"])
    assert sha(auto / "pareto.png") != sha(forced / "pareto.png")
    assert "e2e_p99_ms" in (forced / "index.html").read_text()


def test_max_points_decimates_and_says_so(tmp_path):
    out = tmp_path / "thin"
    run([fixture_path("two_obj_slo.json"), "--out-dir", str(out), "--max-points", "8"])
    page = (out / "index.html").read_text()
    assert "decimated" in page and "the front is never sampled" in page


def test_a_file_without_flags_warns_on_stderr_but_still_reports(tmp_path):
    done = run([fixture_path("front_only.json"), "--out-dir", str(tmp_path / "o")])
    assert "no deploy commands" in done.stderr
    assert "wrote " in done.stdout                # a warning is not a failure


def test_a_space_file_restores_the_commands(tmp_path):
    out = tmp_path / "spaced"
    done = run([fixture_path("front_only.json"), "--out-dir", str(out),
                "--space", os.path.join(SEARCH_DIR, "full-stack.yaml")])
    assert "no deploy commands" not in done.stderr
    assert "./blis run " in (out / "picks.md").read_text()


def test_an_unreadable_space_is_a_named_error(tmp_path):
    done = run([fixture_path("front_only.json"), "--out-dir", str(tmp_path / "o"),
                "--space", "/nonexistent/space.yaml"], expect=2)
    assert "/nonexistent/space.yaml" in done.stderr


def test_the_module_entry_point_works_too(tmp_path):
    """`python -m viz` must work from tools/blis-search/, which needs the
    sys.path insert in __main__.py."""
    done = subprocess.run(
        [sys.executable, "-m", "viz", fixture_path("two_obj_slo.json"),
         "--out-dir", str(tmp_path / "mod")],
        capture_output=True, text=True, cwd=SEARCH_DIR)
    assert done.returncode == 0, done.stderr
    assert (tmp_path / "mod" / "index.html").exists()


def test_an_unwritable_out_dir_is_a_named_error_not_a_traceback(tmp_path):
    """A read-only parent is the ordinary way this fails in the wild, and a
    traceback there would violate the never-crash rule."""
    locked = tmp_path / "locked"
    locked.mkdir()
    locked.chmod(0o500)
    try:
        done = run([fixture_path("two_obj_slo.json"),
                    "--out-dir", str(locked / "nested")], expect=2)
        assert done.stderr.startswith("error: ")
        assert "Traceback" not in done.stderr
    finally:
        locked.chmod(0o700)          # so pytest can clean the directory up


def test_malformed_json_is_a_named_error_naming_the_file(tmp_path):
    bad = tmp_path / "bad.json"
    bad.write_text("{]")
    done = run([str(bad), "--out-dir", str(tmp_path / "out")], expect=2)
    assert "is not valid JSON" in done.stderr
    assert str(bad) in done.stderr
    assert "Traceback" not in done.stderr


def test_max_points_must_be_positive(tmp_path):
    """Zero is falsy downstream, so accepting it would silently ignore the flag."""
    for bad in ("0", "-5"):
        done = subprocess.run(
            [sys.executable, CLI_PY, fixture_path("two_obj_slo.json"),
             "--out-dir", str(tmp_path / "out"), "--max-points", bad],
            capture_output=True, text=True, cwd=SEARCH_DIR)
        assert done.returncode == 2, bad
        assert "must be 1 or more" in done.stderr, bad
