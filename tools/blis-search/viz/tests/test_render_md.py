"""picks.md: the content laws, over every fixture."""
import pytest

import analysis
import commands
import data
import render_md
from conftest import VALID_FIXTURES, fixture_path


def rendered(name, space=None):
    ds = data.load(fixture_path(name))
    an = analysis.analyse(ds)
    return an, render_md.document(an, commands.resolve_param_flags(ds, space))


@pytest.mark.parametrize("name", VALID_FIXTURES)
def test_every_fixture_renders_a_complete_document(name):
    an, text = rendered(name)
    assert text.startswith("# BLIS search picks")
    assert text.endswith("\n")
    assert "## Picks (" in text
    assert "## Which knobs the front exploits" in text
    assert an.ds.source_sha256[:8] in text          # provenance, not a timestamp
    assert an.picks.header in text


@pytest.mark.parametrize("name", VALID_FIXTURES)
def test_the_document_is_a_pure_function_of_the_input(name):
    """§10: no timestamps, no environment. Two renders must be identical."""
    _, first = rendered(name)
    _, second = rendered(name)
    assert first == second
    assert "202" not in first.split("generated from")[-1]   # no date in provenance


@pytest.mark.parametrize("name", VALID_FIXTURES)
def test_every_banner_and_every_omission_reaches_the_page(name):
    an, text = rendered(name)
    for banner in an.banners:
        assert banner in text
    for note in an.picks.notes:
        assert note in text


@pytest.mark.parametrize("name", VALID_FIXTURES)
def test_every_pick_gets_a_row_and_a_command_block(name):
    an, text = rendered(name)
    for row in an.picks.rows:
        label = analysis.describe_config(an.ds, row.index)
        assert label in text
        assert f"### {', '.join(row.titles)} — {label}" in text
    if an.picks.rows and an.ds.param_flags:
        assert "```bash" in text
        assert "./blis run " in text


def test_a_file_without_flags_shows_the_note_instead_of_a_command():
    """§7 tier 3, as the user sees it."""
    an, text = rendered("front_only.json")
    assert commands.NO_FLAGS_NOTE in text
    assert "```bash" not in text


def test_a_space_file_restores_the_commands():
    """§7 tier 2, as the user sees it."""
    import os
    from conftest import VIZ_DIR
    space = os.path.join(os.path.dirname(VIZ_DIR), "full-stack.yaml")
    an, text = rendered("front_only.json", space=space)
    assert "```bash" in text
    assert "./blis run " in text
    assert commands.NO_FLAGS_NOTE not in text


def test_the_command_block_is_fenced_bash_and_one_line():
    an, text = rendered("two_obj_slo.json")
    blocks = text.split("```bash\n")[1:]
    assert blocks
    for block in blocks:
        body = block.split("```")[0]
        assert body.count("\n") == 1                # exactly one command line
        assert body.startswith("./blis run ")
        assert "--metrics-path" not in body


def test_the_knob_table_is_ranked_and_carries_a_strip_per_knob():
    an, text = rendered("two_obj_slo.json")
    lines = [ln for ln in text.splitlines() if ln.startswith("| ") and "|" in ln[2:]]
    knob_lines = [ln for ln in lines if any(f"| {s.name} |" in ln for s in an.strips)]
    assert len(knob_lines) == len(an.strips)
    order = [ln.split("|")[1].strip() for ln in knob_lines]
    assert order == [s.name for s in an.strips]     # analysis' ranking, unchanged
    for s in an.strips:
        assert s.headline in text


def test_front_only_input_says_the_strips_are_not_attributions():
    an, text = rendered("front_only.json")
    assert "not attributions" in text


def test_the_objective_columns_carry_their_direction_arrow():
    an, text = rendered("two_obj_slo.json")
    assert "responses_per_sec ↑" in text
    assert "ttft_p99_ms ↓" in text


def test_a_file_with_no_gpu_dimension_drops_the_gpu_column():
    an, text = rendered("no_gpu_dimension.json")
    header = [ln for ln in text.splitlines() if ln.startswith("| Pick |")][0]
    assert "GPUs" not in header
