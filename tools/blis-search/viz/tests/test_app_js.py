"""The interaction layer, exercised by node against a real generated page.

There is no JS test framework here, so this is deliberately narrow: a syntax
gate, and a smoke run proving the script decodes the plotspec and draws one
point per plotted config. It skips when node is absent.
"""
import json
import os
import shutil
import subprocess

import pytest

import analysis
import commands
import data
import render_html
from conftest import TESTS_DIR, VIZ_DIR, VALID_FIXTURES, fixture_path

NODE = shutil.which("node")
needs_node = pytest.mark.skipif(NODE is None, reason="node is not installed")
APP_JS = os.path.join(VIZ_DIR, "assets", "app.js")
RUNNER = os.path.join(TESTS_DIR, "run_app.js")


@needs_node
def test_app_js_parses():
    subprocess.run([NODE, "--check", APP_JS], check=True, capture_output=True)


def test_app_js_contains_no_url_at_all():
    """Including the SVG namespace: the script builds markup instead of calling
    createElementNS, so the offline assertion has no exceptions to carve out."""
    with open(APP_JS) as f:
        source = f.read()
    assert "http" not in source
    # The call, not the word: the comment explaining why we avoid it may stay.
    assert "createElementNS(" not in source


@needs_node
@pytest.mark.parametrize("name", VALID_FIXTURES)
def test_the_script_draws_one_point_per_plotted_config(name, tmp_path):
    ds = data.load(fixture_path(name))
    an = analysis.analyse(ds)
    page = tmp_path / "index.html"
    page.write_text(render_html.document(an, commands.resolve_param_flags(ds)))

    done = subprocess.run([NODE, RUNNER, str(page)], check=True,
                          capture_output=True, text=True)
    drawn = json.loads(done.stdout)

    assert drawn["table_rows"] == len(an.plotted)
    assert drawn["count_text"] == f"{len(an.plotted)} of {len(an.plotted)} configs shown"
    assert drawn["pins_prompt"] is True
    if an.hero.form == analysis.HERO_CARD:
        # The baked card is left alone; only the table and panel come alive.
        assert drawn["circles"] == 0
    else:
        assert drawn["circles"] == len(an.plotted)
        assert drawn["stars"] == (0 if an.knee.index is None else 1)
        assert drawn["axis_labels"] >= 3


def test_the_interaction_surface_is_wired():
    """The panel's behaviours exist. Pointer semantics need a browser, so this
    asserts the handlers are registered rather than simulating a drag."""
    with open(APP_JS) as f:
        source = f.read()
    for handler in ('"mousedown"', '"mousemove"', '"mouseup"', '"mouseleave"',
                    '"mouseover"', '"click"', '"change"'):
        assert handler in source, handler
    # A drag must not also pin the point under the pointer.
    assert "dragged" in source
    # Reset clears every filter, the pins, and the brush.
    assert "state.brush = null" in source
