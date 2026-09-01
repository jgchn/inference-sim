"""index.html: the offline guarantee, the banners, and plotspec/SVG agreement."""
import json
import re

import pytest

import analysis
import commands
import data
import render_html
from conftest import VALID_FIXTURES, fixture_path, load_raw

#: Anything that would make the browser reach out. A hit here is a hard failure.
EXTERNAL = re.compile(r"https?://|//[a-z0-9.-]+\.[a-z]{2,}/|\bsrc\s*=|@import")


def page(name, space=None):
    ds = data.load(fixture_path(name))
    an = analysis.analyse(ds)
    return an, render_html.document(an, commands.resolve_param_flags(ds, space))


def embedded_spec(text):
    match = re.search(r'id="plotspec">(.*?)</script>', text, re.S)
    assert match, "the page must carry its plotspec"
    return json.loads(match.group(1))


def baked_svg(text):
    """Only the baked hero. Task 16 inlines app.js, whose source contains the
    string '<circle ' — counting elements across the whole page would count the
    script's own markup templates too."""
    start = text.index("<svg viewBox=")
    return text[start:text.index("</svg>", start)]


@pytest.mark.parametrize("name", VALID_FIXTURES)
def test_the_page_contains_no_external_reference(name):
    """§11's offline guarantee, as an assertion rather than a promise."""
    _, text = page(name)
    assert not EXTERNAL.search(text), EXTERNAL.search(text).group(0)


@pytest.mark.parametrize("name", VALID_FIXTURES)
def test_every_fixture_renders_a_complete_page(name):
    an, text = page(name)
    assert text.startswith("<!doctype html>")
    assert "<style>" in text and "</style>" in text          # css inlined
    assert "<svg" in text                                     # hero baked
    assert an.picks.header in text
    assert an.ds.source_sha256[:8] in text


@pytest.mark.parametrize("name", VALID_FIXTURES)
def test_the_page_is_byte_identical_across_renders(name):
    """§10: no timestamp anywhere, so identical input yields identical HTML."""
    _, first = page(name)
    _, second = page(name)
    assert first == second


@pytest.mark.parametrize("name", VALID_FIXTURES)
def test_the_embedded_json_is_key_sorted_and_html_inert(name):
    """§10: an unordered dict dump would break byte-identity. And because HTML
    does not escape a script element's content, the blob must contain nothing the
    HTML parser can act on — otherwise a knob value holding `</script>` would end
    the block early and the rest of the page would be parsed as live markup."""
    _, text = page(name)
    raw = re.search(r'id="plotspec">(.*?)</script>', text, re.S).group(1)
    spec = json.loads(raw)
    assert raw == render_html.embed_json(spec)
    for char in "<>&":
        assert char not in raw, char


@pytest.mark.parametrize("name", VALID_FIXTURES)
def test_the_baked_svg_and_the_plotspec_describe_the_same_points(name):
    """D8's law. If these two ever disagree, the page lies to the reader as soon
    as the first interaction redraws it."""
    an, text = page(name)
    spec = embedded_spec(text)
    assert len(spec["data"]["r"]) == len(an.plotted)
    assert len(spec["data"]["front"]) == len(an.ds.front_indices())
    svg = baked_svg(text)
    if an.hero.form != analysis.HERO_CARD:
        assert len(re.findall(r"<circle ", svg)) == len(an.plotted)
    stars = len(re.findall(r'<path d="M', svg))
    if an.knee.index is None:
        assert spec["knee"] is None and stars == 0
    else:
        assert spec["knee"] == an.plotted.index(an.knee.index)
        assert stars == 1


@pytest.mark.parametrize("name", VALID_FIXTURES)
def test_the_columnar_payload_decodes_back_to_the_plotted_points(name):
    an, text = page(name)
    rows = data.decode_columnar(embedded_spec(text)["data"])
    assert len(rows) == len(an.plotted)
    first = rows[0]
    original = an.ds.points[an.plotted[0]]
    for key, value in original.metrics.items():
        assert first[data.FIELD_METRIC + key] == value


@pytest.mark.parametrize("name", VALID_FIXTURES)
def test_every_banner_reaches_the_page(name):
    an, text = page(name)
    for banner in an.banners:
        assert render_html.esc(banner) in text
    for note in an.picks.notes:
        assert render_html.esc(note) in text


def test_a_file_without_flags_shows_the_note_instead_of_a_command():
    _, text = page("front_only.json")
    assert render_html.esc(commands.NO_FLAGS_NOTE) in text


def test_the_commands_are_html_escaped():
    """A config value containing < or & must not break the page."""
    _, text = page("two_obj_slo.json")
    assert "<pre>./blis run " in text
    body = text.split("<pre>")[1].split("</pre>")[0]
    assert "<script" not in body


def test_the_slo_band_appears_only_when_constraints_exist():
    with_slo, text_with = page("two_obj_slo.json")
    assert with_slo.ds.slo_constraints
    assert "SLO infeasible" in text_with

    without, text_without = page("no_gpu_dimension.json")
    assert not without.ds.slo_constraints
    assert "SLO infeasible" not in text_without


def test_a_single_objective_page_bakes_the_card_not_a_scatter():
    an, text = page("single_objective.json")
    svg = baked_svg(text)
    assert an.hero.form == analysis.HERO_CARD
    assert "<circle " not in svg                  # no scatter
    assert "<rect " in svg                        # the cloud histogram


def test_the_page_states_decimation_in_its_provenance():
    ds = data.load(fixture_path("two_obj_slo.json"))
    an = analysis.analyse(ds, max_points=8)
    text = render_html.document(an, commands.resolve_param_flags(ds))
    assert an.plot_notes
    assert render_html.esc(an.plot_notes[0]) in text


def test_a_hostile_config_value_cannot_break_out_of_the_page():
    """A knob value containing `</script>` is the one input that can turn this
    page into an injection: inside a script element the HTML parser stops at the
    first literal `</script>`, whatever the JSON says. It must survive as data."""
    payload = "<script>alert(1)</script>"
    raw = load_raw("two_obj_slo.json")
    for collection in ("all_results", "pareto_front"):
        for entry in raw[collection]:
            if entry["config"].get("scheduler") == "sjf":
                entry["config"]["scheduler"] = payload
    ds = data.build_dataset(raw, source_sha256="0" * 64,
                            source_name="hostile.json")
    an = analysis.analyse(ds, color_override="scheduler")
    text = render_html.document(an, commands.resolve_param_flags(ds))

    assert payload not in text                    # never present literally
    blob = re.search(r'id="plotspec">(.*?)</script>', text, re.S).group(1)
    assert payload in json.loads(blob)["data"]["s"]["c:scheduler"]   # kept as data

# --------------------------------------------------- the advanced panel


@pytest.mark.parametrize("name", VALID_FIXTURES)
def test_the_panel_offers_only_fields_the_data_supports(name):
    an, text = page(name)
    spec = embedded_spec(text)
    fields = spec["fields"]
    assert fields["metrics"], "there is always at least one numeric metric"
    for field in fields["metrics"]:
        assert field.startswith(data.FIELD_METRIC)
        assert field[2:] in an.ds.metric_names()
    for field in fields["numeric_knobs"]:
        assert analysis.knob_kind(an.ds, field[2:]) == "numeric"
    for field in fields["categorical"]:
        assert analysis.knob_kind(an.ds, field[2:]) == "categorical"
    for field, span in fields["ranges"].items():
        assert span[0] <= span[1]
    # Every offered axis field appears as an <option>.
    for field in fields["metrics"] + fields["numeric_knobs"]:
        assert f'value="{field}"' in text


@pytest.mark.parametrize("name", VALID_FIXTURES)
def test_the_geometry_is_shared_not_duplicated(name):
    """app.js must place points where the baked SVG placed them, so the geometry
    is a Python decision like every other."""
    _, text = page(name)
    geom = embedded_spec(text)["geom"]
    assert set(geom) == {"w", "h", "ml", "mr", "mt", "mb"}
    assert f'viewBox="0 0 {geom["w"]:g}' in text


@pytest.mark.parametrize("name", VALID_FIXTURES)
def test_the_page_carries_the_script_and_the_controls(name):
    _, text = page(name)
    assert 'id="advanced"' in text and "hidden" in text
    for control in ('id="pick-x"', 'id="pick-y"', 'id="log-x"', 'id="log-y"',
                    'id="pick-color"', 'id="pick-show"', 'id="reset"',
                    'id="filters"', 'id="hover"', 'id="pins"', 'id="configs"',
                    'id="toggle"'):
        assert control in text, control
    assert "front only" in text and "SLO-feasible only" in text
    assert "<noscript>" in text                    # the page says what needs JS


@pytest.mark.parametrize("name", VALID_FIXTURES)
def test_the_panel_does_not_break_the_offline_guarantee(name):
    """app.js builds SVG as markup precisely so this stays true."""
    _, text = page(name)
    assert not EXTERNAL.search(text)


def test_gpu_colouring_is_available_only_when_a_count_exists():
    with_gpu, text_with = page("two_obj_slo.json")
    assert embedded_spec(text_with)["gpu_values"]
    assert '<option value="gpus"' in text_with

    without, text_without = page("no_gpu_dimension.json")
    assert embedded_spec(text_without)["gpu_values"] == []
    assert '<option value="gpus"' not in text_without


def test_categorical_filter_values_match_what_javascript_will_compare():
    """JS compares String(value); null must arrive as "null", not "None"."""
    _, text = page("two_obj_slo.json")
    categorical = embedded_spec(text)["fields"]["categorical"]
    flat = [v for values in categorical.values() for v in values]
    assert "None" not in flat
    assert any(v == "null" for v in flat)          # unset knobs are real values
    assert all(isinstance(v, str) for v in flat)
