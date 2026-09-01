"""index.html — one self-contained page (design §6).

Two hard rules. **Offline:** every byte the browser needs is in the file; a
generated page containing an external URL is a test failure, not a warning.
**Decisions in Python:** the initial hero is a baked SVG *and* a plotspec (D8);
app.js redraws from the same plotspec on interaction, so drawing exists twice
but deciding exists once.
"""
import html
import json
import math
import os
from typing import Any, Dict, List, Optional

import commands
from analysis import (Analysis, HERO_CARD, color_values, context_line,
                      describe_config, fmt, gpu_count, gpu_source, knob_kind,
                      knob_value, metric_is_numeric, page_banners,
                      provenance_bits, screen_fractions, strip_cells,
                      summary_bits)
from data import FIELD_KNOB, FIELD_METRIC, encode_columnar
from render_mpl import PALETTE

ASSETS = os.path.join(os.path.dirname(os.path.abspath(__file__)), "assets")

CLOUD_COLOR = "#c8ccd4"
FRONT_COLOR = "#2b6cb0"
KNEE_COLOR = "#b7791f"
BAND_COLOR = "#f56565"

# SVG geometry, in user units. The viewBox scales to the container width.
_W, _H = 820.0, 440.0
_ML, _MR, _MT, _MB = 74.0, 24.0, 18.0, 52.0


def read_asset(name: str) -> str:
    """Read one asset for inlining. Missing assets are an error, not a silent skip."""
    with open(os.path.join(ASSETS, name)) as f:
        return f.read()


def esc(text: Any) -> str:
    return html.escape(str(text), quote=True)


def embed_json(obj: Any) -> str:
    r"""Serialise for a <script type="application/json"> block.

    HTML does not escape a script element's content, so the browser ends the
    block at the first literal `</script>` — and knob values, metric names and
    deploy commands are all dataset text. One config value containing
    `</script>` would terminate the JSON early and everything after it would be
    parsed as live HTML. Escaping `<`, `>` and `&` as \uXXXX keeps the JSON
    valid and lossless (JSON.parse restores the characters) while leaving no
    sequence the HTML parser can act on. Key-sorted and separator-pinned so the
    page stays byte-identical.
    """
    return (json.dumps(obj, sort_keys=True, separators=(",", ":"))
            .replace("<", "\\u003c").replace(">", "\\u003e")
            .replace("&", "\\u0026"))


# ------------------------------------------------------------- plotspec


def _field_spec(an: Analysis) -> Dict[str, Any]:
    """The advanced panel's option lists and filter bounds, decided in Python.

    app.js populates nothing itself: which fields may go on an axis, which are
    categorical, and what each numeric range is are all decisions, and decisions
    live here (§4.2).
    """
    ds = an.ds
    metrics = [FIELD_METRIC + m for m in ds.metric_names()
               if metric_is_numeric(ds, m)]
    numeric_knobs = [FIELD_KNOB + k for k in ds.knob_names()
                     if knob_kind(ds, k) == "numeric"]
    categorical: Dict[str, List[str]] = {}
    for name in ds.knob_names():
        if knob_kind(ds, name) != "numeric":
            values = sorted({knob_value(ds.points[i], name) for i in an.plotted},
                            key=lambda v: (v is None, str(v)))
            categorical[FIELD_KNOB + name] = [_value_text(v) for v in values]
    ranges: Dict[str, List[float]] = {}
    for field in metrics + numeric_knobs:
        raw = [_field_value(ds, field, i) for i in an.plotted]
        numbers = [float(v) for v in raw if isinstance(v, (int, float))
                   and not isinstance(v, bool)]
        if numbers:
            ranges[field] = [min(numbers), max(numbers)]
    return {"metrics": metrics, "numeric_knobs": numeric_knobs,
            "categorical": categorical, "ranges": ranges}


def _field_value(ds, field: str, idx: int) -> Any:
    if field.startswith(FIELD_KNOB):
        return ds.points[idx].config.get(field[len(FIELD_KNOB):])
    return ds.points[idx].metrics.get(field[len(FIELD_METRIC):])


def _value_text(value: Any) -> str:
    """The string app.js compares against, matching JS String() for these types."""
    if value is None:
        return "null"
    if isinstance(value, bool):
        return "true" if value else "false"
    return str(value)


def _axis_spec(an: Analysis, which: str) -> Optional[Dict[str, Any]]:
    axis = an.axes.x if which == "x" else an.axes.y
    if axis is None:
        return None
    values = [float(p.metrics[axis.key]) for p in an.ds.points]
    return {"key": axis.key, "label": an.axes.x_label if which == "x" else an.axes.y_label,
            "direction": axis.direction,
            "invert": an.axes.x_invert if which == "x" else an.axes.y_invert,
            "domain": [min(values), max(values)]}


def plotspec(an: Analysis, flag_source) -> Dict[str, Any]:
    """Everything a redraw needs, decided in Python (D8).

    Row positions are indices into `data.r`; `index` maps them back to dataset
    indices so a hover can name the same point the picks table names.
    """
    ds = an.ds
    position = {idx: row for row, idx in enumerate(an.plotted)}
    picks = []
    for pick in an.picks.rows:
        cmd = commands.deploy_command(ds, pick.index, flag_source.flags)
        picks.append({"row": position.get(pick.index), "titles": list(pick.titles),
                      "label": describe_config(ds, pick.index),
                      "command": commands.command_text(cmd) if cmd else "",
                      "gpus": gpu_count(ds, pick.index)})
    return {
        "data": encode_columnar(ds, an.plotted),
        "index": list(an.plotted),
        "geom": {"w": _W, "h": _H, "ml": _ML, "mr": _MR, "mt": _MT, "mb": _MB},
        "fields": _field_spec(an),
        "gpus": [gpu_count(ds, i) for i in an.plotted],
        "gpu_values": sorted({c for c in (gpu_count(ds, i) for i in an.plotted)
                              if c is not None}),
        "axes": {"x": _axis_spec(an, "x"), "y": _axis_spec(an, "y")},
        "color": {"key": an.color.key, "kind": an.color.kind, "label": an.color.label,
                  "caption": an.color.caption, "values": list(an.color.values),
                  "palette": list(PALETTE)},
        "objectives": [{"key": o.key, "direction": o.direction} for o in ds.objectives],
        "slo": [{"metric": c["metric"], "op": c["op"],
                 "threshold": float(c["threshold"])} for c in ds.slo_constraints],
        "hero": {"form": an.hero.form, "caption": an.hero.caption,
                 "with_scatter": an.hero.with_scatter,
                 "with_distribution": an.hero.with_distribution},
        "knee": position.get(an.knee.index) if an.knee.index is not None else None,
        "knee_note": an.knee.note,
        "picks": picks,
        "captions": {"axes": an.axes.caption, "color": an.color.caption,
                     "plot_notes": list(an.plot_notes)},
        "strips": [{"name": s.name, "kind": s.kind, "score": s.score,
                    "headline": s.headline, "cells": strip_cells(s)}
                   for s in an.strips],
        "colors": {"cloud": CLOUD_COLOR, "front": FRONT_COLOR, "knee": KNEE_COLOR,
                   "band": BAND_COLOR},
    }


# ------------------------------------------------------------- baked SVG


def _point_fill(an: Analysis, idx: int) -> str:
    choice = an.color
    if choice is None or choice.kind == "none":
        return FRONT_COLOR
    value = color_values(an.ds, choice, [idx])[0]
    if choice.kind == "categorical":
        order = {v: i for i, v in enumerate(choice.values)}
        return PALETTE[order.get(value, 0) % len(PALETTE)]
    numbers = [v for v in color_values(an.ds, choice, an.ds.front_indices())
               if isinstance(v, (int, float)) and not isinstance(v, bool)]
    if not numbers or max(numbers) == min(numbers) or not isinstance(value, (int, float)):
        return FRONT_COLOR
    lo, hi = min(numbers), max(numbers)
    return _ramp((float(value) - lo) / (hi - lo))


def _ramp(t: float) -> str:
    """A five-stop viridis-like ramp. Baked so the page needs no colormap code."""
    stops = ["#440154", "#3b528b", "#21918c", "#5ec962", "#fde725"]
    t = min(1.0, max(0.0, t))
    return stops[min(len(stops) - 1, int(t * len(stops)))]


def _px(an: Analysis, indices: List[int]) -> List[Any]:
    """Screen coordinates from the one orientation rule (analysis.screen_fractions)."""
    width, height = _W - _ML - _MR, _H - _MT - _MB
    return [(_ML + fx * width, _MT + fy * height)
            for fx, fy in screen_fractions(an.ds, an.axes, indices)]


def hero_svg(an: Analysis) -> str:
    """The initial hero, baked so the page renders with JavaScript disabled."""
    if an.hero.form == HERO_CARD or an.axes.y is None:
        return _card_svg(an)
    parts = [f'<svg viewBox="0 0 {_W:g} {_H:g}" role="img" '
             f'aria-label="{esc(an.hero.caption)}" id="hero">']
    parts.append(f'<rect x="{_ML:g}" y="{_MT:g}" width="{_W - _ML - _MR:g}" '
                 f'height="{_H - _MT - _MB:g}" fill="none" stroke="#e2e8f0"/>')
    parts.extend(_band_svg(an))

    for x, y in _px(an, an.cloud_plotted):
        parts.append(f'<circle cx="{x:.1f}" cy="{y:.1f}" r="2.6" '
                     f'fill="{CLOUD_COLOR}"/>')
    front = an.ds.front_indices()
    for idx, (x, y) in zip(front, _px(an, front)):
        parts.append(f'<circle cx="{x:.1f}" cy="{y:.1f}" r="5" '
                     f'fill="{_point_fill(an, idx)}" stroke="#fff" stroke-width="1"/>')
    if an.knee.index is not None:
        (kx, ky), = _px(an, [an.knee.index])
        parts.append(f'<path d="{_star(kx, ky, 9.0)}" fill="{KNEE_COLOR}" '
                     'stroke="#fff" stroke-width="1"/>')

    parts.extend(_axis_svg(an))
    parts.append("</svg>")
    return "".join(parts)


def _band_svg(an: Analysis) -> List[str]:
    """Shade the SLO-infeasible region, in fraction space so it needs no scale."""
    out = []
    width, height = _W - _ML - _MR, _H - _MT - _MB
    for constraint in an.ds.slo_constraints:
        metric, op = constraint["metric"], constraint["op"]
        threshold = float(constraint["threshold"])
        for which, axis in (("x", an.axes.x), ("y", an.axes.y)):
            if axis is None or axis.key != metric:
                continue
            values = [float(p.metrics[axis.key]) for p in an.ds.points]
            lo, hi = min(values), max(values)
            if hi == lo:
                continue
            frac = (threshold - lo) / (hi - lo)
            frac = min(1.0, max(0.0, frac))
            good_high = axis.direction == "maximize"
            # In fraction space "better" already points right/down, so the
            # infeasible side is always the low-fraction side.
            cut = frac if good_high else 1.0 - frac
            if which == "x":
                out.append(f'<rect x="{_ML:g}" y="{_MT:g}" width="{cut * width:.1f}" '
                           f'height="{height:g}" fill="{BAND_COLOR}" opacity="0.07"/>')
            else:
                out.append(f'<rect x="{_ML:g}" y="{_MT:g}" width="{width:g}" '
                           f'height="{cut * height:.1f}" fill="{BAND_COLOR}" '
                           'opacity="0.07"/>')
    return out


def _axis_svg(an: Analysis) -> List[str]:
    xs = [float(p.metrics[an.axes.x.key]) for p in an.ds.points]
    ys = [float(p.metrics[an.axes.y.key]) for p in an.ds.points]
    left = max(xs) if an.axes.x_invert else min(xs)
    right = min(xs) if an.axes.x_invert else max(xs)
    top = min(ys) if an.axes.y_invert else max(ys)
    bottom = max(ys) if an.axes.y_invert else min(ys)
    mid_x, mid_y = _ML + (_W - _ML - _MR) / 2.0, _MT + (_H - _MT - _MB) / 2.0
    return [
        f'<text x="{_ML:g}" y="{_H - _MB + 16:g}" font-size="11" fill="#4a5568">'
        f'{esc(fmt(left))}</text>',
        f'<text x="{_W - _MR:g}" y="{_H - _MB + 16:g}" font-size="11" fill="#4a5568" '
        f'text-anchor="end">{esc(fmt(right))}</text>',
        f'<text x="{mid_x:g}" y="{_H - _MB + 34:g}" font-size="12" fill="#1a202c" '
        f'text-anchor="middle">{esc(an.axes.x_label)}</text>',
        f'<text x="{_ML - 8:g}" y="{_MT + 12:g}" font-size="11" fill="#4a5568" '
        f'text-anchor="end">{esc(fmt(top))}</text>',
        f'<text x="{_ML - 8:g}" y="{_H - _MB:g}" font-size="11" fill="#4a5568" '
        f'text-anchor="end">{esc(fmt(bottom))}</text>',
        f'<text x="14" y="{mid_y:g}" font-size="12" fill="#1a202c" '
        f'text-anchor="middle" transform="rotate(-90 14 {mid_y:g})">'
        f'{esc(an.axes.y_label)}</text>',
    ]


def _star(cx: float, cy: float, r: float) -> str:
    """A five-pointed star as a path, so no font or marker library is needed."""
    points = []
    for i in range(10):
        angle = -math.pi / 2 + i * math.pi / 5
        radius = r if i % 2 == 0 else r * 0.45
        points.append(f"{cx + radius * math.cos(angle):.1f},"
                      f"{cy + radius * math.sin(angle):.1f}")
    return "M" + "L".join(points) + "Z"


def _card_svg(an: Analysis) -> str:
    """One objective: the winner in words over a histogram of the cloud (§9)."""
    ds = an.ds
    obj = ds.objectives[0]
    best = an.picks.rows[0].index if an.picks.rows else ds.front_indices()[0]
    lines = [describe_config(ds, best),
             f"{obj.key} = {fmt(ds.value(ds.points[best], obj))}"]
    gpus = gpu_count(ds, best)
    if gpus is not None:
        lines.append(f"{gpus} GPUs")
    parts = [f'<svg viewBox="0 0 {_W:g} 220" role="img" '
             f'aria-label="{esc(an.hero.caption)}" id="hero">']
    for i, line in enumerate(lines):
        parts.append(f'<text x="{_ML:g}" y="{40 + i * 30:g}" font-size="{22 - i * 5:g}" '
                     f'fill="{FRONT_COLOR}">{esc(line)}</text>')
    if an.hero.with_distribution and an.cloud_plotted:
        values = [float(ds.points[i].metrics[obj.key]) for i in an.cloud_plotted]
        lo, hi = min(values), max(values)
        span = (hi - lo) or 1.0
        bins = [0] * 16
        for value in values:
            bins[min(15, int((value - lo) / span * 16))] += 1
        tallest = max(bins) or 1
        width = (_W - _ML - _MR) / 16.0
        for i, count in enumerate(bins):
            height = 70.0 * count / tallest
            parts.append(f'<rect x="{_ML + i * width:.1f}" y="{190 - height:.1f}" '
                         f'width="{width * 0.9:.1f}" height="{height:.1f}" '
                         f'fill="{CLOUD_COLOR}"/>')
        parts.append(f'<text x="{_ML:g}" y="206" font-size="11" fill="#4a5568">'
                     f'{esc(fmt(lo))}</text>')
        parts.append(f'<text x="{_W - _MR:g}" y="206" font-size="11" fill="#4a5568" '
                     f'text-anchor="end">{esc(fmt(hi))}</text>')
    parts.append("</svg>")
    return "".join(parts)


# ------------------------------------------------------------- the page


def _legend_html(an: Analysis) -> str:
    items = []
    if an.cloud_plotted:
        items.append((CLOUD_COLOR, f"dominated ({len(an.cloud_plotted)})"))
    if an.color is not None and an.color.kind == "categorical":
        drawn = set(color_values(an.ds, an.color, an.ds.front_indices()))
        for i, value in enumerate(v for v in an.color.values if v in drawn):
            items.append((PALETTE[i % len(PALETTE)], f"{esc(an.color.label)} {esc(value)}"))
    else:
        items.append((FRONT_COLOR, f"on front ({len(an.ds.front_indices())})"))
    if an.knee.index is not None:
        items.append((KNEE_COLOR, "best balance"))
    if an.ds.slo_constraints:
        items.append((BAND_COLOR, "SLO infeasible"))
    spans = "".join(f'<span><i style="background:{c}"></i>{esc(label)}</span>'
                    for c, label in items)
    return f'<div class="legend">{spans}</div>'


def _picks_html(an: Analysis, flag_source) -> str:
    ds = an.ds
    if not an.picks.rows:
        return "<p>No configs to pick from.</p>"
    has_gpu = bool(gpu_source(ds))
    head = ["Pick", "Config"] + [f"{o.key} {o.arrow}".strip() for o in ds.objectives]
    if has_gpu:
        head.append("GPUs")
    rows = ["<tr>" + "".join(
        f'<th class="{"num" if i >= 2 else ""}">{esc(h)}</th>'
        for i, h in enumerate(head)) + "</tr>"]
    blocks = []
    for pick in an.picks.rows:
        cells = [esc(", ".join(pick.titles)), esc(describe_config(ds, pick.index))]
        cells += [f'<td class="num">{esc(fmt(ds.value(ds.points[pick.index], o)))}</td>'
                  for o in ds.objectives]
        gpu_cell = (f'<td class="num">{esc(gpu_count(ds, pick.index))}</td>'
                    if has_gpu else "")
        rows.append(f"<tr><td>{cells[0]}</td><td>{cells[1]}</td>"
                    + "".join(cells[2:]) + gpu_cell + "</tr>")
        cmd = commands.deploy_command(ds, pick.index, flag_source.flags)
        text = commands.command_text(cmd) if cmd else commands.NO_FLAGS_NOTE
        blocks.append(f"<h3>{esc(', '.join(pick.titles))} — "
                      f"{esc(describe_config(ds, pick.index))}</h3>"
                      f"<pre>{esc(text)}</pre>")
    notes = "".join(f"<li>{esc(n)}</li>" for n in an.picks.notes)
    notes_html = f'<ul class="notes">{notes}</ul>' if notes else ""
    return (f"<table>{''.join(rows)}</table>{notes_html}" + "".join(blocks))


def _knobs_html(an: Analysis) -> str:
    if not an.strips:
        return "<p>This file records no configuration knobs.</p>"
    rows = ['<tr><th>Knob</th><th>Front</th><th class="num">Separation</th>'
            "<th>What the front does</th></tr>"]
    for strip in an.strips:
        cells = "".join(
            f'<i style="opacity:{0.15 + 0.85 * c:.2f}"></i>'
            for c in strip_cells(strip))
        rows.append(f"<tr><td>{esc(strip.name)}</td>"
                    f'<td><span class="strip">{cells}</span></td>'
                    f'<td class="num">{strip.score:.2f}</td>'
                    f"<td>{esc(strip.headline)}</td></tr>")
    lead = ""
    if not an.ds.has_cloud:
        lead = ('<p class="sub">No dominated cloud to compare against, so these are '
                "ranges the front occupies, not attributions.</p>")
    return lead + f"<table>{''.join(rows)}</table>"


def _options(fields: List[str], selected: Optional[str],
             blank: Optional[str] = None) -> str:
    out = []
    if blank is not None:
        out.append(f'<option value="">{esc(blank)}</option>')
    for field in fields:
        mark = " selected" if field == selected else ""
        out.append(f'<option value="{esc(field)}"{mark}>{esc(field[2:])}</option>')
    return "".join(out)


def _advanced_html(an: Analysis, spec: Dict[str, Any]) -> str:
    """The panel §6 describes: axes, log toggles, color, show, and filters.

    Every option list and every slider bound comes from the plotspec, so the
    panel can offer nothing the data does not support.
    """
    fields = spec["fields"]
    axis_fields = fields["metrics"] + fields["numeric_knobs"]
    color_fields = axis_fields + sorted(fields["categorical"])
    x_selected = FIELD_METRIC + an.axes.x.key
    y_selected = FIELD_METRIC + an.axes.y.key if an.axes.y else None
    if an.color.key == "gpus":
        color_selected = "gpus"
    elif an.color.key:
        color_selected = (FIELD_METRIC + an.color.key
                          if FIELD_METRIC + an.color.key in fields["metrics"]
                          else FIELD_KNOB + an.color.key)
    else:
        color_selected = "none"

    color_options = ['<option value="none">front / cloud only</option>']
    if spec["gpu_values"]:
        mark = " selected" if color_selected == "gpus" else ""
        color_options.append(f'<option value="gpus"{mark}>GPU count</option>')
    color_options.append(_options(color_fields, color_selected))

    sliders = []
    for field in axis_fields:
        span = fields["ranges"].get(field)
        if not span:
            continue
        step = (span[1] - span[0]) / 100.0 or 1.0
        sliders.append(
            f'<label>{esc(field[2:])}'
            f'<input type="range" data-range="{esc(field)}" data-edge="lo" '
            f'min="{span[0]}" max="{span[1]}" step="{step}" value="{span[0]}" '
            f'data-default="{span[0]}">'
            f'<input type="range" data-range="{esc(field)}" data-edge="hi" '
            f'min="{span[0]}" max="{span[1]}" step="{step}" value="{span[1]}" '
            f'data-default="{span[1]}"></label>')
    boxes = []
    for field, values in sorted(fields["categorical"].items()):
        checks = "".join(
            f'<label><input type="checkbox" data-knob="{esc(field)}" '
            f'value="{esc(v)}" checked>{esc(v)}</label>' for v in values)
        boxes.append(f'<div class="filter"><span>{esc(field[2:])}</span>{checks}</div>')

    return f"""<div class="controls">
<label>x <select id="pick-x">{_options(axis_fields, x_selected)}</select></label>
<label><input type="checkbox" id="log-x"> log x</label>
<label>y <select id="pick-y">{_options(axis_fields, y_selected, blank="(none)")}</select></label>
<label><input type="checkbox" id="log-y"> log y</label>
<label>color <select id="pick-color">{''.join(color_options)}</select></label>
<label>show <select id="pick-show">
<option value="both" selected>front + cloud</option>
<option value="front">front only</option>
<option value="slo">SLO-feasible only</option>
</select></label>
<button type="button" id="reset">reset</button>
<span class="sub" id="count"></span>
</div>
<div id="filters" class="filters">{''.join(sliders)}{''.join(boxes)}</div>
<div class="split">
<div id="hover" class="hover sub">Hover a point for its full config and metrics.</div>
<div id="pins" class="pins"></div>
</div>"""


def document(an: Analysis, flag_source) -> str:
    """The whole page, as one string: no I/O, no network, no timestamp (§10)."""
    spec_obj = plotspec(an, flag_source)
    spec = embed_json(spec_obj)
    banners = "".join(f'<div class="banner">{esc(b)}</div>'
                      for b in page_banners(an, flag_source))
    caption = "\n".join([an.hero.caption, an.axes.caption]
                        + ([an.color.caption] if an.color.caption else [])
                        + an.plot_notes)
    return f"""<!doctype html>
<html lang="en">
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>BLIS search — {esc(context_line(an.ds))}</title>
<style>
{read_asset('style.css')}
</style>
</head>
<body>
<main>
<section>
<h1>{esc(context_line(an.ds))}</h1>
<div class="sub">{esc(" · ".join(summary_bits(an.ds)))}</div>
{banners}
</section>
<section>
<h2>Tradeoff</h2>
{hero_svg(an)}
{_legend_html(an)}
<div class="caption">{esc(caption)}</div>
</section>
<section>
<h2>Picks <span class="sub">({esc(an.picks.header)})</span></h2>
{_picks_html(an, flag_source)}
</section>
<section>
<h2>Which knobs the front exploits</h2>
{_knobs_html(an)}
</section>
<section>
<h2>All configs
<button type="button" id="toggle">Advanced ▸</button>
</h2>
<div id="advanced" hidden>
{_advanced_html(an, spec_obj)}
</div>
<table id="configs"></table>
<noscript><p class="sub">The table and the advanced panel need JavaScript. The
chart above, the picks, and the knob strips are baked into this file and need
nothing.</p></noscript>
</section>
</main>
<footer>{esc(" · ".join(provenance_bits(an)))}</footer>
<script type="application/json" id="plotspec">{spec}</script>
<script>
{read_asset('app.js')}
</script>
</body>
</html>
"""
