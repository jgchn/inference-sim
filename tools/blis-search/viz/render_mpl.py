"""pareto.png, knobs.png, parallel.png — the print-resolution figures (design §3).

Deterministic by construction: the Agg backend, a pinned rcParams block, and
explicit empty PNG metadata, so two runs on one machine produce byte-identical
files (§10). Every number, caption, and choice comes from the Analysis bundle;
this module only draws.
"""
from typing import Any, Dict, List, Optional, Tuple

import matplotlib
matplotlib.use("Agg")                                          # noqa: E402
import matplotlib.pyplot as plt                                # noqa: E402
from matplotlib.lines import Line2D                            # noqa: E402

from analysis import (Analysis, HERO_CARD, axis_label, color_values,
                      describe_config, fmt, goodness, gpu_count, strip_cells)

#: Pinned so a figure is a function of the data, not of the user's matplotlibrc.
RCPARAMS = {
    "figure.figsize": (9.0, 5.5),
    "figure.dpi": 150,
    "savefig.dpi": 150,
    "font.size": 9,
    "axes.grid": True,
    "grid.alpha": 0.25,
    "grid.linewidth": 0.5,
    "axes.spines.top": False,
    "axes.spines.right": False,
    "svg.hashsalt": "blis-search-viz",
    "path.simplify": False,
}

#: Explicit and empty: matplotlib otherwise stamps its version into the PNG.
PNG_METADATA = {"Software": None}

CLOUD_COLOR = "#c8ccd4"
FRONT_COLOR = "#2b6cb0"
KNEE_COLOR = "#b7791f"
BAND_COLOR = "#f56565"

#: Categorical palette, colorblind-safe order, cycled if a knob has more values.
PALETTE = ["#2b6cb0", "#dd6b20", "#38a169", "#805ad5", "#d53f8c", "#319795",
           "#b7791f", "#4a5568"]


def save(fig, path: str) -> None:
    """One writing path, so every figure gets the same deterministic treatment."""
    fig.savefig(path, format="png", metadata=PNG_METADATA, bbox_inches="tight")
    plt.close(fig)


def _categorical_colors(values: List[Any], legend: List[Any]) -> List[str]:
    order = {v: i for i, v in enumerate(legend)}
    return [PALETTE[order.get(v, 0) % len(PALETTE)] for v in values]


def _numeric_values(values: List[Any]) -> Optional[List[float]]:
    """Raw floats for a continuous ramp, so the colorbar reads in real units.
    None when the column is unusable (no values, or every value identical)."""
    clean = [float(v) for v in values if isinstance(v, (int, float))
             and not isinstance(v, bool)]
    if not clean or max(clean) == min(clean):
        return None
    floor = min(clean)
    return [float(v) if isinstance(v, (int, float)) and not isinstance(v, bool)
            else floor for v in values]


def _point_colors(an: Analysis, indices: List[int]) -> Tuple[Any, Optional[str]]:
    """(color argument for scatter, colormap name). Never decides the channel."""
    choice = an.color
    if choice is None or choice.kind == "none":
        return FRONT_COLOR, None
    vals = color_values(an.ds, choice, indices)
    if choice.kind == "categorical":
        return _categorical_colors(vals, choice.values), None
    numeric = _numeric_values(vals)
    if numeric is None:
        return FRONT_COLOR, None
    return numeric, "viridis"


def _caption(an: Analysis) -> str:
    bits = [an.hero.caption, an.axes.caption]
    # The card has no color channel to explain.
    if (an.hero.form != HERO_CARD and an.color is not None and an.color.caption):
        bits.append(an.color.caption)
    bits.extend(an.plot_notes)
    return "\n".join(b for b in bits if b)


def _title(an: Analysis) -> str:
    ctx = an.ds.context
    head = " · ".join(str(ctx[k]) for k in ("model", "hardware", "workload") if k in ctx)
    return head or an.ds.source_name


def _raw(an: Analysis, key: str, indices: List[int]) -> List[float]:
    return [float(an.ds.points[i].metrics[key]) for i in indices]


def _slo_bands(an: Analysis, ax) -> bool:
    """Shade the region no config may occupy. Returns whether anything was drawn."""
    drawn = False
    for constraint in an.ds.slo_constraints:
        metric, op = constraint["metric"], constraint["op"]
        threshold = float(constraint["threshold"])
        if metric == an.axes.x.key:
            lo, hi = ax.get_xlim()
            span = (threshold, max(hi, threshold)) if op == "<" else (min(lo, threshold), threshold)
            ax.axvspan(*span, color=BAND_COLOR, alpha=0.08, zorder=0)
            drawn = True
        elif an.axes.y is not None and metric == an.axes.y.key:
            lo, hi = ax.get_ylim()
            span = (threshold, max(hi, threshold)) if op == "<" else (min(lo, threshold), threshold)
            ax.axhspan(*span, color=BAND_COLOR, alpha=0.08, zorder=0)
            drawn = True
    return drawn


def _scatter(an: Analysis, ax) -> None:
    ds = an.ds
    xkey = an.axes.x.key
    ykey = an.axes.y.key

    cloud = [i for i in an.cloud_plotted]
    if cloud:
        ax.scatter(_raw(an, xkey, cloud), _raw(an, ykey, cloud), s=12,
                   c=CLOUD_COLOR, edgecolors="none", zorder=1)

    front = ds.front_indices()
    if front:
        ordered = sorted(front, key=lambda i: (float(ds.points[i].metrics[xkey]), i))
        if len(ds.objectives) == 2:
            # A staircase is only truthful for a 2D front. With 3+ objectives this
            # is a projection, and some plotted members are dominated *in the
            # projection* — joining them would draw a frontier that does not exist.
            ax.plot(_raw(an, xkey, ordered), _raw(an, ykey, ordered),
                    drawstyle="steps-post", color=FRONT_COLOR, linewidth=1.2,
                    alpha=0.7, zorder=2)
        colors, cmap = _point_colors(an, ordered)
        sc = ax.scatter(_raw(an, xkey, ordered), _raw(an, ykey, ordered), s=46,
                        c=colors, cmap=cmap, edgecolors="white", linewidths=0.6,
                        zorder=3)
        if cmap is not None:
            bar = ax.figure.colorbar(sc, ax=ax, pad=0.02)
            bar.set_label(an.color.label, fontsize=8)
            bar.ax.tick_params(labelsize=7)

    if an.knee.index is not None:
        ax.scatter(_raw(an, xkey, [an.knee.index]), _raw(an, ykey, [an.knee.index]),
                   s=190, marker="*", c=KNEE_COLOR, edgecolors="white",
                   linewidths=0.6, zorder=4)

    ax.set_xlabel(an.axes.x_label)
    ax.set_ylabel(an.axes.y_label)
    if an.axes.x_invert:
        ax.invert_xaxis()
    if an.axes.y_invert:
        ax.invert_yaxis()
    banded = _slo_bands(an, ax)
    _legend(an, ax, banded)


def _legend(an: Analysis, ax, banded: bool) -> None:
    handles = []
    if an.cloud_plotted:
        handles.append(Line2D([], [], marker="o", linestyle="", color=CLOUD_COLOR,
                              label=f"dominated ({len(an.cloud_plotted)})"))
    handles.append(Line2D([], [], marker="o", linestyle="", color=FRONT_COLOR,
                          label=f"on front ({len(an.ds.front_indices())})"))
    if an.knee.index is not None:
        handles.append(Line2D([], [], marker="*", linestyle="", color=KNEE_COLOR,
                              markersize=11, label="best balance"))
    if banded:
        handles.append(Line2D([], [], marker="s", linestyle="", color=BAND_COLOR,
                              alpha=0.35, label="SLO infeasible"))
    if an.color is not None and an.color.kind == "categorical":
        # Only the front is colored, so the legend lists only values that appear
        # there. A legend entry for a color drawn nowhere is a lie.
        drawn = set(color_values(an.ds, an.color, an.ds.front_indices()))
        shown = [v for v in an.color.values if v in drawn]
        for value, color in zip(shown, _categorical_colors(shown, an.color.values)):
            handles.append(Line2D([], [], marker="o", linestyle="", color=color,
                                  label=f"{an.color.label} {value}"))
    ax.legend(handles=handles, loc="best", frameon=False, fontsize=8)


def _card(an: Analysis, ax) -> None:
    """One objective: the winner in words, and where the rest of the search landed."""
    ds = an.ds
    obj = ds.objectives[0]
    best = an.picks.rows[0].index if an.picks.rows else ds.front_indices()[0]
    lines = [f"{describe_config(ds, best)}",
             f"{obj.key} = {fmt(ds.value(ds.points[best], obj))}"]
    gpus = gpu_count(ds, best)
    if gpus is not None:
        lines.append(f"{gpus} GPUs")
    ax.text(0.02, 0.95, "\n".join(lines), transform=ax.transAxes, va="top", ha="left",
            fontsize=13, color=FRONT_COLOR)

    if an.hero.with_distribution and an.cloud_plotted:
        ax.hist(_raw(an, obj.key, an.cloud_plotted), bins=12, color=CLOUD_COLOR)
        ax.axvline(ds.value(ds.points[best], obj), color=FRONT_COLOR, linewidth=1.5)
        ax.set_xlabel(axis_label(obj))
        ax.set_ylabel("evaluated configs")
    else:
        ax.set_xticks([])
        ax.set_yticks([])
        ax.grid(False)


def pareto(an: Analysis, path: str) -> None:
    """The hero figure: a card at one objective, a scatter otherwise (§5)."""
    with plt.rc_context(RCPARAMS):
        fig, ax = plt.subplots()
        if an.hero.form == HERO_CARD:
            _card(an, ax)
        else:
            _scatter(an, ax)
        ax.set_title(_title(an), loc="left", fontsize=11)
        fig.text(0.0, -0.06, _caption(an), fontsize=7.5, color="#4a5568", va="top")
        save(fig, path)


# ------------------------------------------------------------- knobs.png


def knobs(an: Analysis, path: str) -> None:
    """One strip per knob, ranked, with its headline printed beside it (§5).

    Fill comes from strip_cells(), the same numbers picks.md turns into glyphs,
    so the two figures cannot tell different stories about the same knob.
    """
    strips = an.strips
    with plt.rc_context(RCPARAMS):
        height = max(2.0, 0.34 * max(len(strips), 1) + 1.1)
        fig, ax = plt.subplots(figsize=(9.0, height))
        ax.set_xlim(0.0, 1.0)
        ax.set_ylim(-0.5, max(len(strips), 1) - 0.5)
        ax.invert_yaxis()
        ax.set_yticks(range(len(strips)))
        ax.set_yticklabels([s.name for s in strips], fontsize=8)
        ax.set_xticks([])
        ax.grid(False)
        for spine in ax.spines.values():
            spine.set_visible(False)

        for row, strip in enumerate(strips):
            cells = strip_cells(strip)
            width = 0.34 / max(len(cells), 1)
            for i, fill in enumerate(cells):
                ax.add_patch(plt.Rectangle(
                    (i * width, row - 0.28), width * 0.92, 0.56,
                    facecolor=FRONT_COLOR, alpha=0.15 + 0.85 * fill,
                    edgecolor="none"))
            ax.text(0.37, row, f"{strip.score:.2f}", va="center", fontsize=7.5,
                    color="#4a5568")
            ax.text(0.43, row, strip.headline, va="center", fontsize=8)

        ax.set_title("Which knobs the front exploits", loc="left", fontsize=11)
        caption = ("front share per value (categorical) or the front's span of the "
                   "searched range (numeric), ranked by separation from the cloud")
        if not an.ds.has_cloud:
            caption = ("no dominated cloud to compare against: these are the ranges "
                       "the front occupies, not attributions")
        fig.text(0.0, -0.02, caption, fontsize=7.5, color="#4a5568", va="top")
        save(fig, path)


# ---------------------------------------------------------- parallel.png


def parallel(an: Analysis, path: str) -> None:
    """Parallel coordinates over the objectives: the hero at 4+ (§5).

    Every axis is normalised so up is better, which makes a front member a line
    that cannot be beaten everywhere — the visual reading of non-domination.
    """
    ds = an.ds
    objectives = ds.objectives
    with plt.rc_context(RCPARAMS):
        fig, ax = plt.subplots()
        columns = [goodness(_raw(an, o.key, list(range(len(ds.points)))), o)
                   for o in objectives]
        xs = list(range(len(objectives)))

        for i in an.cloud_plotted:
            ax.plot(xs, [col[i] for col in columns], color=CLOUD_COLOR,
                    linewidth=0.6, alpha=0.6, zorder=1)

        front = ds.front_indices()
        colors, cmap = _point_colors(an, front)
        mapped = _line_colors(colors, cmap)
        for i, color in zip(front, mapped):
            ax.plot(xs, [col[i] for col in columns], color=color, linewidth=1.3,
                    alpha=0.9, zorder=2)
        if an.knee.index is not None:
            ax.plot(xs, [col[an.knee.index] for col in columns], color=KNEE_COLOR,
                    linewidth=2.2, zorder=3)

        # The per-axis range goes into the tick label rather than floating text,
        # which would collide with the title and the labels at 4+ axes.
        labels = []
        for obj in objectives:
            raw = _raw(an, obj.key, list(range(len(ds.points))))
            best = max(raw) if obj.direction == "maximize" else min(raw)
            worst = min(raw) if obj.direction == "maximize" else max(raw)
            labels.append(f"{axis_label(obj)}\n{fmt(worst)} → {fmt(best)}")
        ax.set_xticks(xs)
        ax.set_xticklabels(labels, fontsize=8)
        ax.set_ylim(-0.05, 1.05)
        ax.set_yticks([0.0, 1.0])
        ax.set_yticklabels(["worst", "best"], fontsize=8)
        ax.grid(axis="x", alpha=0.3)
        ax.grid(axis="y", visible=False)
        _parallel_legend(an, ax)

        ax.set_title(_title(an), loc="left", fontsize=11, pad=10)
        caption = ["every axis normalised so up is better; front in color, "
                   "dominated cloud in gray"]
        if an.knee.index is not None:
            caption.append("the thick gold line is the best-balance config")
        caption.extend(an.plot_notes)
        fig.text(0.0, -0.10, "\n".join(caption), fontsize=7.5, color="#4a5568",
                 va="top")
        save(fig, path)


def _parallel_legend(an: Analysis, ax) -> None:
    handles = []
    if an.cloud_plotted:
        handles.append(Line2D([], [], color=CLOUD_COLOR,
                              label=f"dominated ({len(an.cloud_plotted)})"))
    if an.color is not None and an.color.kind == "categorical":
        drawn = set(color_values(an.ds, an.color, an.ds.front_indices()))
        shown = [v for v in an.color.values if v in drawn]
        for value, color in zip(shown, _categorical_colors(shown, an.color.values)):
            handles.append(Line2D([], [], color=color,
                                  label=f"{an.color.label} {value}"))
    else:
        handles.append(Line2D([], [], color=FRONT_COLOR,
                              label=f"on front ({len(an.ds.front_indices())})"))
    if an.knee.index is not None:
        handles.append(Line2D([], [], color=KNEE_COLOR, linewidth=2.2,
                              label="best balance"))
    ax.legend(handles=handles, loc="lower left", frameon=False, fontsize=7.5,
              ncol=max(1, len(handles) // 3))


def _line_colors(colors: Any, cmap: Optional[str]) -> List[Any]:
    """Per-line colors from whatever _point_colors returned (a single color, a
    list of colors, or numeric values plus a colormap)."""
    if isinstance(colors, str):
        return [colors]
    if cmap is None:
        return list(colors)
    values = [float(v) for v in colors]
    lo, hi = min(values), max(values)
    span = (hi - lo) or 1.0
    mapper = plt.get_cmap(cmap)
    return [mapper((v - lo) / span) for v in values]
