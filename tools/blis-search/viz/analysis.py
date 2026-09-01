"""Every decision the renderers consume (design §5).

This module knows objectives and points. It knows nothing about pixels and
nothing about the JSON schema: that is data.py's job. Both renderers consume
one Analysis object, which is why the PNG and the HTML can never disagree.
"""
import math
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Tuple

from data import Dataset, Objective, Point


def dominates(a_vals: List[float], b_vals: List[float], objectives: List[Objective]) -> bool:
    """True when a is no worse than b on every objective and strictly better once."""
    strictly = False
    for val_a, val_b, obj in zip(a_vals, b_vals, objectives):
        if obj.direction == "maximize":
            if val_a < val_b:
                return False
            if val_a > val_b:
                strictly = True
        else:
            if val_a > val_b:
                return False
            if val_a < val_b:
                strictly = True
    return strictly


def recompute_front(ds: Dataset) -> List[int]:
    """Point indices of the non-dominated set, recomputed from objectives (§5)."""
    vals = [[ds.value(p, o) for o in ds.objectives] for p in ds.points]
    front = []
    for i, vi in enumerate(vals):
        if not any(dominates(vj, vi, ds.objectives) for j, vj in enumerate(vals) if j != i):
            front.append(i)
    # Sorted explicitly: the return contract promises ascending indices, and
    # relying on enumerate's order to supply that makes the guarantee invisible
    # to anyone refactoring this loop.
    return sorted(front)


@dataclass(frozen=True)
class FrontCheck:
    verifiable: bool                     # False in front-only mode
    agrees: bool
    reported_only: List[int]             # on the reported front, not on the recomputed one
    recomputed_only: List[int]           # non-dominated, but not reported
    message: Optional[str]               # banner text; None when silent agreement


def _bare_label(ds: Dataset, idx: int) -> str:
    cfg = ds.points[idx].config
    bits = []
    if cfg.get("tp") is not None:
        bits.append("tp%s" % cfg["tp"])
    # dp is part of the topology, not a policy knob: an instance occupies tp x dp
    # GPUs. Omitting it made two genuinely different topologies (tp4xdp4 and
    # tp4xdp1) share the label "tp4 x1", while the "x1" spent the label's remaining
    # channel on a replica count that was constant. Only emitted when the config
    # carries dp, so a results file from a space without a dp axis keeps its old
    # label exactly.
    if cfg.get("dp") is not None:
        bits.append("dp%s" % cfg["dp"])
    if cfg.get("replicas") is not None:
        bits.append("×%s" % cfg["replicas"])
    return " ".join(bits)


def describe_config(ds: Dataset, idx: int) -> str:
    """A short stable label for a point, for banners and picks rows.

    tp/replicas alone is not unique — a 60-evaluation search has many configs
    sharing a topology — so the point index is appended whenever another point
    would carry the same label. Two identical-looking rows in a picks table are
    worse than a slightly longer label.
    """
    bare = _bare_label(ds, idx)
    if not bare:
        return "#%d" % idx
    for other in range(len(ds.points)):
        if other != idx and _bare_label(ds, other) == bare:
            return "%s #%d" % (bare, idx)
    return bare


def check_front(ds: Dataset) -> FrontCheck:
    """Compare the reported front against a recomputation (§5).

    Never overwrites ds: disagreement is surfaced, not applied.
    """
    if not ds.has_cloud:
        return FrontCheck(
            verifiable=False, agrees=True, reported_only=[], recomputed_only=[],
            message=("Dominance cannot be verified: this file has no all_results, "
                     "so the reported front is taken on trust. "
                     "Re-run the search with --output-all to check it."))

    reported = set(ds.front_indices())
    recomputed = set(recompute_front(ds))
    reported_only = sorted(reported - recomputed)
    recomputed_only = sorted(recomputed - reported)
    if not reported_only and not recomputed_only:
        return FrontCheck(verifiable=True, agrees=True, reported_only=[],
                          recomputed_only=[], message=None)

    parts = []
    if reported_only:
        parts.append("reported on the front but dominated by another evaluated config: "
                     + ", ".join(describe_config(ds, i) for i in reported_only))
    if recomputed_only:
        parts.append("non-dominated but absent from the reported front: "
                     + ", ".join(describe_config(ds, i) for i in recomputed_only))
    return FrontCheck(
        verifiable=True, agrees=False, reported_only=reported_only,
        recomputed_only=recomputed_only,
        message=("The reported Pareto front disagrees with a recomputation from "
                 "all_results. " + "; ".join(parts)
                 + ". The chart draws the front as reported; this is a searcher bug, "
                   "not a plotting choice."))


def normalise(values: List[float]) -> List[float]:
    """Map to [0,1]. A zero-range input maps to the midpoint, not a divide by zero."""
    if not values:
        return []
    lo, hi = min(values), max(values)
    if hi == lo:
        return [0.5] * len(values)
    span = hi - lo
    return [(v - lo) / span for v in values]


def goodness(values: List[float], objective: Objective) -> List[float]:
    """Normalise so 1.0 is the best value for this objective's direction."""
    norm = normalise(values)
    if objective.direction == "minimize":
        return [1.0 - v for v in norm]
    return norm


def axis_label(objective: Objective) -> str:
    """'responses_per_sec ↑ better', or just the metric name for a non-objective."""
    if objective.direction not in ("minimize", "maximize"):
        return objective.key
    return "%s %s better" % (objective.key, objective.arrow)


@dataclass(frozen=True)
class AxisChoice:
    x: Objective
    y: Optional[Objective]        # None only in the single-objective case
    x_invert: bool
    y_invert: bool
    x_label: str
    y_label: str
    caption: str
    auto: bool


def _stdev(values: List[float]) -> float:
    if len(values) < 2:
        return 0.0
    mean = sum(values) / len(values)
    return math.sqrt(sum((v - mean) ** 2 for v in values) / len(values))


def _as_axis_objective(ds: Dataset, metric: str) -> Objective:
    """An objective keeps its direction; any other metric gets no direction."""
    for obj in ds.objectives:
        if obj.key == metric:
            return obj
    if metric not in ds.metric_names():
        raise ValueError(
            "unknown metric %r for an axis override; available metrics: %s"
            % (metric, ", ".join(ds.metric_names())))
    return Objective(key=metric, direction="")


def _distinct_positions(ds: Dataset, a: Objective, b: Objective) -> int:
    xs = normalise([float(p.metrics[a.key]) for p in ds.points])
    ys = normalise([float(p.metrics[b.key]) for p in ds.points])
    return len({(round(x, 9), round(y, 9)) for x, y in zip(xs, ys)})


def _spread_sum(ds: Dataset, a: Objective, b: Objective) -> float:
    xs = normalise([float(p.metrics[a.key]) for p in ds.points])
    ys = normalise([float(p.metrics[b.key]) for p in ds.points])
    return _stdev(xs) + _stdev(ys)


def _best_pair(ds: Dataset) -> Tuple[Objective, Objective, str]:
    """The pair maximising distinct 2D positions (§5 'Pair selection, pinned').

    Ties break by the sum of the two per-axis standard deviations, then by
    declaration order. Runs over all valid points, before decimation (D5).
    """
    objectives = ds.objectives
    best = None
    for i in range(len(objectives)):
        for j in range(i + 1, len(objectives)):
            a, b = objectives[i], objectives[j]
            key = (-_distinct_positions(ds, a, b), -_spread_sum(ds, a, b), i, j)
            if best is None or key < best[0]:
                best = (key, a, b)
    _, a, b = best
    caption = ("axes are the widest-spread pair of %d objectives: %s and %s"
               % (len(objectives), a.key, b.key))
    return a, b, caption


def choose_axes(
    ds: Dataset,
    x_override: Optional[str] = None,
    y_override: Optional[str] = None,
) -> AxisChoice:
    """Pick the hero's x and y and orient them so better is toward lower-right (§5)."""
    notes: List[str] = []
    auto = x_override is None and y_override is None

    if len(ds.objectives) == 1 and x_override is None and y_override is None:
        x = ds.objectives[0]
        return AxisChoice(
            x=x, y=None, x_invert=x.direction == "minimize", y_invert=False,
            x_label=axis_label(x), y_label="",
            caption="single objective: no tradeoff surface exists, "
                    "so the hero is the best-config card",
            auto=True)

    if x_override is not None or y_override is not None:
        if len(ds.objectives) >= 2:
            default_x, default_y, _ = (ds.objectives[0], ds.objectives[1], "") \
                if len(ds.objectives) == 2 else _best_pair(ds)
        else:
            default_x, default_y = ds.objectives[0], ds.objectives[0]
        x = _as_axis_objective(ds, x_override) if x_override else default_x
        y = _as_axis_objective(ds, y_override) if y_override else default_y
        notes.append("axes set by --x/--y")
        for axis in (x, y):
            if axis.direction not in ("minimize", "maximize"):
                notes.append("%s is not an objective, so no better direction is assumed"
                             % axis.key)
    elif len(ds.objectives) == 2:
        x, y = ds.objectives[0], ds.objectives[1]
        notes.append("axes are the 2 objectives")
    else:
        x, y, caption = _best_pair(ds)
        notes.append(caption)

    for axis in (x, y):
        vals = [float(p.metrics[axis.key]) for p in ds.points]
        if vals and max(vals) == min(vals):
            notes.append("%s has zero spread across all points" % axis.key)

    return AxisChoice(
        x=x, y=y,
        # Better must be to the right and toward the bottom, so a minimised x and a
        # maximised y are the ones that need their axis reversed.
        x_invert=x.direction == "minimize",
        y_invert=y.direction == "maximize",
        x_label=axis_label(x), y_label=axis_label(y),
        caption=" · ".join(notes),
        auto=auto)


def screen_fractions(
    ds: Dataset, axes: AxisChoice, indices: List[int],
) -> List[Tuple[float, float]]:
    """(fx, fy) in [0,1]^2 for each index: larger is further right and further down.

    Both mean 'better'. This is the single place 'better is toward the lower-right'
    is encoded; both renderers call it, so they cannot disagree.
    """
    xs_raw = [float(ds.points[i].metrics[axes.x.key]) for i in indices]
    if axes.y is None:
        fx = _fractions(ds, axes.x, xs_raw)
        return [(v, 0.5) for v in fx]
    ys_raw = [float(ds.points[i].metrics[axes.y.key]) for i in indices]
    return list(zip(_fractions(ds, axes.x, xs_raw), _fractions(ds, axes.y, ys_raw)))


def _fractions(ds: Dataset, axis: Objective, values: List[float]) -> List[float]:
    """Normalise against the FULL point set so a subset keeps the same scale."""
    full = [float(p.metrics[axis.key]) for p in ds.points]
    lo, hi = min(full), max(full)
    if hi == lo:
        base = [0.5] * len(values)
    else:
        base = [(v - lo) / (hi - lo) for v in values]
    if axis.direction == "minimize":
        return [1.0 - v for v in base]
    return base


# ------------------------------------------------------------------- knee


@dataclass
class Knee:
    """The front's best-balance point, or the reason there isn't one."""
    index: Optional[int]
    note: str               # "" when index is not None, else why it is suppressed
    extremes: Tuple[Optional[int], Optional[int]] = (None, None)


def _front_costs(
    ds: Dataset, front: List[int], axis: Objective,
) -> Optional[List[float]]:
    """Front members on one axis, normalised over the FRONT ONLY and oriented so
    lower is better (D12). None when the front is flat on this axis.

    Deliberately not screen_fractions(): that normalises against the full point
    set, and per-axis rescaling is not distance-preserving, so a single dominated
    outlier could otherwise move the knee.
    """
    raw = [float(ds.points[i].metrics[axis.key]) for i in front]
    lo, hi = min(raw), max(raw)
    if hi == lo:
        return None
    norm = [(v - lo) / (hi - lo) for v in raw]
    if axis.direction == "maximize":
        return [1.0 - v for v in norm]
    return norm


def knee(ds: Dataset, axes: AxisChoice) -> Knee:
    """Max perpendicular distance from the chord joining the two extremes (§5).

    Only non-extreme members are candidates, which is what makes "the knee is
    never an extreme" a law rather than an accident. Ties break to the lowest
    index, so a collinear front yields its lowest-index interior member.
    """
    if axes.y is None:
        return Knee(None, "only one objective — there is no tradeoff to balance")
    front = ds.front_indices()
    if len(front) < 3:
        return Knee(None, f"front has {len(front)} member(s) — a knee needs 3")

    cx = _front_costs(ds, front, axes.x)
    cy = _front_costs(ds, front, axes.y)
    if cx is None or cy is None:
        flat = axes.x.key if cx is None else axes.y.key
        return Knee(None, f"front is flat in {flat} — there is no tradeoff to balance")

    # Extremes: cheapest on x, cheapest on y. Ties break on the other axis, then
    # on the lowest point index (front is in ascending index order).
    lo_x = min(range(len(front)), key=lambda k: (cx[k], cy[k], front[k]))
    lo_y = min(range(len(front)), key=lambda k: (cy[k], cx[k], front[k]))
    if lo_x == lo_y:
        return Knee(None, "one front member is best on both axes — no tradeoff to balance",
                    (front[lo_x], front[lo_y]))

    ax, ay = cx[lo_x], cy[lo_x]
    bx, by = cx[lo_y], cy[lo_y]
    span = math.hypot(bx - ax, by - ay)

    best_k, best_d = None, -1.0
    for k in range(len(front)):
        if k in (lo_x, lo_y):
            continue
        cross = abs((bx - ax) * (cy[k] - ay) - (by - ay) * (cx[k] - ax))
        d = cross / span
        if d > best_d:                      # strict >: ties keep the lowest index
            best_k, best_d = k, d
    return Knee(front[best_k], "", (front[lo_x], front[lo_y]))


# --------------------------------------------------------- knob attribution

ABSENT = None


@dataclass
class KnobStrip:
    """One knob's front-vs-cloud separation, ready to draw."""
    name: str
    kind: str                                   # "numeric" | "categorical"
    score: float                                # [0,1]; 0.0 when not comparable
    headline: str                               # printed verbatim by every renderer
    narrows: bool                               # front covers less than the search did
    comparable: bool                            # False with front-only input
    values: List[Any] = field(default_factory=list)        # categorical, sorted
    front_fracs: List[float] = field(default_factory=list)
    cloud_fracs: List[float] = field(default_factory=list)
    full_range: Optional[Tuple[float, float]] = None       # numeric
    front_range: Optional[Tuple[float, float]] = None


def knob_value(point: Point, name: str) -> Any:
    """The knob's value for one point. A missing key reads as None (D13)."""
    return point.config.get(name, ABSENT)


def _value_sort_key(val: Any) -> Tuple[int, float, str]:
    """Deterministic ordering across None / bool / number / string."""
    if val is None:
        return (0, 0.0, "")
    if isinstance(val, bool):
        return (1, float(val), "")
    if isinstance(val, (int, float)):
        return (2, float(val), "")
    return (3, 0.0, str(val))


def knob_kind(ds: Dataset, name: str) -> str:
    """"numeric" iff every value is a non-bool number (D3); else "categorical"."""
    for p in ds.points:
        val = knob_value(p, name)
        if isinstance(val, bool) or not isinstance(val, (int, float)):
            return "categorical"
    return "numeric"


def _pct(frac: float) -> int:
    return int(round(100.0 * frac))


def _show(val: Any) -> str:
    if val is None:
        return "unset"
    if isinstance(val, bool):
        return "true" if val else "false"
    return str(val)


def _shares(values: List[Any], picked: List[Any]) -> List[float]:
    if not picked:
        return [0.0] * len(values)
    return [picked.count(v) / len(picked) for v in values]


def _categorical_strip(
    ds: Dataset, name: str, front: List[int], cloud: List[int],
) -> KnobStrip:
    # all_vals spans ALL points, not just the front: a value that appears only in
    # the cloud must still contribute to the total-variation distance.
    all_vals = [knob_value(p, name) for p in ds.points]
    values = sorted({_value_sort_key(v): v for v in all_vals}.values(),
                    key=_value_sort_key)
    f_vals = [knob_value(ds.points[i], name) for i in front]
    c_vals = [knob_value(ds.points[i], name) for i in cloud]
    f_fracs = _shares(values, f_vals)
    c_fracs = _shares(values, c_vals)

    # Dominant front value; ties go to the lowest position, so the headline is
    # stable under reordering of equally common values.
    top = max(range(len(values)), key=lambda k: (f_fracs[k], -k)) if values else 0
    narrows = len({_value_sort_key(v) for v in f_vals}) < len(values)

    if len(values) == 1:
        headline = f"constant {_show(values[0])} across the search"
        score = 0.0
        comparable = False
    elif not cloud:
        headline = (f"front is {_pct(f_fracs[top])}% {_show(values[top])} "
                    "(no cloud to compare)")
        score = 0.0
        comparable = False
    elif not f_vals:
        # An empty front is reachable: a results file may report pareto_front: []
        # (every evaluation failed). There is nothing to attribute, and a
        # total-variation distance against an all-zero front distribution would
        # score a meaningless 0.5 for every knob.
        headline = f"no front to compare ({len(values)} values searched)"
        score = 0.0
        comparable = False
    else:
        score = 0.5 * sum(abs(a - b) for a, b in zip(f_fracs, c_fracs))
        headline = (f"front is {_pct(f_fracs[top])}% {_show(values[top])} "
                    f"(cloud {_pct(c_fracs[top])}%)")
        comparable = True
    return KnobStrip(name=name, kind="categorical", score=score, headline=headline,
                     narrows=narrows, comparable=comparable, values=values,
                     front_fracs=f_fracs, cloud_fracs=c_fracs)


def _numeric_strip(
    ds: Dataset, name: str, front: List[int], cloud: List[int],
) -> KnobStrip:
    all_vals = [float(knob_value(p, name)) for p in ds.points]
    f_vals = [float(knob_value(ds.points[i], name)) for i in front]
    c_vals = [float(knob_value(ds.points[i], name)) for i in cloud]
    lo, hi = min(all_vals), max(all_vals)
    flo, fhi = (min(f_vals), max(f_vals)) if f_vals else (lo, hi)
    narrows = (flo, fhi) != (lo, hi)

    def num(val: float) -> str:
        return f"{val:g}"

    if hi == lo:
        headline = f"constant {num(lo)} across the search"
        score = 0.0
        comparable = False
    elif not cloud:
        headline = f"front spans {num(flo)}–{num(fhi)} (no cloud to compare)"
        score = 0.0
        comparable = False
    elif not f_vals:
        # An empty front is reachable: a results file may report pareto_front: []
        # (every evaluation failed). There is nothing to attribute, and a mean
        # over no values would divide by zero.
        headline = f"no front to compare against {num(lo)}–{num(hi)}"
        score = 0.0
        comparable = False
    else:
        f_mean = sum(f_vals) / len(f_vals)
        c_mean = sum(c_vals) / len(c_vals)
        score = abs(f_mean - c_mean) / (hi - lo)
        comparable = True
        if not narrows:
            headline = f"spans the full searched range {num(lo)}–{num(hi)}"
        elif flo == fhi:
            headline = f"front pins {num(flo)} of {num(lo)}–{num(hi)}"
        else:
            headline = f"narrows to {num(flo)}–{num(fhi)} of {num(lo)}–{num(hi)}"
    return KnobStrip(name=name, kind="numeric", score=score, headline=headline,
                     narrows=narrows, comparable=comparable,
                     full_range=(lo, hi), front_range=(flo, fhi))


def knob_strips(ds: Dataset) -> List[KnobStrip]:
    """One strip per knob, ranked by separation score then name (§5).

    With front-only input every score is 0.0 and the ranking degrades to
    alphabetical; each headline then says "no cloud to compare" so the page never
    implies an attribution it could not make.
    """
    front, cloud = ds.front_indices(), ds.cloud_indices()
    strips = []
    for name in ds.knob_names():
        if knob_kind(ds, name) == "numeric":
            strips.append(_numeric_strip(ds, name, front, cloud))
        else:
            strips.append(_categorical_strip(ds, name, front, cloud))
    return sorted(strips, key=lambda s: (-s.score, s.name))


def strip_cells(strip: KnobStrip, width: int = 4) -> List[float]:
    """Per-cell fill in [0,1] — the one decision behind every strip drawing.

    Categorical: one cell per value, filled with that value's front share.
    Numeric: `width` equal slices of the full searched range, each filled by how
    much of it the front's range covers.
    """
    if strip.kind == "categorical":
        return list(strip.front_fracs)
    lo, hi = strip.full_range
    flo, fhi = strip.front_range
    if hi == lo:
        return [1.0] * width
    if flo == fhi:
        # A pinned front has zero width and would otherwise light up no cell at
        # all. Fill the one slice that contains it.
        k = min(width - 1, int(width * (flo - lo) / (hi - lo)))
        return [1.0 if i == k else 0.0 for i in range(width)]
    cells = []
    for i in range(width):
        c0 = lo + (hi - lo) * i / width
        c1 = lo + (hi - lo) * (i + 1) / width
        overlap = max(0.0, min(c1, fhi) - max(c0, flo))
        cells.append(overlap / (c1 - c0))
    return cells

# ------------------------------------------------------------------ picks


def plural(count: int, word: str) -> str:
    return f"{count} {word}" if count == 1 else f"{count} {word}s"


def gpu_source(ds: Dataset) -> str:
    """Where a GPU count can be derived from: "gpus_used", "tp x dp x replicas",
    "tp x replicas", or ""."""
    def numeric_everywhere(name: str) -> bool:
        return all(isinstance(knob_value(p, name), (int, float))
                   and not isinstance(knob_value(p, name), bool) for p in ds.points)

    if ds.points and numeric_everywhere("gpus_used"):
        return "gpus_used"
    if ds.points and numeric_everywhere("tp") and numeric_everywhere("replicas"):
        # An instance occupies tp x dp GPUs (its "world"): tensor parallelism
        # shards a rank, data parallelism adds ranks. Omitting dp under-reports a
        # wide-EP topology's cost (tp8 dp2 is 16 GPUs, not 8) and merges distinct
        # GPU tiers into one. dp is an OPT-IN axis, so a space without it keeps the
        # old source string and the old value exactly.
        if numeric_everywhere("dp"):
            return "tp x dp x replicas"
        return "tp x replicas"
    return ""


def gpu_count(ds: Dataset, idx: int) -> Optional[int]:
    """GPU count for one point, or None when no GPU dimension is derivable (§9)."""
    src = gpu_source(ds)
    cfg = ds.points[idx].config
    if src == "gpus_used":
        return int(cfg["gpus_used"])
    if src == "tp x dp x replicas":
        return int(cfg["tp"]) * int(cfg["dp"]) * int(cfg["replicas"])
    if src == "tp x replicas":
        return int(cfg["tp"]) * int(cfg["replicas"])
    return None


@dataclass
class Pick:
    """One row of the picks table: a point and every category it won."""
    index: int
    titles: List[str]


@dataclass
class Picks:
    rows: List[Pick]
    pool: List[int]                 # the indices picks were drawn from
    header: str                     # "from 17 SLO-feasible configs"
    banner: str = ""                # non-empty when no config met the SLO (§9)
    notes: List[str] = field(default_factory=list)   # rows deliberately omitted


def _pool(ds: Dataset) -> Tuple[List[int], str, str]:
    front = ds.front_indices()
    if not ds.slo_constraints:
        return front, "from " + plural(len(front), "front config"), ""
    feasible = [i for i in front if ds.points[i].slo_ok]
    if feasible:
        return feasible, "from " + plural(len(feasible), "SLO-feasible config"), ""
    return (front, "from " + plural(len(front), "front config") + " — none met the SLO",
            "No evaluated config met the SLO constraints. These picks are the best "
            "available and they violate it.")


def picks(ds: Dataset, axes: AxisChoice) -> Picks:
    """Best per objective, the knee, and lowest GPU count (§5).

    Drawn from the SLO-feasible set when the file carries constraints, otherwise
    from the front; the header states which pool. A config that wins several
    categories appears once, listing all its titles. Anything not shown is listed
    in `notes` — an omission a reader cannot see is a silent one.
    """
    pool, header, banner = _pool(ds)
    titles: Dict[int, List[str]] = {}
    order: List[int] = []
    notes: List[str] = []

    def claim(idx: int, title: str) -> None:
        if idx not in titles:
            titles[idx] = []
            order.append(idx)
        titles[idx].append(title)

    if not pool:
        return Picks([], [], "no configs to pick from", banner, notes)

    for obj in ds.objectives:
        verb = "Max" if obj.direction == "maximize" else "Min"
        sign = -1.0 if obj.direction == "maximize" else 1.0
        best = min(pool, key=lambda i: (sign * ds.value(ds.points[i], obj), i))
        claim(best, f"{verb} {obj.key}")

    k = knee(ds, axes)
    if k.index is not None:
        if k.index in pool:
            claim(k.index, "★ Best balance")
        else:
            notes.append("best-balance row omitted: the knee is not in the "
                         + ("SLO-feasible set" if ds.slo_constraints else "pool"))
    elif k.note:
        notes.append("best-balance row omitted: " + k.note)

    if not gpu_source(ds):
        notes.append("fewest-GPUs row omitted: no GPU dimension is derivable "
                     "(no gpus_used, and not both tp and replicas)")
    else:
        counts = {gpu_count(ds, i) for i in pool}
        if len(counts) == 1:
            # Every candidate costs the same; the row would be an arbitrary
            # tie-break dressed up as a recommendation (§5's wasted-channel rule).
            notes.append("fewest-GPUs row omitted: every config in the pool uses "
                         f"{counts.pop()} GPUs")
        else:
            claim(min(pool, key=lambda i: (gpu_count(ds, i), i)), "Fewest GPUs")

    return Picks([Pick(i, titles[i]) for i in order], pool, header, banner, notes)


# --------------------------------------------------- shared page furniture


def fmt(value: float) -> str:
    """The one number format every output uses.

    It lives here rather than in a renderer because picks.md, the figures, and
    the page must print the same digits for the same value: enough to compare two
    configs, few enough to scan a column.
    """
    if value == int(value) and abs(value) < 1e15:
        return str(int(value))
    if abs(value) >= 100:
        return f"{value:.0f}"
    if abs(value) >= 10:
        return f"{value:.1f}"
    return f"{value:.3g}"


def context_line(ds: Dataset) -> str:
    """What this search was, from `spec`. Falls back to the file name."""
    ctx = ds.context
    bits = [str(ctx[k]) for k in ("model", "hardware", "workload") if k in ctx]
    if "rate" in ctx:
        bits.append(f"{ctx['rate']} req/s")
    if "num-requests" in ctx:
        bits.append(f"{ctx['num-requests']} req")
    return " · ".join(bits) or ds.source_name


def summary_bits(ds: Dataset) -> List[str]:
    """How big the search was, and what it was constrained by."""
    bits = [f"{ds.stats.get('total_evaluated', len(ds.points))} evaluated",
            plural(len(ds.front_indices()), "config") + " on front"]
    if not ds.has_cloud:
        bits.append("front-only file (no all_results)")
    if ds.slo_constraints:
        bits.append("SLO: " + ", ".join(
            f"{c['metric']} {c['op']} {fmt(float(c['threshold']))}"
            for c in ds.slo_constraints))
    return bits


def provenance_bits(an: "Analysis") -> List[str]:
    """The input's sha256 and the searcher's own numbers — never a generation
    timestamp, which would make byte-identity untestable (§10)."""
    ds = an.ds
    bits = [f"generated from {ds.source_name} (sha {ds.source_sha256[:8]})"]
    for key in ("strategy", "converged_at_eval", "hypervolume", "wall_time_seconds"):
        if key not in ds.stats:
            continue
        value = ds.stats[key]
        if key == "hypervolume":
            bits.append(f"HV {float(value):.3f}")
        elif key == "wall_time_seconds":
            bits.append(f"{fmt(float(value))}s")
        elif key == "converged_at_eval":
            bits.append(f"converged at eval {value}")
        else:
            bits.append(str(value))
    return bits + list(an.plot_notes)


def page_banners(an: "Analysis", flag_source) -> List[str]:
    """Every statement an output must make: the data's own (data_banners) plus
    the CLI's flag availability. Both renderers call this, so picks.md and
    index.html cannot carry different sentences.
    """
    out = list(an.banners)
    if getattr(flag_source, "note", ""):
        out.append(f"No deploy commands: {flag_source.note}.")
    if getattr(flag_source, "warning", ""):
        out.append(flag_source.warning)
    return out


# ------------------------------------------------- hero form, color, cloud

HERO_CARD = "card"
HERO_SCATTER = "scatter"
HERO_PARALLEL = "parallel"


@dataclass
class Hero:
    """Which figure leads the page, by objective count (§5)."""
    form: str                       # "card" | "scatter" | "parallel"
    caption: str
    with_scatter: bool = False      # parallel hero also draws the best-pair scatter
    with_distribution: bool = False  # single-objective card + cloud distribution


def third_objective(ds: Dataset, axes: AxisChoice) -> Optional[Objective]:
    """The objective that is on neither axis — the one the color channel gets."""
    on_axes = {axes.x.key, axes.y.key if axes.y else None}
    rest = [o for o in ds.objectives if o.key not in on_axes]
    return rest[0] if rest else None


def choose_hero(ds: Dataset, axes: AxisChoice) -> Hero:
    n = len(ds.objectives)
    if n == 1:
        obj = ds.objectives[0]
        return Hero(HERO_CARD,
                    f"One objective ({axis_label(obj)}), so there is no tradeoff: "
                    "the best config, and where the rest of the search landed.",
                    with_distribution=ds.has_cloud)
    if n == 2:
        return Hero(HERO_SCATTER, "The two objectives, front drawn as a staircase "
                                  "over the dominated cloud.")
    if n == 3:
        third = third_objective(ds, axes)
        extra = third.key if third else ds.objectives[2].key
        return Hero(HERO_SCATTER,
                    f"Three objectives: the two widest-spread on the axes, {extra} "
                    "on the color channel.")
    return Hero(HERO_PARALLEL,
                f"{len(ds.objectives)} objectives: parallel coordinates lead, with the "
                "widest-spread pair as a scatter beneath.",
                with_scatter=True)


@dataclass
class ColorChoice:
    """The color channel. `key` is a knob, a metric, or the synthetic "gpus"."""
    key: Optional[str]              # None: front/cloud coloring only
    kind: str                       # "numeric" | "categorical" | "none"
    label: str
    caption: str
    values: List[Any] = field(default_factory=list)   # legend entries, categorical


def metric_is_numeric(ds: Dataset, name: str) -> bool:
    """True when every point carries this metric as a number. Metrics are numeric
    by construction, but a non-objective metric may be missing on some points
    (D2), and a column with holes cannot drive a continuous ramp."""
    return all(isinstance(p.metrics.get(name), (int, float))
               and not isinstance(p.metrics.get(name), bool) for p in ds.points)


def gpu_counts(ds: Dataset, indices: List[int]) -> List[Optional[int]]:
    return [gpu_count(ds, i) for i in indices]


def color_values(ds: Dataset, choice: ColorChoice, indices: List[int]) -> List[Any]:
    """Per-point color values, so a renderer never re-derives the channel."""
    if choice.key is None:
        return [None] * len(indices)
    if choice.key == "gpus":
        return gpu_counts(ds, indices)
    if choice.key in ds.metric_names():
        return [ds.points[i].metrics.get(choice.key) for i in indices]
    return [knob_value(ds.points[i], choice.key) for i in indices]


def choose_color(
    ds: Dataset, axes: AxisChoice, plotted: List[int],
    override: Optional[str] = None,
) -> ColorChoice:
    """GPU count for predictability, the third objective at exactly 3, or nothing (§5).

    A channel that carries one distinct value is not a channel: the caption says so
    and the points fall back to front/cloud coloring rather than showing a
    one-entry legend.
    """
    if override:
        if override in ds.metric_names():
            kind = "numeric" if metric_is_numeric(ds, override) else "categorical"
        elif override in ds.knob_names():
            kind = knob_kind(ds, override)
        else:
            raise ValueError(
                "unknown field %r for --color; available metrics: %s; available knobs: %s"
                % (override, ", ".join(ds.metric_names()), ", ".join(ds.knob_names())))
        values = []
        if kind == "categorical":
            values = sorted({v for v in color_values(
                ds, ColorChoice(override, kind, override, ""), plotted)},
                key=_value_sort_key)
        return ColorChoice(override, kind, override,
                           f"colored by {override} (--color)", values)

    if len(ds.objectives) == 3:
        # The objective on neither axis, which is not necessarily objectives[2]:
        # with 3 objectives the axes are the widest-spread pair (§5).
        third = third_objective(ds, axes)
        if third is not None:
            return ColorChoice(third.key, "numeric", axis_label(third),
                               f"colored by {third.key}, the third objective")

    src = gpu_source(ds)
    if not src:
        return ColorChoice(None, "none", "",
                           "no GPU count is derivable from this file, so points are "
                           "colored by front/cloud only")
    counts = sorted({c for c in gpu_counts(ds, plotted) if c is not None})
    if len(counts) <= 1:
        only = counts[0] if counts else "?"
        return ColorChoice(None, "none", "",
                           f"all {plural(len(plotted), 'plotted config')} use {only} "
                           "GPUs, so color carries nothing here")
    return ColorChoice("gpus", "categorical", "GPU count",
                       f"colored by GPU count (from {src})", counts)


def decimate(
    ds: Dataset, axes: AxisChoice, max_points: Optional[int],
) -> Tuple[List[int], str]:
    """Grid-decimate the dominated cloud; never the front (§4.3, pinned).

    Bin the normalised hero plane into k x k with k = ceil(sqrt(N)), keep the
    lowest-index point per occupied bin, and if that still exceeds N keep the
    first N in index order. Preserves the cloud's visual envelope instead of
    thinning it uniformly, and is a pure function of the input.
    """
    cloud = ds.cloud_indices()
    if not max_points or len(cloud) <= max_points:
        return cloud, ""
    k = math.ceil(math.sqrt(max_points))
    fracs = screen_fractions(ds, axes, cloud)
    kept: Dict[Tuple[int, int], int] = {}
    for idx, (fx, fy) in zip(cloud, fracs):
        cell = (min(k - 1, int(fx * k)), min(k - 1, int(fy * k)))
        if cell not in kept:                      # lowest index wins
            kept[cell] = idx
    chosen = sorted(kept.values())[:max_points]
    note = (f"dominated cloud decimated from {len(cloud)} to {len(chosen)} points "
            f"(--max-points {max_points}, {k}x{k} grid); the front is never sampled")
    return chosen, note


# ---------------------------------------------------------- the one bundle


def data_banners(ds: Dataset, check: FrontCheck, chosen: Picks) -> List[str]:
    """Every statement the data itself forces the page to make (§9).

    Computed here, not in a renderer, so picks.md and index.html carry exactly the
    same sentences. Flag-availability notes are the CLI's business, not the data's,
    and are appended by the renderers.
    """
    out = []
    if chosen.banner:
        out.append(chosen.banner)
    if check.message:
        out.append(check.message)
    if len(ds.front_indices()) == 1:
        out.append("The search found a single non-dominated config, so there is no "
                   "tradeoff curve to show.")
    if ds.excluded:
        out.append(plural(len(ds.excluded), "evaluation") + " excluded: an objective "
                   "metric was missing, NaN, or infinite.")
    if ds.slo_mismatch:
        out.append(plural(len(ds.slo_mismatch), "front config")
                   + " disagree with the file's own slo_feasible list; this page "
                     "recomputes SLO feasibility from slo_constraints.")
    return out


@dataclass
class Analysis:
    """Every decision, computed once. Renderers read; they never decide."""
    ds: Dataset
    axes: AxisChoice
    check: FrontCheck
    knee: Knee
    picks: Picks
    strips: List[KnobStrip]
    banners: List[str] = field(default_factory=list)
    hero: Optional[Hero] = None
    color: Optional[ColorChoice] = None
    cloud_plotted: List[int] = field(default_factory=list)   # after decimation
    plotted: List[int] = field(default_factory=list)         # front + cloud_plotted
    plot_notes: List[str] = field(default_factory=list)      # provenance strip


def analyse(
    ds: Dataset,
    x_override: Optional[str] = None,
    y_override: Optional[str] = None,
    color_override: Optional[str] = None,
    max_points: Optional[int] = None,
) -> Analysis:
    """Run every §5 rule over one dataset. The renderers add nothing to this."""
    axes = choose_axes(ds, x_override, y_override)
    check = check_front(ds)
    chosen = picks(ds, axes)
    cloud, note = decimate(ds, axes, max_points)
    plotted = sorted(ds.front_indices() + cloud)
    return Analysis(ds=ds, axes=axes, check=check, knee=knee(ds, axes), picks=chosen,
                    strips=knob_strips(ds),
                    banners=data_banners(ds, check, chosen),
                    hero=choose_hero(ds, axes),
                    color=choose_color(ds, axes, plotted, color_override),
                    cloud_plotted=cloud, plotted=plotted,
                    plot_notes=[note] if note else [])
