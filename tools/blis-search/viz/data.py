"""Load, validate, and normalise blis-search results into one internal shape.

This module owns the JSON schema (design §2) and knows nothing about charts.
"""
import hashlib
import json
import math
import os
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional

#: Top-level keys every blis-search results file carries (design §2).
REQUIRED_KEYS = ("pareto_front", "objectives", "spec", "search_stats")

#: Optional top-level keys, present depending on the search's flags.
OPTIONAL_KEYS = ("all_results", "slo_feasible", "slo_constraints",
                 "param_flags", "space_file")

DIRECTIONS = ("minimize", "maximize")


class SchemaError(ValueError):
    """The input file is not a blis-search results file."""


def validate_schema(raw: Any) -> None:
    """Raise SchemaError unless raw is a blis-search results object.

    The error names expected vs found top-level keys so a user who pointed the
    tool at search/pareto_search.py output can see why (design §2, §9).
    """
    if not isinstance(raw, dict):
        raise SchemaError(
            "expected a JSON object at the top level, found %s" % type(raw).__name__)

    missing = [k for k in REQUIRED_KEYS if k not in raw]
    if missing:
        raise SchemaError(
            "not a blis-search results file: missing top-level key(s) %s\n"
            "  expected: %s\n"
            "  found:    %s\n"
            "  (search/pareto_search.py emits an unrelated schema and is not supported)"
            % (", ".join(missing), ", ".join(REQUIRED_KEYS), ", ".join(sorted(raw))))

    objectives = raw["objectives"]
    if not isinstance(objectives, list) or not objectives:
        raise SchemaError("objectives must be a non-empty list, found %r" % (objectives,))
    for i, obj in enumerate(objectives):
        if not isinstance(obj, dict) or "metric" not in obj or "direction" not in obj:
            raise SchemaError(
                "objectives[%d] must be {metric, direction}, found %r" % (i, obj))
        if obj["direction"] not in DIRECTIONS:
            raise SchemaError(
                "objectives[%d].direction is %r; expected one of %s"
                % (i, obj["direction"], ", ".join(DIRECTIONS)))

    for key in ("pareto_front", "all_results", "slo_feasible"):
        entries = raw.get(key)
        if entries is None:
            continue
        if not isinstance(entries, list):
            raise SchemaError("%s must be a list, found %s" % (key, type(entries).__name__))
        for i, entry in enumerate(entries):
            if not isinstance(entry, dict) or "config" not in entry or "metrics" not in entry:
                raise SchemaError(
                    "%s[%d] must be {config, metrics}, found keys %s"
                    % (key, i, sorted(entry) if isinstance(entry, dict) else type(entry).__name__))

    for key in ("spec", "search_stats"):
        if not isinstance(raw[key], dict):
            raise SchemaError("%s must be an object, found %s" % (key, type(raw[key]).__name__))

    constraints = raw.get("slo_constraints")
    if constraints is not None:
        if not isinstance(constraints, list):
            raise SchemaError("slo_constraints must be a list")
        for i, c in enumerate(constraints):
            if not isinstance(c, dict) or set(c) < {"metric", "op", "threshold"}:
                raise SchemaError(
                    "slo_constraints[%d] must be {metric, op, threshold}, found %r" % (i, c))
            if c["op"] not in ("<", ">"):
                raise SchemaError(
                    "slo_constraints[%d].op is %r; expected < or >" % (i, c["op"]))


@dataclass(frozen=True)
class Objective:
    key: str
    direction: str          # "minimize" | "maximize"

    @property
    def arrow(self) -> str:
        """Glyph for axis labels. Empty for an axis that is not an objective:
        --x / --y may name any metric, whose "better" direction is unknown."""
        if self.direction == "maximize":
            return "↑"
        if self.direction == "minimize":
            return "↓"
        return ""


@dataclass
class Point:
    idx: int                # dense position in Dataset.points, source order
    config: Dict[str, Any]
    metrics: Dict[str, Any]
    on_front: bool
    slo_ok: Optional[bool]  # None when the file carries no slo_constraints


@dataclass
class Dataset:
    points: List[Point]
    objectives: List[Objective]
    context: Dict[str, Any]                        # spec, verbatim
    stats: Dict[str, Any]                          # search_stats, verbatim
    slo_constraints: List[Dict[str, Any]] = field(default_factory=list)
    param_flags: Optional[Dict[str, str]] = None
    space_file: Optional[str] = None
    has_cloud: bool = False
    excluded: List[Dict[str, Any]] = field(default_factory=list)
    slo_mismatch: List[int] = field(default_factory=list)
    source_sha256: str = ""
    source_name: str = ""

    def front_indices(self) -> List[int]:
        return [p.idx for p in self.points if p.on_front]

    def cloud_indices(self) -> List[int]:
        return [p.idx for p in self.points if not p.on_front]

    def value(self, point: Point, objective: Objective) -> float:
        """The raw metric value. No orientation, no normalisation."""
        return float(point.metrics[objective.key])

    def knob_names(self) -> List[str]:
        names = set()
        for p in self.points:
            names.update(p.config)
        return sorted(names)

    def metric_names(self) -> List[str]:
        names = set()
        for p in self.points:
            names.update(p.metrics)
        return sorted(names)


def entry_key(entry: Dict[str, Any]) -> str:
    """Canonical identity of a {config, metrics} entry, for front matching (D7)."""
    return json.dumps({"config": entry.get("config"), "metrics": entry.get("metrics")},
                      sort_keys=True, allow_nan=True)


def satisfies_slo(metrics: Dict[str, Any], constraints: List[Dict[str, Any]]) -> bool:
    """Mirror of search.py's filter_by_slo, evaluated per point (D4)."""
    for c in constraints:
        val = metrics.get(c["metric"])
        if val is None or not isinstance(val, (int, float)) or isinstance(val, bool):
            return False
        if not math.isfinite(float(val)):
            return False
        if c["op"] == "<" and float(val) >= c["threshold"]:
            return False
        if c["op"] == ">" and float(val) <= c["threshold"]:
            return False
    return True


def _exclusion_reason(metrics: Dict[str, Any], objectives: List[Objective]) -> Optional[str]:
    """Why this point cannot be plotted, or None when it can (D2, §9)."""
    for obj in objectives:
        if obj.key not in metrics:
            return "missing"
        val = metrics[obj.key]
        if isinstance(val, bool) or not isinstance(val, (int, float)):
            return "missing"
        f = float(val)
        if math.isnan(f):
            return "nan"
        if math.isinf(f):
            return "inf"
    return None


def file_sha256(path: str) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def build_dataset(
    raw: Dict[str, Any],
    *,
    source_sha256: str,
    source_name: str,
    param_flags: Optional[Dict[str, str]] = None,
    space_file: Optional[str] = None,
) -> Dataset:
    """Normalise a validated results object into the one internal shape (§4)."""
    validate_schema(raw)

    objectives = [Objective(key=o["metric"], direction=o["direction"])
                  for o in raw["objectives"]]
    constraints = list(raw.get("slo_constraints") or [])

    has_cloud = "all_results" in raw and bool(raw["all_results"])
    source_entries = raw["all_results"] if has_cloud else raw["pareto_front"]

    # Front membership: one all_results row per pareto_front entry, lowest index
    # first, each row claimed at most once (D7).
    front_wanted: Dict[str, int] = {}
    for entry in raw["pareto_front"]:
        front_wanted[entry_key(entry)] = front_wanted.get(entry_key(entry), 0) + 1

    reported_feasible: Dict[str, int] = {}
    for entry in raw.get("slo_feasible") or []:
        k = entry_key(entry)
        reported_feasible[k] = reported_feasible.get(k, 0) + 1

    points: List[Point] = []
    excluded: List[Dict[str, Any]] = []
    slo_mismatch: List[int] = []

    for source_idx, entry in enumerate(source_entries):
        metrics = entry["metrics"]
        reason = _exclusion_reason(metrics, objectives)
        if reason is not None:
            excluded.append({"source_index": source_idx, "reason": reason,
                             "config": entry["config"]})
            continue

        key = entry_key(entry)
        on_front = True
        if has_cloud:
            on_front = front_wanted.get(key, 0) > 0
            if on_front:
                front_wanted[key] -= 1

        slo_ok = satisfies_slo(metrics, constraints) if constraints else None
        idx = len(points)
        points.append(Point(idx=idx, config=dict(entry["config"]), metrics=dict(metrics),
                            on_front=on_front, slo_ok=slo_ok))

        # Cross-check the computed verdict against the searcher's own list, for
        # front members only — slo_feasible never covers the cloud (D4).
        if constraints and on_front:
            claimed = reported_feasible.get(key, 0) > 0
            if claimed:
                reported_feasible[key] -= 1
            if claimed != bool(slo_ok):
                slo_mismatch.append(idx)

    return Dataset(
        points=points,
        objectives=objectives,
        context=dict(raw["spec"]),
        stats=dict(raw["search_stats"]),
        slo_constraints=constraints,
        param_flags=param_flags if param_flags is not None else raw.get("param_flags"),
        space_file=space_file if space_file is not None else raw.get("space_file"),
        has_cloud=has_cloud,
        excluded=excluded,
        slo_mismatch=slo_mismatch,
        source_sha256=source_sha256,
        source_name=source_name,
    )


def load(
    path: str,
    *,
    param_flags: Optional[Dict[str, str]] = None,
    space_file: Optional[str] = None,
) -> Dataset:
    """Read, validate, and normalise a results file."""
    with open(path) as f:
        raw = json.load(f)
    return build_dataset(raw, source_sha256=file_sha256(path),
                         source_name=os.path.basename(path),
                         param_flags=param_flags, space_file=space_file)


FIELD_KNOB = "c:"
FIELD_METRIC = "m:"


def _is_number(val: Any) -> bool:
    return isinstance(val, (int, float)) and not isinstance(val, bool)


def encode_columnar(ds: Dataset, indices: Optional[List[int]] = None) -> Dict[str, Any]:
    """Columnar encoding with categorical strings interned (§4.3).

    Fields are namespaced c:<knob> / m:<metric>: gpus_used appears in both a
    config and a metrics block in real files, so unprefixed names would collide.
    A field is numeric only when every value across the encoded rows is a number
    (bools excluded); otherwise its values are interned into s[field] and the
    row payload carries the index. null is an interned value, not missing data.
    """
    idxs = list(range(len(ds.points))) if indices is None else list(indices)
    rows_src = [ds.points[i] for i in idxs]

    fields = sorted([FIELD_KNOB + k for k in ds.knob_names()]
                    + [FIELD_METRIC + m for m in ds.metric_names()])

    def cell(point: Point, fieldname: str) -> Any:
        if fieldname.startswith(FIELD_KNOB):
            return point.config.get(fieldname[len(FIELD_KNOB):])
        return point.metrics.get(fieldname[len(FIELD_METRIC):])

    columns = {f: [cell(p, f) for p in rows_src] for f in fields}

    interned: Dict[str, List[Any]] = {}
    for f in fields:
        col = columns[f]
        if all(_is_number(v) for v in col):
            continue
        # Deterministic order: numbers and strings sorted within kind, None last.
        distinct = {json.dumps(v, sort_keys=True): v for v in col}
        vals = [distinct[k] for k in sorted(distinct)]
        interned[f] = vals

    rows: List[List[Any]] = []
    for p in rows_src:
        row: List[Any] = []
        for f in fields:
            val = cell(p, f)
            if f in interned:
                row.append(interned[f].index(val))
            else:
                row.append(val)
        rows.append(row)

    has_slo = bool(ds.slo_constraints)
    return {
        "f": fields,
        "s": {f: interned[f] for f in sorted(interned)},
        "r": rows,
        "front": [i for i, p in enumerate(rows_src) if p.on_front],
        "slo": [i for i, p in enumerate(rows_src) if p.slo_ok] if has_slo else None,
        "n": len(rows),
    }


def decode_columnar(enc: Dict[str, Any]) -> List[Dict[str, Any]]:
    """Inverse of encode_columnar: one flat {field: value} dict per row."""
    fields = enc["f"]
    interned = enc.get("s") or {}
    out = []
    for row in enc["r"]:
        rec = {}
        for f, cell in zip(fields, row):
            rec[f] = interned[f][cell] if f in interned else cell
        out.append(rec)
    return out
