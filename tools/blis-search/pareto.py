"""Pareto front extraction and hypervolume computation."""
from typing import List, Dict, Optional, Tuple


def is_dominated(candidate: dict, other: dict, objectives: List[Dict]) -> bool:
    """Check if candidate is dominated by other given objective directions."""
    dominated = True
    strictly_better = False
    for obj in objectives:
        metric = obj["metric"]
        direction = obj["direction"]
        c_val = candidate["metrics"][metric]
        o_val = other["metrics"][metric]
        if direction == "maximize":
            if o_val < c_val:
                dominated = False
                break
            if o_val > c_val:
                strictly_better = True
        else:
            if o_val > c_val:
                dominated = False
                break
            if o_val < c_val:
                strictly_better = True
    return dominated and strictly_better


def extract_pareto_front(
    results: List[dict], objectives: List[Dict]
) -> List[dict]:
    """Extract non-dominated solutions from results.

    Each result is {"config": {...}, "metrics": {...}}.
    """
    pareto = []
    for r in results:
        if any(is_dominated(r, other, objectives) for other in results if other is not r):
            continue
        pareto.append(r)
    return pareto


def compute_hypervolume_2d(
    pareto_front: List[dict],
    objectives: List[Dict],
    reference_point: Dict[str, float],
) -> float:
    """2D hypervolume via sweep-line. Objectives must be exactly 2."""
    if not pareto_front or len(objectives) != 2:
        return 0.0

    obj0 = objectives[0]
    obj1 = objectives[1]
    m0, m1 = obj0["metric"], obj1["metric"]
    ref0, ref1 = reference_point[m0], reference_point[m1]

    points = []
    for r in pareto_front:
        v0 = r["metrics"][m0]
        v1 = r["metrics"][m1]
        if obj0["direction"] == "maximize":
            n0 = v0 / ref0 if ref0 != 0 else 0.0
        else:
            n0 = max(0.0, (ref0 - v0) / ref0) if ref0 != 0 else 0.0
        if obj1["direction"] == "maximize":
            n1 = v1 / ref1 if ref1 != 0 else 0.0
        else:
            n1 = max(0.0, (ref1 - v1) / ref1) if ref1 != 0 else 0.0
        points.append((n0, n1))

    points.sort(key=lambda p: -p[0])
    hv = 0.0
    max_n1 = 0.0
    for i, (n0, n1) in enumerate(points):
        if n1 > max_n1:
            max_n1 = n1
        next_n0 = points[i + 1][0] if i < len(points) - 1 else 0.0
        hv += (n0 - next_n0) * max_n1
    return hv
