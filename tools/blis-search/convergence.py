"""HV convergence detection and diversity-biased Phase 2 targeting."""
from typing import List, Optional, Set


def detect_convergence(
    hv_values: List[float], K: int, epsilon: float
) -> Optional[int]:
    """Detect first eval index where HV has not improved by >= epsilon over K evals.

    Returns 1-based eval count or None if no convergence detected.
    """
    n = len(hv_values)
    for i in range(K, n):
        if hv_values[i] - hv_values[i - K] < epsilon:
            return i + 1
    return None


def get_pareto_tp_classes(pareto_front: List[dict]) -> Set[int]:
    """Return set of TP values represented in Pareto front (legacy mode)."""
    tp_values = set()
    for r in pareto_front:
        cfg = r.get("config", {})
        if "tp" in cfg:
            tp_values.add(cfg["tp"])
    return tp_values


def get_pareto_gpu_classes(pareto_front: List[dict]) -> Set[int]:
    """Return set of GPU-count tiers represented in Pareto front."""
    gpu_counts = set()
    for r in pareto_front:
        cfg = r.get("config", {})
        gpus = cfg.get("gpus_used")
        if gpus is not None:
            gpu_counts.add(gpus)
        elif "tp" in cfg and "replicas" in cfg:
            gpu_counts.add(cfg["tp"] * cfg["replicas"])
    return gpu_counts
