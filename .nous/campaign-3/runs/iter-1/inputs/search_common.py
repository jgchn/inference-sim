"""
Common parameter space definition, constraint validation, and evaluation logic
for BLIS multi-objective configuration search.
"""

import json
import os
import random
import subprocess
import tempfile
import time
from pathlib import Path

# Worktree directory where blis binary lives and where we can write temp files
# inputs/ is at: .nous/campaign-3/runs/iter-1/inputs/ — 6 parents up = repo root
BLIS_DIR = Path(__file__).parent.parent.parent.parent.parent.parent / ".nous-experiments" / "iter-1-01b1b82a"
BLIS_BIN = BLIS_DIR / "blis"

# Discrete parameter values
PARAMS = {
    "tp": [1, 2, 4, 8],
    "num_instances": [1, 2, 3, 4, 5, 6, 7, 8],  # filtered by TP constraint
    "scheduler": ["fcfs", "priority-fcfs", "sjf", "reverse-priority"],
    "max_num_running_reqs": [32, 64, 128, 256, 512],
    "max_num_scheduled_tokens": [2048, 4096, 8192],
    "long_prefill_token_threshold": [0, 1024, 2048, 4096],
    "block_size_in_tokens": [16, 32],
    "routing_policy": ["round-robin", "least-loaded", "weighted"],
    "routing_scorer_config": [
        "precise-prefix-cache:2,queue-depth:1,kv-utilization:1",
        "queue-depth:1,kv-utilization:1",
        "precise-prefix-cache:2,load-balance:1",
        "load-balance:1,kv-utilization:1",
    ],
    "admission_policy": ["always-admit", "tier-shed"],
    "preemption_policy": ["fcfs", "priority"],
    "gpu_memory_utilization": [0.85, 0.9, 0.95],
}

# Fixed parameters
FIXED = {
    "model": "qwen/qwen3-14b",
    "hardware": "H100",
    "latency_model": "trained-physics",
    "num_requests": 100,
    "rate": 20,
    "seed": 42,
}

# Reference point for hypervolume (worst possible + margin)
HV_REF = (0.0, 20000.0)  # (rps worst, ttft_p99 worst)

# Temp directory for metrics output (must be inside worktree)
METRICS_TMPDIR = BLIS_DIR / ".search_tmp"


def ensure_tmp_dir():
    METRICS_TMPDIR.mkdir(exist_ok=True)


def is_valid_config(cfg):
    """Check all constraints for a config dict."""
    tp = cfg["tp"]
    n = cfg["num_instances"]
    lptt = cfg["long_prefill_token_threshold"]
    mnst = cfg["max_num_scheduled_tokens"]
    rp = cfg["routing_policy"]
    rs = cfg["routing_scorer_config"]

    # TP * instances <= 8
    if tp * n > 8:
        return False

    # long_prefill_token_threshold < max_num_scheduled_tokens (or 0 = disabled)
    if lptt != 0 and lptt >= mnst:
        return False

    # routing_scorer_config only valid when routing_policy == weighted
    if rp != "weighted" and rs is not None:
        return False

    return True


def random_valid_config(rng=None):
    """Sample a uniformly random valid configuration."""
    if rng is None:
        rng = random
    while True:
        tp = rng.choice(PARAMS["tp"])
        max_instances = 8 // tp
        n = rng.randint(1, max_instances)
        scheduler = rng.choice(PARAMS["scheduler"])
        mnrr = rng.choice(PARAMS["max_num_running_reqs"])
        mnst = rng.choice(PARAMS["max_num_scheduled_tokens"])

        # long_prefill: pick from valid values (0 or < mnst)
        valid_lptt = [v for v in PARAMS["long_prefill_token_threshold"] if v == 0 or v < mnst]
        lptt = rng.choice(valid_lptt)

        bs = rng.choice(PARAMS["block_size_in_tokens"])
        rp = rng.choice(PARAMS["routing_policy"])
        rs = rng.choice(PARAMS["routing_scorer_config"]) if rp == "weighted" else None
        ap = rng.choice(PARAMS["admission_policy"])
        pp = rng.choice(PARAMS["preemption_policy"])
        gmu = rng.choice(PARAMS["gpu_memory_utilization"])

        cfg = {
            "tp": tp,
            "num_instances": n,
            "scheduler": scheduler,
            "max_num_running_reqs": mnrr,
            "max_num_scheduled_tokens": mnst,
            "long_prefill_token_threshold": lptt,
            "block_size_in_tokens": bs,
            "routing_policy": rp,
            "routing_scorer_config": rs,
            "admission_policy": ap,
            "preemption_policy": pp,
            "gpu_memory_utilization": gmu,
        }
        return cfg  # constraint already guaranteed by construction


def config_to_cmd(cfg, metrics_path):
    """Build the blis run command for a config."""
    cmd = [
        str(BLIS_BIN),
        "run",
        "--model", FIXED["model"],
        "--hardware", FIXED["hardware"],
        "--latency-model", FIXED["latency_model"],
        "--num-requests", str(FIXED["num_requests"]),
        "--rate", str(FIXED["rate"]),
        "--seed", str(FIXED["seed"]),
        "--tp", str(cfg["tp"]),
        "--num-instances", str(cfg["num_instances"]),
        "--scheduler", cfg["scheduler"],
        "--max-num-running-reqs", str(cfg["max_num_running_reqs"]),
        "--max-num-scheduled-tokens", str(cfg["max_num_scheduled_tokens"]),
        "--long-prefill-token-threshold", str(cfg["long_prefill_token_threshold"]),
        "--block-size-in-tokens", str(cfg["block_size_in_tokens"]),
        "--preemption-policy", cfg["preemption_policy"],
        "--gpu-memory-utilization", str(cfg["gpu_memory_utilization"]),
        "--metrics-path", str(metrics_path),
    ]
    # Only add routing/admission for multi-instance
    if cfg["num_instances"] > 1:
        cmd += ["--routing-policy", cfg["routing_policy"]]
        if cfg["routing_policy"] == "weighted" and cfg["routing_scorer_config"]:
            cmd += ["--routing-scorer-config", cfg["routing_scorer_config"]]
        cmd += ["--admission-policy", cfg["admission_policy"]]
    return cmd


_eval_counter = 0


def evaluate(cfg):
    """Run BLIS for a config and return (rps, ttft_p99) or None on failure."""
    global _eval_counter
    _eval_counter += 1
    ensure_tmp_dir()
    metrics_path = METRICS_TMPDIR / f"eval_{_eval_counter}_{os.getpid()}.json"
    cmd = config_to_cmd(cfg, metrics_path)
    try:
        result = subprocess.run(
            cmd,
            capture_output=True,
            text=True,
            timeout=30,
            cwd=str(BLIS_DIR),
        )
        if result.returncode != 0:
            return None
        with open(metrics_path) as f:
            data = json.load(f)
        return (data["responses_per_sec"], data["ttft_p99_ms"])
    except Exception:
        return None
    finally:
        try:
            metrics_path.unlink(missing_ok=True)
        except Exception:
            pass


def pareto_front(points):
    """
    Compute Pareto front from list of (rps, ttft_p99) tuples.
    Maximizing rps, minimizing ttft_p99.
    Returns list of non-dominated points.
    """
    dominated = set()
    n = len(points)
    for i in range(n):
        if i in dominated:
            continue
        for j in range(n):
            if i == j or j in dominated:
                continue
            # j dominates i if j is at least as good on all objectives and strictly better on one
            # Better: higher rps AND lower ttft_p99
            if points[j][0] >= points[i][0] and points[j][1] <= points[i][1]:
                if points[j][0] > points[i][0] or points[j][1] < points[i][1]:
                    dominated.add(i)
                    break
    return [points[i] for i in range(n) if i not in dominated]


def hypervolume_2d(front, ref=HV_REF):
    """
    Compute 2D hypervolume indicator for a Pareto front.
    Maximizing objective 0 (rps), minimizing objective 1 (ttft_p99).
    ref = (rps_min, ttft_p99_max) — the worst reference point.

    Hypervolume = area dominated by front but not by reference point.
    We maximize rps (obj0) and minimize ttft_p99 (obj1).
    Transform: maximize obj0, maximize (-obj1).
    ref in transformed space: (ref[0], -ref[1]) = (0, -20000)
    """
    if not front:
        return 0.0
    # Transform to maximization problem
    # obj0: rps (maximize), obj1: -ttft_p99 (maximize, since we minimize ttft_p99)
    transformed = [(p[0], -p[1]) for p in front]
    ref_t = (ref[0], -ref[1])  # (0, 20000) in maximization space

    # Filter dominated by reference
    valid = [(x, y) for x, y in transformed if x > ref_t[0] and y > ref_t[1]]
    if not valid:
        return 0.0

    # Sort by first objective descending
    valid.sort(key=lambda p: p[0], reverse=True)

    hv = 0.0
    prev_y = ref_t[1]
    for x, y in valid:
        if y > prev_y:
            hv += (x - ref_t[0]) * (y - prev_y)
            prev_y = y
    return hv


def config_key(cfg):
    """Hashable key for deduplication."""
    return (
        cfg["tp"], cfg["num_instances"], cfg["scheduler"],
        cfg["max_num_running_reqs"], cfg["max_num_scheduled_tokens"],
        cfg["long_prefill_token_threshold"], cfg["block_size_in_tokens"],
        cfg["routing_policy"], cfg["routing_scorer_config"],
        cfg["admission_policy"], cfg["preemption_policy"],
        cfg["gpu_memory_utilization"],
    )
