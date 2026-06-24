"""
h-robustness: Latin Hypercube Sampling over BLIS configuration space.
500 LHS-stratified evaluations, Pareto front and hypervolume computed.
Tests whether stratified coverage alone (without evolutionary selection) improves over random.
"""

import json
import random
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))
from search_common import (
    PARAMS, random_valid_config, evaluate, pareto_front, hypervolume_2d,
    config_key, HV_REF
)

RESULTS_DIR = Path(__file__).parent.parent / "results" / "h-robustness"
NUM_EVALS = 500
SEED = 99999


def lhs_sample(n_samples, rng):
    """
    Generate Latin Hypercube samples for the BLIS parameter space.
    Returns list of n_samples config dicts, all valid.

    Strategy: For each parameter, create n_samples strata and sample one value
    from each stratum. Then pair using a random permutation per parameter.
    For discrete parameters, strata map to discrete values with repetition
    if n_samples > num_values.
    """
    params_ordered = [
        ("tp", PARAMS["tp"]),
        ("scheduler", PARAMS["scheduler"]),
        ("max_num_running_reqs", PARAMS["max_num_running_reqs"]),
        ("max_num_scheduled_tokens", PARAMS["max_num_scheduled_tokens"]),
        ("long_prefill_token_threshold", PARAMS["long_prefill_token_threshold"]),
        ("block_size_in_tokens", PARAMS["block_size_in_tokens"]),
        ("routing_policy", PARAMS["routing_policy"]),
        ("routing_scorer_config", PARAMS["routing_scorer_config"]),
        ("admission_policy", PARAMS["admission_policy"]),
        ("preemption_policy", PARAMS["preemption_policy"]),
        ("gpu_memory_utilization", PARAMS["gpu_memory_utilization"]),
        # num_instances is handled separately (depends on tp)
    ]

    # For each parameter, generate a permuted assignment of strata to samples
    assignments = {}
    for name, values in params_ordered:
        n_vals = len(values)
        # Create n_samples assignments cycling through strata
        strata = list(range(n_samples))
        rng.shuffle(strata)
        assignments[name] = [values[s % n_vals] for s in strata]

    # Build configs
    configs = []
    for i in range(n_samples):
        tp = assignments["tp"][i]
        max_n = 8 // tp

        # num_instances: LHS over [1, max_n]
        # Use a separate stratum assignment per sample
        n_inst = rng.randint(1, max_n)

        mnst = assignments["max_num_scheduled_tokens"][i]
        lptt_raw = assignments["long_prefill_token_threshold"][i]
        # Enforce constraint: lptt < mnst or 0
        if lptt_raw != 0 and lptt_raw >= mnst:
            valid_lptt = [v for v in PARAMS["long_prefill_token_threshold"]
                          if v == 0 or v < mnst]
            lptt = rng.choice(valid_lptt)
        else:
            lptt = lptt_raw

        rp = assignments["routing_policy"][i]
        rs = assignments["routing_scorer_config"][i] if rp == "weighted" else None

        cfg = {
            "tp": tp,
            "num_instances": n_inst,
            "scheduler": assignments["scheduler"][i],
            "max_num_running_reqs": assignments["max_num_running_reqs"][i],
            "max_num_scheduled_tokens": mnst,
            "long_prefill_token_threshold": lptt,
            "block_size_in_tokens": assignments["block_size_in_tokens"][i],
            "routing_policy": rp,
            "routing_scorer_config": rs,
            "admission_policy": assignments["admission_policy"][i],
            "preemption_policy": assignments["preemption_policy"][i],
            "gpu_memory_utilization": assignments["gpu_memory_utilization"][i],
        }
        configs.append(cfg)

    return configs


def run():
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    rng = random.Random(SEED)
    t_start = time.time()

    print(f"[lhs] Generating {NUM_EVALS} LHS configs...", flush=True)
    lhs_configs = lhs_sample(NUM_EVALS, rng)

    results = []
    seen = set()
    n_eval = 0
    n_fail = 0
    n_dup = 0

    print(f"[lhs] Starting {NUM_EVALS} evaluations...", flush=True)

    for cfg in lhs_configs:
        key = config_key(cfg)
        if key in seen:
            n_dup += 1
            # Replace with a fresh random config to maintain budget
            cfg = random_valid_config(rng)
            key = config_key(cfg)
            seen.add(key)
        else:
            seen.add(key)

        obj = evaluate(cfg)
        n_eval += 1

        if obj is None:
            n_fail += 1
            continue

        rps, ttft = obj
        results.append({
            "config": dict(cfg),
            "rps": rps,
            "ttft_p99": ttft,
        })

        if n_eval % 50 == 0:
            elapsed = time.time() - t_start
            print(f"[lhs] {n_eval}/{NUM_EVALS} evals, {elapsed:.1f}s elapsed, "
                  f"best rps={max(r['rps'] for r in results):.2f}, "
                  f"best ttft={min(r['ttft_p99'] for r in results):.1f}ms", flush=True)

    elapsed = time.time() - t_start

    # Compute Pareto front and hypervolume
    points = [(r["rps"], r["ttft_p99"]) for r in results]
    front = pareto_front(points)
    hv = hypervolume_2d(front, HV_REF)

    summary = {
        "arm": "h-robustness",
        "method": "lhs",
        "num_evals": n_eval,
        "num_failures": n_fail,
        "num_duplicates_replaced": n_dup,
        "num_results": len(results),
        "pareto_front_size": len(front),
        "hypervolume": hv,
        "hypervolume_ref": list(HV_REF),
        "elapsed_s": elapsed,
        "pareto_front": sorted(front, key=lambda p: p[0]),
        "all_points": [[r["rps"], r["ttft_p99"]] for r in results],
        "best_rps": max(r["rps"] for r in results) if results else 0,
        "best_ttft": min(r["ttft_p99"] for r in results) if results else 0,
    }

    out_path = RESULTS_DIR / "summary.json"
    with open(out_path, "w") as f:
        json.dump(summary, f, indent=2)

    print(f"\n[lhs] Done. {n_eval} evals in {elapsed:.1f}s")
    print(f"[lhs] Pareto front: {len(front)} points, hypervolume={hv:.4f}")
    print(f"[lhs] Best rps={summary['best_rps']:.2f}, best ttft_p99={summary['best_ttft']:.1f}ms")
    print(f"[lhs] Results written to {out_path}")
    return summary


if __name__ == "__main__":
    run()
