"""
h-control-negative: Uniform random search over BLIS configuration space.
500 random valid configurations evaluated, Pareto front and hypervolume computed.
"""

import json
import random
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))
from search_common import (
    random_valid_config, evaluate, pareto_front, hypervolume_2d,
    config_key, HV_REF
)

RESULTS_DIR = Path(__file__).parent.parent / "results" / "h-control-negative"
NUM_EVALS = 500
SEED = 12345


def run():
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    rng = random.Random(SEED)
    t_start = time.time()

    results = []
    seen = set()
    n_eval = 0
    n_fail = 0

    print(f"[random] Starting {NUM_EVALS} evaluations...", flush=True)

    while n_eval < NUM_EVALS:
        cfg = random_valid_config(rng)
        key = config_key(cfg)
        if key in seen:
            continue
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
            print(f"[random] {n_eval}/{NUM_EVALS} evals, {elapsed:.1f}s elapsed, "
                  f"best rps={max(r['rps'] for r in results):.2f}, "
                  f"best ttft={min(r['ttft_p99'] for r in results):.1f}ms", flush=True)

    elapsed = time.time() - t_start

    # Compute Pareto front and hypervolume
    points = [(r["rps"], r["ttft_p99"]) for r in results]
    front = pareto_front(points)
    hv = hypervolume_2d(front, HV_REF)

    summary = {
        "arm": "h-control-negative",
        "method": "random",
        "num_evals": n_eval,
        "num_failures": n_fail,
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

    print(f"\n[random] Done. {n_eval} evals in {elapsed:.1f}s")
    print(f"[random] Pareto front: {len(front)} points, hypervolume={hv:.4f}")
    print(f"[random] Best rps={summary['best_rps']:.2f}, best ttft_p99={summary['best_ttft']:.1f}ms")
    print(f"[random] Results written to {out_path}")
    return summary


if __name__ == "__main__":
    run()
