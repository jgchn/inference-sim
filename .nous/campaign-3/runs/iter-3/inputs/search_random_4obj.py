#!/usr/bin/env python3
"""Random search on 4-objective space: maximize rps, minimize ttft_p99, gpu_count, kv_blocks."""

import subprocess, json, os, sys, time
import numpy as np

BLIS = "/Users/jchen/go/src/inference-sim/inference-sim/.nous-experiments/iter-3-3e85f30b/blis"
RESULTS_DIR = "/Users/jchen/go/src/inference-sim/inference-sim/.nous/campaign-3/runs/iter-3/results/h-control-negative"
TMPDIR = os.environ.get("TMPDIR", "/tmp/claude")
REF_4D = [0.0, 50000.0, 9.0, 11000.0]

FIXED = [
    "--model", "qwen/qwen3-14b", "--hardware", "H100",
    "--latency-model", "trained-physics", "--num-requests", "200",
    "--rate", "50", "--prefix-tokens", "512", "--seed", "42",
    "--max-num-scheduled-tokens", "4096", "--long-prefill-token-threshold", "0",
    "--admission-policy", "always-admit", "--preemption-policy", "fcfs",
]

TP_OPTIONS = [1, 2, 4, 8]
SCHEDULER_OPTIONS = ["fcfs", "sjf"]
BATCH_OPTIONS = [32, 64, 128, 256, 512]
KV_OPTIONS = [2000, 3000, 4000, 5000, 7500, 10000]
BLOCK_SIZE_OPTIONS = [16, 32]
ROUTING_OPTIONS = ["round-robin", "least-loaded", "weighted"]
SCORER_OPTIONS = [
    "precise-prefix-cache:2,queue-depth:1,kv-utilization:1",
    "queue-depth:1,kv-utilization:1",
    "precise-prefix-cache:2,load-balance:1",
    "load-balance:1,kv-utilization:1",
]
GPU_MEM_OPTIONS = [0.85, 0.9, 0.95]

TOTAL_EVALS = 200
CHECKPOINT_EVERY = 40


def random_config(rng):
    tp = int(rng.choice(TP_OPTIONS))
    num_inst = int(rng.randint(1, 8 // tp + 1))
    scheduler = str(rng.choice(SCHEDULER_OPTIONS))
    max_batch = int(rng.choice(BATCH_OPTIONS))
    kv_blocks = int(rng.choice(KV_OPTIONS))
    block_size = int(rng.choice(BLOCK_SIZE_OPTIONS))
    gpu_mem = float(rng.choice(GPU_MEM_OPTIONS))
    if num_inst == 1:
        routing = "round-robin"
        scorers = None
    else:
        routing = str(rng.choice(ROUTING_OPTIONS))
        scorers = str(rng.choice(SCORER_OPTIONS)) if routing == "weighted" else None
    return {
        "tp": tp, "num_instances": num_inst,
        "scheduler": scheduler, "max_num_running_reqs": max_batch,
        "total_kv_blocks": kv_blocks, "block_size_in_tokens": block_size,
        "routing_policy": routing, "routing_scorers": scorers,
        "gpu_memory_utilization": gpu_mem,
    }


eval_counter = [0]


def evaluate(config):
    eval_counter[0] += 1
    mfile = os.path.join(TMPDIR, f"rand_4obj_{eval_counter[0]}.json")
    cmd = [BLIS, "run"] + FIXED + [
        "--tp", str(config["tp"]),
        "--num-instances", str(config["num_instances"]),
        "--scheduler", config["scheduler"],
        "--max-num-running-reqs", str(config["max_num_running_reqs"]),
        "--total-kv-blocks", str(config["total_kv_blocks"]),
        "--block-size-in-tokens", str(config["block_size_in_tokens"]),
        "--routing-policy", config["routing_policy"],
        "--gpu-memory-utilization", str(config["gpu_memory_utilization"]),
        "--metrics-path", mfile,
    ]
    if config.get("routing_scorers"):
        cmd += ["--routing-scorers", config["routing_scorers"]]
    try:
        r = subprocess.run(cmd, capture_output=True, timeout=300)
        if r.returncode != 0:
            print(f"  [eval {eval_counter[0]}] FAIL rc={r.returncode}", file=sys.stderr)
            return None
        with open(mfile) as f:
            d = json.load(f)
        return {"rps": d["responses_per_sec"], "ttft_p99": d["ttft_p99_ms"],
                "preemption_count": d.get("preemption_count", 0)}
    except Exception as e:
        print(f"  [eval {eval_counter[0]}] ERROR: {e}", file=sys.stderr)
        return None


def objectives(metrics, config):
    gpu = config["tp"] * config["num_instances"]
    return [-metrics["rps"], metrics["ttft_p99"], float(gpu), float(config["total_kv_blocks"])]


def dominates(a, b):
    return all(ai <= bi for ai, bi in zip(a, b)) and any(ai < bi for ai, bi in zip(a, b))


def get_pareto_front(all_objs):
    n = len(all_objs)
    dominated = [False] * n
    for i in range(n):
        if dominated[i]:
            continue
        for j in range(n):
            if i != j and not dominated[j] and dominates(all_objs[j], all_objs[i]):
                dominated[i] = True
                break
    return [i for i in range(n) if not dominated[i]]


def hypervolume_mc(pareto_objs, ref, n_samples=200000, seed=0):
    if not pareto_objs:
        return 0.0
    pts = np.array(pareto_objs, dtype=float)
    ref = np.array(ref, dtype=float)
    lb = pts.min(axis=0)
    if np.any(ref <= lb):
        return 0.0
    rng = np.random.RandomState(seed)
    samples = rng.uniform(lb, ref, size=(n_samples, len(ref)))
    dominated = np.zeros(n_samples, dtype=bool)
    for p in pts:
        dominated |= np.all(p[np.newaxis, :] <= samples, axis=1)
    return float(np.prod(ref - lb) * dominated.mean())


def main():
    rng = np.random.RandomState(456)  # different seed from NSGA-II arm
    os.makedirs(RESULTS_DIR, exist_ok=True)

    all_configs, all_metrics, all_objs = [], [], []
    convergence = []

    t0 = time.time()
    print(f"=== Random Search 4-Objective ({TOTAL_EVALS} evals) ===", file=sys.stderr)

    n_evals = 0
    while n_evals < TOTAL_EVALS:
        config = random_config(rng)
        metrics = evaluate(config)
        if metrics is None:
            continue
        objs = objectives(metrics, config)
        all_configs.append(config)
        all_metrics.append(metrics)
        all_objs.append(objs)
        n_evals += 1

        if n_evals % CHECKPOINT_EVERY == 0 or n_evals == TOTAL_EVALS:
            pf_idx = get_pareto_front(all_objs)
            hv = hypervolume_mc([all_objs[k] for k in pf_idx], REF_4D)
            convergence.append({"eval": n_evals, "hypervolume": hv, "pareto_size": len(pf_idx)})
            print(f"  [eval={n_evals}] PF={len(pf_idx)}, HV={hv:.4e}", file=sys.stderr)

    # Final Pareto front
    pf_idx_final = get_pareto_front(all_objs)
    pareto_front = []
    for i in pf_idx_final:
        c = all_configs[i]
        pareto_front.append({
            "config": c, "metrics": all_metrics[i],
            "objectives": all_objs[i],
            "gpu_count": c["tp"] * c["num_instances"],
        })
    pareto_front.sort(key=lambda x: x["objectives"][0])

    wall = time.time() - t0

    # Count unique non-dominated configs to measure Pareto density
    n_nd = len(pf_idx_final)
    density_pct = 100.0 * n_nd / len(all_objs)

    summary = {
        "algorithm": "random_4obj",
        "objectives": ["-rps", "ttft_p99", "gpu_count", "kv_blocks"],
        "reference_point": REF_4D,
        "total_evals": len(all_objs),
        "pareto_size": n_nd,
        "pareto_density_pct": density_pct,
        "final_hypervolume": convergence[-1]["hypervolume"],
        "wall_time_s": wall,
        "convergence": convergence,
        "pareto_front": pareto_front,
    }
    with open(os.path.join(RESULTS_DIR, "pareto_front.json"), "w") as f:
        json.dump(summary, f, indent=2)
    with open(os.path.join(RESULTS_DIR, "convergence.json"), "w") as f:
        json.dump(convergence, f, indent=2)

    print(f"\n=== Done: {len(all_objs)} evals in {wall:.1f}s, PF={n_nd} ({density_pct:.1f}%), HV={convergence[-1]['hypervolume']:.4e} ===", file=sys.stderr)


if __name__ == "__main__":
    main()
