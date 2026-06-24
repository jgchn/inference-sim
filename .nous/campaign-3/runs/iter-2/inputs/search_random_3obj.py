"""
Random search over BLIS 3-objective space.
Arm: h-control-negative

Objectives (all minimize):
  obj0 = -responses_per_sec
  obj1 = ttft_p99_ms
  obj2 = gpu_count = tp * num_instances

Reference point: (0, 40000, 9)
Evaluations: 550, tracked every 50.
"""

import json
import os
import random
import subprocess
import sys
import tempfile
import time

BLIS = "/Users/jchen/go/src/inference-sim/inference-sim/blis"
OUT_DIR = "/Users/jchen/go/src/inference-sim/inference-sim/.nous/campaign-3/runs/iter-2/results/h-control-negative"
REF = (0.0, 40000.0, 9.0)

TP_VALUES = [1, 2, 4, 8]
SCHEDULER_VALUES = ["fcfs", "priority-fcfs", "sjf", "reverse-priority"]
MAX_RUNNING_VALUES = [32, 64, 128, 256, 512]
MAX_TOKENS_VALUES = [2048, 4096, 8192]
PREFILL_THRESHOLD_VALUES = [0, 1024, 2048, 4096]
BLOCK_SIZE_VALUES = [16, 32]
ROUTING_POLICY_VALUES = ["round-robin", "least-loaded", "weighted"]
ROUTING_SCORER_CONFIGS = [
    "precise-prefix-cache:2,queue-depth:1,kv-utilization:1",
    "queue-depth:1,kv-utilization:1",
    "precise-prefix-cache:2,load-balance:1",
    "load-balance:1,kv-utilization:1",
]
ADMISSION_POLICY_VALUES = ["always-admit", "tier-shed"]
PREEMPTION_POLICY_VALUES = ["fcfs", "priority"]
GPU_MEM_UTIL_VALUES = [0.85, 0.9, 0.95]


def sample_config(rng):
    tp = rng.choice(TP_VALUES)
    max_inst = 8 // tp
    num_instances = rng.randint(1, max_inst)
    scheduler = rng.choice(SCHEDULER_VALUES)
    max_running = rng.choice(MAX_RUNNING_VALUES)
    max_tokens = rng.choice(MAX_TOKENS_VALUES)
    valid_prefill = [t for t in PREFILL_THRESHOLD_VALUES if t == 0 or t < max_tokens]
    prefill_threshold = rng.choice(valid_prefill)
    block_size = rng.choice(BLOCK_SIZE_VALUES)
    preemption = rng.choice(PREEMPTION_POLICY_VALUES)
    gpu_mem_util = rng.choice(GPU_MEM_UTIL_VALUES)

    if num_instances == 1:
        routing_policy = "round-robin"
        admission_policy = "always-admit"
        routing_scorers = None
    else:
        routing_policy = rng.choice(ROUTING_POLICY_VALUES)
        admission_policy = rng.choice(ADMISSION_POLICY_VALUES)
        routing_scorers = rng.choice(ROUTING_SCORER_CONFIGS) if routing_policy == "weighted" else None

    return {
        "tp": tp,
        "num_instances": num_instances,
        "scheduler": scheduler,
        "max_num_running_reqs": max_running,
        "max_num_scheduled_tokens": max_tokens,
        "long_prefill_token_threshold": prefill_threshold,
        "block_size_in_tokens": block_size,
        "routing_policy": routing_policy,
        "routing_scorers": routing_scorers,
        "admission_policy": admission_policy,
        "preemption_policy": preemption,
        "gpu_memory_utilization": gpu_mem_util,
    }


def build_cmd(config, metrics_path):
    cmd = [
        BLIS, "run",
        "--model", "qwen/qwen3-14b",
        "--hardware", "H100",
        "--latency-model", "trained-physics",
        "--num-requests", "500",
        "--rate", "50",
        "--seed", "42",
        "--tp", str(config["tp"]),
        "--num-instances", str(config["num_instances"]),
        "--scheduler", config["scheduler"],
        "--max-num-running-reqs", str(config["max_num_running_reqs"]),
        "--max-num-scheduled-tokens", str(config["max_num_scheduled_tokens"]),
        "--long-prefill-token-threshold", str(config["long_prefill_token_threshold"]),
        "--block-size-in-tokens", str(config["block_size_in_tokens"]),
        "--preemption-policy", config["preemption_policy"],
        "--gpu-memory-utilization", str(config["gpu_memory_utilization"]),
        "--metrics-path", metrics_path,
    ]
    if config["num_instances"] > 1:
        cmd += ["--routing-policy", config["routing_policy"]]
        cmd += ["--admission-policy", config["admission_policy"]]
        if config["routing_policy"] == "weighted" and config["routing_scorers"]:
            cmd += ["--routing-scorers", config["routing_scorers"]]
    return cmd


def evaluate(config):
    fd, path = tempfile.mkstemp(suffix=".json", dir=os.environ.get("TMPDIR", "/tmp"))
    os.close(fd)
    try:
        cmd = build_cmd(config, path)
        result = subprocess.run(cmd, capture_output=True, timeout=60)
        if result.returncode != 0:
            return None
        with open(path) as f:
            m = json.load(f)
        rps = float(m["responses_per_sec"])
        ttft = float(m["ttft_p99_ms"])
        gpu = config["tp"] * config["num_instances"]
        return (-rps, ttft, float(gpu))
    except Exception:
        return None
    finally:
        try:
            os.unlink(path)
        except OSError:
            pass


def pareto_front_2d(points):
    nd = []
    for p in points:
        dominated = False
        to_remove = []
        for i, q in enumerate(nd):
            q_dom_p = (q[0] <= p[0] and q[1] <= p[1]) and (q[0] < p[0] or q[1] < p[1])
            p_dom_q = (p[0] <= q[0] and p[1] <= q[1]) and (p[0] < q[0] or p[1] < q[1])
            if q_dom_p:
                dominated = True
                break
            if p_dom_q:
                to_remove.append(i)
        if not dominated:
            nd = [q for i, q in enumerate(nd) if i not in set(to_remove)]
            nd.append(p)
    return nd


def hv_2d(points_2d, ref2d):
    nd = pareto_front_2d(points_2d)
    if not nd:
        return 0.0
    nd.sort(key=lambda p: p[0])
    hv = 0.0
    for i, p in enumerate(nd):
        if p[1] >= ref2d[1]:
            continue
        next_x = nd[i + 1][0] if i < len(nd) - 1 else ref2d[0]
        width = next_x - p[0]
        height = ref2d[1] - p[1]
        if width > 0 and height > 0:
            hv += width * height
    return hv


def pareto_front_3d(points):
    nd = []
    for p in points:
        dominated = False
        to_remove = []
        for i, q in enumerate(nd):
            q_dom_p = all(q[j] <= p[j] for j in range(3)) and any(q[j] < p[j] for j in range(3))
            p_dom_q = all(p[j] <= q[j] for j in range(3)) and any(p[j] < q[j] for j in range(3))
            if q_dom_p:
                dominated = True
                break
            if p_dom_q:
                to_remove.append(i)
        if not dominated:
            nd = [q for i, q in enumerate(nd) if i not in set(to_remove)]
            nd.append(p)
    return nd


def hv_3d(points_3d, ref=REF):
    valid = [p for p in points_3d if p[0] < ref[0] and p[1] < ref[1] and p[2] < ref[2]]
    if not valid:
        return 0.0
    nd = pareto_front_3d(valid)
    if not nd:
        return 0.0
    nd.sort(key=lambda p: p[0])
    hv = 0.0
    for i in range(len(nd)):
        slab = (nd[i + 1][0] if i < len(nd) - 1 else ref[0]) - nd[i][0]
        if slab <= 0:
            continue
        proj = [(p[1], p[2]) for p in nd[: i + 1]]
        hv += slab * hv_2d(proj, (ref[1], ref[2]))
    return hv


def main():
    rng = random.Random(123)
    n_evals = 550
    checkpoint_every = 50

    all_results = []  # list of (config, objectives)
    convergence = []  # list of {"eval": N, "hv": V}
    all_objectives = []  # running list of valid objectives

    print(f"Running random search: {n_evals} evaluations", flush=True)
    t0 = time.time()

    for i in range(n_evals):
        config = sample_config(rng)
        obj = evaluate(config)
        if obj is not None:
            all_results.append({"config": config, "objectives": list(obj), "metrics": {
                "responses_per_sec": -obj[0],
                "ttft_p99_ms": obj[1],
            }, "eval_num": i + 1})
            all_objectives.append(obj)

        eval_num = i + 1
        if eval_num % checkpoint_every == 0 or eval_num == n_evals:
            hv = hv_3d(all_objectives)
            convergence.append({"eval": eval_num, "hv": hv})
            elapsed = time.time() - t0
            print(f"  eval={eval_num:3d}  hv={hv:.1f}  elapsed={elapsed:.1f}s", flush=True)

    pareto_objs = pareto_front_3d(all_objectives)
    pareto_records = [r for r in all_results if tuple(r["objectives"]) in [tuple(p) for p in pareto_objs]]
    # Deduplicate by objectives
    seen = set()
    pareto_out = []
    for r in pareto_records:
        key = tuple(r["objectives"])
        if key not in seen:
            seen.add(key)
            pareto_out.append(r)

    os.makedirs(OUT_DIR, exist_ok=True)
    with open(os.path.join(OUT_DIR, "pareto_front.json"), "w") as f:
        json.dump(pareto_out, f, indent=2)
    with open(os.path.join(OUT_DIR, "convergence.json"), "w") as f:
        json.dump(convergence, f, indent=2)

    print(f"\nDone. Pareto front size: {len(pareto_out)}")
    print(f"Final HV: {convergence[-1]['hv']:.1f}")
    print(f"Total time: {time.time() - t0:.1f}s")


if __name__ == "__main__":
    main()
