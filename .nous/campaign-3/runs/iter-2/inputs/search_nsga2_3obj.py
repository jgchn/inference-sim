"""
NSGA-II search over BLIS 3-objective space.
Arm: h-main

Population: 50, Generations: 11 → 550 total evaluations (50 initial + 10 * 50 offspring)
Objectives (all minimize):
  obj0 = -responses_per_sec
  obj1 = ttft_p99_ms
  obj2 = gpu_count = tp * num_instances

Reference point: (0, 40000, 9)
HV tracked after each generation (every 50 evals).
"""

import json
import os
import random
import subprocess
import sys
import tempfile
import time

BLIS = "/Users/jchen/go/src/inference-sim/inference-sim/blis"
OUT_DIR = "/Users/jchen/go/src/inference-sim/inference-sim/.nous/campaign-3/runs/iter-2/results/h-main"
REF = (0.0, 40000.0, 9.0)
POP_SIZE = 50
N_GENERATIONS = 10  # 50 initial + 10 * 50 = 550 total

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

# All mutable parameters as lists (for uniform crossover indexing)
PARAM_KEYS = [
    "tp", "num_instances", "scheduler", "max_num_running_reqs",
    "max_num_scheduled_tokens", "long_prefill_token_threshold",
    "block_size_in_tokens", "routing_policy", "routing_scorers",
    "admission_policy", "preemption_policy", "gpu_memory_utilization",
]


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


def repair_config(config):
    """Ensure all constraints hold after crossover/mutation."""
    tp = config["tp"]
    max_inst = 8 // tp
    config["num_instances"] = max(1, min(config["num_instances"], max_inst))

    max_tokens = config["max_num_scheduled_tokens"]
    pft = config["long_prefill_token_threshold"]
    if pft != 0 and pft >= max_tokens:
        config["long_prefill_token_threshold"] = 0

    if config["num_instances"] == 1:
        config["routing_policy"] = "round-robin"
        config["admission_policy"] = "always-admit"
        config["routing_scorers"] = None
    else:
        if config["routing_policy"] != "weighted":
            config["routing_scorers"] = None
        elif config["routing_scorers"] is None:
            config["routing_scorers"] = ROUTING_SCORER_CONFIGS[0]

    return config


def crossover(p1, p2, rng):
    child = {}
    for key in PARAM_KEYS:
        child[key] = rng.choice([p1[key], p2[key]])
    return repair_config(child)


def mutate(config, rng, mutation_rate=None):
    if mutation_rate is None:
        mutation_rate = 1.0 / len(PARAM_KEYS)
    config = dict(config)
    for key in PARAM_KEYS:
        if rng.random() < mutation_rate:
            if key == "tp":
                config["tp"] = rng.choice(TP_VALUES)
            elif key == "num_instances":
                max_inst = 8 // config["tp"]
                config["num_instances"] = rng.randint(1, max_inst)
            elif key == "scheduler":
                config["scheduler"] = rng.choice(SCHEDULER_VALUES)
            elif key == "max_num_running_reqs":
                config["max_num_running_reqs"] = rng.choice(MAX_RUNNING_VALUES)
            elif key == "max_num_scheduled_tokens":
                config["max_num_scheduled_tokens"] = rng.choice(MAX_TOKENS_VALUES)
            elif key == "long_prefill_token_threshold":
                valid = [t for t in PREFILL_THRESHOLD_VALUES if t == 0 or t < config["max_num_scheduled_tokens"]]
                config["long_prefill_token_threshold"] = rng.choice(valid)
            elif key == "block_size_in_tokens":
                config["block_size_in_tokens"] = rng.choice(BLOCK_SIZE_VALUES)
            elif key == "routing_policy":
                if config["num_instances"] > 1:
                    config["routing_policy"] = rng.choice(ROUTING_POLICY_VALUES)
            elif key == "routing_scorers":
                if config["num_instances"] > 1 and config["routing_policy"] == "weighted":
                    config["routing_scorers"] = rng.choice(ROUTING_SCORER_CONFIGS)
            elif key == "admission_policy":
                if config["num_instances"] > 1:
                    config["admission_policy"] = rng.choice(ADMISSION_POLICY_VALUES)
            elif key == "preemption_policy":
                config["preemption_policy"] = rng.choice(PREEMPTION_POLICY_VALUES)
            elif key == "gpu_memory_utilization":
                config["gpu_memory_utilization"] = rng.choice(GPU_MEM_UTIL_VALUES)
    return repair_config(config)


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


# --- NSGA-II core ---

def dominates(a, b):
    """Returns True if a dominates b (all <=, at least one <) in minimize."""
    return all(a[i] <= b[i] for i in range(3)) and any(a[i] < b[i] for i in range(3))


def fast_nondominated_sort(objectives):
    """Returns list of fronts (each front is list of indices into objectives)."""
    n = len(objectives)
    dominated_count = [0] * n
    dominated_by = [[] for _ in range(n)]

    for i in range(n):
        for j in range(i + 1, n):
            if dominates(objectives[i], objectives[j]):
                dominated_by[i].append(j)
                dominated_count[j] += 1
            elif dominates(objectives[j], objectives[i]):
                dominated_by[j].append(i)
                dominated_count[i] += 1

    fronts = []
    current = [i for i in range(n) if dominated_count[i] == 0]
    while current:
        fronts.append(current)
        next_front = []
        for i in current:
            for j in dominated_by[i]:
                dominated_count[j] -= 1
                if dominated_count[j] == 0:
                    next_front.append(j)
        current = next_front
    return fronts


def crowding_distance(objectives, front):
    """Returns dict {idx: distance} for indices in front."""
    k = len(front)
    n_obj = 3
    distances = {idx: 0.0 for idx in front}
    if k <= 2:
        for idx in front:
            distances[idx] = float("inf")
        return distances

    for m in range(n_obj):
        sorted_front = sorted(front, key=lambda i: objectives[i][m])
        distances[sorted_front[0]] = float("inf")
        distances[sorted_front[-1]] = float("inf")
        obj_range = objectives[sorted_front[-1]][m] - objectives[sorted_front[0]][m]
        if obj_range == 0:
            continue
        for j in range(1, k - 1):
            distances[sorted_front[j]] += (
                objectives[sorted_front[j + 1]][m] - objectives[sorted_front[j - 1]][m]
            ) / obj_range
    return distances


def select_next_population(combined_objs, combined_configs, pop_size):
    fronts = fast_nondominated_sort(combined_objs)
    selected_idx = []
    for front in fronts:
        if len(selected_idx) + len(front) <= pop_size:
            selected_idx.extend(front)
        else:
            needed = pop_size - len(selected_idx)
            dist = crowding_distance(combined_objs, front)
            sorted_front = sorted(front, key=lambda i: dist[i], reverse=True)
            selected_idx.extend(sorted_front[:needed])
            break

    new_objs = [combined_objs[i] for i in selected_idx]
    new_configs = [combined_configs[i] for i in selected_idx]
    return new_objs, new_configs


def tournament_select(objectives, fronts, distances, k=2, rng=None):
    candidates = rng.sample(range(len(objectives)), k)
    rank = {}
    for r, front in enumerate(fronts):
        for idx in front:
            rank[idx] = r

    def better(a, b):
        if rank.get(a, 999) < rank.get(b, 999):
            return a
        if rank.get(a, 999) > rank.get(b, 999):
            return b
        return a if distances.get(a, 0) >= distances.get(b, 0) else b

    winner = candidates[0]
    for c in candidates[1:]:
        winner = better(winner, c)
    return winner


# --- HV helpers (same as random arm) ---

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
    rng = random.Random(456)  # different seed from random arm
    t0 = time.time()
    total_evals = 0

    print(f"NSGA-II: pop={POP_SIZE}, gen={N_GENERATIONS}, total_evals={POP_SIZE + N_GENERATIONS * POP_SIZE}", flush=True)

    # --- Initialize population ---
    print("Initializing population...", flush=True)
    pop_configs = [sample_config(rng) for _ in range(POP_SIZE)]
    pop_objs = []
    for i, cfg in enumerate(pop_configs):
        obj = evaluate(cfg)
        if obj is None:
            obj = (REF[0], REF[1], REF[2])  # failed eval: use reference (worst)
        pop_objs.append(obj)
        total_evals += 1

    all_objectives_seen = list(pop_objs)
    convergence = []
    all_results_log = []

    for i, (cfg, obj) in enumerate(zip(pop_configs, pop_objs)):
        all_results_log.append({"config": cfg, "objectives": list(obj),
                                 "metrics": {"responses_per_sec": -obj[0], "ttft_p99_ms": obj[1]},
                                 "eval_num": i + 1, "generation": 0})

    hv = hv_3d(all_objectives_seen)
    convergence.append({"eval": total_evals, "hv": hv, "generation": 0})
    print(f"  gen=0 eval={total_evals} hv={hv:.1f} elapsed={time.time()-t0:.1f}s", flush=True)

    # --- Evolutionary loop ---
    for gen in range(1, N_GENERATIONS + 1):
        fronts = fast_nondominated_sort(pop_objs)
        dist_all = {}
        for front in fronts:
            dist_all.update(crowding_distance(pop_objs, front))

        # Create offspring
        offspring_configs = []
        offspring_objs = []
        gen_start = total_evals

        while len(offspring_configs) < POP_SIZE:
            p1_idx = tournament_select(pop_objs, fronts, dist_all, k=2, rng=rng)
            p2_idx = tournament_select(pop_objs, fronts, dist_all, k=2, rng=rng)
            child = crossover(pop_configs[p1_idx], pop_configs[p2_idx], rng)
            child = mutate(child, rng)
            obj = evaluate(child)
            if obj is None:
                obj = (REF[0], REF[1], REF[2])
            offspring_configs.append(child)
            offspring_objs.append(obj)
            total_evals += 1
            all_objectives_seen.append(obj)
            all_results_log.append({
                "config": child, "objectives": list(obj),
                "metrics": {"responses_per_sec": -obj[0], "ttft_p99_ms": obj[1]},
                "eval_num": total_evals, "generation": gen
            })

        # Combine and select
        combined_objs = pop_objs + offspring_objs
        combined_configs = pop_configs + offspring_configs
        pop_objs, pop_configs = select_next_population(combined_objs, combined_configs, POP_SIZE)

        hv = hv_3d(all_objectives_seen)
        convergence.append({"eval": total_evals, "hv": hv, "generation": gen})
        print(f"  gen={gen} eval={total_evals} hv={hv:.1f} elapsed={time.time()-t0:.1f}s", flush=True)

    # Final Pareto front from all evaluated points
    pareto_objs = pareto_front_3d(all_objectives_seen)
    pareto_obj_set = {tuple(p) for p in pareto_objs}
    seen = set()
    pareto_out = []
    for r in all_results_log:
        key = tuple(r["objectives"])
        if key in pareto_obj_set and key not in seen:
            seen.add(key)
            pareto_out.append(r)

    os.makedirs(OUT_DIR, exist_ok=True)
    with open(os.path.join(OUT_DIR, "pareto_front.json"), "w") as f:
        json.dump(pareto_out, f, indent=2)
    with open(os.path.join(OUT_DIR, "convergence.json"), "w") as f:
        json.dump(convergence, f, indent=2)

    print(f"\nDone. Pareto front size: {len(pareto_out)}")
    print(f"Final HV: {convergence[-1]['hv']:.1f}")
    print(f"Total evals: {total_evals}, Total time: {time.time()-t0:.1f}s")


if __name__ == "__main__":
    main()
