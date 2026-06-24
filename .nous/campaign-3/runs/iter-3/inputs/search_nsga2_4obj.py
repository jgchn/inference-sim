#!/usr/bin/env python3
"""NSGA-II search on 4-objective space: maximize rps, minimize ttft_p99, gpu_count, kv_blocks."""

import subprocess, json, os, sys, time
import numpy as np

BLIS = "/Users/jchen/go/src/inference-sim/inference-sim/.nous-experiments/iter-3-3e85f30b/blis"
RESULTS_DIR = "/Users/jchen/go/src/inference-sim/inference-sim/.nous/campaign-3/runs/iter-3/results/h-main"
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

POP_SIZE = 40
N_GEN = 4  # 40 init + 4*40 offspring = 200 total


def random_config(rng):
    tp = int(rng.choice(TP_OPTIONS))
    num_inst = int(rng.randint(1, 8 // tp + 1))
    return _build_config(tp, num_inst, rng)


def _build_config(tp, num_inst, rng):
    scheduler = rng.choice(SCHEDULER_OPTIONS)
    max_batch = int(rng.choice(BATCH_OPTIONS))
    kv_blocks = int(rng.choice(KV_OPTIONS))
    block_size = int(rng.choice(BLOCK_SIZE_OPTIONS))
    gpu_mem = float(rng.choice(GPU_MEM_OPTIONS))
    if num_inst == 1:
        routing = "round-robin"
        scorers = None
    else:
        routing = rng.choice(ROUTING_OPTIONS)
        scorers = rng.choice(SCORER_OPTIONS) if routing == "weighted" else None
    return {
        "tp": tp, "num_instances": num_inst,
        "scheduler": str(scheduler), "max_num_running_reqs": max_batch,
        "total_kv_blocks": kv_blocks, "block_size_in_tokens": block_size,
        "routing_policy": str(routing),
        "routing_scorers": str(scorers) if scorers is not None else None,
        "gpu_memory_utilization": gpu_mem,
    }


eval_counter = [0]


def evaluate(config):
    eval_counter[0] += 1
    mfile = os.path.join(TMPDIR, f"nsga2_4obj_{eval_counter[0]}.json")
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


def fast_non_dominated_sort(objs):
    n = len(objs)
    dom_count = [0] * n
    dom_set = [[] for _ in range(n)]
    for i in range(n):
        for j in range(n):
            if i == j:
                continue
            if dominates(objs[i], objs[j]):
                dom_set[i].append(j)
            elif dominates(objs[j], objs[i]):
                dom_count[i] += 1
    fronts = []
    current = [i for i in range(n) if dom_count[i] == 0]
    while current:
        fronts.append(current)
        nxt = []
        for i in current:
            for j in dom_set[i]:
                dom_count[j] -= 1
                if dom_count[j] == 0:
                    nxt.append(j)
        current = nxt
    return fronts


def crowding_distance(objs):
    n = len(objs)
    if n <= 2:
        return [float("inf")] * n
    d = len(objs[0])
    dist = [0.0] * n
    for dim in range(d):
        si = sorted(range(n), key=lambda i: objs[i][dim])
        dist[si[0]] = float("inf")
        dist[si[-1]] = float("inf")
        fmin, fmax = objs[si[0]][dim], objs[si[-1]][dim]
        if fmax == fmin:
            continue
        for k in range(1, n - 1):
            dist[si[k]] += (objs[si[k+1]][dim] - objs[si[k-1]][dim]) / (fmax - fmin)
    return dist


def tournament_select(pop, ranks, cdists, rng):
    i, j = rng.choice(len(pop), 2, replace=False)
    ri, rj = ranks[i], ranks[j]
    ci, cj = cdists[i], cdists[j]
    if ri < rj or (ri == rj and ci > cj):
        return pop[i]
    return pop[j]


def crossover(p1, p2, rng):
    tp = p1["tp"] if rng.rand() < 0.5 else p2["tp"]
    max_inst = 8 // tp
    num_inst = min(p1["num_instances"] if rng.rand() < 0.5 else p2["num_instances"], max_inst)
    child = {"tp": tp, "num_instances": num_inst}
    for k in ["scheduler", "max_num_running_reqs", "total_kv_blocks",
              "block_size_in_tokens", "gpu_memory_utilization"]:
        child[k] = p1[k] if rng.rand() < 0.5 else p2[k]
    if num_inst == 1:
        child["routing_policy"] = "round-robin"
        child["routing_scorers"] = None
    else:
        child["routing_policy"] = p1["routing_policy"] if rng.rand() < 0.5 else p2["routing_policy"]
        if child["routing_policy"] == "weighted":
            s1 = p1.get("routing_scorers") or SCORER_OPTIONS[0]
            s2 = p2.get("routing_scorers") or SCORER_OPTIONS[0]
            child["routing_scorers"] = s1 if rng.rand() < 0.5 else s2
        else:
            child["routing_scorers"] = None
    return child


def mutate(config, rng, prob=0.15):
    c = dict(config)
    if rng.rand() < prob:
        c["tp"] = int(rng.choice(TP_OPTIONS))
        c["num_instances"] = min(c["num_instances"], 8 // c["tp"])
    if rng.rand() < prob:
        c["num_instances"] = int(rng.randint(1, 8 // c["tp"] + 1))
    if rng.rand() < prob:
        c["scheduler"] = str(rng.choice(SCHEDULER_OPTIONS))
    if rng.rand() < prob:
        c["max_num_running_reqs"] = int(rng.choice(BATCH_OPTIONS))
    if rng.rand() < prob:
        c["total_kv_blocks"] = int(rng.choice(KV_OPTIONS))
    if rng.rand() < prob:
        c["block_size_in_tokens"] = int(rng.choice(BLOCK_SIZE_OPTIONS))
    if rng.rand() < prob:
        c["gpu_memory_utilization"] = float(rng.choice(GPU_MEM_OPTIONS))
    if c["num_instances"] == 1:
        c["routing_policy"] = "round-robin"
        c["routing_scorers"] = None
    elif rng.rand() < prob:
        c["routing_policy"] = str(rng.choice(ROUTING_OPTIONS))
        if c["routing_policy"] == "weighted":
            c["routing_scorers"] = str(rng.choice(SCORER_OPTIONS))
        else:
            c["routing_scorers"] = None
    return c


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
    rng = np.random.RandomState(123)
    os.makedirs(RESULTS_DIR, exist_ok=True)

    all_configs, all_metrics, all_objs = [], [], []
    convergence = []

    t0 = time.time()
    print("=== NSGA-II 4-Objective Search (pop=40, gen=4) ===", file=sys.stderr)

    # --- Initial population ---
    print(f"Initializing population (N={POP_SIZE})...", file=sys.stderr)
    population = []
    i = 0
    while len(population) < POP_SIZE:
        config = random_config(rng)
        metrics = evaluate(config)
        if metrics is None:
            continue
        objs = objectives(metrics, config)
        population.append({"config": config, "metrics": metrics, "objectives": objs})
        all_configs.append(config)
        all_metrics.append(metrics)
        all_objs.append(objs)
        i += 1
        if i % 10 == 0:
            print(f"  init {len(population)}/{POP_SIZE}", file=sys.stderr)

    pf_idx = get_pareto_front(all_objs)
    hv = hypervolume_mc([all_objs[k] for k in pf_idx], REF_4D)
    convergence.append({"eval": len(all_objs), "hypervolume": hv, "pareto_size": len(pf_idx)})
    print(f"  [eval={len(all_objs)}] PF={len(pf_idx)}, HV={hv:.4e}", file=sys.stderr)

    # --- Generations ---
    for gen in range(N_GEN):
        print(f"\n=== Generation {gen+1}/{N_GEN} ===", file=sys.stderr)
        pop_objs = [ind["objectives"] for ind in population]
        fronts = fast_non_dominated_sort(pop_objs)
        ranks = [0] * len(population)
        for rank, front in enumerate(fronts):
            for idx in front:
                ranks[idx] = rank
        cdists = [0.0] * len(population)
        for front in fronts:
            fobjs = [population[idx]["objectives"] for idx in front]
            cd = crowding_distance(fobjs)
            for k, idx in enumerate(front):
                cdists[idx] = cd[k]

        offspring = []
        attempts = 0
        while len(offspring) < POP_SIZE and attempts < POP_SIZE * 10:
            attempts += 1
            p1 = tournament_select(population, ranks, cdists, rng)
            p2 = tournament_select(population, ranks, cdists, rng)
            child = crossover(p1["config"], p2["config"], rng)
            child = mutate(child, rng)
            metrics = evaluate(child)
            if metrics is None:
                continue
            objs = objectives(metrics, child)
            offspring.append({"config": child, "metrics": metrics, "objectives": objs})
            all_configs.append(child)
            all_metrics.append(metrics)
            all_objs.append(objs)

        # Selection from combined
        combined = population + offspring
        comb_objs = [ind["objectives"] for ind in combined]
        fronts_c = fast_non_dominated_sort(comb_objs)
        ranks_c = [0] * len(combined)
        for rank, front in enumerate(fronts_c):
            for idx in front:
                ranks_c[idx] = rank
        cdists_c = [0.0] * len(combined)
        for front in fronts_c:
            fobjs = [combined[idx]["objectives"] for idx in front]
            cd = crowding_distance(fobjs)
            for k, idx in enumerate(front):
                cdists_c[idx] = cd[k]
        sorted_idx = sorted(range(len(combined)), key=lambda i: (ranks_c[i], -cdists_c[i]))
        population = [combined[i] for i in sorted_idx[:POP_SIZE]]

        pf_idx = get_pareto_front(all_objs)
        hv = hypervolume_mc([all_objs[k] for k in pf_idx], REF_4D)
        convergence.append({"eval": len(all_objs), "hypervolume": hv, "pareto_size": len(pf_idx)})
        print(f"  [eval={len(all_objs)}] PF={len(pf_idx)}, HV={hv:.4e}", file=sys.stderr)

    # --- Output ---
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
    summary = {
        "algorithm": "nsga2_4obj",
        "objectives": ["-rps", "ttft_p99", "gpu_count", "kv_blocks"],
        "reference_point": REF_4D,
        "total_evals": len(all_objs),
        "pareto_size": len(pareto_front),
        "final_hypervolume": convergence[-1]["hypervolume"],
        "wall_time_s": wall,
        "convergence": convergence,
        "pareto_front": pareto_front,
    }
    with open(os.path.join(RESULTS_DIR, "pareto_front.json"), "w") as f:
        json.dump(summary, f, indent=2)
    with open(os.path.join(RESULTS_DIR, "convergence.json"), "w") as f:
        json.dump(convergence, f, indent=2)

    print(f"\n=== Done: {len(all_objs)} evals in {wall:.1f}s, PF={len(pareto_front)}, HV={convergence[-1]['hypervolume']:.4e} ===", file=sys.stderr)


if __name__ == "__main__":
    main()
