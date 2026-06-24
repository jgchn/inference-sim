"""
h-main: NSGA-II evolutionary search over BLIS configuration space.
Population size=50, 10 generations, 500 total evaluations.
Discrete crossover/mutation operators, constraint-aware offspring generation.
"""

import json
import random
import sys
import time
from copy import deepcopy
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))
from search_common import (
    PARAMS, random_valid_config, evaluate, pareto_front, hypervolume_2d,
    config_key, is_valid_config, HV_REF
)

RESULTS_DIR = Path(__file__).parent.parent / "results" / "h-main"
POP_SIZE = 50
N_GENERATIONS = 10
SEED = 42000
MUTATION_RATE = 0.2


def non_dominated_sort(pop_with_obj):
    """
    Sort population into Pareto fronts (NSGA-II non-dominated sorting).
    pop_with_obj: list of (cfg, (rps, ttft_p99))
    Returns list of fronts (each front is list of indices).
    """
    n = len(pop_with_obj)
    dominated_by = [0] * n  # domination count
    dominates = [[] for _ in range(n)]  # whom i dominates
    fronts = [[]]

    for i in range(n):
        for j in range(n):
            if i == j:
                continue
            ri, ti = pop_with_obj[i][1]
            rj, tj = pop_with_obj[j][1]
            # j dominates i: j >= i on all objectives and strictly better on one
            # maximize rps, minimize ttft
            if rj >= ri and tj <= ti and (rj > ri or tj < ti):
                dominated_by[i] += 1
            elif ri >= rj and ti <= tj and (ri > rj or ti < tj):
                dominates[i].append(j)

        if dominated_by[i] == 0:
            fronts[0].append(i)

    i = 0
    while fronts[i]:
        next_front = []
        for p in fronts[i]:
            for q in dominates[p]:
                dominated_by[q] -= 1
                if dominated_by[q] == 0:
                    next_front.append(q)
        i += 1
        fronts.append(next_front)

    return [f for f in fronts if f]


def crowding_distance(front_indices, pop_with_obj):
    """Compute crowding distance for individuals in a front."""
    n = len(front_indices)
    if n <= 2:
        return {idx: float('inf') for idx in front_indices}

    distances = {idx: 0.0 for idx in front_indices}
    objectives = [(pop_with_obj[idx][1][0], pop_with_obj[idx][1][1]) for idx in front_indices]

    for obj_i in range(2):
        # Sort by this objective
        sorted_front = sorted(range(n), key=lambda k: objectives[k][obj_i])
        obj_vals = [objectives[sorted_front[k]][obj_i] for k in range(n)]
        obj_range = obj_vals[-1] - obj_vals[0]
        if obj_range == 0:
            continue

        distances[front_indices[sorted_front[0]]] = float('inf')
        distances[front_indices[sorted_front[-1]]] = float('inf')

        for k in range(1, n - 1):
            distances[front_indices[sorted_front[k]]] += (
                (obj_vals[k + 1] - obj_vals[k - 1]) / obj_range
            )

    return distances


def tournament_select(pop_with_obj, fronts, crowd_dist, rng):
    """Binary tournament selection by rank then crowding distance."""
    n = len(pop_with_obj)
    # Build rank and distance maps
    rank = {}
    for r, front in enumerate(fronts):
        for idx in front:
            rank[idx] = r

    candidates = [rng.randrange(n), rng.randrange(n)]
    a, b = candidates
    if rank[a] < rank[b]:
        return pop_with_obj[a][0]
    elif rank[b] < rank[a]:
        return pop_with_obj[b][0]
    else:
        if crowd_dist.get(a, 0) >= crowd_dist.get(b, 0):
            return pop_with_obj[a][0]
        else:
            return pop_with_obj[b][0]


def crossover(p1, p2, rng):
    """Uniform discrete crossover between two parent configs."""
    child = {}
    for key in p1:
        child[key] = p1[key] if rng.random() < 0.5 else p2[key]
    return child


def mutate(cfg, rng):
    """Randomly mutate one parameter of the config."""
    child = deepcopy(cfg)
    param = rng.choice(list(PARAMS.keys()))

    if param == "tp":
        child["tp"] = rng.choice(PARAMS["tp"])
    elif param == "num_instances":
        max_n = 8 // child["tp"]
        child["num_instances"] = rng.randint(1, max_n)
    elif param == "scheduler":
        child["scheduler"] = rng.choice(PARAMS["scheduler"])
    elif param == "max_num_running_reqs":
        child["max_num_running_reqs"] = rng.choice(PARAMS["max_num_running_reqs"])
    elif param == "max_num_scheduled_tokens":
        child["max_num_scheduled_tokens"] = rng.choice(PARAMS["max_num_scheduled_tokens"])
    elif param == "long_prefill_token_threshold":
        valid = [v for v in PARAMS["long_prefill_token_threshold"]
                 if v == 0 or v < child["max_num_scheduled_tokens"]]
        child["long_prefill_token_threshold"] = rng.choice(valid)
    elif param == "block_size_in_tokens":
        child["block_size_in_tokens"] = rng.choice(PARAMS["block_size_in_tokens"])
    elif param == "routing_policy":
        child["routing_policy"] = rng.choice(PARAMS["routing_policy"])
    elif param == "routing_scorer_config":
        if child["routing_policy"] == "weighted":
            child["routing_scorer_config"] = rng.choice(PARAMS["routing_scorer_config"])
    elif param == "admission_policy":
        child["admission_policy"] = rng.choice(PARAMS["admission_policy"])
    elif param == "preemption_policy":
        child["preemption_policy"] = rng.choice(PARAMS["preemption_policy"])
    elif param == "gpu_memory_utilization":
        child["gpu_memory_utilization"] = rng.choice(PARAMS["gpu_memory_utilization"])

    return child


def repair(cfg, rng):
    """Fix constraint violations in a config after crossover/mutation."""
    # TP * instances <= 8
    max_n = 8 // cfg["tp"]
    if cfg["num_instances"] > max_n:
        cfg["num_instances"] = rng.randint(1, max_n)

    # long_prefill constraint
    if cfg["long_prefill_token_threshold"] != 0:
        if cfg["long_prefill_token_threshold"] >= cfg["max_num_scheduled_tokens"]:
            valid = [v for v in PARAMS["long_prefill_token_threshold"]
                     if v == 0 or v < cfg["max_num_scheduled_tokens"]]
            cfg["long_prefill_token_threshold"] = rng.choice(valid)

    # scorer config only valid for weighted routing
    if cfg["routing_policy"] != "weighted":
        cfg["routing_scorer_config"] = None

    return cfg


def run():
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    rng = random.Random(SEED)
    t_start = time.time()
    total_evals = 0
    generation_hv = []

    print(f"[nsga2] Starting NSGA-II: pop={POP_SIZE}, gen={N_GENERATIONS}", flush=True)

    # Initialize population
    print(f"[nsga2] Initializing population...", flush=True)
    population = []  # list of (cfg, (rps, ttft_p99))
    seen = set()
    while len(population) < POP_SIZE:
        cfg = random_valid_config(rng)
        key = config_key(cfg)
        if key in seen:
            continue
        seen.add(key)
        obj = evaluate(cfg)
        total_evals += 1
        if obj is None:
            continue
        population.append((cfg, obj))

    print(f"[nsga2] Population initialized ({total_evals} evals)", flush=True)

    # Record gen 0 hypervolume
    front0 = pareto_front([p[1] for p in population])
    hv0 = hypervolume_2d(front0, HV_REF)
    generation_hv.append({"generation": 0, "evals": total_evals, "hv": hv0, "front_size": len(front0)})
    print(f"[nsga2] Gen 0: hv={hv0:.4f}, front_size={len(front0)}", flush=True)

    for gen in range(1, N_GENERATIONS + 1):
        # Build fronts and crowding distances for selection
        fronts = non_dominated_sort(population)
        crowd_dist = {}
        for front in fronts:
            cd = crowding_distance(front, population)
            crowd_dist.update(cd)

        # Generate offspring
        offspring = []
        attempts = 0
        while len(offspring) < POP_SIZE and attempts < POP_SIZE * 20:
            attempts += 1
            p1 = tournament_select(population, fronts, crowd_dist, rng)
            p2 = tournament_select(population, fronts, crowd_dist, rng)

            child = crossover(p1, p2, rng)
            if rng.random() < MUTATION_RATE:
                child = mutate(child, rng)
            child = repair(child, rng)

            if not is_valid_config(child):
                continue
            key = config_key(child)
            if key in seen:
                continue
            seen.add(key)

            obj = evaluate(child)
            total_evals += 1
            if obj is None:
                continue
            offspring.append((child, obj))

        # Combine parent + offspring, select next generation
        combined = population + offspring
        combined_fronts = non_dominated_sort(combined)

        next_pop = []
        for front in combined_fronts:
            if len(next_pop) + len(front) <= POP_SIZE:
                for idx in front:
                    next_pop.append(combined[idx])
            else:
                # Fill remaining with highest crowding distance
                cd = crowding_distance(front, combined)
                remaining = POP_SIZE - len(next_pop)
                sorted_front = sorted(front, key=lambda i: cd.get(i, 0), reverse=True)
                for idx in sorted_front[:remaining]:
                    next_pop.append(combined[idx])
                break

        population = next_pop

        front_pts = pareto_front([p[1] for p in population])
        hv = hypervolume_2d(front_pts, HV_REF)
        generation_hv.append({"generation": gen, "evals": total_evals, "hv": hv, "front_size": len(front_pts)})
        print(f"[nsga2] Gen {gen}: hv={hv:.4f}, front_size={len(front_pts)}, total_evals={total_evals}", flush=True)

    elapsed = time.time() - t_start

    # Final Pareto front
    all_points = [p[1] for p in population]
    final_front = pareto_front(all_points)
    final_hv = hypervolume_2d(final_front, HV_REF)

    summary = {
        "arm": "h-main",
        "method": "nsga2",
        "pop_size": POP_SIZE,
        "n_generations": N_GENERATIONS,
        "total_evals": total_evals,
        "pareto_front_size": len(final_front),
        "hypervolume": final_hv,
        "hypervolume_ref": list(HV_REF),
        "elapsed_s": elapsed,
        "generation_hypervolume": generation_hv,
        "pareto_front": sorted(final_front, key=lambda p: p[0]),
        "all_points": [[p[1][0], p[1][1]] for p in population],
        "best_rps": max(p[1][0] for p in population) if population else 0,
        "best_ttft": min(p[1][1] for p in population) if population else 0,
    }

    out_path = RESULTS_DIR / "summary.json"
    with open(out_path, "w") as f:
        json.dump(summary, f, indent=2)

    print(f"\n[nsga2] Done. {total_evals} evals in {elapsed:.1f}s")
    print(f"[nsga2] Final Pareto front: {len(final_front)} points, hypervolume={final_hv:.4f}")
    print(f"[nsga2] Best rps={summary['best_rps']:.2f}, best ttft_p99={summary['best_ttft']:.1f}ms")
    print(f"[nsga2] Generation HV progression: {[g['hv'] for g in generation_hv]}")
    print(f"[nsga2] Results written to {out_path}")
    return summary


if __name__ == "__main__":
    run()
