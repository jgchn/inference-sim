"""Shared utilities for BLIS iter-6 experiments."""
import json
import os
import subprocess
import sys
import time

import numpy as np

# ── Parameter space ──────────────────────────────────────────────────────────
TP_VALUES = [1, 2, 4, 8]
SCHEDULER_VALUES = ['fcfs', 'sjf']
BATCH_VALUES = [32, 64, 128, 256, 512]
KV_VALUES = [2000, 3000, 4000, 5000, 7500, 10000]
BLOCK_SIZE_VALUES = [16, 32]

# 4-objective space: [-rps, ttft_p99_ms, gpu_count, kv_blocks] — all minimized
REF_POINT = np.array([0.0, 50000.0, 9.0, 11000.0])
HV_LO = np.array([-100.0, 0.0, 1.0, 2000.0])
MC_SAMPLES = 200_000
MC_SEED = 42

# ── Config enumeration ────────────────────────────────────────────────────────
def all_configs():
    """Return list of all 1800 configs as dicts."""
    configs = []
    for tp_idx, tp in enumerate(TP_VALUES):
        max_inst = 8 // tp
        for inst_idx in range(max_inst):
            num_instances = inst_idx + 1
            for sched_idx, scheduler in enumerate(SCHEDULER_VALUES):
                for batch_idx, batch in enumerate(BATCH_VALUES):
                    for kv_idx, kv in enumerate(KV_VALUES):
                        for bs_idx, bs in enumerate(BLOCK_SIZE_VALUES):
                            configs.append({
                                'tp_idx': tp_idx, 'tp': tp,
                                'inst_idx': inst_idx, 'num_instances': num_instances,
                                'sched_idx': sched_idx, 'scheduler': scheduler,
                                'batch_idx': batch_idx, 'max_num_running_reqs': batch,
                                'kv_idx': kv_idx, 'total_kv_blocks': kv,
                                'bs_idx': bs_idx, 'block_size_in_tokens': bs,
                            })
    return configs


def genes_to_config(genes):
    tp_idx, inst_idx, sched_idx, batch_idx, kv_idx, bs_idx = [int(g) for g in genes]
    tp = TP_VALUES[tp_idx]
    return {
        'tp_idx': tp_idx, 'tp': tp,
        'inst_idx': inst_idx, 'num_instances': inst_idx + 1,
        'sched_idx': sched_idx, 'scheduler': SCHEDULER_VALUES[sched_idx],
        'batch_idx': batch_idx, 'max_num_running_reqs': BATCH_VALUES[batch_idx],
        'kv_idx': kv_idx, 'total_kv_blocks': KV_VALUES[kv_idx],
        'bs_idx': bs_idx, 'block_size_in_tokens': BLOCK_SIZE_VALUES[bs_idx],
    }


def config_key(cfg):
    return (cfg['tp'], cfg['num_instances'], cfg['scheduler'],
            cfg['max_num_running_reqs'], cfg['total_kv_blocks'], cfg['block_size_in_tokens'])

# ── BLIS evaluation ───────────────────────────────────────────────────────────
_tmpdir = os.environ.get('TMPDIR', '/tmp')
_blis_path = os.path.join(os.getcwd(), 'blis')
_eval_counter = [0]
_eval_cache = {}


def eval_config(cfg, rate, blis_binary=None):
    """Evaluate a single config, return [obj0, obj1, obj2, obj3].

    Objectives (all minimized):
      obj0 = -responses_per_sec
      obj1 = ttft_p99_ms
      obj2 = gpu_count (tp * num_instances)
      obj3 = total_kv_blocks
    """
    key = (config_key(cfg), rate)
    if key in _eval_cache:
        return _eval_cache[key]

    _eval_counter[0] += 1
    eid = _eval_counter[0]
    mpath = os.path.join(_tmpdir, f'blis_iter6_{eid}_{os.getpid()}.json')

    binary = blis_binary or _blis_path
    cmd = [
        binary, 'run',
        '--model', 'qwen/qwen3-14b',
        '--hardware', 'H100',
        '--latency-model', 'trained-physics',
        '--num-requests', '200',
        '--rate', str(rate),
        '--prefix-tokens', '512',
        '--seed', '42',
        '--max-num-scheduled-tokens', '4096',
        '--long-prefill-token-threshold', '0',
        '--admission-policy', 'always-admit',
        '--preemption-policy', 'fcfs',
        '--routing-policy', 'round-robin',
        '--gpu-memory-utilization', '0.9',
        '--tp', str(cfg['tp']),
        '--num-instances', str(cfg['num_instances']),
        '--scheduler', cfg['scheduler'],
        '--max-num-running-reqs', str(cfg['max_num_running_reqs']),
        '--total-kv-blocks', str(cfg['total_kv_blocks']),
        '--block-size-in-tokens', str(cfg['block_size_in_tokens']),
        '--metrics-path', mpath,
    ]
    result = subprocess.run(cmd, capture_output=True, text=True)
    if result.returncode != 0:
        sys.exit(f"BLIS failed for config {config_key(cfg)}: {result.stderr[:300]}")

    with open(mpath) as f:
        metrics = json.load(f)

    rps = metrics['responses_per_sec']
    ttft = metrics['ttft_p99_ms']
    gpu_count = cfg['tp'] * cfg['num_instances']
    kv = cfg['total_kv_blocks']

    objs = [-rps, ttft, float(gpu_count), float(kv)]
    _eval_cache[key] = objs
    return objs

# ── Pareto dominance ──────────────────────────────────────────────────────────
def dominates(a, b):
    """True if a dominates b (all obj minimized). a, b are lists/arrays."""
    return all(a[i] <= b[i] for i in range(len(a))) and any(a[i] < b[i] for i in range(len(a)))


def pareto_front(objectives):
    """Return indices of non-dominated solutions in objectives list."""
    n = len(objectives)
    dominated_mask = [False] * n
    for i in range(n):
        if dominated_mask[i]:
            continue
        for j in range(n):
            if i == j or dominated_mask[j]:
                continue
            if dominates(objectives[j], objectives[i]):
                dominated_mask[i] = True
                break
    return [i for i in range(n) if not dominated_mask[i]]

# ── MC Hypervolume ────────────────────────────────────────────────────────────
def mc_hypervolume(pareto_objs_list, ref_point=None, hv_lo=None,
                   n_samples=MC_SAMPLES, seed=MC_SEED):
    """MC estimate of hypervolume. All objectives minimized."""
    if ref_point is None:
        ref_point = REF_POINT
    if hv_lo is None:
        hv_lo = HV_LO

    ref = np.array(ref_point, dtype=float)
    lo = np.array(hv_lo, dtype=float)
    box_vol = float(np.prod(ref - lo))

    if not pareto_objs_list:
        return 0.0

    rng = np.random.default_rng(seed)
    # Shape: (n_samples, n_obj)
    samples = rng.uniform(lo, ref, size=(n_samples, len(ref)))
    pareto = np.array(pareto_objs_list, dtype=float)  # (n_pareto, n_obj)

    # dominated[i] = any(all(pareto[k] <= samples[i]) for k)
    # broadcast: (n_pareto, n_obj) vs (n_samples, n_obj)
    # pareto[:, np.newaxis, :] = (n_pareto, 1, n_obj)
    # samples[np.newaxis, :, :] = (1, n_samples, n_obj)
    dominated = np.any(
        np.all(pareto[:, np.newaxis, :] <= samples[np.newaxis, :, :], axis=2),
        axis=0
    )
    return float(np.sum(dominated) / n_samples * box_vol)

# ── NSGA-II ───────────────────────────────────────────────────────────────────
def fast_nondominated_sort(objectives):
    """Returns list of fronts (each front = list of indices)."""
    n = len(objectives)
    S = [[] for _ in range(n)]  # S[i] = solutions that i dominates
    n_dom = [0] * n              # number of solutions dominating i
    fronts = [[]]

    for i in range(n):
        for j in range(n):
            if i == j:
                continue
            if dominates(objectives[i], objectives[j]):
                S[i].append(j)
            elif dominates(objectives[j], objectives[i]):
                n_dom[i] += 1
        if n_dom[i] == 0:
            fronts[0].append(i)

    k = 0
    while fronts[k]:
        next_front = []
        for i in fronts[k]:
            for j in S[i]:
                n_dom[j] -= 1
                if n_dom[j] == 0:
                    next_front.append(j)
        k += 1
        fronts.append(next_front)

    return [f for f in fronts if f]


def crowding_distance(objectives, front):
    n = len(front)
    if n <= 2:
        return {idx: float('inf') for idx in front}

    n_obj = len(objectives[0])
    dist = {idx: 0.0 for idx in front}

    for m in range(n_obj):
        sorted_front = sorted(front, key=lambda i: objectives[i][m])
        dist[sorted_front[0]] = float('inf')
        dist[sorted_front[-1]] = float('inf')
        f_min = objectives[sorted_front[0]][m]
        f_max = objectives[sorted_front[-1]][m]
        if f_max == f_min:
            continue
        span = f_max - f_min
        for k in range(1, n - 1):
            dist[sorted_front[k]] += (
                objectives[sorted_front[k + 1]][m] -
                objectives[sorted_front[k - 1]][m]
            ) / span

    return dist


def nsga2_selection(population_objectives, pop_size):
    """Select pop_size individuals from combined pool using NSGA-II rank+crowding."""
    n = len(population_objectives)
    fronts = fast_nondominated_sort(population_objectives)

    rank = {}
    for r, front in enumerate(fronts):
        for i in front:
            rank[i] = r

    crowding = {}
    for front in fronts:
        cd = crowding_distance(population_objectives, front)
        crowding.update(cd)

    selected = []
    for front in fronts:
        if len(selected) + len(front) <= pop_size:
            selected.extend(front)
        else:
            # Fill remaining spots by crowding distance (descending)
            remaining = pop_size - len(selected)
            sorted_front = sorted(front, key=lambda i: crowding.get(i, 0), reverse=True)
            selected.extend(sorted_front[:remaining])
            break

    return selected


def tournament_select_pair(population_objectives, rank, crowding, rng):
    """Binary tournament: returns index of winner."""
    a, b = rng.integers(0, len(population_objectives), size=2)
    ra, rb = rank[a], rank[b]
    if ra < rb:
        return int(a)
    elif rb < ra:
        return int(b)
    else:
        return int(a) if crowding.get(int(a), 0) >= crowding.get(int(b), 0) else int(b)


def repair_genes(genes):
    """Clamp genes to valid ranges and fix inst_idx for current tp."""
    g = list(genes)
    g[0] = int(np.clip(g[0], 0, 3))               # tp_idx
    max_inst = 8 // TP_VALUES[g[0]]
    g[1] = int(np.clip(g[1], 0, max_inst - 1))    # inst_idx
    g[2] = int(np.clip(g[2], 0, 1))               # sched_idx
    g[3] = int(np.clip(g[3], 0, 4))               # batch_idx
    g[4] = int(np.clip(g[4], 0, 5))               # kv_idx
    g[5] = int(np.clip(g[5], 0, 1))               # bs_idx
    return g


def crossover_mutate(p1, p2, rng, mutation_rate=0.15):
    """Uniform crossover + per-gene mutation. Returns repaired offspring."""
    child = []
    for i in range(6):
        child.append(p1[i] if rng.random() < 0.5 else p2[i])

    # Mutate
    for i in range(6):
        if rng.random() < mutation_rate:
            if i == 0:
                child[i] = int(rng.integers(0, 4))
            elif i == 1:
                # Use the (possibly mutated) tp_idx
                max_inst = 8 // TP_VALUES[int(np.clip(child[0], 0, 3))]
                child[i] = int(rng.integers(0, max_inst))
            elif i == 2:
                child[i] = int(rng.integers(0, 2))
            elif i == 3:
                child[i] = int(rng.integers(0, 5))
            elif i == 4:
                child[i] = int(rng.integers(0, 6))
            elif i == 5:
                child[i] = int(rng.integers(0, 2))

    return repair_genes(child)


def config_to_genes(cfg):
    return [cfg['tp_idx'], cfg['inst_idx'], cfg['sched_idx'],
            cfg['batch_idx'], cfg['kv_idx'], cfg['bs_idx']]


def run_nsga2(rate, pop_size=40, n_gens=4, mutation_rate=0.15, rng_seed=42,
              checkpoint_every=40):
    """Run NSGA-II. Returns (all_results, convergence_checkpoints).

    all_results: list of (cfg, objs) in evaluation order
    convergence_checkpoints: list of (eval_count, hv, pareto_size)
    """
    rng = np.random.default_rng(rng_seed)
    all_results = []  # (cfg, objs)
    checkpoints = []

    def record_checkpoint():
        all_objs = [r[1] for r in all_results]
        pf_idx = pareto_front(all_objs)
        pf_objs = [all_objs[i] for i in pf_idx]
        hv = mc_hypervolume(pf_objs)
        checkpoints.append({
            'eval': len(all_results),
            'hv': hv,
            'pareto_size': len(pf_idx),
        })
        print(f"  checkpoint eval={len(all_results)}: hv={hv:.4e}, pareto={len(pf_idx)}",
              flush=True)

    # Initial population: random genes
    population_genes = []
    seen_keys = {}

    def random_genes():
        tp_idx = int(rng.integers(0, 4))
        max_inst = 8 // TP_VALUES[tp_idx]
        return [tp_idx,
                int(rng.integers(0, max_inst)),
                int(rng.integers(0, 2)),
                int(rng.integers(0, 5)),
                int(rng.integers(0, 6)),
                int(rng.integers(0, 2))]

    # Evaluate initial population
    print(f"NSGA-II: evaluating initial pop ({pop_size} evals)...", flush=True)
    population_genes = [random_genes() for _ in range(pop_size)]
    population_objs = []
    for genes in population_genes:
        cfg = genes_to_config(genes)
        objs = eval_config(cfg, rate)
        all_results.append((cfg, objs))
        population_objs.append(objs)

    if len(all_results) % checkpoint_every == 0:
        record_checkpoint()

    # Generational loop
    for gen in range(n_gens):
        print(f"NSGA-II gen {gen+1}/{n_gens}...", flush=True)

        # Build rank + crowding for current population
        fronts = fast_nondominated_sort(population_objs)
        rank = {}
        for r, front in enumerate(fronts):
            for i in front:
                rank[i] = r
        crowding = {}
        for front in fronts:
            cd = crowding_distance(population_objs, front)
            crowding.update(cd)

        # Generate offspring
        offspring_genes = []
        offspring_objs = []
        while len(offspring_genes) < pop_size:
            p1_idx = tournament_select_pair(population_objs, rank, crowding, rng)
            p2_idx = tournament_select_pair(population_objs, rank, crowding, rng)
            child_genes = crossover_mutate(
                population_genes[p1_idx], population_genes[p2_idx],
                rng, mutation_rate
            )
            cfg = genes_to_config(child_genes)
            objs = eval_config(cfg, rate)
            offspring_genes.append(child_genes)
            offspring_objs.append(objs)
            all_results.append((cfg, objs))

            if len(all_results) % checkpoint_every == 0:
                record_checkpoint()

        # Combine + NSGA-II selection
        combined_genes = population_genes + offspring_genes
        combined_objs = population_objs + offspring_objs
        selected_idx = nsga2_selection(combined_objs, pop_size)
        population_genes = [combined_genes[i] for i in selected_idx]
        population_objs = [combined_objs[i] for i in selected_idx]

    # Final checkpoint if not already at a multiple of checkpoint_every
    if len(all_results) % checkpoint_every != 0:
        record_checkpoint()

    return all_results, checkpoints


def run_random(rate, budget=200, rng_seed=43, checkpoint_every=40):
    """Uniform random search. Returns (all_results, convergence_checkpoints)."""
    rng = np.random.default_rng(rng_seed)
    all_results = []
    checkpoints = []

    def record_checkpoint():
        all_objs = [r[1] for r in all_results]
        pf_idx = pareto_front(all_objs)
        pf_objs = [all_objs[i] for i in pf_idx]
        hv = mc_hypervolume(pf_objs)
        checkpoints.append({
            'eval': len(all_results),
            'hv': hv,
            'pareto_size': len(pf_idx),
        })
        print(f"  checkpoint eval={len(all_results)}: hv={hv:.4e}, pareto={len(pf_idx)}",
              flush=True)

    configs = all_configs()
    indices = rng.choice(len(configs), size=budget, replace=False)

    print(f"Random search: evaluating {budget} configs...", flush=True)
    for i, idx in enumerate(indices):
        cfg = configs[int(idx)]
        objs = eval_config(cfg, rate)
        all_results.append((cfg, objs))
        if len(all_results) % checkpoint_every == 0:
            record_checkpoint()

    if len(all_results) % checkpoint_every != 0:
        record_checkpoint()

    return all_results, checkpoints


def compute_exhaustive_hv(rate, checkpoint_every=40):
    """Run all 1800 configs and compute HV + convergence. Returns dict."""
    configs = all_configs()
    all_results = []
    checkpoints = []

    def record_checkpoint():
        all_objs = [r[1] for r in all_results]
        pf_idx = pareto_front(all_objs)
        pf_objs = [all_objs[i] for i in pf_idx]
        hv = mc_hypervolume(pf_objs)
        checkpoints.append({
            'eval': len(all_results),
            'hv': hv,
            'pareto_size': len(pf_idx),
        })
        print(f"  checkpoint eval={len(all_results)}: hv={hv:.4e}, pareto={len(pf_idx)}",
              flush=True)

    print(f"Exhaustive sweep at rate={rate}: {len(configs)} configs...", flush=True)
    t0 = time.time()
    for i, cfg in enumerate(configs):
        objs = eval_config(cfg, rate)
        all_results.append((cfg, objs))
        if (i + 1) % checkpoint_every == 0:
            record_checkpoint()
        if (i + 1) % 100 == 0:
            elapsed = time.time() - t0
            print(f"  {i+1}/{len(configs)} evals, {elapsed:.1f}s elapsed", flush=True)

    if len(all_results) % checkpoint_every != 0:
        record_checkpoint()

    # Final Pareto front
    all_objs = [r[1] for r in all_results]
    pf_idx = pareto_front(all_objs)
    pf_objs = [all_objs[i] for i in pf_idx]
    pf_configs = [all_results[i][0] for i in pf_idx]
    exhaustive_hv = mc_hypervolume(pf_objs)

    elapsed = time.time() - t0
    print(f"Exhaustive complete: {len(pf_idx)} Pareto configs, "
          f"density={100*len(pf_idx)/len(configs):.2f}%, "
          f"HV={exhaustive_hv:.4e}, {elapsed:.1f}s", flush=True)

    return {
        'count': len(configs),
        'pareto_count': len(pf_idx),
        'density_pct': 100.0 * len(pf_idx) / len(configs),
        'hypervolume': exhaustive_hv,
        'convergence': checkpoints,
        'pareto_configs': [
            {**cfg, 'objectives': objs}
            for cfg, objs in zip(pf_configs, pf_objs)
        ],
    }


def add_hv_pct(checkpoints, exhaustive_hv):
    """Add hv_pct field to each checkpoint."""
    for cp in checkpoints:
        cp['hv_pct'] = 100.0 * cp['hv'] / exhaustive_hv if exhaustive_hv > 0 else 0.0
    return checkpoints
