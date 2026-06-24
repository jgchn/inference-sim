"""Shared utilities for BLIS iter-7 experiments (model-portability study)."""
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

# Model flag sets
FIXED_FLAGS_LLAMA = [
    '--model', 'meta-llama/llama-3.1-8b-instruct',
    '--hardware', 'H100',
    '--latency-model', 'trained-physics',
    '--num-requests', '200',
    '--rate', '50',
    '--prefix-tokens', '512',
    '--seed', '42',
    '--max-num-scheduled-tokens', '4096',
    '--long-prefill-token-threshold', '0',
    '--admission-policy', 'always-admit',
    '--preemption-policy', 'fcfs',
    '--routing-policy', 'round-robin',
    '--gpu-memory-utilization', '0.9',
]

FIXED_FLAGS_QWEN = [
    '--model', 'qwen/qwen3-14b',
    '--hardware', 'H100',
    '--latency-model', 'trained-physics',
    '--num-requests', '200',
    '--rate', '50',
    '--prefix-tokens', '512',
    '--seed', '42',
    '--max-num-scheduled-tokens', '4096',
    '--long-prefill-token-threshold', '0',
    '--admission-policy', 'always-admit',
    '--preemption-policy', 'fcfs',
    '--routing-policy', 'round-robin',
    '--gpu-memory-utilization', '0.9',
]

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


def eval_config(cfg, fixed_flags, cache_key_prefix=''):
    """Evaluate a single config, return [obj0, obj1, obj2, obj3].

    Objectives (all minimized):
      obj0 = -responses_per_sec
      obj1 = ttft_p99_ms
      obj2 = gpu_count (tp * num_instances)
      obj3 = total_kv_blocks
    """
    key = (cache_key_prefix, config_key(cfg))
    if key in _eval_cache:
        return _eval_cache[key]

    _eval_counter[0] += 1
    eid = _eval_counter[0]
    mpath = os.path.join(_tmpdir, f'blis_iter7_{eid}_{os.getpid()}.json')

    cmd = [
        _blis_path, 'run',
    ] + fixed_flags + [
        '--tp', str(cfg['tp']),
        '--num-instances', str(cfg['num_instances']),
        '--scheduler', cfg['scheduler'],
        '--max-num-running-reqs', str(cfg['max_num_running_reqs']),
        '--total-kv-blocks', str(cfg['total_kv_blocks']),
        '--block-size-in-tokens', str(cfg['block_size_in_tokens']),
        '--metrics-path', mpath,
    ]
    result = subprocess.run(cmd, capture_output=True, text=True, timeout=120)
    if result.returncode != 0:
        sys.exit(f"BLIS failed for config {config_key(cfg)}: {result.stderr[:300]}")

    with open(mpath) as f:
        metrics = json.load(f)
    os.unlink(mpath)

    rps = metrics['responses_per_sec']
    ttft = metrics['ttft_p99_ms']
    gpu_count = cfg['tp'] * cfg['num_instances']
    kv = cfg['total_kv_blocks']

    objs = [-rps, ttft, float(gpu_count), float(kv)]
    _eval_cache[key] = objs
    return objs


# ── Pareto dominance ──────────────────────────────────────────────────────────
def dominates(a, b):
    """True if a dominates b (all obj minimized)."""
    return all(a[i] <= b[i] for i in range(len(a))) and any(a[i] < b[i] for i in range(len(a)))


def pareto_front(objectives):
    """Return indices of non-dominated solutions."""
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


def pareto_update(nd_list, new_pt):
    """Update running non-dominated set with a new point."""
    for p in nd_list:
        if all(p[k] <= new_pt[k] for k in range(len(p))) and any(p[k] < new_pt[k] for k in range(len(p))):
            return nd_list
    return [p for p in nd_list if not (
        all(new_pt[k] <= p[k] for k in range(len(p))) and any(new_pt[k] < p[k] for k in range(len(p)))
    )] + [new_pt]


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
    samples = rng.uniform(lo, ref, size=(n_samples, len(ref)))
    pareto = np.array(pareto_objs_list, dtype=float)  # (n_pareto, n_obj)

    dominated = np.any(
        np.all(pareto[:, np.newaxis, :] <= samples[np.newaxis, :, :], axis=2),
        axis=0
    )
    return float(np.sum(dominated) / n_samples * box_vol)


# ── NSGA-II helpers ───────────────────────────────────────────────────────────
def fast_nondominated_sort(objectives):
    """Returns list of fronts (each front = list of indices)."""
    n = len(objectives)
    S = [[] for _ in range(n)]
    n_dom = [0] * n
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
        for k2 in range(1, n - 1):
            dist[sorted_front[k2]] += (
                objectives[sorted_front[k2 + 1]][m] -
                objectives[sorted_front[k2 - 1]][m]
            ) / span

    return dist


def nsga2_select(combined_genes, combined_objs, pop_size):
    """Select pop_size individuals from combined pool using NSGA-II rank+crowding."""
    fronts = fast_nondominated_sort(combined_objs)
    rank = {}
    for r, front in enumerate(fronts):
        for i in front:
            rank[i] = r
    cd = {}
    for front in fronts:
        cd.update(crowding_distance(combined_objs, front))
    order = sorted(range(len(combined_genes)), key=lambda i: (rank[i], -cd.get(i, 0)))
    sel = order[:pop_size]
    return [combined_genes[i] for i in sel], [combined_objs[i] for i in sel]


def tournament_select_one(pop_objs, rank, cd, rng):
    """Binary tournament selection, returns index of winner."""
    a = int(rng.integers(0, len(pop_objs)))
    b = int(rng.integers(0, len(pop_objs)))
    if rank[a] < rank[b]:
        return a
    elif rank[b] < rank[a]:
        return b
    else:
        return a if cd.get(a, 0) >= cd.get(b, 0) else b


def repair_genes(genes):
    """Clamp genes to valid ranges and fix inst_idx for current tp."""
    g = list(genes)
    g[0] = int(np.clip(g[0], 0, 3))
    max_inst = 8 // TP_VALUES[g[0]]
    g[1] = int(np.clip(g[1], 0, max_inst - 1))
    g[2] = int(np.clip(g[2], 0, 1))
    g[3] = int(np.clip(g[3], 0, 4))
    g[4] = int(np.clip(g[4], 0, 5))
    g[5] = int(np.clip(g[5], 0, 1))
    return g


def crossover_mutate(p1, p2, rng, mutation_rate=0.15):
    """Uniform crossover + per-gene mutation. Returns repaired offspring."""
    child = [p1[i] if rng.random() < 0.5 else p2[i] for i in range(6)]
    # Mutate
    if rng.random() < mutation_rate:
        child[0] = int(rng.integers(0, 4))
    tp = TP_VALUES[int(np.clip(child[0], 0, 3))]
    if rng.random() < mutation_rate:
        child[1] = int(rng.integers(0, 8 // tp))
    if rng.random() < mutation_rate:
        child[2] = int(rng.integers(0, 2))
    if rng.random() < mutation_rate:
        child[3] = int(rng.integers(0, 5))
    if rng.random() < mutation_rate:
        child[4] = int(rng.integers(0, 6))
    if rng.random() < mutation_rate:
        child[5] = int(rng.integers(0, 2))
    return repair_genes(child)


def random_genes(rng):
    tp_idx = int(rng.integers(0, 4))
    tp = TP_VALUES[tp_idx]
    return [tp_idx,
            int(rng.integers(0, 8 // tp)),
            int(rng.integers(0, 2)),
            int(rng.integers(0, 5)),
            int(rng.integers(0, 6)),
            int(rng.integers(0, 2))]


# ── NSGA-II runner ────────────────────────────────────────────────────────────
def run_nsga2(fixed_flags, pop_size=40, n_gens=4, mutation_rate=0.15, rng_seed=42,
              checkpoint_every=40, exhaustive_hv=None, cache_key_prefix=''):
    """Run NSGA-II. Returns (current_nd, checkpoints).

    checkpoints: list of dicts with eval, hv, pareto_size, pct_of_exhaustive
    """
    rng = np.random.default_rng(rng_seed)
    all_results = []
    current_nd = []
    checkpoints = []
    t0 = time.time()

    def record_checkpoint():
        hv = mc_hypervolume(current_nd)
        pct = (100.0 * hv / exhaustive_hv) if exhaustive_hv else None
        entry = {
            'eval': len(all_results),
            'hv': hv,
            'pareto_size': len(current_nd),
            'pct_of_exhaustive': round(pct, 2) if pct is not None else None,
            'elapsed_s': round(time.time() - t0, 1),
        }
        checkpoints.append(entry)
        pct_str = f' ({pct:.1f}%)' if pct is not None else ''
        print(f"  eval={len(all_results)}: hv={hv:.4e}, pareto={len(current_nd)}{pct_str}",
              flush=True)

    # Evaluate initial population
    print(f"NSGA-II (seed={rng_seed}): initializing pop={pop_size}...", flush=True)
    pop_genes = [random_genes(rng) for _ in range(pop_size)]
    pop_objs = []
    for genes in pop_genes:
        cfg = genes_to_config(genes)
        objs = eval_config(cfg, fixed_flags, cache_key_prefix)
        pop_objs.append(objs)
        current_nd = pareto_update(current_nd, objs)
        all_results.append((cfg, objs))
        if len(all_results) % checkpoint_every == 0:
            record_checkpoint()

    # Generational loop
    for gen in range(n_gens):
        print(f"  gen {gen + 1}/{n_gens}...", flush=True)

        # Build rank + crowding for current population
        fronts = fast_nondominated_sort(pop_objs)
        rank = {}
        for r, front in enumerate(fronts):
            for i in front:
                rank[i] = r
        cd = {}
        for front in fronts:
            cd.update(crowding_distance(pop_objs, front))

        # Generate offspring
        offspring_genes = []
        offspring_objs = []
        while len(offspring_genes) < pop_size:
            p1_idx = tournament_select_one(pop_objs, rank, cd, rng)
            p2_idx = tournament_select_one(pop_objs, rank, cd, rng)
            child_genes = crossover_mutate(pop_genes[p1_idx], pop_genes[p2_idx], rng, mutation_rate)
            cfg = genes_to_config(child_genes)
            objs = eval_config(cfg, fixed_flags, cache_key_prefix)
            offspring_genes.append(child_genes)
            offspring_objs.append(objs)
            current_nd = pareto_update(current_nd, objs)
            all_results.append((cfg, objs))
            if len(all_results) % checkpoint_every == 0:
                record_checkpoint()

        # NSGA-II survivor selection
        pop_genes, pop_objs = nsga2_select(pop_genes + offspring_genes,
                                           pop_objs + offspring_objs, pop_size)

    # Final checkpoint if not already at a multiple
    if len(all_results) % checkpoint_every != 0:
        record_checkpoint()

    return current_nd, checkpoints


# ── Random search runner ──────────────────────────────────────────────────────
def run_random(fixed_flags, budget=200, rng_seed=43, checkpoint_every=40,
               exhaustive_hv=None, cache_key_prefix=''):
    """Uniform random sampling. Returns (current_nd, checkpoints)."""
    rng = np.random.default_rng(rng_seed)
    configs = all_configs()
    indices = rng.choice(len(configs), size=budget, replace=False)

    all_results = []
    current_nd = []
    checkpoints = []
    t0 = time.time()

    def record_checkpoint():
        hv = mc_hypervolume(current_nd)
        pct = (100.0 * hv / exhaustive_hv) if exhaustive_hv else None
        entry = {
            'eval': len(all_results),
            'hv': hv,
            'pareto_size': len(current_nd),
            'pct_of_exhaustive': round(pct, 2) if pct is not None else None,
            'elapsed_s': round(time.time() - t0, 1),
        }
        checkpoints.append(entry)
        pct_str = f' ({pct:.1f}%)' if pct is not None else ''
        print(f"  eval={len(all_results)}: hv={hv:.4e}, pareto={len(current_nd)}{pct_str}",
              flush=True)

    print(f"Random search (seed={rng_seed}): {budget} evals...", flush=True)
    for idx in indices:
        cfg = configs[int(idx)]
        objs = eval_config(cfg, fixed_flags, cache_key_prefix)
        current_nd = pareto_update(current_nd, objs)
        all_results.append((cfg, objs))
        if len(all_results) % checkpoint_every == 0:
            record_checkpoint()

    if len(all_results) % checkpoint_every != 0:
        record_checkpoint()

    return current_nd, checkpoints


def add_hv_pct(checkpoints, exhaustive_hv):
    """Add/update hv_pct field in each checkpoint."""
    for cp in checkpoints:
        cp['hv_pct'] = 100.0 * cp['hv'] / exhaustive_hv if exhaustive_hv > 0 else 0.0
    return checkpoints
