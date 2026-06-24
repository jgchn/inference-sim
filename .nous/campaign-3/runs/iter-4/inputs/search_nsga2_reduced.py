#!/usr/bin/env python3
"""Arm h-main: NSGA-II on reduced 5-parameter space. pop=40, 4 gens, 200 total evals."""
import json, os, random, subprocess, sys, time
import numpy as np

BLIS = os.environ.get('BLIS_BIN', os.path.join(os.getcwd(), 'blis'))
RESULTS_DIR = os.path.join(os.path.dirname(os.path.dirname(__file__)), 'results', 'h-main')
os.makedirs(RESULTS_DIR, exist_ok=True)

TP_VALUES = [1, 2, 4, 8]
SCHEDULER_VALUES = ['fcfs', 'sjf']
BATCH_VALUES = [32, 64, 128, 256, 512]
KV_VALUES = [2000, 3000, 4000, 5000, 7500, 10000]
REF_POINT = np.array([0.0, 50000.0, 9.0, 11000.0])
POP_SIZE = 40
N_GENS = 4
BUDGET = POP_SIZE + N_GENS * POP_SIZE  # 200
TRACK_EVERY = 40
SEED = 42
MUTATION_RATE = 0.15

FIXED_FLAGS = [
    '--model', 'qwen/qwen3-14b', '--hardware', 'H100',
    '--latency-model', 'trained-physics',
    '--num-requests', '200', '--rate', '50', '--prefix-tokens', '512', '--seed', '42',
    '--max-num-scheduled-tokens', '4096', '--long-prefill-token-threshold', '0',
    '--admission-policy', 'always-admit', '--preemption-policy', 'fcfs',
    '--block-size-in-tokens', '16', '--routing-policy', 'round-robin',
    '--gpu-memory-utilization', '0.9',
]

# Config encoded as (tp_idx, inst_idx, sched_idx, batch_idx, kv_idx)
# inst_idx is 0-based within the valid range for the current tp


def rand_config(rng):
    tp_idx = int(rng.integers(0, 4))
    tp = TP_VALUES[tp_idx]
    max_inst = 8 // tp
    inst_idx = int(rng.integers(0, max_inst))
    return (tp_idx, inst_idx, int(rng.integers(0, 2)), int(rng.integers(0, 5)), int(rng.integers(0, 6)))


def decode(gene):
    tp_idx, inst_idx, sched_idx, batch_idx, kv_idx = gene
    tp = TP_VALUES[tp_idx]
    inst = inst_idx + 1
    return {
        'tp': tp, 'num_instances': inst,
        'scheduler': SCHEDULER_VALUES[sched_idx],
        'max_num_running_reqs': BATCH_VALUES[batch_idx],
        'total_kv_blocks': KV_VALUES[kv_idx],
    }


def repair(gene):
    tp_idx, inst_idx, sched_idx, batch_idx, kv_idx = gene
    tp_idx = max(0, min(3, tp_idx))
    tp = TP_VALUES[tp_idx]
    max_inst = 8 // tp
    inst_idx = max(0, min(max_inst - 1, inst_idx))
    sched_idx = max(0, min(1, sched_idx))
    batch_idx = max(0, min(4, batch_idx))
    kv_idx = max(0, min(5, kv_idx))
    return (tp_idx, inst_idx, sched_idx, batch_idx, kv_idx)


def crossover(p1, p2, rng):
    child = tuple(p1[i] if rng.random() < 0.5 else p2[i] for i in range(5))
    return repair(child)


def mutate(gene, rng):
    tp_idx, inst_idx, sched_idx, batch_idx, kv_idx = gene
    if rng.random() < MUTATION_RATE:
        tp_idx = int(rng.integers(0, 4))
    tp = TP_VALUES[tp_idx]
    max_inst = 8 // tp
    if rng.random() < MUTATION_RATE:
        inst_idx = int(rng.integers(0, max_inst))
    if rng.random() < MUTATION_RATE:
        sched_idx = int(rng.integers(0, 2))
    if rng.random() < MUTATION_RATE:
        batch_idx = int(rng.integers(0, 5))
    if rng.random() < MUTATION_RATE:
        kv_idx = int(rng.integers(0, 6))
    return repair((tp_idx, inst_idx, sched_idx, batch_idx, kv_idx))


def run_config(cfg, idx):
    mp = os.path.join(os.environ['TMPDIR'], f'metrics_nsga2r_{idx}.json')
    cmd = [BLIS, 'run'] + FIXED_FLAGS + [
        '--tp', str(cfg['tp']),
        '--num-instances', str(cfg['num_instances']),
        '--scheduler', cfg['scheduler'],
        '--max-num-running-reqs', str(cfg['max_num_running_reqs']),
        '--total-kv-blocks', str(cfg['total_kv_blocks']),
        '--metrics-path', mp,
    ]
    r = subprocess.run(cmd, capture_output=True, timeout=60)
    if r.returncode != 0:
        return None
    with open(mp) as f:
        data = json.load(f)
    if isinstance(data, list):
        for rec in data:
            if isinstance(rec, dict) and rec.get('instance_id') == 'cluster':
                return rec
        for rec in data:
            if isinstance(rec, dict) and 'responses_per_sec' in rec:
                return rec
    return data if isinstance(data, dict) else None


def dominates(a, b):
    return all(ai <= bi for ai, bi in zip(a, b)) and any(ai < bi for ai, bi in zip(a, b))


def fast_nondominated_sort(objs):
    n = len(objs)
    dominated_by = [[] for _ in range(n)]
    n_dom = [0] * n
    ranks = [0] * n
    fronts = [[]]
    for i in range(n):
        for j in range(n):
            if i == j:
                continue
            if dominates(objs[i], objs[j]):
                dominated_by[i].append(j)
            elif dominates(objs[j], objs[i]):
                n_dom[i] += 1
        if n_dom[i] == 0:
            ranks[i] = 0
            fronts[0].append(i)
    k = 0
    while fronts[k]:
        nxt = []
        for i in fronts[k]:
            for j in dominated_by[i]:
                n_dom[j] -= 1
                if n_dom[j] == 0:
                    ranks[j] = k + 1
                    nxt.append(j)
        k += 1
        fronts.append(nxt)
    return ranks, fronts[:-1]


def crowding_distance(objs, front):
    if len(front) <= 2:
        return {i: float('inf') for i in front}
    n_obj = len(objs[0])
    dist = {i: 0.0 for i in front}
    for m in range(n_obj):
        sf = sorted(front, key=lambda i: objs[i][m])
        obj_min = objs[sf[0]][m]
        obj_max = objs[sf[-1]][m]
        dist[sf[0]] = float('inf')
        dist[sf[-1]] = float('inf')
        if obj_max == obj_min:
            continue
        for k in range(1, len(sf) - 1):
            dist[sf[k]] += (objs[sf[k+1]][m] - objs[sf[k-1]][m]) / (obj_max - obj_min)
    return dist


def tournament_select(pop, ranks, crowding, rng):
    i, j = int(rng.integers(0, len(pop))), int(rng.integers(0, len(pop)))
    ri, rj = ranks[i], ranks[j]
    ci, cj = crowding.get(i, 0.0), crowding.get(j, 0.0)
    if ri < rj or (ri == rj and ci > cj):
        return pop[i]
    return pop[j]


def compute_hv(pareto_objs, ref=REF_POINT, n_samples=500000, seed=42):
    pts = np.array(pareto_objs)
    if len(pts) == 0:
        return 0.0
    lb = pts.min(axis=0)
    ub = ref
    if np.any(lb >= ub):
        return 0.0
    rng = np.random.default_rng(seed)
    samples = rng.uniform(lb, ub, (n_samples, len(ref)))
    dominated = np.zeros(n_samples, dtype=bool)
    for p in pts:
        dominated |= np.all(samples >= p, axis=1)
    return float(np.prod(ub - lb) * dominated.mean())


def pareto_indices(objs):
    nd = []
    for i, a in enumerate(objs):
        if not any(dominates(b, a) for j, b in enumerate(objs) if i != j):
            nd.append(i)
    return nd


def main():
    rng = np.random.default_rng(SEED)
    eval_count = 0
    all_genes = []
    all_objs = []
    convergence = []
    t0 = time.time()

    # Initialize population
    print(f'Initializing population of {POP_SIZE}...', flush=True)
    pop = [rand_config(rng) for _ in range(POP_SIZE)]
    pop_objs = []
    for i, gene in enumerate(pop):
        cfg = decode(gene)
        metrics = run_config(cfg, eval_count)
        eval_count += 1
        if metrics is None:
            # fallback: use worst objective
            obj = [0.0, 50000.0, 9.0, 11000.0]
        else:
            gpu_count = cfg['tp'] * cfg['num_instances']
            obj = [-metrics['responses_per_sec'], metrics['ttft_p99_ms'], gpu_count, cfg['total_kv_blocks']]
        pop_objs.append(obj)
        all_genes.append(gene)
        all_objs.append(obj)

        if eval_count % TRACK_EVERY == 0:
            pf_idx = pareto_indices(all_objs)
            hv = compute_hv([all_objs[j] for j in pf_idx])
            convergence.append({'eval': eval_count, 'hv': hv, 'pareto_size': len(pf_idx)})
            print(f'  eval={eval_count}: HV={hv:.4e}, pareto_size={len(pf_idx)}', flush=True)

    # NSGA-II generations
    for gen in range(N_GENS):
        print(f'Generation {gen+1}/{N_GENS}...', flush=True)
        # Sort current population
        ranks, fronts = fast_nondominated_sort(pop_objs)
        crowding = {}
        for front in fronts:
            cd = crowding_distance(pop_objs, front)
            crowding.update(cd)

        # Generate offspring
        offspring = []
        offspring_objs = []
        for _ in range(POP_SIZE):
            p1 = tournament_select(pop, ranks, crowding, rng)
            p2 = tournament_select(pop, ranks, crowding, rng)
            child = crossover(p1, p2, rng)
            child = mutate(child, rng)
            cfg = decode(child)
            metrics = run_config(cfg, eval_count)
            eval_count += 1
            if metrics is None:
                obj = [0.0, 50000.0, 9.0, 11000.0]
            else:
                gpu_count = cfg['tp'] * cfg['num_instances']
                obj = [-metrics['responses_per_sec'], metrics['ttft_p99_ms'], gpu_count, cfg['total_kv_blocks']]
            offspring.append(child)
            offspring_objs.append(obj)
            all_genes.append(child)
            all_objs.append(obj)

            if eval_count % TRACK_EVERY == 0:
                pf_idx = pareto_indices(all_objs)
                hv = compute_hv([all_objs[j] for j in pf_idx])
                convergence.append({'eval': eval_count, 'hv': hv, 'pareto_size': len(pf_idx)})
                print(f'  eval={eval_count}: HV={hv:.4e}, pareto_size={len(pf_idx)}', flush=True)

        # Combine and select next generation (NSGA-II selection)
        combined = pop + offspring
        combined_objs = pop_objs + offspring_objs
        c_ranks, c_fronts = fast_nondominated_sort(combined_objs)
        c_crowding = {}
        for front in c_fronts:
            cd = crowding_distance(combined_objs, front)
            c_crowding.update(cd)

        # Select top POP_SIZE by rank then crowding
        selected = []
        for front in c_fronts:
            if len(selected) + len(front) <= POP_SIZE:
                selected.extend(front)
            else:
                needed = POP_SIZE - len(selected)
                sorted_front = sorted(front, key=lambda i: -c_crowding.get(i, 0.0))
                selected.extend(sorted_front[:needed])
                break

        pop = [combined[i] for i in selected]
        pop_objs = [combined_objs[i] for i in selected]

    # Final
    pf_idx = pareto_indices(all_objs)
    pf_objs = [all_objs[j] for j in pf_idx]
    final_hv = compute_hv(pf_objs)
    print(f'Final HV={final_hv:.4e}, pareto_size={len(pf_idx)}, time={time.time()-t0:.1f}s', flush=True)

    pf_out = [{'config': decode(all_genes[j]), 'objectives': {
        'neg_rps': all_objs[j][0], 'ttft_p99_ms': all_objs[j][1],
        'gpu_count': all_objs[j][2], 'total_kv_blocks': all_objs[j][3]
    }} for j in pf_idx]

    with open(os.path.join(RESULTS_DIR, 'pareto_front.json'), 'w') as f:
        json.dump({'pareto_front': pf_out, 'final_hv': final_hv,
                   'total_evals': eval_count, 'pareto_size': len(pf_idx)}, f, indent=2)
    with open(os.path.join(RESULTS_DIR, 'convergence.json'), 'w') as f:
        json.dump(convergence, f, indent=2)
    print('Done.', flush=True)


if __name__ == '__main__':
    main()
