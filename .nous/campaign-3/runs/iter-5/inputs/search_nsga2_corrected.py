#!/usr/bin/env python3
"""h-main: NSGA-II on 1800-config corrected space. pop=40, 4 gens, 200 evals."""

import subprocess, json, os, sys, time
import numpy as np

BLIS = '/Users/jchen/go/src/inference-sim/inference-sim/.nous-experiments/iter-5-9ae74af5/blis'
RESULTS_DIR = '/Users/jchen/go/src/inference-sim/inference-sim/.nous/campaign-3/runs/iter-5/results/h-main'
EXHAUSTIVE_RESULTS = '/Users/jchen/go/src/inference-sim/inference-sim/.nous/campaign-3/runs/iter-5/results/h-robustness/pareto_front.json'
TMPDIR = os.environ.get('TMPDIR', '/tmp')
REF = np.array([0.0, 50000.0, 9.0, 11000.0])
MC_SAMPLES = 200_000
MC_SEED = 42
HV_LO = np.array([-100.0, 0.0, 1.0, 2000.0])

FIXED_FLAGS = [
    '--model', 'qwen/qwen3-14b', '--hardware', 'H100',
    '--latency-model', 'trained-physics', '--num-requests', '200',
    '--rate', '50', '--prefix-tokens', '512', '--seed', '42',
    '--max-num-scheduled-tokens', '4096', '--long-prefill-token-threshold', '0',
    '--admission-policy', 'always-admit', '--preemption-policy', 'fcfs',
    '--routing-policy', 'round-robin', '--gpu-memory-utilization', '0.9',
]

TP_VALUES = [1, 2, 4, 8]
SCHED_VALUES = ['fcfs', 'sjf']
BATCH_VALUES = [32, 64, 128, 256, 512]
KV_VALUES = [2000, 3000, 4000, 5000, 7500, 10000]
BS_VALUES = [16, 32]
POP_SIZE = 40
N_GENS = 4
MUTATION_RATE = 0.15
TOTAL_EVALS = POP_SIZE * (1 + N_GENS)  # 200


def decode_gene(gene):
    tp_idx, inst_idx, sched_idx, batch_idx, kv_idx, bs_idx = gene
    tp = TP_VALUES[tp_idx]
    return tp, inst_idx + 1, SCHED_VALUES[sched_idx], BATCH_VALUES[batch_idx], KV_VALUES[kv_idx], BS_VALUES[bs_idx]


def random_gene(rng):
    tp_idx = int(rng.integers(0, 4))
    tp = TP_VALUES[tp_idx]
    inst_idx = int(rng.integers(0, 8 // tp))
    return (tp_idx, inst_idx, int(rng.integers(0, 2)), int(rng.integers(0, 5)),
            int(rng.integers(0, 6)), int(rng.integers(0, 2)))


def crossover(p1, p2, rng):
    child = [p1[i] if rng.random() < 0.5 else p2[i] for i in range(6)]
    tp = TP_VALUES[child[0]]
    child[1] = min(child[1], 8 // tp - 1)
    return tuple(child)


def mutate(gene, rng):
    g = list(gene)
    if rng.random() < MUTATION_RATE:
        g[0] = int(rng.integers(0, 4))
    tp = TP_VALUES[g[0]]
    if rng.random() < MUTATION_RATE:
        g[1] = int(rng.integers(0, 8 // tp))
    g[1] = min(g[1], 8 // tp - 1)
    if rng.random() < MUTATION_RATE:
        g[2] = int(rng.integers(0, 2))
    if rng.random() < MUTATION_RATE:
        g[3] = int(rng.integers(0, 5))
    if rng.random() < MUTATION_RATE:
        g[4] = int(rng.integers(0, 6))
    if rng.random() < MUTATION_RATE:
        g[5] = int(rng.integers(0, 2))
    return tuple(g)


def dominates(p, q):
    better = False
    for a, b in zip(p, q):
        if a > b:
            return False
        if a < b:
            better = True
    return better


def fast_nds(objs):
    """Fast non-dominated sort. Returns list of fronts (each front = list of indices)."""
    n = len(objs)
    S = [[] for _ in range(n)]
    nc = [0] * n
    fronts = [[]]
    for i in range(n):
        for j in range(n):
            if i == j:
                continue
            if dominates(objs[i], objs[j]):
                S[i].append(j)
            elif dominates(objs[j], objs[i]):
                nc[i] += 1
        if nc[i] == 0:
            fronts[0].append(i)
    fi = 0
    while fronts[fi]:
        nxt = []
        for i in fronts[fi]:
            for j in S[i]:
                nc[j] -= 1
                if nc[j] == 0:
                    nxt.append(j)
        fi += 1
        fronts.append(nxt)
    return fronts[:-1]


def crowding_dist(objs, front):
    n = len(front)
    if n <= 2:
        return {i: float('inf') for i in front}
    d = len(objs[0])
    dist = {i: 0.0 for i in front}
    for dim in range(d):
        s = sorted(front, key=lambda i: objs[i][dim])
        dist[s[0]] = float('inf')
        dist[s[-1]] = float('inf')
        rng_d = objs[s[-1]][dim] - objs[s[0]][dim]
        if rng_d == 0:
            continue
        for k in range(1, n - 1):
            dist[s[k]] += (objs[s[k+1]][dim] - objs[s[k-1]][dim]) / rng_d
    return dist


def nsga2_select(pop, objs, n_select):
    """Select n_select individuals using NSGA-II ranking."""
    fronts = fast_nds(objs)
    rank = [0] * len(pop)
    for r, front in enumerate(fronts):
        for i in front:
            rank[i] = r
    cd = {}
    for front in fronts:
        cd.update(crowding_dist(objs, front))
    order = sorted(range(len(pop)), key=lambda i: (rank[i], -cd.get(i, 0)))
    return [pop[i] for i in order[:n_select]], [objs[i] for i in order[:n_select]]


def tournament_select(pop, objs, rng, n_select):
    fronts = fast_nds(objs)
    rank = [0] * len(pop)
    for r, front in enumerate(fronts):
        for i in front:
            rank[i] = r
    cd = {}
    for front in fronts:
        cd.update(crowding_dist(objs, front))
    selected = []
    for _ in range(n_select):
        a, b = int(rng.integers(0, len(pop))), int(rng.integers(0, len(pop)))
        if rank[a] < rank[b] or (rank[a] == rank[b] and cd.get(a, 0) >= cd.get(b, 0)):
            selected.append(pop[a])
        else:
            selected.append(pop[b])
    return selected


def evaluate(tp, inst, sched, batch, kv, bs, idx, label=''):
    mf = os.path.join(TMPDIR, f'blis_iter5_nsga2_{idx}.json')
    cmd = [BLIS, 'run'] + FIXED_FLAGS + [
        '--tp', str(tp), '--num-instances', str(inst),
        '--scheduler', sched, '--max-num-running-reqs', str(batch),
        '--total-kv-blocks', str(kv), '--block-size-in-tokens', str(bs),
        '--metrics-path', mf,
    ]
    try:
        r = subprocess.run(cmd, capture_output=True, timeout=60)
        if r.returncode != 0 or not os.path.exists(mf):
            return (0.0, 50000.0, float(tp * inst), float(kv))
        with open(mf) as f:
            d = json.load(f)
        os.unlink(mf)
        return (-d['responses_per_sec'], d['ttft_p99_ms'], float(tp * inst), float(kv))
    except Exception as e:
        print(f"  WARN {label}: {e}", file=sys.stderr)
        return (0.0, 50000.0, float(tp * inst), float(kv))


def pareto_update(nd, new_pt):
    for p in nd:
        if all(p[k] <= new_pt[k] for k in range(4)) and any(p[k] < new_pt[k] for k in range(4)):
            return nd
    return [p for p in nd if not (
        all(new_pt[k] <= p[k] for k in range(4)) and any(new_pt[k] < p[k] for k in range(4))
    )] + [new_pt]


def hypervolume_mc(pareto_pts):
    if not pareto_pts:
        return 0.0
    pts = np.array(pareto_pts, dtype=np.float64)
    rng_mc = np.random.default_rng(MC_SEED)
    box_vol = float(np.prod(REF - HV_LO))
    dominated = 0
    batch = 50_000
    for _ in range(MC_SAMPLES // batch):
        s = rng_mc.uniform(HV_LO, REF, size=(batch, 4))
        dom = np.any(np.all(pts[np.newaxis, :, :] <= s[:, np.newaxis, :], axis=2), axis=1)
        dominated += int(np.sum(dom))
    return box_vol * dominated / MC_SAMPLES


def main():
    os.makedirs(RESULTS_DIR, exist_ok=True)
    rng = np.random.default_rng(42)

    # Load exhaustive HV for normalization
    exhaustive_hv = None
    if os.path.exists(EXHAUSTIVE_RESULTS):
        with open(EXHAUSTIVE_RESULTS) as f:
            exhaustive_hv = json.load(f).get('exhaustive_hv')
        print(f"Loaded exhaustive HV: {exhaustive_hv:.6e}", file=sys.stderr)

    eval_counter = [0]
    current_nd = []
    convergence = []
    all_evaluated = []  # (gene, obj)
    t0 = time.time()

    def eval_gene(gene, label=''):
        tp, inst, sched, batch, kv, bs = decode_gene(gene)
        obj = evaluate(tp, inst, sched, batch, kv, bs, eval_counter[0], label)
        eval_counter[0] += 1
        return obj

    def record_checkpoint():
        hv = hypervolume_mc(current_nd)
        pct = (hv / exhaustive_hv * 100) if exhaustive_hv else None
        entry = {
            'eval': eval_counter[0],
            'hv': hv,
            'pareto_size': len(current_nd),
            'pct_of_exhaustive': round(pct, 2) if pct is not None else None,
            'elapsed_s': round(time.time() - t0, 1),
        }
        convergence.append(entry)
        print(f"Eval {eval_counter[0]}: HV={hv:.4e} Pareto={len(current_nd)}"
              + (f" ({pct:.1f}%)" if pct else ""), file=sys.stderr)

    # Initialize population
    print("Initializing population...", file=sys.stderr)
    pop = [random_gene(rng) for _ in range(POP_SIZE)]
    pop_objs = []
    for gene in pop:
        obj = eval_gene(gene, f'init')
        pop_objs.append(obj)
        current_nd = pareto_update(current_nd, obj)
        all_evaluated.append((gene, obj))
    record_checkpoint()

    # Evolve for N_GENS generations
    for gen in range(N_GENS):
        print(f"Generation {gen+1}/{N_GENS}...", file=sys.stderr)
        parents = tournament_select(pop, pop_objs, rng, POP_SIZE)
        offspring = []
        offspring_objs = []
        for i in range(0, POP_SIZE, 2):
            p1, p2 = parents[i], parents[(i + 1) % POP_SIZE]
            c1 = mutate(crossover(p1, p2, rng), rng)
            c2 = mutate(crossover(p2, p1, rng), rng)
            for child in [c1, c2]:
                obj = eval_gene(child, f'gen{gen+1}')
                offspring.append(child)
                offspring_objs.append(obj)
                current_nd = pareto_update(current_nd, obj)
                all_evaluated.append((child, obj))
        # Select survivors: combine parent + offspring, keep best POP_SIZE
        combined_pop = pop + offspring
        combined_objs = pop_objs + offspring_objs
        pop, pop_objs = nsga2_select(combined_pop, combined_objs, POP_SIZE)
        record_checkpoint()

    # Save results
    pareto_out = {
        'total_evals': eval_counter[0],
        'pareto_size': len(current_nd),
        'exhaustive_hv': exhaustive_hv,
        'final_hv': convergence[-1]['hv'],
        'final_pct_of_exhaustive': convergence[-1]['pct_of_exhaustive'],
        'reference_point': list(REF),
        'pareto_front': [{'objectives': list(p)} for p in current_nd],
    }
    with open(os.path.join(RESULTS_DIR, 'pareto_front.json'), 'w') as f:
        json.dump(pareto_out, f, indent=2)
    with open(os.path.join(RESULTS_DIR, 'convergence.json'), 'w') as f:
        json.dump(convergence, f, indent=2)

    print(f"\nTotal evals: {eval_counter[0]}")
    print(f"Final HV: {convergence[-1]['hv']:.6e}")
    if exhaustive_hv:
        print(f"% of exhaustive: {convergence[-1]['pct_of_exhaustive']:.1f}%")
    print(f"Results saved to {RESULTS_DIR}")


if __name__ == '__main__':
    main()
