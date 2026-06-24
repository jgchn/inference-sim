#!/usr/bin/env python3
"""Arm h-control-negative: Random search on reduced 5-parameter space, 200 evals."""
import json, os, random, subprocess, sys, time
import numpy as np

BLIS = os.environ.get('BLIS_BIN', os.path.join(os.getcwd(), 'blis'))
RESULTS_DIR = os.path.join(os.path.dirname(os.path.dirname(__file__)), 'results', 'h-control-negative')
os.makedirs(RESULTS_DIR, exist_ok=True)

TP_VALUES = [1, 2, 4, 8]
SCHEDULER_VALUES = ['fcfs', 'sjf']
BATCH_VALUES = [32, 64, 128, 256, 512]
KV_VALUES = [2000, 3000, 4000, 5000, 7500, 10000]
REF_POINT = np.array([0.0, 50000.0, 9.0, 11000.0])
BUDGET = 200
TRACK_EVERY = 40
SEED = 42

FIXED_FLAGS = [
    '--model', 'qwen/qwen3-14b', '--hardware', 'H100',
    '--latency-model', 'trained-physics',
    '--num-requests', '200', '--rate', '50', '--prefix-tokens', '512', '--seed', '42',
    '--max-num-scheduled-tokens', '4096', '--long-prefill-token-threshold', '0',
    '--admission-policy', 'always-admit', '--preemption-policy', 'fcfs',
    '--block-size-in-tokens', '16', '--routing-policy', 'round-robin',
    '--gpu-memory-utilization', '0.9',
]


def random_config(rng):
    tp = int(rng.choice(TP_VALUES))
    max_inst = 8 // tp
    inst = int(rng.integers(1, max_inst + 1))
    return {
        'tp': tp, 'num_instances': inst,
        'scheduler': str(rng.choice(SCHEDULER_VALUES)),
        'max_num_running_reqs': int(rng.choice(BATCH_VALUES)),
        'total_kv_blocks': int(rng.choice(KV_VALUES)),
    }


def run_config(cfg, idx):
    mp = os.path.join(os.environ['TMPDIR'], f'metrics_rnd_{idx}.json')
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


def pareto_front(objs):
    nd_mask = [True] * len(objs)
    for i in range(len(objs)):
        if not nd_mask[i]:
            continue
        for j in range(len(objs)):
            if i != j and nd_mask[j] and dominates(objs[j], objs[i]):
                nd_mask[i] = False
                break
    return [i for i, m in enumerate(nd_mask) if m]


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


def main():
    rng = np.random.default_rng(SEED)
    all_objs = []
    all_configs_run = []
    convergence = []
    t0 = time.time()

    for i in range(BUDGET):
        cfg = random_config(rng)
        metrics = run_config(cfg, i)
        if metrics is None:
            print(f'  FAILED config {i}: {cfg}', flush=True)
            continue
        gpu_count = cfg['tp'] * cfg['num_instances']
        obj = [-metrics['responses_per_sec'], metrics['ttft_p99_ms'], gpu_count, cfg['total_kv_blocks']]
        all_objs.append(obj)
        all_configs_run.append({'config': cfg, 'metrics': {
            'responses_per_sec': metrics['responses_per_sec'],
            'ttft_p99_ms': metrics['ttft_p99_ms'],
            'preemption_count': metrics.get('preemption_count', 0),
        }, 'objectives': obj})

        # Track HV every TRACK_EVERY evals
        if (i + 1) % TRACK_EVERY == 0:
            pf_idx = pareto_front(all_objs)
            pf_objs = [all_objs[j] for j in pf_idx]
            hv = compute_hv(pf_objs)
            convergence.append({'eval': i + 1, 'hv': hv, 'pareto_size': len(pf_idx)})
            print(f'  eval={i+1}: HV={hv:.4e}, pareto_size={len(pf_idx)}', flush=True)

    # Final Pareto front
    pf_idx = pareto_front(all_objs)
    pf_objs = [all_objs[j] for j in pf_idx]
    final_hv = compute_hv(pf_objs)
    print(f'Final HV={final_hv:.4e}, pareto_size={len(pf_idx)}, time={time.time()-t0:.1f}s', flush=True)

    pf_out = [{'config': all_configs_run[j]['config'], 'objectives': {
        'neg_rps': all_objs[j][0], 'ttft_p99_ms': all_objs[j][1],
        'gpu_count': all_objs[j][2], 'total_kv_blocks': all_objs[j][3]
    }} for j in pf_idx]

    with open(os.path.join(RESULTS_DIR, 'pareto_front.json'), 'w') as f:
        json.dump({'pareto_front': pf_out, 'final_hv': final_hv,
                   'total_evals': len(all_configs_run), 'pareto_size': len(pf_idx)}, f, indent=2)
    with open(os.path.join(RESULTS_DIR, 'convergence.json'), 'w') as f:
        json.dump(convergence, f, indent=2)
    print('Done.', flush=True)


if __name__ == '__main__':
    main()
