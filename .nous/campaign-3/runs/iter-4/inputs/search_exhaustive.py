#!/usr/bin/env python3
"""Arm h-robustness: Exhaustive sweep of all 900 reduced-space configs."""
import json, os, subprocess, sys, time
import numpy as np

BLIS = os.environ.get('BLIS_BIN', os.path.join(os.getcwd(), 'blis'))
RESULTS_DIR = os.path.join(os.path.dirname(os.path.dirname(__file__)), 'results', 'h-robustness')
os.makedirs(RESULTS_DIR, exist_ok=True)

TP_VALUES = [1, 2, 4, 8]
SCHEDULER_VALUES = ['fcfs', 'sjf']
BATCH_VALUES = [32, 64, 128, 256, 512]
KV_VALUES = [2000, 3000, 4000, 5000, 7500, 10000]
REF_POINT = np.array([0.0, 50000.0, 9.0, 11000.0])

FIXED_FLAGS = [
    '--model', 'qwen/qwen3-14b', '--hardware', 'H100',
    '--latency-model', 'trained-physics',
    '--num-requests', '200', '--rate', '50', '--prefix-tokens', '512', '--seed', '42',
    '--max-num-scheduled-tokens', '4096', '--long-prefill-token-threshold', '0',
    '--admission-policy', 'always-admit', '--preemption-policy', 'fcfs',
    '--block-size-in-tokens', '16', '--routing-policy', 'round-robin',
    '--gpu-memory-utilization', '0.9',
]


def all_configs():
    configs = []
    for tp in TP_VALUES:
        max_inst = 8 // tp
        for inst in range(1, max_inst + 1):
            for sched in SCHEDULER_VALUES:
                for batch in BATCH_VALUES:
                    for kv in KV_VALUES:
                        configs.append({'tp': tp, 'num_instances': inst, 'scheduler': sched,
                                        'max_num_running_reqs': batch, 'total_kv_blocks': kv})
    return configs


def run_config(cfg, idx):
    mp = os.path.join(os.environ['TMPDIR'], f'metrics_exh_{idx}.json')
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
    for rec in (data if isinstance(data, list) else [data]):
        if isinstance(rec, dict) and rec.get('instance_id') == 'cluster':
            return rec
    # single-instance case: look for non-cluster record or first dict
    if isinstance(data, list):
        for rec in data:
            if isinstance(rec, dict) and 'responses_per_sec' in rec:
                return rec
    return data if isinstance(data, dict) else None


def dominates(a, b):
    return all(ai <= bi for ai, bi in zip(a, b)) and any(ai < bi for ai, bi in zip(a, b))


def pareto_front(points_objs):
    nd = []
    for i, (cfg, obj) in enumerate(points_objs):
        dominated = False
        for j, (_, obj2) in enumerate(points_objs):
            if i != j and dominates(obj2, obj):
                dominated = True
                break
        if not dominated:
            nd.append((cfg, obj))
    return nd


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


def to_obj(metrics, kv):
    rps = metrics['responses_per_sec']
    ttft = metrics['ttft_p99_ms']
    gpu = metrics.get('gpu_count', None)
    if gpu is None:
        # derive from instance_id being 'cluster' — we need to pass it
        gpu = metrics.get('_gpu_count', 1)
    return [-rps, ttft, gpu, kv]


def main():
    configs = all_configs()
    print(f'Total configs: {len(configs)}', flush=True)
    all_results = []
    t0 = time.time()
    for i, cfg in enumerate(configs):
        metrics = run_config(cfg, i)
        if metrics is None:
            print(f'  FAILED config {i}: {cfg}', flush=True)
            continue
        gpu_count = cfg['tp'] * cfg['num_instances']
        obj = [-metrics['responses_per_sec'], metrics['ttft_p99_ms'], gpu_count, cfg['total_kv_blocks']]
        all_results.append({'config': cfg, 'metrics': {
            'responses_per_sec': metrics['responses_per_sec'],
            'ttft_p99_ms': metrics['ttft_p99_ms'],
            'preemption_count': metrics.get('preemption_count', 0),
        }, 'objectives': obj})
        if (i + 1) % 100 == 0:
            elapsed = time.time() - t0
            print(f'  {i+1}/{len(configs)} evals, {elapsed:.1f}s elapsed', flush=True)

    print(f'Completed {len(all_results)}/{len(configs)} evals in {time.time()-t0:.1f}s', flush=True)

    # Compute Pareto front
    pts_objs = [(r['config'], r['objectives']) for r in all_results]
    pf = pareto_front(pts_objs)
    pf_objs = [obj for _, obj in pf]
    ref_hv = compute_hv(pf_objs)

    print(f'Pareto front size: {len(pf)} / {len(all_results)} ({100*len(pf)/len(all_results):.1f}%)', flush=True)
    print(f'Reference HV: {ref_hv:.4e}', flush=True)

    # Save Pareto front
    pf_out = []
    for cfg, obj in pf:
        pf_out.append({'config': cfg, 'objectives': {
            'neg_rps': obj[0], 'ttft_p99_ms': obj[1], 'gpu_count': obj[2], 'total_kv_blocks': obj[3]
        }})
    with open(os.path.join(RESULTS_DIR, 'pareto_front.json'), 'w') as f:
        json.dump({'pareto_front': pf_out, 'reference_hv': ref_hv,
                   'total_configs': len(all_results), 'pareto_size': len(pf),
                   'pareto_density_pct': 100*len(pf)/len(all_results)}, f, indent=2)

    # Save all results for reference
    with open(os.path.join(RESULTS_DIR, 'all_results.json'), 'w') as f:
        json.dump(all_results, f, indent=2)

    # Convergence: not meaningful for exhaustive, but record reference
    with open(os.path.join(RESULTS_DIR, 'convergence.json'), 'w') as f:
        json.dump({'reference_hv': ref_hv, 'total_evals': len(all_results),
                   'pareto_size': len(pf), 'pareto_density_pct': 100*len(pf)/len(all_results)}, f, indent=2)

    print('Done.', flush=True)


if __name__ == '__main__':
    main()
