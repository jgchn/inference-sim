"""Probe 3-objective Pareto frontier richness at rate=50."""
import subprocess, json, os, random

tmpdir = os.environ["TMPDIR"]

configs = []
for tp, inst in [(4,1), (2,2), (1,4)]:
    for batch in [32, 64, 128, 256, 512]:
        for sched_tokens in [2048, 4096, 8192]:
            if inst == 1:
                configs.append({'tp': tp, 'inst': inst, 'batch': batch, 'sched': sched_tokens, 'routing': None})
            else:
                for routing in ['least-loaded', 'round-robin']:
                    configs.append({'tp': tp, 'inst': inst, 'batch': batch, 'sched': sched_tokens, 'routing': routing})

random.seed(42)
sample = random.sample(configs, min(20, len(configs)))

results = []
for c in sample:
    mpath = f"{tmpdir}/frontier_probe.json"
    cmd = ["./blis", "run", "--model", "qwen/qwen3-14b", "--hardware", "H100",
           "--latency-model", "trained-physics", "--num-requests", "500", "--rate", "50",
           "--tp", str(c['tp']), "--num-instances", str(c['inst']),
           "--scheduler", "fcfs", "--max-num-running-reqs", str(c['batch']),
           "--max-num-scheduled-tokens", str(c['sched']),
           "--long-prefill-token-threshold", "0", "--block-size-in-tokens", "16",
           "--preemption-policy", "fcfs", "--gpu-memory-utilization", "0.9",
           "--seed", "42", "--metrics-path", mpath]
    if c['inst'] > 1 and c['routing']:
        cmd += ["--routing-policy", c['routing'], "--admission-policy", "always-admit"]
    subprocess.run(cmd, capture_output=True)
    try:
        d = json.load(open(mpath))
        gpus = c['tp'] * c['inst']
        results.append((d['responses_per_sec'], d['ttft_p99_ms'], gpus, c))
    except:
        pass

# Find Pareto-optimal in 3D (max rps, min ttft, min gpus)
pareto = []
for i, (rps_i, ttft_i, gpu_i, _) in enumerate(results):
    dominated = False
    for j, (rps_j, ttft_j, gpu_j, _) in enumerate(results):
        if i == j:
            continue
        if rps_j >= rps_i and ttft_j <= ttft_i and gpu_j <= gpu_i:
            if rps_j > rps_i or ttft_j < ttft_i or gpu_j < gpu_i:
                dominated = True
                break
    if not dominated:
        pareto.append(i)

print(f"Total configs tested: {len(results)}")
print(f"Pareto-optimal configs: {len(pareto)} ({100*len(pareto)/max(1,len(results)):.0f}%)")
print(f"\nAll results (sorted by rps):")
for idx, (rps, ttft, gpu, c) in enumerate(sorted(results, key=lambda x: x[0])):
    orig_idx = results.index((rps, ttft, gpu, c))
    marker = "***" if orig_idx in pareto else "   "
    r = c['routing'] or 'N/A'
    print(f"{marker} tp={c['tp']} i={c['inst']} b={c['batch']:3d} st={c['sched']:4d} r={r:13s} | rps={rps:.2f} ttft={ttft:.1f} gpus={gpu}")
