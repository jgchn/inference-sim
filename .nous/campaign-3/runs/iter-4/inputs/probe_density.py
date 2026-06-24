#!/usr/bin/env python3
"""Exhaustive sweep of tp=2,i=2 tier to measure within-tier Pareto density."""
import subprocess, json, os

BLIS = "/Users/jchen/go/src/inference-sim/inference-sim/blis"
TMPDIR = os.environ.get("TMPDIR", "/tmp/claude")

FIXED = [
    "--model", "qwen/qwen3-14b", "--hardware", "H100",
    "--latency-model", "trained-physics", "--num-requests", "200",
    "--rate", "50", "--prefix-tokens", "512", "--seed", "42",
    "--max-num-scheduled-tokens", "4096", "--long-prefill-token-threshold", "0",
    "--admission-policy", "always-admit", "--preemption-policy", "fcfs",
    "--block-size-in-tokens", "16", "--routing-policy", "round-robin",
    "--gpu-memory-utilization", "0.9",
]

results = []
for sched in ["fcfs", "sjf"]:
    for batch in [32, 64, 128, 256, 512]:
        for kv in [2000, 3000, 4000, 5000, 7500, 10000]:
            mfile = os.path.join(TMPDIR, f"exhaust_{sched}_{batch}_{kv}.json")
            cmd = [BLIS, "run"] + FIXED + [
                "--tp", "2", "--num-instances", "2",
                "--scheduler", sched,
                "--max-num-running-reqs", str(batch),
                "--total-kv-blocks", str(kv),
                "--metrics-path", mfile,
            ]
            r = subprocess.run(cmd, capture_output=True, timeout=30)
            if r.returncode == 0:
                d = json.load(open(mfile))
                results.append({
                    "sched": sched, "batch": batch, "kv": kv,
                    "rps": d["responses_per_sec"],
                    "ttft": d["ttft_p99_ms"],
                    "gpu": 4,
                })


def dominates(a, b):
    objs_a = [-a["rps"], a["ttft"], a["gpu"], a["kv"]]
    objs_b = [-b["rps"], b["ttft"], b["gpu"], b["kv"]]
    return all(ai <= bi for ai, bi in zip(objs_a, objs_b)) and any(ai < bi for ai, bi in zip(objs_a, objs_b))


pareto = []
for i, a in enumerate(results):
    dominated = False
    for j, b in enumerate(results):
        if i != j and dominates(b, a):
            dominated = True
            break
    if not dominated:
        pareto.append(a)

print(f"Total configs evaluated: {len(results)}")
print(f"Pareto-optimal: {len(pareto)} ({len(pareto)/len(results)*100:.1f}%)")
print("\nPareto front (within tp=2,i=2 tier):")
for p in sorted(pareto, key=lambda x: -x["rps"]):
    print(f"  sched={p['sched']:4s} batch={p['batch']:3d} kv={p['kv']:5d} -> rps={p['rps']:.2f} ttft={p['ttft']:.1f}")
