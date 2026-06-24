"""Validate baseline command and benchmark eval throughput."""
import subprocess, json, os, time

tmpdir = os.environ["TMPDIR"]

# Part 1: Validate baseline
mpath = os.path.join(tmpdir, "baseline_iter2.json")
start = time.time()
cmd = ["./blis", "run", "--model", "qwen/qwen3-14b", "--hardware", "H100",
       "--latency-model", "trained-physics", "--num-requests", "500", "--rate", "50",
       "--tp", "2", "--num-instances", "2", "--scheduler", "fcfs",
       "--max-num-running-reqs", "128", "--max-num-scheduled-tokens", "4096",
       "--long-prefill-token-threshold", "0", "--block-size-in-tokens", "16",
       "--routing-policy", "least-loaded", "--admission-policy", "always-admit",
       "--preemption-policy", "fcfs", "--gpu-memory-utilization", "0.9",
       "--seed", "42", "--metrics-path", mpath]
r = subprocess.run(cmd, capture_output=True, text=True)
elapsed = time.time() - start
print(f"Exit code: {r.returncode}")
print(f"Eval time: {elapsed*1000:.0f}ms")
d = json.load(open(mpath))
print(f"responses_per_sec: {d['responses_per_sec']:.4f}")
print(f"ttft_p99_ms: {d['ttft_p99_ms']:.4f}")
print(f"Metrics file size: {os.path.getsize(mpath)} bytes")

# Part 2: Benchmark throughput (20 evals)
print("\n--- Benchmarking 20 sequential evals ---")
start = time.time()
for i in range(20):
    mpath_i = os.path.join(tmpdir, f"timing_{i}.json")
    batch = [128, 256, 32, 64, 512][i % 5]
    tp = [1, 2, 4, 8][i % 4]
    inst = max(1, min(8 // tp, [1, 2, 4][i % 3]))
    cmd_i = ["./blis", "run", "--model", "qwen/qwen3-14b", "--hardware", "H100",
             "--latency-model", "trained-physics", "--num-requests", "500", "--rate", "50",
             "--tp", str(tp), "--num-instances", str(inst),
             "--scheduler", "fcfs", "--max-num-running-reqs", str(batch),
             "--max-num-scheduled-tokens", "4096",
             "--long-prefill-token-threshold", "0", "--block-size-in-tokens", "16",
             "--preemption-policy", "fcfs", "--gpu-memory-utilization", "0.9",
             "--seed", "42", "--metrics-path", mpath_i]
    if inst > 1:
        cmd_i += ["--routing-policy", "least-loaded", "--admission-policy", "always-admit"]
    subprocess.run(cmd_i, capture_output=True)
elapsed = time.time() - start
print(f"20 evals in {elapsed:.1f}s = {elapsed/20*1000:.0f}ms/eval")
print(f"Projected 550 evals: {elapsed/20*550:.0f}s = {elapsed/20*550/60:.1f}min")
