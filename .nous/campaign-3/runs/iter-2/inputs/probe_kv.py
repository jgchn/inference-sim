"""Test KV-constrained and multi-tier scenarios."""
import subprocess, json, os

tmpdir = os.environ["TMPDIR"]

# Part 1: KV-constrained at rate=50
configs_kv = [
    {"tp": 2, "inst": 2, "batch": 32},
    {"tp": 2, "inst": 2, "batch": 64},
    {"tp": 2, "inst": 2, "batch": 128},
    {"tp": 2, "inst": 2, "batch": 256},
    {"tp": 2, "inst": 2, "batch": 512},
]
print("=== KV-constrained (gpu-mem-util=0.85, rate=50, 500 reqs, tp=2/i=2) ===")
for c in configs_kv:
    label = f"b{c['batch']}"
    mpath = os.path.join(tmpdir, f"kv_probe_{label}.json")
    cmd = ["./blis", "run", "--model", "qwen/qwen3-14b", "--hardware", "H100",
           "--latency-model", "trained-physics", "--num-requests", "500", "--rate", "50",
           "--tp", str(c["tp"]), "--num-instances", str(c["inst"]),
           "--scheduler", "fcfs", "--max-num-running-reqs", str(c["batch"]),
           "--max-num-scheduled-tokens", "4096",
           "--long-prefill-token-threshold", "0", "--block-size-in-tokens", "16",
           "--routing-policy", "least-loaded", "--admission-policy", "always-admit",
           "--preemption-policy", "fcfs", "--gpu-memory-utilization", "0.85",
           "--seed", "42", "--metrics-path", mpath]
    subprocess.run(cmd, capture_output=True)
    try:
        d = json.load(open(mpath))
        print(f"  {label:5s}: rps={d['responses_per_sec']:.2f} ttft_p99={d['ttft_p99_ms']:.1f} itl_p99={d['itl_p99_ms']:.2f} preempt={d['preemption_count']}")
    except Exception as e:
        print(f"  {label:5s}: FAILED ({e})")

# Part 2: Multi-tier best configs at rate=50
print("\n=== Multi-tier best configs at rate=50, 500 reqs ===")
configs_tier = [
    {"tp": 1, "inst": 1, "batch": 256, "routing": None, "label": "1gpu"},
    {"tp": 1, "inst": 2, "batch": 256, "routing": "least-loaded", "label": "2gpu-tp1"},
    {"tp": 2, "inst": 1, "batch": 256, "routing": None, "label": "2gpu-tp2"},
    {"tp": 1, "inst": 4, "batch": 256, "routing": "least-loaded", "label": "4gpu-tp1"},
    {"tp": 2, "inst": 2, "batch": 256, "routing": "least-loaded", "label": "4gpu-tp2"},
    {"tp": 4, "inst": 1, "batch": 256, "routing": None, "label": "4gpu-tp4"},
    {"tp": 2, "inst": 4, "batch": 256, "routing": "least-loaded", "label": "8gpu-tp2"},
    {"tp": 4, "inst": 2, "batch": 256, "routing": "least-loaded", "label": "8gpu-tp4"},
    {"tp": 8, "inst": 1, "batch": 256, "routing": None, "label": "8gpu-tp8"},
]
print(f"{'label':12s} gpus  rps      ttft_p99    e2e_p99      itl_p99")
for c in configs_tier:
    mpath = os.path.join(tmpdir, f"tier_{c['label']}.json")
    cmd = ["./blis", "run", "--model", "qwen/qwen3-14b", "--hardware", "H100",
           "--latency-model", "trained-physics", "--num-requests", "500", "--rate", "50",
           "--tp", str(c["tp"]), "--num-instances", str(c["inst"]),
           "--scheduler", "fcfs", "--max-num-running-reqs", str(c["batch"]),
           "--max-num-scheduled-tokens", "4096",
           "--long-prefill-token-threshold", "0", "--block-size-in-tokens", "16",
           "--preemption-policy", "fcfs", "--gpu-memory-utilization", "0.9",
           "--seed", "42", "--metrics-path", mpath]
    if c["inst"] > 1 and c["routing"]:
        cmd += ["--routing-policy", c["routing"], "--admission-policy", "always-admit"]
    subprocess.run(cmd, capture_output=True)
    try:
        d = json.load(open(mpath))
        gpus = c["tp"] * c["inst"]
        print(f"{c['label']:12s} {gpus:4d}  {d['responses_per_sec']:7.2f}  {d['ttft_p99_ms']:10.1f}  {d['e2e_p99_ms']:10.1f}  {d['itl_p99_ms']:7.2f}")
    except Exception as e:
        print(f"{c['label']:12s} FAILED ({e})")
