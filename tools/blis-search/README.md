# blis-search — Multi-Objective Configuration Search for BLIS

A fast, parallel configuration search tool that discovers Pareto-optimal BLIS
configurations across a user-defined parameter space. Finds the best tradeoffs
between throughput, latency, and cost without requiring real GPUs.

Two search spaces ship with the tool:

| Space | Scope | Topology |
|---|---|---|
| `defaults.yaml` | instance scheduling + routing/admission policy | `tp` × `replicas` sampled directly, `tp * replicas <= 8` |
| `full-stack.yaml` | the above **plus** flow control and P/D disaggregation | enumerated from an `infrastructure:` block (16-GPU budget, 185 topologies) |

## Quick Start

```bash
# Build BLIS first — the search shells out to this binary
go build -o blis main.go
```

### Simple search

Instance-level tuning on the built-in `defaults.yaml` space. Throughput vs TTFT
P99, up to 8 GPUs, no P/D and no flow control.

```bash
python3 tools/blis-search/search.py \
  --model qwen/qwen3-14b --hardware H100 \
  --workload chatbot --rate 200 \
  --budget 500 --workers 8 \
  --output simple.json --verbose
```

```
BLIS Search: model=qwen/qwen3-14b hardware=H100 workload=chatbot rate=200 budget=500 workers=8 strategy=diversity
  DONE: HV=0.991623, valid=500/500, pareto_size=110, tp_classes=[1, 2, 4, 8], t=14.0s
```

Variations:

```bash
# SLO-feasible subset alongside the front
python3 tools/blis-search/search.py \
  --model qwen/qwen3-14b --hardware H100 --workload chatbot --rate 200 \
  --objectives "responses_per_sec:maximize,ttft_p99_ms:minimize" \
  --slo "ttft_p99_ms<200,responses_per_sec>50"

# Different workload and load point
python3 tools/blis-search/search.py \
  --model qwen/qwen3-14b --hardware H100 \
  --workload summarization --rate 1000 --budget 500 --workers 8
```

### Full-stack search

Joint search over routing, admission control, gateway flow control, and
prefill/decode disaggregation, with the topology itself enumerated from a GPU
budget. Because `gpus_used` is reported per config, this is the mode to use for
"cheapest deployment that meets my SLO".

```bash
python3 tools/blis-search/search.py \
  --model qwen/qwen3-14b --hardware H100 \
  --workload chatbot --rate 200 --num-requests 2000 \
  --search-space tools/blis-search/full-stack.yaml \
  --objectives "responses_per_sec:maximize,ttft_p99_ms:minimize,gpus_used:minimize" \
  --slo "ttft_p99_ms<500,responses_per_sec>50" \
  --budget 1000 --workers 8 \
  --output full-stack.json --output-all --verbose
```

```
  Phase 1 converged at eval 31, HV=0.0000, missing GPU tiers: [1, 3, 4, 5, 7, 9, 10, 13, 14]
  DONE: HV=0.000000, valid=1000/1000, pareto_size=13, gpu_tiers=[1, 3, 4, 5, 6, 7, 8, 9, 10, 12, 16], t=281.0s
```

`HV=0.0000` is expected here, not a failure — hypervolume is 2-objective only
(see [Sharp Edges](#sharp-edges)). The 13-point front is correct, and because
both SLO metrics are also objectives, `slo_feasible` is populated and answers
the actual question: 5 GPUs is the cheapest topology clearing TTFT P99 < 500ms
at > 50 req/s.

| gpus_used | responses_per_sec | ttft_p99_ms |
|---|---|---|
| 5 | 107.3 | 62.4 |
| 6 | 112.8 | 56.8 |
| 7 | 113.8 | 51.0 |
| 9 | 116.8 | 48.1 |
| 10 | 143.0 | 34.8 |
| 12 | 155.1 | 486.1 |
| 16 | 169.1 | 26.6 |

**When you want a live convergence signal**, drop to two objectives — cost
against throughput — and accept that `--slo` can then only reference those two
metrics:

```bash
python3 tools/blis-search/search.py \
  --model qwen/qwen3-14b --hardware H100 \
  --workload chatbot --rate 200 --num-requests 2000 \
  --search-space tools/blis-search/full-stack.yaml \
  --objectives "responses_per_sec:maximize,gpus_used:minimize" \
  --budget 1000 --workers 8 --output-all --output full-stack.json --verbose
```

```
  Phase 1 converged at eval 41, HV=0.8452, missing GPU tiers: [1, 3, 4, 7, 9, 10, 11, 13, 14, 15]
  DONE: HV=0.845256, valid=1000/1000, pareto_size=6, gpu_tiers=[1, 3, 4, 8, 12, 16], t=323.8s
```

The front is a plain cost/throughput curve — one entry per GPU tier that buys
real throughput, with the diminishing return past 4 GPUs visible directly (this
workload gives the same six points at budget 200 and at budget 1000):

| gpus_used | responses_per_sec |
|---|---|
| 1 | 48.0 |
| 3 | 98.5 |
| 4 | 136.0 |
| 8 | 141.4 |
| 12 | 155.1 |
| 16 | 169.1 |

**Always put something cost-shaped on an axis.** With `responses_per_sec` +
`ttft_p99_ms` and nothing penalising cost, the full 16-GPU topology dominates on
*both* axes and the front collapses to a single point:

```bash
# pareto_size=1 (tp=8, replicas=2, 16 GPUs) — a front with nothing to choose between
python3 tools/blis-search/search.py \
  --model qwen/qwen3-14b --hardware H100 \
  --workload chatbot --rate 200 --num-requests 2000 \
  --search-space tools/blis-search/full-stack.yaml --budget 300
```

Long-context workloads are where P/D disaggregation earns its keep:

```bash
python3 tools/blis-search/search.py \
  --model qwen/qwen3-14b --hardware H100 \
  --workload summarization --rate 50 --num-requests 500 \
  --search-space tools/blis-search/full-stack.yaml \
  --objectives "responses_per_sec:maximize,e2e_p99_ms:minimize,gpus_used:minimize" \
  --budget 1000 --workers 8
```

On that run 12 of the 18 front entries are P/D splits, against a chatbot front
where aggregated topologies mostly win — the disaggregation knobs are worth
searching exactly when prefill dominates.

## Sharp Edges

One thing is load-bearing for how you read a result:

* **Hypervolume is 2-objective only.** With 3+ objectives `final_hv` reports
  `0.0`, so convergence detection sees a flat trajectory and fires at eval
  `--convergence-k + 1` (eval 31 by default) regardless of the real front.
  Pareto extraction itself is N-dimensional and correct — only the HV number
  and the Phase 1/Phase 2 boundary degrade.

## Algorithm

Two-phase hierarchical random search with HV convergence detection and
diversity-biased Phase 2 targeting. Developed and validated across 15
iterations of hypothesis-driven experimentation.

**Phase 1 — Exploration:** Hierarchical random sampling over the search space.
Samples topology first (a pre-enumerated topology in infrastructure mode; `tp`
then `replicas` in legacy mode), then policy parameters, then conditionally
samples scorer profiles and flow-control parameters only for eligible configs.
This preserves the probability of finding the global optimum regardless of how
large the conditional expansion is.

**Convergence Detection:** Monitors the hypervolume indicator after each valid
evaluation. Declares convergence when HV has not improved by ≥ε (default
0.001) over K (default 30) consecutive evaluations.

**Phase 2 — Diversity Targeting:** After convergence, switches to targeting
under-represented cost classes in ascending order — **GPU-count tiers** in
infrastructure mode, **TP classes** in legacy mode. Uses safe parameter values
from `diversity_safe` (excluding pathological configs like always-busiest
routing and token-bucket admission) to maximize the chance of finding viable
configs in each class.

### Why This Algorithm

The research campaign tested random search, Latin Hypercube Sampling, TPE
(Tree-structured Parzen Estimator), and NSGA-II. None outperformed random
search at sufficient budget on BLIS's configuration Pareto surface. The key
insight: BLIS's surface has strong TP dominance with only ~4 meaningfully
distinct regions, all discoverable by random sampling.

What does matter:
- **Hierarchical sampling** preserves P(global optimum) under conditional
  expansion (scorer profiles, flow control). Flat uniform sampling suffers
  proportional dilution.
- **Diversity-biased Phase 2** improves mean HV on non-degenerate surfaces by
  +0.34 percentage points and achieves 4/5 seeds ≥99% quality vs 3/5 for
  continued random.

## Search Space Configuration

The search space is defined in a YAML file. Use `--search-space` to provide a
custom one, or omit it to use the built-in `defaults.yaml`.

### Legacy mode — topology as ordinary parameters (`defaults.yaml`)

```yaml
parameters:
  tp:
    flag: "--tp"
    values: [1, 2, 4, 8]
  replicas:
    flag: "--num-instances"
    max_replicas_by_tp: {1: 8, 2: 4, 4: 2, 8: 1}   # or gpu_budget: 8
  scheduler:
    flag: "--scheduler"
    values: [fcfs, priority-fcfs, sjf, reverse-priority]
  max_batch:
    flag: "--max-num-running-reqs"
    values: [32, 64, 128, 256, 512]
  # ... (see defaults.yaml for the full set)

constraints:
  - "tp * replicas <= 8"
  - "replicas == 1 -> routing_policy == 'round-robin'"
  - "replicas == 1 -> admission_policy == 'always-admit'"
  - "long_prefill == 0 or long_prefill < max_scheduled_tokens"

conditional_subsystems:
  routing_scorers:
    enabled_when: "routing_policy == 'weighted' and replicas > 1"
    flag: "--routing-scorers"
    format_b:
      scorers: [queue-depth, kv-utilization, load-balance]   # 26 profiles
      weights: [1, 2]

diversity_safe:                      # Phase 2 draws policy values from here
  routing_policy: [round-robin, least-loaded, weighted]
  admission_policy: [always-admit, tier-shed]

objectives:
  - metric: responses_per_sec
    direction: maximize
  - metric: ttft_p99_ms
    direction: minimize

hypervolume_reference:               # HV normalisation, per objective metric
  responses_per_sec: 100.0
  ttft_p99_ms: 50000.0
```

Constraints and `enabled_when` conditions are Python expressions over the
config dict, with `A -> B` sugar for implication. A condition that raises
evaluates to `false`.

### Infrastructure mode — topology enumerated from a GPU budget (`full-stack.yaml`)

An `infrastructure:` block replaces the `tp`/`replicas` parameters. The tool
enumerates every valid `(tp × instances)` topology up to `gpu_budget`, and when
`explore_pd` is true also every pure-disjoint prefill/decode split of each
multi-instance topology. Each topology carries a derived `gpus_used`, which is
injected into the metrics of every result so it can serve as an objective or a
reporting column.

```yaml
infrastructure:
  gpu_budget: 16
  tp: [1, 2, 4, 8]
  min_instances: 1                 # optional, default 1
  explore_pd: true
  pd_transfer_bandwidth: 25.0      # GB/s (NIXL RDMA default, fixed constant)
```

`full-stack.yaml`'s block yields **185 topologies** (30 aggregated, 155 P/D
splits) spanning GPU tiers 1–16. Flags emitted per topology: `--tp`,
`--num-instances`, and for P/D splits `--pd-decider always`,
`--prefill-instances`, `--decode-instances`, `--pd-transfer-bandwidth`. Single-
instance topologies are forced back to `pd_decider: never` with round-robin
routing and always-admit admission.

Flow control is a second conditional subsystem, gated on the admission policy
and sampled probabilistically so both the on and off branches get coverage:

```yaml
conditional_subsystems:
  routing_scorers:
    enabled_when: "routing_policy == 'weighted'"
    flag: "--routing-scorers"
    format_b:
      scorers: [queue-depth, kv-utilization, precise-prefix-cache,
                active-requests, load-balance, no-hit-lru]
      weights: [1, 2, 3, 4, 5]     # 46,655 profiles
  flow_control:
    enabled_when: "admission_policy == 'tier-shed'"
    probability: 0.6               # P(flow control on | eligible)
    flag: "--flow-control"
    parameters:
      saturation_detector: {flag: "--saturation-detector", values: [utilization, concurrency]}
      queue_depth_threshold: {flag: "--queue-depth-threshold", values: [3, 5, 10, 20]}
      # ... kv_util_threshold, dispatch_order, max_gateway_queue_depth, request_ttl
```

When a subsystem is not enabled its flags are omitted entirely rather than
passed with a default, so an ineligible config produces exactly the command a
hand-written run would.

The joint space is far too large to enumerate (185 topologies × 38,880 policy
combinations × 46,655 scorer profiles × the flow-control expansion), which is
why the algorithm is sampling plus convergence detection rather than a sweep.
Budget ≥ 800 is recommended; convergence typically lands around eval 40–60 and
the remaining budget refines coverage across GPU tiers and P/D topologies.

### Custom Search Spaces

For models with non-standard requirements (e.g. large MoE models needing
TP≥16), write a custom YAML or a self-contained Python script. See
`../../blis-search-sim2real/experiments/` for examples with GLM-5.2-FP8.

## CLI Reference

| Flag | Default | Description |
|------|---------|-------------|
| `--model` | (required) | LLM model name |
| `--hardware` | (required) | GPU type (H100, A100-80, etc.) |
| `--workload` | (required†) | Workload preset (chatbot, summarization, contentgen, multidoc) |
| `--rate` | (required†) | Request arrival rate (req/s) |
| `--num-requests` | 300† | Requests per evaluation |
| `--seed` | 42 | BLIS simulation seed |
| `--budget` | 500 | Maximum evaluations |
| `--workers` | 8 | Parallel evaluation workers |
| `--search-seed` | 1 | Search RNG seed |
| `--search-space` | defaults.yaml | YAML search space file |
| `--strategy` | diversity | Phase 2: `diversity` or `random` (no Phase 2) |
| `--convergence-k` | 30 | Convergence window size |
| `--convergence-epsilon` | 0.001 | HV improvement threshold |
| `--eval-timeout` | 20.0 | Per-eval timeout (seconds) |
| `--objectives` | (from YAML) | Objectives as `metric:direction,...` |
| `--slo` | (none) | SLO constraints as `metric<value,...` (must be objective metrics) |
| `--output` | stdout | Output file path |
| `--output-all` | false | Also emit every evaluated config, not just the front |
| `--blis-binary` | ./blis | Path to BLIS binary |
| `--latency-model` | trained-physics | Latency model backend |
| `--verbose` | false | Print progress to stderr |

† **Trace mode.** When the search space declares a `--trace-header` parameter, the
workload is a captured TraceV2 corpus replayed through `blis replay` (see
`weka-qwen3-14b.yaml` and `blis-replay-shim.sh`), arrivals come entirely from the
trace, and these three flags are optional — omit them. Omitted, they are absent
from the generated command line and from the output's `spec` block, so results do
not claim a preset the run never used. Any other space still requires
`--workload` and `--rate`; leaving them out is a hard argparse error naming the
space file.

## Output Format

One JSON object. `config` is the full sampled config including inactive
subsystem parameters; `metrics` holds only the objective metrics plus
`gpus_used`. Excerpt from the three-objective full-stack run above:

```json
{
  "pareto_front": [
    {
      "config": {
        "tp": 8, "replicas": 2, "gpus_used": 16,
        "prefill_instances": 1, "decode_instances": 1,
        "pd_decider": "always", "pd_transfer_bandwidth": 25.0,
        "scheduler": "fcfs", "max_batch": 512, "max_scheduled_tokens": 8192,
        "long_prefill": 4096, "block_size": 32, "gpu_mem_util": 0.95,
        "preemption_policy": "fcfs", "snapshot_refresh": 50000,
        "routing_policy": "weighted", "admission_policy": "tier-shed",
        "scorer_profile": "queue-depth:4,kv-utilization:4,precise-prefix-cache:4,active-requests:5,load-balance:3",
        "flow_control": true, "saturation_detector": "utilization",
        "queue_depth_threshold": 20, "kv_util_threshold": 0.9,
        "dispatch_order": "slo-deadline", "max_gateway_queue_depth": 500,
        "request_ttl": 10000000
      },
      "metrics": {
        "responses_per_sec": 169.05890654727952,
        "ttft_p99_ms": 26.58415,
        "gpus_used": 16,
        "completed_requests": 2000,
        "still_queued": 0,
        "still_running": 0,
        "injected_requests": 2000,
        "total_input_tokens": 497382,
        "total_output_tokens": 495611,
        "tokens_per_sec": 41859.1,
        "e2e_mean_ms": 1204.3,
        "e2e_p90_ms": 1811.2,
        "e2e_p95_ms": 1922.7,
        "e2e_p99_ms": 2140.5,
        "ttft_mean_ms": 18.9,
        "ttft_p90_ms": 24.1,
        "ttft_p95_ms": 25.4,
        "itl_mean_ms": 4.8,
        "itl_p90_ms": 5.2,
        "itl_p95_ms": 5.4,
        "itl_p99_ms": 6.0,
        "scheduling_delay_p99_ms": 11.7,
        "preemption_count": 0,
        "dropped_unservable": 0,
        "length_capped_requests": 0,
        "timed_out_requests": 0
      }
    }
  ],
  "search_stats": {
    "budget_used": 1000,
    "valid_evals": 1000,
    "convergence_eval": 31,
    "wall_time_seconds": 281.0,
    "final_hv": 0.0,
    "strategy": "diversity",
    "phase1_evals": 31,
    "phase2_evals": 969
  },
  "objectives": [ ... ],
  "spec": { "model": "...", "hardware": "...", "workload": "...", "rate": "200" },
  "param_flags": { "replicas": "--num-instances", "scheduler": "--scheduler", ... },
  "space_file": "full-stack.yaml",
  "all_results": [ ... ],   // only with --output-all
  "slo_feasible": [ ... ],  // only with --slo
  "slo_constraints": [ ... ]
}
```

`final_hv: 0.0` is this run being three-objective, not an empty front — see
[Sharp Edges](#sharp-edges). Inactive subsystem parameters appear as `null` in
`config` (e.g. `scorer_profile: null` when routing is not `weighted`) and are
not passed to BLIS.

**`metrics` carries every metric BLIS reported for that config**, not just the
search objectives — the objectives are listed first, then the rest of BLIS's
`--metrics-path` JSON in the order BLIS emits it. Only two things are dropped:
the per-request `requests` array (bulk data, not a metric) and non-numeric
metadata (`instance_id`). So `--slo` and post-hoc analysis can reference any
BLIS metric, whether or not it is an objective, and a front entry showing
`completed_requests: 0` is visibly a degenerate config (e.g. `reject-all`
admission scoring a free `ttft_p99_ms: 0`) rather than a real winner. The
objectives alone drive Pareto extraction and hypervolume; recording the rest
changes no search behavior.

**`param_flags` and `space_file` make the output self-describing.**
`param_flags` is the knob → CLI flag map the search used, and `space_file` is
the basename of the space YAML it came from. They exist because knob names do
not map mechanically to flags — `replicas` is `--num-instances`,
`scorer_profile` is `--routing-scorers`, `flow_control` is a bare boolean — so
without the map a reader cannot reconstruct the `blis run` command that
produced a front entry. `viz/` uses them to print exact deploy commands with no
extra arguments; files predating this can pass `--space` instead. Both keys are
optional on read.

## Visualization

```bash
../../.venv/bin/python viz/cli.py search_output.json
```

Writes `search_output-viz/` containing `index.html` (self-contained, opens
offline, interactive), `pareto.png`, `knobs.png`, `parallel.png` at 3+
objectives, and `picks.md` — a markdown picks table with pasteable `blis run`
commands for the best config per objective, the knee, and the cheapest
topology.

The report recomputes the Pareto front from `all_results` and raises a banner if
it disagrees with the reported one, states which SLO pool the picks came from,
and ranks the configuration knobs by how differently the front uses them
compared to the dominated cloud. Every choice it makes is captioned on the page.

For files produced before `param_flags` existed, pass `--space full-stack.yaml`
so the deploy commands can be built; without it the picks table shows the config's
metrics and a note in place of the command.

Flags, output details, and the full set of analysis rules: [`viz/README.md`](viz/README.md).

## Available Metrics

Any metric from BLIS `--metrics-path` JSON can be used as an objective or SLO,
plus `gpus_used`, which the search derives from the topology rather than
reading from BLIS:

| Metric | JSON key | Description |
|--------|----------|-------------|
| Throughput | `responses_per_sec` | Completed requests per second |
| Token throughput | `tokens_per_sec` | Output tokens per second |
| TTFT P99 | `ttft_p99_ms` | Time to first token (P99) |
| E2E P99 | `e2e_p99_ms` | End-to-end latency (P99) |
| ITL mean | `itl_mean_ms` | Inter-token latency (mean) |
| Scheduling delay P99 | `scheduling_delay_p99_ms` | Queue wait time (P99) |
| Goodput | `goodput_rps` | Requests meeting SLO per second |
| Preemptions | `preemption_count` | Preempted requests |
| KV failures | `kv_allocation_failures` | KV cache allocation failures |
| GPU count | `gpus_used` | GPUs the topology occupies (`tp × instances`) — derived |

A config whose BLIS run fails, times out, or omits a requested metric is
discarded (it counts against `--budget` but not `valid_evals`).

## Performance

Each BLIS evaluation is a full CPU-only simulation; its cost scales with
`--num-requests` and with the size of the topology being simulated. Measured on
an 8-worker M-series laptop:

| Space | Workload | `--num-requests` | Budget | Wall time |
|---|---|---|---|---|
| `defaults.yaml` | chatbot @ 200 req/s | 300 | 500 | 14s |
| `defaults.yaml` | summarization @ 1000 req/s | 300 | 500 | 24s |
| `full-stack.yaml` | chatbot @ 200 req/s | 2000 | 300 | 21s |
| `full-stack.yaml` | summarization @ 50 req/s | 500 | 1000 | 207s |
| `full-stack.yaml` | chatbot @ 200 req/s | 2000 | 1000 | 282–324s |

Wall time per evaluation is not stable across runs at equal budget: Phase 2
deliberately targets under-represented GPU tiers, and a 16-GPU topology
simulates far more slowly than a 1-GPU one, so a run that spends its Phase 2
budget on large topologies costs several times one that does not. Read the table
as an order of magnitude, not a rate.

Convergence typically occurs within 30–60 evaluations for common workloads.
The remaining budget refines coverage across cost classes.

## File Structure

```
tools/blis-search/
├── search.py          # CLI entry point + two-phase search orchestrator
├── search_space.py    # YAML space loading, topology enumeration, samplers
├── evaluator.py       # BLIS subprocess invocation + parallel batch eval
├── pareto.py          # Pareto front extraction + 2D hypervolume
├── convergence.py     # HV convergence detection + cost-class tracking
├── defaults.yaml      # Legacy space (single-node, TP≤8, no P/D)
├── full-stack.yaml    # Infrastructure space (16 GPUs, P/D, flow control)
└── viz/               # Reports from a results file (see viz/README.md)
    ├── cli.py         # Flags, out-dir handling, orchestration
    ├── data.py        # Schema validation, normalisation, columnar encoding
    ├── analysis.py    # Every decision: front check, axes, knee, picks, knobs
    ├── commands.py    # param_flags resolution + blis run construction
    ├── render_mpl.py  # pareto.png, knobs.png, parallel.png
    ├── render_md.py   # picks.md
    ├── render_html.py # index.html
    ├── assets/        # style.css and app.js, inlined at generation time
    └── tests/         # Fixtures, generator, and the test suite
```
