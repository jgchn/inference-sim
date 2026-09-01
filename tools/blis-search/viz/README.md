# viz — reports from a blis-search results file

Turns one `search.py` results JSON into a report directory: an interactive
self-contained page, three print-resolution figures, and a markdown picks table
with pasteable `blis run` commands.

```bash
cd tools/blis-search
../../.venv/bin/python viz/cli.py search_output.json
# wrote search_output-viz/pareto.png
# wrote search_output-viz/knobs.png
# wrote search_output-viz/picks.md
# wrote search_output-viz/index.html
```

No pip install, no network, no per-run code changes. The only dependency beyond
the standard library is matplotlib (already in `.venv`), and only for the PNGs.

## Outputs

| File | Contents |
|---|---|
| `index.html` | Self-contained interactive page: simple view, advanced panel behind a toggle |
| `pareto.png` | The hero chart at print resolution, caption baked in |
| `knobs.png` | Knob-attribution strips |
| `parallel.png` | Parallel coordinates — emitted at 3+ objectives (becomes the hero form at 4+) |
| `picks.md` | Markdown picks table with fenced `blis run` blocks, for pasting into a PR |

`index.html` opens offline: CSS, JavaScript, and data are all inlined, and the
page contains no external URL of any kind. A test asserts that.

## Flags

| Flag | Default | Meaning |
|---|---|---|
| `--out-dir DIR` | `<results-basename>-viz/` | Output directory |
| `--space FILE` | (none) | Space YAML, for results files predating `param_flags` |
| `--x M` / `--y M` | auto | Override the chosen axes |
| `--color M` | auto | Override the color dimension (any metric or knob) |
| `--max-points N` | unlimited | Grid-decimate the dominated cloud; the front is never sampled |
| `--png-only` / `--html-only` | both | Restrict outputs |

Written paths go to stdout; every warning goes to stderr. Identical input
produces byte-identical output, so a report can be regenerated and diffed.

## Input it reads

`search.py`'s output. Required: `pareto_front`, `objectives`, `spec`,
`search_stats`. Optional and used when present: `all_results` (without it there
is no dominated cloud, dominance cannot be verified, and the knob strips
degrade to "where the front sits in its own range"), `slo_feasible`,
`slo_constraints`, `param_flags`, `space_file`.

`search/pareto_search.py` emits an unrelated schema and is rejected with a hard
error naming the expected and found keys.

## What the tool will choose

Every rule is deterministic and captioned on the page. Nothing is a hidden
heuristic, so you can predict a report before generating it.

**Dominance is recomputed.** With `all_results` present, the front is
recomputed from `objectives` and compared to the reported `pareto_front`.
Agreement is silent; disagreement raises a banner and a stderr warning naming
the differing configs. The chart still draws the front as reported — a searcher
bug is surfaced, never laundered.

**Axes.** Objective 1 → x, objective 2 → y, oriented so better is toward the
lower-right, with the direction in each label (`responses_per_sec ↑ better`).
With 3+ objectives the pair maximising the count of distinct 2D positions wins;
ties break by the sum of per-axis standard deviations, then declaration order.
The caption names the choice. `--x` / `--y` override.

**Hero form.**

| Objectives | Hero |
|---|---|
| 1 | Best-config card, plus a distribution of that metric over the cloud |
| 2 | Scatter with staircase front, dominated cloud behind |
| 3 | Scatter of the widest-spread pair; the third objective on the color channel |
| 4+ | Parallel coordinates, best-pair scatter beneath |

**Color** is GPU count, for predictability across reports: `gpus_used` when
present, else `tp × replicas`, else no color channel. At exactly 3 objectives
the third objective takes the channel instead. When the plotted set has one
distinct GPU count the channel carries nothing — the caption says so and points
fall back to front/cloud coloring rather than a one-entry legend.

**Knee.** On coordinates normalised over the front members only, both oriented
as minimise, the knee is the front member furthest from the chord joining the
two extremes. Extremes are not candidates, so the knee is never an extreme.
Ties break to the lowest index. It is suppressed — with a printed reason — when
the front has fewer than 3 members, when the front is flat on an axis, or when
one member is best on both.

**Picks.** Best per objective, the knee, and lowest GPU count, drawn from the
SLO-feasible set when the file carries constraints and from the front otherwise;
the header states which pool. One config winning several categories is one row
listing all its titles. A row is omitted, and the omission listed, when it would
be an arbitrary tie-break (every candidate uses the same GPU count) or
misleading (the knee is not SLO-feasible).

**Knob attribution.** Per knob, one separation score in [0,1] comparing its
distribution on the front against the dominated cloud: total-variation distance
for categoricals (booleans and `null` are values, not missing data), and
`|mean_front − mean_cloud| / (max − min)` for numerics so both land on one
scale. A knob is numeric only if every value is a non-bool number; a missing key
counts as `null`. Strips are ranked by score, and each says whether the front
narrows the knob or spans its full searched range.

**Decimation.** `--max-points N` bins the normalised hero plane into `k × k`
with `k = ceil(sqrt(N))` and keeps the lowest-index point per occupied bin,
which preserves the cloud's envelope. The front is never sampled, and the page
states the sampling in its provenance strip.

## Deploy commands

The picks table prints the exact `blis run` invocation per pick, built through
`evaluator.build_command` — the same function the search itself uses. It needs
the knob → flag map, which does not follow mechanically from knob names
(`replicas` → `--num-instances`, `flow_control` is a bare boolean, flow-control
sub-params are suppressed when FC is off). Three tiers, in order:

1. the results file's own `param_flags` (emitted by `search.py`),
2. `--space FILE`, reconstructed with `get_all_param_flags`,
3. neither — the config is shown and the command is replaced by a note.

There is no fourth tier. A silently wrong deploy command is worse than no
command.

## Provenance and determinism

The page carries no generation timestamp: provenance is the input file's sha256
plus the searcher's own `wall_time_seconds`. Embedded JSON is key-sorted, and
the figures pin matplotlib's `rcParams` and write empty PNG metadata. Identical
input therefore yields byte-identical output, which is what makes the
regression tests meaningful.

## Tests

```bash
.venv/bin/python -m pytest tools/blis-search/viz/tests/ -q
```

The JavaScript is covered by `node --check` plus `tests/run_app.js`, a DOM stub
that runs the real script against a real generated page. Both skip when `node`
is absent.

### Regenerating the fixtures

Fixtures are generated, not hand-written:

```bash
.venv/bin/python tools/blis-search/viz/tests/make_fixtures.py
# add --from-real /path/to/search_output.json for the real-derived subsample
```
