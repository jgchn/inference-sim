#!/bin/bash
# Experiment runner for iteration 5: Simulation Duration Confound
# Captures per-condition scores to JSON result files

set -e
RESULTS_DIR="/Users/jchen/go/src/inference-sim/inference-sim/.nous/campign-2/runs/iter-5/results"
BLIS="./blis"

# Helper: run one condition and save result
run_condition() {
    local arm=$1
    local name=$2
    local hw=$3
    local tp=$4
    local inst=$5
    local rate=$6
    local nreq=$7
    local batch=$8
    local seed=$9

    local routing="least-loaded"
    if [ "$inst" = "1" ]; then
        routing="round-robin"
    fi

    local output="$RESULTS_DIR/$arm/${name}.json"

    local stdout
    stdout=$($BLIS run --model qwen3-14b --hardware $hw --tp $tp --num-instances $inst \
        --rate $rate --num-requests $nreq --seed $seed \
        --latency-model trained-physics --scheduler fcfs \
        --max-num-running-reqs $batch --max-num-scheduled-tokens 8192 \
        --block-size-in-tokens 16 --routing-policy $routing \
        --admission-policy always-admit --preemption-policy fcfs \
        --fitness-weights "throughput:0.4,p99_ttft:0.3,p99_e2e:0.3" 2>/dev/null)

    local score
    score=$(echo "$stdout" | grep "Score:" | awk '{print $2}')

    cat > "$output" <<EOF
{
  "arm": "$arm",
  "name": "$name",
  "hardware": "$hw",
  "tp": $tp,
  "instances": $inst,
  "rate": $rate,
  "nreq": $nreq,
  "batch": $batch,
  "seed": $seed,
  "score": $score
}
EOF
    echo "$name: $score"
}

echo "=== Phase 2: Executing experiment conditions ==="
echo ""

# =============================================================
# h-main: Lean vs Standard Phase 1 at nreq=2000
# Rates: 500, 2000, 5000 | HW: H100, A100-SXM | Seeds: 42, 123, 456
# Tiers: TP=1/8, TP=2/4, TP=4/2, TP=8/1 | Batch: 128, 512
# =============================================================
echo "--- h-main ---"
mkdir -p "$RESULTS_DIR/h-main"

for hw in H100 A100-SXM; do
  for rate in 500 2000 5000; do
    for seed in 42 123 456; do
      for batch in 128 512; do
        for tp_inst in "1 8" "2 4" "4 2" "8 1"; do
          tp=$(echo $tp_inst | awk '{print $1}')
          inst=$(echo $tp_inst | awk '{print $2}')
          name="${hw}-r${rate}-tp${tp}i${inst}-b${batch}-s${seed}"
          run_condition "h-main" "$name" "$hw" "$tp" "$inst" "$rate" 2000 "$batch" "$seed"
        done
      done
    done
  done
done

echo ""
echo "--- h-control-negative ---"
mkdir -p "$RESULTS_DIR/h-control-negative"

# nreq=250 — same matrix
for hw in H100 A100-SXM; do
  for rate in 500 2000 5000; do
    for seed in 42 123 456; do
      for batch in 128 512; do
        for tp_inst in "1 8" "2 4" "4 2" "8 1"; do
          tp=$(echo $tp_inst | awk '{print $1}')
          inst=$(echo $tp_inst | awk '{print $2}')
          name="${hw}-r${rate}-tp${tp}i${inst}-b${batch}-s${seed}-nreq250"
          run_condition "h-control-negative" "$name" "$hw" "$tp" "$inst" "$rate" 250 "$batch" "$seed"
        done
      done
    done
  done
done

echo ""
echo "--- h-ablation ---"
mkdir -p "$RESULTS_DIR/h-ablation"

# H100, rate=500, seeds=42,123,456, all 4 batch×nreq combos, all 4 tiers
for seed in 42 123 456; do
  for batch_nreq in "128 500" "512 500" "128 2000" "512 2000"; do
    batch=$(echo $batch_nreq | awk '{print $1}')
    nreq=$(echo $batch_nreq | awk '{print $2}')
    for tp_inst in "1 8" "2 4" "4 2" "8 1"; do
      tp=$(echo $tp_inst | awk '{print $1}')
      inst=$(echo $tp_inst | awk '{print $2}')
      name="H100-r500-tp${tp}i${inst}-b${batch}-nreq${nreq}-s${seed}"
      run_condition "h-ablation" "$name" "H100" "$tp" "$inst" 500 "$nreq" "$batch" "$seed"
    done
  done
done

echo ""
echo "--- h-robustness ---"
mkdir -p "$RESULTS_DIR/h-robustness"

# Sweep nreq={250,500,750,1000,1500,2000} at H100 rate=500, batch=512, seed=42
for nreq in 250 500 750 1000 1500 2000; do
  for tp_inst in "1 8" "2 4" "4 2" "8 1"; do
    tp=$(echo $tp_inst | awk '{print $1}')
    inst=$(echo $tp_inst | awk '{print $2}')
    name="H100-r500-tp${tp}i${inst}-b512-s42-nreq${nreq}"
    run_condition "h-robustness" "$name" "H100" "$tp" "$inst" 500 "$nreq" 512 42
  done
done

echo ""
echo "=== All conditions complete ==="
