Validation passes. Here's a summary of iteration 5:

**Key finding:** The "rate-dependent crossover" (TP=8/1 at low rates, TP=4/2 at high rates) discovered in prior iterations was entirely an artifact of insufficient simulation duration (`num_requests=500`). At `nreq=2000`, **TP=4/2 wins universally** across all rates (500-5000) and both hardware (H100, A100-SXM). The batch=512 requirement (RP-C2-5) was compensating for short simulations, not providing independent value.

**Experiment design:** Tests whether lean Phase 1 (batch=128, nreq=2000) matches standard Phase 1 across all 18 condition×seed combinations, with control arms proving the transient regime (nreq=250) gives wrong results and that nreq is the dominant factor over batch size.