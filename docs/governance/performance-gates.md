# Mixture Performance Gates

Phase 6 sparse dispatch remains opt-in until measurements prove a benefit on
the target hardware. Run:

```bash
python benchmarks/benchmark_mot_dispatch.py --batch 16 --size 32
python benchmarks/benchmark_molora_dispatch.py --batch 16
```

Record p50/p95 latency, peak memory, throughput, expert calls, activated
experts, and model parity in `benchmarks/mixture_baselines.yaml`.

The default promotion gate is a measured 20% reduction in expert FLOPs with
maximum absolute output error below `1e-4`. If the gate is not met, retain the
grouped path as an explicit opt-in and keep the dense path as the default.
