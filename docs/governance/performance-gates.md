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

## Compile Gate

`compile=False` remains the default. Routed modules may be marked
`component_compile` only after they execute through the shared
`attempt_compile()` entrypoint and match eager output. A backend failure during
the first compiled forward must return the original eager module.

This marker does not claim a throughput improvement. Promotion requires a
hardware-specific benchmark showing positive p50 and p95 latency impact after
including initial compilation cost and recompilations.

## Export Gate

The registry distinguishes component and full-model evidence:

- `component_roundtrip`: representative routed blocks export, reload, execute,
  and match eager output for the tested shape.
- `full_model_roundtrip`: the complete registered model passes the same gate.
- `unverified`: no executable evidence exists for that scope/backend.

Legacy tracer warnings about Python shape/control-flow conversion prevent a
component result from being described as generally dynamic-shape safe.
