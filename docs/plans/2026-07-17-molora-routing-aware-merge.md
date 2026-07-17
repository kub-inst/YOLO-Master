# MoLoRA Routing-Aware Merge Implementation Plan

> **For Claude:** REQUIRED SUB-SKILL: Use superpowers:executing-plans to implement this plan task-by-task.

**Goal:** Complete MoLoRA routing-aware merge with uniform, training-EMA, and calibration-data modes while preserving checkpoint and adapter backend compatibility.

**Architecture:** Each `MoLoRALayer` records the actual normalized top-k expert contribution used by the sparse forward path. `MoLoRAModel` orchestrates calibration forwards and produces independent weights for every wrapped layer; the unified adapter backend delegates to the same implementation for wrapper and raw-model callers.

**Tech Stack:** Python 3.11, PyTorch, pytest

---

### Task 1: Specify sparse usage and calibrated merge behavior

**Files:**
- Create: `tests/test_molora_routing_aware_merge.py`

**Step 1: Test top-k contribution EMA**

Force a two-expert router to choose one expert with `top_k=1`, set EMA decay to zero, run a training forward, and assert the stored EMA is exactly the selected sparse contribution rather than the dense softmax probability.

**Step 2: Test layer-specific calibration**

Wrap two linear layers, bias their routers toward different experts, calibrate with one batch, and assert each layer merges with its own observed expert weights.

**Step 3: Test calibration state restoration**

Start from training mode, calibrate and merge, and assert the model's prior training flags are restored after the no-grad calibration pass.

**Step 4: Test missing calibration rejection**

Call calibrated model merge without calibration data or explicit weights and assert a clear `ValueError`.

**Step 5: Test backend delegation**

Call `merge_adapters(..., mode="calibrated", calibration_data=...)` and assert merge metadata records calibrated weights.

**Step 6: Run the tests to verify they fail**

Run: `python -m pytest -q tests/test_molora_routing_aware_merge.py --tb=long`

Expected: FAIL because model-level calibration and sparse contribution EMA are not implemented.

### Task 2: Record actual sparse routing contribution

**Files:**
- Modify: `ultralytics/nn/peft/molora/layer.py`

**Step 1: Add weighted usage computation**

Scatter-add the final normalized `top_k_weights` into an expert vector using `top_k_indices`, then normalize by total contribution.

**Step 2: Move EMA updates after final routing decisions**

Update `_usage_ema` after expert dropout, top-k normalization, and capacity adjustment so it represents the expert mixture actually applied to outputs.

**Step 3: Add ephemeral calibration accumulation**

Add start, collect, finish, and cancel helpers. Calibration accumulators must not enter checkpoints or state dictionaries.

**Step 4: Preserve existing manual calibration weights**

Keep `merge_weights(mode="calibrated", calibration=[...])` working for direct layer callers, with finite/non-negative validation.

### Task 3: Add model-level calibration orchestration

**Files:**
- Modify: `ultralytics/nn/peft/molora/model.py`

**Step 1: Add calibration batch dispatch**

Support Tensor batches, positional tuple/list batches, keyword dict batches, and an optional `forward_fn(model, batch)` override.

**Step 2: Add `calibrate_merge_weights`**

Reject already merged layers, start all layer accumulators, switch to eval/no-grad, run up to `max_batches`, finish per-layer weights, and restore all original training flags in `finally`.

**Step 3: Extend `merge`**

Accept `calibration_data`, `calibration`, `max_batches`, and `forward_fn`. In calibrated mode, derive per-layer weights from data unless explicit weights were supplied.

**Step 4: Record calibration evidence**

Store calibration batch counts and source metadata in every layer's merge record.

### Task 4: Integrate backend and documentation

**Files:**
- Modify: `ultralytics/utils/lora/backend.py`
- Modify: `docs/molora_guide.md`

**Step 1: Delegate wrapper calibration through the backend**

When a `MoLoRAModel` is supplied, call its `merge` method with all calibration options.

**Step 2: Support raw models**

Use the shared calibration helper for raw models containing `MoLoRALayer` modules.

**Step 3: Document all three modes**

Document `uniform`, default `ema`, and recommended release-time `calibrated` usage, including the `forward_fn` escape hatch for YOLO dataloader dictionaries.

### Task 5: Verify compatibility

**Files:**
- Test: `tests/test_molora_routing_aware_merge.py`
- Test: `tests/test_molora.py`
- Test: `tests/test_molora_merge_semantics.py`
- Test: `tests/test_adapter_backend_contract.py`
- Test: `tests/test_mixture_export.py`

**Step 1: Run focused tests**

Run: `python -m pytest -q tests/test_molora_routing_aware_merge.py tests/test_molora.py tests/test_molora_merge_semantics.py tests/test_adapter_backend_contract.py --tb=long`

Expected: PASS.

**Step 2: Run export regressions**

Run: `python -m pytest -q tests/test_mixture_export.py -k molora --tb=long`

Expected: PASS or dependency skips only.

**Step 3: Run lint and compile checks**

Run: `ruff check ultralytics/nn/peft/molora/layer.py ultralytics/nn/peft/molora/model.py ultralytics/utils/lora/backend.py tests/test_molora_routing_aware_merge.py`

Run: `python -m compileall -q ultralytics/nn/peft/molora ultralytics/utils/lora/backend.py`

Expected: PASS.
