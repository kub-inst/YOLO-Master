"""Routing-aware MoLoRA merge tests."""

import pytest
import torch
import torch.nn as nn

from ultralytics.nn.peft.molora import MoLoRAConfig, MoLoRALayer, MoLoRAModel
from ultralytics.utils.lora import adapter_metadata, merge_adapters


def _set_router_bias(layer: MoLoRALayer, bias: list[float]) -> None:
    with torch.no_grad():
        layer.router.fc[-1].weight.zero_()
        layer.router.fc[-1].bias.copy_(torch.tensor(bias))


def _two_layer_model() -> MoLoRAModel:
    wrapper = MoLoRAModel(
        nn.Sequential(nn.Linear(2, 2, bias=False), nn.ReLU(), nn.Linear(2, 2, bias=False)),
        MoLoRAConfig(
            r=1,
            alpha=1,
            num_experts=2,
            top_k=1,
            use_rslora=False,
            target_modules=["0", "2"],
        ),
    )
    layers = [module for module in wrapper.model.modules() if isinstance(module, MoLoRALayer)]
    _set_router_bias(layers[0], [8.0, -8.0])
    _set_router_bias(layers[1], [-8.0, 8.0])
    return wrapper


def test_usage_ema_tracks_final_sparse_top_k_contribution():
    layer = MoLoRALayer(nn.Linear(2, 2), r=1, alpha=1, num_experts=2, top_k=1, use_rslora=False)
    layer.usage_ema_decay = 0.0
    _set_router_bias(layer, [2.0, -2.0])

    layer.train()
    layer(torch.randn(4, 2))

    assert torch.equal(layer._usage_ema, torch.tensor([1.0, 0.0]))


def test_calibrated_merge_collects_independent_weights_per_layer():
    wrapper = _two_layer_model().eval()

    result = wrapper.merge(mode="calibrated", calibration_data=[torch.ones(4, 2)])

    assert result["batches"] == 1
    layers = [module for module in wrapper.model.modules() if isinstance(module, MoLoRALayer)]
    assert layers[0]._merge_metadata["expert_weights"] == pytest.approx([1.0, 0.0])
    assert layers[1]._merge_metadata["expert_weights"] == pytest.approx([0.0, 1.0])
    assert all(layer._merge_metadata["calibration_batches"] == 1 for layer in layers)


def test_calibrated_merge_restores_training_flags():
    wrapper = _two_layer_model().train()
    wrapper.model[1].eval()
    original = {name: module.training for name, module in wrapper.named_modules()}

    wrapper.merge(mode="calibrated", calibration_data=[torch.ones(2, 2)])

    restored = {name: module.training for name, module in wrapper.named_modules()}
    assert restored == original


def test_calibrated_merge_requires_data_or_explicit_weights():
    wrapper = _two_layer_model()

    with pytest.raises(ValueError, match="calibration_data or explicit calibration weights"):
        wrapper.merge(mode="calibrated")


def test_invalid_per_layer_calibration_is_atomic():
    wrapper = _two_layer_model()
    layers = {name: module for name, module in wrapper.model.named_modules() if isinstance(module, MoLoRALayer)}
    first_name = next(iter(layers))

    with pytest.raises(ValueError, match="Missing explicit calibration weights"):
        wrapper.merge(mode="calibrated", calibration={first_name: [0.5, 0.5]})

    assert all(not layer.merged for layer in layers.values())


def test_calibrated_merge_supports_forward_override():
    wrapper = _two_layer_model().eval()
    batches = [{"images": torch.ones(2, 2)}]

    wrapper.merge(
        mode="calibrated",
        calibration_data=batches,
        forward_fn=lambda model, batch: model(batch["images"]),
    )

    assert all(
        module._merge_metadata["mode"] == "calibrated"
        for module in wrapper.model.modules()
        if isinstance(module, MoLoRALayer)
    )


def test_adapter_backend_delegates_calibrated_data_merge():
    wrapper = _two_layer_model().eval()

    assert merge_adapters(wrapper, mode="calibrated", calibration_data=[torch.ones(2, 2)])

    records = adapter_metadata(wrapper)["merge_records"]
    assert records and all(record["mode"] == "calibrated" for record in records)
    assert all(record["calibration_batches"] == 1 for record in records)
