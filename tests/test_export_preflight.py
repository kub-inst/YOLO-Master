import torch

from ultralytics.nn.modules.mot import MoTBlock
from ultralytics.nn.peft.molora.layer import MoLoRALayer
from ultralytics.utils.export_preflight import export_preflight


def test_mixture_export_preflight_selects_dense_fallback():
    model = torch.nn.Sequential(MoTBlock(16, num_heads=2, top_k=1), MoLoRALayer(torch.nn.Linear(16, 16), r=2, num_experts=2, top_k=1))
    report = export_preflight(model, "onnx", strict=True)
    assert report["supported"] is True
    assert all(item["strategy"] == "dense_fallback" for item in report["decisions"])


def test_export_preflight_reports_safe_eager_strategy():
    model = MoTBlock(16, num_heads=2, top_k=1)
    report = export_preflight(model, "pytorch", strict=True)
    assert report["decisions"][0]["strategy"] == "dynamic"
