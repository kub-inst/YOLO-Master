"""Regression tests for deterministic AutoBackend warmup inputs."""

import torch

from ultralytics.nn.autobackend import AutoBackend


class _WarmupBackend:
    device = torch.device("cpu")
    fp16 = False
    pt = jit = onnx = engine = saved_model = pb = nn_module = False
    triton = True

    def __init__(self):
        self.inputs = []

    def forward(self, image):
        self.inputs.append(image.detach().clone())
        return torch.zeros(1, 84, 16)


def test_warmup_uses_finite_initialized_input():
    backend = _WarmupBackend()

    AutoBackend.warmup(backend, imgsz=(1, 3, 8, 8))

    assert len(backend.inputs) == 1
    assert torch.isfinite(backend.inputs[0]).all()
    assert torch.count_nonzero(backend.inputs[0]) == 0
