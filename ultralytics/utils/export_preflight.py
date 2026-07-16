"""Preflight checks for dynamic mixture and adapter export."""
from __future__ import annotations

from dataclasses import dataclass, asdict
from typing import Any

import torch.nn as nn

from ultralytics.nn.modules.routing_protocol import export_capabilities


@dataclass
class ExportDecision:
    module: str
    module_type: str
    backend: str
    supported: bool
    strategy: str
    dense_fallback: bool
    requires_merge: bool
    known_error: str | None = None


_DYNAMIC_UNSAFE = {"onnx", "engine", "ncnn", "mnn", "paddle", "executorch", "imx", "rknn"}


def _format_name(fmt: str) -> str:
    fmt = fmt.lower()
    return "engine" if fmt in {"tensorrt", "trt"} else fmt


def export_preflight(model: nn.Module, fmt: str, *, strict: bool = True) -> dict[str, Any]:
    """Scan routed modules and decide dynamic, dense fallback, or refusal."""

    fmt = _format_name(fmt)
    decisions: list[ExportDecision] = []
    errors: list[str] = []
    for name, module in model.named_modules():
        kind = ""
        if module.__class__.__name__ in {"MoABlock", "C2fMoA", "NeckMoAFusion"}:
            kind = "moa"
        elif module.__class__.__name__ in {"MoTBlock", "C2fMoT"}:
            kind = "mot"
        elif module.__class__.__name__ == "MoLoRALayer":
            kind = "molora"
        elif hasattr(module, "aux_loss") and hasattr(module, "last_routing_snapshot"):
            kind = "moe"
        if not kind:
            continue
        caps = export_capabilities(module)
        unsupported_dynamic = fmt in _DYNAMIC_UNSAFE and (
            bool(caps.get("dynamic_routing")) or bool(caps.get("sparse_dispatch"))
        )
        if unsupported_dynamic and caps.get("export_safe_dense_fallback", False):
            decision = ExportDecision(
                name or "<root>", type(module).__name__, fmt, True, "dense_fallback", True, False,
                "backend does not guarantee data-dependent routing control flow",
            )
        elif unsupported_dynamic:
            decision = ExportDecision(
                name or "<root>", type(module).__name__, fmt, False, "refuse", False, True,
                "dynamic routing has no declared safe fallback",
            )
            errors.append(f"{name or '<root>'}: {decision.known_error}")
        else:
            decision = ExportDecision(
                name or "<root>", type(module).__name__, fmt, True, "dynamic", False, False,
            )
        decisions.append(decision)
    report = {
        "format": fmt,
        "supported": not errors,
        "decisions": [asdict(item) for item in decisions],
        "errors": errors,
    }
    if errors and strict:
        raise RuntimeError("Export preflight failed: " + "; ".join(errors))
    return report


__all__ = ["ExportDecision", "export_preflight"]
