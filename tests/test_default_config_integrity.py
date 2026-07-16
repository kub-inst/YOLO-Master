"""Configuration integrity tests for the public default YAML surface."""

from collections import Counter
from pathlib import Path
import re

import yaml

from ultralytics.nn.peft.molora.config import MoLoRAConfig


ROOT = Path(__file__).resolve().parents[1]
DEFAULT_PATH = ROOT / "ultralytics/cfg/default.yaml"


def _top_level_keys(text):
    return [match.group(1) for match in re.finditer(r"^([A-Za-z_][A-Za-z0-9_]*):", text, re.MULTILINE)]


def test_default_yaml_has_unique_top_level_keys():
    keys = _top_level_keys(DEFAULT_PATH.read_text(encoding="utf-8"))
    duplicates = {key: count for key, count in Counter(keys).items() if count > 1}
    assert not duplicates, f"Duplicate default config keys: {duplicates}"


def test_default_yaml_exposes_molora_from_args_fields():
    text = DEFAULT_PATH.read_text(encoding="utf-8")
    data = yaml.safe_load(text)
    mapping = {
        "router_hidden_dim": "molora_router_hidden_dim",
        "top_k_warmup": "molora_top_k_warmup",
        "domain_experts": "molora_domain_experts",
        "freeze_experts": "molora_freeze_experts",
    }

    assert all(key in data for key in mapping.values())
    assert set(mapping).issubset(MoLoRAConfig.__dataclass_fields__)


def test_default_yaml_exposes_mixture_runtime_override_fields():
    """Resolver CLI fields that affect MoA/MoT must exist in default.yaml."""
    data = yaml.safe_load(DEFAULT_PATH.read_text(encoding="utf-8"))

    assert data["moa_temperature"] == 1.0
    assert data["moa_aux_loss_coeff"] == 0.01
    assert data["mot_temperature"] == 1.0


def test_molora_from_args_preserves_explicit_empty_collections():
    """None disables optional routing features; empty collections stay explicit."""
    from types import SimpleNamespace

    args = SimpleNamespace(
        molora_num_experts=2,
        molora_top_k=1,
        molora_top_k_warmup=0,
        molora_domain_experts={},
        molora_freeze_experts=[],
    )
    config = MoLoRAConfig.from_args(args)

    assert config.top_k_warmup == 0
    assert config.domain_experts == {}
    assert config.freeze_experts == []


def test_molora_optional_defaults_are_none_or_safe_scalars():
    """Default optional collections do not accidentally activate a feature."""
    config = MoLoRAConfig.from_args()

    assert config.router_hidden_dim is None
    assert config.top_k_warmup is None
    assert config.domain_experts is None
    assert config.freeze_experts is None
