"""Governance checks for the declared model/config registry."""

from pathlib import Path

import yaml


ROOT = Path(__file__).resolve().parents[1]
REGISTRY_PATH = ROOT / "docs/governance/model-registry.yaml"
REQUIRED_FIELDS = {"name", "path", "task", "status", "blocks", "verified", "export"}
VALID_STATUSES = {"stable", "experimental", "legacy", "blocked"}
VALID_EXPORT_STATUSES = {"unverified", "component_roundtrip", "full_model_roundtrip", "blocked"}


def test_model_registry_entries_are_well_formed_and_unique():
    registry = yaml.safe_load(REGISTRY_PATH.read_text(encoding="utf-8"))
    entries = registry["models"]

    assert registry["schema_version"] == 1
    assert "component_export_roundtrip" in registry["verification_policy"]
    assert entries
    assert len({entry["name"] for entry in entries}) == len(entries)

    for entry in entries:
        assert REQUIRED_FIELDS <= entry.keys()
        assert entry["status"] in VALID_STATUSES
        assert (ROOT / entry["path"]).is_file()
        assert isinstance(entry["blocks"], list)
        assert isinstance(entry["verified"], list)
        assert {"onnx", "tensorrt"} <= entry["export"].keys()
        assert set(entry["export"].values()) <= VALID_EXPORT_STATUSES


def test_component_export_evidence_is_not_reported_as_full_model_verification():
    registry = yaml.safe_load(REGISTRY_PATH.read_text(encoding="utf-8"))
    for entry in registry["models"]:
        if entry["export"]["onnx"] == "component_roundtrip":
            assert "component_export_roundtrip" in entry["verified"]
            assert "full_model_export_roundtrip" not in entry["verified"]


def test_registry_covers_phase1_model_smoke_configs():
    registry = yaml.safe_load(REGISTRY_PATH.read_text(encoding="utf-8"))
    paths = {entry["path"] for entry in registry["models"]}
    required = {
        "ultralytics/cfg/models/26/yolo26-master-n.yaml",
        "ultralytics/cfg/models/26/yolo26.yaml",
        "ultralytics/cfg/models/26/yolo26-seg.yaml",
        "ultralytics/cfg/models/26/yoloe-26.yaml",
        "ultralytics/cfg/models/26/yoloe-26-seg.yaml",
    }

    assert required <= paths
