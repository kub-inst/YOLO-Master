from scripts.audit_moe_usage import audit_usage


def test_moe_usage_audit_classifies_every_canonical_class():
    rows = audit_usage()
    assert rows
    assert len({row.name for row in rows}) == len(rows)
    assert all(row.disposition in {"retain", "freeze", "archive-candidate"} for row in rows)


def test_yaml_referenced_classes_are_never_archive_candidates():
    assert all(row.disposition == "retain" for row in audit_usage() if row.yaml)
