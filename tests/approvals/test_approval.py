import pytest

from industrial_ai.approvals.approval import (
    ApprovalStatus,
    approve_incident,
    create_approval,
    get_approval,
    reject_incident,
)
from industrial_ai.policy.severity import Severity


def test_create_approval_requires_human_for_sev1(tmp_path):
    store_path = tmp_path / "approvals.json"

    record = create_approval("INCIDENT-001", "SEV1", store_path)

    assert record.incident_id == "INCIDENT-001"
    assert record.severity == Severity.SEV1
    assert record.approval_required is True
    assert record.status == ApprovalStatus.PENDING


def test_create_approval_marks_non_sev1_as_not_required(tmp_path):
    store_path = tmp_path / "approvals.json"

    record = create_approval("INCIDENT-002", "SEV2", store_path)

    assert record.approval_required is False
    assert record.status == ApprovalStatus.NOT_REQUIRED


def test_get_approval_reads_persisted_record(tmp_path):
    store_path = tmp_path / "approvals.json"
    create_approval("INCIDENT-001", "SEV1", store_path)

    record = get_approval("INCIDENT-001", store_path)

    assert record.status == ApprovalStatus.PENDING


def test_approve_incident_updates_pending_record(tmp_path):
    store_path = tmp_path / "approvals.json"
    create_approval("INCIDENT-001", "SEV1", store_path)

    record = approve_incident("INCIDENT-001", store_path)

    assert record.status == ApprovalStatus.APPROVED
    assert get_approval("INCIDENT-001", store_path).status == ApprovalStatus.APPROVED


def test_reject_incident_updates_pending_record(tmp_path):
    store_path = tmp_path / "approvals.json"
    create_approval("INCIDENT-001", "SEV1", store_path)

    record = reject_incident("INCIDENT-001", store_path)

    assert record.status == ApprovalStatus.REJECTED


def test_cannot_approve_incident_that_does_not_require_approval(tmp_path):
    store_path = tmp_path / "approvals.json"
    create_approval("INCIDENT-002", "SEV2", store_path)

    with pytest.raises(ValueError):
        approve_incident("INCIDENT-002", store_path)


def test_missing_approval_record_raises_key_error(tmp_path):
    with pytest.raises(KeyError):
        get_approval("INCIDENT-404", tmp_path / "approvals.json")
