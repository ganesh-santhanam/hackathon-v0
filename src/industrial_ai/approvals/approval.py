import argparse
import json
from dataclasses import asdict, dataclass
from enum import StrEnum
from pathlib import Path

from industrial_ai.paths import APPROVALS_STORE_PATH
from industrial_ai.policy.severity import Severity, approval_required_for_severity_value


class ApprovalStatus(StrEnum):
    PENDING = "pending"
    APPROVED = "approved"
    REJECTED = "rejected"
    NOT_REQUIRED = "not_required"


@dataclass(frozen=True)
class ApprovalRecord:
    incident_id: str
    severity: Severity
    approval_required: bool
    status: ApprovalStatus


def approval_required_for_severity(severity: Severity) -> bool:
    return approval_required_for_severity_value(severity)


def initial_status_for_severity(severity: Severity) -> ApprovalStatus:
    if approval_required_for_severity(severity):
        return ApprovalStatus.PENDING
    return ApprovalStatus.NOT_REQUIRED


def load_store(store_path: Path = APPROVALS_STORE_PATH) -> dict[str, dict]:
    if not store_path.exists():
        return {}
    return json.loads(store_path.read_text(encoding="utf-8"))


def save_store(store: dict[str, dict], store_path: Path = APPROVALS_STORE_PATH) -> None:
    store_path.parent.mkdir(parents=True, exist_ok=True)
    store_path.write_text(json.dumps(store, indent=2, sort_keys=True), encoding="utf-8")


def record_to_dict(record: ApprovalRecord) -> dict[str, str | bool]:
    data = asdict(record)
    data["severity"] = record.severity.value
    data["status"] = record.status.value
    return data


def dict_to_record(data: dict) -> ApprovalRecord:
    return ApprovalRecord(
        incident_id=data["incident_id"],
        severity=Severity(data["severity"]),
        approval_required=bool(data["approval_required"]),
        status=ApprovalStatus(data["status"]),
    )


def create_approval(
    incident_id: str,
    severity: str,
    store_path: Path = APPROVALS_STORE_PATH,
) -> ApprovalRecord:
    severity_value = Severity(severity)
    record = ApprovalRecord(
        incident_id=incident_id,
        severity=severity_value,
        approval_required=approval_required_for_severity(severity_value),
        status=initial_status_for_severity(severity_value),
    )
    store = load_store(store_path)
    store[incident_id] = record_to_dict(record)
    save_store(store, store_path)
    return record


def get_approval(
    incident_id: str,
    store_path: Path = APPROVALS_STORE_PATH,
) -> ApprovalRecord:
    store = load_store(store_path)
    if incident_id not in store:
        raise KeyError(f"Approval record not found for incident: {incident_id}")
    return dict_to_record(store[incident_id])


def update_approval_status(
    incident_id: str,
    status: ApprovalStatus,
    store_path: Path = APPROVALS_STORE_PATH,
) -> ApprovalRecord:
    record = get_approval(incident_id, store_path)
    if not record.approval_required:
        raise ValueError(f"Approval is not required for incident: {incident_id}")

    updated = ApprovalRecord(
        incident_id=record.incident_id,
        severity=record.severity,
        approval_required=record.approval_required,
        status=status,
    )
    store = load_store(store_path)
    store[incident_id] = record_to_dict(updated)
    save_store(store, store_path)
    return updated


def approve_incident(incident_id: str, store_path: Path = APPROVALS_STORE_PATH) -> ApprovalRecord:
    return update_approval_status(incident_id, ApprovalStatus.APPROVED, store_path)


def reject_incident(incident_id: str, store_path: Path = APPROVALS_STORE_PATH) -> ApprovalRecord:
    return update_approval_status(incident_id, ApprovalStatus.REJECTED, store_path)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Manage JSON-backed human approvals.")
    parser.add_argument("--store-path", default=APPROVALS_STORE_PATH, type=Path)
    subparsers = parser.add_subparsers(dest="command", required=True)

    create_parser = subparsers.add_parser("create", help="Create or replace an approval record.")
    create_parser.add_argument("incident_id")
    create_parser.add_argument("--severity", required=True, choices=[item.value for item in Severity])

    show_parser = subparsers.add_parser("show", help="Show an approval record.")
    show_parser.add_argument("incident_id")

    approve_parser = subparsers.add_parser("approve", help="Approve an incident.")
    approve_parser.add_argument("incident_id")

    reject_parser = subparsers.add_parser("reject", help="Reject an incident.")
    reject_parser.add_argument("incident_id")
    return parser


def main() -> None:
    args = build_parser().parse_args()
    if args.command == "create":
        record = create_approval(args.incident_id, args.severity, args.store_path)
    elif args.command == "show":
        record = get_approval(args.incident_id, args.store_path)
    elif args.command == "approve":
        record = approve_incident(args.incident_id, args.store_path)
    elif args.command == "reject":
        record = reject_incident(args.incident_id, args.store_path)
    else:
        raise ValueError(f"Unsupported command: {args.command}")

    print(json.dumps(record_to_dict(record), indent=2))


if __name__ == "__main__":
    main()
