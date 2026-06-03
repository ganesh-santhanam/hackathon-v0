import json
from dataclasses import asdict, dataclass, field
from datetime import UTC, datetime, timedelta
from enum import StrEnum
from pathlib import Path
from typing import Any, Callable

from industrial_ai.demo.investigation import InvestigationResult, run_investigation
from industrial_ai.paths import DATA_DIR, MVTEC_DATASET_DIR
from industrial_ai.telemetry.predict import TelemetryReading


PLANT_DATA_DIR = DATA_DIR / "plant"
DEFAULT_EVENT_STREAM_PATH = PLANT_DATA_DIR / "events.jsonl"
CONCERNING_OPERATOR_TERMS = (
    "alarm",
    "burning",
    "hot",
    "noise",
    "overheat",
    "smell",
    "smoke",
    "stop",
    "urgent",
    "vibration",
)


class PlantEventType(StrEnum):
    TELEMETRY = "telemetry"
    VISION_INSPECTION = "vision_inspection"
    OPERATOR_NOTE = "operator_note"
    MAINTENANCE_NOTE = "maintenance_note"


@dataclass(frozen=True)
class PlantEvent:
    event_id: str
    timestamp: str
    machine_id: str
    event_type: PlantEventType
    payload: dict[str, Any]


@dataclass
class MachineState:
    machine_id: str
    latest_telemetry: dict[str, Any] | None = None
    latest_failure_probability: float | None = None
    latest_risk_level: str | None = None
    latest_visual_defect_detected: bool = False
    latest_vision_category: str | None = None
    latest_operator_note: str | None = None
    latest_maintenance_note: str | None = None
    event_count: int = 0


@dataclass(frozen=True)
class TriggeredInvestigation:
    event_id: str
    machine_id: str
    reason: str
    severity: str
    approval_status: str
    incident_id: str


@dataclass
class PlantStreamState:
    machines: dict[str, MachineState] = field(default_factory=dict)
    processed_events: list[PlantEvent] = field(default_factory=list)
    triggered_investigations: list[TriggeredInvestigation] = field(default_factory=list)


InvestigationRunner = Callable[..., InvestigationResult]


def utc_timestamp(offset_seconds: int = 0) -> str:
    return (datetime.now(UTC) + timedelta(seconds=offset_seconds)).isoformat()


def event_to_dict(event: PlantEvent) -> dict[str, Any]:
    data = asdict(event)
    data["event_type"] = event.event_type.value
    return data


def event_from_dict(data: dict[str, Any]) -> PlantEvent:
    return PlantEvent(
        event_id=data["event_id"],
        timestamp=data["timestamp"],
        machine_id=data["machine_id"],
        event_type=PlantEventType(data["event_type"]),
        payload=dict(data["payload"]),
    )


def write_events_jsonl(events: list[PlantEvent], path: Path = DEFAULT_EVENT_STREAM_PATH) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        "\n".join(json.dumps(event_to_dict(event), sort_keys=True) for event in events) + "\n",
        encoding="utf-8",
    )


def append_event_jsonl(event: PlantEvent, path: Path = DEFAULT_EVENT_STREAM_PATH) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as event_file:
        event_file.write(json.dumps(event_to_dict(event), sort_keys=True) + "\n")


def read_events_jsonl(path: Path = DEFAULT_EVENT_STREAM_PATH) -> list[PlantEvent]:
    if not path.exists():
        return []
    return [
        event_from_dict(json.loads(line))
        for line in path.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]


def telemetry_payload(
    machine_type: str,
    air_temperature_k: float,
    process_temperature_k: float,
    rotational_speed_rpm: float,
    torque_nm: float,
    tool_wear_min: float,
    failure_probability: float,
    risk_level: str,
) -> dict[str, Any]:
    return {
        "type": machine_type,
        "air_temperature_k": air_temperature_k,
        "process_temperature_k": process_temperature_k,
        "rotational_speed_rpm": rotational_speed_rpm,
        "torque_nm": torque_nm,
        "tool_wear_min": tool_wear_min,
        "failure_probability": failure_probability,
        "risk_level": risk_level,
    }


def demo_visual_defect_path() -> Path | None:
    path = MVTEC_DATASET_DIR / "cable/test/cut_outer_insulation/000.png"
    return path if path.exists() else None


def generate_demo_events(base_time: datetime | None = None) -> list[PlantEvent]:
    base_time = base_time or datetime.now(UTC)
    visual_path = demo_visual_defect_path()

    def timestamp(index: int) -> str:
        return (base_time + timedelta(seconds=index * 10)).isoformat()

    return [
        PlantEvent(
            event_id="evt-001",
            timestamp=timestamp(0),
            machine_id="FAN-023",
            event_type=PlantEventType.TELEMETRY,
            payload=telemetry_payload("M", 300.2, 309.8, 1510, 40.0, 70, 0.18, "LOW"),
        ),
        PlantEvent(
            event_id="evt-002",
            timestamp=timestamp(1),
            machine_id="FAN-023",
            event_type=PlantEventType.OPERATOR_NOTE,
            payload={"note": "Operator reports mild vibration during startup."},
        ),
        PlantEvent(
            event_id="evt-003",
            timestamp=timestamp(2),
            machine_id="PUMP-007",
            event_type=PlantEventType.TELEMETRY,
            payload=telemetry_payload("M", 301.1, 311.6, 1266, 55.5, 230, 0.86, "HIGH"),
        ),
        PlantEvent(
            event_id="evt-004",
            timestamp=timestamp(3),
            machine_id="PUMP-007",
            event_type=PlantEventType.MAINTENANCE_NOTE,
            payload={"note": "Last tool replacement was delayed by one shift."},
        ),
        PlantEvent(
            event_id="evt-005",
            timestamp=timestamp(4),
            machine_id="CAM-014",
            event_type=PlantEventType.VISION_INSPECTION,
            payload={
                "category": "cable",
                "method": "auto",
                "defect_detected": True,
                "defect_type": "cut_outer_insulation",
                "image_path": str(visual_path) if visual_path else None,
            },
        ),
        PlantEvent(
            event_id="evt-006",
            timestamp=timestamp(5),
            machine_id="MOTOR-042",
            event_type=PlantEventType.OPERATOR_NOTE,
            payload={"note": "Burning smell near motor housing, urgent inspection requested."},
        ),
    ]


def generate_and_store_demo_events(path: Path = DEFAULT_EVENT_STREAM_PATH) -> list[PlantEvent]:
    events = generate_demo_events()
    write_events_jsonl(events, path)
    return events


def telemetry_reading_from_payload(machine_id: str, payload: dict[str, Any]) -> TelemetryReading:
    return TelemetryReading(
        machine_id=machine_id,
        type=str(payload.get("type", "M")),
        air_temperature_k=float(payload.get("air_temperature_k", 301.1)),
        process_temperature_k=float(payload.get("process_temperature_k", 311.6)),
        rotational_speed_rpm=float(payload.get("rotational_speed_rpm", 1266.0)),
        torque_nm=float(payload.get("torque_nm", 55.5)),
        tool_wear_min=float(payload.get("tool_wear_min", 210.0)),
    )


def is_concerning_operator_note(note: str) -> bool:
    normalized = note.lower()
    return any(term in normalized for term in CONCERNING_OPERATOR_TERMS)


def trigger_reason(event: PlantEvent) -> str | None:
    if event.event_type == PlantEventType.TELEMETRY:
        probability = float(event.payload.get("failure_probability", 0.0))
        risk_level = str(event.payload.get("risk_level", "")).upper()
        if probability >= 0.7 or risk_level == "HIGH":
            return "High-risk telemetry event"
    if event.event_type == PlantEventType.VISION_INSPECTION and bool(event.payload.get("defect_detected")):
        return "Visual defect event"
    if event.event_type == PlantEventType.OPERATOR_NOTE and is_concerning_operator_note(str(event.payload.get("note", ""))):
        return "Concerning operator note"
    return None


def update_machine_state(machine: MachineState, event: PlantEvent) -> None:
    machine.event_count += 1
    if event.event_type == PlantEventType.TELEMETRY:
        machine.latest_telemetry = dict(event.payload)
        machine.latest_failure_probability = float(event.payload.get("failure_probability", 0.0))
        machine.latest_risk_level = str(event.payload.get("risk_level", "")).upper()
    elif event.event_type == PlantEventType.VISION_INSPECTION:
        machine.latest_visual_defect_detected = bool(event.payload.get("defect_detected"))
        machine.latest_vision_category = event.payload.get("category")
    elif event.event_type == PlantEventType.OPERATOR_NOTE:
        machine.latest_operator_note = str(event.payload.get("note", ""))
    elif event.event_type == PlantEventType.MAINTENANCE_NOTE:
        machine.latest_maintenance_note = str(event.payload.get("note", ""))


def build_trigger_reading(state: PlantStreamState, event: PlantEvent) -> TelemetryReading:
    if event.event_type == PlantEventType.TELEMETRY:
        return telemetry_reading_from_payload(event.machine_id, event.payload)
    machine = state.machines[event.machine_id]
    if machine.latest_telemetry:
        return telemetry_reading_from_payload(event.machine_id, machine.latest_telemetry)
    return telemetry_reading_from_payload(event.machine_id, {})


def process_event(
    state: PlantStreamState,
    event: PlantEvent,
    investigation_runner: InvestigationRunner = run_investigation,
    rag_mode: str = "deterministic",
) -> TriggeredInvestigation | None:
    machine = state.machines.setdefault(event.machine_id, MachineState(machine_id=event.machine_id))
    update_machine_state(machine, event)
    state.processed_events.append(event)

    reason = trigger_reason(event)
    if reason is None:
        return None

    vision_image_path = None
    vision_category = None
    vision_method = "auto"
    if event.event_type == PlantEventType.VISION_INSPECTION:
        image_path = event.payload.get("image_path")
        if image_path:
            vision_image_path = Path(str(image_path))
        vision_category = event.payload.get("category")
        vision_method = str(event.payload.get("method", "auto"))

    result = investigation_runner(
        reading=build_trigger_reading(state, event),
        vision_image_path=vision_image_path,
        vision_category=vision_category,
        vision_method=vision_method,
        rag_mode=rag_mode,
    )
    triggered = TriggeredInvestigation(
        event_id=event.event_id,
        machine_id=event.machine_id,
        reason=reason,
        severity=result.severity.severity.value,
        approval_status=result.approval.status.value,
        incident_id=result.approval.incident_id,
    )
    state.triggered_investigations.append(triggered)
    return triggered


def process_events(
    events: list[PlantEvent],
    investigation_runner: InvestigationRunner = run_investigation,
    rag_mode: str = "deterministic",
) -> PlantStreamState:
    state = PlantStreamState()
    for event in events:
        process_event(
            state=state,
            event=event,
            investigation_runner=investigation_runner,
            rag_mode=rag_mode,
        )
    return state
