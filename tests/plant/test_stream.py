from datetime import UTC, datetime
from types import SimpleNamespace

from industrial_ai.plant.stream import (
    PlantEvent,
    PlantEventType,
    PlantStreamState,
    event_from_dict,
    event_to_dict,
    generate_demo_events,
    process_event,
    read_events_jsonl,
    telemetry_payload,
    trigger_reason,
    write_events_jsonl,
)


def fake_investigation_runner(**kwargs):
    return SimpleNamespace(
        severity=SimpleNamespace(severity=SimpleNamespace(value="SEV1")),
        approval=SimpleNamespace(
            status=SimpleNamespace(value="pending"),
            incident_id=f"{kwargs['reading'].machine_id}-INVESTIGATION",
        ),
    )


def test_event_jsonl_round_trip(tmp_path):
    event = PlantEvent(
        event_id="evt-1",
        timestamp="2026-06-03T00:00:00+00:00",
        machine_id="M-001",
        event_type=PlantEventType.OPERATOR_NOTE,
        payload={"note": "mild vibration"},
    )
    path = tmp_path / "events.jsonl"

    write_events_jsonl([event], path)

    assert read_events_jsonl(path) == [event]
    assert event_from_dict(event_to_dict(event)) == event


def test_generate_demo_events_includes_required_event_types():
    events = generate_demo_events(base_time=datetime(2026, 6, 3, tzinfo=UTC))

    assert {event.event_type for event in events} == {
        PlantEventType.TELEMETRY,
        PlantEventType.VISION_INSPECTION,
        PlantEventType.OPERATOR_NOTE,
        PlantEventType.MAINTENANCE_NOTE,
    }


def test_trigger_reason_detects_high_risk_visual_and_operator_note():
    telemetry = PlantEvent(
        event_id="evt-telemetry",
        timestamp="2026-06-03T00:00:00+00:00",
        machine_id="M-001",
        event_type=PlantEventType.TELEMETRY,
        payload=telemetry_payload("M", 301.1, 311.6, 1266, 55.5, 230, 0.86, "HIGH"),
    )
    vision = PlantEvent(
        event_id="evt-vision",
        timestamp="2026-06-03T00:00:10+00:00",
        machine_id="M-001",
        event_type=PlantEventType.VISION_INSPECTION,
        payload={"defect_detected": True, "category": "cable"},
    )
    note = PlantEvent(
        event_id="evt-note",
        timestamp="2026-06-03T00:00:20+00:00",
        machine_id="M-001",
        event_type=PlantEventType.OPERATOR_NOTE,
        payload={"note": "urgent burning smell"},
    )

    assert trigger_reason(telemetry) == "High-risk telemetry event"
    assert trigger_reason(vision) == "Visual defect event"
    assert trigger_reason(note) == "Concerning operator note"


def test_process_event_updates_machine_state_and_triggers_investigation():
    state = PlantStreamState()
    event = PlantEvent(
        event_id="evt-telemetry",
        timestamp="2026-06-03T00:00:00+00:00",
        machine_id="M-001",
        event_type=PlantEventType.TELEMETRY,
        payload=telemetry_payload("M", 301.1, 311.6, 1266, 55.5, 230, 0.86, "HIGH"),
    )

    triggered = process_event(
        state=state,
        event=event,
        investigation_runner=fake_investigation_runner,
    )

    assert state.machines["M-001"].latest_risk_level == "HIGH"
    assert state.machines["M-001"].latest_failure_probability == 0.86
    assert triggered is not None
    assert triggered.reason == "High-risk telemetry event"
    assert triggered.severity == "SEV1"
    assert triggered.approval_status == "pending"


def test_process_event_ignores_non_triggering_maintenance_note():
    state = PlantStreamState()
    event = PlantEvent(
        event_id="evt-maint",
        timestamp="2026-06-03T00:00:00+00:00",
        machine_id="M-001",
        event_type=PlantEventType.MAINTENANCE_NOTE,
        payload={"note": "routine inspection complete"},
    )

    triggered = process_event(
        state=state,
        event=event,
        investigation_runner=fake_investigation_runner,
    )

    assert triggered is None
    assert state.machines["M-001"].latest_maintenance_note == "routine inspection complete"
    assert state.triggered_investigations == []
