from pathlib import Path
import re

import streamlit as st

from industrial_ai.demo.graph_workflow import run_investigation_graph
from industrial_ai.demo.investigation import run_investigation
from industrial_ai.evaluation.test_rig import RigReport, run_rig
from industrial_ai.paths import MVTEC_DATASET_DIR
from industrial_ai.plant.stream import (
    DEFAULT_EVENT_STREAM_PATH,
    PlantEvent,
    PlantStreamState,
    generate_and_store_demo_events,
    process_event,
    read_events_jsonl,
)
from industrial_ai.policy.severity import SeverityDecision, severity_policy, severity_policy_rules
from industrial_ai.rag.answer import deterministic_metadata, test_ollama_connection
from industrial_ai.telemetry.predict import TelemetryReading


VISION_UPLOAD_DIR = Path("/tmp/industrial_ai_vision_uploads")
VISION_CATEGORIES = ["cable", "grid", "metal_nut", "screw", "transistor"]
EVALUATION_DATA_SOURCES = [
    "AI4I held-out test split",
    "MVTec test images",
    "Severity policy scenarios",
    "JSON approval workflow",
]
PLANT_STREAM_RAG_MODE = "ollama"
SAFE_FILENAME_PATTERN = re.compile(r"[^A-Za-z0-9_.-]+")
ALLOWED_UPLOAD_SUFFIXES = {".png", ".jpg", ".jpeg"}
DEFAULT_TELEMETRY_INPUTS = {
    "machine_id": "FAN-023",
    "machine_type": "M",
    "air_temperature_k": 301.1,
    "process_temperature_k": 311.6,
    "rotational_speed_rpm": 1266.0,
    "torque_nm": 55.5,
    "tool_wear_min": 210.0,
    "score_threshold": 0.5,
    "rag_mode_label": "Local Ollama",
    "vision_enabled": False,
    "vision_category": "cable",
    "vision_method": "auto",
}
DEMO_SCENARIOS = {
    "tool_wear_failure": {
        "label": "Injected Tool Wear Failure",
        "machine_id": "DEMO-TOOL-WEAR",
        "machine_type": "M",
        "air_temperature_k": 301.1,
        "process_temperature_k": 311.6,
        "rotational_speed_rpm": 1266.0,
        "torque_nm": 55.5,
        "tool_wear_min": 230.0,
        "vision_enabled": False,
        "vision_category": "cable",
        "vision_method": "auto",
        "vision_image_path": None,
    },
    "power_failure": {
        "label": "Injected Power Failure",
        "machine_id": "DEMO-POWER",
        "machine_type": "L",
        "air_temperature_k": 299.8,
        "process_temperature_k": 309.4,
        "rotational_speed_rpm": 1420.0,
        "torque_nm": 64.0,
        "tool_wear_min": 180.0,
        "vision_enabled": False,
        "vision_category": "cable",
        "vision_method": "auto",
        "vision_image_path": None,
    },
    "cooling_failure": {
        "label": "Injected Cooling Failure",
        "machine_id": "DEMO-COOLING",
        "machine_type": "H",
        "air_temperature_k": 304.5,
        "process_temperature_k": 318.2,
        "rotational_speed_rpm": 1380.0,
        "torque_nm": 48.0,
        "tool_wear_min": 120.0,
        "vision_enabled": False,
        "vision_category": "cable",
        "vision_method": "auto",
        "vision_image_path": None,
    },
    "visual_defect": {
        "label": "Injected Visual Defect",
        "machine_id": "DEMO-VISION",
        "machine_type": "M",
        "air_temperature_k": 300.5,
        "process_temperature_k": 310.2,
        "rotational_speed_rpm": 1500.0,
        "torque_nm": 42.0,
        "tool_wear_min": 90.0,
        "vision_enabled": True,
        "vision_category": "cable",
        "vision_method": "auto",
        "vision_image_path": MVTEC_DATASET_DIR / "cable/test/cut_outer_insulation/000.png",
    },
    "multi_modal_sev1": {
        "label": "Injected Multi-Modal SEV1",
        "machine_id": "DEMO-SEV1",
        "machine_type": "M",
        "air_temperature_k": 301.1,
        "process_temperature_k": 311.6,
        "rotational_speed_rpm": 1266.0,
        "torque_nm": 55.5,
        "tool_wear_min": 230.0,
        "vision_enabled": True,
        "vision_category": "cable",
        "vision_method": "auto",
        "vision_image_path": MVTEC_DATASET_DIR / "cable/test/cut_outer_insulation/000.png",
    },
}


def safe_filename_part(value: str, fallback: str) -> str:
    cleaned = SAFE_FILENAME_PATTERN.sub("_", value).strip("._")
    return cleaned or fallback


def build_vision_upload_path(
    machine_id: str,
    vision_category: str,
    uploaded_filename: str,
    upload_dir: Path = VISION_UPLOAD_DIR,
) -> Path:
    suffix = Path(uploaded_filename).suffix.lower()
    if suffix not in ALLOWED_UPLOAD_SUFFIXES:
        suffix = ".png"
    filename = (
        f"{safe_filename_part(machine_id, 'machine')}-"
        f"{safe_filename_part(vision_category, 'category')}{suffix}"
    )
    return upload_dir / filename


def initialize_demo_state() -> None:
    for key, value in DEFAULT_TELEMETRY_INPUTS.items():
        st.session_state.setdefault(key, value)
    st.session_state.setdefault("injected_scenario_label", None)
    st.session_state.setdefault("injected_vision_image_path", None)


def apply_demo_scenario(scenario_key: str) -> None:
    scenario = DEMO_SCENARIOS[scenario_key]
    for key in [
        "machine_id",
        "machine_type",
        "air_temperature_k",
        "process_temperature_k",
        "rotational_speed_rpm",
        "torque_nm",
        "tool_wear_min",
        "vision_enabled",
        "vision_category",
        "vision_method",
    ]:
        st.session_state[key] = scenario[key]
    st.session_state["score_threshold"] = 0.3
    st.session_state["injected_scenario_label"] = scenario["label"]
    image_path = scenario["vision_image_path"]
    st.session_state["injected_vision_image_path"] = str(image_path) if image_path and image_path.exists() else None


def render_demo_controls() -> bool:
    st.subheader("Demo Controls")
    left, middle, right = st.columns(3)
    with left:
        if st.button("Inject Tool Wear Failure"):
            apply_demo_scenario("tool_wear_failure")
        if st.button("Inject Power Failure"):
            apply_demo_scenario("power_failure")
    with middle:
        if st.button("Inject Cooling Failure"):
            apply_demo_scenario("cooling_failure")
        if st.button("Inject Visual Defect"):
            apply_demo_scenario("visual_defect")
    with right:
        if st.button("Inject Multi-Modal SEV1"):
            apply_demo_scenario("multi_modal_sev1")

    scenario_label = st.session_state.get("injected_scenario_label")
    if scenario_label:
        st.info(f"Loaded scenario: {scenario_label}")
        injected_image = st.session_state.get("injected_vision_image_path")
        if injected_image:
            st.write(f"Injected visual image: `{Path(injected_image).name}`")
        elif st.session_state.get("vision_enabled"):
            st.warning("No injected visual image was found locally; scenario will run without image input.")
    return st.button("Run Injected Scenario", disabled=not bool(scenario_label))


def rag_metadata_display(metadata) -> dict:
    return {
        "rag_mode": getattr(metadata, "rag_mode", getattr(metadata, "mode", "deterministic")),
        "llm_provider": getattr(metadata, "llm_provider", getattr(metadata, "provider", None)),
        "llm_model": getattr(metadata, "llm_model", getattr(metadata, "model_name", None)),
        "endpoint_url": getattr(metadata, "endpoint_url", None),
        "fallback_used": getattr(metadata, "fallback_used", False),
        "fallback_reason": getattr(metadata, "fallback_reason", None),
        "latency_ms": getattr(metadata, "latency_ms", None),
        "raw_error": getattr(metadata, "raw_error", getattr(metadata, "error_message", None)),
    }


def render_kpi_card(label: str, value: str) -> None:
    st.markdown(
        f"""
        <div style="
            border: 1px solid #3f4652;
            border-radius: 8px;
            padding: 18px 20px;
            min-height: 112px;
            background: #171a21;
        ">
            <div style="
                color: #aeb6c2;
                font-size: 0.82rem;
                font-weight: 700;
                text-transform: uppercase;
                letter-spacing: 0.04em;
                margin-bottom: 10px;
            ">{label}</div>
            <div style="
                color: #f7f9fc;
                font-size: 2rem;
                font-weight: 800;
                line-height: 1.1;
            ">{value}</div>
        </div>
        """,
        unsafe_allow_html=True,
    )


def render_evidence(items: list[str]) -> None:
    for item in items:
        st.write(f"- {item}")


def format_optional_score(value: float | None) -> str:
    return "n/a" if value is None else f"{value:.2f}"


def similar_incident_score_label(incident) -> str:
    if incident.combined_score is not None:
        return f"combined {format_optional_score(incident.combined_score)}"
    return f"score {format_optional_score(incident.score)}"


def similar_incident_score_breakdown(incident) -> list[str]:
    return [
        f"Vector score: {format_optional_score(incident.vector_score)}",
        f"Telemetry similarity: {format_optional_score(incident.telemetry_similarity_score)}",
        f"Combined score: {format_optional_score(incident.combined_score)}",
    ]


def similar_incident_metadata(incident) -> dict[str, str]:
    return {
        "document_type": incident.document_type,
        "machine_id": incident.machine_id,
        "failure_mode": incident.failure_mode or "n/a",
    }


def telemetry_comparison_table(incident) -> list[dict[str, str]]:
    rows = []
    for row in incident.telemetry_comparison:
        rows.append(
            {
                "signal": str(row["signal"]),
                "current": f"{row['current']} {row['unit']}",
                "incident": (
                    "n/a" if row["incident"] == "n/a" else f"{row['incident']} {row['unit']}"
                ),
                "similarity": format_optional_score(row.get("similarity")),
            }
        )
    return rows


def similar_incident_match_reasoning(incident) -> list[str]:
    return incident.match_reasons or ["No structured match reasons available."]


def render_approval_status(approval) -> None:
    status_label = (
        "Pending Review"
        if approval.status.value == "pending"
        else "Approved Automatically"
        if approval.status.value == "not_required"
        else approval.status.value.replace("_", " ").title()
    )
    approval_required = "Yes" if approval.approval_required else "No"
    st.markdown(
        f"""
        <div style="
            border: 1px solid #3f4652;
            border-radius: 8px;
            padding: 18px 20px;
            background: #171a21;
        ">
            <div style="
                color: #f7f9fc;
                font-size: 1rem;
                font-weight: 800;
                margin-bottom: 10px;
            ">Approval Status</div>
            <div style="
                width: 200px;
                height: 1px;
                background: #aeb6c2;
                margin-bottom: 14px;
            "></div>
            <div style="line-height: 1.9; color: #f7f9fc;">
                <div>Incident: <strong>{approval.incident_id}</strong></div>
                <div>Severity: <strong>{approval.severity.value}</strong></div>
                <div>Approval Required: <strong>{approval_required}</strong></div>
                <div>Status: <strong>{status_label}</strong></div>
            </div>
        </div>
        """,
        unsafe_allow_html=True,
    )


def severity_rule_rows() -> list[dict[str, str]]:
    return [
        {
            "severity": rule.severity.value,
            "criteria": rule.criteria,
            "approval_required": "Yes" if rule.approval_required else "No",
        }
        for rule in severity_policy_rules()
    ]


def policy_summary() -> dict[str, str]:
    policy = severity_policy()
    return {
        "name": policy.name,
        "version": policy.version,
        "last_modified": policy.last_modified,
    }


def render_severity_rules(decision: SeverityDecision) -> None:
    with st.expander("Severity Rules"):
        st.write("Policy rules")
        st.dataframe(severity_rule_rows(), use_container_width=True, hide_index=True)
        st.write("Triggered rule")
        st.write(f"Severity: **{decision.severity.value}**")
        st.write(f"Reason: {decision.reason}")
        st.write("Inputs used")
        st.json(decision.inputs)


def pass_rate(report: RigReport) -> float:
    return report.passed / report.total if report.total else 0.0


def format_key_inputs(details: dict) -> str:
    if not details:
        return ""
    priority_keys = [
        "probability",
        "risk_level",
        "heldout_target",
        "image_path",
        "score",
        "threshold",
        "confidence",
        "defect_type",
        "visual_defect_detected",
        "severity",
        "approval_required",
        "approval_status",
        "reason",
    ]
    parts = []
    for key in priority_keys:
        if key not in details:
            continue
        value = details[key]
        if isinstance(value, float):
            value = round(value, 4)
        if key == "image_path":
            value = Path(value).name
        parts.append(f"{key}={value}")
    return ", ".join(parts)


def evaluation_rows(report: RigReport, filter_status: str = "All") -> list[dict[str, str]]:
    rows = []
    for item in report.results:
        if filter_status == "Passed" and not item.passed:
            continue
        if filter_status == "Failed" and item.passed:
            continue
        rows.append(
            {
                "area": item.area,
                "scenario": item.scenario,
                "expected": item.expected,
                "actual": item.actual,
                "result": "Passed" if item.passed else "Failed",
                "key_inputs": format_key_inputs(item.details),
            }
        )
    return rows


def plant_event_rows(events: list[PlantEvent], processed_count: int = 0) -> list[dict[str, str]]:
    rows = []
    for index, event in enumerate(events):
        rows.append(
            {
                "status": "processed" if index < processed_count else "pending",
                "timestamp": event.timestamp,
                "machine_id": event.machine_id,
                "event_type": event.event_type.value,
                "event_id": event.event_id,
                "payload": event.payload,
            }
        )
    return rows


def plant_machine_rows(state: PlantStreamState) -> list[dict]:
    return [
        {
            "machine_id": machine.machine_id,
            "event_count": machine.event_count,
            "risk_level": machine.latest_risk_level,
            "failure_probability": machine.latest_failure_probability,
            "visual_defect": machine.latest_visual_defect_detected,
            "vision_category": machine.latest_vision_category,
            "operator_note": machine.latest_operator_note,
            "maintenance_note": machine.latest_maintenance_note,
        }
        for machine in state.machines.values()
    ]


def plant_trigger_rows(state: PlantStreamState) -> list[dict]:
    return [
        {
            "event_id": item.event_id,
            "machine_id": item.machine_id,
            "reason": item.reason,
            "severity": item.severity,
            "approval_status": item.approval_status,
            "incident_id": item.incident_id,
        }
        for item in state.triggered_investigations
    ]


def initialize_plant_stream_state() -> None:
    st.session_state.setdefault("plant_events", [])
    st.session_state.setdefault("plant_state", PlantStreamState())
    st.session_state.setdefault("plant_event_index", 0)


def reset_plant_stream(events: list[PlantEvent]) -> None:
    st.session_state["plant_events"] = events
    st.session_state["plant_state"] = PlantStreamState()
    st.session_state["plant_event_index"] = 0


def process_next_plant_event() -> None:
    events = st.session_state["plant_events"]
    index = st.session_state["plant_event_index"]
    if index >= len(events):
        return
    process_event(
        state=st.session_state["plant_state"],
        event=events[index],
        rag_mode=PLANT_STREAM_RAG_MODE,
    )
    st.session_state["plant_event_index"] = index + 1


def process_all_plant_events() -> None:
    while st.session_state["plant_event_index"] < len(st.session_state["plant_events"]):
        process_next_plant_event()


def render_investigation_tab() -> None:
    initialize_demo_state()
    run_injected = render_demo_controls()

    st.write("LLM runtime")
    if st.button("Test Ollama Connection"):
        with st.spinner("Testing Ollama"):
            check = test_ollama_connection()
        if check.success:
            st.success(
                f"Ollama responded with {check.model_name} at {check.endpoint_url} "
                f"in {check.latency_ms} ms."
            )
        else:
            st.error(
                f"Ollama check failed for {check.model_name} at {check.endpoint_url} "
                f"after {check.latency_ms} ms."
            )
            st.write(f"Error: {check.error_message}")

    with st.form("telemetry_form"):
        machine_id = st.text_input("Machine ID", key="machine_id")
        machine_type = st.selectbox("Machine Type", options=["L", "M", "H"], key="machine_type")

        left, right = st.columns(2)
        with left:
            air_temperature_k = st.number_input("Air Temperature [K]", key="air_temperature_k")
            process_temperature_k = st.number_input("Process Temperature [K]", key="process_temperature_k")
            rotational_speed_rpm = st.number_input("Rotational Speed [rpm]", key="rotational_speed_rpm")
        with right:
            torque_nm = st.number_input("Torque [Nm]", key="torque_nm")
            tool_wear_min = st.number_input("Tool Wear [min]", key="tool_wear_min")
            score_threshold = st.slider(
                "Retrieval Score Threshold",
                0.0,
                1.0,
                step=0.05,
                key="score_threshold",
            )
            rag_mode_label = st.selectbox(
                "RAG Answer Mode",
                options=["Deterministic", "Local Ollama"],
                key="rag_mode_label",
                help="Local Ollama uses OLLAMA_MODEL, defaulting to gemma3:4b, and falls back if unavailable.",
            )
            run_with_langgraph = st.checkbox("Run with LangGraph", value=False)

        vision_enabled = st.checkbox("Include visual inspection", key="vision_enabled")
        vision_image = None
        vision_category = st.session_state.get("vision_category")
        vision_method = st.session_state.get("vision_method")
        if vision_enabled:
            vision_category = st.selectbox("Vision Category", options=VISION_CATEGORIES, key="vision_category")
            vision_method = st.selectbox(
                "Vision Method",
                options=["auto", "resnet", "comparison"],
                key="vision_method",
                help="Auto uses a saved calibrated ResNet profile. Comparison is available manually.",
            )
            vision_image = st.file_uploader("Inspection Image", type=["png", "jpg", "jpeg"])

        submitted = st.form_submit_button("Run Investigation")

    if not submitted and not run_injected:
        return

    reading = TelemetryReading(
        machine_id=machine_id,
        type=machine_type,
        air_temperature_k=air_temperature_k,
        process_temperature_k=process_temperature_k,
        rotational_speed_rpm=rotational_speed_rpm,
        torque_nm=torque_nm,
        tool_wear_min=tool_wear_min,
    )

    vision_image_path = None
    if vision_enabled and vision_image is not None and vision_category is not None:
        VISION_UPLOAD_DIR.mkdir(parents=True, exist_ok=True)
        vision_image_path = build_vision_upload_path(
            machine_id=machine_id,
            vision_category=vision_category,
            uploaded_filename=vision_image.name,
        )
        vision_image_path.write_bytes(vision_image.getvalue())
    elif vision_enabled and st.session_state.get("injected_vision_image_path"):
        vision_image_path = Path(st.session_state["injected_vision_image_path"])

    with st.spinner("Running investigation"):
        try:
            workflow = run_investigation_graph if run_with_langgraph else run_investigation
            result = workflow(
                reading=reading,
                vision_image_path=vision_image_path,
                vision_category=vision_category,
                vision_method=vision_method,
                score_threshold=score_threshold,
                rag_mode="ollama" if rag_mode_label == "Local Ollama" else "deterministic",
            )
        except FileNotFoundError as exc:
            st.error(str(exc))
            return
    st.session_state["current_severity_decision"] = result.severity

    prediction = result.prediction
    rag_answer = result.rag_answer
    severity = result.severity
    approval = result.approval

    scenario_label = st.session_state.get("injected_scenario_label") if run_injected else None
    if scenario_label:
        st.subheader("Injected Scenario")
        st.write(f"Scenario: **{scenario_label}**")

    if result.agent_trace:
        st.subheader("Agent Trace")
        render_evidence(result.agent_trace)

    probability, risk, severity_col, approval_col = st.columns(4)
    with probability:
        render_kpi_card("Failure", f"{prediction.failure_probability_percent}%")
    with risk:
        render_kpi_card("Risk", prediction.risk_level.value)
    with severity_col:
        render_kpi_card("Severity", severity.severity.value)
    with approval_col:
        render_kpi_card("Approval", approval.status.value)
    st.write(f"Severity reason: {severity.reason}")
    render_severity_rules(severity)

    st.subheader("Combined Evidence")
    for evidence in result.evidence:
        with st.expander(f"{evidence.source.title()} | {evidence.status}"):
            st.write(evidence.summary)
            render_evidence(evidence.details)

    if result.vision:
        st.subheader("Visual Inspection")
        vision = result.vision
        vision_status = "DEFECT DETECTED" if vision.defect_detected else "NO DEFECT DETECTED"
        method_col, status_col, confidence_col, score_col = st.columns(4)
        with method_col:
            render_kpi_card("Vision Method", vision.method.upper())
        with status_col:
            render_kpi_card("Image Status", vision_status)
        with confidence_col:
            render_kpi_card("Vision Confidence", f"{int(vision.confidence * 100)}%")
        with score_col:
            render_kpi_card("Anomaly Score", f"{vision.anomaly_score:.3f}")
        st.write(f"Vision threshold: {vision.threshold:.3f}")
        if vision.defect_type:
            st.write(f"Detected defect type: **{vision.defect_type}**")
        if vision.image_path:
            localization = vision.localization
            if localization and localization.available:
                st.write("Defect localization")
                image_col, annotated_col = st.columns(2)
                with image_col:
                    st.image(vision.image_path, caption="Original image")
                with annotated_col:
                    st.image(localization.annotated_image_path, caption="Annotated image")
                heatmap_col, metadata_col = st.columns(2)
                with heatmap_col:
                    st.image(localization.heatmap_path, caption="Anomaly heatmap")
                with metadata_col:
                    label = (
                        "Approximate anomaly region"
                        if localization.confidence < 0.7
                        else "Localized anomaly region"
                    )
                    st.write(label)
                    st.write(
                        {
                            "anomaly_score": round(localization.top_anomaly_score, 4),
                            "localization_confidence": localization.confidence,
                            "bounding_box": localization.bounding_box,
                            "method": localization.method,
                        }
                    )
            elif vision.defect_detected:
                st.warning("Defect detected, but localization is unavailable for this image.")
            else:
                st.info("No defect detected; localization was not generated.")
        render_evidence(vision.evidence)

    st.subheader("Similar Incidents")
    if not result.similar_incidents:
        st.info("No relevant incidents found.")
    for incident in result.similar_incidents:
        with st.expander(f"{incident.title} | {similar_incident_score_label(incident)}"):
            st.write(similar_incident_metadata(incident))
            st.write(incident.body)
            st.write("Match reasoning")
            render_evidence(similar_incident_match_reasoning(incident))
            st.write("Score breakdown")
            render_evidence(similar_incident_score_breakdown(incident))
            telemetry_rows = telemetry_comparison_table(incident)
            if telemetry_rows:
                st.write("Telemetry comparison")
                st.dataframe(telemetry_rows, use_container_width=True, hide_index=True)
            st.write("Evidence")
            render_evidence(incident.evidence)

    st.subheader("Root Cause And Recommendation")
    st.write(f"Likely root cause: **{rag_answer.likely_root_cause}**")
    st.write(f"RAG confidence: **{rag_answer.confidence}**")
    st.write("RAG execution")
    metadata = getattr(rag_answer, "metadata", deterministic_metadata())
    st.write(rag_metadata_display(metadata))
    st.write(rag_answer.recommended_action)
    if rag_answer.limitations:
        st.write("Limitations")
        render_evidence(rag_answer.limitations)

    st.subheader("Approval")
    render_approval_status(approval)


def render_evaluation_tab() -> None:
    st.subheader("Evaluation")
    st.write("Data sources used")
    render_evidence(EVALUATION_DATA_SOURCES)

    filter_status = st.radio(
        "Scenario Filter",
        options=["All", "Passed", "Failed"],
        horizontal=True,
    )
    if st.button("Run Evaluation"):
        with st.spinner("Running evaluation"):
            st.session_state["evaluation_report"] = run_rig(categories=VISION_CATEGORIES)

    report = st.session_state.get("evaluation_report")
    if report is None:
        st.info("Run evaluation to view scenario results.")
        return

    total_col, passed_col, failed_col, rate_col = st.columns(4)
    with total_col:
        render_kpi_card("Scenarios", str(report.total))
    with passed_col:
        render_kpi_card("Passed", str(report.passed))
    with failed_col:
        render_kpi_card("Failed", str(report.failed))
    with rate_col:
        render_kpi_card("Pass Rate", f"{pass_rate(report) * 100:.1f}%")

    rows = evaluation_rows(report, filter_status)
    if not rows:
        st.info("No scenarios match the selected filter.")
        return
    st.dataframe(rows, use_container_width=True, hide_index=True)


def render_live_plant_stream_tab() -> None:
    st.subheader("Live Plant Stream")
    initialize_plant_stream_state()

    generate_col, replay_col, next_col, all_col = st.columns(4)
    with generate_col:
        if st.button("Generate Demo Events"):
            reset_plant_stream(generate_and_store_demo_events())
    with replay_col:
        if st.button("Replay JSONL"):
            reset_plant_stream(read_events_jsonl(DEFAULT_EVENT_STREAM_PATH))
    with next_col:
        if st.button("Process Next Event"):
            process_next_plant_event()
    with all_col:
        if st.button("Process All Events"):
            process_all_plant_events()

    st.write(f"Event stream: `{DEFAULT_EVENT_STREAM_PATH}`")
    events = st.session_state["plant_events"]
    state = st.session_state["plant_state"]
    processed_count = st.session_state["plant_event_index"]

    if not events:
        st.info("Generate or replay local JSONL events to start the simulated stream.")
        return

    total_col, processed_col, triggered_col = st.columns(3)
    with total_col:
        render_kpi_card("Events", str(len(events)))
    with processed_col:
        render_kpi_card("Processed", str(processed_count))
    with triggered_col:
        render_kpi_card("Triggered", str(len(state.triggered_investigations)))

    st.subheader("Incoming Events")
    st.dataframe(plant_event_rows(events, processed_count), use_container_width=True, hide_index=True)

    st.subheader("Machine State")
    machine_rows = plant_machine_rows(state)
    if machine_rows:
        st.dataframe(machine_rows, use_container_width=True, hide_index=True)
    else:
        st.info("No machine state yet.")

    st.subheader("Triggered Investigations")
    trigger_rows = plant_trigger_rows(state)
    if trigger_rows:
        st.dataframe(trigger_rows, use_container_width=True, hide_index=True)
    else:
        st.info("No investigations triggered yet.")


def render_policy_management_tab() -> None:
    st.subheader("Policy Management")
    summary = policy_summary()
    name_col, version_col, modified_col = st.columns(3)
    with name_col:
        render_kpi_card("Policy", summary["name"])
    with version_col:
        render_kpi_card("Version", summary["version"])
    with modified_col:
        render_kpi_card("Last Modified", summary["last_modified"])

    st.write("Active rules")
    st.dataframe(severity_rule_rows(), use_container_width=True, hide_index=True)

    decision = st.session_state.get("current_severity_decision")
    st.write("Current triggered rule")
    if decision is None:
        st.info("Run an investigation to see the currently triggered rule.")
    else:
        st.write(f"Severity: **{decision.severity.value}**")
        st.write(f"Reason: {decision.reason}")
        st.write("Inputs used")
        st.json(decision.inputs)

    if st.button("Edit Policy"):
        st.session_state["policy_edit_requested"] = True
    if st.session_state.get("policy_edit_requested"):
        st.info("Policy editing is simulated for this demo. Production policy remains read-only.")
        st.text_input("Policy Name", value=summary["name"], disabled=True)
        st.text_input("Version", value=summary["version"], disabled=True)
        st.dataframe(severity_rule_rows(), use_container_width=True, hide_index=True)


def main() -> None:
    st.set_page_config(page_title="Industrial AI Investigation", layout="wide")
    st.title("Industrial AI Investigation")

    investigation_tab, stream_tab, evaluation_tab, policy_tab = st.tabs(
        ["Investigation", "Live Plant Stream", "Evaluation", "Policy Management"]
    )
    with investigation_tab:
        render_investigation_tab()
    with stream_tab:
        render_live_plant_stream_tab()
    with evaluation_tab:
        render_evaluation_tab()
    with policy_tab:
        render_policy_management_tab()


if __name__ == "__main__":
    main()
