from pathlib import Path
import re

import streamlit as st

from industrial_ai.demo.investigation import run_investigation
from industrial_ai.evaluation.test_rig import RigReport, run_rig
from industrial_ai.policy.severity import SeverityDecision, severity_policy, severity_policy_rules
from industrial_ai.telemetry.predict import TelemetryReading


VISION_UPLOAD_DIR = Path("/tmp/industrial_ai_vision_uploads")
VISION_CATEGORIES = ["cable", "grid", "metal_nut", "screw", "transistor"]
EVALUATION_DATA_SOURCES = [
    "AI4I held-out test split",
    "MVTec test images",
    "Severity policy scenarios",
    "JSON approval workflow",
]
SAFE_FILENAME_PATTERN = re.compile(r"[^A-Za-z0-9_.-]+")
ALLOWED_UPLOAD_SUFFIXES = {".png", ".jpg", ".jpeg"}


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


def render_investigation_tab() -> None:
    with st.form("telemetry_form"):
        machine_id = st.text_input("Machine ID", value="FAN-023")
        machine_type = st.selectbox("Machine Type", options=["L", "M", "H"], index=1)

        left, right = st.columns(2)
        with left:
            air_temperature_k = st.number_input("Air Temperature [K]", value=301.1)
            process_temperature_k = st.number_input("Process Temperature [K]", value=311.6)
            rotational_speed_rpm = st.number_input("Rotational Speed [rpm]", value=1266.0)
        with right:
            torque_nm = st.number_input("Torque [Nm]", value=55.5)
            tool_wear_min = st.number_input("Tool Wear [min]", value=210.0)
            score_threshold = st.slider("Retrieval Score Threshold", 0.0, 1.0, 0.5, 0.05)
            rag_mode_label = st.selectbox(
                "RAG Answer Mode",
                options=["Deterministic", "Local Ollama"],
                help="Local Ollama uses OLLAMA_MODEL, defaulting to gemma3:4b, and falls back if unavailable.",
            )

        vision_enabled = st.checkbox("Include visual inspection", value=False)
        vision_image = None
        vision_category = None
        vision_method = "auto"
        if vision_enabled:
            vision_category = st.selectbox("Vision Category", options=VISION_CATEGORIES)
            vision_method = st.selectbox(
                "Vision Method",
                options=["auto", "resnet", "comparison"],
                help="Auto uses a saved calibrated ResNet profile. Comparison is available manually.",
            )
            vision_image = st.file_uploader("Inspection Image", type=["png", "jpg", "jpeg"])

        submitted = st.form_submit_button("Run Investigation")

    if not submitted:
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

    with st.spinner("Running investigation"):
        try:
            result = run_investigation(
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
        render_evidence(vision.evidence)

    st.subheader("Similar Incidents")
    if not result.similar_incidents:
        st.info("No relevant incidents found.")
    for incident in result.similar_incidents:
        score_label = (
            f"combined {incident.combined_score:.3f}"
            if incident.combined_score is not None
            else f"score {incident.score:.3f}"
        )
        with st.expander(f"{incident.title} | {score_label}"):
            st.write(incident.body)
            st.write(
                {
                    "vector_score": incident.vector_score,
                    "telemetry_similarity_score": incident.telemetry_similarity_score,
                    "combined_score": incident.combined_score,
                }
            )
            st.write("Evidence")
            render_evidence(incident.evidence)

    st.subheader("Root Cause And Recommendation")
    st.write(f"Likely root cause: **{rag_answer.likely_root_cause}**")
    st.write(f"RAG confidence: **{rag_answer.confidence}**")
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

    investigation_tab, evaluation_tab, policy_tab = st.tabs(
        ["Investigation", "Evaluation", "Policy Management"]
    )
    with investigation_tab:
        render_investigation_tab()
    with evaluation_tab:
        render_evaluation_tab()
    with policy_tab:
        render_policy_management_tab()


if __name__ == "__main__":
    main()
