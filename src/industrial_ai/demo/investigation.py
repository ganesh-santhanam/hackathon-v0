from dataclasses import asdict, dataclass
from pathlib import Path

from industrial_ai.approvals.approval import (
    ApprovalRecord,
    create_approval,
    record_to_dict,
)
from industrial_ai.incidents.memory import (
    DEFAULT_COLLECTION_NAME,
    DEFAULT_EMBEDDING_MODEL,
    DEFAULT_SCORE_THRESHOLD,
    Embedder,
    SearchResult,
    TelemetryQuery,
    load_embedder,
    retrieve_incidents,
)
from industrial_ai.paths import APPROVALS_STORE_PATH, QDRANT_DATA_DIR
from industrial_ai.policy.severity import SeverityDecision, assign_severity
from industrial_ai.rag.answer import RagAnswer, answer_to_dict, build_answer_from_results, build_llm_answer_from_results
from industrial_ai.telemetry.predict import FailurePrediction, TelemetryReading, predict_failure
from industrial_ai.vision.mvtec_compare import compare_mvtec_image
from industrial_ai.vision.mvtec_resnet import model_path_for_category, predict_resnet


DEFAULT_TOP_K = 3


@dataclass(frozen=True)
class VisionCheck:
    method: str
    category: str
    defect_detected: bool
    defect_type: str | None
    confidence: float
    anomaly_score: float
    threshold: float
    evidence: list[str]


@dataclass(frozen=True)
class EvidenceItem:
    source: str
    signal: str
    status: str
    summary: str
    details: list[str]


@dataclass(frozen=True)
class InvestigationResult:
    prediction: FailurePrediction
    vision: VisionCheck | None
    evidence: list[EvidenceItem]
    similar_incidents: list[SearchResult]
    rag_answer: RagAnswer
    severity: SeverityDecision
    approval: ApprovalRecord


def build_retrieval_query(
    prediction: FailurePrediction,
    vision: VisionCheck | None = None,
) -> str:
    return " ".join(item.summary for item in build_evidence_items(prediction, vision))


def build_evidence_items(
    prediction: FailurePrediction,
    vision: VisionCheck | None = None,
) -> list[EvidenceItem]:
    evidence = [
        EvidenceItem(
            source="telemetry",
            signal="failure_risk",
            status=prediction.risk_level.value,
            summary=(
                f"Telemetry failure probability is "
                f"{prediction.failure_probability_percent}% ({prediction.risk_level.value})"
            ),
            details=list(prediction.evidence),
        )
    ]
    if vision:
        defect = vision.defect_type or "visual anomaly"
        status = "defect_detected" if vision.defect_detected else "no_defect_detected"
        summary = (
            f"Visual inspection detected {defect} in {vision.category}"
            if vision.defect_detected
            else f"Visual inspection found no defect in {vision.category}"
        )
        evidence.append(
            EvidenceItem(
                source="vision",
                signal=vision.method,
                status=status,
                summary=summary,
                details=[
                    f"confidence={vision.confidence:.3f}",
                    f"anomaly_score={vision.anomaly_score:.4f}",
                    f"threshold={vision.threshold:.4f}",
                    *vision.evidence,
                ],
            )
        )
    return evidence


def build_incident_id(machine_id: str) -> str:
    return f"{machine_id}-INVESTIGATION"


def run_vision_check(
    image_path: Path,
    category: str,
    method: str = "auto",
) -> VisionCheck:
    model_path = model_path_for_category(category)
    if method in {"auto", "resnet"}:
        if not model_path.exists():
            raise FileNotFoundError(
                f"No calibrated ResNet profile found at {model_path}. "
                "Run: PYTHONPATH=src .venv/bin/python -m industrial_ai.vision.mvtec_resnet "
                f"train {category} --reference-limit 50 && "
                "PYTHONPATH=src .venv/bin/python -m industrial_ai.vision.mvtec_resnet "
                f"calibrate --model-path {model_path} --metric f1"
            )
        result = predict_resnet(image_path=image_path, model_path=model_path)
        return VisionCheck(
            method="resnet",
            category=result.category,
            defect_detected=result.defect_detected,
            defect_type=result.defect_type,
            confidence=result.confidence,
            anomaly_score=result.anomaly_score,
            threshold=result.threshold,
            evidence=result.evidence,
        )
    result = compare_mvtec_image(image_path=image_path, category=category)
    return VisionCheck(
        method="comparison",
        category=result.category,
        defect_detected=result.defect_detected,
        defect_type=result.defect_type,
        confidence=result.confidence,
        anomaly_score=result.anomaly_score,
        threshold=result.threshold,
        evidence=result.evidence,
    )


def run_investigation(
    reading: TelemetryReading,
    vision_image_path: Path | None = None,
    vision_category: str | None = None,
    vision_method: str = "auto",
    top_k: int = DEFAULT_TOP_K,
    score_threshold: float = DEFAULT_SCORE_THRESHOLD,
    qdrant_path: Path = QDRANT_DATA_DIR,
    collection_name: str = DEFAULT_COLLECTION_NAME,
    embedding_model: str = DEFAULT_EMBEDDING_MODEL,
    approvals_store_path: Path = APPROVALS_STORE_PATH,
    embedder: Embedder | None = None,
    rag_mode: str = "deterministic",
    rag_fallback: bool = True,
) -> InvestigationResult:
    prediction = predict_failure(reading)
    vision = (
        run_vision_check(
            image_path=vision_image_path,
            category=vision_category,
            method=vision_method,
        )
        if vision_image_path and vision_category
        else None
    )
    evidence = build_evidence_items(prediction, vision)
    retrieval_query = " ".join(item.summary for item in evidence)
    embedder = embedder or load_embedder(embedding_model)
    retrieval = retrieve_incidents(
        query=retrieval_query,
        top_k=top_k,
        score_threshold=score_threshold,
        qdrant_path=qdrant_path,
        collection_name=collection_name,
        embedder=embedder,
        telemetry_query=TelemetryQuery(
            tool_wear_min=reading.tool_wear_min,
            torque_nm=reading.torque_nm,
            rotational_speed_rpm=reading.rotational_speed_rpm,
            air_temperature_k=reading.air_temperature_k,
            process_temperature_k=reading.process_temperature_k,
        ),
    )
    rag_answer = (
        build_llm_answer_from_results(retrieval_query, retrieval.results, fallback=rag_fallback)
        if rag_mode == "ollama"
        else build_answer_from_results(retrieval_query, retrieval.results)
    )
    severity = assign_severity(
        failure_probability=prediction.failure_probability,
        rag_confidence=rag_answer.confidence,
        visual_defect_detected=bool(vision and vision.defect_detected),
    )
    approval = create_approval(
        incident_id=build_incident_id(reading.machine_id),
        severity=severity.severity.value,
        store_path=approvals_store_path,
    )
    return InvestigationResult(
        prediction=prediction,
        vision=vision,
        evidence=evidence,
        similar_incidents=retrieval.results,
        rag_answer=rag_answer,
        severity=severity,
        approval=approval,
    )


def investigation_to_dict(result: InvestigationResult) -> dict:
    return {
        "prediction": asdict(result.prediction),
        "vision": asdict(result.vision) if result.vision else None,
        "evidence": [asdict(item) for item in result.evidence],
        "similar_incidents": [asdict(incident) for incident in result.similar_incidents],
        "rag_answer": answer_to_dict(result.rag_answer),
        "severity": asdict(result.severity),
        "approval": record_to_dict(result.approval),
    }
