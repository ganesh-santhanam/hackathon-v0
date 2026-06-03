from pathlib import Path
from typing import Any, TypedDict

from industrial_ai.approvals.approval import ApprovalRecord, create_approval
from industrial_ai.demo.investigation import (
    DEFAULT_TOP_K,
    EvidenceItem,
    InvestigationResult,
    VisionCheck,
    build_evidence_items,
    build_incident_id,
    run_vision_check,
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
from industrial_ai.rag.answer import RagAnswer, build_answer_from_results, build_llm_answer_from_results
from industrial_ai.telemetry.predict import FailurePrediction, TelemetryReading, predict_failure

try:
    from langgraph.graph import END, START, StateGraph
except ModuleNotFoundError:  # pragma: no cover - exercised implicitly when dependency is absent.
    END = "__end__"
    START = "__start__"
    StateGraph = None


class InvestigationGraphState(TypedDict, total=False):
    reading: TelemetryReading
    vision_image_path: Path | None
    vision_category: str | None
    vision_method: str
    top_k: int
    score_threshold: float
    qdrant_path: Path
    collection_name: str
    embedding_model: str
    approvals_store_path: Path
    embedder: Embedder | None
    rag_mode: str
    rag_fallback: bool
    prediction: FailurePrediction
    vision: VisionCheck | None
    evidence: list[EvidenceItem]
    retrieval_query: str
    retrieved_incidents: list[SearchResult]
    rag_answer: RagAnswer
    severity: SeverityDecision
    approval: ApprovalRecord
    agent_trace: list[str]


NODE_ORDER = (
    "telemetry_agent",
    "vision_agent",
    "memory_agent",
    "rag_agent",
    "severity_agent",
    "approval_agent",
)


def append_trace(state: InvestigationGraphState, message: str) -> None:
    state.setdefault("agent_trace", []).append(message)


def telemetry_agent(state: InvestigationGraphState) -> InvestigationGraphState:
    prediction = predict_failure(state["reading"])
    state["prediction"] = prediction
    append_trace(
        state,
        f"telemetry_agent: predicted {prediction.risk_level.value} risk at "
        f"{prediction.failure_probability_percent}%.",
    )
    return state


def vision_agent(state: InvestigationGraphState) -> InvestigationGraphState:
    image_path = state.get("vision_image_path")
    category = state.get("vision_category")
    if image_path and category:
        vision = run_vision_check(
            image_path=image_path,
            category=category,
            method=state.get("vision_method", "auto"),
        )
        state["vision"] = vision
        append_trace(
            state,
            f"vision_agent: {'detected' if vision.defect_detected else 'did not detect'} "
            f"a defect in {vision.category}.",
        )
    else:
        state["vision"] = None
        append_trace(state, "vision_agent: skipped because no inspection image was provided.")
    return state


def memory_agent(state: InvestigationGraphState) -> InvestigationGraphState:
    evidence = build_evidence_items(state["prediction"], state.get("vision"))
    retrieval_query = " ".join(item.summary for item in evidence)
    embedder = state.get("embedder") or load_embedder(state.get("embedding_model", DEFAULT_EMBEDDING_MODEL))
    reading = state["reading"]
    retrieval = retrieve_incidents(
        query=retrieval_query,
        top_k=state.get("top_k", DEFAULT_TOP_K),
        score_threshold=state.get("score_threshold", DEFAULT_SCORE_THRESHOLD),
        qdrant_path=state.get("qdrant_path", QDRANT_DATA_DIR),
        collection_name=state.get("collection_name", DEFAULT_COLLECTION_NAME),
        embedder=embedder,
        telemetry_query=TelemetryQuery(
            tool_wear_min=reading.tool_wear_min,
            torque_nm=reading.torque_nm,
            rotational_speed_rpm=reading.rotational_speed_rpm,
            air_temperature_k=reading.air_temperature_k,
            process_temperature_k=reading.process_temperature_k,
        ),
    )
    state["evidence"] = evidence
    state["retrieval_query"] = retrieval_query
    state["retrieved_incidents"] = retrieval.results
    append_trace(state, f"memory_agent: retrieved {len(retrieval.results)} incident(s).")
    return state


def rag_agent(state: InvestigationGraphState) -> InvestigationGraphState:
    if state.get("rag_mode") == "ollama":
        rag_answer = build_llm_answer_from_results(
            state["retrieval_query"],
            state["retrieved_incidents"],
            fallback=state.get("rag_fallback", True),
        )
    else:
        rag_answer = build_answer_from_results(state["retrieval_query"], state["retrieved_incidents"])
    state["rag_answer"] = rag_answer
    append_trace(
        state,
        f"rag_agent: produced {rag_answer.confidence} confidence answer "
        f"using {rag_answer.metadata.rag_mode} mode.",
    )
    return state


def severity_agent(state: InvestigationGraphState) -> InvestigationGraphState:
    vision = state.get("vision")
    severity = assign_severity(
        failure_probability=state["prediction"].failure_probability,
        rag_confidence=state["rag_answer"].confidence,
        visual_defect_detected=bool(vision and vision.defect_detected),
    )
    state["severity"] = severity
    append_trace(state, f"severity_agent: assigned {severity.severity.value}.")
    return state


def approval_agent(state: InvestigationGraphState) -> InvestigationGraphState:
    approval = create_approval(
        incident_id=build_incident_id(state["reading"].machine_id),
        severity=state["severity"].severity.value,
        store_path=state.get("approvals_store_path", APPROVALS_STORE_PATH),
    )
    state["approval"] = approval
    append_trace(state, f"approval_agent: approval status is {approval.status.value}.")
    return state


NODE_FUNCTIONS = {
    "telemetry_agent": telemetry_agent,
    "vision_agent": vision_agent,
    "memory_agent": memory_agent,
    "rag_agent": rag_agent,
    "severity_agent": severity_agent,
    "approval_agent": approval_agent,
}


def build_investigation_graph():
    if StateGraph is None:
        return None
    graph = StateGraph(InvestigationGraphState)
    for node_name in NODE_ORDER:
        graph.add_node(node_name, NODE_FUNCTIONS[node_name])
    graph.add_edge(START, "telemetry_agent")
    for current_node, next_node in zip(NODE_ORDER, NODE_ORDER[1:]):
        graph.add_edge(current_node, next_node)
    graph.add_edge("approval_agent", END)
    return graph.compile()


def run_sequential_graph(state: InvestigationGraphState) -> InvestigationGraphState:
    for node_name in NODE_ORDER:
        state = NODE_FUNCTIONS[node_name](state)
    return state


def initial_graph_state(
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
) -> InvestigationGraphState:
    return {
        "reading": reading,
        "vision_image_path": vision_image_path,
        "vision_category": vision_category,
        "vision_method": vision_method,
        "top_k": top_k,
        "score_threshold": score_threshold,
        "qdrant_path": qdrant_path,
        "collection_name": collection_name,
        "embedding_model": embedding_model,
        "approvals_store_path": approvals_store_path,
        "embedder": embedder,
        "rag_mode": rag_mode,
        "rag_fallback": rag_fallback,
        "agent_trace": [],
    }


def result_from_graph_state(state: InvestigationGraphState) -> InvestigationResult:
    return InvestigationResult(
        prediction=state["prediction"],
        vision=state.get("vision"),
        evidence=state["evidence"],
        similar_incidents=state["retrieved_incidents"],
        rag_answer=state["rag_answer"],
        severity=state["severity"],
        approval=state["approval"],
        agent_trace=state["agent_trace"],
    )


def run_investigation_graph(
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
    state = initial_graph_state(
        reading=reading,
        vision_image_path=vision_image_path,
        vision_category=vision_category,
        vision_method=vision_method,
        top_k=top_k,
        score_threshold=score_threshold,
        qdrant_path=qdrant_path,
        collection_name=collection_name,
        embedding_model=embedding_model,
        approvals_store_path=approvals_store_path,
        embedder=embedder,
        rag_mode=rag_mode,
        rag_fallback=rag_fallback,
    )
    graph = build_investigation_graph()
    final_state: dict[str, Any] = graph.invoke(state) if graph else run_sequential_graph(state)
    return result_from_graph_state(final_state)
