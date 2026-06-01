import argparse
import json
from dataclasses import asdict, dataclass
from enum import StrEnum


class Severity(StrEnum):
    SEV1 = "SEV1"
    SEV2 = "SEV2"
    SEV3 = "SEV3"


class RagConfidence(StrEnum):
    NONE = "none"
    LOW = "low"
    MEDIUM = "medium"
    HIGH = "high"


@dataclass(frozen=True)
class SeverityDecision:
    severity: Severity
    reason: str
    inputs: dict[str, float | str]


def normalize_probability(probability: float) -> float:
    if probability > 1:
        return probability / 100
    return probability


def assign_severity(
    failure_probability: float,
    rag_confidence: str,
) -> SeverityDecision:
    probability = normalize_probability(failure_probability)
    confidence = RagConfidence(rag_confidence.lower())

    if probability > 0.8 and confidence == RagConfidence.HIGH:
        return SeverityDecision(
            severity=Severity.SEV1,
            reason="Failure probability is above 80% and RAG confidence is high.",
            inputs={
                "failure_probability": probability,
                "rag_confidence": confidence.value,
            },
        )

    if probability > 0.5:
        return SeverityDecision(
            severity=Severity.SEV2,
            reason="Failure probability is above 50%.",
            inputs={
                "failure_probability": probability,
                "rag_confidence": confidence.value,
            },
        )

    return SeverityDecision(
        severity=Severity.SEV3,
        reason="Failure probability is at or below 50%.",
        inputs={
            "failure_probability": probability,
            "rag_confidence": confidence.value,
        },
    )


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Assign deterministic incident severity.")
    parser.add_argument("--failure-probability", required=True, type=float)
    parser.add_argument("--rag-confidence", required=True, choices=[item.value for item in RagConfidence])
    return parser


def main() -> None:
    args = build_parser().parse_args()
    decision = assign_severity(
        failure_probability=args.failure_probability,
        rag_confidence=args.rag_confidence,
    )
    print(json.dumps(asdict(decision), indent=2))


if __name__ == "__main__":
    main()
