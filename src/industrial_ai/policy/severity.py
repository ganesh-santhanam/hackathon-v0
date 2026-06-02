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


@dataclass(frozen=True)
class SeverityRule:
    severity: Severity
    criteria: str
    approval_required: bool


@dataclass(frozen=True)
class SeverityPolicy:
    name: str
    version: str
    last_modified: str
    rules: tuple[SeverityRule, ...]


SEVERITY_POLICY_RULES = (
    SeverityRule(
        severity=Severity.SEV1,
        criteria="Failure probability > 80% and visual defect detected",
        approval_required=True,
    ),
    SeverityRule(
        severity=Severity.SEV1,
        criteria="Failure probability > 80% and RAG confidence is high",
        approval_required=True,
    ),
    SeverityRule(
        severity=Severity.SEV2,
        criteria="Failure probability > 50%",
        approval_required=False,
    ),
    SeverityRule(
        severity=Severity.SEV3,
        criteria="Failure probability <= 50%",
        approval_required=False,
    ),
)
SEVERITY_POLICY = SeverityPolicy(
    name="Tier 0 Severity Policy",
    version="1.0.0",
    last_modified="2026-06-02T00:00:00Z",
    rules=SEVERITY_POLICY_RULES,
)


def severity_policy() -> SeverityPolicy:
    return SEVERITY_POLICY


def severity_policy_rules() -> tuple[SeverityRule, ...]:
    return severity_policy().rules


def approval_required_for_severity_value(severity: Severity) -> bool:
    return any(rule.approval_required for rule in severity_policy_rules() if rule.severity == severity)


def normalize_probability(probability: float) -> float:
    if probability > 1:
        return probability / 100
    return probability


def assign_severity(
    failure_probability: float,
    rag_confidence: str,
    visual_defect_detected: bool = False,
) -> SeverityDecision:
    probability = normalize_probability(failure_probability)
    confidence = RagConfidence(rag_confidence.lower())

    if probability > 0.8 and visual_defect_detected:
        return SeverityDecision(
            severity=Severity.SEV1,
            reason="Failure probability is above 80% and a visual defect was detected.",
            inputs={
                "failure_probability": probability,
                "rag_confidence": confidence.value,
                "visual_defect_detected": str(visual_defect_detected).lower(),
            },
        )

    if probability > 0.8 and confidence == RagConfidence.HIGH:
        return SeverityDecision(
            severity=Severity.SEV1,
            reason="Failure probability is above 80% and RAG confidence is high.",
            inputs={
                "failure_probability": probability,
                "rag_confidence": confidence.value,
                "visual_defect_detected": str(visual_defect_detected).lower(),
            },
        )

    if probability > 0.5:
        return SeverityDecision(
            severity=Severity.SEV2,
            reason="Failure probability is above 50%.",
            inputs={
                "failure_probability": probability,
                "rag_confidence": confidence.value,
                "visual_defect_detected": str(visual_defect_detected).lower(),
            },
        )

    return SeverityDecision(
        severity=Severity.SEV3,
        reason="Failure probability is at or below 50%.",
        inputs={
            "failure_probability": probability,
            "rag_confidence": confidence.value,
            "visual_defect_detected": str(visual_defect_detected).lower(),
        },
    )


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Assign deterministic incident severity.")
    parser.add_argument("--failure-probability", required=True, type=float)
    parser.add_argument("--rag-confidence", required=True, choices=[item.value for item in RagConfidence])
    parser.add_argument("--visual-defect-detected", action="store_true")
    return parser


def main() -> None:
    args = build_parser().parse_args()
    decision = assign_severity(
        failure_probability=args.failure_probability,
        rag_confidence=args.rag_confidence,
        visual_defect_detected=args.visual_defect_detected,
    )
    print(json.dumps(asdict(decision), indent=2))


if __name__ == "__main__":
    main()
