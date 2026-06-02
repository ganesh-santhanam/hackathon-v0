import pytest

from industrial_ai.policy.severity import (
    RagConfidence,
    Severity,
    approval_required_for_severity_value,
    assign_severity,
    normalize_probability,
    severity_policy,
    severity_policy_rules,
)


def test_assigns_sev1_when_probability_above_80_percent_and_rag_confidence_high():
    decision = assign_severity(failure_probability=0.81, rag_confidence="high")

    assert decision.severity == Severity.SEV1
    assert "above 80%" in decision.reason


def test_assigns_sev1_when_probability_above_80_percent_and_visual_defect_detected():
    decision = assign_severity(
        failure_probability=0.81,
        rag_confidence="medium",
        visual_defect_detected=True,
    )

    assert decision.severity == Severity.SEV1
    assert "visual defect" in decision.reason
    assert decision.inputs["visual_defect_detected"] == "true"


def test_assigns_sev2_when_probability_above_80_percent_but_confidence_is_not_high():
    decision = assign_severity(failure_probability=0.81, rag_confidence="medium")

    assert decision.severity == Severity.SEV2


def test_assigns_sev2_when_probability_above_50_percent():
    decision = assign_severity(failure_probability=0.51, rag_confidence="none")

    assert decision.severity == Severity.SEV2


def test_visual_defect_alone_does_not_escalate_low_probability():
    decision = assign_severity(
        failure_probability=0.2,
        rag_confidence="none",
        visual_defect_detected=True,
    )

    assert decision.severity == Severity.SEV3


def test_assigns_sev3_at_exact_50_percent_boundary():
    decision = assign_severity(failure_probability=0.5, rag_confidence="high")

    assert decision.severity == Severity.SEV3


def test_assigns_sev3_for_low_probability():
    decision = assign_severity(failure_probability=0.2, rag_confidence="high")

    assert decision.severity == Severity.SEV3


def test_accepts_percentage_style_probability():
    decision = assign_severity(failure_probability=81, rag_confidence="high")

    assert decision.severity == Severity.SEV1
    assert decision.inputs["failure_probability"] == 0.81


def test_rejects_unknown_rag_confidence():
    with pytest.raises(ValueError):
        assign_severity(failure_probability=0.9, rag_confidence="certain")


def test_rag_confidence_values_match_cli_choices():
    assert [item.value for item in RagConfidence] == ["none", "low", "medium", "high"]


def test_severity_policy_rules_expose_criteria_and_approval_requirements():
    rules = severity_policy_rules()

    assert [rule.severity for rule in rules] == [
        Severity.SEV1,
        Severity.SEV1,
        Severity.SEV2,
        Severity.SEV3,
    ]
    assert approval_required_for_severity_value(Severity.SEV1) is True
    assert approval_required_for_severity_value(Severity.SEV2) is False
    assert approval_required_for_severity_value(Severity.SEV3) is False


def test_severity_policy_exposes_management_metadata():
    policy = severity_policy()

    assert policy.name == "Tier 0 Severity Policy"
    assert policy.version == "1.0.0"
    assert policy.last_modified == "2026-06-02T00:00:00Z"
    assert policy.rules == severity_policy_rules()


def test_normalize_probability_leaves_fractional_probability_unchanged():
    assert normalize_probability(0.82) == 0.82
