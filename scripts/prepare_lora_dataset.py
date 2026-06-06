#!/usr/bin/env python3
import argparse
import json
import random
from collections import Counter
from pathlib import Path
from typing import Any


DEFAULT_INPUT_PATH = Path("data/incidents/ai4i_incident_corpus.jsonl")
DEFAULT_OUTPUT_DIR = Path("data/lora")
DEFAULT_EVAL_RATIO = 0.2
DEFAULT_SEED = 42


def load_jsonl(path: Path) -> list[dict[str, Any]]:
    with path.open(encoding="utf-8") as input_file:
        return [json.loads(line) for line in input_file if line.strip()]


def failure_modes(document: dict[str, Any]) -> list[str]:
    modes = document.get("metadata", {}).get("failure_modes", [])
    return [str(mode) for mode in modes] if isinstance(modes, list) else []


def evidence_text(document: dict[str, Any]) -> str:
    evidence = document.get("evidence", [])
    if not isinstance(evidence, list):
        return ""
    return "\n".join(f"- {item}" for item in evidence)


def telemetry_summary(document: dict[str, Any]) -> str:
    telemetry = document.get("metadata", {}).get("telemetry", {})
    if not isinstance(telemetry, dict):
        return "Telemetry unavailable."
    return (
        f"Tool wear: {telemetry.get('tool_wear_min', 'n/a')} min\n"
        f"Torque: {telemetry.get('torque_nm', 'n/a')} Nm\n"
        f"Rotational speed: {telemetry.get('rotational_speed_rpm', 'n/a')} rpm\n"
        f"Air temperature: {telemetry.get('air_temperature_k', 'n/a')} K\n"
        f"Process temperature: {telemetry.get('process_temperature_k', 'n/a')} K"
    )


def base_input(document: dict[str, Any]) -> str:
    return (
        f"Document title: {document.get('title', '')}\n"
        f"Machine ID: {document.get('machine_id', '')}\n"
        f"Document type: {document.get('document_type', '')}\n"
        f"Body:\n{document.get('body', '')}\n\n"
        f"Telemetry:\n{telemetry_summary(document)}\n\n"
        f"Evidence:\n{evidence_text(document)}"
    )


def source_fields(document: dict[str, Any], task_type: str) -> dict[str, str]:
    return {
        "task_type": task_type,
        "source_document_id": str(document.get("document_id", "")),
        "source_document_type": str(document.get("document_type", "")),
    }


def rca_output(document: dict[str, Any]) -> str:
    modes = ", ".join(failure_modes(document)) or "unknown machine failure"
    return (
        f"Likely root cause: {modes}.\n"
        f"Supporting evidence:\n{evidence_text(document)}"
    )


def recommended_action_output(document: dict[str, Any]) -> str:
    modes = failure_modes(document)
    primary_mode = modes[0] if modes else "machine failure"
    return (
        f"Inspect the asset for {primary_mode}, validate operating load and telemetry, "
        "review recent maintenance history, and confirm the machine returns to normal bands before closeout."
    )


def severity_output(document: dict[str, Any]) -> str:
    modes = failure_modes(document)
    severity = "SEV2" if modes else "SEV3"
    if "tool wear failure" in modes or "power failure" in modes or "overstrain failure" in modes:
        severity = "SEV2"
    return (
        f"Suggested severity: {severity}. "
        "The event has a positive machine failure label and should be reviewed with telemetry evidence. "
        "Escalate further if high failure probability or visual defect evidence is present."
    )


def maintenance_summary_output(document: dict[str, Any]) -> str:
    modes = ", ".join(failure_modes(document)) or "machine failure"
    return (
        f"Maintenance summary for {document.get('machine_id', 'unknown machine')}: "
        f"investigate {modes}, inspect tooling and load conditions, record corrective action, "
        "and verify telemetry normalization."
    )


def evidence_extraction_output(document: dict[str, Any]) -> str:
    return evidence_text(document)


def examples_for_document(document: dict[str, Any]) -> list[dict[str, str]]:
    input_text = base_input(document)
    return [
        {
            "instruction": "Generate a concise root cause analysis from the incident evidence.",
            "input": input_text,
            "output": rca_output(document),
            **source_fields(document, "rca_generation"),
        },
        {
            "instruction": "Generate recommended maintenance actions for this industrial incident.",
            "input": input_text,
            "output": recommended_action_output(document),
            **source_fields(document, "recommended_action_generation"),
        },
        {
            "instruction": "Explain the likely severity using the available incident evidence.",
            "input": input_text,
            "output": severity_output(document),
            **source_fields(document, "severity_explanation"),
        },
        {
            "instruction": "Write a maintenance summary for the operator handoff.",
            "input": input_text,
            "output": maintenance_summary_output(document),
            **source_fields(document, "maintenance_summary"),
        },
        {
            "instruction": "Extract the key evidence signals from this incident document.",
            "input": input_text,
            "output": evidence_extraction_output(document),
            **source_fields(document, "evidence_extraction"),
        },
    ]


def validate_example(example: dict[str, str]) -> None:
    for field_name in ["instruction", "input", "output"]:
        if not example.get(field_name, "").strip():
            raise ValueError(f"Empty {field_name} in example from {example.get('source_document_id')}")


def split_examples(
    examples: list[dict[str, str]],
    eval_ratio: float,
    seed: int,
) -> tuple[list[dict[str, str]], list[dict[str, str]]]:
    if not 0 < eval_ratio < 1:
        raise ValueError("--eval-ratio must be greater than 0 and less than 1")
    shuffled = list(examples)
    random.Random(seed).shuffle(shuffled)
    eval_count = max(1, round(len(shuffled) * eval_ratio)) if len(shuffled) > 1 else 0
    return shuffled[eval_count:], shuffled[:eval_count]


def write_jsonl(path: Path, examples: list[dict[str, str]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as output_file:
        for example in examples:
            output_file.write(json.dumps(example, sort_keys=True) + "\n")


def task_counts(examples: list[dict[str, str]]) -> dict[str, int]:
    return dict(sorted(Counter(example["task_type"] for example in examples).items()))


def build_dataset(
    input_path: Path,
    output_dir: Path,
    eval_ratio: float,
    seed: int,
    limit: int | None,
) -> dict[str, Any]:
    documents = load_jsonl(input_path)
    if limit is not None:
        documents = documents[:limit]
    examples = [example for document in documents for example in examples_for_document(document)]
    for example in examples:
        validate_example(example)
    train_examples, eval_examples = split_examples(examples, eval_ratio=eval_ratio, seed=seed)
    if not train_examples or not eval_examples:
        raise ValueError("Both train and eval splits must contain at least one example.")

    train_path = output_dir / "train.jsonl"
    eval_path = output_dir / "eval.jsonl"
    write_jsonl(train_path, train_examples)
    write_jsonl(eval_path, eval_examples)
    return {
        "input_path": str(input_path),
        "train_path": str(train_path),
        "eval_path": str(eval_path),
        "document_count": len(documents),
        "total_examples": len(examples),
        "train_examples": len(train_examples),
        "eval_examples": len(eval_examples),
        "task_type_counts": task_counts(examples),
    }


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Prepare LoRA train/eval JSONL from incident corpus.")
    parser.add_argument("--input", type=Path, default=DEFAULT_INPUT_PATH)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--eval-ratio", type=float, default=DEFAULT_EVAL_RATIO)
    parser.add_argument("--seed", type=int, default=DEFAULT_SEED)
    parser.add_argument("--limit", type=int, default=None)
    return parser


def main() -> None:
    args = build_parser().parse_args()
    if args.limit is not None and args.limit < 1:
        raise SystemExit("--limit must be >= 1")
    summary = build_dataset(
        input_path=args.input,
        output_dir=args.output_dir,
        eval_ratio=args.eval_ratio,
        seed=args.seed,
        limit=args.limit,
    )
    print(json.dumps(summary, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
