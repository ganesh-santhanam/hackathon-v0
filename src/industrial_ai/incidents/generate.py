import argparse
import json
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

import pandas as pd

from industrial_ai.paths import INCIDENTS_DATA_DIR
from industrial_ai.telemetry.ai4i import FAILURE_MODE_COLUMNS, TARGET_COLUMN, load_ai4i_dataset


DOCUMENT_TYPES = ("incident_report", "rca_report", "maintenance_note")
DEFAULT_SOURCE_FAILURE_ROWS = 100

FAILURE_MODE_LABELS = {
    "twf": "tool wear failure",
    "hdf": "heat dissipation failure",
    "pwf": "power failure",
    "osf": "overstrain failure",
    "rnf": "random failure",
}
DOCUMENT_TYPE_TITLES = {
    "incident_report": "Incident Report",
    "rca_report": "RCA Report",
    "maintenance_note": "Maintenance Note",
}


@dataclass(frozen=True)
class IncidentDocument:
    document_id: str
    document_type: str
    source_dataset: str
    source_row_id: int
    machine_id: str
    title: str
    body: str
    metadata: dict[str, Any]
    evidence: list[str]


def infer_failure_modes(row: pd.Series) -> list[str]:
    modes = [
        FAILURE_MODE_LABELS[column]
        for column in FAILURE_MODE_COLUMNS
        if column in row and int(row[column]) == 1
    ]
    return modes or ["unclassified machine failure"]


def build_machine_id(row: pd.Series) -> str:
    return f"AI4I-{int(row['udi']):05d}"


def build_evidence(row: pd.Series) -> list[str]:
    evidence = [
        f"Machine type: {row['type']}",
        f"Tool wear: {row['tool_wear_min']} min",
        f"Torque: {row['torque_nm']} Nm",
        f"Rotational speed: {row['rotational_speed_rpm']} rpm",
        f"Air temperature: {row['air_temperature_k']} K",
        f"Process temperature: {row['process_temperature_k']} K",
    ]
    for mode in infer_failure_modes(row):
        evidence.append(f"AI4I label: {mode}")
    return evidence


def build_metadata(row: pd.Series, failure_modes: list[str]) -> dict[str, Any]:
    return {
        "product_id": row["product_id"],
        "machine_type": row["type"],
        "failure_modes": failure_modes,
        "machine_failure": int(row[TARGET_COLUMN]),
        "telemetry": {
            "air_temperature_k": float(row["air_temperature_k"]),
            "process_temperature_k": float(row["process_temperature_k"]),
            "rotational_speed_rpm": int(row["rotational_speed_rpm"]),
            "torque_nm": float(row["torque_nm"]),
            "tool_wear_min": int(row["tool_wear_min"]),
        },
    }


def render_body(document_type: str, machine_id: str, row: pd.Series, failure_modes: list[str]) -> str:
    primary_mode = failure_modes[0]
    if document_type == "incident_report":
        return (
            f"{machine_id} reported a {primary_mode}. Telemetry showed "
            f"{row['tool_wear_min']} minutes of tool wear, {row['torque_nm']} Nm torque, "
            f"and {row['rotational_speed_rpm']} rpm rotational speed. "
            "The event was opened for investigation because the AI4I machine failure label "
            "was positive."
        )
    if document_type == "rca_report":
        return (
            f"Root cause analysis for {machine_id} points to {', '.join(failure_modes)}. "
            f"The strongest contributing signals were tool wear at {row['tool_wear_min']} "
            f"minutes, torque at {row['torque_nm']} Nm, and process temperature at "
            f"{row['process_temperature_k']} K. Recommended action is to inspect the asset, "
            "validate operating load, and review recent maintenance history."
        )
    if document_type == "maintenance_note":
        return (
            f"Maintenance note for {machine_id}: inspect tooling, drivetrain load, and thermal "
            f"conditions after {primary_mode}. Record corrective action, replace worn tooling "
            "if needed, and confirm the machine returns to normal telemetry bands before closeout."
        )
    raise ValueError(f"Unsupported document type: {document_type}")


def generate_documents(source_failure_rows: int = DEFAULT_SOURCE_FAILURE_ROWS) -> list[IncidentDocument]:
    dataset = load_ai4i_dataset()
    failure_rows = dataset.raw.loc[dataset.raw[TARGET_COLUMN] == 1].head(source_failure_rows)
    documents = []

    for _, row in failure_rows.iterrows():
        source_row_id = int(row["udi"])
        machine_id = build_machine_id(row)
        failure_modes = infer_failure_modes(row)
        evidence = build_evidence(row)
        metadata = build_metadata(row, failure_modes)

        for document_type in DOCUMENT_TYPES:
            document_id = f"ai4i-{source_row_id:05d}-{document_type}"
            title = f"{DOCUMENT_TYPE_TITLES[document_type]} - {machine_id}"
            documents.append(
                IncidentDocument(
                    document_id=document_id,
                    document_type=document_type,
                    source_dataset="AI4I",
                    source_row_id=source_row_id,
                    machine_id=machine_id,
                    title=title,
                    body=render_body(document_type, machine_id, row, failure_modes),
                    metadata=metadata,
                    evidence=evidence,
                )
            )
    return documents


def write_documents(
    documents: list[IncidentDocument],
    output_dir: Path = INCIDENTS_DATA_DIR,
) -> tuple[Path, Path]:
    output_dir.mkdir(parents=True, exist_ok=True)
    corpus_path = output_dir / "ai4i_incident_corpus.jsonl"
    manifest_path = output_dir / "manifest.json"

    with corpus_path.open("w", encoding="utf-8") as corpus_file:
        for document in documents:
            corpus_file.write(json.dumps(asdict(document), sort_keys=True) + "\n")

    manifest = {
        "generator_version": 1,
        "source_dataset": "AI4I",
        "document_count": len(documents),
        "document_types": sorted({document.document_type for document in documents}),
        "source_row_count": len({document.source_row_id for document in documents}),
        "corpus_file": corpus_path.name,
    }
    manifest_path.write_text(json.dumps(manifest, indent=2, sort_keys=True), encoding="utf-8")
    return corpus_path, manifest_path


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Generate structured AI4I incident documents.")
    parser.add_argument("--source-failure-rows", default=DEFAULT_SOURCE_FAILURE_ROWS, type=int)
    parser.add_argument("--output-dir", default=INCIDENTS_DATA_DIR, type=Path)
    return parser


def main() -> None:
    args = build_parser().parse_args()
    documents = generate_documents(source_failure_rows=args.source_failure_rows)
    corpus_path, manifest_path = write_documents(documents, output_dir=args.output_dir)
    print(
        json.dumps(
            {
                "document_count": len(documents),
                "corpus_path": str(corpus_path),
                "manifest_path": str(manifest_path),
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
