import json

from industrial_ai.incidents.generate import (
    DOCUMENT_TYPES,
    generate_documents,
    infer_failure_modes,
    write_documents,
)
from industrial_ai.telemetry.ai4i import load_ai4i_dataset


def test_generate_documents_creates_three_structured_docs_per_failure_row():
    documents = generate_documents(source_failure_rows=2)

    assert len(documents) == 6
    assert {document.document_type for document in documents} == set(DOCUMENT_TYPES)
    assert all(document.source_dataset == "AI4I" for document in documents)
    assert all(document.body for document in documents)
    assert all(document.evidence for document in documents)
    assert all("telemetry" in document.metadata for document in documents)


def test_infer_failure_modes_falls_back_for_unclassified_ai4i_failures():
    dataset = load_ai4i_dataset()
    row = dataset.raw.loc[dataset.raw["machine_failure"] == 1].iloc[0]

    assert infer_failure_modes(row)


def test_write_documents_writes_jsonl_and_manifest(tmp_path):
    documents = generate_documents(source_failure_rows=1)

    corpus_path, manifest_path = write_documents(documents, output_dir=tmp_path)

    lines = corpus_path.read_text(encoding="utf-8").splitlines()
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))

    assert len(lines) == 3
    assert json.loads(lines[0])["document_id"].startswith("ai4i-")
    assert manifest["document_count"] == 3
    assert manifest["source_row_count"] == 1
