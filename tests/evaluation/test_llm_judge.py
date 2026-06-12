import json

from industrial_ai.evaluation.llm_judge import (
    build_eval_dataset,
    generate_report,
    parse_json_object,
    run_candidate_evaluation,
    run_judge,
    summarize_scores,
)


def write_jsonl(path, records):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        "".join(json.dumps(record, sort_keys=True) + "\n" for record in records),
        encoding="utf-8",
    )


def incident_documents():
    metadata = {
        "failure_modes": ["tool wear failure"],
        "machine_failure": 1,
        "machine_type": "L",
        "telemetry": {
            "air_temperature_k": 298.8,
            "process_temperature_k": 308.9,
            "rotational_speed_rpm": 1455,
            "tool_wear_min": 208,
            "torque_nm": 41.3,
        },
    }
    return [
        {
            "document_id": "ai4i-00078-incident_report",
            "document_type": "incident_report",
            "machine_id": "AI4I-00078",
            "title": "Incident Report - AI4I-00078",
            "body": "AI4I-00078 reported a tool wear failure.",
            "evidence": ["Tool wear: 208 min", "AI4I label: tool wear failure"],
            "metadata": metadata,
            "source_dataset": "AI4I",
            "source_row_id": 78,
        },
        {
            "document_id": "ai4i-00078-rca_report",
            "document_type": "rca_report",
            "machine_id": "AI4I-00078",
            "title": "RCA Report - AI4I-00078",
            "body": "Root cause analysis points to tool wear failure.",
            "evidence": ["Tool wear: 208 min", "AI4I label: tool wear failure"],
            "metadata": metadata,
            "source_dataset": "AI4I",
            "source_row_id": 78,
        },
    ]


def test_build_eval_dataset_includes_telemetry_evidence_and_expectations(tmp_path):
    corpus_path = tmp_path / "corpus.jsonl"
    output_path = tmp_path / "eval_dataset.jsonl"
    write_jsonl(corpus_path, incident_documents())

    summary = build_eval_dataset(corpus_path=corpus_path, output_path=output_path)
    rows = output_path.read_text(encoding="utf-8").splitlines()
    example = json.loads(rows[0])

    assert summary["examples"] == 1
    assert example["telemetry"]["tool_wear_min"] == 208
    assert example["retrieved_incident_evidence"][0]["document_id"] == "ai4i-00078-rca_report"
    assert example["expected_failure_mode"] == "tool wear failure"
    assert example["expected_severity"] == "SEV2"
    assert "Current telemetry" in example["prompt"]


def test_candidate_and_judge_dry_run_create_expected_outputs(tmp_path):
    corpus_path = tmp_path / "corpus.jsonl"
    dataset_path = tmp_path / "eval_dataset.jsonl"
    base_path = tmp_path / "base_results.jsonl"
    lora_path = tmp_path / "lora_results.jsonl"
    judge_path = tmp_path / "judge_scores.jsonl"
    write_jsonl(corpus_path, incident_documents())
    build_eval_dataset(corpus_path=corpus_path, output_path=dataset_path)

    base_summary = run_candidate_evaluation(
        dataset_path=dataset_path,
        output_path=base_path,
        candidate_name="base",
        model="gemma3:4b",
        dry_run=True,
    )
    lora_summary = run_candidate_evaluation(
        dataset_path=dataset_path,
        output_path=lora_path,
        candidate_name="lora",
        model="gemma3-lora:latest",
        dry_run=True,
    )
    judge_summary = run_judge(
        dataset_path=dataset_path,
        candidate_result_paths=[base_path, lora_path],
        output_path=judge_path,
        dry_run=True,
    )

    assert base_summary["records"] == 1
    assert lora_summary["records"] == 1
    assert judge_summary["records"] == 2
    scores = [json.loads(line) for line in judge_path.read_text(encoding="utf-8").splitlines()]
    assert {record["candidate_name"] for record in scores} == {"base", "lora"}
    assert scores[0]["scores"]["hallucination_score"] == 1.0


def test_summarize_scores_writes_json_and_csv(tmp_path):
    judge_path = tmp_path / "judge_scores.jsonl"
    summary_json = tmp_path / "summary.json"
    summary_csv = tmp_path / "summary.csv"
    write_jsonl(
        judge_path,
        [
            {
                "eval_id": "case-1",
                "candidate_name": "base",
                "candidate_model": "gemma3:4b",
                "judge_model": "gpt-oss:20b",
                "judge_provider": "openai-compatible",
                "judge_endpoint": "http://localhost:8000/v1/chat/completions",
                "scores": {
                    "hallucination_score": 2,
                    "rca_quality": 3,
                    "actionability": 3,
                    "severity_reasoning": 3,
                    "rationale": "ok",
                },
            },
            {
                "eval_id": "case-1",
                "candidate_name": "lora",
                "candidate_model": "gemma3-lora:latest",
                "judge_model": "gpt-oss:20b",
                "judge_provider": "openai-compatible",
                "judge_endpoint": "http://localhost:8000/v1/chat/completions",
                "scores": {
                    "hallucination_score": 1,
                    "rca_quality": 4,
                    "actionability": 4,
                    "severity_reasoning": 5,
                    "rationale": "better",
                },
            },
        ],
    )

    summary = summarize_scores(judge_path, summary_json, summary_csv)

    assert summary["examples_evaluated"] == 1
    assert summary["candidates"]["base"]["metrics"]["rca_quality"]["mean"] == 3.0
    assert summary["improvement_percentage"]["hallucination_score"] == 50.0
    assert "candidate,model,metric" in summary_csv.read_text(encoding="utf-8")


def test_generate_report_writes_slide_ready_markdown(tmp_path):
    summary_json = tmp_path / "summary.json"
    report_path = tmp_path / "report.md"
    summary_json.write_text(
        json.dumps(
            {
                "examples_evaluated": 1,
                "judge": {"models": ["gpt-oss:20b"], "provider": "openai-compatible"},
                "candidates": {
                    "base": {
                        "model": "gemma3:4b",
                        "examples_evaluated": 1,
                        "metrics": {
                            "hallucination_score": {"mean": 2.0},
                            "rca_quality": {"mean": 3.0},
                            "actionability": {"mean": 3.0},
                            "severity_reasoning": {"mean": 3.0},
                        },
                    },
                    "lora": {
                        "model": "gemma3-lora:latest",
                        "examples_evaluated": 1,
                        "metrics": {
                            "hallucination_score": {"mean": 1.0},
                            "rca_quality": {"mean": 4.0},
                            "actionability": {"mean": 4.0},
                            "severity_reasoning": {"mean": 5.0},
                        },
                    },
                },
                "improvement_percentage": {
                    "hallucination_score": 50.0,
                    "rca_quality": 33.33,
                    "actionability": 33.33,
                    "severity_reasoning": 66.67,
                },
            }
        ),
        encoding="utf-8",
    )

    result = generate_report(summary_json, report_path)

    assert result["report_path"] == str(report_path)
    report = report_path.read_text(encoding="utf-8")
    assert "# AMD Hackathon LLM-as-Judge Evaluation" in report
    assert "| Field | Value |" in report
    assert "| Base Model | gemma3:4b |" in report
    assert "| LoRA Model | gemma3-lora:latest |" in report
    assert "| Judge Model | gpt-oss:20b |" in report
    assert "| Metric | Base Mean | LoRA Mean | LoRA Improvement % |" in report


def test_parse_json_object_handles_markdown_fence():
    parsed = parse_json_object('```json\n{"rca_quality": 4}\n```')

    assert parsed == {"rca_quality": 4}
