import argparse
import csv
import json
import os
import statistics
import time
import urllib.error
import urllib.request
from collections import Counter, defaultdict
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from industrial_ai.paths import DATA_DIR, INCIDENTS_DATA_DIR


DEFAULT_CORPUS_PATH = INCIDENTS_DATA_DIR / "ai4i_incident_corpus.jsonl"
DEFAULT_EVAL_DIR = DATA_DIR / "evals"
DEFAULT_DATASET_PATH = DEFAULT_EVAL_DIR / "eval_dataset.jsonl"
DEFAULT_BASE_RESULTS_PATH = DEFAULT_EVAL_DIR / "base_results.jsonl"
DEFAULT_LORA_RESULTS_PATH = DEFAULT_EVAL_DIR / "lora_results.jsonl"
DEFAULT_JUDGE_SCORES_PATH = DEFAULT_EVAL_DIR / "judge_scores.jsonl"
DEFAULT_SUMMARY_JSON_PATH = DEFAULT_EVAL_DIR / "summary.json"
DEFAULT_SUMMARY_CSV_PATH = DEFAULT_EVAL_DIR / "summary.csv"
DEFAULT_REPORT_PATH = DEFAULT_EVAL_DIR / "llm_judge_report.md"
DEFAULT_OLLAMA_ENDPOINT = "http://localhost:11434/api/generate"
DEFAULT_OPENAI_COMPATIBLE_ENDPOINT = "http://localhost:8000/v1/chat/completions"
DEFAULT_BASE_MODEL = "gemma3:4b"
DEFAULT_LORA_MODEL = "gemma3-lora:latest"
DEFAULT_JUDGE_MODEL = "gpt-oss:20b"
DEFAULT_TIMEOUT_SECONDS = 180.0

SCORE_FIELDS = (
    "hallucination_score",
    "rca_quality",
    "actionability",
    "severity_reasoning",
)
HIGHER_IS_BETTER = {
    "hallucination_score": False,
    "rca_quality": True,
    "actionability": True,
    "severity_reasoning": True,
}


def utc_timestamp() -> str:
    return datetime.now(UTC).isoformat()


def load_jsonl(path: Path) -> list[dict[str, Any]]:
    with path.open(encoding="utf-8") as input_file:
        return [json.loads(line) for line in input_file if line.strip()]


def write_jsonl(path: Path, records: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as output_file:
        for record in records:
            output_file.write(json.dumps(record, sort_keys=True) + "\n")


def append_jsonl(path: Path, records: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as output_file:
        for record in records:
            output_file.write(json.dumps(record, sort_keys=True) + "\n")


def failure_modes(document: dict[str, Any]) -> list[str]:
    modes = document.get("metadata", {}).get("failure_modes", [])
    return [str(mode) for mode in modes] if isinstance(modes, list) else []


def expected_severity(document: dict[str, Any]) -> str | None:
    metadata = document.get("metadata", {})
    if metadata.get("machine_failure") == 1:
        return "SEV2"
    return None


def telemetry(document: dict[str, Any]) -> dict[str, Any]:
    value = document.get("metadata", {}).get("telemetry", {})
    return value if isinstance(value, dict) else {}


def evidence_documents_for_machine(
    documents: list[dict[str, Any]],
    machine_id: str,
    source_document_id: str,
    limit: int,
) -> list[dict[str, Any]]:
    same_machine = [
        document
        for document in documents
        if document.get("machine_id") == machine_id and document.get("document_id") != source_document_id
    ]
    priority = {"rca_report": 0, "maintenance_note": 1, "incident_report": 2}
    return sorted(
        same_machine,
        key=lambda document: (
            priority.get(str(document.get("document_type")), 99),
            str(document.get("document_id")),
        ),
    )[:limit]


def compact_document(document: dict[str, Any]) -> dict[str, Any]:
    return {
        "document_id": document.get("document_id"),
        "document_type": document.get("document_type"),
        "machine_id": document.get("machine_id"),
        "title": document.get("title"),
        "body": document.get("body"),
        "evidence": document.get("evidence", []),
        "failure_modes": failure_modes(document),
        "telemetry": telemetry(document),
    }


def build_candidate_prompt(example: dict[str, Any]) -> str:
    telemetry_lines = "\n".join(
        f"- {name}: {value}" for name, value in sorted(example.get("telemetry", {}).items())
    )
    evidence_sections = []
    for item in example.get("retrieved_incident_evidence", []):
        evidence = "\n".join(f"  - {line}" for line in item.get("evidence", []))
        evidence_sections.append(
            f"{item.get('document_id')} ({item.get('document_type')}): {item.get('title')}\n"
            f"Body: {item.get('body')}\n"
            f"Evidence:\n{evidence}"
        )
    evidence_text = "\n\n".join(evidence_sections) or "No retrieved incident evidence."
    return (
        "You are an industrial incident investigation assistant.\n"
        "Use only the provided telemetry and retrieved incident evidence. Do not invent facts.\n"
        "Return a concise investigation with these sections: Root Cause, Evidence, Actions, "
        "Severity Reasoning, Limitations.\n\n"
        f"Evaluation ID: {example['eval_id']}\n"
        f"Machine ID: {example.get('machine_id')}\n"
        f"Incident summary: {example.get('incident_body')}\n\n"
        f"Current telemetry:\n{telemetry_lines}\n\n"
        f"Retrieved incident evidence:\n{evidence_text}"
    )


def build_eval_dataset(
    corpus_path: Path = DEFAULT_CORPUS_PATH,
    output_path: Path = DEFAULT_DATASET_PATH,
    limit: int | None = None,
    evidence_limit: int = 3,
) -> dict[str, Any]:
    documents = load_jsonl(corpus_path)
    source_documents = [
        document for document in documents if document.get("document_type") == "incident_report"
    ]
    if limit is not None:
        source_documents = source_documents[:limit]

    examples = []
    for index, document in enumerate(source_documents, start=1):
        machine_id = str(document.get("machine_id", "unknown"))
        retrieved = evidence_documents_for_machine(
            documents,
            machine_id=machine_id,
            source_document_id=str(document.get("document_id", "")),
            limit=evidence_limit,
        )
        example = {
            "eval_id": f"industrial-incident-{index:04d}",
            "source_document_id": document.get("document_id"),
            "source_row_id": document.get("source_row_id"),
            "machine_id": machine_id,
            "machine_type": document.get("metadata", {}).get("machine_type"),
            "incident_title": document.get("title"),
            "incident_body": document.get("body"),
            "telemetry": telemetry(document),
            "retrieved_incident_evidence": [compact_document(item) for item in retrieved],
            "expected_failure_mode": failure_modes(document)[0] if failure_modes(document) else None,
            "expected_failure_modes": failure_modes(document),
            "expected_severity": expected_severity(document),
            "metadata": {
                "source_dataset": document.get("source_dataset"),
                "generated_at": utc_timestamp(),
                "corpus_path": str(corpus_path),
            },
        }
        example["prompt"] = build_candidate_prompt(example)
        examples.append(example)

    write_jsonl(output_path, examples)
    return {
        "dataset_path": str(output_path),
        "corpus_path": str(corpus_path),
        "examples": len(examples),
        "evidence_limit": evidence_limit,
    }


def request_headers(extra_headers: dict[str, str] | None = None) -> dict[str, str]:
    return {"Content-Type": "application/json", **(extra_headers or {})}


def post_json(
    endpoint: str,
    payload: dict[str, Any],
    timeout_seconds: float,
    headers: dict[str, str] | None = None,
) -> dict[str, Any]:
    request = urllib.request.Request(
        endpoint,
        data=json.dumps(payload).encode("utf-8"),
        headers=request_headers(headers),
        method="POST",
    )
    with urllib.request.urlopen(request, timeout=timeout_seconds) as response:
        body = response.read().decode("utf-8")
    parsed = json.loads(body)
    if not isinstance(parsed, dict):
        raise ValueError("Expected model endpoint to return a JSON object.")
    return parsed


def openai_headers() -> dict[str, str]:
    api_key = os.environ.get("OPENAI_API_KEY")
    return {"Authorization": f"Bearer {api_key}"} if api_key else {}


def call_ollama(
    endpoint: str,
    model: str,
    prompt: str,
    timeout_seconds: float,
    json_format: bool = False,
) -> str:
    payload: dict[str, Any] = {"model": model, "prompt": prompt, "stream": False}
    if json_format:
        payload["format"] = "json"
    response = post_json(endpoint, payload, timeout_seconds=timeout_seconds)
    return str(response.get("response", ""))


def call_openai_compatible(
    endpoint: str,
    model: str,
    prompt: str,
    timeout_seconds: float,
    json_format: bool = False,
) -> str:
    payload: dict[str, Any] = {
        "model": model,
        "messages": [{"role": "user", "content": prompt}],
        "stream": False,
    }
    if json_format:
        payload["response_format"] = {"type": "json_object"}
    response = post_json(
        endpoint,
        payload,
        timeout_seconds=timeout_seconds,
        headers=openai_headers(),
    )
    choices = response.get("choices", [])
    if not choices or not isinstance(choices[0], dict):
        return ""
    message = choices[0].get("message", {})
    return str(message.get("content", "")) if isinstance(message, dict) else ""


def call_model(
    provider: str,
    endpoint: str,
    model: str,
    prompt: str,
    timeout_seconds: float,
    json_format: bool = False,
) -> str:
    if provider == "ollama":
        return call_ollama(endpoint, model, prompt, timeout_seconds, json_format=json_format)
    if provider == "openai-compatible":
        return call_openai_compatible(
            endpoint,
            model,
            prompt,
            timeout_seconds,
            json_format=json_format,
        )
    raise ValueError(f"Unsupported provider: {provider}")


def dry_run_candidate_response(example: dict[str, Any], candidate_name: str) -> str:
    expected_mode = example.get("expected_failure_mode") or "unknown machine failure"
    severity = example.get("expected_severity") or "SEV3"
    action = "inspect the asset, validate telemetry bands, and document corrective action"
    if candidate_name == "lora":
        action = (
            "inspect the specific subsystem tied to the retrieved RCA, validate telemetry "
            "normalization, and keep the machine out of service until checks pass"
        )
    return (
        f"Root Cause: {expected_mode}.\n"
        f"Evidence: telemetry and retrieved incident records support {expected_mode}.\n"
        f"Actions: {action}.\n"
        f"Severity Reasoning: Suggested severity is {severity} based on failure label and evidence.\n"
        "Limitations: Dry-run response generated without model inference."
    )


def run_candidate_evaluation(
    dataset_path: Path,
    output_path: Path,
    candidate_name: str,
    model: str,
    provider: str = "ollama",
    endpoint: str = DEFAULT_OLLAMA_ENDPOINT,
    timeout_seconds: float = DEFAULT_TIMEOUT_SECONDS,
    limit: int | None = None,
    dry_run: bool = False,
) -> dict[str, Any]:
    examples = load_jsonl(dataset_path)
    if limit is not None:
        examples = examples[:limit]

    records = []
    for example in examples:
        started_at = time.perf_counter()
        error = None
        try:
            response_text = (
                dry_run_candidate_response(example, candidate_name)
                if dry_run
                else call_model(provider, endpoint, model, example["prompt"], timeout_seconds)
            )
        except (OSError, urllib.error.URLError, TimeoutError, json.JSONDecodeError, ValueError) as exc:
            response_text = ""
            error = str(exc)
        latency_ms = int((time.perf_counter() - started_at) * 1000)
        records.append(
            {
                "timestamp": utc_timestamp(),
                "eval_id": example["eval_id"],
                "candidate_name": candidate_name,
                "provider": provider,
                "endpoint": endpoint,
                "model": model,
                "prompt": example["prompt"],
                "response_text": response_text,
                "metadata": {
                    "latency_ms": latency_ms,
                    "success": error is None,
                    "error": error,
                    "dry_run": dry_run,
                    "source_document_id": example.get("source_document_id"),
                    "expected_failure_mode": example.get("expected_failure_mode"),
                    "expected_severity": example.get("expected_severity"),
                },
            }
        )

    write_jsonl(output_path, records)
    return {
        "candidate_name": candidate_name,
        "model": model,
        "output_path": str(output_path),
        "records": len(records),
        "successes": sum(1 for record in records if record["metadata"]["success"]),
    }


def parse_json_object(raw: str) -> dict[str, Any]:
    text = raw.strip()
    if text.startswith("```"):
        text = text.strip("`")
        if text.lower().startswith("json"):
            text = text[4:].strip()
    try:
        parsed = json.loads(text)
    except json.JSONDecodeError:
        start = text.find("{")
        end = text.rfind("}")
        if start == -1 or end == -1 or end <= start:
            raise
        parsed = json.loads(text[start : end + 1])
    if not isinstance(parsed, dict):
        raise ValueError("Expected a JSON object.")
    return parsed


def build_judge_prompt(example: dict[str, Any], candidate_record: dict[str, Any]) -> str:
    return (
        "You are GPT-OSS 20B acting as an LLM-as-Judge for industrial incident investigation.\n"
        "Score the candidate output against only the evaluation case and retrieved evidence.\n"
        "Return only valid JSON with keys: hallucination_score, rca_quality, actionability, "
        "severity_reasoning, rationale.\n"
        "Scales: hallucination_score is 1-5 where 1 means no unsupported claims and 5 means many "
        "unsupported claims. rca_quality, actionability, and severity_reasoning are 1-5 where 5 is best.\n\n"
        f"Expected failure modes: {example.get('expected_failure_modes')}\n"
        f"Expected severity: {example.get('expected_severity')}\n"
        f"Telemetry: {json.dumps(example.get('telemetry', {}), sort_keys=True)}\n"
        f"Retrieved evidence: {json.dumps(example.get('retrieved_incident_evidence', []), sort_keys=True)}\n\n"
        f"Candidate model: {candidate_record.get('model')}\n"
        f"Candidate output:\n{candidate_record.get('response_text', '')}"
    )


def clamp_score(value: Any) -> float:
    try:
        score = float(value)
    except (TypeError, ValueError):
        return 1.0
    return min(5.0, max(1.0, score))


def normalize_judge_scores(raw_scores: dict[str, Any]) -> dict[str, Any]:
    return {
        "hallucination_score": clamp_score(raw_scores.get("hallucination_score")),
        "rca_quality": clamp_score(raw_scores.get("rca_quality")),
        "actionability": clamp_score(raw_scores.get("actionability")),
        "severity_reasoning": clamp_score(raw_scores.get("severity_reasoning")),
        "rationale": str(raw_scores.get("rationale", "")),
    }


def dry_run_judge_scores(example: dict[str, Any], candidate_record: dict[str, Any]) -> dict[str, Any]:
    response = str(candidate_record.get("response_text", "")).lower()
    expected_mode = str(example.get("expected_failure_mode") or "").lower()
    expected_sev = str(example.get("expected_severity") or "").lower()
    mentions_mode = bool(expected_mode and expected_mode in response)
    mentions_severity = bool(expected_sev and expected_sev in response)
    action_words = ["inspect", "validate", "replace", "review", "confirm", "service"]
    actionability = 4.0 if sum(word in response for word in action_words) >= 2 else 2.0
    return {
        "hallucination_score": 1.0 if mentions_mode else 3.0,
        "rca_quality": 4.0 if mentions_mode else 2.0,
        "actionability": actionability,
        "severity_reasoning": 4.0 if mentions_severity else 2.0,
        "rationale": "Dry-run heuristic judge used for local pipeline validation.",
    }


def load_candidate_records(paths: list[Path]) -> list[dict[str, Any]]:
    records = []
    for path in paths:
        records.extend(load_jsonl(path))
    return records


def run_judge(
    dataset_path: Path,
    candidate_result_paths: list[Path],
    output_path: Path = DEFAULT_JUDGE_SCORES_PATH,
    judge_model: str = DEFAULT_JUDGE_MODEL,
    provider: str = "openai-compatible",
    endpoint: str = DEFAULT_OPENAI_COMPATIBLE_ENDPOINT,
    timeout_seconds: float = DEFAULT_TIMEOUT_SECONDS,
    limit: int | None = None,
    dry_run: bool = False,
) -> dict[str, Any]:
    examples = {example["eval_id"]: example for example in load_jsonl(dataset_path)}
    candidate_records = load_candidate_records(candidate_result_paths)
    if limit is not None:
        candidate_records = candidate_records[:limit]

    score_records = []
    for candidate_record in candidate_records:
        example = examples[candidate_record["eval_id"]]
        prompt = build_judge_prompt(example, candidate_record)
        started_at = time.perf_counter()
        error = None
        try:
            raw_scores = (
                dry_run_judge_scores(example, candidate_record)
                if dry_run
                else parse_json_object(
                    call_model(
                        provider,
                        endpoint,
                        judge_model,
                        prompt,
                        timeout_seconds,
                        json_format=True,
                    )
                )
            )
            scores = normalize_judge_scores(raw_scores)
        except (OSError, urllib.error.URLError, TimeoutError, json.JSONDecodeError, ValueError) as exc:
            error = str(exc)
            scores = {
                "hallucination_score": 5.0,
                "rca_quality": 1.0,
                "actionability": 1.0,
                "severity_reasoning": 1.0,
                "rationale": f"Judge failed; pessimistic scores assigned. Error: {error}",
            }
        latency_ms = int((time.perf_counter() - started_at) * 1000)
        score_records.append(
            {
                "timestamp": utc_timestamp(),
                "eval_id": candidate_record["eval_id"],
                "candidate_name": candidate_record["candidate_name"],
                "candidate_model": candidate_record["model"],
                "judge_model": judge_model,
                "judge_provider": provider,
                "judge_endpoint": endpoint,
                "scores": scores,
                "metadata": {
                    "latency_ms": latency_ms,
                    "success": error is None,
                    "error": error,
                    "dry_run": dry_run,
                },
            }
        )

    write_jsonl(output_path, score_records)
    return {
        "judge_model": judge_model,
        "output_path": str(output_path),
        "records": len(score_records),
        "successes": sum(1 for record in score_records if record["metadata"]["success"]),
    }


def mean(values: list[float]) -> float | None:
    return round(statistics.fmean(values), 4) if values else None


def score_distribution(values: list[float]) -> dict[str, int]:
    counts = Counter(str(int(round(value))) for value in values)
    return {str(score): counts.get(str(score), 0) for score in range(1, 6)}


def improvement_percentage(base: float | None, lora: float | None, metric: str) -> float | None:
    if base is None or lora is None or base == 0:
        return None
    delta = lora - base if HIGHER_IS_BETTER[metric] else base - lora
    return round((delta / abs(base)) * 100, 2)


def summarize_scores(
    judge_scores_path: Path = DEFAULT_JUDGE_SCORES_PATH,
    summary_json_path: Path = DEFAULT_SUMMARY_JSON_PATH,
    summary_csv_path: Path = DEFAULT_SUMMARY_CSV_PATH,
) -> dict[str, Any]:
    records = load_jsonl(judge_scores_path)
    by_candidate: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for record in records:
        by_candidate[record["candidate_name"]].append(record)

    candidate_summaries = {}
    for candidate_name, candidate_records in sorted(by_candidate.items()):
        metrics = {}
        for field in SCORE_FIELDS:
            values = [float(record["scores"][field]) for record in candidate_records]
            metrics[field] = {
                "mean": mean(values),
                "distribution": score_distribution(values),
            }
        candidate_summaries[candidate_name] = {
            "examples_evaluated": len(candidate_records),
            "model": candidate_records[0].get("candidate_model") if candidate_records else None,
            "metrics": metrics,
        }

    improvements = {}
    if "base" in candidate_summaries and "lora" in candidate_summaries:
        for field in SCORE_FIELDS:
            improvements[field] = improvement_percentage(
                candidate_summaries["base"]["metrics"][field]["mean"],
                candidate_summaries["lora"]["metrics"][field]["mean"],
                field,
            )

    judge_models = sorted({str(record.get("judge_model")) for record in records})
    summary = {
        "generated_at": utc_timestamp(),
        "examples_evaluated": len({record["eval_id"] for record in records}),
        "judge": {
            "models": judge_models,
            "provider": records[0].get("judge_provider") if records else None,
            "endpoint": records[0].get("judge_endpoint") if records else None,
        },
        "candidates": candidate_summaries,
        "improvement_percentage": improvements,
    }

    summary_json_path.parent.mkdir(parents=True, exist_ok=True)
    summary_json_path.write_text(json.dumps(summary, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    write_summary_csv(summary, summary_csv_path)
    return summary


def write_summary_csv(summary: dict[str, Any], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as output_file:
        writer = csv.DictWriter(
            output_file,
            fieldnames=[
                "candidate",
                "model",
                "metric",
                "mean",
                "score_1_count",
                "score_2_count",
                "score_3_count",
                "score_4_count",
                "score_5_count",
                "lora_improvement_pct",
            ],
        )
        writer.writeheader()
        improvements = summary.get("improvement_percentage", {})
        for candidate, candidate_summary in summary.get("candidates", {}).items():
            for metric, metric_summary in candidate_summary["metrics"].items():
                distribution = metric_summary["distribution"]
                writer.writerow(
                    {
                        "candidate": candidate,
                        "model": candidate_summary.get("model"),
                        "metric": metric,
                        "mean": metric_summary["mean"],
                        "score_1_count": distribution["1"],
                        "score_2_count": distribution["2"],
                        "score_3_count": distribution["3"],
                        "score_4_count": distribution["4"],
                        "score_5_count": distribution["5"],
                        "lora_improvement_pct": improvements.get(metric),
                    }
                )


def markdown_table(headers: list[str], rows: list[list[Any]]) -> str:
    header = "| " + " | ".join(headers) + " |"
    separator = "| " + " | ".join(["---"] * len(headers)) + " |"
    body = ["| " + " | ".join(str(value) for value in row) + " |" for row in rows]
    return "\n".join([header, separator, *body])


def generate_report(
    summary_json_path: Path = DEFAULT_SUMMARY_JSON_PATH,
    output_path: Path = DEFAULT_REPORT_PATH,
) -> dict[str, Any]:
    summary = json.loads(summary_json_path.read_text(encoding="utf-8"))
    metric_rows = []
    for metric in SCORE_FIELDS:
        base_mean = summary.get("candidates", {}).get("base", {}).get("metrics", {}).get(metric, {}).get("mean")
        lora_mean = summary.get("candidates", {}).get("lora", {}).get("metrics", {}).get(metric, {}).get("mean")
        metric_rows.append(
            [
                metric,
                base_mean,
                lora_mean,
                summary.get("improvement_percentage", {}).get(metric),
            ]
        )

    candidate_rows = [
        [
            candidate,
            details.get("model"),
            details.get("examples_evaluated"),
        ]
        for candidate, details in summary.get("candidates", {}).items()
    ]
    judge = summary.get("judge", {})
    report = (
        "# AMD Hackathon LLM-as-Judge Evaluation\n\n"
        "## Executive Summary\n\n"
        f"- Examples evaluated: {summary.get('examples_evaluated')}\n"
        f"- Judge model: {', '.join(judge.get('models', []))}\n"
        f"- Judge provider: {judge.get('provider')}\n"
        "- Hallucination score is lower-is-better; all other metrics are higher-is-better.\n\n"
        "## Candidate Models\n\n"
        f"{markdown_table(['Candidate', 'Model', 'Examples'], candidate_rows)}\n\n"
        "## Mean Scores\n\n"
        f"{markdown_table(['Metric', 'Gemma Base Mean', 'Gemma LoRA Mean', 'LoRA Improvement %'], metric_rows)}\n\n"
        "## Key Findings\n\n"
        "- Use the mean-score table directly in PowerPoint to compare base and LoRA behavior.\n"
        "- Use hallucination score to show evidence grounding; lower LoRA values indicate better grounding.\n"
        "- Use RCA quality, actionability, and severity reasoning to show domain-specific investigation gains.\n\n"
        "## Conclusions\n\n"
        "- The pipeline preserves prompts, responses, judge scores, and model metadata for auditability.\n"
        "- Results are generated from local JSONL artifacts and can run on a laptop or AMD Cloud instance.\n"
    )
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(report, encoding="utf-8")
    return {"report_path": str(output_path)}


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Run AMD hackathon LLM-as-Judge evaluations.")
    subparsers = parser.add_subparsers(dest="command", required=True)

    dataset = subparsers.add_parser("build-dataset", help="Generate eval JSONL from incident corpus.")
    dataset.add_argument("--corpus-path", type=Path, default=DEFAULT_CORPUS_PATH)
    dataset.add_argument("--output", type=Path, default=DEFAULT_DATASET_PATH)
    dataset.add_argument("--limit", type=int, default=None)
    dataset.add_argument("--evidence-limit", type=int, default=3)

    candidate = subparsers.add_parser("run-candidate", help="Run one candidate model.")
    candidate.add_argument("--dataset", type=Path, default=DEFAULT_DATASET_PATH)
    candidate.add_argument("--output", type=Path, required=True)
    candidate.add_argument("--candidate-name", choices=["base", "lora"], required=True)
    candidate.add_argument("--model", required=True)
    candidate.add_argument("--provider", choices=["ollama", "openai-compatible"], default="ollama")
    candidate.add_argument("--endpoint", default=DEFAULT_OLLAMA_ENDPOINT)
    candidate.add_argument("--timeout-seconds", type=float, default=DEFAULT_TIMEOUT_SECONDS)
    candidate.add_argument("--limit", type=int, default=None)
    candidate.add_argument("--dry-run", action="store_true")

    judge = subparsers.add_parser("judge", help="Judge candidate outputs.")
    judge.add_argument("--dataset", type=Path, default=DEFAULT_DATASET_PATH)
    judge.add_argument("--base-results", type=Path, default=DEFAULT_BASE_RESULTS_PATH)
    judge.add_argument("--lora-results", type=Path, default=DEFAULT_LORA_RESULTS_PATH)
    judge.add_argument("--output", type=Path, default=DEFAULT_JUDGE_SCORES_PATH)
    judge.add_argument("--judge-model", default=DEFAULT_JUDGE_MODEL)
    judge.add_argument("--provider", choices=["ollama", "openai-compatible"], default="openai-compatible")
    judge.add_argument("--endpoint", default=DEFAULT_OPENAI_COMPATIBLE_ENDPOINT)
    judge.add_argument("--timeout-seconds", type=float, default=DEFAULT_TIMEOUT_SECONDS)
    judge.add_argument("--limit", type=int, default=None)
    judge.add_argument("--dry-run", action="store_true")

    summarize = subparsers.add_parser("summarize", help="Write summary JSON and CSV.")
    summarize.add_argument("--judge-scores", type=Path, default=DEFAULT_JUDGE_SCORES_PATH)
    summarize.add_argument("--summary-json", type=Path, default=DEFAULT_SUMMARY_JSON_PATH)
    summarize.add_argument("--summary-csv", type=Path, default=DEFAULT_SUMMARY_CSV_PATH)

    report = subparsers.add_parser("report", help="Generate slide-ready markdown report.")
    report.add_argument("--summary-json", type=Path, default=DEFAULT_SUMMARY_JSON_PATH)
    report.add_argument("--output", type=Path, default=DEFAULT_REPORT_PATH)

    run_all = subparsers.add_parser("run-all", help="Run dataset, candidates, judge, summary, report.")
    run_all.add_argument("--corpus-path", type=Path, default=DEFAULT_CORPUS_PATH)
    run_all.add_argument("--eval-dir", type=Path, default=DEFAULT_EVAL_DIR)
    run_all.add_argument("--limit", type=int, default=None)
    run_all.add_argument("--base-model", default=DEFAULT_BASE_MODEL)
    run_all.add_argument("--lora-model", default=DEFAULT_LORA_MODEL)
    run_all.add_argument("--candidate-provider", choices=["ollama", "openai-compatible"], default="ollama")
    run_all.add_argument("--candidate-endpoint", default=DEFAULT_OLLAMA_ENDPOINT)
    run_all.add_argument("--judge-model", default=DEFAULT_JUDGE_MODEL)
    run_all.add_argument("--judge-provider", choices=["ollama", "openai-compatible"], default="openai-compatible")
    run_all.add_argument("--judge-endpoint", default=DEFAULT_OPENAI_COMPATIBLE_ENDPOINT)
    run_all.add_argument("--timeout-seconds", type=float, default=DEFAULT_TIMEOUT_SECONDS)
    run_all.add_argument("--dry-run", action="store_true")
    return parser


def main() -> None:
    args = build_parser().parse_args()
    if getattr(args, "limit", None) is not None and args.limit < 1:
        raise SystemExit("--limit must be >= 1")

    if args.command == "build-dataset":
        result = build_eval_dataset(args.corpus_path, args.output, args.limit, args.evidence_limit)
    elif args.command == "run-candidate":
        result = run_candidate_evaluation(
            dataset_path=args.dataset,
            output_path=args.output,
            candidate_name=args.candidate_name,
            model=args.model,
            provider=args.provider,
            endpoint=args.endpoint,
            timeout_seconds=args.timeout_seconds,
            limit=args.limit,
            dry_run=args.dry_run,
        )
    elif args.command == "judge":
        result = run_judge(
            dataset_path=args.dataset,
            candidate_result_paths=[args.base_results, args.lora_results],
            output_path=args.output,
            judge_model=args.judge_model,
            provider=args.provider,
            endpoint=args.endpoint,
            timeout_seconds=args.timeout_seconds,
            limit=args.limit,
            dry_run=args.dry_run,
        )
    elif args.command == "summarize":
        result = summarize_scores(args.judge_scores, args.summary_json, args.summary_csv)
    elif args.command == "report":
        result = generate_report(args.summary_json, args.output)
    elif args.command == "run-all":
        dataset_path = args.eval_dir / "eval_dataset.jsonl"
        base_path = args.eval_dir / "base_results.jsonl"
        lora_path = args.eval_dir / "lora_results.jsonl"
        judge_path = args.eval_dir / "judge_scores.jsonl"
        summary_json_path = args.eval_dir / "summary.json"
        summary_csv_path = args.eval_dir / "summary.csv"
        report_path = args.eval_dir / "llm_judge_report.md"
        result = {
            "dataset": build_eval_dataset(args.corpus_path, dataset_path, args.limit),
            "base": run_candidate_evaluation(
                dataset_path,
                base_path,
                "base",
                args.base_model,
                args.candidate_provider,
                args.candidate_endpoint,
                args.timeout_seconds,
                dry_run=args.dry_run,
            ),
            "lora": run_candidate_evaluation(
                dataset_path,
                lora_path,
                "lora",
                args.lora_model,
                args.candidate_provider,
                args.candidate_endpoint,
                args.timeout_seconds,
                dry_run=args.dry_run,
            ),
            "judge": run_judge(
                dataset_path,
                [base_path, lora_path],
                judge_path,
                args.judge_model,
                args.judge_provider,
                args.judge_endpoint,
                args.timeout_seconds,
                dry_run=args.dry_run,
            ),
        }
        result["summary"] = summarize_scores(judge_path, summary_json_path, summary_csv_path)
        result["report"] = generate_report(summary_json_path, report_path)
    else:
        raise SystemExit(f"Unknown command: {args.command}")

    print(json.dumps(result, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
