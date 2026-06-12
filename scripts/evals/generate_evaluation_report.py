#!/usr/bin/env python3
"""Generate a complete file-based evaluation package."""

from __future__ import annotations

import argparse
import csv
import html
import json
import math
import shutil
import subprocess
import textwrap
from collections import Counter, defaultdict
from datetime import UTC, datetime
from pathlib import Path
from typing import Any


NOT_AVAILABLE = "NOT AVAILABLE"
PROJECT_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_OUTPUT_DIR = PROJECT_ROOT / "data" / "evals" / "full_run"


def utc_now() -> str:
    return datetime.now(UTC).isoformat()


def load_json(path: Path) -> dict[str, Any] | None:
    if not path.exists():
        return None
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except json.JSONDecodeError:
        return None


def load_jsonl(path: Path) -> list[dict[str, Any]]:
    if not path.exists():
        return []
    records = []
    for line in path.read_text(encoding="utf-8").splitlines():
        if line.strip():
            try:
                records.append(json.loads(line))
            except json.JSONDecodeError:
                continue
    return records


def read_csv(path: Path) -> list[dict[str, str]]:
    if not path.exists():
        return []
    with path.open(encoding="utf-8", newline="") as input_file:
        return list(csv.DictReader(input_file))


def na(reason: str) -> dict[str, str]:
    return {"value": NOT_AVAILABLE, "reason": reason}


def numeric(value: Any) -> float | None:
    try:
        if value is None or value == NOT_AVAILABLE:
            return None
        result = float(value)
    except (TypeError, ValueError):
        return None
    if math.isnan(result) or math.isinf(result):
        return None
    return result


def pct(value: Any) -> str:
    number = numeric(value)
    return NOT_AVAILABLE if number is None else f"{number * 100:.1f}%"


def metric_value(value: Any, reason: str) -> Any:
    return na(reason) if value is None else value


def annotate_known_unavailable(summary: dict[str, Any]) -> None:
    system = summary["sections"].get("system_hardware_metrics", {}).get("system_metrics", {})
    runtime = system.get("runtime", {})
    if runtime.get("cuda_version") == NOT_AVAILABLE:
        runtime["cuda_version"] = na("CUDA is not available in the captured runtime.")
    if runtime.get("rocm_version") == NOT_AVAILABLE:
        runtime["rocm_version"] = na("ROCm/HIP is not available in the captured runtime.")
    gpu_metrics = system.get("gpu_metrics", {})
    for key in ("peak_vram_gb", "average_vram_gb"):
        if gpu_metrics.get(key) == NOT_AVAILABLE:
            gpu_metrics[key] = na("GPU VRAM telemetry was not returned by nvidia-smi or rocm-smi.")
    timing = system.get("timing", {})
    timing_reasons = {
        "training_runtime_seconds": "No training command was supplied to run_full_evaluation.py.",
        "inference_runtime_seconds": "No inference command was supplied to run_full_evaluation.py.",
        "benchmark_runtime_seconds": "No benchmark command was supplied to run_full_evaluation.py.",
    }
    for key, reason in timing_reasons.items():
        if timing.get(key) == NOT_AVAILABLE:
            timing[key] = na(reason)


def display_value(value: Any) -> str:
    if isinstance(value, dict) and set(value.keys()) == {"value", "reason"}:
        return f"{value['value']} - {value['reason']}"
    if isinstance(value, dict):
        return json.dumps(value, sort_keys=True)
    if isinstance(value, list):
        return json.dumps(value)
    return str(value)


def ensure_dirs(output_dir: Path) -> Path:
    charts_dir = output_dir / "charts"
    charts_dir.mkdir(parents=True, exist_ok=True)
    return charts_dir


def svg_bar_chart(
    title: str,
    rows: list[tuple[str, float]],
    output_path: Path,
    unit: str = "",
    width: int = 920,
    bar_height: int = 34,
) -> None:
    if not rows:
        output_path.write_text(empty_svg(title, "No measured values available."), encoding="utf-8")
        return
    max_value = max(value for _, value in rows) or 1.0
    height = 92 + len(rows) * (bar_height + 18)
    labels_width = 260
    plot_width = width - labels_width - 90
    parts = [
        f'<svg xmlns="http://www.w3.org/2000/svg" width="{width}" height="{height}" viewBox="0 0 {width} {height}">',
        '<rect width="100%" height="100%" fill="#ffffff"/>',
        f'<text x="24" y="34" font-family="Arial" font-size="22" font-weight="700" fill="#111827">{html.escape(title)}</text>',
    ]
    palette = ["#2563eb", "#059669", "#dc2626", "#7c3aed", "#ea580c", "#0891b2"]
    for index, (label, value) in enumerate(rows):
        y = 70 + index * (bar_height + 18)
        bar_width = max(2, int((value / max_value) * plot_width))
        color = palette[index % len(palette)]
        parts.extend(
            [
                f'<text x="24" y="{y + 23}" font-family="Arial" font-size="14" fill="#374151">{html.escape(label[:38])}</text>',
                f'<rect x="{labels_width}" y="{y}" width="{bar_width}" height="{bar_height}" rx="3" fill="{color}"/>',
                f'<text x="{labels_width + bar_width + 10}" y="{y + 23}" font-family="Arial" font-size="14" fill="#111827">{value:.3g}{html.escape(unit)}</text>',
            ]
        )
    parts.append("</svg>")
    output_path.write_text("\n".join(parts), encoding="utf-8")


def empty_svg(title: str, message: str) -> str:
    return "\n".join(
        [
            '<svg xmlns="http://www.w3.org/2000/svg" width="920" height="180" viewBox="0 0 920 180">',
            '<rect width="100%" height="100%" fill="#ffffff"/>',
            f'<text x="24" y="42" font-family="Arial" font-size="22" font-weight="700" fill="#111827">{html.escape(title)}</text>',
            f'<text x="24" y="92" font-family="Arial" font-size="16" fill="#6b7280">{html.escape(message)}</text>',
            "</svg>",
        ]
    )


def copy_vision_examples(charts_dir: Path) -> list[str]:
    source_dir = PROJECT_ROOT / "data" / "vision_examples"
    output_paths = []
    if not source_dir.exists():
        return output_paths
    for source in sorted(source_dir.glob("*.png")):
        destination = charts_dir / source.name
        shutil.copy2(source, destination)
        output_paths.append(str(destination))
    return output_paths


def telemetry_metrics() -> dict[str, Any]:
    metrics = load_json(PROJECT_ROOT / "models" / "telemetry_metrics.json")
    if not metrics:
        reason = "Telemetry model metrics file models/telemetry_metrics.json is not present in this checkout."
        return {
            "accuracy": na(reason),
            "precision": na(reason),
            "recall": na(reason),
            "f1": na(reason),
            "roc_auc": na(reason),
            "pr_auc": na(reason),
            "confusion_matrix": na(reason),
            "shap_global_importance": na("SHAP was not run; no SHAP artifact was found."),
            "shap_local_example": na("SHAP was not run; no SHAP artifact was found."),
        }
    threshold = metrics.get("threshold_metrics", {}).get("0.5", {})
    confusion = threshold.get("confusion_matrix", {})
    total = sum(int(confusion.get(key, 0)) for key in ("tn", "fp", "fn", "tp"))
    accuracy = None
    if total:
        accuracy = (int(confusion.get("tn", 0)) + int(confusion.get("tp", 0))) / total
    return {
        "accuracy": metric_value(accuracy, "Missing confusion matrix at threshold 0.5."),
        "precision": metric_value(threshold.get("precision"), "Missing precision at threshold 0.5."),
        "recall": metric_value(threshold.get("recall"), "Missing recall at threshold 0.5."),
        "f1": metric_value(threshold.get("f1"), "Missing F1 at threshold 0.5."),
        "roc_auc": metric_value(metrics.get("test_roc_auc"), "Missing telemetry ROC-AUC."),
        "pr_auc": metric_value(metrics.get("test_average_precision"), "Missing telemetry PR-AUC."),
        "confusion_matrix": metric_value(confusion or None, "Missing telemetry confusion matrix."),
        "shap_global_importance": na("SHAP was not run; no SHAP artifact was found."),
        "shap_local_example": na("SHAP was not run; no SHAP artifact was found."),
    }


def vision_metrics(charts_dir: Path) -> dict[str, Any]:
    examples = copy_vision_examples(charts_dir)
    reason = "No persisted aggregate vision metrics artifact was found; example heatmaps/overlays were copied when available."
    return {
        "accuracy": na(reason),
        "precision": na(reason),
        "recall": na(reason),
        "f1": na(reason),
        "defect_examples": examples,
        "good_examples": examples,
        "heatmaps": [path for path in examples if "heatmap" in path],
        "overlays": [path for path in examples if "annotated" in path],
    }


def retrieval_metrics(charts_dir: Path) -> dict[str, Any]:
    corpus = load_jsonl(PROJECT_ROOT / "data" / "incidents" / "ai4i_incident_corpus.jsonl")
    document_ids = [str(item.get("document_id")) for item in corpus if item.get("document_id")]
    duplicate_count = sum(count - 1 for count in Counter(document_ids).values() if count > 1)
    duplicate_rate = duplicate_count / len(document_ids) if document_ids else None
    reason = "No labeled retrieval relevance set was found, so ranking metrics cannot be measured."
    svg_bar_chart(
        "Retrieval Corpus Profile",
        [("corpus size", float(len(corpus))), ("duplicate rate x100", float((duplicate_rate or 0) * 100))],
        charts_dir / "retrieval_corpus_profile.svg",
    )
    return {
        "recall_at_k": na(reason),
        "precision_at_k": na(reason),
        "mrr": na(reason),
        "ndcg": na(reason),
        "retrieval_latency_ms": na("No persisted retrieval latency artifact was found."),
        "corpus_size": len(corpus) if corpus else na("Incident corpus file is missing."),
        "duplicate_rate": metric_value(duplicate_rate, "Incident corpus file is missing."),
    }


def llm_metrics(charts_dir: Path) -> dict[str, Any]:
    summary = load_json(PROJECT_ROOT / "data" / "evals" / "summary.json") or {}
    candidates = summary.get("candidates", {})
    lora = candidates.get("lora") or candidates.get("base") or {}
    metrics = lora.get("metrics", {})
    hallucination = metrics.get("hallucination_score", {}).get("mean")
    relevance = metrics.get("rca_quality", {}).get("mean")
    groundedness = None
    if hallucination is not None:
        groundedness = max(0.0, 6.0 - float(hallucination))
    rows = []
    for candidate_name, candidate in candidates.items():
        for metric_name, metric in candidate.get("metrics", {}).items():
            value = numeric(metric.get("mean"))
            if value is not None:
                rows.append((f"{candidate_name} {metric_name}", value))
    svg_bar_chart("LLM Judge Mean Scores", rows, charts_dir / "llm_judge_scores.svg")
    return {
        "hallucination_score": metric_value(hallucination, "No LLM judge summary was found."),
        "groundedness": metric_value(groundedness, "Groundedness inferred from hallucination score was unavailable."),
        "relevance": metric_value(relevance, "No RCA/relevance judge score was found."),
        "citation_accuracy": na("The current judge rubric does not score citation accuracy."),
        "first_token_latency_ms": na("Candidate generation traces do not include first-token latency."),
        "full_response_latency_ms": na("Candidate generation traces do not include aggregate response latency."),
        "prompt_tokens": na("Token accounting was not emitted by the local model endpoint."),
        "completion_tokens": na("Token accounting was not emitted by the local model endpoint."),
        "total_tokens": na("Token accounting was not emitted by the local model endpoint."),
        "judge": summary.get("judge", {}),
        "examples_evaluated": summary.get("examples_evaluated", na("No LLM judge summary was found.")),
    }


def agent_metrics(charts_dir: Path) -> dict[str, Any]:
    scenarios = load_json(PROJECT_ROOT / "data" / "evaluation" / "scenarios.json") or []
    scenario_count = len(scenarios) if isinstance(scenarios, list) else 0
    reason = "No persisted agent trace artifact was found."
    (charts_dir / "agent_trace_placeholder.svg").write_text(
        empty_svg("Agent Trace Visualization", reason), encoding="utf-8"
    )
    return {
        "workflow_completion_rate": na(reason),
        "scenario_pass_rate": na(reason),
        "structured_output_success_rate": na(reason),
        "tool_call_success_rate": na(reason),
        "retry_count": na(reason),
        "average_steps": na(reason),
        "average_runtime_seconds": na(reason),
        "scenario_count": scenario_count,
        "trace_visualizations": [str(charts_dir / "agent_trace_placeholder.svg")],
        "failure_breakdown": na(reason),
    }


def policy_metrics(charts_dir: Path) -> dict[str, Any]:
    scenarios = load_json(PROJECT_ROOT / "data" / "evaluation" / "scenarios.json") or []
    if not isinstance(scenarios, list) or not scenarios:
        reason = "Policy scenario file is missing."
        return {
            "severity_accuracy": na(reason),
            "confusion_matrix": na(reason),
            "false_sev1": na(reason),
            "missed_sev1": na(reason),
            "override_rate": na("No override log artifact was found."),
            "escalation_rate": na("No escalation log artifact was found."),
        }
    try:
        from industrial_ai.policy.severity import assign_severity
    except Exception:
        reason = "industrial_ai.policy.severity could not be imported."
        return {
            "severity_accuracy": na(reason),
            "confusion_matrix": na(reason),
            "false_sev1": na(reason),
            "missed_sev1": na(reason),
            "override_rate": na("No override log artifact was found."),
            "escalation_rate": na("No escalation log artifact was found."),
        }
    pairs = []
    for scenario in scenarios:
        decision = assign_severity(
            failure_probability=float(scenario["failure_probability"]),
            rag_confidence=str(scenario["rag_confidence"]),
        )
        pairs.append((str(scenario["expected_severity"]), str(decision.severity.value)))
    labels = sorted({label for pair in pairs for label in pair})
    matrix = {
        expected: {actual: sum(1 for exp, act in pairs if exp == expected and act == actual) for actual in labels}
        for expected in labels
    }
    accuracy = sum(1 for exp, act in pairs if exp == act) / len(pairs)
    false_sev1 = sum(1 for exp, act in pairs if exp != "SEV1" and act == "SEV1")
    missed_sev1 = sum(1 for exp, act in pairs if exp == "SEV1" and act != "SEV1")
    svg_bar_chart(
        "Policy Severity Outcomes",
        [("accuracy", accuracy), ("false SEV1", float(false_sev1)), ("missed SEV1", float(missed_sev1))],
        charts_dir / "policy_severity.svg",
    )
    return {
        "severity_accuracy": accuracy,
        "confusion_matrix": matrix,
        "false_sev1": false_sev1,
        "missed_sev1": missed_sev1,
        "override_rate": na("No override log artifact was found."),
        "escalation_rate": na("No escalation log artifact was found."),
    }


def lora_metrics(charts_dir: Path) -> dict[str, Any]:
    training = load_json(PROJECT_ROOT / "data" / "amd" / "lora" / "training_metrics.json") or {}
    judge = load_json(PROJECT_ROOT / "data" / "evals" / "summary.json") or {}
    adapter_dir = PROJECT_ROOT / str(training.get("adapter_dir", ""))
    adapter_size = None
    if adapter_dir.exists():
        adapter_size = sum(path.stat().st_size for path in adapter_dir.glob("**/*") if path.is_file()) / 1024**2
    candidates = judge.get("candidates", {})
    rows = []
    for candidate_name, candidate in candidates.items():
        for metric_name in ("hallucination_score", "rca_quality", "actionability", "severity_reasoning"):
            value = numeric(candidate.get("metrics", {}).get(metric_name, {}).get("mean"))
            if value is not None:
                rows.append((f"{candidate_name} {metric_name}", value))
    svg_bar_chart("Base vs LoRA Judge Scores", rows, charts_dir / "lora_side_by_side.svg")
    return {
        "base_model": candidates.get("base", {}).get("model", training.get("model_name", NOT_AVAILABLE)),
        "lora_model": candidates.get("lora", {}).get("model", training.get("adapter_dir", NOT_AVAILABLE)),
        "adapter_size_mb": metric_value(None if adapter_size is None else round(adapter_size, 2), "Adapter directory was not found."),
        "train_loss": metric_value(training.get("train_metrics", {}).get("train_loss"), "No LoRA training metrics found."),
        "eval_loss": metric_value(training.get("eval_metrics", {}).get("eval_loss"), "No LoRA eval metrics found."),
        "train_runtime_seconds": metric_value(training.get("train_metrics", {}).get("train_runtime"), "No LoRA train runtime found."),
        "peak_vram_gb": na("LoRA training did not persist peak VRAM in training_metrics.json."),
        "gpu_hours": metric_value(
            (numeric(training.get("train_metrics", {}).get("train_runtime")) or 0) / 3600
            if training.get("train_metrics")
            else None,
            "No LoRA train runtime found.",
        ),
        "judge_model": (judge.get("judge", {}).get("models") or [NOT_AVAILABLE])[0],
        "examples": judge.get("examples_evaluated", na("No judge summary found.")),
        "successes": judge.get("examples_evaluated", na("No judge summary found.")),
        "quality": {
            "hallucination_score": candidates.get("lora", {}).get("metrics", {}).get("hallucination_score", {}).get("mean", NOT_AVAILABLE),
            "rca_quality": candidates.get("lora", {}).get("metrics", {}).get("rca_quality", {}).get("mean", NOT_AVAILABLE),
            "actionability": candidates.get("lora", {}).get("metrics", {}).get("actionability", {}).get("mean", NOT_AVAILABLE),
            "severity_reasoning": candidates.get("lora", {}).get("metrics", {}).get("severity_reasoning", {}).get("mean", NOT_AVAILABLE),
        },
    }


def benchmark_metrics(charts_dir: Path) -> dict[str, Any]:
    kernel = load_json(PROJECT_ROOT / "data" / "benchmarks" / "kernel_comparison.json") or {}
    fused = load_json(PROJECT_ROOT / "data" / "benchmarks" / "rocm_fused_rerank_results.json") or {}
    kernel_results = kernel.get("results", [])
    fused_results = fused.get("results", [])
    latency_rows = []
    speedup_rows = []
    for item in kernel_results:
        latency = numeric(item.get("latency_ms"))
        speedup = numeric(item.get("speedup_vs_baseline"))
        label = f"{item.get('implementation_name')} {item.get('precision')}"
        if latency is not None:
            latency_rows.append((label, latency))
        if speedup is not None:
            speedup_rows.append((label, speedup))
    svg_bar_chart("AMD Kernel Latency", latency_rows, charts_dir / "amd_kernel_latency.svg", unit=" ms")
    svg_bar_chart("AMD Kernel Speedup", speedup_rows, charts_dir / "amd_kernel_speedup.svg", unit="x")
    return {
        "rocm_benchmark": {
            "metadata": kernel.get("metadata", {}),
            "results": kernel_results,
            "fused_rerank_results": fused_results,
        },
        "required_coverage": {
            "pytorch_eager": any(item.get("implementation_name") == "pytorch_eager" for item in kernel_results),
            "torch_compile": any("compile" in str(item.get("implementation_name", "")).lower() for item in kernel_results),
            "rocblas": any("rocblas" in str(item.get("implementation_name", "")).lower() for item in kernel_results),
            "triton": any("triton" in str(item.get("implementation_name", "")).lower() for item in kernel_results),
            "bf16": any(item.get("precision") == "bf16" for item in kernel_results + fused_results),
            "fp16": any(item.get("precision") == "fp16" or item.get("mode") == "fp16" for item in kernel_results + fused_results),
            "fp32": any(item.get("precision") == "fp32" or item.get("mode") == "fp32" for item in kernel_results + fused_results),
            "fp8": any(item.get("precision") == "fp8" or item.get("mode") == "fp8" for item in kernel_results + fused_results),
        },
    }


def business_metrics() -> dict[str, Any]:
    return {
        "label": "Illustrative estimates only; not measured production impact.",
        "rca_time_reduction": "30-50% illustrative estimate",
        "documentation_search_reduction": "40-60% illustrative estimate",
        "estimated_downtime_avoided": "Scenario-dependent illustrative estimate",
        "estimated_cost_savings": "Scenario-dependent illustrative estimate",
    }


def lineage(hardware: dict[str, Any], system: dict[str, Any], lora: dict[str, Any], llm: dict[str, Any]) -> dict[str, Any]:
    return {
        "model_lineage": {
            "base_model": lora.get("base_model", NOT_AVAILABLE),
            "lora_model": lora.get("lora_model", NOT_AVAILABLE),
            "judge_model": lora.get("judge_model", NOT_AVAILABLE),
            "precision": (load_json(PROJECT_ROOT / "data" / "evals" / "summary.json") or {})
            .get("environment", {})
            .get("precision", NOT_AVAILABLE),
        },
        "hardware_lineage": {
            "current_capture": hardware,
            "runtime": system.get("runtime", {}),
            "local_rtx_5070_results": na("No artifact explicitly labeled Local RTX 5070 was found."),
            "amd_mi300x_results": (load_json(PROJECT_ROOT / "data" / "evals" / "summary.json") or {})
            .get("environment", {})
            .get("hardware", NOT_AVAILABLE),
        },
    }


def flatten_metrics(section: str, value: Any, rows: list[dict[str, Any]], prefix: str = "") -> None:
    if isinstance(value, dict) and set(value.keys()) == {"value", "reason"}:
        rows.append({"section": section, "metric": prefix, "value": value["value"], "reason": value["reason"]})
        return
    if isinstance(value, dict):
        for key, nested in value.items():
            flatten_metrics(section, nested, rows, f"{prefix}.{key}" if prefix else str(key))
        return
    if isinstance(value, list):
        rows.append({"section": section, "metric": prefix, "value": json.dumps(value), "reason": ""})
        return
    rows.append({"section": section, "metric": prefix, "value": value, "reason": ""})


def completeness(summary: dict[str, Any]) -> dict[str, Any]:
    rows: list[dict[str, Any]] = []
    for section, value in summary["sections"].items():
        flatten_metrics(section, value, rows)
    measured = [row for row in rows if row["value"] != NOT_AVAILABLE]
    missing = [row for row in rows if row["value"] == NOT_AVAILABLE]
    percent = round((len(measured) / len(rows)) * 100, 1) if rows else 0.0
    return {
        "total_metric_fields": len(rows),
        "measured_metric_fields": len(measured),
        "missing_metric_fields": len(missing),
        "estimated_report_completeness_percent": percent,
        "missing": missing,
    }


def write_csv_summary(summary: dict[str, Any], output_path: Path) -> None:
    rows: list[dict[str, Any]] = []
    for section, value in summary["sections"].items():
        flatten_metrics(section, value, rows)
    with output_path.open("w", encoding="utf-8", newline="") as output_file:
        writer = csv.DictWriter(output_file, fieldnames=["section", "metric", "value", "reason"])
        writer.writeheader()
        writer.writerows(rows)


def markdown_table(rows: list[list[Any]], headers: list[str]) -> str:
    def cell(value: Any) -> str:
        return str(value).replace("\n", " ").replace("|", "\\|")

    lines = [
        "| " + " | ".join(headers) + " |",
        "| " + " | ".join("---" for _ in headers) + " |",
    ]
    for row in rows:
        lines.append("| " + " | ".join(cell(item) for item in row) + " |")
    return "\n".join(lines)


def render_report(summary: dict[str, Any], charts: list[str]) -> str:
    sections = summary["sections"]
    lora = sections["lora"]
    llm = sections["llm"]
    benchmark = sections["amd_gpu_benchmarks"]
    best_benchmark = None
    for item in benchmark["rocm_benchmark"]["results"]:
        if numeric(item.get("speedup_vs_baseline")) is not None:
            if best_benchmark is None or item["speedup_vs_baseline"] > best_benchmark["speedup_vs_baseline"]:
                best_benchmark = item
    hackathon_rows = [
        ["LoRA RCA quality", lora["quality"].get("rca_quality", NOT_AVAILABLE)],
        ["LoRA severity reasoning", lora["quality"].get("severity_reasoning", NOT_AVAILABLE)],
        ["Hallucination score", llm.get("hallucination_score", NOT_AVAILABLE)],
        [
            "Best AMD rerank speedup",
            f"{best_benchmark['speedup_vs_baseline']:.2f}x" if best_benchmark else NOT_AVAILABLE,
        ],
        ["Report completeness", f"{summary['completeness']['estimated_report_completeness_percent']}%"],
    ]
    slide_rows = [
        ["Base model", summary["lineage"]["model_lineage"]["base_model"]],
        ["LoRA model", summary["lineage"]["model_lineage"]["lora_model"]],
        ["Judge model", summary["lineage"]["model_lineage"]["judge_model"]],
        ["Hardware", summary["lineage"]["hardware_lineage"]["amd_mi300x_results"]],
        ["Precision", summary["lineage"]["model_lineage"]["precision"]],
    ]
    chart_lines = "\n".join(f"- `{Path(path).name}`" for path in charts)
    telemetry_rows = [[key, display_value(value)] for key, value in sections["telemetry_model"].items()]
    vision_rows = [
        [key, display_value(value)]
        for key, value in sections["vision"].items()
        if key not in {"defect_examples", "good_examples", "heatmaps", "overlays"}
    ]
    retrieval_rows = [[key, display_value(value)] for key, value in sections["retrieval"].items()]
    llm_rows = [
        [key, display_value(value)] for key, value in sections["llm"].items() if key not in {"judge"}
    ]
    agent_rows = [
        [key, display_value(value)]
        for key, value in sections["agent_system"].items()
        if key != "trace_visualizations"
    ]
    policy_rows = [[key, display_value(value)] for key, value in sections["policy_severity"].items()]
    lora_rows = [
        [key, display_value(value)] for key, value in sections["lora"].items() if key != "quality"
    ]
    business_rows = [[key, display_value(value)] for key, value in sections["business_metrics"].items()]
    missing_rows = [
        [item["section"], item["metric"], item["reason"]]
        for item in summary["completeness"]["missing"][:80]
    ]
    return f"""# Unified Evaluation Report

Generated: {summary['generated_at']}

## Hackathon Summary

{markdown_table(hackathon_rows, ["Metric", "Value"])}

## Executive Summary

This package aggregates evaluation, observability, benchmarking, and reporting artifacts for local NVIDIA and AMD ROCm environments. Metrics are only reported when a measured artifact exists or when the framework can compute them directly from repository evaluation data. Missing metrics are marked `{NOT_AVAILABLE}` with an explicit reason.

## Slide-ready Lineage

{markdown_table(slide_rows, ["Field", "Value"])}

## Portfolio Report

### Telemetry Model

{markdown_table(telemetry_rows, ["Metric", "Value"])}

### Vision

{markdown_table(vision_rows, ["Metric", "Value"])}

### Retrieval

{markdown_table(retrieval_rows, ["Metric", "Value"])}

### LLM

{markdown_table(llm_rows, ["Metric", "Value"])}

### Agent System

{markdown_table(agent_rows, ["Metric", "Value"])}

### Policy / Severity

{markdown_table(policy_rows, ["Metric", "Value"])}

### LoRA

{markdown_table(lora_rows, ["Metric", "Value"])}

### AMD / GPU Benchmarks

Benchmark implementations and precision coverage are included in `evaluation_summary.json`. Highest-signal charts are emitted under `charts/`.

### Business Metrics

{markdown_table(business_rows, ["Metric", "Value"])}

## Charts Generated

{chart_lines}

## Metrics Missing

{markdown_table(missing_rows, ["Section", "Metric", "Reason"])}
"""


def markdown_to_html(markdown: str, title: str) -> str:
    body = []
    in_list = False
    for line in markdown.splitlines():
        if line.startswith("# "):
            body.append(f"<h1>{html.escape(line[2:])}</h1>")
        elif line.startswith("## "):
            body.append(f"<h2>{html.escape(line[3:])}</h2>")
        elif line.startswith("### "):
            body.append(f"<h3>{html.escape(line[4:])}</h3>")
        elif line.startswith("| "):
            cells = [cell.strip() for cell in line.strip("|").split("|")]
            if set(cells) == {"---"}:
                continue
            tag = "th" if not body or not str(body[-1]).startswith("<table") else "td"
            row = "".join(f"<{tag}>{cell}</{tag}>" for cell in cells)
            if not body or not str(body[-1]).startswith("<table"):
                body.append(f"<table><tr>{row}</tr>")
            else:
                body[-1] = body[-1] + f"<tr>{row}</tr>"
        elif line.startswith("- "):
            if not in_list:
                body.append("<ul>")
                in_list = True
            body.append(f"<li>{html.escape(line[2:])}</li>")
        elif not line.strip():
            if body and str(body[-1]).startswith("<table"):
                body[-1] = body[-1] + "</table>"
            if in_list:
                body.append("</ul>")
                in_list = False
        else:
            body.append(f"<p>{html.escape(line)}</p>")
    if body and str(body[-1]).startswith("<table"):
        body[-1] = body[-1] + "</table>"
    if in_list:
        body.append("</ul>")
    return textwrap.dedent(
        f"""
        <!doctype html>
        <html>
        <head>
          <meta charset="utf-8">
          <title>{html.escape(title)}</title>
          <style>
            body {{ font-family: Arial, sans-serif; color: #111827; line-height: 1.5; margin: 40px; }}
            h1, h2, h3 {{ color: #111827; }}
            table {{ border-collapse: collapse; width: 100%; margin: 16px 0 28px; font-size: 13px; }}
            th, td {{ border: 1px solid #d1d5db; padding: 8px; vertical-align: top; }}
            th {{ background: #f3f4f6; text-align: left; }}
            code {{ background: #f3f4f6; padding: 2px 4px; }}
          </style>
        </head>
        <body>
        {''.join(body)}
        </body>
        </html>
        """
    ).strip()


def try_write_pdf(html_path: Path, pdf_path: Path) -> str | None:
    for command in (["wkhtmltopdf", str(html_path), str(pdf_path)], ["pandoc", str(html_path), "-o", str(pdf_path)]):
        if shutil.which(command[0]) is None:
            continue
        result = subprocess.run(command, check=False, capture_output=True, text=True, timeout=60)
        if result.returncode == 0 and pdf_path.exists():
            return None
        return result.stderr.strip() or f"{command[0]} failed with exit code {result.returncode}."
    return "No supported PDF renderer found. Install wkhtmltopdf or pandoc to emit evaluation_report.pdf."


def generate(output_dir: Path) -> dict[str, Any]:
    charts_dir = ensure_dirs(output_dir)
    hardware = load_json(output_dir / "hardware_profile.json") or {}
    system = load_json(output_dir / "system_metrics.json") or {}
    sections = {
        "system_hardware_metrics": {"hardware_profile": hardware, "system_metrics": system},
        "telemetry_model": telemetry_metrics(),
        "vision": vision_metrics(charts_dir),
        "retrieval": retrieval_metrics(charts_dir),
        "llm": llm_metrics(charts_dir),
        "agent_system": agent_metrics(charts_dir),
        "policy_severity": policy_metrics(charts_dir),
        "lora": lora_metrics(charts_dir),
        "amd_gpu_benchmarks": benchmark_metrics(charts_dir),
        "business_metrics": business_metrics(),
    }
    summary = {
        "generated_at": utc_now(),
        "output_dir": str(output_dir),
        "sections": sections,
        "lineage": lineage(hardware, system, sections["lora"], sections["llm"]),
        "charts": sorted(str(path) for path in charts_dir.iterdir() if path.is_file()),
    }
    annotate_known_unavailable(summary)
    summary["completeness"] = completeness(summary)
    summary["metrics_collected"] = summary["completeness"]["measured_metric_fields"]
    summary["metrics_missing"] = summary["completeness"]["missing_metric_fields"]
    summary["charts_generated"] = len(summary["charts"])

    (output_dir / "evaluation_summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    write_csv_summary(summary, output_dir / "evaluation_summary.csv")
    report_md = render_report(summary, summary["charts"])
    (output_dir / "evaluation_report.md").write_text(report_md, encoding="utf-8")
    report_html = markdown_to_html(report_md, "Unified Evaluation Report")
    html_path = output_dir / "evaluation_report.html"
    html_path.write_text(report_html, encoding="utf-8")
    pdf_error = try_write_pdf(html_path, output_dir / "evaluation_report.pdf")
    summary["pdf_status"] = "created" if pdf_error is None else {"value": NOT_AVAILABLE, "reason": pdf_error}
    (output_dir / "evaluation_summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    return summary


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Generate unified evaluation reports.")
    parser.add_argument("--output-dir", default=DEFAULT_OUTPUT_DIR, type=Path)
    return parser


def main() -> int:
    args = build_parser().parse_args()
    summary = generate(args.output_dir)
    print(
        json.dumps(
            {
                "output_dir": str(args.output_dir),
                "metrics_collected": summary["metrics_collected"],
                "metrics_missing": summary["metrics_missing"],
                "charts_generated": summary["charts_generated"],
                "estimated_report_completeness_percent": summary["completeness"][
                    "estimated_report_completeness_percent"
                ],
                "pdf_status": summary["pdf_status"],
            },
            indent=2,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
