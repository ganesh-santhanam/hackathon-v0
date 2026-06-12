# Unified Evaluation Report

Generated: 2026-06-12T14:43:24.545166+00:00

## Hackathon Summary

| Metric | Value |
| --- | --- |
| LoRA RCA quality | 4.6 |
| LoRA severity reasoning | 4.6 |
| Hallucination score | 1.0 |
| Best AMD rerank speedup | 6.05x |
| Report completeness | 71.1% |

## Executive Summary

This package aggregates evaluation, observability, benchmarking, and reporting artifacts for local NVIDIA and AMD ROCm environments. Metrics are only reported when a measured artifact exists or when the framework can compute them directly from repository evaluation data. Missing metrics are marked `NOT AVAILABLE` with an explicit reason.

## Slide-ready Lineage

| Field | Value |
| --- | --- |
| Base model | Qwen/Qwen3-4B-Instruct-2507 |
| LoRA model | Qwen/Qwen3-4B-Instruct-2507+data/amd/lora/qwen4b_adapter |
| Judge model | Qwen/Qwen3-14B |
| Hardware | AMD MI300X-class gfx942 GPU via ROCm 7.0 / HIP 7.0 |
| Precision | BF16 for LoRA training, candidate generation, and vLLM judge serving |

## Portfolio Report

### Telemetry Model

| Metric | Value |
| --- | --- |
| accuracy | NOT AVAILABLE - Telemetry model metrics file models/telemetry_metrics.json is not present in this checkout. |
| precision | NOT AVAILABLE - Telemetry model metrics file models/telemetry_metrics.json is not present in this checkout. |
| recall | NOT AVAILABLE - Telemetry model metrics file models/telemetry_metrics.json is not present in this checkout. |
| f1 | NOT AVAILABLE - Telemetry model metrics file models/telemetry_metrics.json is not present in this checkout. |
| roc_auc | NOT AVAILABLE - Telemetry model metrics file models/telemetry_metrics.json is not present in this checkout. |
| pr_auc | NOT AVAILABLE - Telemetry model metrics file models/telemetry_metrics.json is not present in this checkout. |
| confusion_matrix | NOT AVAILABLE - Telemetry model metrics file models/telemetry_metrics.json is not present in this checkout. |
| shap_global_importance | NOT AVAILABLE - SHAP was not run; no SHAP artifact was found. |
| shap_local_example | NOT AVAILABLE - SHAP was not run; no SHAP artifact was found. |

### Vision

| Metric | Value |
| --- | --- |
| accuracy | NOT AVAILABLE - No persisted aggregate vision metrics artifact was found; example heatmaps/overlays were copied when available. |
| precision | NOT AVAILABLE - No persisted aggregate vision metrics artifact was found; example heatmaps/overlays were copied when available. |
| recall | NOT AVAILABLE - No persisted aggregate vision metrics artifact was found; example heatmaps/overlays were copied when available. |
| f1 | NOT AVAILABLE - No persisted aggregate vision metrics artifact was found; example heatmaps/overlays were copied when available. |

### Retrieval

| Metric | Value |
| --- | --- |
| recall_at_k | NOT AVAILABLE - No labeled retrieval relevance set was found, so ranking metrics cannot be measured. |
| precision_at_k | NOT AVAILABLE - No labeled retrieval relevance set was found, so ranking metrics cannot be measured. |
| mrr | NOT AVAILABLE - No labeled retrieval relevance set was found, so ranking metrics cannot be measured. |
| ndcg | NOT AVAILABLE - No labeled retrieval relevance set was found, so ranking metrics cannot be measured. |
| retrieval_latency_ms | NOT AVAILABLE - No persisted retrieval latency artifact was found. |
| corpus_size | 300 |
| duplicate_rate | 0.0 |

### LLM

| Metric | Value |
| --- | --- |
| hallucination_score | 1.0 |
| groundedness | 5.0 |
| relevance | 4.6 |
| citation_accuracy | NOT AVAILABLE - The current judge rubric does not score citation accuracy. |
| first_token_latency_ms | NOT AVAILABLE - Candidate generation traces do not include first-token latency. |
| full_response_latency_ms | NOT AVAILABLE - Candidate generation traces do not include aggregate response latency. |
| prompt_tokens | NOT AVAILABLE - Token accounting was not emitted by the local model endpoint. |
| completion_tokens | NOT AVAILABLE - Token accounting was not emitted by the local model endpoint. |
| total_tokens | NOT AVAILABLE - Token accounting was not emitted by the local model endpoint. |
| examples_evaluated | 10 |

### Agent System

| Metric | Value |
| --- | --- |
| workflow_completion_rate | NOT AVAILABLE - No persisted agent trace artifact was found. |
| scenario_pass_rate | NOT AVAILABLE - No persisted agent trace artifact was found. |
| structured_output_success_rate | NOT AVAILABLE - No persisted agent trace artifact was found. |
| tool_call_success_rate | NOT AVAILABLE - No persisted agent trace artifact was found. |
| retry_count | NOT AVAILABLE - No persisted agent trace artifact was found. |
| average_steps | NOT AVAILABLE - No persisted agent trace artifact was found. |
| average_runtime_seconds | NOT AVAILABLE - No persisted agent trace artifact was found. |
| scenario_count | 12 |
| failure_breakdown | NOT AVAILABLE - No persisted agent trace artifact was found. |

### Policy / Severity

| Metric | Value |
| --- | --- |
| severity_accuracy | 1.0 |
| confusion_matrix | {"SEV1": {"SEV1": 3, "SEV2": 0, "SEV3": 0}, "SEV2": {"SEV1": 0, "SEV2": 5, "SEV3": 0}, "SEV3": {"SEV1": 0, "SEV2": 0, "SEV3": 4}} |
| false_sev1 | 0 |
| missed_sev1 | 0 |
| override_rate | NOT AVAILABLE - No override log artifact was found. |
| escalation_rate | NOT AVAILABLE - No escalation log artifact was found. |

### LoRA

| Metric | Value |
| --- | --- |
| base_model | Qwen/Qwen3-4B-Instruct-2507 |
| lora_model | Qwen/Qwen3-4B-Instruct-2507+data/amd/lora/qwen4b_adapter |
| adapter_size_mb | 141.22 |
| train_loss | 0.1870122145116329 |
| eval_loss | 0.02745771035552025 |
| train_runtime_seconds | 289.3995 |
| peak_vram_gb | NOT AVAILABLE - LoRA training did not persist peak VRAM in training_metrics.json. |
| gpu_hours | 0.08038875 |
| judge_model | Qwen/Qwen3-14B |
| examples | 10 |
| successes | 10 |

### AMD / GPU Benchmarks

Benchmark implementations and precision coverage are included in `evaluation_summary.json`. Highest-signal charts are emitted under `charts/`.

### Business Metrics

| Metric | Value |
| --- | --- |
| label | Illustrative estimates only; not measured production impact. |
| rca_time_reduction | 30-50% illustrative estimate |
| documentation_search_reduction | 40-60% illustrative estimate |
| estimated_downtime_avoided | Scenario-dependent illustrative estimate |
| estimated_cost_savings | Scenario-dependent illustrative estimate |

## Charts Generated

- `agent_trace_placeholder.svg`
- `amd_kernel_latency.svg`
- `amd_kernel_speedup.svg`
- `cable-000-patch_distance-bb12792ea7-annotated.png`
- `cable-000-patch_distance-bb12792ea7-heatmap.png`
- `grid-000-patch_distance-b959c41301-annotated.png`
- `grid-000-patch_distance-b959c41301-heatmap.png`
- `llm_judge_scores.svg`
- `lora_side_by_side.svg`
- `metal_nut-000-patch_distance-43e41440b0-annotated.png`
- `metal_nut-000-patch_distance-43e41440b0-heatmap.png`
- `policy_severity.svg`
- `retrieval_corpus_profile.svg`
- `screw-000-patch_distance-36b875c977-annotated.png`
- `screw-000-patch_distance-36b875c977-heatmap.png`

## Metrics Missing

| Section | Metric | Reason |
| --- | --- | --- |
| system_hardware_metrics | system_metrics.runtime.cuda_version | CUDA is not available in the captured runtime. |
| system_hardware_metrics | system_metrics.timing.training_runtime_seconds | No training command was supplied to run_full_evaluation.py. |
| system_hardware_metrics | system_metrics.timing.inference_runtime_seconds | No inference command was supplied to run_full_evaluation.py. |
| system_hardware_metrics | system_metrics.timing.benchmark_runtime_seconds | No benchmark command was supplied to run_full_evaluation.py. |
| telemetry_model | accuracy | Telemetry model metrics file models/telemetry_metrics.json is not present in this checkout. |
| telemetry_model | precision | Telemetry model metrics file models/telemetry_metrics.json is not present in this checkout. |
| telemetry_model | recall | Telemetry model metrics file models/telemetry_metrics.json is not present in this checkout. |
| telemetry_model | f1 | Telemetry model metrics file models/telemetry_metrics.json is not present in this checkout. |
| telemetry_model | roc_auc | Telemetry model metrics file models/telemetry_metrics.json is not present in this checkout. |
| telemetry_model | pr_auc | Telemetry model metrics file models/telemetry_metrics.json is not present in this checkout. |
| telemetry_model | confusion_matrix | Telemetry model metrics file models/telemetry_metrics.json is not present in this checkout. |
| telemetry_model | shap_global_importance | SHAP was not run; no SHAP artifact was found. |
| telemetry_model | shap_local_example | SHAP was not run; no SHAP artifact was found. |
| vision | accuracy | No persisted aggregate vision metrics artifact was found; example heatmaps/overlays were copied when available. |
| vision | precision | No persisted aggregate vision metrics artifact was found; example heatmaps/overlays were copied when available. |
| vision | recall | No persisted aggregate vision metrics artifact was found; example heatmaps/overlays were copied when available. |
| vision | f1 | No persisted aggregate vision metrics artifact was found; example heatmaps/overlays were copied when available. |
| retrieval | recall_at_k | No labeled retrieval relevance set was found, so ranking metrics cannot be measured. |
| retrieval | precision_at_k | No labeled retrieval relevance set was found, so ranking metrics cannot be measured. |
| retrieval | mrr | No labeled retrieval relevance set was found, so ranking metrics cannot be measured. |
| retrieval | ndcg | No labeled retrieval relevance set was found, so ranking metrics cannot be measured. |
| retrieval | retrieval_latency_ms | No persisted retrieval latency artifact was found. |
| llm | citation_accuracy | The current judge rubric does not score citation accuracy. |
| llm | first_token_latency_ms | Candidate generation traces do not include first-token latency. |
| llm | full_response_latency_ms | Candidate generation traces do not include aggregate response latency. |
| llm | prompt_tokens | Token accounting was not emitted by the local model endpoint. |
| llm | completion_tokens | Token accounting was not emitted by the local model endpoint. |
| llm | total_tokens | Token accounting was not emitted by the local model endpoint. |
| agent_system | workflow_completion_rate | No persisted agent trace artifact was found. |
| agent_system | scenario_pass_rate | No persisted agent trace artifact was found. |
| agent_system | structured_output_success_rate | No persisted agent trace artifact was found. |
| agent_system | tool_call_success_rate | No persisted agent trace artifact was found. |
| agent_system | retry_count | No persisted agent trace artifact was found. |
| agent_system | average_steps | No persisted agent trace artifact was found. |
| agent_system | average_runtime_seconds | No persisted agent trace artifact was found. |
| agent_system | failure_breakdown | No persisted agent trace artifact was found. |
| policy_severity | override_rate | No override log artifact was found. |
| policy_severity | escalation_rate | No escalation log artifact was found. |
| lora | peak_vram_gb | LoRA training did not persist peak VRAM in training_metrics.json. |
