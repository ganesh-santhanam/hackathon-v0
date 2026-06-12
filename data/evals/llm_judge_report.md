# AMD Hackathon LLM-as-Judge Evaluation

## Executive Summary

- Examples evaluated: 10
- Base model: Qwen/Qwen3-4B-Instruct-2507
- LoRA model: Qwen/Qwen3-4B-Instruct-2507+data/amd/lora/qwen4b_adapter
- Judge model: Qwen/Qwen3-14B
- Judge provider: openai-compatible
- Hardware: AMD MI300X-class gfx942 GPU via ROCm 7.0 / HIP 7.0
- Precision: BF16 for LoRA training, candidate generation, and vLLM judge serving
- Hallucination score is lower-is-better; all other metrics are higher-is-better.

## Evaluation Metadata

| Field | Value |
| --- | --- |
| Base Model | Qwen/Qwen3-4B-Instruct-2507 |
| LoRA Model | Qwen/Qwen3-4B-Instruct-2507+data/amd/lora/qwen4b_adapter |
| Judge Model | Qwen/Qwen3-14B |
| Hardware | AMD MI300X-class gfx942 GPU via ROCm 7.0 / HIP 7.0 |
| Precision | BF16 for LoRA training, candidate generation, and vLLM judge serving |
| Judge Endpoint | http://localhost:8000/v1/chat/completions |
| Judge Provider | openai-compatible |

## Candidate Models

| Candidate | Model | Examples |
| --- | --- | --- |
| base | Qwen/Qwen3-4B-Instruct-2507 | 10 |
| lora | Qwen/Qwen3-4B-Instruct-2507+data/amd/lora/qwen4b_adapter | 10 |

## Mean Scores

| Metric | Base Mean | LoRA Mean | LoRA Improvement % |
| --- | --- | --- | --- |
| hallucination_score | 1.0 | 1.0 | 0.0 |
| rca_quality | 3.7 | 4.2 | 13.51 |
| actionability | 4.4 | 4.1 | -6.82 |
| severity_reasoning | 3.8 | 4.2 | 10.53 |

## Key Findings

- Use the mean-score table directly in PowerPoint to compare base and LoRA behavior.
- Use hallucination score to show evidence grounding; lower LoRA values indicate better grounding.
- Use RCA quality, actionability, and severity reasoning to show domain-specific investigation gains.

## Conclusions

- The pipeline preserves prompts, responses, judge scores, and model metadata for auditability.
- Results are generated from local JSONL artifacts and can run on a laptop or AMD Cloud instance.
