#!/usr/bin/env python3
"""Compare ROCm reranking implementation strategies on a fixed workload."""

from __future__ import annotations

import csv
import importlib.util
import json
import math
import subprocess
import sys
import time
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import Any, Callable


OUTPUT_DIR = Path("data/benchmarks")
CSV_OUTPUT = OUTPUT_DIR / "kernel_comparison.csv"
JSON_OUTPUT = OUTPUT_DIR / "kernel_comparison.json"
REPORT_OUTPUT = OUTPUT_DIR / "kernel_comparison_report.md"
CHART_OUTPUT = OUTPUT_DIR / "kernel_comparison_chart.svg"

TELEMETRY_SCALE = [12.0, 14.0, 900.0, 45.0, 250.0]
TELEMETRY_WEIGHT = [0.15, 0.15, 0.25, 0.2, 0.25]


@dataclass
class Workload:
    batch_size: int = 32
    candidates: int = 1_000_000
    embedding_dim: int = 768
    top_k: int = 10
    runs: int = 5
    warmup_runs: int = 2
    bonus_weight: float = 0.05
    seed: int = 7


def import_torch():
    import torch
    import torch.nn.functional as functional

    return torch, functional


def now() -> str:
    return datetime.now(UTC).isoformat()


def module_available(name: str) -> bool:
    return importlib.util.find_spec(name) is not None


def synchronize(torch_module: Any) -> None:
    torch_module.cuda.synchronize()


def reset_peak_vram(torch_module: Any) -> None:
    torch_module.cuda.reset_peak_memory_stats()


def peak_vram_gb(torch_module: Any) -> float:
    return torch_module.cuda.max_memory_allocated() / (1024**3)


def make_inputs(torch_module: Any, workload: Workload) -> dict[str, Any]:
    generator = torch_module.Generator(device="cpu")
    generator.manual_seed(workload.seed)
    query_embeddings = torch_module.randn(
        workload.batch_size, workload.embedding_dim, generator=generator
    )
    incident_embeddings = torch_module.randn(
        workload.candidates, workload.embedding_dim, generator=generator
    )
    query_telemetry = torch_module.rand(workload.batch_size, 5, generator=generator)
    incident_telemetry = torch_module.rand(workload.candidates, 5, generator=generator)
    rerank_bonus = torch_module.rand(workload.candidates, generator=generator)

    telemetry_min = torch_module.tensor([290.0, 300.0, 1_100.0, 5.0, 0.0])
    telemetry_span = torch_module.tensor([18.0, 20.0, 1_900.0, 80.0, 253.0])
    query_telemetry = telemetry_min + query_telemetry * telemetry_span
    incident_telemetry = telemetry_min + incident_telemetry * telemetry_span

    return {
        "query_embeddings": query_embeddings.cuda(),
        "incident_embeddings": incident_embeddings.cuda(),
        "query_telemetry": query_telemetry.cuda(),
        "incident_telemetry": incident_telemetry.cuda(),
        "rerank_bonus": rerank_bonus.cuda(),
        "telemetry_scale": torch_module.tensor(
            TELEMETRY_SCALE, device="cuda", dtype=torch_module.float32
        ),
        "telemetry_weights": torch_module.tensor(
            TELEMETRY_WEIGHT, device="cuda", dtype=torch_module.float32
        ),
    }


def scaled_telemetry_constants(torch_module: Any) -> tuple[Any, Any]:
    scale = torch_module.tensor(TELEMETRY_SCALE, device="cuda", dtype=torch_module.float32)
    weights = torch_module.tensor(TELEMETRY_WEIGHT, device="cuda", dtype=torch_module.float32)
    return scale, weights


def full_rerank_eager(
    torch_module: Any,
    functional: Any,
    inputs: dict[str, Any],
    dtype: Any,
    workload: Workload,
) -> tuple[Any, Any]:
    query_embeddings = functional.normalize(inputs["query_embeddings"].to(dtype), dim=1)
    incident_embeddings = functional.normalize(inputs["incident_embeddings"].to(dtype), dim=1)
    embedding_score = query_embeddings @ incident_embeddings.T
    scale = inputs["telemetry_scale"]
    weights = inputs["telemetry_weights"]
    query_telemetry = inputs["query_telemetry"].float()
    incident_telemetry = inputs["incident_telemetry"].float()
    telemetry_delta = (query_telemetry[:, None, :] - incident_telemetry[None, :, :]).abs()
    telemetry_penalty = ((telemetry_delta / scale) * weights).sum(dim=2)
    final_score = embedding_score.float() - telemetry_penalty
    final_score = final_score + inputs["rerank_bonus"].float()[None, :] * workload.bonus_weight
    _, top_indices = torch_module.topk(final_score, k=workload.top_k, dim=1)
    return final_score, top_indices


def make_precomputed_inputs(
    torch_module: Any,
    functional: Any,
    inputs: dict[str, Any],
    dtype: Any,
    workload: Workload,
) -> dict[str, Any]:
    scale = inputs["telemetry_scale"]
    weights = inputs["telemetry_weights"]
    return {
        "query_embeddings": inputs["query_embeddings"],
        "incident_embeddings_norm": functional.normalize(
            inputs["incident_embeddings"].to(dtype), dim=1
        ).contiguous(),
        "query_telemetry_scaled": inputs["query_telemetry"].float() / scale,
        "incident_telemetry_scaled": (inputs["incident_telemetry"].float() / scale).contiguous(),
        "weights": weights,
        "weighted_bonus": (inputs["rerank_bonus"].float() * workload.bonus_weight).contiguous(),
    }


def rerank_precomputed_index(
    torch_module: Any,
    functional: Any,
    precomputed: dict[str, Any],
    dtype: Any,
    workload: Workload,
) -> tuple[Any, Any]:
    query_embeddings = functional.normalize(precomputed["query_embeddings"].to(dtype), dim=1)
    embedding_score = query_embeddings @ precomputed["incident_embeddings_norm"].T
    telemetry_delta = (
        precomputed["query_telemetry_scaled"][:, None, :]
        - precomputed["incident_telemetry_scaled"][None, :, :]
    ).abs()
    telemetry_penalty = (telemetry_delta * precomputed["weights"]).sum(dim=2)
    final_score = embedding_score.float() - telemetry_penalty + precomputed["weighted_bonus"][None, :]
    _, top_indices = torch_module.topk(final_score, k=workload.top_k, dim=1)
    return final_score, top_indices


def build_triton_score_kernel() -> Any | None:
    if not module_available("triton"):
        return None
    import triton
    import triton.language as tl

    @triton.jit
    def _score_kernel(
        embedding_score,
        query_tel_scaled,
        incident_tel_scaled,
        weights,
        weighted_bonus,
        output,
        n_candidates: tl.constexpr,
        block_size: tl.constexpr,
    ):
        row = tl.program_id(0)
        block = tl.program_id(1)
        offsets = block * block_size + tl.arange(0, block_size)
        mask = offsets < n_candidates
        base_score = tl.load(embedding_score + row * n_candidates + offsets, mask=mask, other=-float("inf"))
        penalty = tl.zeros((block_size,), dtype=tl.float32)
        for idx in range(0, 5):
            q = tl.load(query_tel_scaled + row * 5 + idx)
            inc = tl.load(incident_tel_scaled + offsets * 5 + idx, mask=mask, other=0.0)
            weight = tl.load(weights + idx)
            penalty += tl.abs(q - inc) * weight
        bonus = tl.load(weighted_bonus + offsets, mask=mask, other=0.0)
        tl.store(output + row * n_candidates + offsets, base_score - penalty + bonus, mask=mask)

    return _score_kernel


def make_triton_precomputed_runner(
    torch_module: Any,
    functional: Any,
    precomputed: dict[str, Any],
    dtype: Any,
    workload: Workload,
) -> Callable[[], tuple[Any, Any]]:
    import triton

    score_kernel = build_triton_score_kernel()
    if score_kernel is None:
        raise RuntimeError("Triton is not importable.")

    block_size = 256
    grid = (workload.batch_size, triton.cdiv(workload.candidates, block_size))

    def run() -> tuple[Any, Any]:
        query_embeddings = functional.normalize(precomputed["query_embeddings"].to(dtype), dim=1)
        embedding_score = query_embeddings @ precomputed["incident_embeddings_norm"].T
        output = torch_module.empty_like(embedding_score, dtype=torch_module.float32)
        score_kernel[grid](
            embedding_score,
            precomputed["query_telemetry_scaled"],
            precomputed["incident_telemetry_scaled"],
            precomputed["weights"],
            precomputed["weighted_bonus"],
            output,
            workload.candidates,
            block_size,
        )
        _, top_indices = torch_module.topk(output, k=workload.top_k, dim=1)
        return output, top_indices

    return run


def make_fp8_runner(
    torch_module: Any,
    functional: Any,
    precomputed_fp32: dict[str, Any],
    workload: Workload,
) -> Callable[[], tuple[Any, Any]]:
    if not hasattr(torch_module, "_scaled_mm") or not hasattr(torch_module, "float8_e4m3fnuz"):
        raise RuntimeError("torch._scaled_mm with float8_e4m3fnuz is not available.")

    fp8_dtype = torch_module.float8_e4m3fnuz
    incident_norm = precomputed_fp32["incident_embeddings_norm"].float()
    inc_scale = incident_norm.abs().amax().clamp(min=1e-8) / 224.0
    # torch._scaled_mm on ROCm requires the right operand to be column-major.
    incident_fp8_colmajor = (incident_norm / inc_scale).to(fp8_dtype).T

    def run() -> tuple[Any, Any]:
        query_norm = functional.normalize(precomputed_fp32["query_embeddings"].float(), dim=1)
        query_scale = query_norm.abs().amax().clamp(min=1e-8) / 224.0
        query_fp8 = (query_norm / query_scale).to(fp8_dtype)
        embedding_score = torch_module._scaled_mm(
            query_fp8,
            incident_fp8_colmajor,
            scale_a=query_scale,
            scale_b=inc_scale,
            out_dtype=torch_module.float32,
        )
        telemetry_delta = (
            precomputed_fp32["query_telemetry_scaled"][:, None, :]
            - precomputed_fp32["incident_telemetry_scaled"][None, :, :]
        ).abs()
        telemetry_penalty = (telemetry_delta * precomputed_fp32["weights"]).sum(dim=2)
        final_score = embedding_score - telemetry_penalty + precomputed_fp32["weighted_bonus"][None, :]
        _, top_indices = torch_module.topk(final_score, k=workload.top_k, dim=1)
        return final_score, top_indices

    return run


def topk_overlap(torch_module: Any, baseline_topk: Any, candidate_topk: Any) -> float:
    baseline_cpu = baseline_topk.detach().cpu()
    candidate_cpu = candidate_topk.detach().cpu()
    overlaps = []
    for row_index in range(baseline_cpu.shape[0]):
        left = set(baseline_cpu[row_index].tolist())
        right = set(candidate_cpu[row_index].tolist())
        overlaps.append(len(left & right) / max(1, len(left)))
    return float(sum(overlaps) / len(overlaps))


def measure_runner(
    torch_module: Any,
    name: str,
    precision: str,
    runner: Callable[[], tuple[Any, Any]],
    workload: Workload,
    baseline_latency: float,
    baseline_scores: Any,
    baseline_topk: Any,
    complexity: str,
    notes: str = "",
) -> dict[str, Any]:
    reset_peak_vram(torch_module)
    for _ in range(workload.warmup_runs):
        runner()
    synchronize(torch_module)

    latencies = []
    final_scores = None
    top_indices = None
    for _ in range(workload.runs):
        start = time.perf_counter()
        final_scores, top_indices = runner()
        synchronize(torch_module)
        latencies.append((time.perf_counter() - start) * 1000.0)

    latency = min(latencies)
    diff = (final_scores.float() - baseline_scores.float()).abs()
    row = {
        "implementation_name": name,
        "precision": precision,
        "latency_ms": latency,
        "throughput_candidates_per_sec": workload.batch_size
        * workload.candidates
        / max(latency / 1000.0, 1e-12),
        "speedup_vs_baseline": baseline_latency / latency,
        "peak_vram_gb": peak_vram_gb(torch_module),
        "top_k_overlap_vs_fp32": topk_overlap(torch_module, baseline_topk, top_indices),
        "max_abs_error_vs_fp32": float(diff.max().item()),
        "mean_abs_error_vs_fp32": float(diff.mean().item()),
        "engineering_complexity": complexity,
        "supported_on_current_AMD_image": "yes",
        "notes": notes,
    }
    del final_scores, top_indices, diff
    torch_module.cuda.empty_cache()
    return row


def unsupported_row(
    name: str,
    precision: str,
    complexity: str,
    notes: str,
) -> dict[str, Any]:
    return {
        "implementation_name": name,
        "precision": precision,
        "latency_ms": None,
        "throughput_candidates_per_sec": None,
        "speedup_vs_baseline": None,
        "peak_vram_gb": None,
        "top_k_overlap_vs_fp32": None,
        "max_abs_error_vs_fp32": None,
        "mean_abs_error_vs_fp32": None,
        "engineering_complexity": complexity,
        "supported_on_current_AMD_image": "no",
        "notes": notes,
    }


def write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as output_file:
        writer = csv.DictWriter(output_file, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)


def write_json(path: Path, rows: list[dict[str, Any]], metadata: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps({"metadata": metadata, "results": rows}, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )


def fmt(value: Any, digits: int = 3) -> str:
    if value is None:
        return ""
    if isinstance(value, float) and math.isfinite(value):
        return f"{value:.{digits}f}"
    return str(value)


def write_report(path: Path, rows: list[dict[str, Any]], metadata: dict[str, Any]) -> None:
    supported = [row for row in rows if row["supported_on_current_AMD_image"] == "yes"]
    best = max(supported, key=lambda row: row["speedup_vs_baseline"]) if supported else None
    lines = [
        "# ROCm Kernel Comparison Report",
        "",
        f"- Generated: {metadata['generated_at']}",
        f"- Torch: {metadata['torch_version']}",
        f"- HIP: {metadata['hip_version']}",
        f"- Device: {metadata['device_name']}",
        f"- Workload: batch={metadata['batch_size']}, candidates={metadata['candidates']}, dim={metadata['embedding_dim']}, top_k={metadata['top_k']}",
        "",
        "## Results",
        "",
        "| implementation | precision | supported | latency_ms | candidates/s | speedup | peak_vram_gb | top_k_overlap | max_error | mean_error | complexity |",
        "| --- | --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | --- |",
    ]
    for row in rows:
        lines.append(
            "| {implementation_name} | {precision} | {supported_on_current_AMD_image} | "
            "{latency_ms} | {throughput_candidates_per_sec} | {speedup_vs_baseline} | "
            "{peak_vram_gb} | {top_k_overlap_vs_fp32} | {max_abs_error_vs_fp32} | "
            "{mean_abs_error_vs_fp32} | {engineering_complexity} |".format(
                **{key: fmt(value) for key, value in row.items()}
            )
        )
    lines.extend(["", "## Unsupported Or Limited Paths", ""])
    for row in rows:
        if row["supported_on_current_AMD_image"] == "no" or row["notes"]:
            lines.append(
                f"- {row['implementation_name']} ({row['precision']}): {row['notes']}"
            )
    if best:
        lines.extend(
            [
                "",
                "## Recommendation",
                "",
                (
                    f"Use `{best['implementation_name']}` with `{best['precision']}` for the hackathon slide: "
                    f"{fmt(best['latency_ms'])} ms, {fmt(best['throughput_candidates_per_sec'], 0)} candidates/s, "
                    f"{fmt(best['speedup_vs_baseline'])}x vs the PyTorch FP32 eager baseline, "
                    f"top-k overlap {fmt(best['top_k_overlap_vs_fp32'])}."
                ),
            ]
        )
    lines.append("")
    path.write_text("\n".join(lines), encoding="utf-8")


def write_chart(path: Path, rows: list[dict[str, Any]]) -> None:
    plotted = [row for row in rows if row["supported_on_current_AMD_image"] == "yes"]
    if not plotted:
        return
    width = 1100
    height = 460
    margin_left = 80
    margin_bottom = 155
    chart_height = height - margin_bottom - 55
    chart_width = width - margin_left - 40
    max_speedup = max(float(row["speedup_vs_baseline"]) for row in plotted)
    step = chart_width / len(plotted)
    bar_width = max(10, step * 0.64)
    colors = {
        "fp32": "#4C78A8",
        "fp16": "#59A14F",
        "bf16": "#B07AA1",
        "fp8": "#F28E2B",
    }
    svg = [
        f'<svg xmlns="http://www.w3.org/2000/svg" width="{width}" height="{height}" viewBox="0 0 {width} {height}">',
        '<rect width="100%" height="100%" fill="white"/>',
        '<text x="36" y="32" font-family="Arial" font-size="22" font-weight="700">ROCm rerank implementation speedup</text>',
        f'<line x1="{margin_left}" y1="55" x2="{margin_left}" y2="{height - margin_bottom}" stroke="#333"/>',
        f'<line x1="{margin_left}" y1="{height - margin_bottom}" x2="{width - 35}" y2="{height - margin_bottom}" stroke="#333"/>',
    ]
    for index, row in enumerate(plotted):
        speedup = float(row["speedup_vs_baseline"])
        bar_height = speedup / max_speedup * chart_height
        x = margin_left + index * step + (step - bar_width) / 2
        y = height - margin_bottom - bar_height
        label = f"{row['implementation_name']} {row['precision']}"
        svg.extend(
            [
                f'<rect x="{x:.1f}" y="{y:.1f}" width="{bar_width:.1f}" height="{bar_height:.1f}" fill="{colors.get(row["precision"], "#777")}"/>',
                f'<text x="{x + bar_width / 2:.1f}" y="{y - 6:.1f}" font-family="Arial" font-size="10" text-anchor="middle">{speedup:.2f}x</text>',
                f'<text x="{x + bar_width / 2:.1f}" y="{height - margin_bottom + 16}" font-family="Arial" font-size="9" text-anchor="end" transform="rotate(-48 {x + bar_width / 2:.1f} {height - margin_bottom + 16})">{label}</text>',
            ]
        )
    svg.append("</svg>")
    path.write_text("\n".join(svg) + "\n", encoding="utf-8")


def run_graph_capture_only() -> int:
    torch_module, functional = import_torch()
    workload = Workload()
    inputs = make_inputs(torch_module, workload)
    baseline_scores, baseline_topk = full_rerank_eager(
        torch_module, functional, inputs, torch_module.float32, workload
    )
    synchronize(torch_module)
    baseline_row = measure_runner(
        torch_module,
        "pytorch_eager",
        "fp32",
        lambda: full_rerank_eager(torch_module, functional, inputs, torch_module.float32, workload),
        workload,
        1.0,
        baseline_scores,
        baseline_topk,
        "low",
    )
    baseline_latency = baseline_row["latency_ms"]
    rows = []
    for precision, dtype in {
        "fp32": torch_module.float32,
        "fp16": torch_module.float16,
        "bf16": torch_module.bfloat16,
    }.items():
        try:
            static_runner = lambda dtype=dtype: full_rerank_eager(
                torch_module, functional, inputs, dtype, workload
            )
            for _ in range(3):
                static_runner()
            synchronize(torch_module)
            graph = torch_module.cuda.CUDAGraph()
            with torch_module.cuda.graph(graph):
                graph_outputs = static_runner()

            def replay_graph() -> tuple[Any, Any]:
                graph.replay()
                return graph_outputs

            row = measure_runner(
                torch_module,
                "rocm_graph_capture",
                precision,
                replay_graph,
                workload,
                baseline_latency,
                baseline_scores,
                baseline_topk,
                "medium",
                notes="Measured in an isolated subprocess because ROCm graph capture reserves a private memory pool.",
            )
            rows.append(row)
        except Exception as exc:
            rows.append(
                unsupported_row(
                    "rocm_graph_capture",
                    precision,
                    "medium",
                    f"Graph capture failed in isolated subprocess: {type(exc).__name__}: {str(exc)[:240]}",
                )
            )
    print(json.dumps(rows))
    return 0


def main() -> int:
    premeasured_graph_rows: list[dict[str, Any]] = []
    try:
        graph_result = subprocess.run(
            [sys.executable, __file__, "--graph-only-json"],
            check=False,
            capture_output=True,
            text=True,
            timeout=180,
        )
        if graph_result.returncode == 0:
            premeasured_graph_rows = json.loads(graph_result.stdout.strip().splitlines()[-1])
        else:
            reason = graph_result.stderr.strip() or graph_result.stdout.strip()
            for precision in ("fp32", "fp16", "bf16"):
                premeasured_graph_rows.append(
                    unsupported_row(
                        "rocm_graph_capture",
                        precision,
                        "medium",
                        f"Isolated graph capture subprocess failed: {reason[:240]}",
                    )
                )
    except Exception as exc:
        for precision in ("fp32", "fp16", "bf16"):
            premeasured_graph_rows.append(
                unsupported_row(
                    "rocm_graph_capture",
                    precision,
                    "medium",
                    f"Could not run isolated graph capture subprocess: {type(exc).__name__}: {str(exc)[:240]}",
                )
            )

    torch_module, functional = import_torch()
    if not torch_module.cuda.is_available():
        raise SystemExit("A ROCm-visible GPU is required.")

    workload = Workload()
    inputs = make_inputs(torch_module, workload)

    baseline_scores, baseline_topk = full_rerank_eager(
        torch_module, functional, inputs, torch_module.float32, workload
    )
    synchronize(torch_module)
    baseline_runner = lambda: full_rerank_eager(
        torch_module, functional, inputs, torch_module.float32, workload
    )
    baseline_row = measure_runner(
        torch_module,
        "pytorch_eager",
        "fp32",
        baseline_runner,
        workload,
        baseline_latency=1.0,
        baseline_scores=baseline_scores,
        baseline_topk=baseline_topk,
        complexity="low",
        notes="Baseline PyTorch eager implementation.",
    )
    baseline_latency = baseline_row["latency_ms"]
    baseline_row["speedup_vs_baseline"] = 1.0
    rows = [baseline_row]

    dtype_map = {
        "fp16": torch_module.float16,
        "bf16": torch_module.bfloat16,
    }

    for precision, dtype in dtype_map.items():
        rows.append(
            measure_runner(
                torch_module,
                "pytorch_eager",
                precision,
                lambda dtype=dtype: full_rerank_eager(torch_module, functional, inputs, dtype, workload),
                workload,
                baseline_latency,
                baseline_scores,
                baseline_topk,
                "low",
            )
        )

    precomputed_fp32 = make_precomputed_inputs(
        torch_module, functional, inputs, torch_module.float32, workload
    )
    for precision, dtype in {"fp32": torch_module.float32, **dtype_map}.items():
        precomputed = (
            precomputed_fp32
            if precision == "fp32"
            else make_precomputed_inputs(torch_module, functional, inputs, dtype, workload)
        )
        rows.append(
            measure_runner(
                torch_module,
                "rocblas_precomputed_index",
                precision,
                lambda precomputed=precomputed, dtype=dtype: rerank_precomputed_index(
                    torch_module, functional, precomputed, dtype, workload
                ),
                workload,
                baseline_latency,
                baseline_scores,
                baseline_topk,
                "medium",
                notes="Uses PyTorch/rocBLAS GEMM with static candidate embeddings pre-normalized outside online latency.",
            )
        )

    if module_available("triton"):
        for precision, dtype in {"fp32": torch_module.float32, **dtype_map}.items():
            try:
                precomputed = (
                    precomputed_fp32
                    if precision == "fp32"
                    else make_precomputed_inputs(torch_module, functional, inputs, dtype, workload)
                )
                rows.append(
                    measure_runner(
                        torch_module,
                        "rocblas_plus_triton_score",
                        precision,
                        make_triton_precomputed_runner(
                            torch_module, functional, precomputed, dtype, workload
                        ),
                        workload,
                        baseline_latency,
                        baseline_scores,
                        baseline_topk,
                        "high",
                        notes="rocBLAS GEMM plus a custom Triton kernel for telemetry penalty and score combine; top-k remains PyTorch.",
                    )
                )
            except Exception as exc:
                rows.append(
                    unsupported_row(
                        "rocblas_plus_triton_score",
                        precision,
                        "high",
                        f"Triton path failed on this image: {type(exc).__name__}: {str(exc)[:240]}",
                    )
                )
    else:
        rows.append(
            unsupported_row(
                "triton_kernel",
                "mixed",
                "high",
                "Triton is not installed in the current AMD image.",
            )
        )

    try:
        rows.append(
            measure_runner(
                torch_module,
                "torch_scaled_mm_fp8",
                "fp8",
                make_fp8_runner(torch_module, functional, precomputed_fp32, workload),
                workload,
                baseline_latency,
                baseline_scores,
                baseline_topk,
                "high",
                notes="Uses torch._scaled_mm with float8_e4m3fnuz for the similarity GEMM; quantization and non-GEMM rerank work are included.",
            )
        )
    except Exception as exc:
        rows.append(
            unsupported_row(
                "torch_scaled_mm_fp8",
                "fp8",
                "high",
                "FP8 unavailable for this workload. Required component: a working ROCm hipBLASLt/PyTorch "
                f"scaled FP8 matmul path for this shape/layout. Failure: {type(exc).__name__}: {str(exc)[:240]}",
            )
        )

    if hasattr(torch_module, "compile"):
        for precision, dtype in {"fp32": torch_module.float32, **dtype_map}.items():
            try:
                compiled = torch_module.compile(
                    lambda: full_rerank_eager(torch_module, functional, inputs, dtype, workload),
                    mode="reduce-overhead",
                )
                rows.append(
                    measure_runner(
                        torch_module,
                        "torch_compile_inductor",
                        precision,
                        compiled,
                        workload,
                        baseline_latency,
                        baseline_scores,
                        baseline_topk,
                        "medium",
                        notes="torch.compile/Inductor over the eager PyTorch graph.",
                    )
                )
                del compiled
                torch_module.cuda.empty_cache()
            except Exception as exc:
                rows.append(
                    unsupported_row(
                        "torch_compile_inductor",
                        precision,
                        "medium",
                        f"torch.compile failed on this image: {type(exc).__name__}: {str(exc)[:240]}",
                    )
                )

    for row in premeasured_graph_rows:
        if row["supported_on_current_AMD_image"] == "yes":
            row["speedup_vs_baseline"] = baseline_latency / row["latency_ms"]
        rows.append(row)

    rows.append(
        unsupported_row(
            "aiter_full_rerank",
            "mixed",
            "high",
            "AITER is installed, but no exposed operator implements normalize + similarity + telemetry penalty + weighted rerank + top-k. A separate hipb_mm probe segfaulted on this image, so it was not used in the benchmark.",
        )
    )
    rows.append(
        unsupported_row(
            "composable_kernel_full_rerank",
            "mixed",
            "very_high",
            "Composable Kernel Python bindings are not importable as ck or composable_kernel in this image. Enabling this would require CK development headers/examples or a purpose-built CK extension for this workload.",
        )
    )

    metadata = {
        "generated_at": now(),
        "torch_version": torch_module.__version__,
        "hip_version": getattr(torch_module.version, "hip", None),
        "device_name": torch_module.cuda.get_device_name(0),
        "batch_size": workload.batch_size,
        "candidates": workload.candidates,
        "embedding_dim": workload.embedding_dim,
        "top_k": workload.top_k,
        "runs": workload.runs,
        "warmup_runs": workload.warmup_runs,
    }
    write_csv(CSV_OUTPUT, rows)
    write_json(JSON_OUTPUT, rows, metadata)
    write_report(REPORT_OUTPUT, rows, metadata)
    write_chart(CHART_OUTPUT, rows)
    print(f"Wrote {CSV_OUTPUT}")
    print(f"Wrote {JSON_OUTPUT}")
    print(f"Wrote {REPORT_OUTPUT}")
    print(f"Wrote {CHART_OUTPUT}")
    return 0


if __name__ == "__main__":
    if len(sys.argv) > 1 and sys.argv[1] == "--graph-only-json":
        raise SystemExit(run_graph_capture_only())
    raise SystemExit(main())
