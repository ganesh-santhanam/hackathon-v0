#!/usr/bin/env python3
"""ROCm-ready industrial incident memory reranking benchmark."""

from __future__ import annotations

import argparse
import csv
import json
import math
import shutil
import subprocess
import time
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import Any


DEFAULT_OUTPUT_DIR = Path("data/benchmarks")
CSV_OUTPUT = DEFAULT_OUTPUT_DIR / "rocm_fused_rerank_results.csv"
JSON_OUTPUT = DEFAULT_OUTPUT_DIR / "rocm_fused_rerank_results.json"
REPORT_OUTPUT = DEFAULT_OUTPUT_DIR / "rocm_fused_rerank_report.md"
CHART_OUTPUT = DEFAULT_OUTPUT_DIR / "rocm_fused_rerank_latency.svg"

TELEMETRY_COLUMNS = [
    "air_temperature",
    "process_temperature",
    "rotational_speed",
    "torque",
    "tool_wear",
]
TELEMETRY_SCALE = [12.0, 14.0, 900.0, 45.0, 250.0]
TELEMETRY_WEIGHT = [0.15, 0.15, 0.25, 0.2, 0.25]


@dataclass(frozen=True)
class PrecisionMode:
    name: str
    dtype_attr: str


PRECISION_MODES = [
    PrecisionMode("fp32", "float32"),
    PrecisionMode("fp16", "float16"),
    PrecisionMode("bf16", "bfloat16"),
    PrecisionMode("fp8", "float8_e4m3fn"),
    PrecisionMode("tf32", "float32"),
]


def utc_timestamp() -> str:
    return datetime.now(UTC).isoformat()


def parse_int_list(value: str) -> list[int]:
    return [int(part.strip()) for part in value.split(",") if part.strip()]


def parse_modes(value: str) -> list[str]:
    return [part.strip().lower() for part in value.split(",") if part.strip()]


def import_torch():
    try:
        import torch
        import torch.nn.functional as functional
    except ImportError as exc:
        raise SystemExit("PyTorch is required for this benchmark.") from exc
    return torch, functional


def is_rocm_build(torch_module: Any) -> bool:
    return bool(getattr(getattr(torch_module, "version", None), "hip", None))


def has_amd_gpu(torch_module: Any) -> bool:
    if not torch_module.cuda.is_available():
        return False
    if is_rocm_build(torch_module):
        return True
    try:
        name = torch_module.cuda.get_device_name(0).lower()
    except Exception:
        return False
    return "amd" in name or "radeon" in name or "instinct" in name


def resolve_device(torch_module: Any, requested: str, allow_cpu: bool) -> Any:
    if requested == "cpu":
        return torch_module.device("cpu")
    if requested == "cuda":
        if not torch_module.cuda.is_available():
            raise SystemExit("Requested cuda device, but no GPU is visible to PyTorch.")
        return torch_module.device("cuda")
    if torch_module.cuda.is_available():
        return torch_module.device("cuda")
    if allow_cpu:
        return torch_module.device("cpu")
    raise SystemExit("No GPU is visible to PyTorch. Use --allow-cpu or --dry-run locally.")


def get_gpu_utilization() -> float | None:
    rocm_smi = shutil.which("rocm-smi")
    if not rocm_smi:
        return None
    try:
        result = subprocess.run(
            [rocm_smi, "--showuse", "--json"],
            check=False,
            capture_output=True,
            text=True,
            timeout=5,
        )
    except (OSError, subprocess.TimeoutExpired):
        return None
    if result.returncode != 0 or not result.stdout.strip():
        return None
    try:
        payload = json.loads(result.stdout)
    except json.JSONDecodeError:
        return None
    values: list[float] = []
    for gpu_payload in payload.values():
        if not isinstance(gpu_payload, dict):
            continue
        for key, raw_value in gpu_payload.items():
            if "GPU use" not in key:
                continue
            try:
                values.append(float(str(raw_value).replace("%", "").strip()))
            except ValueError:
                continue
    if not values:
        return None
    return sum(values) / len(values)


def mode_supported(torch_module: Any, mode: PrecisionMode, device: Any) -> tuple[bool, str]:
    if mode.name == "fp32":
        return True, ""
    if mode.name == "tf32":
        if device.type != "cuda":
            return False, "TF32 is only relevant on GPU."
        matmul_backend = getattr(getattr(torch_module.backends, "cuda", None), "matmul", None)
        if matmul_backend is None or not hasattr(matmul_backend, "allow_tf32"):
            return False, "PyTorch does not expose torch.backends.cuda.matmul.allow_tf32."
        if is_rocm_build(torch_module):
            return False, "TF32 is not cleanly exposed for this ROCm build."
        return True, ""
    if mode.name == "fp8":
        if device.type != "cuda":
            return False, "FP8 mode requires GPU support."
        if not hasattr(torch_module, mode.dtype_attr):
            return False, "PyTorch does not expose torch.float8_e4m3fn."
        return False, "Native FP8 matmul is not exposed cleanly by this PyTorch build."
    if device.type != "cuda":
        return False, f"{mode.name.upper()} benchmark is skipped on CPU fallback."
    if not hasattr(torch_module, mode.dtype_attr):
        return False, f"PyTorch does not expose torch.{mode.dtype_attr}."
    if mode.name == "bf16" and hasattr(torch_module.cuda, "is_bf16_supported"):
        try:
            if not torch_module.cuda.is_bf16_supported():
                return False, "GPU does not report BF16 support."
        except Exception:
            return False, "Could not verify BF16 support."
    return True, ""


def make_inputs(
    torch_module: Any,
    device: Any,
    batch_size: int,
    candidates: int,
    embedding_dim: int,
    seed: int,
) -> dict[str, Any]:
    generator = torch_module.Generator(device="cpu")
    generator.manual_seed(seed)
    query_embeddings = torch_module.randn(batch_size, embedding_dim, generator=generator)
    incident_embeddings = torch_module.randn(candidates, embedding_dim, generator=generator)
    query_telemetry = torch_module.rand(batch_size, len(TELEMETRY_COLUMNS), generator=generator)
    incident_telemetry = torch_module.rand(candidates, len(TELEMETRY_COLUMNS), generator=generator)
    rerank_bonus = torch_module.rand(candidates, generator=generator)

    telemetry_min = torch_module.tensor([290.0, 300.0, 1_100.0, 5.0, 0.0])
    telemetry_span = torch_module.tensor([18.0, 20.0, 1_900.0, 80.0, 253.0])
    query_telemetry = telemetry_min + query_telemetry * telemetry_span
    incident_telemetry = telemetry_min + incident_telemetry * telemetry_span

    return {
        "query_embeddings": query_embeddings.to(device),
        "incident_embeddings": incident_embeddings.to(device),
        "query_telemetry": query_telemetry.to(device),
        "incident_telemetry": incident_telemetry.to(device),
        "rerank_bonus": rerank_bonus.to(device),
    }


def rerank_scores(
    torch_module: Any,
    functional: Any,
    inputs: dict[str, Any],
    dtype: Any,
    top_k: int,
    bonus_weight: float,
) -> tuple[Any, Any]:
    query_embeddings = functional.normalize(inputs["query_embeddings"].to(dtype), dim=1)
    incident_embeddings = functional.normalize(inputs["incident_embeddings"].to(dtype), dim=1)
    embedding_score = query_embeddings @ incident_embeddings.T

    penalty_dtype = torch_module.float32
    query_telemetry = inputs["query_telemetry"].to(penalty_dtype)
    incident_telemetry = inputs["incident_telemetry"].to(penalty_dtype)
    scale = torch_module.tensor(TELEMETRY_SCALE, device=query_telemetry.device, dtype=penalty_dtype)
    weights = torch_module.tensor(TELEMETRY_WEIGHT, device=query_telemetry.device, dtype=penalty_dtype)
    telemetry_delta = (query_telemetry[:, None, :] - incident_telemetry[None, :, :]).abs()
    telemetry_penalty = ((telemetry_delta / scale) * weights).sum(dim=2)
    final_score = embedding_score.float() - telemetry_penalty
    final_score = final_score + inputs["rerank_bonus"].float()[None, :] * bonus_weight
    top_values, top_indices = torch_module.topk(final_score, k=min(top_k, final_score.shape[1]), dim=1)
    return final_score, top_indices if top_values is not None else top_indices


def synchronize(torch_module: Any, device: Any) -> None:
    if device.type == "cuda":
        torch_module.cuda.synchronize()


def peak_vram_gb(torch_module: Any, device: Any) -> float | None:
    if device.type != "cuda":
        return None
    return torch_module.cuda.max_memory_allocated(device) / (1024**3)


def effective_ops(batch_size: int, candidates: int, embedding_dim: int) -> int:
    dot_ops = batch_size * candidates * embedding_dim * 2
    telemetry_ops = batch_size * candidates * len(TELEMETRY_COLUMNS) * 4
    combine_ops = batch_size * candidates * 3
    return dot_ops + telemetry_ops + combine_ops


def topk_overlap(torch_module: Any, baseline: Any, candidate: Any) -> float:
    overlaps = []
    baseline_cpu = baseline.detach().cpu()
    candidate_cpu = candidate.detach().cpu()
    for row_index in range(baseline_cpu.shape[0]):
        left = set(baseline_cpu[row_index].tolist())
        right = set(candidate_cpu[row_index].tolist())
        overlaps.append(len(left & right) / max(1, len(left)))
    return float(sum(overlaps) / len(overlaps))


def benchmark_mode(
    torch_module: Any,
    functional: Any,
    mode: PrecisionMode,
    inputs: dict[str, Any],
    device: Any,
    args: argparse.Namespace,
    baseline_scores: Any | None = None,
    baseline_topk: Any | None = None,
) -> dict[str, Any]:
    supported, skip_reason = mode_supported(torch_module, mode, device)
    row = {
        "timestamp": utc_timestamp(),
        "mode": mode.name,
        "status": "ok" if supported else "skipped",
        "skip_reason": skip_reason,
        "device": str(device),
        "rocm_build": is_rocm_build(torch_module),
        "gpu_name": torch_module.cuda.get_device_name(0) if device.type == "cuda" else "cpu",
        "batch_size": args.current_batch_size,
        "candidates": args.current_candidates,
        "embedding_dim": args.current_embedding_dim,
        "top_k": args.top_k,
        "latency_ms": None,
        "candidates_per_second": None,
        "speedup_vs_fp32": None,
        "effective_ops_per_second": None,
        "peak_vram_gb": None,
        "gpu_utilization": None,
        "max_abs_error_vs_fp32": None,
        "mean_abs_error_vs_fp32": None,
        "top_k_overlap_vs_fp32": None,
    }
    if not supported:
        return row

    dtype = getattr(torch_module, mode.dtype_attr)
    previous_tf32 = None
    if mode.name == "tf32":
        previous_tf32 = torch_module.backends.cuda.matmul.allow_tf32
        torch_module.backends.cuda.matmul.allow_tf32 = True
    elif device.type == "cuda" and hasattr(torch_module.backends.cuda.matmul, "allow_tf32"):
        previous_tf32 = torch_module.backends.cuda.matmul.allow_tf32
        torch_module.backends.cuda.matmul.allow_tf32 = False

    try:
        if device.type == "cuda":
            torch_module.cuda.reset_peak_memory_stats(device)

        with torch_module.no_grad():
            for _ in range(args.warmup_runs):
                rerank_scores(torch_module, functional, inputs, dtype, args.top_k, args.bonus_weight)
            synchronize(torch_module, device)

            latencies = []
            final_scores = None
            top_indices = None
            for _ in range(args.runs):
                start = time.perf_counter()
                final_scores, top_indices = rerank_scores(
                    torch_module,
                    functional,
                    inputs,
                    dtype,
                    args.top_k,
                    args.bonus_weight,
                )
                synchronize(torch_module, device)
                latencies.append((time.perf_counter() - start) * 1000.0)

        latency_ms = min(latencies)
        candidates_per_second = (
            args.current_batch_size * args.current_candidates / max(latency_ms / 1000.0, 1e-12)
        )
        ops = effective_ops(args.current_batch_size, args.current_candidates, args.current_embedding_dim)
        row.update(
            {
                "latency_ms": latency_ms,
                "candidates_per_second": candidates_per_second,
                "effective_ops_per_second": ops / max(latency_ms / 1000.0, 1e-12),
                "peak_vram_gb": peak_vram_gb(torch_module, device),
                "gpu_utilization": get_gpu_utilization() if device.type == "cuda" else None,
            }
        )
        if baseline_scores is not None and final_scores is not None and baseline_topk is not None:
            diff = (final_scores.float() - baseline_scores.float()).abs()
            row["max_abs_error_vs_fp32"] = float(diff.max().item())
            row["mean_abs_error_vs_fp32"] = float(diff.mean().item())
            row["top_k_overlap_vs_fp32"] = topk_overlap(torch_module, baseline_topk, top_indices)
    finally:
        if previous_tf32 is not None:
            torch_module.backends.cuda.matmul.allow_tf32 = previous_tf32

    return row


def write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = list(rows[0].keys()) if rows else []
    with path.open("w", newline="", encoding="utf-8") as output_file:
        writer = csv.DictWriter(output_file, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def write_json(path: Path, rows: list[dict[str, Any]], metadata: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = {"metadata": metadata, "results": rows}
    path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def format_number(value: Any, digits: int = 3) -> str:
    if value is None:
        return ""
    if isinstance(value, str):
        return value
    if isinstance(value, bool):
        return str(value)
    if isinstance(value, int):
        return str(value)
    if isinstance(value, (int, float)) and math.isfinite(value):
        return f"{value:.{digits}f}"
    return str(value)


def write_report(path: Path, rows: list[dict[str, Any]], metadata: dict[str, Any]) -> None:
    successful = [row for row in rows if row["status"] == "ok"]
    skipped = [row for row in rows if row["status"] == "skipped"]
    best = min(successful, key=lambda row: row["latency_ms"]) if successful else None
    lines = [
        "# ROCm Fused Rerank Benchmark Report",
        "",
        f"- Generated: {metadata['generated_at']}",
        f"- Device: {metadata['device']}",
        f"- ROCm build: {metadata['rocm_build']}",
        f"- Rows: {len(rows)}",
        f"- Successful runs: {len(successful)}",
        f"- Skipped runs: {len(skipped)}",
        "",
    ]
    if best:
        lines.extend(
            [
                "## Best Latency",
                "",
                (
                    f"{best['mode']} at batch {best['batch_size']}, {best['candidates']} candidates, "
                    f"dim {best['embedding_dim']}: {format_number(best['latency_ms'])} ms"
                ),
                "",
            ]
        )
    lines.extend(
        [
            "## Results",
            "",
            "| mode | batch | candidates | dim | status | latency_ms | candidates/s | "
            "speedup | peak_vram_gb | max_error | mean_error | top_k_overlap |",
            "| --- | ---: | ---: | ---: | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |",
        ]
    )
    for row in rows:
        lines.append(
            "| {mode} | {batch_size} | {candidates} | {embedding_dim} | {status} | "
            "{latency_ms} | {candidates_per_second} | {speedup_vs_fp32} | {peak_vram_gb} | "
            "{max_abs_error_vs_fp32} | {mean_abs_error_vs_fp32} | {top_k_overlap_vs_fp32} |".format(
                **{key: format_number(value) for key, value in row.items()}
            )
        )
    if skipped:
        lines.extend(["", "## Skipped Modes", ""])
        for row in skipped:
            lines.append(
                f"- {row['mode']} batch={row['batch_size']} candidates={row['candidates']} "
                f"dim={row['embedding_dim']}: {row['skip_reason']}"
            )
    lines.append("")
    path.write_text("\n".join(lines), encoding="utf-8")


def write_svg_chart(path: Path, rows: list[dict[str, Any]]) -> None:
    plotted = [row for row in rows if row["status"] == "ok" and row["latency_ms"] is not None]
    if not plotted:
        return
    width = 1100
    height = 420
    margin_left = 90
    margin_bottom = 120
    chart_width = width - margin_left - 40
    chart_height = height - 70 - margin_bottom
    max_latency = max(float(row["latency_ms"]) for row in plotted)
    bar_width = max(8, chart_width / len(plotted) * 0.65)
    step = chart_width / len(plotted)
    colors = {"fp32": "#1f77b4", "fp16": "#2ca02c", "bf16": "#9467bd", "tf32": "#ff7f0e"}
    svg = [
        f'<svg xmlns="http://www.w3.org/2000/svg" width="{width}" height="{height}" '
        f'viewBox="0 0 {width} {height}">',
        '<rect width="100%" height="100%" fill="white"/>',
        '<text x="40" y="32" font-family="Arial" font-size="22" font-weight="700">'
        "ROCm fused rerank latency</text>",
        f'<line x1="{margin_left}" y1="50" x2="{margin_left}" y2="{height - margin_bottom}" '
        'stroke="#333"/>',
        f'<line x1="{margin_left}" y1="{height - margin_bottom}" x2="{width - 35}" '
        f'y2="{height - margin_bottom}" stroke="#333"/>',
    ]
    for index, row in enumerate(plotted):
        latency = float(row["latency_ms"])
        bar_height = latency / max_latency * chart_height
        x = margin_left + index * step + (step - bar_width) / 2
        y = height - margin_bottom - bar_height
        label = f"{row['mode']} b{row['batch_size']} n{row['candidates']//1000}k d{row['embedding_dim']}"
        svg.extend(
            [
                f'<rect x="{x:.1f}" y="{y:.1f}" width="{bar_width:.1f}" height="{bar_height:.1f}" '
                f'fill="{colors.get(row["mode"], "#777")}"/>',
                f'<text x="{x + bar_width / 2:.1f}" y="{y - 6:.1f}" font-family="Arial" '
                f'font-size="10" text-anchor="middle">{latency:.1f}</text>',
                f'<text x="{x + bar_width / 2:.1f}" y="{height - margin_bottom + 14}" '
                'font-family="Arial" font-size="9" text-anchor="end" '
                f'transform="rotate(-45 {x + bar_width / 2:.1f} {height - margin_bottom + 14})">'
                f"{label}</text>",
            ]
        )
    svg.append("</svg>")
    path.write_text("\n".join(svg) + "\n", encoding="utf-8")


def run_benchmark(args: argparse.Namespace) -> int:
    torch_module, functional = import_torch()
    if args.dry_run:
        args.allow_cpu = True
        args.candidates = [min(args.candidates[0], 512)]
        args.embedding_dims = [min(args.embedding_dims[0], 384)]
        args.batch_sizes = [min(args.batch_sizes[0], 2)]
        args.runs = 1
        args.warmup_runs = 0

    device = resolve_device(torch_module, args.device, args.allow_cpu)
    requested_modes = set(args.modes)
    selected_modes = [mode for mode in PRECISION_MODES if mode.name in requested_modes]
    if "fp32" not in requested_modes:
        selected_modes.insert(0, PRECISION_MODES[0])

    rows: list[dict[str, Any]] = []
    for embedding_dim in args.embedding_dims:
        for candidates in args.candidates:
            for batch_size in args.batch_sizes:
                args.current_embedding_dim = embedding_dim
                args.current_candidates = candidates
                args.current_batch_size = batch_size
                try:
                    inputs = make_inputs(
                        torch_module,
                        device,
                        batch_size,
                        candidates,
                        embedding_dim,
                        args.seed,
                    )
                except RuntimeError as exc:
                    rows.append(
                        {
                            "timestamp": utc_timestamp(),
                            "mode": "all",
                            "status": "skipped",
                            "skip_reason": f"input allocation failed: {exc}",
                            "device": str(device),
                            "rocm_build": is_rocm_build(torch_module),
                            "gpu_name": (
                                torch_module.cuda.get_device_name(0)
                                if device.type == "cuda"
                                else "cpu"
                            ),
                            "batch_size": batch_size,
                            "candidates": candidates,
                            "embedding_dim": embedding_dim,
                            "top_k": args.top_k,
                            "latency_ms": None,
                            "candidates_per_second": None,
                            "speedup_vs_fp32": None,
                            "effective_ops_per_second": None,
                            "peak_vram_gb": None,
                            "gpu_utilization": None,
                            "max_abs_error_vs_fp32": None,
                            "mean_abs_error_vs_fp32": None,
                            "top_k_overlap_vs_fp32": None,
                        }
                    )
                    continue

                baseline_row = benchmark_mode(
                    torch_module,
                    functional,
                    PRECISION_MODES[0],
                    inputs,
                    device,
                    args,
                )
                rows.append(baseline_row)
                baseline_latency = baseline_row["latency_ms"]
                baseline_scores = None
                baseline_topk = None
                if baseline_row["status"] == "ok":
                    with torch_module.no_grad():
                        baseline_scores, baseline_topk = rerank_scores(
                            torch_module,
                            functional,
                            inputs,
                            torch_module.float32,
                            args.top_k,
                            args.bonus_weight,
                        )
                    synchronize(torch_module, device)

                for mode in selected_modes:
                    if mode.name == "fp32":
                        continue
                    row = benchmark_mode(
                        torch_module,
                        functional,
                        mode,
                        inputs,
                        device,
                        args,
                        baseline_scores=baseline_scores,
                        baseline_topk=baseline_topk,
                    )
                    if (
                        row["status"] == "ok"
                        and row["latency_ms"] is not None
                        and baseline_latency is not None
                    ):
                        row["speedup_vs_fp32"] = baseline_latency / row["latency_ms"]
                    rows.append(row)

                del inputs
                if device.type == "cuda":
                    torch_module.cuda.empty_cache()

    metadata = {
        "generated_at": utc_timestamp(),
        "device": str(device),
        "rocm_build": is_rocm_build(torch_module),
        "amd_gpu_visible": has_amd_gpu(torch_module),
        "dry_run": args.dry_run,
        "telemetry_columns": TELEMETRY_COLUMNS,
        "formula": "final_score = embedding_score - telemetry_penalty + weighted_rerank_bonus",
    }
    write_csv(args.csv_output, rows)
    write_json(args.json_output, rows, metadata)
    write_report(args.report_output, rows, metadata)
    if args.chart:
        write_svg_chart(args.chart_output, rows)

    print(f"Wrote CSV: {args.csv_output}")
    print(f"Wrote JSON: {args.json_output}")
    print(f"Wrote report: {args.report_output}")
    if args.chart and args.chart_output.exists():
        print(f"Wrote chart: {args.chart_output}")
    return 0


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--device", choices=["auto", "cuda", "cpu"], default="auto")
    parser.add_argument("--allow-cpu", action="store_true")
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--candidates", type=parse_int_list, default=parse_int_list("10000,100000,1000000"))
    parser.add_argument("--embedding-dims", type=parse_int_list, default=parse_int_list("384,768"))
    parser.add_argument("--batch-sizes", type=parse_int_list, default=parse_int_list("1,8,32"))
    parser.add_argument("--modes", type=parse_modes, default=parse_modes("fp32,fp16,bf16,fp8,tf32"))
    parser.add_argument("--top-k", type=int, default=10)
    parser.add_argument("--runs", type=int, default=5)
    parser.add_argument("--warmup-runs", type=int, default=2)
    parser.add_argument("--bonus-weight", type=float, default=0.05)
    parser.add_argument("--seed", type=int, default=7)
    parser.add_argument("--csv-output", type=Path, default=CSV_OUTPUT)
    parser.add_argument("--json-output", type=Path, default=JSON_OUTPUT)
    parser.add_argument("--report-output", type=Path, default=REPORT_OUTPUT)
    parser.add_argument("--chart-output", type=Path, default=CHART_OUTPUT)
    parser.add_argument("--chart", action="store_true")
    return parser


def main() -> int:
    parser = build_parser()
    args = parser.parse_args()
    if args.top_k < 1:
        parser.error("--top-k must be positive")
    if args.runs < 1:
        parser.error("--runs must be positive")
    if not args.candidates or not args.embedding_dims or not args.batch_sizes:
        parser.error("--candidates, --embedding-dims, and --batch-sizes must not be empty")
    return run_benchmark(args)


if __name__ == "__main__":
    raise SystemExit(main())
