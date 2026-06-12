#!/usr/bin/env python3
"""Run the unified evaluation packaging flow."""

from __future__ import annotations

import argparse
import importlib.util
import json
import subprocess
import sys
import time
from pathlib import Path
from typing import Any


PROJECT_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_OUTPUT_DIR = PROJECT_ROOT / "data" / "evals" / "full_run"


def load_module(path: Path, name: str):
    spec = importlib.util.spec_from_file_location(name, path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"Could not load {path}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def run_capture(output_dir: Path, sample_seconds: float, interval_seconds: float) -> None:
    capture = load_module(Path(__file__).with_name("capture_system_metrics.py"), "capture_system_metrics")
    profile = capture.hardware_profile()
    gpu_metrics = capture.sample_gpu_metrics(sample_seconds, interval_seconds)
    system_metrics = {
        "captured_at": capture.utc_now(),
        "runtime": capture.torch_runtime(),
        "gpu_metrics": gpu_metrics["summary"],
        "timing": {
            "wall_clock_runtime_seconds": "NOT AVAILABLE",
            "training_runtime_seconds": {
                "value": "NOT AVAILABLE",
                "reason": "No training command was supplied to run_full_evaluation.py.",
            },
            "inference_runtime_seconds": {
                "value": "NOT AVAILABLE",
                "reason": "No inference command was supplied to run_full_evaluation.py.",
            },
            "benchmark_runtime_seconds": {
                "value": "NOT AVAILABLE",
                "reason": "No benchmark command was supplied to run_full_evaluation.py.",
            },
        },
        "raw_gpu_samples": gpu_metrics["samples"],
    }
    capture.write_metrics(output_dir, system_metrics, profile)


def update_timing(output_dir: Path, timing: dict[str, Any]) -> None:
    metrics_path = output_dir / "system_metrics.json"
    if not metrics_path.exists():
        return
    metrics = json.loads(metrics_path.read_text(encoding="utf-8"))
    metrics["timing"] = {**metrics.get("timing", {}), **timing}
    metrics_path.write_text(json.dumps(metrics, indent=2), encoding="utf-8")


def generate_report(output_dir: Path) -> dict[str, Any]:
    report = load_module(Path(__file__).with_name("generate_evaluation_report.py"), "generate_evaluation_report")
    return report.generate(output_dir)


def maybe_run_command(command: list[str] | None, cwd: Path) -> tuple[float, int | None]:
    if not command:
        return 0.0, None
    started = time.perf_counter()
    result = subprocess.run(command, cwd=cwd, check=False)
    return time.perf_counter() - started, result.returncode


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Build the full evaluation package.")
    parser.add_argument("--output-dir", default=DEFAULT_OUTPUT_DIR, type=Path)
    parser.add_argument("--sample-seconds", default=2.0, type=float)
    parser.add_argument("--interval-seconds", default=1.0, type=float)
    parser.add_argument("--training-command", nargs="+")
    parser.add_argument("--inference-command", nargs="+")
    parser.add_argument("--benchmark-command", nargs="+")
    return parser


def main() -> int:
    args = build_parser().parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    started = time.perf_counter()

    training_runtime, training_return_code = maybe_run_command(args.training_command, PROJECT_ROOT)
    inference_runtime, inference_return_code = maybe_run_command(args.inference_command, PROJECT_ROOT)
    benchmark_runtime, benchmark_return_code = maybe_run_command(args.benchmark_command, PROJECT_ROOT)

    run_capture(args.output_dir, args.sample_seconds, args.interval_seconds)
    update_timing(
        args.output_dir,
        {
            "wall_clock_runtime_seconds": round(time.perf_counter() - started, 4),
            "training_runtime_seconds": round(training_runtime, 4)
            if args.training_command
            else {
                "value": "NOT AVAILABLE",
                "reason": "No training command was supplied to run_full_evaluation.py.",
            },
            "inference_runtime_seconds": round(inference_runtime, 4)
            if args.inference_command
            else {
                "value": "NOT AVAILABLE",
                "reason": "No inference command was supplied to run_full_evaluation.py.",
            },
            "benchmark_runtime_seconds": round(benchmark_runtime, 4)
            if args.benchmark_command
            else {
                "value": "NOT AVAILABLE",
                "reason": "No benchmark command was supplied to run_full_evaluation.py.",
            },
            "training_return_code": training_return_code,
            "inference_return_code": inference_return_code,
            "benchmark_return_code": benchmark_return_code,
        },
    )
    summary = generate_report(args.output_dir)
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
