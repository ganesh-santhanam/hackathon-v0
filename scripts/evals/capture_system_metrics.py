#!/usr/bin/env python3
"""Capture host, runtime, and GPU telemetry for evaluation runs."""

from __future__ import annotations

import argparse
import json
import os
import platform
import socket
import statistics
import subprocess
import time
from datetime import UTC, datetime
from pathlib import Path
from typing import Any


NOT_AVAILABLE = "NOT AVAILABLE"
DEFAULT_OUTPUT_DIR = Path("data/evals/full_run")


def utc_now() -> str:
    return datetime.now(UTC).isoformat()


def run_command(command: list[str]) -> str | None:
    try:
        result = subprocess.run(command, check=False, capture_output=True, text=True, timeout=10)
    except (FileNotFoundError, subprocess.SubprocessError):
        return None
    if result.returncode != 0:
        return None
    return result.stdout.strip()


def import_version(module_name: str, version_attr: str = "__version__") -> str:
    try:
        module = __import__(module_name)
        return str(getattr(module, version_attr, NOT_AVAILABLE))
    except Exception:
        return NOT_AVAILABLE


def cpu_model() -> str:
    if platform.system() == "Linux":
        cpuinfo = Path("/proc/cpuinfo")
        if cpuinfo.exists():
            for line in cpuinfo.read_text(encoding="utf-8", errors="ignore").splitlines():
                if line.lower().startswith("model name"):
                    return line.split(":", 1)[1].strip()
    value = platform.processor()
    return value or NOT_AVAILABLE


def ram_total_gb() -> float | str:
    if platform.system() == "Linux":
        meminfo = Path("/proc/meminfo")
        if meminfo.exists():
            for line in meminfo.read_text(encoding="utf-8", errors="ignore").splitlines():
                if line.startswith("MemTotal:"):
                    kb = float(line.split()[1])
                    return round(kb / 1024 / 1024, 2)
    try:
        pages = os.sysconf("SC_PHYS_PAGES")
        page_size = os.sysconf("SC_PAGE_SIZE")
        return round((pages * page_size) / 1024**3, 2)
    except (AttributeError, ValueError, OSError):
        return NOT_AVAILABLE


def torch_runtime() -> dict[str, Any]:
    runtime = {
        "python_version": platform.python_version(),
        "torch_version": NOT_AVAILABLE,
        "rocm_version": NOT_AVAILABLE,
        "cuda_version": NOT_AVAILABLE,
        "transformers_version": import_version("transformers"),
        "peft_version": import_version("peft"),
        "vllm_version": import_version("vllm"),
    }
    try:
        import torch

        runtime["torch_version"] = str(torch.__version__)
        runtime["rocm_version"] = str(getattr(torch.version, "hip", None) or NOT_AVAILABLE)
        runtime["cuda_version"] = str(getattr(torch.version, "cuda", None) or NOT_AVAILABLE)
    except Exception:
        pass
    return runtime


def torch_gpu_profile() -> dict[str, Any] | None:
    try:
        import torch

        if not torch.cuda.is_available():
            return None
        devices = []
        for index in range(torch.cuda.device_count()):
            props = torch.cuda.get_device_properties(index)
            devices.append(
                {
                    "index": index,
                    "name": torch.cuda.get_device_name(index),
                    "memory_total_gb": round(props.total_memory / 1024**3, 2),
                }
            )
        return {"gpu_count": len(devices), "gpus": devices}
    except Exception:
        return None


def nvidia_profile() -> dict[str, Any] | None:
    output = run_command(
        [
            "nvidia-smi",
            "--query-gpu=index,name,memory.total",
            "--format=csv,noheader,nounits",
        ]
    )
    if not output:
        return None
    gpus = []
    for line in output.splitlines():
        parts = [part.strip() for part in line.split(",")]
        if len(parts) >= 3:
            gpus.append(
                {
                    "index": int(parts[0]) if parts[0].isdigit() else parts[0],
                    "name": parts[1],
                    "memory_total_gb": round(float(parts[2]) / 1024, 2),
                }
            )
    return {"gpu_count": len(gpus), "gpus": gpus} if gpus else None


def rocm_profile() -> dict[str, Any] | None:
    output = run_command(["rocm-smi", "--showproductname", "--showmeminfo", "vram", "--json"])
    if not output:
        return None
    try:
        parsed = json.loads(output)
    except json.JSONDecodeError:
        return None
    gpus = []
    for key, value in parsed.items():
        if not isinstance(value, dict):
            continue
        name = value.get("Card series") or value.get("Card model") or value.get("GPU ID") or key
        total_bytes = None
        for metric_name, metric_value in value.items():
            if "VRAM Total Memory" in metric_name:
                total_bytes = metric_value
                break
        try:
            memory_gb: float | str = round(float(total_bytes) / 1024**3, 2)
        except (TypeError, ValueError):
            memory_gb = NOT_AVAILABLE
        gpus.append({"index": key, "name": str(name), "memory_total_gb": memory_gb})
    return {"gpu_count": len(gpus), "gpus": gpus} if gpus else None


def hardware_profile() -> dict[str, Any]:
    gpu_profile = nvidia_profile() or rocm_profile() or torch_gpu_profile()
    gpus = gpu_profile["gpus"] if gpu_profile else []
    return {
        "captured_at": utc_now(),
        "hostname": socket.gethostname(),
        "os": platform.platform(),
        "cpu_model": cpu_model(),
        "ram_total_gb": ram_total_gb(),
        "gpu_name": ", ".join(str(gpu.get("name")) for gpu in gpus) if gpus else NOT_AVAILABLE,
        "gpu_count": gpu_profile["gpu_count"] if gpu_profile else 0,
        "gpu_memory": [gpu.get("memory_total_gb", NOT_AVAILABLE) for gpu in gpus],
        "gpus": gpus,
    }


def parse_float(value: str) -> float | None:
    cleaned = value.strip().replace("%", "").replace("W", "").replace("C", "").replace("MiB", "")
    try:
        return float(cleaned)
    except ValueError:
        return None


def nvidia_sample() -> list[dict[str, Any]]:
    output = run_command(
        [
            "nvidia-smi",
            "--query-gpu=memory.used,utilization.gpu,power.draw,temperature.gpu",
            "--format=csv,noheader,nounits",
        ]
    )
    if not output:
        return []
    samples = []
    for line in output.splitlines():
        parts = [parse_float(part) for part in line.split(",")]
        if len(parts) >= 4:
            samples.append(
                {
                    "vram_gb": None if parts[0] is None else parts[0] / 1024,
                    "gpu_utilization_percent": parts[1],
                    "power_draw_w": parts[2],
                    "temperature_c": parts[3],
                }
            )
    return samples


def rocm_sample() -> list[dict[str, Any]]:
    output = run_command(
        [
            "rocm-smi",
            "--showuse",
            "--showmemuse",
            "--showmeminfo",
            "vram",
            "--showpower",
            "--showtemp",
            "--json",
        ]
    )
    if not output:
        return []
    try:
        parsed = json.loads(output)
    except json.JSONDecodeError:
        return []
    samples = []
    for value in parsed.values():
        if not isinstance(value, dict):
            continue
        sample: dict[str, Any] = {
            "vram_gb": None,
            "gpu_utilization_percent": None,
            "power_draw_w": None,
            "temperature_c": None,
        }
        for metric_name, metric_value in value.items():
            numeric = parse_float(str(metric_value))
            lower_name = metric_name.lower()
            if "vram total used memory" in lower_name:
                sample["vram_gb"] = None if numeric is None else numeric / 1024**3
            elif "gpu use" in lower_name:
                sample["gpu_utilization_percent"] = numeric
            elif "power" in lower_name:
                sample["power_draw_w"] = numeric
            elif "temperature" in lower_name or "temp" in lower_name:
                sample["temperature_c"] = numeric
        samples.append(sample)
    return samples


def summarize_samples(samples: list[dict[str, Any]]) -> dict[str, Any]:
    summary: dict[str, Any] = {}
    field_map = {
        "vram_gb": ("peak_vram_gb", "average_vram_gb"),
        "gpu_utilization_percent": ("peak_gpu_utilization", "average_gpu_utilization"),
        "power_draw_w": ("peak_power_draw", "average_power_draw"),
        "temperature_c": ("peak_temperature", "average_temperature"),
    }
    for source, (peak_key, average_key) in field_map.items():
        values = [sample[source] for sample in samples if isinstance(sample.get(source), (int, float))]
        if values:
            summary[peak_key] = round(max(values), 4)
            summary[average_key] = round(statistics.fmean(values), 4)
        else:
            summary[peak_key] = NOT_AVAILABLE
            summary[average_key] = NOT_AVAILABLE
    summary["sample_count"] = len(samples)
    return summary


def sample_gpu_metrics(duration_seconds: float, interval_seconds: float) -> dict[str, Any]:
    started = time.perf_counter()
    raw_samples: list[dict[str, Any]] = []
    while time.perf_counter() - started < duration_seconds:
        timestamp = utc_now()
        samples = nvidia_sample() or rocm_sample()
        for sample in samples:
            raw_samples.append({"timestamp": timestamp, **sample})
        time.sleep(interval_seconds)
    return {"summary": summarize_samples(raw_samples), "samples": raw_samples}


def sample_while_process(command: list[str], interval_seconds: float) -> tuple[int, dict[str, Any]]:
    process = subprocess.Popen(command)
    raw_samples: list[dict[str, Any]] = []
    while process.poll() is None:
        timestamp = utc_now()
        for sample in nvidia_sample() or rocm_sample():
            raw_samples.append({"timestamp": timestamp, **sample})
        time.sleep(interval_seconds)
    return process.returncode or 0, {"summary": summarize_samples(raw_samples), "samples": raw_samples}


def write_metrics(output_dir: Path, metrics: dict[str, Any], profile: dict[str, Any]) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    (output_dir / "system_metrics.json").write_text(json.dumps(metrics, indent=2), encoding="utf-8")
    (output_dir / "hardware_profile.json").write_text(json.dumps(profile, indent=2), encoding="utf-8")


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Capture evaluation hardware and runtime metrics.")
    parser.add_argument("--output-dir", default=DEFAULT_OUTPUT_DIR, type=Path)
    parser.add_argument("--sample-seconds", default=1.0, type=float)
    parser.add_argument("--interval-seconds", default=1.0, type=float)
    parser.add_argument("--command", nargs=argparse.REMAINDER)
    return parser


def main() -> int:
    args = build_parser().parse_args()
    started = time.perf_counter()
    profile = hardware_profile()
    return_code = 0

    if args.command:
        return_code, gpu_metrics = sample_while_process(args.command, args.interval_seconds)
    else:
        gpu_metrics = sample_gpu_metrics(args.sample_seconds, args.interval_seconds)

    wall_clock_runtime = round(time.perf_counter() - started, 4)
    metrics = {
        "captured_at": utc_now(),
        "runtime": torch_runtime(),
        "gpu_metrics": gpu_metrics["summary"],
        "timing": {
            "wall_clock_runtime_seconds": wall_clock_runtime,
            "training_runtime_seconds": {
                "value": NOT_AVAILABLE,
                "reason": "No training command was wrapped by capture_system_metrics.py.",
            },
            "inference_runtime_seconds": {
                "value": NOT_AVAILABLE,
                "reason": "No inference command was wrapped by capture_system_metrics.py.",
            },
            "benchmark_runtime_seconds": {
                "value": NOT_AVAILABLE,
                "reason": "No benchmark command was wrapped by capture_system_metrics.py.",
            },
        },
        "raw_gpu_samples": gpu_metrics["samples"],
    }
    write_metrics(args.output_dir, metrics, profile)
    print(json.dumps({"output_dir": str(args.output_dir), "return_code": return_code}, indent=2))
    return return_code


if __name__ == "__main__":
    raise SystemExit(main())
