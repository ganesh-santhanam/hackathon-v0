#!/usr/bin/env python3
import argparse
import json
import os
import statistics
import time
import urllib.error
import urllib.request
from datetime import UTC, datetime
from pathlib import Path
from typing import Any


DEFAULT_OUTPUT_PATH = Path("data/benchmarks/llm_benchmark.jsonl")
DEFAULT_SUMMARY_OUTPUT_PATH = Path("data/benchmarks/llm_benchmark_summary.json")
DEFAULT_PROMPT = "Summarize the likely cause of a tool wear failure in one sentence."
DEFAULT_OLLAMA_ENDPOINT = "http://localhost:11434/api/generate"
DEFAULT_OPENAI_COMPATIBLE_ENDPOINT = "http://localhost:8000/v1/chat/completions"
DEFAULT_TIMEOUT_SECONDS = 120.0


def utc_timestamp() -> str:
    return datetime.now(UTC).isoformat()


def normalize_endpoint(endpoint: str | None, provider: str) -> str:
    if endpoint:
        return endpoint
    if provider == "ollama":
        return DEFAULT_OLLAMA_ENDPOINT
    return DEFAULT_OPENAI_COMPATIBLE_ENDPOINT


def request_headers(extra_headers: dict[str, str] | None = None) -> dict[str, str]:
    return {"Content-Type": "application/json", **(extra_headers or {})}


def post_json(
    endpoint: str,
    payload: dict[str, Any],
    timeout_seconds: float,
    headers: dict[str, str] | None = None,
) -> tuple[dict[str, Any], int | None]:
    request = urllib.request.Request(
        endpoint,
        data=json.dumps(payload).encode("utf-8"),
        headers=request_headers(headers),
        method="POST",
    )
    with urllib.request.urlopen(request, timeout=timeout_seconds) as response:
        status_code = getattr(response, "status", None)
        body = response.read().decode("utf-8")
    parsed = json.loads(body)
    if not isinstance(parsed, dict):
        raise ValueError("Expected endpoint to return a JSON object.")
    return parsed, status_code


def post_stream(
    endpoint: str,
    payload: dict[str, Any],
    timeout_seconds: float,
    headers: dict[str, str] | None = None,
):
    request = urllib.request.Request(
        endpoint,
        data=json.dumps(payload).encode("utf-8"),
        headers=request_headers(headers),
        method="POST",
    )
    response = urllib.request.urlopen(request, timeout=timeout_seconds)
    return response, getattr(response, "status", None)


def ollama_payload(model: str, prompt: str, stream: bool) -> dict[str, Any]:
    return {
        "model": model,
        "prompt": prompt,
        "stream": stream,
    }


def openai_compatible_payload(model: str, prompt: str, stream: bool) -> dict[str, Any]:
    return {
        "model": model,
        "messages": [{"role": "user", "content": prompt}],
        "stream": stream,
    }


def openai_headers() -> dict[str, str]:
    api_key = os.environ.get("OPENAI_API_KEY")
    return {"Authorization": f"Bearer {api_key}"} if api_key else {}


def token_total(prompt_tokens: Any, completion_tokens: Any) -> int | None:
    if isinstance(prompt_tokens, int) and isinstance(completion_tokens, int):
        return prompt_tokens + completion_tokens
    return None


def ollama_non_streaming(
    endpoint: str,
    model: str,
    prompt: str,
    timeout_seconds: float,
) -> tuple[dict[str, Any], int | None]:
    response, status_code = post_json(
        endpoint,
        ollama_payload(model, prompt, stream=False),
        timeout_seconds=timeout_seconds,
    )
    prompt_tokens = response.get("prompt_eval_count")
    completion_tokens = response.get("eval_count")
    return {
        "output_text": str(response.get("response", "")),
        "prompt_tokens": prompt_tokens,
        "completion_tokens": completion_tokens,
        "total_tokens": token_total(prompt_tokens, completion_tokens),
        "ttft_ms": None,
    }, status_code


def ollama_streaming(
    endpoint: str,
    model: str,
    prompt: str,
    timeout_seconds: float,
    started_at: float,
) -> tuple[dict[str, Any], int | None]:
    response, status_code = post_stream(
        endpoint,
        ollama_payload(model, prompt, stream=True),
        timeout_seconds=timeout_seconds,
    )
    output_parts = []
    ttft_ms = None
    prompt_tokens = None
    completion_tokens = None
    try:
        for raw_line in response:
            line = raw_line.decode("utf-8").strip()
            if not line:
                continue
            chunk = json.loads(line)
            content = str(chunk.get("response", ""))
            if content and ttft_ms is None:
                ttft_ms = (time.perf_counter() - started_at) * 1000
            output_parts.append(content)
            if chunk.get("done"):
                prompt_tokens = chunk.get("prompt_eval_count")
                completion_tokens = chunk.get("eval_count")
    finally:
        response.close()
    return {
        "output_text": "".join(output_parts),
        "prompt_tokens": prompt_tokens,
        "completion_tokens": completion_tokens,
        "total_tokens": token_total(prompt_tokens, completion_tokens),
        "ttft_ms": ttft_ms,
    }, status_code


def openai_non_streaming(
    endpoint: str,
    model: str,
    prompt: str,
    timeout_seconds: float,
) -> tuple[dict[str, Any], int | None]:
    response, status_code = post_json(
        endpoint,
        openai_compatible_payload(model, prompt, stream=False),
        timeout_seconds=timeout_seconds,
        headers=openai_headers(),
    )
    choices = response.get("choices", [])
    output_text = ""
    if choices and isinstance(choices[0], dict):
        message = choices[0].get("message", {})
        if isinstance(message, dict):
            output_text = str(message.get("content", ""))
    usage = response.get("usage", {})
    if not isinstance(usage, dict):
        usage = {}
    return {
        "output_text": output_text,
        "prompt_tokens": usage.get("prompt_tokens"),
        "completion_tokens": usage.get("completion_tokens"),
        "total_tokens": usage.get("total_tokens"),
        "ttft_ms": None,
    }, status_code


def parse_sse_payload(line: str) -> dict[str, Any] | None:
    if not line.startswith("data:"):
        return None
    data = line.removeprefix("data:").strip()
    if not data or data == "[DONE]":
        return None
    parsed = json.loads(data)
    return parsed if isinstance(parsed, dict) else None


def openai_streaming(
    endpoint: str,
    model: str,
    prompt: str,
    timeout_seconds: float,
    started_at: float,
) -> tuple[dict[str, Any], int | None]:
    response, status_code = post_stream(
        endpoint,
        openai_compatible_payload(model, prompt, stream=True),
        timeout_seconds=timeout_seconds,
        headers=openai_headers(),
    )
    output_parts = []
    ttft_ms = None
    prompt_tokens = None
    completion_tokens = None
    total_tokens = None
    try:
        for raw_line in response:
            line = raw_line.decode("utf-8").strip()
            if not line:
                continue
            chunk = parse_sse_payload(line)
            if chunk is None:
                continue
            usage = chunk.get("usage")
            if isinstance(usage, dict):
                prompt_tokens = usage.get("prompt_tokens", prompt_tokens)
                completion_tokens = usage.get("completion_tokens", completion_tokens)
                total_tokens = usage.get("total_tokens", total_tokens)
            choices = chunk.get("choices", [])
            if not choices or not isinstance(choices[0], dict):
                continue
            delta = choices[0].get("delta", {})
            if not isinstance(delta, dict):
                continue
            content = str(delta.get("content", ""))
            if content and ttft_ms is None:
                ttft_ms = (time.perf_counter() - started_at) * 1000
            output_parts.append(content)
    finally:
        response.close()
    return {
        "output_text": "".join(output_parts),
        "prompt_tokens": prompt_tokens,
        "completion_tokens": completion_tokens,
        "total_tokens": total_tokens,
        "ttft_ms": ttft_ms,
    }, status_code


def call_provider(
    provider: str,
    endpoint: str,
    model: str,
    prompt: str,
    stream: bool,
    dry_run: bool,
    timeout_seconds: float,
    started_at: float,
) -> tuple[dict[str, Any], int | None]:
    if dry_run:
        return {
            "output_text": "dry-run response",
            "prompt_tokens": None,
            "completion_tokens": None,
            "total_tokens": None,
            "ttft_ms": 0.0 if stream else None,
        }, 200

    if provider == "ollama":
        if stream:
            return ollama_streaming(endpoint, model, prompt, timeout_seconds, started_at)
        return ollama_non_streaming(endpoint, model, prompt, timeout_seconds)

    if stream:
        return openai_streaming(endpoint, model, prompt, timeout_seconds, started_at)
    return openai_non_streaming(endpoint, model, prompt, timeout_seconds)


def error_type(exc: Exception) -> str:
    if isinstance(exc, urllib.error.HTTPError):
        return "http_error"
    if isinstance(exc, urllib.error.URLError):
        return "url_error"
    if isinstance(exc, TimeoutError):
        return "timeout"
    if isinstance(exc, json.JSONDecodeError):
        return "json_decode_error"
    return type(exc).__name__


def tokens_per_second(completion_tokens: Any, generation_latency_ms: float | None) -> float | None:
    if (
        not isinstance(completion_tokens, int)
        or completion_tokens <= 0
        or generation_latency_ms is None
        or generation_latency_ms <= 0
    ):
        return None
    return completion_tokens / (generation_latency_ms / 1000)


def benchmark_once(
    provider: str,
    endpoint: str,
    model: str,
    hardware_label: str,
    prompt: str,
    run_index: int,
    dry_run: bool,
    stream: bool,
    timeout_seconds: float,
) -> dict[str, Any]:
    started_at = time.perf_counter()
    status_code = None
    error = None
    error_name = None
    try:
        provider_result, status_code = call_provider(
            provider=provider,
            endpoint=endpoint,
            model=model,
            prompt=prompt,
            stream=stream,
            dry_run=dry_run,
            timeout_seconds=timeout_seconds,
            started_at=started_at,
        )
    except (OSError, urllib.error.URLError, TimeoutError, json.JSONDecodeError, ValueError) as exc:
        provider_result = {
            "output_text": "",
            "prompt_tokens": None,
            "completion_tokens": None,
            "total_tokens": None,
            "ttft_ms": None,
        }
        error = str(exc)
        error_name = error_type(exc)
        if isinstance(exc, urllib.error.HTTPError):
            status_code = exc.code

    total_latency_ms = (time.perf_counter() - started_at) * 1000
    ttft_ms = provider_result["ttft_ms"]
    generation_latency_ms = total_latency_ms - ttft_ms if ttft_ms is not None else total_latency_ms
    output_text = str(provider_result["output_text"])
    completion_tokens = provider_result["completion_tokens"]
    tps = tokens_per_second(completion_tokens, generation_latency_ms)
    success = error is None
    return {
        "timestamp": utc_timestamp(),
        "provider": provider,
        "endpoint": endpoint,
        "model": model,
        "hardware_label": hardware_label,
        "run_index": run_index,
        "latency_ms": round(total_latency_ms, 3),
        "total_latency_ms": round(total_latency_ms, 3),
        "ttft_ms": round(ttft_ms, 3) if ttft_ms is not None else None,
        "generation_latency_ms": round(generation_latency_ms, 3),
        "prompt_char_count": len(prompt),
        "output_char_count": len(output_text),
        "prompt_tokens": provider_result["prompt_tokens"],
        "completion_tokens": completion_tokens,
        "total_tokens": provider_result["total_tokens"],
        "tokens_per_second": round(tps, 3) if tps is not None else None,
        "success": success,
        "error": error,
        "error_type": error_name,
        "status_code": status_code,
        "timeout_seconds": timeout_seconds,
        "streaming": stream,
        "dry_run": dry_run,
    }


def percentile(values: list[float], percentile_value: float) -> float | None:
    if not values:
        return None
    if len(values) == 1:
        return round(values[0], 3)
    sorted_values = sorted(values)
    rank = (len(sorted_values) - 1) * percentile_value
    lower = int(rank)
    upper = min(lower + 1, len(sorted_values) - 1)
    weight = rank - lower
    value = sorted_values[lower] * (1 - weight) + sorted_values[upper] * weight
    return round(value, 3)


def mean(values: list[float]) -> float | None:
    return round(statistics.fmean(values), 3) if values else None


def build_summary(records: list[dict[str, Any]]) -> dict[str, Any]:
    successful = [record for record in records if record["success"]]
    total_latencies = [record["total_latency_ms"] for record in successful]
    ttfts = [record["ttft_ms"] for record in successful if record["ttft_ms"] is not None]
    tokens_per_second_values = [
        record["tokens_per_second"] for record in successful if record["tokens_per_second"] is not None
    ]
    return {
        "timestamp": utc_timestamp(),
        "run_count": len(records),
        "success_count": len(successful),
        "error_count": len(records) - len(successful),
        "p50_total_latency_ms": percentile(total_latencies, 0.50),
        "p95_total_latency_ms": percentile(total_latencies, 0.95),
        "p99_total_latency_ms": percentile(total_latencies, 0.99),
        "p50_ttft_ms": percentile(ttfts, 0.50),
        "p95_ttft_ms": percentile(ttfts, 0.95),
        "mean_tokens_per_second": mean(tokens_per_second_values),
    }


def append_jsonl(records: list[dict[str, Any]], output_path: Path) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("a", encoding="utf-8") as output_file:
        for record in records:
            output_file.write(json.dumps(record, sort_keys=True) + "\n")


def write_summary(summary: dict[str, Any], output_path: Path) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(summary, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Benchmark local or OpenAI-compatible LLM endpoints.")
    parser.add_argument("--provider", choices=["ollama", "openai-compatible"], default="ollama")
    parser.add_argument("--endpoint", default=None)
    parser.add_argument("--model", default=os.environ.get("OLLAMA_MODEL", "gemma3:4b"))
    parser.add_argument("--hardware-label", default="local-cpu")
    parser.add_argument("--prompt", default=DEFAULT_PROMPT)
    parser.add_argument("--runs", type=int, default=1)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT_PATH)
    parser.add_argument("--summary-output", type=Path, default=DEFAULT_SUMMARY_OUTPUT_PATH)
    parser.add_argument("--timeout-seconds", type=float, default=DEFAULT_TIMEOUT_SECONDS)
    parser.add_argument("--stream", action="store_true")
    parser.add_argument("--dry-run", action="store_true")
    return parser


def main() -> None:
    args = build_parser().parse_args()
    if args.runs < 1:
        raise SystemExit("--runs must be >= 1")
    if args.timeout_seconds <= 0:
        raise SystemExit("--timeout-seconds must be > 0")

    endpoint = normalize_endpoint(args.endpoint, args.provider)
    records = [
        benchmark_once(
            provider=args.provider,
            endpoint=endpoint,
            model=args.model,
            hardware_label=args.hardware_label,
            prompt=args.prompt,
            run_index=index + 1,
            dry_run=args.dry_run,
            stream=args.stream,
            timeout_seconds=args.timeout_seconds,
        )
        for index in range(args.runs)
    ]
    append_jsonl(records, args.output)
    summary = build_summary(records)
    if args.summary_output:
        write_summary(summary, args.summary_output)
    print(
        json.dumps(
            {
                "output": str(args.output),
                "records_written": len(records),
                "summary_output": str(args.summary_output) if args.summary_output else None,
                "summary": summary,
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
