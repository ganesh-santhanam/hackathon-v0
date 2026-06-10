#!/usr/bin/env python3
import argparse
import json
import sys
import time
from datetime import UTC, datetime
from pathlib import Path
from typing import Any


DEFAULT_MODEL_NAME = "google/gemma-3-4b-it"
DEFAULT_TRAIN_FILE = Path("data/lora/train.jsonl")
DEFAULT_EVAL_FILE = Path("data/lora/eval.jsonl")
DEFAULT_OUTPUT_DIR = Path("data/amd/lora")
DEFAULT_ADAPTER_DIR = DEFAULT_OUTPUT_DIR / "gemma3b_adapter"
DEFAULT_METRICS_PATH = DEFAULT_OUTPUT_DIR / "training_metrics.json"
DEFAULT_LOG_PATH = DEFAULT_OUTPUT_DIR / "training_log.txt"
DEFAULT_BASE_RESULTS_PATH = Path("data/evals/base_results.jsonl")
DEFAULT_LORA_RESULTS_PATH = Path("data/evals/lora_results.jsonl")


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


def write_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def append_log(path: Path, message: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as output_file:
        output_file.write(f"[{utc_timestamp()}] {message}\n")
    print(message)


def format_instruction_prompt(example: dict[str, Any]) -> str:
    return (
        "You are an industrial incident investigation assistant.\n"
        "Follow the instruction using only the provided incident context.\n\n"
        f"Instruction:\n{example.get('instruction', '')}\n\n"
        f"Input:\n{example.get('input', '')}\n\n"
        "Response:\n"
    )


def format_training_text(example: dict[str, Any], eos_token: str) -> str:
    return f"{format_instruction_prompt(example)}{example.get('output', '')}{eos_token}"


def fail_if_no_gpu(force_cpu: bool = False) -> None:
    if force_cpu:
        return
    try:
        import torch
    except ImportError as exc:
        raise SystemExit(
            "PyTorch is not installed. Run scripts/amd/setup_amd_lora.sh on AMD Cloud first."
        ) from exc
    if not torch.cuda.is_available():
        raise SystemExit(
            "No ROCm/GPU device is visible to PyTorch. Use --dry-run for local checks or run on AMD Cloud."
        )


def load_hf_stack():
    try:
        import torch
        from datasets import Dataset
        from peft import LoraConfig, PeftModel, get_peft_model
        from transformers import (
            AutoModelForCausalLM,
            AutoTokenizer,
            DataCollatorForLanguageModeling,
            Trainer,
            TrainingArguments,
        )
    except ImportError as exc:
        raise SystemExit(
            "Missing training dependencies. Run scripts/amd/setup_amd_lora.sh on AMD Cloud."
        ) from exc
    return {
        "torch": torch,
        "Dataset": Dataset,
        "LoraConfig": LoraConfig,
        "PeftModel": PeftModel,
        "get_peft_model": get_peft_model,
        "AutoModelForCausalLM": AutoModelForCausalLM,
        "AutoTokenizer": AutoTokenizer,
        "DataCollatorForLanguageModeling": DataCollatorForLanguageModeling,
        "Trainer": Trainer,
        "TrainingArguments": TrainingArguments,
    }


def dry_run(args: argparse.Namespace) -> dict[str, Any]:
    train_rows = load_jsonl(args.train_file)
    eval_rows = load_jsonl(args.eval_file)
    if args.eval_subset_size:
        eval_rows = eval_rows[: args.eval_subset_size]
    sample_prompt = format_instruction_prompt(train_rows[0]) if train_rows else ""
    metrics = {
        "timestamp": utc_timestamp(),
        "mode": "dry-run",
        "model_name": args.model_name,
        "train_file": str(args.train_file),
        "eval_file": str(args.eval_file),
        "train_examples": len(train_rows),
        "eval_examples": len(eval_rows),
        "sample_prompt_characters": len(sample_prompt),
        "adapter_dir": str(args.adapter_dir),
        "status": "ok",
    }
    write_json(args.metrics_path, metrics)
    append_log(args.log_path, "Dry-run completed without loading Hugging Face models.")
    return metrics


def train(args: argparse.Namespace) -> dict[str, Any]:
    fail_if_no_gpu(force_cpu=args.force_cpu)
    stack = load_hf_stack()
    torch = stack["torch"]

    train_rows = load_jsonl(args.train_file)
    eval_rows = load_jsonl(args.eval_file)
    if args.eval_subset_size:
        eval_rows = eval_rows[: args.eval_subset_size]
    if args.smoke_test:
        train_rows = train_rows[: min(len(train_rows), 8)]
        eval_rows = eval_rows[: min(len(eval_rows), 4)]
        args.max_steps = min(args.max_steps or 4, 4)

    if not train_rows:
        raise SystemExit(f"No training rows found in {args.train_file}")
    if not eval_rows:
        raise SystemExit(f"No eval rows found in {args.eval_file}")

    args.output_dir.mkdir(parents=True, exist_ok=True)
    args.log_path.write_text("", encoding="utf-8")
    append_log(args.log_path, f"Loading tokenizer: {args.model_name}")
    tokenizer = stack["AutoTokenizer"].from_pretrained(args.model_name)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    append_log(args.log_path, f"Loading base model: {args.model_name}")
    dtype = torch.bfloat16 if args.bf16 and torch.cuda.is_bf16_supported() else torch.float16
    model = stack["AutoModelForCausalLM"].from_pretrained(
        args.model_name,
        torch_dtype=dtype,
        device_map="auto",
    )
    model.config.use_cache = False
    if args.gradient_checkpointing:
        model.gradient_checkpointing_enable()

    lora_config = stack["LoraConfig"](
        r=args.lora_r,
        lora_alpha=args.lora_alpha,
        lora_dropout=args.lora_dropout,
        bias="none",
        task_type="CAUSAL_LM",
        target_modules=[
            "q_proj",
            "k_proj",
            "v_proj",
            "o_proj",
            "gate_proj",
            "up_proj",
            "down_proj",
        ],
    )
    model = stack["get_peft_model"](model, lora_config)
    model.print_trainable_parameters()

    eos_token = tokenizer.eos_token or ""
    train_dataset = stack["Dataset"].from_list(
        [{"text": format_training_text(row, eos_token)} for row in train_rows]
    )
    eval_dataset = stack["Dataset"].from_list(
        [{"text": format_training_text(row, eos_token)} for row in eval_rows]
    )

    def tokenize(batch: dict[str, list[str]]) -> dict[str, Any]:
        return tokenizer(
            batch["text"],
            truncation=True,
            max_length=args.max_length,
            padding=False,
        )

    train_dataset = train_dataset.map(tokenize, batched=True, remove_columns=["text"])
    eval_dataset = eval_dataset.map(tokenize, batched=True, remove_columns=["text"])
    collator = stack["DataCollatorForLanguageModeling"](tokenizer=tokenizer, mlm=False)

    training_args = stack["TrainingArguments"](
        output_dir=str(args.output_dir / "trainer_checkpoints"),
        num_train_epochs=args.num_epochs,
        max_steps=args.max_steps,
        per_device_train_batch_size=args.batch_size,
        per_device_eval_batch_size=args.batch_size,
        gradient_accumulation_steps=args.gradient_accumulation_steps,
        learning_rate=args.learning_rate,
        logging_steps=args.logging_steps,
        eval_strategy="steps",
        eval_steps=args.eval_steps,
        save_steps=args.save_steps,
        save_total_limit=1,
        bf16=bool(args.bf16 and torch.cuda.is_bf16_supported()),
        fp16=not bool(args.bf16 and torch.cuda.is_bf16_supported()),
        report_to=[],
        gradient_checkpointing=args.gradient_checkpointing,
    )

    trainer = stack["Trainer"](
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=eval_dataset,
        data_collator=collator,
    )

    started_at = time.perf_counter()
    append_log(args.log_path, "Starting LoRA training.")
    train_result = trainer.train()
    eval_metrics = trainer.evaluate()
    elapsed_seconds = round(time.perf_counter() - started_at, 3)

    append_log(args.log_path, f"Saving LoRA adapter to {args.adapter_dir}")
    args.adapter_dir.mkdir(parents=True, exist_ok=True)
    trainer.model.save_pretrained(args.adapter_dir)
    tokenizer.save_pretrained(args.adapter_dir)

    metrics = {
        "timestamp": utc_timestamp(),
        "mode": "train",
        "model_name": args.model_name,
        "train_file": str(args.train_file),
        "eval_file": str(args.eval_file),
        "adapter_dir": str(args.adapter_dir),
        "train_examples": len(train_rows),
        "eval_examples": len(eval_rows),
        "elapsed_seconds": elapsed_seconds,
        "bf16_requested": args.bf16,
        "bf16_used": bool(args.bf16 and torch.cuda.is_bf16_supported()),
        "gradient_checkpointing": args.gradient_checkpointing,
        "train_metrics": train_result.metrics,
        "eval_metrics": eval_metrics,
    }
    write_json(args.metrics_path, metrics)
    append_log(args.log_path, "Training complete.")
    return metrics


def load_generation_model(args: argparse.Namespace, use_lora: bool):
    stack = load_hf_stack()
    torch = stack["torch"]
    tokenizer = stack["AutoTokenizer"].from_pretrained(args.model_name)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    dtype = torch.bfloat16 if args.bf16 and torch.cuda.is_available() else torch.float16
    model = stack["AutoModelForCausalLM"].from_pretrained(
        args.model_name,
        torch_dtype=dtype,
        device_map="auto" if torch.cuda.is_available() else None,
    )
    if use_lora:
        model = stack["PeftModel"].from_pretrained(model, args.adapter_dir)
    model.eval()
    return stack, tokenizer, model


def generate_text(args: argparse.Namespace, prompt: str, use_lora: bool) -> str:
    stack, tokenizer, model = load_generation_model(args, use_lora=use_lora)
    torch = stack["torch"]
    inputs = tokenizer(prompt, return_tensors="pt", truncation=True, max_length=args.max_length)
    if torch.cuda.is_available():
        inputs = {key: value.to(model.device) for key, value in inputs.items()}
    with torch.no_grad():
        output_ids = model.generate(
            **inputs,
            max_new_tokens=args.max_new_tokens,
            do_sample=False,
            pad_token_id=tokenizer.eos_token_id,
        )
    decoded = tokenizer.decode(output_ids[0], skip_special_tokens=True)
    return decoded[len(prompt) :].strip() if decoded.startswith(prompt) else decoded.strip()


def dry_run_generation_response(example: dict[str, Any], candidate_name: str) -> str:
    expected_modes = example.get("expected_failure_modes") or [example.get("expected_failure_mode")]
    expected_mode = next((mode for mode in expected_modes if mode), "unknown machine failure")
    expected_severity = example.get("expected_severity") or "SEV3"
    suffix = "using the base model dry-run path"
    if candidate_name == "lora":
        suffix = "using the LoRA adapter dry-run path"
    return (
        f"Root Cause: {expected_mode}.\n"
        f"Evidence: Retrieved incident evidence supports {expected_mode}.\n"
        "Actions: Inspect the affected subsystem, validate telemetry, and document closeout.\n"
        f"Severity Reasoning: Suggested severity is {expected_severity}.\n"
        f"Limitations: Generated {suffix}."
    )


def generate_for_judge(args: argparse.Namespace) -> dict[str, Any]:
    fail_if_no_gpu(force_cpu=args.force_cpu or args.dry_run)
    examples = load_jsonl(args.judge_dataset)
    if args.eval_subset_size:
        examples = examples[: args.eval_subset_size]
    if args.smoke_test:
        examples = examples[: min(len(examples), 2)]

    base_records = []
    lora_records = []
    for example in examples:
        prompt = example["prompt"]
        started_at = time.perf_counter()
        base_response = (
            dry_run_generation_response(example, "base")
            if args.dry_run
            else generate_text(args, prompt, use_lora=False)
        )
        base_latency_ms = int((time.perf_counter() - started_at) * 1000)

        started_at = time.perf_counter()
        lora_response = (
            dry_run_generation_response(example, "lora")
            if args.dry_run
            else generate_text(args, prompt, use_lora=True)
        )
        lora_latency_ms = int((time.perf_counter() - started_at) * 1000)

        common_metadata = {
            "source_document_id": example.get("source_document_id"),
            "expected_failure_mode": example.get("expected_failure_mode"),
            "expected_severity": example.get("expected_severity"),
            "dry_run": args.dry_run,
        }
        base_records.append(
            {
                "timestamp": utc_timestamp(),
                "eval_id": example["eval_id"],
                "candidate_name": "base",
                "provider": "huggingface-transformers",
                "endpoint": None,
                "model": args.model_name,
                "prompt": prompt,
                "response_text": base_response,
                "metadata": {
                    **common_metadata,
                    "latency_ms": base_latency_ms,
                    "success": True,
                    "error": None,
                },
            }
        )
        lora_records.append(
            {
                "timestamp": utc_timestamp(),
                "eval_id": example["eval_id"],
                "candidate_name": "lora",
                "provider": "huggingface-transformers",
                "endpoint": None,
                "model": f"{args.model_name}+{args.adapter_dir}",
                "prompt": prompt,
                "response_text": lora_response,
                "metadata": {
                    **common_metadata,
                    "latency_ms": lora_latency_ms,
                    "success": True,
                    "error": None,
                },
            }
        )

    write_jsonl(args.base_results_output, base_records)
    write_jsonl(args.lora_results_output, lora_records)
    return {
        "base_results": str(args.base_results_output),
        "lora_results": str(args.lora_results_output),
        "examples": len(examples),
        "dry_run": args.dry_run,
    }


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Train and evaluate Gemma LoRA on AMD Cloud.")
    parser.add_argument("--mode", choices=["train", "generate"], default="train")
    parser.add_argument("--model-name", default=DEFAULT_MODEL_NAME)
    parser.add_argument("--train-file", type=Path, default=DEFAULT_TRAIN_FILE)
    parser.add_argument("--eval-file", type=Path, default=DEFAULT_EVAL_FILE)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--adapter-dir", type=Path, default=DEFAULT_ADAPTER_DIR)
    parser.add_argument("--metrics-path", type=Path, default=DEFAULT_METRICS_PATH)
    parser.add_argument("--log-path", type=Path, default=DEFAULT_LOG_PATH)
    parser.add_argument("--max-steps", type=int, default=100)
    parser.add_argument("--num-epochs", type=float, default=1.0)
    parser.add_argument("--learning-rate", type=float, default=2e-4)
    parser.add_argument("--batch-size", type=int, default=1)
    parser.add_argument("--gradient-accumulation-steps", type=int, default=8)
    parser.add_argument("--bf16", action="store_true")
    parser.add_argument("--gradient-checkpointing", action="store_true", default=True)
    parser.add_argument("--no-gradient-checkpointing", dest="gradient_checkpointing", action="store_false")
    parser.add_argument("--eval-subset-size", type=int, default=64)
    parser.add_argument("--max-length", type=int, default=1024)
    parser.add_argument("--max-new-tokens", type=int, default=256)
    parser.add_argument("--logging-steps", type=int, default=5)
    parser.add_argument("--eval-steps", type=int, default=25)
    parser.add_argument("--save-steps", type=int, default=50)
    parser.add_argument("--lora-r", type=int, default=16)
    parser.add_argument("--lora-alpha", type=int, default=32)
    parser.add_argument("--lora-dropout", type=float, default=0.05)
    parser.add_argument("--smoke-test", action="store_true")
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--force-cpu", action="store_true")
    parser.add_argument("--judge-dataset", type=Path, default=Path("data/evals/eval_dataset.jsonl"))
    parser.add_argument("--base-results-output", type=Path, default=DEFAULT_BASE_RESULTS_PATH)
    parser.add_argument("--lora-results-output", type=Path, default=DEFAULT_LORA_RESULTS_PATH)
    return parser


def validate_args(args: argparse.Namespace) -> None:
    if args.max_steps is not None and args.max_steps < 0:
        raise SystemExit("--max-steps must be >= 0")
    if args.num_epochs <= 0:
        raise SystemExit("--num-epochs must be > 0")
    if args.batch_size < 1:
        raise SystemExit("--batch-size must be >= 1")
    if args.gradient_accumulation_steps < 1:
        raise SystemExit("--gradient-accumulation-steps must be >= 1")
    if args.mode == "train" and not args.dry_run:
        if not args.train_file.exists():
            raise SystemExit(f"Missing train file: {args.train_file}")
        if not args.eval_file.exists():
            raise SystemExit(f"Missing eval file: {args.eval_file}")
    if args.mode == "generate" and not args.judge_dataset.exists():
        raise SystemExit(f"Missing judge dataset: {args.judge_dataset}")


def main() -> int:
    args = build_parser().parse_args()
    validate_args(args)
    if args.dry_run:
        result = dry_run(args) if args.mode == "train" else generate_for_judge(args)
    elif args.mode == "train":
        result = train(args)
    else:
        result = generate_for_judge(args)
    print(json.dumps(result, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except KeyboardInterrupt:
        print("Interrupted.", file=sys.stderr)
        raise SystemExit(130)
