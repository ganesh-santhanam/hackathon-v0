# ROCm Fused Rerank Benchmark Report

- Generated: 2026-06-12T13:35:18.828904+00:00
- Device: cpu
- ROCm build: False
- Rows: 5
- Successful runs: 1
- Skipped runs: 4

## Best Latency

fp32 at batch 1, 512 candidates, dim 384: 21.442 ms

## Results

| mode | batch | candidates | dim | status | latency_ms | candidates/s | speedup | peak_vram_gb | max_error | mean_error | top_k_overlap |
| --- | ---: | ---: | ---: | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| fp32 | 1 | 512 | 384 | ok | 21.442 | 23878.515 |  |  |  |  |  |
| fp16 | 1 | 512 | 384 | skipped |  |  |  |  |  |  |  |
| bf16 | 1 | 512 | 384 | skipped |  |  |  |  |  |  |  |
| fp8 | 1 | 512 | 384 | skipped |  |  |  |  |  |  |  |
| tf32 | 1 | 512 | 384 | skipped |  |  |  |  |  |  |  |

## Skipped Modes

- fp16 batch=1 candidates=512 dim=384: FP16 benchmark is skipped on CPU fallback.
- bf16 batch=1 candidates=512 dim=384: BF16 benchmark is skipped on CPU fallback.
- fp8 batch=1 candidates=512 dim=384: FP8 mode requires GPU support.
- tf32 batch=1 candidates=512 dim=384: TF32 is only relevant on GPU.
