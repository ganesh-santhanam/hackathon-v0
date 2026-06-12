# ROCm Fused Rerank Benchmark Report

- Generated: 2026-06-12T15:04:25.872862+00:00
- Device: cuda
- ROCm build: True
- Rows: 72
- Successful runs: 54
- Skipped runs: 18

## Best Latency

fp32 at batch 8, 10000 candidates, dim 384: 0.331 ms

## Results

| mode | batch | candidates | dim | status | latency_ms | candidates/s | speedup | peak_vram_gb | max_error | mean_error | top_k_overlap |
| --- | ---: | ---: | ---: | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| fp32 | 1 | 10000 | 384 | ok | 0.351 | 28462699.636 |  | 0.229 |  |  |  |
| fp16 | 1 | 10000 | 384 | ok | 0.364 | 27458043.748 | 0.965 | 0.228 | 0.000 | 0.000 | 1.000 |
| bf16 | 1 | 10000 | 384 | ok | 0.369 | 27082216.093 | 0.951 | 0.228 | 0.001 | 0.000 | 1.000 |
| fp8 | 1 | 10000 | 384 | skipped |  |  |  |  |  |  |  |
| fp32 | 8 | 10000 | 384 | ok | 0.331 | 241947667.983 |  | 0.233 |  |  |  |
| fp16 | 8 | 10000 | 384 | ok | 0.349 | 229353870.104 | 0.948 | 0.229 | 0.000 | 0.000 | 1.000 |
| bf16 | 8 | 10000 | 384 | ok | 0.556 | 143984865.416 | 0.595 | 0.229 | 0.001 | 0.000 | 1.000 |
| fp8 | 8 | 10000 | 384 | skipped |  |  |  |  |  |  |  |
| fp32 | 32 | 10000 | 384 | ok | 0.588 | 543856063.313 |  | 0.249 |  |  |  |
| fp16 | 32 | 10000 | 384 | ok | 0.836 | 382684490.529 | 0.704 | 0.242 | 0.000 | 0.000 | 1.000 |
| bf16 | 32 | 10000 | 384 | ok | 1.115 | 287078933.881 | 0.528 | 0.242 | 0.002 | 0.000 | 1.000 |
| fp8 | 32 | 10000 | 384 | skipped |  |  |  |  |  |  |  |
| fp32 | 1 | 100000 | 384 | ok | 0.457 | 218962098.800 |  | 0.495 |  |  |  |
| fp16 | 1 | 100000 | 384 | ok | 0.496 | 201417190.396 | 0.920 | 0.488 | 0.000 | 0.000 | 1.000 |
| bf16 | 1 | 100000 | 384 | ok | 0.641 | 156043239.966 | 0.713 | 0.488 | 0.001 | 0.000 | 1.000 |
| fp8 | 1 | 100000 | 384 | skipped |  |  |  |  |  |  |  |
| fp32 | 8 | 100000 | 384 | ok | 0.486 | 1646829277.981 |  | 0.541 |  |  |  |
| fp16 | 8 | 100000 | 384 | ok | 0.535 | 1494324302.069 | 0.907 | 0.494 | 0.000 | 0.000 | 1.000 |
| bf16 | 8 | 100000 | 384 | ok | 0.629 | 1271560944.278 | 0.772 | 0.494 | 0.002 | 0.000 | 0.988 |
| fp8 | 8 | 100000 | 384 | skipped |  |  |  |  |  |  |  |
| fp32 | 32 | 100000 | 384 | ok | 0.657 | 4867497571.262 |  | 0.697 |  |  |  |
| fp16 | 32 | 100000 | 384 | ok | 0.670 | 4777631222.809 | 0.982 | 0.628 | 0.000 | 0.000 | 1.000 |
| bf16 | 32 | 100000 | 384 | ok | 0.720 | 4446939723.981 | 0.914 | 0.628 | 0.002 | 0.000 | 0.994 |
| fp8 | 32 | 100000 | 384 | skipped |  |  |  |  |  |  |  |
| fp32 | 1 | 1000000 | 384 | ok | 2.294 | 435960840.877 |  | 3.161 |  |  |  |
| fp16 | 1 | 1000000 | 384 | ok | 2.241 | 446231156.564 | 1.024 | 3.092 | 0.000 | 0.000 | 1.000 |
| bf16 | 1 | 1000000 | 384 | ok | 2.341 | 427081593.864 | 0.980 | 3.092 | 0.001 | 0.000 | 1.000 |
| fp8 | 1 | 1000000 | 384 | skipped |  |  |  |  |  |  |  |
| fp32 | 8 | 1000000 | 384 | ok | 2.775 | 2883000260.194 |  | 3.594 |  |  |  |
| fp16 | 8 | 1000000 | 384 | ok | 2.711 | 2951265007.449 | 1.024 | 3.145 | 0.000 | 0.000 | 1.000 |
| bf16 | 8 | 1000000 | 384 | ok | 2.772 | 2886447489.715 | 1.001 | 3.145 | 0.002 | 0.000 | 0.988 |
| fp8 | 8 | 1000000 | 384 | skipped |  |  |  |  |  |  |  |
| fp32 | 32 | 1000000 | 384 | ok | 4.739 | 6751999333.834 |  | 5.139 |  |  |  |
| fp16 | 32 | 1000000 | 384 | ok | 4.615 | 6934357410.370 | 1.027 | 4.454 | 0.000 | 0.000 | 0.997 |
| bf16 | 32 | 1000000 | 384 | ok | 4.680 | 6837695945.187 | 1.013 | 4.454 | 0.002 | 0.000 | 0.991 |
| fp8 | 32 | 1000000 | 384 | skipped |  |  |  |  |  |  |  |
| fp32 | 1 | 10000 | 768 | ok | 0.367 | 27248029.830 |  | 0.377 |  |  |  |
| fp16 | 1 | 10000 | 768 | ok | 0.508 | 19686123.325 | 0.722 | 0.257 | 0.000 | 0.000 | 1.000 |
| bf16 | 1 | 10000 | 768 | ok | 1.037 | 9640579.936 | 0.354 | 0.257 | 0.001 | 0.000 | 1.000 |
| fp8 | 1 | 10000 | 768 | skipped |  |  |  |  |  |  |  |
| fp32 | 8 | 10000 | 768 | ok | 0.389 | 205732738.313 |  | 0.263 |  |  |  |
| fp16 | 8 | 10000 | 768 | ok | 0.489 | 163472155.846 | 0.795 | 0.258 | 0.000 | 0.000 | 1.000 |
| bf16 | 8 | 10000 | 768 | ok | 0.829 | 96532320.714 | 0.469 | 0.258 | 0.001 | 0.000 | 1.000 |
| fp8 | 8 | 10000 | 768 | skipped |  |  |  |  |  |  |  |
| fp32 | 32 | 10000 | 768 | ok | 0.484 | 661569851.516 |  | 0.279 |  |  |  |
| fp16 | 32 | 10000 | 768 | ok | 0.773 | 413763875.441 | 0.625 | 0.265 | 0.000 | 0.000 | 1.000 |
| bf16 | 32 | 10000 | 768 | ok | 1.111 | 287961419.648 | 0.435 | 0.265 | 0.001 | 0.000 | 0.997 |
| fp8 | 32 | 10000 | 768 | skipped |  |  |  |  |  |  |  |
| fp32 | 1 | 100000 | 768 | ok | 0.608 | 164462040.226 |  | 0.782 |  |  |  |
| fp16 | 1 | 100000 | 768 | ok | 0.619 | 161457758.353 | 0.982 | 0.775 | 0.000 | 0.000 | 1.000 |
| bf16 | 1 | 100000 | 768 | ok | 0.703 | 142275230.000 | 0.865 | 0.775 | 0.001 | 0.000 | 1.000 |
| fp8 | 1 | 100000 | 768 | skipped |  |  |  |  |  |  |  |
| fp32 | 8 | 100000 | 768 | ok | 0.658 | 1215334460.238 |  | 0.827 |  |  |  |
| fp16 | 8 | 100000 | 768 | ok | 0.648 | 1234560275.026 | 1.016 | 0.781 | 0.000 | 0.000 | 1.000 |
| bf16 | 8 | 100000 | 768 | ok | 0.737 | 1085173962.133 | 0.893 | 0.781 | 0.001 | 0.000 | 1.000 |
| fp8 | 8 | 100000 | 768 | skipped |  |  |  |  |  |  |  |
| fp32 | 32 | 100000 | 768 | ok | 0.830 | 3853420679.098 |  | 0.983 |  |  |  |
| fp16 | 32 | 100000 | 768 | ok | 0.801 | 3997471413.477 | 1.037 | 0.843 | 0.000 | 0.000 | 0.997 |
| bf16 | 32 | 100000 | 768 | ok | 0.823 | 3888691125.949 | 1.009 | 0.843 | 0.001 | 0.000 | 0.997 |
| fp8 | 32 | 100000 | 768 | skipped |  |  |  |  |  |  |  |
| fp32 | 1 | 1000000 | 768 | ok | 3.899 | 256478519.589 |  | 6.023 |  |  |  |
| fp16 | 1 | 1000000 | 768 | ok | 3.671 | 272410905.836 | 1.062 | 5.954 | 0.000 | 0.000 | 1.000 |
| bf16 | 1 | 1000000 | 768 | ok | 3.960 | 252522320.086 | 0.985 | 5.954 | 0.001 | 0.000 | 1.000 |
| fp8 | 1 | 1000000 | 768 | skipped |  |  |  |  |  |  |  |
| fp32 | 8 | 1000000 | 768 | ok | 4.415 | 1812123160.285 |  | 6.456 |  |  |  |
| fp16 | 8 | 1000000 | 768 | ok | 4.194 | 1907684303.216 | 1.053 | 6.007 | 0.000 | 0.000 | 1.000 |
| bf16 | 8 | 1000000 | 768 | ok | 4.274 | 1871835869.470 | 1.033 | 6.007 | 0.001 | 0.000 | 0.988 |
| fp8 | 8 | 1000000 | 768 | skipped |  |  |  |  |  |  |  |
| fp32 | 32 | 1000000 | 768 | ok | 6.474 | 4942814723.424 |  | 8.001 |  |  |  |
| fp16 | 32 | 1000000 | 768 | ok | 6.197 | 5163417295.806 | 1.045 | 6.600 | 0.000 | 0.000 | 1.000 |
| bf16 | 32 | 1000000 | 768 | ok | 6.272 | 5101797616.785 | 1.032 | 6.600 | 0.001 | 0.000 | 0.997 |
| fp8 | 32 | 1000000 | 768 | skipped |  |  |  |  |  |  |  |

## Skipped Modes

- fp8 batch=1 candidates=10000 dim=384: Native FP8 matmul is not exposed cleanly by this PyTorch build.
- fp8 batch=8 candidates=10000 dim=384: Native FP8 matmul is not exposed cleanly by this PyTorch build.
- fp8 batch=32 candidates=10000 dim=384: Native FP8 matmul is not exposed cleanly by this PyTorch build.
- fp8 batch=1 candidates=100000 dim=384: Native FP8 matmul is not exposed cleanly by this PyTorch build.
- fp8 batch=8 candidates=100000 dim=384: Native FP8 matmul is not exposed cleanly by this PyTorch build.
- fp8 batch=32 candidates=100000 dim=384: Native FP8 matmul is not exposed cleanly by this PyTorch build.
- fp8 batch=1 candidates=1000000 dim=384: Native FP8 matmul is not exposed cleanly by this PyTorch build.
- fp8 batch=8 candidates=1000000 dim=384: Native FP8 matmul is not exposed cleanly by this PyTorch build.
- fp8 batch=32 candidates=1000000 dim=384: Native FP8 matmul is not exposed cleanly by this PyTorch build.
- fp8 batch=1 candidates=10000 dim=768: Native FP8 matmul is not exposed cleanly by this PyTorch build.
- fp8 batch=8 candidates=10000 dim=768: Native FP8 matmul is not exposed cleanly by this PyTorch build.
- fp8 batch=32 candidates=10000 dim=768: Native FP8 matmul is not exposed cleanly by this PyTorch build.
- fp8 batch=1 candidates=100000 dim=768: Native FP8 matmul is not exposed cleanly by this PyTorch build.
- fp8 batch=8 candidates=100000 dim=768: Native FP8 matmul is not exposed cleanly by this PyTorch build.
- fp8 batch=32 candidates=100000 dim=768: Native FP8 matmul is not exposed cleanly by this PyTorch build.
- fp8 batch=1 candidates=1000000 dim=768: Native FP8 matmul is not exposed cleanly by this PyTorch build.
- fp8 batch=8 candidates=1000000 dim=768: Native FP8 matmul is not exposed cleanly by this PyTorch build.
- fp8 batch=32 candidates=1000000 dim=768: Native FP8 matmul is not exposed cleanly by this PyTorch build.
