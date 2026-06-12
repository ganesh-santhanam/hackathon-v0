# ROCm Fused Rerank Benchmark Report

- Generated: 2026-06-12T14:00:03.777587+00:00
- Device: cuda
- ROCm build: True
- Rows: 90
- Successful runs: 54
- Skipped runs: 36

## Best Latency

fp32 at batch 8, 10000 candidates, dim 384: 0.329 ms

## Results

| mode | batch | candidates | dim | status | latency_ms | candidates/s | speedup | peak_vram_gb | max_error | mean_error | top_k_overlap |
| --- | ---: | ---: | ---: | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| fp32 | 1 | 10000 | 384 | ok | 0.348 | 28748685.020 |  | 0.229 |  |  |  |
| fp16 | 1 | 10000 | 384 | ok | 0.381 | 26219945.173 | 0.912 | 0.228 | 0.000 | 0.000 | 1.000 |
| bf16 | 1 | 10000 | 384 | ok | 0.381 | 26225176.428 | 0.912 | 0.228 | 0.001 | 0.000 | 1.000 |
| fp8 | 1 | 10000 | 384 | skipped |  |  |  |  |  |  |  |
| tf32 | 1 | 10000 | 384 | skipped |  |  |  |  |  |  |  |
| fp32 | 8 | 10000 | 384 | ok | 0.329 | 243167029.583 |  | 0.233 |  |  |  |
| fp16 | 8 | 10000 | 384 | ok | 0.345 | 231765140.854 | 0.953 | 0.229 | 0.000 | 0.000 | 1.000 |
| bf16 | 8 | 10000 | 384 | ok | 0.492 | 162523669.194 | 0.668 | 0.229 | 0.001 | 0.000 | 1.000 |
| fp8 | 8 | 10000 | 384 | skipped |  |  |  |  |  |  |  |
| tf32 | 8 | 10000 | 384 | skipped |  |  |  |  |  |  |  |
| fp32 | 32 | 10000 | 384 | ok | 0.643 | 497870805.353 |  | 0.249 |  |  |  |
| fp16 | 32 | 10000 | 384 | ok | 1.017 | 314598356.076 | 0.632 | 0.242 | 0.000 | 0.000 | 1.000 |
| bf16 | 32 | 10000 | 384 | ok | 1.099 | 291219900.980 | 0.585 | 0.242 | 0.002 | 0.000 | 1.000 |
| fp8 | 32 | 10000 | 384 | skipped |  |  |  |  |  |  |  |
| tf32 | 32 | 10000 | 384 | skipped |  |  |  |  |  |  |  |
| fp32 | 1 | 100000 | 384 | ok | 0.450 | 222193571.444 |  | 0.495 |  |  |  |
| fp16 | 1 | 100000 | 384 | ok | 0.501 | 199679710.190 | 0.899 | 0.488 | 0.000 | 0.000 | 1.000 |
| bf16 | 1 | 100000 | 384 | ok | 0.643 | 155435190.035 | 0.700 | 0.488 | 0.001 | 0.000 | 1.000 |
| fp8 | 1 | 100000 | 384 | skipped |  |  |  |  |  |  |  |
| tf32 | 1 | 100000 | 384 | skipped |  |  |  |  |  |  |  |
| fp32 | 8 | 100000 | 384 | ok | 0.482 | 1658873380.214 |  | 0.541 |  |  |  |
| fp16 | 8 | 100000 | 384 | ok | 0.555 | 1442296457.689 | 0.869 | 0.494 | 0.000 | 0.000 | 1.000 |
| bf16 | 8 | 100000 | 384 | ok | 0.913 | 876540373.091 | 0.528 | 0.494 | 0.002 | 0.000 | 0.988 |
| fp8 | 8 | 100000 | 384 | skipped |  |  |  |  |  |  |  |
| tf32 | 8 | 100000 | 384 | skipped |  |  |  |  |  |  |  |
| fp32 | 32 | 100000 | 384 | ok | 0.659 | 4852771376.622 |  | 0.697 |  |  |  |
| fp16 | 32 | 100000 | 384 | ok | 0.673 | 4752329644.619 | 0.979 | 0.628 | 0.000 | 0.000 | 1.000 |
| bf16 | 32 | 100000 | 384 | ok | 0.784 | 4079587694.684 | 0.841 | 0.628 | 0.002 | 0.000 | 0.994 |
| fp8 | 32 | 100000 | 384 | skipped |  |  |  |  |  |  |  |
| tf32 | 32 | 100000 | 384 | skipped |  |  |  |  |  |  |  |
| fp32 | 1 | 1000000 | 384 | ok | 2.280 | 438639971.451 |  | 3.161 |  |  |  |
| fp16 | 1 | 1000000 | 384 | ok | 2.304 | 434074128.096 | 0.990 | 3.092 | 0.000 | 0.000 | 1.000 |
| bf16 | 1 | 1000000 | 384 | ok | 2.341 | 427125552.772 | 0.974 | 3.092 | 0.001 | 0.000 | 1.000 |
| fp8 | 1 | 1000000 | 384 | skipped |  |  |  |  |  |  |  |
| tf32 | 1 | 1000000 | 384 | skipped |  |  |  |  |  |  |  |
| fp32 | 8 | 1000000 | 384 | ok | 2.730 | 2930669176.235 |  | 3.594 |  |  |  |
| fp16 | 8 | 1000000 | 384 | ok | 2.698 | 2964975858.303 | 1.012 | 3.145 | 0.000 | 0.000 | 1.000 |
| bf16 | 8 | 1000000 | 384 | ok | 2.751 | 2907901289.033 | 0.992 | 3.145 | 0.002 | 0.000 | 0.988 |
| fp8 | 8 | 1000000 | 384 | skipped |  |  |  |  |  |  |  |
| tf32 | 8 | 1000000 | 384 | skipped |  |  |  |  |  |  |  |
| fp32 | 32 | 1000000 | 384 | ok | 4.710 | 6794629377.657 |  | 5.139 |  |  |  |
| fp16 | 32 | 1000000 | 384 | ok | 4.670 | 6851944705.771 | 1.008 | 4.454 | 0.000 | 0.000 | 0.997 |
| bf16 | 32 | 1000000 | 384 | ok | 4.730 | 6765304820.651 | 0.996 | 4.454 | 0.002 | 0.000 | 0.991 |
| fp8 | 32 | 1000000 | 384 | skipped |  |  |  |  |  |  |  |
| tf32 | 32 | 1000000 | 384 | skipped |  |  |  |  |  |  |  |
| fp32 | 1 | 10000 | 768 | ok | 0.375 | 26632437.756 |  | 0.377 |  |  |  |
| fp16 | 1 | 10000 | 768 | ok | 0.513 | 19481026.474 | 0.731 | 0.257 | 0.000 | 0.000 | 1.000 |
| bf16 | 1 | 10000 | 768 | ok | 0.730 | 13705877.436 | 0.515 | 0.257 | 0.001 | 0.000 | 1.000 |
| fp8 | 1 | 10000 | 768 | skipped |  |  |  |  |  |  |  |
| tf32 | 1 | 10000 | 768 | skipped |  |  |  |  |  |  |  |
| fp32 | 8 | 10000 | 768 | ok | 0.415 | 192626261.943 |  | 0.263 |  |  |  |
| fp16 | 8 | 10000 | 768 | ok | 0.470 | 170361754.328 | 0.884 | 0.258 | 0.000 | 0.000 | 1.000 |
| bf16 | 8 | 10000 | 768 | ok | 0.635 | 125945772.044 | 0.654 | 0.258 | 0.001 | 0.000 | 1.000 |
| fp8 | 8 | 10000 | 768 | skipped |  |  |  |  |  |  |  |
| tf32 | 8 | 10000 | 768 | skipped |  |  |  |  |  |  |  |
| fp32 | 32 | 10000 | 768 | ok | 0.467 | 685358435.103 |  | 0.279 |  |  |  |
| fp16 | 32 | 10000 | 768 | ok | 0.642 | 498304977.811 | 0.727 | 0.265 | 0.000 | 0.000 | 1.000 |
| bf16 | 32 | 10000 | 768 | ok | 1.010 | 316898195.730 | 0.462 | 0.265 | 0.001 | 0.000 | 0.997 |
| fp8 | 32 | 10000 | 768 | skipped |  |  |  |  |  |  |  |
| tf32 | 32 | 10000 | 768 | skipped |  |  |  |  |  |  |  |
| fp32 | 1 | 100000 | 768 | ok | 0.620 | 161179312.262 |  | 0.782 |  |  |  |
| fp16 | 1 | 100000 | 768 | ok | 0.624 | 160346354.347 | 0.995 | 0.775 | 0.000 | 0.000 | 1.000 |
| bf16 | 1 | 100000 | 768 | ok | 0.729 | 137130005.056 | 0.851 | 0.775 | 0.001 | 0.000 | 1.000 |
| fp8 | 1 | 100000 | 768 | skipped |  |  |  |  |  |  |  |
| tf32 | 1 | 100000 | 768 | skipped |  |  |  |  |  |  |  |
| fp32 | 8 | 100000 | 768 | ok | 0.663 | 1206018074.895 |  | 0.827 |  |  |  |
| fp16 | 8 | 100000 | 768 | ok | 0.671 | 1192524017.230 | 0.989 | 0.781 | 0.000 | 0.000 | 1.000 |
| bf16 | 8 | 100000 | 768 | ok | 0.944 | 847594060.728 | 0.703 | 0.781 | 0.001 | 0.000 | 1.000 |
| fp8 | 8 | 100000 | 768 | skipped |  |  |  |  |  |  |  |
| tf32 | 8 | 100000 | 768 | skipped |  |  |  |  |  |  |  |
| fp32 | 32 | 100000 | 768 | ok | 0.818 | 3912296135.859 |  | 0.983 |  |  |  |
| fp16 | 32 | 100000 | 768 | ok | 0.809 | 3954581625.787 | 1.011 | 0.843 | 0.000 | 0.000 | 0.997 |
| bf16 | 32 | 100000 | 768 | ok | 0.852 | 3757213475.525 | 0.960 | 0.843 | 0.001 | 0.000 | 0.997 |
| fp8 | 32 | 100000 | 768 | skipped |  |  |  |  |  |  |  |
| tf32 | 32 | 100000 | 768 | skipped |  |  |  |  |  |  |  |
| fp32 | 1 | 1000000 | 768 | ok | 4.021 | 248713342.863 |  | 6.023 |  |  |  |
| fp16 | 1 | 1000000 | 768 | ok | 3.669 | 272558957.887 | 1.096 | 5.954 | 0.000 | 0.000 | 1.000 |
| bf16 | 1 | 1000000 | 768 | ok | 3.905 | 256094670.006 | 1.030 | 5.954 | 0.001 | 0.000 | 1.000 |
| fp8 | 1 | 1000000 | 768 | skipped |  |  |  |  |  |  |  |
| tf32 | 1 | 1000000 | 768 | skipped |  |  |  |  |  |  |  |
| fp32 | 8 | 1000000 | 768 | ok | 4.433 | 1804605856.545 |  | 6.456 |  |  |  |
| fp16 | 8 | 1000000 | 768 | ok | 4.227 | 1892660144.614 | 1.049 | 6.007 | 0.000 | 0.000 | 1.000 |
| bf16 | 8 | 1000000 | 768 | ok | 4.256 | 1879903757.496 | 1.042 | 6.007 | 0.001 | 0.000 | 0.988 |
| fp8 | 8 | 1000000 | 768 | skipped |  |  |  |  |  |  |  |
| tf32 | 8 | 1000000 | 768 | skipped |  |  |  |  |  |  |  |
| fp32 | 32 | 1000000 | 768 | ok | 6.511 | 4915114425.318 |  | 8.001 |  |  |  |
| fp16 | 32 | 1000000 | 768 | ok | 6.173 | 5183845885.925 | 1.055 | 6.600 | 0.000 | 0.000 | 1.000 |
| bf16 | 32 | 1000000 | 768 | ok | 6.483 | 4936003218.985 | 1.004 | 6.600 | 0.001 | 0.000 | 0.997 |
| fp8 | 32 | 1000000 | 768 | skipped |  |  |  |  |  |  |  |
| tf32 | 32 | 1000000 | 768 | skipped |  |  |  |  |  |  |  |

## Skipped Modes

- fp8 batch=1 candidates=10000 dim=384: Native FP8 matmul is not exposed cleanly by this PyTorch build.
- tf32 batch=1 candidates=10000 dim=384: TF32 is not cleanly exposed for this ROCm build.
- fp8 batch=8 candidates=10000 dim=384: Native FP8 matmul is not exposed cleanly by this PyTorch build.
- tf32 batch=8 candidates=10000 dim=384: TF32 is not cleanly exposed for this ROCm build.
- fp8 batch=32 candidates=10000 dim=384: Native FP8 matmul is not exposed cleanly by this PyTorch build.
- tf32 batch=32 candidates=10000 dim=384: TF32 is not cleanly exposed for this ROCm build.
- fp8 batch=1 candidates=100000 dim=384: Native FP8 matmul is not exposed cleanly by this PyTorch build.
- tf32 batch=1 candidates=100000 dim=384: TF32 is not cleanly exposed for this ROCm build.
- fp8 batch=8 candidates=100000 dim=384: Native FP8 matmul is not exposed cleanly by this PyTorch build.
- tf32 batch=8 candidates=100000 dim=384: TF32 is not cleanly exposed for this ROCm build.
- fp8 batch=32 candidates=100000 dim=384: Native FP8 matmul is not exposed cleanly by this PyTorch build.
- tf32 batch=32 candidates=100000 dim=384: TF32 is not cleanly exposed for this ROCm build.
- fp8 batch=1 candidates=1000000 dim=384: Native FP8 matmul is not exposed cleanly by this PyTorch build.
- tf32 batch=1 candidates=1000000 dim=384: TF32 is not cleanly exposed for this ROCm build.
- fp8 batch=8 candidates=1000000 dim=384: Native FP8 matmul is not exposed cleanly by this PyTorch build.
- tf32 batch=8 candidates=1000000 dim=384: TF32 is not cleanly exposed for this ROCm build.
- fp8 batch=32 candidates=1000000 dim=384: Native FP8 matmul is not exposed cleanly by this PyTorch build.
- tf32 batch=32 candidates=1000000 dim=384: TF32 is not cleanly exposed for this ROCm build.
- fp8 batch=1 candidates=10000 dim=768: Native FP8 matmul is not exposed cleanly by this PyTorch build.
- tf32 batch=1 candidates=10000 dim=768: TF32 is not cleanly exposed for this ROCm build.
- fp8 batch=8 candidates=10000 dim=768: Native FP8 matmul is not exposed cleanly by this PyTorch build.
- tf32 batch=8 candidates=10000 dim=768: TF32 is not cleanly exposed for this ROCm build.
- fp8 batch=32 candidates=10000 dim=768: Native FP8 matmul is not exposed cleanly by this PyTorch build.
- tf32 batch=32 candidates=10000 dim=768: TF32 is not cleanly exposed for this ROCm build.
- fp8 batch=1 candidates=100000 dim=768: Native FP8 matmul is not exposed cleanly by this PyTorch build.
- tf32 batch=1 candidates=100000 dim=768: TF32 is not cleanly exposed for this ROCm build.
- fp8 batch=8 candidates=100000 dim=768: Native FP8 matmul is not exposed cleanly by this PyTorch build.
- tf32 batch=8 candidates=100000 dim=768: TF32 is not cleanly exposed for this ROCm build.
- fp8 batch=32 candidates=100000 dim=768: Native FP8 matmul is not exposed cleanly by this PyTorch build.
- tf32 batch=32 candidates=100000 dim=768: TF32 is not cleanly exposed for this ROCm build.
- fp8 batch=1 candidates=1000000 dim=768: Native FP8 matmul is not exposed cleanly by this PyTorch build.
- tf32 batch=1 candidates=1000000 dim=768: TF32 is not cleanly exposed for this ROCm build.
- fp8 batch=8 candidates=1000000 dim=768: Native FP8 matmul is not exposed cleanly by this PyTorch build.
- tf32 batch=8 candidates=1000000 dim=768: TF32 is not cleanly exposed for this ROCm build.
- fp8 batch=32 candidates=1000000 dim=768: Native FP8 matmul is not exposed cleanly by this PyTorch build.
- tf32 batch=32 candidates=1000000 dim=768: TF32 is not cleanly exposed for this ROCm build.
