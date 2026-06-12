# ROCm Kernel Comparison Report

- Generated: 2026-06-12T15:05:45.510730+00:00
- Torch: 2.8.0+gitb2fb688
- HIP: 7.0.51831-a3e329ad8
- Device: 
- Workload: batch=32, candidates=1000000, dim=768, top_k=10

## Results

| implementation | precision | supported | latency_ms | candidates/s | speedup | peak_vram_gb | top_k_overlap | max_error | mean_error | complexity |
| --- | --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | --- |
| pytorch_eager | fp32 | yes | 6.445 | 4964748705.303 | 1.000 | 8.091 | 1.000 | 0.000 | 0.000 | low |
| pytorch_eager | fp16 | yes | 6.100 | 5246188035.915 | 1.057 | 6.601 | 1.000 | 0.000 | 0.000 | low |
| pytorch_eager | bf16 | yes | 6.405 | 4995749717.322 | 1.006 | 6.601 | 0.997 | 0.001 | 0.000 | low |
| rocblas_precomputed_index | fp32 | yes | 3.393 | 9431951908.345 | 1.900 | 7.637 | 1.000 | 0.000 | 0.000 | medium |
| rocblas_precomputed_index | fp16 | yes | 2.882 | 11102298758.155 | 2.236 | 9.030 | 1.000 | 0.000 | 0.000 | medium |
| rocblas_precomputed_index | bf16 | yes | 2.874 | 11135028354.435 | 2.243 | 9.030 | 0.997 | 0.001 | 0.000 | medium |
| rocblas_plus_triton_score | fp32 | yes | 1.566 | 20427992057.487 | 4.115 | 6.445 | 1.000 | 0.000 | 0.000 | high |
| rocblas_plus_triton_score | fp16 | yes | 1.053 | 30399265110.749 | 6.123 | 7.897 | 1.000 | 0.000 | 0.000 | high |
| rocblas_plus_triton_score | bf16 | yes | 1.039 | 30802253299.700 | 6.204 | 7.897 | 0.997 | 0.001 | 0.000 | high |
| torch_scaled_mm_fp8 | fp8 | yes | 2.557 | 12513892353.848 | 2.521 | 9.805 | 0.956 | 0.008 | 0.001 | high |
| torch_compile_inductor | fp32 | yes | 5.314 | 6021988881.998 | 1.213 | 13.086 | 1.000 | 0.000 | 0.000 | medium |
| torch_compile_inductor | fp16 | yes | 4.214 | 7593812657.105 | 1.530 | 11.655 | 1.000 | 0.000 | 0.000 | medium |
| torch_compile_inductor | bf16 | yes | 4.265 | 7502206135.491 | 1.511 | 11.655 | 1.000 | 0.001 | 0.000 | medium |
| rocm_graph_capture | fp32 | yes | 6.561 | 4877524580.361 | 0.982 | 3.635 | 1.000 | 0.000 | 0.000 | medium |
| rocm_graph_capture | fp16 | yes | 6.344 | 5044165631.416 | 1.016 | 3.635 | 1.000 | 0.000 | 0.000 | medium |
| rocm_graph_capture | bf16 | yes | 6.333 | 5052790638.825 | 1.018 | 3.635 | 0.997 | 0.001 | 0.000 | medium |
| aiter_full_rerank | mixed | no |  |  |  |  |  |  |  | high |
| composable_kernel_full_rerank | mixed | no |  |  |  |  |  |  |  | very_high |

## Unsupported Or Limited Paths

- pytorch_eager (fp32): Baseline PyTorch eager implementation.
- rocblas_precomputed_index (fp32): Uses PyTorch/rocBLAS GEMM with static candidate embeddings pre-normalized outside online latency.
- rocblas_precomputed_index (fp16): Uses PyTorch/rocBLAS GEMM with static candidate embeddings pre-normalized outside online latency.
- rocblas_precomputed_index (bf16): Uses PyTorch/rocBLAS GEMM with static candidate embeddings pre-normalized outside online latency.
- rocblas_plus_triton_score (fp32): rocBLAS GEMM plus a custom Triton kernel for telemetry penalty and score combine; top-k remains PyTorch.
- rocblas_plus_triton_score (fp16): rocBLAS GEMM plus a custom Triton kernel for telemetry penalty and score combine; top-k remains PyTorch.
- rocblas_plus_triton_score (bf16): rocBLAS GEMM plus a custom Triton kernel for telemetry penalty and score combine; top-k remains PyTorch.
- torch_scaled_mm_fp8 (fp8): Uses torch._scaled_mm with float8_e4m3fnuz for the similarity GEMM; quantization and non-GEMM rerank work are included.
- torch_compile_inductor (fp32): torch.compile/Inductor over the eager PyTorch graph.
- torch_compile_inductor (fp16): torch.compile/Inductor over the eager PyTorch graph.
- torch_compile_inductor (bf16): torch.compile/Inductor over the eager PyTorch graph.
- rocm_graph_capture (fp32): Measured in an isolated subprocess because ROCm graph capture reserves a private memory pool.
- rocm_graph_capture (fp16): Measured in an isolated subprocess because ROCm graph capture reserves a private memory pool.
- rocm_graph_capture (bf16): Measured in an isolated subprocess because ROCm graph capture reserves a private memory pool.
- aiter_full_rerank (mixed): AITER is installed, but no exposed operator implements normalize + similarity + telemetry penalty + weighted rerank + top-k. A separate hipb_mm probe segfaulted on this image, so it was not used in the benchmark.
- composable_kernel_full_rerank (mixed): Composable Kernel Python bindings are not importable as ck or composable_kernel in this image. Enabling this would require CK development headers/examples or a purpose-built CK extension for this workload.

## Recommendation

Use `rocblas_plus_triton_score` with `bf16` for the hackathon slide: 1.039 ms, 30802253300 candidates/s, 6.204x vs the PyTorch FP32 eager baseline, top-k overlap 0.997.
