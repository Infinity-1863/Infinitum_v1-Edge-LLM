# Performance

Raw weights and benchmark artifacts are not committed.

## Hardware

### PC

- ASUS Vivobook S 14 S5406SA_S5406SA
- Intel(R) Core(TM) Ultra 7 258V, 8 cores / 8 logical processors
- 33,805,971,456 bytes RAM
- Intel(R) Arc(TM) 140V GPU (16GB)

### Phone

- Model: 22041216G
- Device/product: xaga / xaga_global
- SoC: MT6895Z/TCZA
- Hardware: mt6895
- ABI: arm64-v8a
- Android: 14, SDK 34
- RAM: 7,664,796 KiB total

## Model Package

Phone package: `/data/local/tmp/infinitum-edge-gpt-oss-20b`

- Core GGUF: `model/gpt-oss-24l-32e-core-only.gguf`, 3,613,724,192 bytes
- Expert index: `moe/gpt_oss_expert_store_index.fixed.json`, 3,039,941 bytes
- Expert pack: `moe_ggml_pack/experts.ggml_mxfp4.bin`, 10,178,887,680 bytes
- Runtime: Android aarch64 llama.cpp-compatible server

## PC

| State | Profile | Prompt tok/s | Decode tok/s | Notes |
| --- | --- | ---: | ---: | --- |
| before | baseline | 10.618 | 8.995 | `20260603-202609-baseline.result.json` |
| after | page-prefetch | 10.803 | 8.534 | `20260603-202742-page-prefetch.result.json` |

### GPT-OSS 120B Laptop

| State | Profile | Prompt tok/s | Decode tok/s | Notes |
| --- | --- | ---: | ---: | --- |
| baseline | graph | 0.9 | 2.4 | `baseline_120b_laptop_cpu_20260604.result.json` |
| current | graph | 2.9 | 1.9 | `gpt_oss_120b_laptop_prefetch_summary_20260604.json` |
| before | pack-slots | 2.7 | 0.5 | `gpt_oss_120b_laptop_prefetch_summary_20260604.json` |
| after | pack-slots early-prefetch | 3.2 | 0.8 | `gpt_oss_120b_laptop_prefetch_summary_20260604.json` |
| before | graph, 4 threads | 1.4 | 2.35 | `pc-bench-120b-queue/20260604-134409-baseline.result.json` |
| after | graph predictor-prefetch, 4 threads | 1.41 | 3.16 | `pc-bench-120b-queue/20260604-135330-page-prefetch.result.json` |
| best | graph, auto threads | 2.82 | 3.24 | `pc-bench-120b-queue/20260604-135530-baseline.result.json` |
| before | Vulkan baseline, 32 tokens | 2.11 | 3.05 | `pc-bench-router-shadow-120b/20260605-100445-baseline.result.json` |
| current | router-shadow top-6, 32 tokens | 2.11 | 0.97 | `pc-bench-router-shadow-120b/20260605-100308-router-shadow.result.json` |
| before | Vulkan baseline, 64 tokens | 2.19 | 1.91 | `pc-bench-vulkan-120b/20260605-105410-baseline.result.json` |
| after | Vulkan row4, 64 tokens | 2.15 | 3.23 | `pc-bench-vulkan-120b/20260605-110229-baseline.result.json` |
| after | oneAPI/SYCL row6, 64 tokens | 2.96 | 2.79 | `pc-bench-sycl-120b/20260605-112241-baseline.result.json` |
| before | Vulkan baseline, 64 tokens | 2.19 | 1.91 | `pc-bench-vulkan-120b/20260605-105410-baseline.result.json`, wall 57.2s |
| after | oneAPI/SYCL merged, 64 tokens | 2.97 | 2.58 | `pc-bench-sycl-merged-auto-120b/20260605-153852-baseline.result.json`, wall 42.4s |
| before | oneAPI/SYCL merged, 64 tokens | 3.02 | 2.53 | `pc-bench-sycl-merged-current-120b/20260605-165411-baseline.result.json`, wall 42.3s |
| after | oneAPI/SYCL merged + flash attention, 64 tokens | 2.91 | 2.86 | `pc-bench-sycl-fa-on-120b/20260605-165902-baseline.result.json`, wall 40.0s |
| final | oneAPI/SYCL merged + flash attention, 64 tokens | 2.40 | 2.82 | `pc-bench-sycl-fa-on-final-local-120b/20260605-184404-baseline.result.json`, wall 39.0s |

Delta: `57.2s -> 42.4s` wall time (`-25.9%`).
Delta: `42.3s -> 40.0s` wall time (`-5.4%`), decode `2.53 -> 2.86 tok/s` (`+13.2%`).

## Phone

### Page Prefetch

| State | Profile | Prompt tok/s | Decode tok/s | RSS KiB | Notes |
| --- | --- | ---: | ---: | ---: | --- |
| before | baseline | 1.928 | 0.404 | 4,325,940 | `20260603-205628-baseline.result.json` |
| after | page-prefetch | 1.868 | 0.411 | 4,337,744 | `20260603-205954-page-prefetch.result.json` |

### Full-Model Compute

Mode: full selected experts.

| State | Profile | Extra env | Prompt tok/s | Decode tok/s | RSS KiB | Notes |
| --- | --- | --- | ---: | ---: | ---: | --- |
| before | baseline | `LLAMA_INFINITUM_EXPERT_BACKEND=cpu` | 2.083 | 0.330 | 4,056,872 | `20260603-225851-baseline-full-cpu.result.json` |
| after | page-prefetch | `LLAMA_INFINITUM_EXPERT_BACKEND=cpu`, `LLAMA_INFINITUM_EXPERT_ROW_THREADS=4` | 1.903 | 0.503 | 4,372,612 | `20260603-231046-page-prefetch-full-cpu-row4.result.json` |

Delta: `0.330 -> 0.503 tok/s` (`+52.4%`).
