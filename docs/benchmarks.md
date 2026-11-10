# Latency benchmarks

These measurements examine variation across independent process starts. Each process calls `measure_module_latency()` once. Each supported combination has 16 results: eight on each of two physical cards. The two cards were tested concurrently, with case order shuffled. Results retain all timed samples, including outliers. These measurements used device-specific events, before the switch to PyTorch's generic event API.

Each test uses one accelerator, one PyTorch intra-op thread, randomly initialized TorchVision weights, and a fixed FP32 input of shape `(1, 3, 224, 224)`. The model is in evaluation mode, and weights and input use a fixed seed. The warmup and sample counts are shown below. Compiled measurements also use three preparation forwards. Inductor's disk cache was reused between processes separately for each card.

The latency columns are the averages of the 16 returned `mean_ms` and `median_ms` values. CV means *coefficient of variation*: the sample standard deviation divided by the average, expressed as a percentage. Each CV pair reports the variation of the returned mean and median, respectively. Combined CV uses all 16 processes; card CV uses the eight processes on that card. The Ascend test used cards 6 and 7; the 3080 Ti test used cards 0 and 1.

## Ascend 910B2

The Ascend environment used PyTorch 2.10.0, TorchVision 0.25.0, TorchNPU 2.10.0.post7, Triton Ascend 3.6.0, CANN 9.2.0-beta.2, and driver 26.1.1. Inductor used Triton mode and four compiler threads. TorchAir is the default Ascend compiler backend.

| Model | Execution | Warmup / samples | Mean (ms) | Median (ms) | Combined CV | Card 6 CV | Card 7 CV |
| --- | --- | --- | ---: | ---: | ---: | ---: | ---: |
| MobileNetV3-Small | Eager | 5 / 20 | 6.895 | 6.804 | 4.00% / 3.12% | 2.92% / 2.51% | 5.04% / 3.72% |
| MobileNetV3-Small | Eager | 20 / 100 | 6.612 | 6.559 | 4.65% / 4.40% | 4.64% / 4.75% | 4.96% / 4.20% |
| MobileNetV3-Small | Eager | 50 / 20 | 6.365 | 6.343 | 3.86% / 3.49% | 3.72% / 3.00% | 3.74% / 3.77% |
| MobileNetV3-Small | TorchAir | 5 / 20 | 1.724 | 1.704 | 3.66% / 3.77% | 3.89% / 4.05% | 3.55% / 3.62% |
| MobileNetV3-Small | TorchAir | 20 / 100 | 1.744 | 1.739 | 4.28% / 4.29% | 4.28% / 4.30% | 4.47% / 4.50% |
| MobileNetV3-Small | TorchAir | 50 / 20 | 1.748 | 1.723 | 4.46% / 4.37% | 4.87% / 4.51% | 4.19% / 4.42% |
| MobileNetV3-Small | Triton Inductor | 5 / 20 | 7.488 | 7.477 | 4.85% / 4.98% | 5.60% / 5.77% | 3.37% / 3.52% |
| MobileNetV3-Small | Triton Inductor | 20 / 100 | 7.400 | 7.374 | 3.23% / 3.12% | 2.28% / 2.15% | 3.81% / 3.76% |
| MobileNetV3-Small | Triton Inductor | 50 / 20 | 7.303 | 7.293 | 3.91% / 3.88% | 4.09% / 3.95% | 4.01% / 4.07% |
| ResNet-50 | Eager | 5 / 20 | 7.481 | 7.451 | 6.15% / 6.15% | 7.04% / 7.14% | 5.60% / 5.47% |
| ResNet-50 | Eager | 20 / 100 | 7.235 | 7.196 | 4.38% / 4.27% | 3.69% / 3.71% | 5.13% / 4.97% |
| ResNet-50 | Eager | 50 / 20 | 7.181 | 7.162 | 3.21% / 3.25% | 3.07% / 3.28% | 3.38% / 3.31% |
| ResNet-50 | TorchAir | 5 / 20 | 2.378 | 2.361 | 4.05% / 4.07% | 3.98% / 4.02% | 4.38% / 4.40% |
| ResNet-50 | TorchAir | 20 / 100 | 2.375 | 2.371 | 3.77% / 3.76% | 4.26% / 4.25% | 0.37% / 0.36% |
| ResNet-50 | TorchAir | 50 / 20 | 2.359 | 2.338 | 4.03% / 4.12% | 4.24% / 4.33% | 4.09% / 4.20% |
| ResNet-50 | Triton Inductor | 5 / 20 | 12.365 | 12.324 | 0.29% / 0.28% | 0.20% / 0.22% | 0.34% / 0.29% |
| ResNet-50 | Triton Inductor | 20 / 100 | 12.329 | 12.321 | 0.35% / 0.35% | 0.19% / 0.19% | 0.25% / 0.25% |
| ResNet-50 | Triton Inductor | 50 / 20 | 12.376 | 12.321 | 0.32% / 0.35% | 0.13% / 0.24% | 0.18% / 0.21% |
| ViT-B/16 | Eager | 5 / 20 | 8.810 | 8.799 | 4.79% / 4.49% | 3.05% / 2.64% | 6.19% / 5.90% |
| ViT-B/16 | Eager | 20 / 100 | 8.344 | 8.311 | 6.71% / 6.14% | 9.26% / 8.45% | 3.28% / 3.08% |
| ViT-B/16 | Eager | 50 / 20 | 8.278 | 8.253 | 4.42% / 4.39% | 5.82% / 5.72% | 2.72% / 2.69% |
| ViT-B/16 | Triton Inductor | 5 / 20 | 10.945 | 10.899 | 2.51% / 2.43% | 2.02% / 1.94% | 2.98% / 2.88% |
| ViT-B/16 | Triton Inductor | 20 / 100 | 10.578 | 10.553 | 3.05% / 3.15% | 2.44% / 2.44% | 3.48% / 3.66% |
| ViT-B/16 | Triton Inductor | 50 / 20 | 10.672 | 10.643 | 5.36% / 5.21% | 5.91% / 5.74% | 5.01% / 4.89% |

ViT-B/16 with TorchAir failed during compilation preparation on both cards with `only supported softmax_layout=TND`; no latency statistics are available for that combination.

## GeForce RTX 3080 Ti

The 3080 Ti environment used PyTorch 2.15.0, TorchVision 0.29.2, CUDA runtime 13.2, Triton 3.9.0, and NVIDIA driver 595.71.05. Inductor used four compiler threads. Software versions differ between devices, so these results are not a controlled cross-device performance comparison.

| Model | Execution | Warmup / samples | Mean (ms) | Median (ms) | Combined CV | Card 0 CV | Card 1 CV |
| --- | --- | --- | ---: | ---: | ---: | ---: | ---: |
| MobileNetV3-Small | Eager | 5 / 20 | 4.302 | 4.267 | 11.07% / 10.99% | 13.12% / 12.99% | 2.31% / 1.34% |
| MobileNetV3-Small | Eager | 20 / 100 | 4.399 | 4.383 | 12.16% / 12.36% | 11.37% / 11.69% | 13.54% / 13.61% |
| MobileNetV3-Small | Eager | 50 / 20 | 4.149 | 4.127 | 6.94% / 7.10% | 2.00% / 1.86% | 9.47% / 9.66% |
| MobileNetV3-Small | Inductor | 5 / 20 | 2.214 | 2.209 | 14.75% / 14.60% | 15.17% / 14.82% | 15.14% / 15.15% |
| MobileNetV3-Small | Inductor | 20 / 100 | 2.115 | 2.114 | 11.96% / 13.02% | 14.83% / 15.83% | 0.67% / 0.81% |
| MobileNetV3-Small | Inductor | 50 / 20 | 2.046 | 2.038 | 8.14% / 8.22% | 1.54% / 1.30% | 11.18% / 11.33% |
| ResNet-50 | Eager | 5 / 20 | 5.197 | 5.106 | 10.02% / 10.16% | 8.87% / 9.16% | 11.21% / 11.41% |
| ResNet-50 | Eager | 20 / 100 | 4.884 | 4.840 | 1.46% / 1.42% | 1.36% / 1.36% | 1.57% / 1.38% |
| ResNet-50 | Eager | 50 / 20 | 5.006 | 4.981 | 8.10% / 8.31% | 8.33% / 8.61% | 8.44% / 8.59% |
| ResNet-50 | Inductor | 5 / 20 | 2.656 | 2.612 | 15.85% / 13.01% | 10.20% / 10.24% | 18.49% / 14.66% |
| ResNet-50 | Inductor | 20 / 100 | 2.674 | 2.658 | 13.58% / 13.51% | 10.56% / 10.48% | 14.70% / 14.58% |
| ResNet-50 | Inductor | 50 / 20 | 2.613 | 2.590 | 15.67% / 14.44% | 10.95% / 11.09% | 19.15% / 17.22% |
| ViT-B/16 | Eager | 5 / 20 | 4.522 | 4.505 | 7.30% / 7.23% | 6.22% / 5.73% | 8.07% / 8.25% |
| ViT-B/16 | Eager | 20 / 100 | 4.224 | 4.151 | 7.89% / 8.41% | 2.35% / 1.45% | 10.07% / 10.50% |
| ViT-B/16 | Eager | 50 / 20 | 4.284 | 4.249 | 8.91% / 9.10% | 10.22% / 10.27% | 7.66% / 8.15% |
| ViT-B/16 | Inductor | 5 / 20 | 4.655 | 4.670 | 1.43% / 0.17% | 1.79% / 0.23% | 0.09% / 0.07% |
| ViT-B/16 | Inductor | 20 / 100 | 4.315 | 4.240 | 1.62% / 0.98% | 1.60% / 0.18% | 1.04% / 0.34% |
| ViT-B/16 | Inductor | 50 / 20 | 4.282 | 4.240 | 1.11% / 0.82% | 1.51% / 0.46% | 0.38% / 0.29% |

## Interpreting the defaults

The default is five warmup forwards and 20 timed forwards. On Ascend, the combined mean and median CVs at 5/20 ranged from 0.28% to 6.15% across the measured combinations. Increasing warmup or sample counts did not consistently reduce variation across all models and backends.

On the 3080 Ti, the combined median CV at 5/20 ranged from 0.17% to 14.60%. For ViT-B/16 with Inductor, the returned median averaged 4.670 ms at 5/20 and 4.240 ms at 50/20, even though the 5/20 median CV was only 0.17%. A low CV therefore does not by itself establish that the measurement has reached a representative latency. Increasing warmup to 50 did not reduce CV for every measured combination.

These results do not establish either 5/20 or 50/20 as a reliable setting for every workload or device. Compare settings for the workload and environment being measured, and inspect the samples when the mean and median differ substantially.
