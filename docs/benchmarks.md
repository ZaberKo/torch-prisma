# Latency benchmarks

These measurements examine variation across independent process starts. Each process calls `measure_module_latency()` once. Each reported combination has 16 results from independent processes, collected across two cards tested concurrently with case order shuffled. Results retain all timed samples, including outliers. These measurements used device-specific events, before the switch to PyTorch's generic event API.

Each test uses one accelerator, one PyTorch intra-op thread, randomly initialized TorchVision weights, and a fixed FP32 input of shape `(1, 3, 224, 224)`. The model is in evaluation mode, and weights and input use a fixed seed. The warmup and sample counts are shown below. Compiled measurements also use three preparation forwards. Inductor's disk cache was reused between processes on each card.

Each latency value is the average of the 16 returned `mean_ms` or `median_ms` values. CV means *coefficient of variation*: the sample standard deviation divided by the average, expressed as a percentage. Each CV describes the variation across those 16 returned values.

## Ascend 910B2

The Ascend environment used PyTorch 2.10.0, TorchVision 0.25.0, TorchNPU 2.10.0.post7, Triton Ascend 3.6.0, CANN 9.2.0-beta.2, and driver 26.1.1. Inductor used Triton mode and four compiler threads. TorchAir is the default Ascend compiler backend.

| Model | Execution | Warmup / samples | Mean (ms) / CV | Median (ms) / CV |
| --- | --- | --- | ---: | ---: |
| MobileNetV3-Small | Eager | 5 / 20 | 6.895 / 4.00% | 6.804 / 3.12% |
| MobileNetV3-Small | Eager | 20 / 100 | 6.612 / 4.65% | 6.559 / 4.40% |
| MobileNetV3-Small | Eager | 50 / 20 | 6.365 / 3.86% | 6.343 / 3.49% |
| MobileNetV3-Small | TorchAir | 5 / 20 | 1.724 / 3.66% | 1.704 / 3.77% |
| MobileNetV3-Small | TorchAir | 20 / 100 | 1.744 / 4.28% | 1.739 / 4.29% |
| MobileNetV3-Small | TorchAir | 50 / 20 | 1.748 / 4.46% | 1.723 / 4.37% |
| MobileNetV3-Small | Triton Inductor | 5 / 20 | 7.488 / 4.85% | 7.477 / 4.98% |
| MobileNetV3-Small | Triton Inductor | 20 / 100 | 7.400 / 3.23% | 7.374 / 3.12% |
| MobileNetV3-Small | Triton Inductor | 50 / 20 | 7.303 / 3.91% | 7.293 / 3.88% |
| ResNet-50 | Eager | 5 / 20 | 7.481 / 6.15% | 7.451 / 6.15% |
| ResNet-50 | Eager | 20 / 100 | 7.235 / 4.38% | 7.196 / 4.27% |
| ResNet-50 | Eager | 50 / 20 | 7.181 / 3.21% | 7.162 / 3.25% |
| ResNet-50 | TorchAir | 5 / 20 | 2.378 / 4.05% | 2.361 / 4.07% |
| ResNet-50 | TorchAir | 20 / 100 | 2.375 / 3.77% | 2.371 / 3.76% |
| ResNet-50 | TorchAir | 50 / 20 | 2.359 / 4.03% | 2.338 / 4.12% |
| ResNet-50 | Triton Inductor | 5 / 20 | 12.365 / 0.29% | 12.324 / 0.28% |
| ResNet-50 | Triton Inductor | 20 / 100 | 12.329 / 0.35% | 12.321 / 0.35% |
| ResNet-50 | Triton Inductor | 50 / 20 | 12.376 / 0.32% | 12.321 / 0.35% |
| ViT-B/16 | Eager | 5 / 20 | 8.810 / 4.79% | 8.799 / 4.49% |
| ViT-B/16 | Eager | 20 / 100 | 8.344 / 6.71% | 8.311 / 6.14% |
| ViT-B/16 | Eager | 50 / 20 | 8.278 / 4.42% | 8.253 / 4.39% |
| ViT-B/16 | Triton Inductor | 5 / 20 | 10.945 / 2.51% | 10.899 / 2.43% |
| ViT-B/16 | Triton Inductor | 20 / 100 | 10.578 / 3.05% | 10.553 / 3.15% |
| ViT-B/16 | Triton Inductor | 50 / 20 | 10.672 / 5.36% | 10.643 / 5.21% |

ViT-B/16 with TorchAir failed during compilation preparation with `only supported softmax_layout=TND`; no latency statistics are available for that combination.

## GeForce RTX 3080 Ti

The 3080 Ti environment used PyTorch 2.15.0, TorchVision 0.29.2, CUDA runtime 13.2, Triton 3.9.0, and NVIDIA driver 595.71.05. Inductor used four compiler threads. Software versions differ between devices, so these results are not a controlled cross-device performance comparison.

| Model | Execution | Warmup / samples | Mean (ms) / CV | Median (ms) / CV |
| --- | --- | --- | ---: | ---: |
| MobileNetV3-Small | Eager | 5 / 20 | 4.302 / 11.07% | 4.267 / 10.99% |
| MobileNetV3-Small | Eager | 20 / 100 | 4.399 / 12.16% | 4.383 / 12.36% |
| MobileNetV3-Small | Eager | 50 / 20 | 4.149 / 6.94% | 4.127 / 7.10% |
| MobileNetV3-Small | Inductor | 5 / 20 | 2.214 / 14.75% | 2.209 / 14.60% |
| MobileNetV3-Small | Inductor | 20 / 100 | 2.115 / 11.96% | 2.114 / 13.02% |
| MobileNetV3-Small | Inductor | 50 / 20 | 2.046 / 8.14% | 2.038 / 8.22% |
| ResNet-50 | Eager | 5 / 20 | 5.197 / 10.02% | 5.106 / 10.16% |
| ResNet-50 | Eager | 20 / 100 | 4.884 / 1.46% | 4.840 / 1.42% |
| ResNet-50 | Eager | 50 / 20 | 5.006 / 8.10% | 4.981 / 8.31% |
| ResNet-50 | Inductor | 5 / 20 | 2.656 / 15.85% | 2.612 / 13.01% |
| ResNet-50 | Inductor | 20 / 100 | 2.674 / 13.58% | 2.658 / 13.51% |
| ResNet-50 | Inductor | 50 / 20 | 2.613 / 15.67% | 2.590 / 14.44% |
| ViT-B/16 | Eager | 5 / 20 | 4.522 / 7.30% | 4.505 / 7.23% |
| ViT-B/16 | Eager | 20 / 100 | 4.224 / 7.89% | 4.151 / 8.41% |
| ViT-B/16 | Eager | 50 / 20 | 4.284 / 8.91% | 4.249 / 9.10% |
| ViT-B/16 | Inductor | 5 / 20 | 4.655 / 1.43% | 4.670 / 0.17% |
| ViT-B/16 | Inductor | 20 / 100 | 4.315 / 1.62% | 4.240 / 0.98% |
| ViT-B/16 | Inductor | 50 / 20 | 4.282 / 1.11% | 4.240 / 0.82% |

## Interpreting the defaults

The default is 5 warmup forwards and 20 timed forwards. Suitable values depend on the model, backend, and device; compare settings for your workload. A low CV indicates consistent results across runs but does not establish that warmup is sufficient.
