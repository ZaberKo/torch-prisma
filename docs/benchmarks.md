# Latency benchmarks

Each case uses one accelerator, one PyTorch intra-op thread, and a fixed FP32 input of shape `(1, 3, 224, 224)`. TorchVision models run in eval mode with fixed randomly initialized weights. Both stacks use PyTorch 2.14.0 and TorchVision 0.29.0. The A100 uses CUDA 12.6 and NVIDIA driver 550.163.01. The Ascend stack uses TorchNPU 2.14.0 and CANN 9.2.0-beta.2.

## Default settings

Each latency call uses five warmup forwards and twenty timed forwards. Compiled calls also run three untimed preparation forwards. The latency columns average the statistics returned by 12 warmed calls in one process. First-call CV uses four independent starts per A100 case and six for compiled ResNet-50 on Ascend. CV is the sample standard deviation divided by the mean, expressed as a percentage. Each pair shows the CV of the returned mean and median, in that order.

| Device | Model | Execution | Mean (ms) | Median (ms) | Warmed CV | First-call CV |
| --- | --- | --- | ---: | ---: | ---: | ---: |
| Ascend 910B2 | ResNet-50 | Eager | 7.154 | 7.143 | 1.03% / 1.02% | — |
| Ascend 910B2 | ResNet-50 | TorchAir | 2.283 | 2.251 | 0.09% / 0.04% | 4.18% / 4.26% |
| Ascend 910B2 | MobileNetV3-Small | Eager | 5.665 | 5.661 | 1.39% / 1.42% | — |
| Ascend 910B2 | MobileNetV3-Small | TorchAir | 1.820 | 1.789 | 0.08% / 0.02% | — |
| Ascend 910B2 | ViT-B/16 | Eager | 8.132 | 8.128 | 1.14% / 1.14% | — |
| A100 PCIe 40 GB | ResNet-50 | Eager | 4.490 | 4.051 | 11.43% / 0.54% | 2.05% / 1.92% |
| A100 PCIe 40 GB | ResNet-50 | Inductor | 2.491 | 2.223 | 19.31% / 4.53% | 0.79% / 0.77% |
| A100 PCIe 40 GB | MobileNetV3-Small | Eager | 3.965 | 3.642 | 13.58% / 0.27% | 5.03% / 4.99% |
| A100 PCIe 40 GB | MobileNetV3-Small | Inductor | 1.989 | 1.759 | 14.76% / 0.44% | 0.74% / 0.92% |
| A100 PCIe 40 GB | ViT-B/16 | Eager | 4.859 | 4.675 | 5.43% / 0.03% | 15.06% / 14.11% |
| A100 PCIe 40 GB | ViT-B/16 | Inductor | 4.050 | 4.019 | 1.83% / 0.69% | 12.68% / 22.96% |

ViT-B/16 encountered a fused attention error with the tested TorchAir stack.

## Reproducibility across process starts

On Ascend, compiled ResNet-50 median CV stayed near 4.1% across 12 independent processes at settings from 5/20 to 100/200.

The A100 ViT-B/16 sweep used four independent processes for each setting and mode. Each pair shows first-call mean CV and median CV.

| Warmup / samples | Eager CV | Inductor CV |
| --- | ---: | ---: |
| 5 / 20 | 7.39% / 7.36% | 8.81% / 7.93% |
| 5 / 100 | 3.13% / 0.07% | 3.29% / 1.32% |
| 20 / 20 | 14.28% / 19.15% | 18.40% / 28.26% |
| 20 / 100 | 3.00% / 0.18% | 4.90% / 0.36% |
| 50 / 20 | 1.08% / 0.17% | 2.42% / 0.54% |

At 50/20, the first-call median averaged 4.668 ms in eager mode and 4.069 ms with Inductor. This setting gave the lowest mean CV in both modes. Report first-call and warmed results separately when comparing process starts.
