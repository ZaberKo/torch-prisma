# Torch Prisma

Torch Prisma measures PyTorch inference latency, registered parameters, multiply-accumulates (MACs), and nominal floating-point operations (FLOPs). It supports CPU, CUDA, and Huawei Ascend, including compiled latency with TorchAir.

```bash
uv add torch-prisma
```

```python
import torch
from torch import nn

from torch_prisma import calculate_model_complexity, measure_module_latency

model = nn.Linear(8, 4)
x = torch.randn(2, 8)

complexity = calculate_model_complexity(model, x)
latency = measure_module_latency(model, x, warmup=5, repetitions=20)

print(complexity.params, complexity.macs, complexity.flops)
print(complexity.uncovered_ops)
print(latency.median_ms, latency.p90_ms)
```

Requires Python 3.10 or later and PyTorch 2.6 or later. The [measurement guide](https://github.com/ZaberKo/torch-prisma/blob/main/docs/measurement.md) covers input preparation, counting conventions, and the API. See [Ascend setup](https://github.com/ZaberKo/torch-prisma/blob/main/docs/ascend.md) for CANN, `torch_npu`, and TorchAir, and the [release guide](https://github.com/ZaberKo/torch-prisma/blob/main/docs/release.md) for publishing.

## Development

See the [development guide](https://github.com/ZaberKo/torch-prisma/blob/main/DEV.md) for setup, tests, and builds.

## License

Torch Prisma is distributed under the [MIT License](https://github.com/ZaberKo/torch-prisma/blob/main/LICENSE).
