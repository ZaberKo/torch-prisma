# Torch Prisma

Torch Prisma measures PyTorch inference latency, parameters, multiply-accumulates (MACs), and floating-point operations (FLOPs) on CPU, CUDA, and Huawei Ascend.

## Installation

Requires Python 3.10+ and PyTorch 2.6+. Install in a virtual environment.

```bash
uv pip install torch-prisma
```

## Usage

Latency uses `torch.compile` by default. Set `compile=False` for eager timing.

```python
import torch
from torch import nn

from torch_prisma import calculate_model_complexity, measure_module_latency

device = "cpu"
model = nn.Linear(8, 4, device=device).eval()
inputs = torch.randn(2, 8, device=device)

complexity = calculate_model_complexity(model, inputs)
latency = measure_module_latency(model, inputs, device=device)

print(complexity.to_dict())
print(latency.median_ms)
```

## Guides

- [Measurement API and conventions](measurement.md)
- [Latency benchmarks](benchmarks.md)
- [Huawei Ascend and TorchAir](ascend.md)
- [Development and releases](release.md)
