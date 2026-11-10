# Torch Prisma

Torch Prisma measures PyTorch inference latency, parameters, multiply-accumulates (MACs), and floating-point operations (FLOPs) on CPUs and accelerators.

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

# For inference, set the model to evaluation mode and place it and the inputs on the device.
device = "cpu"
model = nn.Linear(8, 4).eval().to(device)
inputs = torch.randn(2, 8, device=device)

complexity = calculate_model_complexity(model, inputs)
latency = measure_module_latency(model, inputs, device=device)

print(complexity.params, complexity.macs, complexity.flops)
# Operations with incomplete counts, grouped by MACs and FLOPs.
print(complexity.uncovered_ops)
print(latency.median_ms)
```

## Guides

- [Measurement API and conventions](measurement.md)
- [Latency benchmarks](benchmarks.md)
- [Huawei Ascend](ascend.md)
