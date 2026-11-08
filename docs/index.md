# Torch Prisma

Torch Prisma measures inference latency and model complexity for PyTorch models on CPU, CUDA, and Huawei Ascend. It reports latency samples and summary statistics alongside parameters, multiply-accumulates (MACs), floating-point operations (FLOPs), and counting coverage.

## Installation

Torch Prisma requires Python 3.10 or later and PyTorch 2.6 or later. Create a virtual environment with `uv venv` if needed, then install a published release.

```bash
uv pip install torch-prisma
```

## Usage

Measure a model with a representative input batch. Latency uses `torch.compile` by default. Pass `compile=False` for eager timing.

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
- [Huawei Ascend and TorchAir](ascend.md)
- [Development and releases](release.md)
