# Torch Prisma

Torch Prisma measures inference latency and model complexity for PyTorch models on CPU, CUDA, and Huawei Ascend. It reports latency samples and summary statistics alongside parameters, multiply-accumulates (MACs), floating-point operations (FLOPs), and counting coverage.

Torch Prisma requires Python 3.10 or later. Install a published release with uv.

```bash
uv add torch-prisma
```

Measure an eager model with a representative input batch.

```python
import torch
from torch import nn

from torch_prisma import calculate_model_complexity, measure_module_latency

model = nn.Linear(8, 4)
inputs = torch.randn(2, 8)

complexity = calculate_model_complexity(model, inputs)
latency = measure_module_latency(model, inputs)

print(complexity.to_dict())
print(latency.median_ms)
```

## Guides

- [Measurement API and conventions](measurement.md)
- [Huawei Ascend and TorchAir](ascend.md)
- [Development and releases](release.md)
