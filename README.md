# Torch Prisma

Measure PyTorch inference latency, parameters, MACs, and FLOPs on CPUs and accelerators.

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
x = torch.randn(2, 8, device=device)

complexity = calculate_model_complexity(model, x)
latency = measure_module_latency(model, x, device=device)

print(complexity.params, complexity.macs, complexity.flops)
# Operations with incomplete counts, grouped by MACs and FLOPs.
print(complexity.uncovered_ops)
print(latency.median_ms)
```

[Measurement API](https://github.com/ZaberKo/torch-prisma/blob/main/docs/measurement.md) · [Latency benchmarks](https://github.com/ZaberKo/torch-prisma/blob/main/docs/benchmarks.md) · [Huawei Ascend](https://github.com/ZaberKo/torch-prisma/blob/main/docs/ascend.md)

## Development

```bash
uv sync
uv run pre-commit install
uv run pytest
uv build
```

Preview the documentation with `uv run --group docs zensical serve`.

## License

Torch Prisma is distributed under the [MIT License](https://github.com/ZaberKo/torch-prisma/blob/main/LICENSE).
