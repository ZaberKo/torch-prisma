# Torch Prisma

[![CI](https://github.com/ZaberKo/torch-prisma/actions/workflows/ci.yml/badge.svg)](https://github.com/ZaberKo/torch-prisma/actions/workflows/ci.yml)
[![PyPI](https://img.shields.io/pypi/v/torch-prisma)](https://pypi.org/project/torch-prisma/)
[![Python](https://img.shields.io/pypi/pyversions/torch-prisma)](https://pypi.org/project/torch-prisma/)
[![Documentation](https://readthedocs.org/projects/torch-prisma/badge/?version=latest)](https://torch-prisma.readthedocs.io/)
[![License](https://img.shields.io/pypi/l/torch-prisma)](LICENSE)

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

# Set evaluation mode and place the model and inputs on the device before measuring.
device = "cpu"
model = nn.Linear(8, 4).eval().to(device)
x = torch.randn(2, 8, device=device)

complexity = calculate_model_complexity(model, x)
# Forward side effects are not managed. Pass deep copies of the model and inputs if needed.
latency = measure_module_latency(model, x, device=device)

print(complexity.params, complexity.macs, complexity.flops)
# Operations with incomplete counts, grouped by MACs and FLOPs.
print(complexity.uncovered_ops)
print(latency.median_ms)
```

Documentation: [torch-prisma.readthedocs.io](https://torch-prisma.readthedocs.io/).

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
