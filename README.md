# Torch Prisma

Torch Prisma measures PyTorch inference latency, registered parameters, multiply-accumulates (MACs), and nominal floating-point operations (FLOPs). It supports CPU, CUDA, and Huawei Ascend, including compiled latency with TorchAir.

## Installation

Torch Prisma requires Python 3.10 or later and PyTorch 2.6 or later. Create a virtual environment with `uv venv` if needed, then install a published release.

```bash
uv pip install torch-prisma
```

## Usage

Measure inference latency and complexity for a representative input batch. Latency uses `torch.compile` by default. Pass `compile=False` for eager timing.

```python
import torch
from torch import nn

from torch_prisma import calculate_model_complexity, measure_module_latency

device = "cpu"
model = nn.Linear(8, 4, device=device).eval()
x = torch.randn(2, 8, device=device)

complexity = calculate_model_complexity(model, x)
latency = measure_module_latency(model, x, device=device, warmup=5, repetitions=20)

print(complexity.params, complexity.macs, complexity.flops)
print(complexity.uncovered_ops)
print(latency.median_ms, latency.p90_ms)
```

The measurement guide covers input preparation, counting conventions, and the API. The Ascend guide covers CANN, `torch_npu`, and TorchAir. [Measurement guide](https://github.com/ZaberKo/torch-prisma/blob/main/docs/measurement.md) · [Ascend setup](https://github.com/ZaberKo/torch-prisma/blob/main/docs/ascend.md)

## Development

```bash
uv sync
uv run pre-commit install
uv run pre-commit run --all-files
uv run pytest
uv build
```

Preview the documentation with `uv run --group docs zensical serve`.

## License

Torch Prisma is distributed under the [MIT License](https://github.com/ZaberKo/torch-prisma/blob/main/LICENSE).
