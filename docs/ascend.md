# Huawei Ascend

## Runtime setup

Install the Ascend driver, CANN, and compatible PyTorch and TorchNPU versions. Initialize CANN before starting Python. Compiled timing requires TorchAir, which is bundled in supported TorchNPU distributions. [Ascend compatibility matrix](https://github.com/Ascend/pytorch/blob/master/COMPATIBILITY.en.md) · [TorchAir](https://github.com/Ascend/torchair)

Install Torch Prisma in the prepared environment.

```bash
uv pip install --no-deps torch-prisma
```

## Measuring latency

```python
import torch
import torch_npu
from torch import nn

from torch_prisma import measure_module_latency

device = torch.device("npu:0")
model = nn.Linear(128, 64, device=device).eval()
inputs = torch.randn(8, 128, device=device)

print(measure_module_latency(model, inputs, device).median_ms)
```

TorchAir is the default compiler backend on Ascend. If it raises a compatibility error, use `compile=False` for eager timing. For a custom TorchAir configuration, pass the result of `torchair.get_npu_backend(compiler_config=config)` as `compile_kwargs={"backend": backend}`.
