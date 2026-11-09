# Huawei Ascend

## Runtime setup

Install the Ascend driver, CANN, and compatible PyTorch and TorchNPU versions. Initialize CANN before starting Python. The default compiled timing backend, TorchAir, is bundled in supported TorchNPU distributions. [Ascend compatibility matrix](https://github.com/Ascend/pytorch/blob/master/COMPATIBILITY.en.md) · [TorchAir](https://github.com/Ascend/torchair)

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

TorchAir is the default compiler backend on Ascend. If it raises a compatibility error, use `compile=False` for eager timing.

To configure TorchAir, create a `CompilerConfig` and pass the resulting backend to the measurement function. This example enables graph dumps.

```python
import torchair
from torchair.configs.compiler_config import CompilerConfig

config = CompilerConfig()
config.debug.graph_dump.type = "pbtxt"
backend = torchair.get_npu_backend(compiler_config=config)
latency = measure_module_latency(model, inputs, device, compile_kwargs={"backend": backend})
```

See the [TorchAir documentation](https://github.com/Ascend/torchair) for configuration options.

## Triton Ascend

To use [Triton Ascend](https://github.com/triton-lang/triton-ascend) through PyTorch Inductor, install a compatible Triton Ascend, CANN, PyTorch, and TorchNPU combination. With `compile=True` (the default), select the Inductor backend explicitly:

```python
latency = measure_module_latency(model, inputs, device, compile_kwargs={"backend": "inductor"})
```

TorchNPU uses Triton as its default Inductor compilation mode. If `TORCHINDUCTOR_NPU_BACKEND` has been changed, set it to `default` before the first compilation to select Triton. See the [TorchNPU backend documentation](https://github.com/Ascend/pytorch/blob/master/docs/zh/api/environment_variable/inductor/TORCHINDUCTOR_NPU_BACKEND.md).

For a callable that launches a Triton kernel directly, use eager timing:

```python
latency = measure_module_latency(triton_callable, (x, y, output), device, compile=False)
```
