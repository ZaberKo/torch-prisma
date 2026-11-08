# Huawei Ascend

Torch Prisma measures eager and compiled inference latency on Ascend NPUs. Eager complexity counting reports the operations visible through PyTorch dispatch and the CPU profiler. Vendor-specific fused operations can have incomplete coverage.

## Runtime setup

Install the Ascend driver and CANN, then install a compatible combination of Python, PyTorch, and `torch_npu`. For compiled latency, install the corresponding TorchAir distribution or use the TorchAir module provided by the runtime. Use the vendor's compatibility matrix to select the versions and initialize the CANN environment before starting Python. [Ascend Extension for PyTorch](https://github.com/Ascend/pytorch) · [TorchAir](https://github.com/Ascend/torchair)

Install Torch Prisma into the prepared Python environment. If the vendor stack is already installed, preserve its package versions with `--no-deps`.

```bash
uv pip install --python /path/to/ascend/python --no-deps torch-prisma
```

## Measuring an Ascend model

```python
import torch
import torch_npu  # Registers the NPU device.
from torch import nn

from torch_prisma import calculate_model_complexity, measure_module_latency

device = torch.device("npu:0")
model = nn.Linear(128, 64).to(device)
inputs = torch.randn(8, 128, device=device)

eager = measure_module_latency(model, inputs, device, warmup=10, repetitions=30)
compiled = measure_module_latency(model, inputs, device, warmup=10, repetitions=30, compile=True)
complexity = calculate_model_complexity(model, inputs, device)

print(eager.median_ms, compiled.median_ms)
print(complexity.to_dict())
```

Compiled timing selects `torchair.get_npu_backend()` by default. To use a custom compiler configuration, build the backend with `torchair.get_npu_backend(compiler_config=config)` and pass it as `compile_kwargs={"backend": backend}`. Keep compiler options and warmup settings with the timing results.

Timing uses NPU events and synchronization on the selected device. Check the returned complexity coverage before comparing counts across devices. The same eager model can be measured on CPU to obtain a CPU-dispatched reference under the same counting conventions.

For CI, the manual **Accelerator validation** workflow accepts `ascend` and runs eager timing, TorchAir timing, and eager counting on a runner labeled `self-hosted`, `linux`, and `ascend`. The runner's `python3` must have the compatible runtime, TorchAir, and pytest installed. Configure the runner service to inherit the CANN environment.
