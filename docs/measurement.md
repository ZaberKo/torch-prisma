# Measurement API

Import `measure_module_latency`, `calculate_model_complexity`, `count_operations`, and `count_parameters` from `torch_prisma`.

## Inputs and execution

Set the model mode and place the model and inputs on their execution devices before measuring. Forward side effects persist. Pass a separate model instance or copies of the inputs to preserve the originals.

A tuple or list in `input_args` supplies positional arguments. Other values supply one positional argument. Use `(items,)` to pass a list as one argument, or `()` with `input_kwargs` for a keyword-only forward.

## Latency

`measure_module_latency(target, input_args, device, input_kwargs=None, *, repetitions=20, warmup=5, compile=True, compile_kwargs=None)` returns a `LatencyStats` dataclass.

Each sample measures one forward in inference mode for the entire batch. CPU uses a wall clock. CUDA and Ascend use events on the current stream of the selected device. The callable must handle any dependencies on other streams.

Compilation runs three untimed forwards before warmup and raises an error if further compilation is required during warmup or sampling. Set `compile=False` for eager timing. Use `compile_kwargs` to configure `torch.compile`. [Ascend backend setup](ascend.md)

| Field | Definition |
| --- | --- |
| `samples_ms` | Timings in measurement order |
| `median_ms`, `mean_ms` | Median and arithmetic mean |
| `std_ms` | Population standard deviation |
| `p90_ms`, `p99_ms` | Samples at index `min(int(n * q), n - 1)` in sorted order for `q = 0.90` or `0.99` |
| `min_ms`, `max_ms` | Minimum and maximum |

Durations are in milliseconds. Use `LatencyStats.from_samples()` to summarize external timings and `to_dict()` to serialize the result. [Benchmark protocol and results](benchmarks.md)

## Complexity

`calculate_model_complexity(target, input_args, input_kwargs=None)` returns a `ModelComplexity` record for one eager forward without gradients. Counting requests math scaled dot-product attention and disables the multihead attention fast path.

| Field | Definition |
| --- | --- |
| `params` | Elements in unique registered parameters, including frozen and unused parameters |
| `macs` | Known dense matrix and convolution multiply-accumulates on the executed path |
| `flops` | Known nominal floating-point arithmetic on the executed path |
| `uncovered_ops` | Sorted operator names with incomplete counts, grouped under `macs` and `flops` |

`count_parameters(model)` counts unique parameter objects without executing a forward. Counts use registered tensor dimensions. Distinct parameters that share storage count separately. A plain callable reports zero parameters.

Each real floating-point dense MAC contributes two FLOPs. Supported arithmetic outside the matrix products adds to FLOPs. Comparisons, ReLU, forward max pooling, indexing, and data movement contribute zero FLOPs. Integer matrix work contributes MACs only. Complex matrix work contributes MACs and a FLOP coverage gap.

MACs and FLOPs are subtotals for the supplied batch. Check `uncovered_ops` for visible operations with incomplete counts. Opaque kernels can conceal operations from the counter.

### Nominal FLOP formulas

Each transcendental operation counts as one FLOP under this convention.

| Operation | FLOP convention |
| --- | --- |
| Addition, subtraction, multiplication, and division | One per output element. Add a multiplication when an add or subtract applies a nonunit scale. |
| Sum and mean over a group of `K` elements | `K - 1` additions for a nonempty group, plus one division for mean. |
| Softmax over a group of `K` elements | `4K - 1` operations for a nonempty group. |
| LayerNorm and GroupNorm | `5K + 2` operations per group of `K` elements, with another `K` for each affine weight or bias. |
| BatchNorm with stored statistics | Two operations per channel and two per input element, with another operation per element for each affine weight or bias. |
| Bias and scaling in a matrix product | An addition per output element when the bias term is active, plus each applicable nonunit scalar multiplication. |
| Convolution with bias | One additional FLOP per output element. |

### Prepared forwards

`count_operations(forward)` returns `OperationCounts` for a zero-argument eager callable. Use it when parameter counting is handled separately.

```python
from torch_prisma import count_operations

model.eval().to(device)
inputs = inputs.to(device)
operations = count_operations(lambda: model(inputs))
```
