# Measurement API

Import `measure_module_latency`, `calculate_model_complexity`, `count_operations`, and `count_parameters` from `torch_prisma`.

## Inputs and execution

Set the model's train or eval mode and place the model and inputs on their execution devices before measuring. Both APIs run the supplied model or callable with the supplied inputs. Forward calls can change parameters, buffers, and inputs. Pass a separate model instance or input copies when you need to preserve the originals.

A tuple or list in `input_args` supplies positional arguments. Any other value supplies one positional argument. Use `()` for a keyword-only forward and `(items,)` to pass a list as one argument. Use `input_kwargs` for keyword arguments.

## Latency

`measure_module_latency(target, input_args, device, input_kwargs=None, *, repetitions=20, warmup=5, compile=True, compile_kwargs=None)` returns a `LatencyStats` dataclass.

Each sample is the duration of one forward for the supplied batch. The required `device` selects the timing device. CPU timing uses `time.perf_counter()`. CUDA and Ascend timing use device events on the current stream and synchronize the selected device around the measurement loop. Include any required stream dependencies in the callable.

Latency measurement uses `torch.compile` by default. Compilation preparation runs three untimed forwards with temporarily raised Dynamo recompilation limits. Warmup and sampling run afterward with the original limits. If either phase requires compilation, the measurement raises an error because the compiled latency would be unreliable. Use `compile=False` to measure eager latency. Preparation and warmup can change model and input state before sampling starts. `compile_kwargs` is forwarded to `torch.compile`. The default backend on Ascend is TorchAir. See [Ascend setup](ascend.md).

| Field | Definition |
| --- | --- |
| `samples_ms` | Timings in measurement order |
| `median_ms`, `mean_ms` | Median and arithmetic mean |
| `std_ms` | Population standard deviation |
| `p90_ms`, `p99_ms` | Samples at index `min(int(n * q), n - 1)` in sorted order for `q = 0.90` or `0.99` |
| `min_ms`, `max_ms` | Minimum and maximum |

All values use milliseconds. `LatencyStats.from_samples(samples_ms)` summarizes timings from any source under the same convention. `to_dict()` returns a JSON-ready representation.

When comparing two measurements, hold the batch size, shape, dtype, device, thread count, compilation options, warmup, and repetition count fixed. Keep the samples when reporting tail percentiles.

## Complexity

`calculate_model_complexity(target, input_args, input_kwargs=None)` returns a `ModelComplexity` dataclass for one eager forward. Run it separately from latency measurement. It disables gradients and the MHA fast path and requests math scaled dot-product attention during counting.

| Field | Definition |
| --- | --- |
| `params` | Elements in unique registered parameters, including frozen and unused parameters |
| `macs` | Known dense matrix and convolution multiply-accumulates on the executed path |
| `flops` | Known nominal floating-point arithmetic on the executed path |
| `uncovered_ops` | Sorted operator names with incomplete counts, grouped under `macs` and `flops` |

`count_parameters(model)` reports the registered parameter count without a forward. It deduplicates by parameter identity and excludes buffers. Distinct parameters with shared storage count separately. Registered tensor dimensions determine the count for masked or zero-valued weights. For a plain callable, `params` defaults to zero. `ModelComplexity.to_dict()` returns JSON-ready counts and coverage.

The counter attributes two nominal FLOPs to each real floating-point dense MAC. Supported bias, elementwise, reduction, and normalization arithmetic adds FLOPs independently. Comparisons, ReLU, forward max pooling, indexing, and data movement contribute zero FLOPs under this convention. Integer matrix work contributes logical MACs and zero FLOPs. Complex matrix work contributes logical MACs and a FLOP coverage gap. Masking and kernel fusion leave the declared dense MAC convention unchanged.

The totals are known subtotals for the full batch. For example, GELU can appear under `uncovered_ops["flops"]` while leaving MAC coverage complete. A fused attention kernel can contribute its matrix products while retaining a FLOP diagnostic for its other arithmetic. Coverage depends on the operations visible to the dispatcher and profiler. An empty coverage list describes the observed trace. Opaque work can remain invisible to both instruments. Use latency to compare hardware execution time.

### Nominal FLOP formulas

The following formulas cover common operations. Each transcendental operation counts as one nominal unit.

| Operation | FLOP convention |
| --- | --- |
| Addition, subtraction, multiplication, and division | One per output element. Add a multiplication when an add or subtract applies a nonunit scale. |
| Sum and mean over a group of `K` elements | `K - 1` additions for a nonempty group, plus one division for mean. |
| Softmax over a group of `K` elements | `4K - 1` operations for a nonempty group. The count includes subtraction, exponentiation, summation, and division. |
| LayerNorm and GroupNorm | `5K + 2` operations per group of `K` elements, with another `K` for each affine weight or bias. |
| BatchNorm with stored statistics | Two operations per channel and two per input element, with another operation per element for each affine weight or bias. |
| Bias and scaling in a matrix product | An addition per output element when the bias term is active, plus each applicable nonunit scalar multiplication. |
| Convolution with bias | One additional FLOP per output element. |

### Prepared forwards

`count_operations(forward)` runs a zero-argument eager callable and returns an `OperationCounts` dataclass with `macs`, `flops`, and `uncovered_ops`. Prepare the model and inputs before calling it. Code that measures a sampled network can use this function and compute its active parameter count separately.

```python
from torch_prisma import count_operations

model.eval().to(device)
inputs = inputs.to(device)
operations = count_operations(lambda: model(inputs))
```
