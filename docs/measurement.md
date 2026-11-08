# Measurement API

Import `measure_module_latency`, `calculate_model_complexity`, `count_parameters`, `LatencyStats`, and `ModelComplexity` from `torch_prisma`.

## Inputs and execution

Both measurement functions take `target`, `input_args`, an optional `device`, and optional `input_kwargs`. A tuple or list in `input_args` supplies positional arguments. Any other value supplies one positional argument. Use `()` for a keyword-only forward and `(items,)` to pass a list as one argument. Nested tuples, lists, and dictionaries can contain tensors.

The selected device defaults to the first registered parameter or buffer, then the first input tensor, then CPU. Place all registered parameters and buffers on that device before measuring. Measurement copies the module and inputs together, sets the copy to eval mode, and moves independent input tensors to the selected device. The copy preserves repeated input references and checks shared storage among dense, non-quantized tensors in the input containers and module dictionaries. Pass storage-sharing views on the selected device. Measurement raises an error if the backend cannot preserve their storage relationship during copying.

Module measurement requires enough memory for a second copy of the module and inputs. Use a prepared callable when the target cannot be deep-copied or when the forward includes additional work. The callable owns its model placement, training mode, and captured state. Measurement copies the explicit inputs but leaves captured objects under the caller's control. Arrange aliases involving captured objects within the callable.

Measurement restores the default CPU RNG and the default RNG of the selected accelerator on success or failure. Separate generators and state shared through hooks, closures, external services, or globals remain under the caller's control. Keep the source module and inputs stable while a measurement is in progress. Coordinate with other code that changes process-wide PyTorch backend settings, consumes default RNG state, or uses the profiler.

## Latency

`measure_module_latency(target, input_args, device=None, input_kwargs=None, *, repetitions=20, warmup=5, compile=False, compile_kwargs=None)` returns a `LatencyStats` dataclass.

Each sample is the duration of one forward for the supplied batch. CPU timing uses `time.perf_counter()`. CUDA and Ascend timing use device events on the current stream and synchronize the selected device around the measurement loop. Include any required stream dependencies in the callable.

Input preparation, event setup, and warmup occur outside the timed loop. With `compile=True`, the first compiled call also occurs outside that loop. Use a representative warmup and stable inputs to limit recompilation during sampling. `compile_kwargs` is forwarded to `torch.compile`. The default backend on Ascend is TorchAir. See [Ascend setup](ascend.md).

| Field | Definition |
| --- | --- |
| `samples_ms` | Timings in measurement order |
| `median_ms`, `mean_ms` | Median and arithmetic mean |
| `std_ms` | Population standard deviation |
| `p90_ms`, `p99_ms` | Percentiles with linear interpolation between ordered samples |
| `min_ms`, `max_ms` | Minimum and maximum |

All values use milliseconds. `LatencyStats.from_samples(samples_ms)` summarizes timings from any source under the same convention. `to_dict()` returns a JSON-ready representation.

When comparing two measurements, hold the batch size, shape, dtype, device, thread count, compilation options, warmup, and repetition count fixed. Keep the samples when reporting tail percentiles.

## Complexity

`calculate_model_complexity(target, input_args, device=None, input_kwargs=None, *, parameter_count=None)` returns a `ModelComplexity` dataclass for one eager forward. Run it separately from latency measurement. It disables gradients and the MHA fast path and requests math scaled dot-product attention during counting.

| Field | Definition |
| --- | --- |
| `params` | Elements in unique registered parameters, including frozen and unused parameters |
| `macs` | Known dense matrix and convolution multiply-accumulates on the executed path |
| `flops` | Known nominal floating-point arithmetic on the executed path |
| `uncovered_ops` | Sorted operator names with incomplete counts, grouped under `macs` and `flops` |

`count_parameters(model)` reports the registered parameter count without a forward. It deduplicates by parameter identity and excludes buffers. Distinct parameters with shared storage count separately. Registered tensor dimensions determine the count for masked or zero-valued weights. For a plain callable, `params` defaults to zero. Supply `parameter_count` when the caller owns the active parameter count. `ModelComplexity.to_dict()` returns JSON-ready counts and coverage.

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

### Sampled networks

The caller selects the subnet and supplies its active parameter count. A callable can measure a prepared supernet without copying its cached sample state.

```python
from torch_prisma import calculate_model_complexity, measure_module_latency

supernet.set_sample_config(config)
supernet.eval()


def forward(x):
    return supernet(x)


complexity = calculate_model_complexity(
    forward, example_input, device=device, parameter_count=active_parameter_count
)
latency = measure_module_latency(forward, example_input, device=device)
```

Keep subnet selection, parameter accounting, and the capture of per-layer inputs in the caller.
