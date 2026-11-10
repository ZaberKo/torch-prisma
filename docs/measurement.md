# Measurement API

Import the measurement functions from `torch_prisma`. Set the model's training or evaluation mode and place the model and inputs on their execution devices before measuring.

## `measure_module_latency`

Measures one forward per sample for the entire batch in inference mode. Forward side effects are not managed. Pass deep copies of the model and inputs if needed. CPU uses a wall clock. Accelerators use timed events on the selected device's current stream. The device type must match PyTorch's current accelerator type, and the backend must support timed events.

**Parameters**

- `target` — Prepared module or callable.
- `input_args` — Positional input or tuple/list of positional inputs. Use `(items,)` to pass a list as one argument.
- `device` — Timing device as a `torch.device` or string. Place the model and inputs on this device before measuring.
- `input_kwargs` — Optional keyword arguments.
- `repetitions` — Number of timed forwards. Defaults to 20.
- `warmup` — Number of untimed forwards after compilation preparation. Defaults to 5.
- `compile` — Enable `torch.compile`. Defaults to `True`.
- `compile_kwargs` — Keyword arguments passed to `torch.compile` when enabled. On Ascend, pass `{"backend": "inductor"}` to use Triton through Inductor. See the [Ascend guide](ascend.md#triton-ascend).

**Returns**

A `LatencyStats` result in milliseconds. Its `samples_ms` preserve sampling order. The remaining fields are `mean_ms`, `median_ms`, `std_ms` (population standard deviation), `min_ms`, `max_ms`, `p90_ms`, and `p99_ms`. Percentiles select the sample at index `min(int(n * q), n - 1)` after sorting, with `q` equal to 0.90 or 0.99. Use `LatencyStats.from_samples()` to summarize external timings and `to_dict()` to serialize the result.

**Raises**

- `ValueError` for invalid timing options or a device type that does not match the current accelerator.
- `RuntimeError` when the accelerator or default Ascend backend is unavailable, the backend does not support timed events, or compilation is required after preparation.

Compilation runs three untimed forwards before warmup. If PyTorch requests compilation after preparation, measurement fails rather than including compilation in the result. Set `compile=False` for eager timing or when compilation is incompatible with the model or device. See [Ascend setup](ascend.md) and [latency benchmarks](benchmarks.md).

## `calculate_model_complexity`

Counts operations for one eager forward without gradients. Counting requests math scaled dot-product attention and disables the multihead attention fast path.

**Parameters**

- `target` — Prepared module or callable.
- `input_args` — Positional input or tuple/list of positional inputs.
- `input_kwargs` — Optional keyword arguments.

**Returns**

A `ModelComplexity` result for the supplied batch. `params` counts unique registered parameter elements. `macs` and `flops` are known operation subtotals. `uncovered_ops` lists visible operations with incomplete counts, grouped by MACs and FLOPs. A plain callable reports zero parameters. Use `to_dict()` to serialize the result.

Each real floating-point dense MAC contributes two FLOPs. Supported arithmetic outside matrix products adds to FLOPs. Comparisons, ReLU, forward max pooling, indexing, and data movement contribute zero FLOPs. Integer matrix work contributes MACs only. Complex matrix work contributes MACs and a FLOP coverage gap.

## `count_parameters`

Counts the elements in unique registered parameter objects without executing a forward.

**Parameters**

- `model` — Module whose parameters are counted.

**Returns**

The parameter count as an integer. Distinct parameters sharing storage count separately.

## `count_operations`

Counts operations for a prepared zero-argument eager callable.

**Parameters**

- `forward` — Callable that executes the forward to count.

**Returns**

An `OperationCounts` result containing `macs`, `flops`, and `uncovered_ops`.

```python
from torch_prisma import count_operations

operations = count_operations(lambda: model(inputs))
```

## Nominal FLOP formulas

Each transcendental operation counts as one FLOP.

- Addition, subtraction, multiplication, and division contribute one FLOP per output element. A nonunit add/subtract scale contributes an additional multiplication.
- Sum over a nonempty group of `K` elements contributes `K - 1` additions. Mean adds one division.
- Softmax over a nonempty group of `K` elements contributes `4K - 1` FLOPs.
- LayerNorm and GroupNorm contribute `5K + 2` FLOPs per group of `K` elements, plus `K` for each affine weight or bias.
- BatchNorm with stored statistics contributes two FLOPs per channel and two per input element, plus one per element for each affine weight or bias.
- Matrix product bias and scaling contribute an addition per output element when the bias term is active, plus each applicable nonunit scalar multiplication.
- Convolution bias contributes one additional FLOP per output element.
