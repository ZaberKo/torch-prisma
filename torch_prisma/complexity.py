"""Parameter counts, nominal FLOPs, dense MACs, and coverage diagnostics."""

from collections.abc import Callable, Iterator
from contextlib import contextmanager
from dataclasses import dataclass, field
from typing import Any

import torch
from torch import nn
from torch.autograd.profiler_util import FunctionEvent
from torch.nn.attention import SDPBackend, sdpa_kernel
from torch.profiler import ProfilerActivity, profile, record_function
from torch.utils.flop_counter import FlopCounterMode, flop_registry

from ._execution import prepare_forward, tensor_leaves
from ._formulas import (
    FLOP_COMPOSITES,
    FLOP_FREE_OPS,
    MAC_COMPOSITES,
    MAC_FORMULAS,
    MAC_FREE_OPS,
    arithmetic_formulas,
    biased_matrix_vector_flops,
    cudnn_convolution_flops,
    dot_flops,
    efficient_attention_flops,
    is_native_pointwise,
    matrix_extra_flops,
    matrix_vector_flops,
    slow_conv2d_flops,
)

_FORMULA_PREFIX = "torch_prisma.formula/"


def count_parameters(model: nn.Module) -> int:
    """Count elements in unique registered Parameters without executing the model.

    Tensor dimensions determine the count for all registered Parameters,
    including frozen, unused, and masked weights. Parameter identity
    deduplicates multiple registrations of the same tensor object.

    Args:
        model: Module whose registered parameters are counted.

    Returns:
        Total number of parameter elements.
    """
    return sum(parameter.numel() for parameter in model.parameters())


@dataclass(frozen=True)
class ModelComplexity:
    """Recorded FLOPs, dense MACs and unique registered parameter elements.

    Operation counts describe the known subtotal for the supplied batch and
    executed path under the documented counting conventions. One real
    floating-point dense MAC contributes two FLOPs. Supported bias,
    normalization and elementwise arithmetic contribute only to FLOPs.

    For each convention, `uncovered_ops` lists observed operations for which
    the counter cannot establish a complete count.

    Attributes:
        macs: Count of audited matrix and convolution multiply-accumulates.
        params: Count of unique registered Parameter elements.
        flops: Known nominal floating-point arithmetic under the registered formulas.
        uncovered_ops: Dictionary with `macs` and `flops` keys. Each value is a
            sorted tuple of unique operator names. Names can occur under both
            keys. The dictionary is mutable, although the record fields are
            frozen and the tuples are immutable.
    """

    macs: int
    params: int
    flops: int
    uncovered_ops: dict[str, tuple[str, ...]] = field(
        default_factory=lambda: {"macs": (), "flops": ()}
    )

    def to_dict(self) -> dict[str, Any]:
        """Return counts and coverage as JSON-ready data."""
        return {
            "macs": self.macs,
            "params": self.params,
            "flops": self.flops,
            "uncovered_ops": {key: list(ops) for key, ops in self.uncovered_ops.items()},
        }


@contextmanager
def _counting_attention() -> Iterator[None]:
    """Expose attention matrix work and restore public backend switches on exit."""
    fastpath = torch.backends.mha.get_fastpath_enabled()
    try:
        torch.backends.mha.set_fastpath_enabled(False)
        with sdpa_kernel(SDPBackend.MATH):
            yield
    finally:
        torch.backends.mha.set_fastpath_enabled(fastpath)


def _find_coverage_gaps(
    events: list[FunctionEvent], mac_free: frozenset[str]
) -> tuple[set[str], set[str]]:
    """Match each formula marker to one invocation and its backend children.

    A C++ composite can invoke an operation below Python dispatch. Follow
    profiler parents to distinguish hidden calls from separately counted calls.
    """
    # Native pointwise semantics prove the absence of dense matrix/convolution
    # MACs without implying a FLOP count. Inspect each observed packet once.
    mac_free = mac_free | {
        name
        for name in {event.name for event in events}
        if name not in mac_free and name not in MAC_FORMULAS and is_native_pointwise(name)
    }
    counted: dict[int, str] = {}
    recorded: set[str] = set()
    for event in events:
        if not event.name.startswith(_FORMULA_PREFIX):
            continue
        name = event.name.removeprefix(_FORMULA_PREFIX)
        recorded.add(name)
        parent = event.cpu_parent
        while parent is not None and parent.name != name:
            parent = parent.cpu_parent
        if (
            parent is None
            and not name.startswith("aten::")
            and event.cpu_parent is not None
            and event.cpu_parent.name == "PythonDispatchMode"
        ):
            # Python custom operators can lack their own profiler event. The
            # immediate dispatch scope still contains this call and its formula.
            parent = event.cpu_parent
        if parent is not None:
            counted[parent.id] = name
    mac_gaps = recorded - MAC_FORMULAS - mac_free
    flop_gaps: set[str] = set()
    mac_ignored = (mac_free | MAC_COMPOSITES) - mac_gaps
    flop_ignored = FLOP_FREE_OPS | MAC_COMPOSITES | FLOP_COMPOSITES
    for event in events:
        if "::" not in event.name or event.name.startswith(_FORMULA_PREFIX):
            continue
        mac_covered = flop_covered = False
        parent = event
        while parent is not None:
            if parent.id in counted:
                flop_covered = True
                mac_covered |= counted[parent.id] in MAC_FORMULAS
            parent = parent.cpu_parent
        if not mac_covered and event.name not in mac_ignored:
            mac_gaps.add(event.name)
        if not flop_covered and event.name not in flop_ignored:
            flop_gaps.add(event.name)
    return mac_gaps, flop_gaps


def calculate_model_complexity(
    target: Callable[..., Any],
    input_args: Any,
    device: torch.device | str | None = None,
    input_kwargs: dict[str, Any] | None = None,
    *,
    parameter_count: int | None = None,
) -> ModelComplexity:
    """Count eager inference FLOPs, MACs and parameters for one forward.

    Args:
        target: An eager module or a prepared callable. Modules are copied and
            evaluated in eval mode. Callables manage their own captured state.
        input_args: One positional input or a tuple/list of positional inputs.
        device: CPU, CUDA or Ascend NPU, inferred from model tensors, then inputs.
        input_kwargs: Optional keyword arguments.
        parameter_count: Override for registered parameters, for example the
            active size of a sampled subnet. A plain callable defaults to zero.

    Returns:
        Counts for the supplied batch and executed path, with coverage gaps
        listed in `uncovered_ops`. Counting uses math scaled dot-product
        attention and temporarily disables the multihead attention fast path.

    Raises:
        TypeError: The target is not callable.
        ValueError: The parameter override, input preparation, or a FLOP
            formula result is invalid.
        RuntimeError: The requested accelerator is unavailable.
    """
    if not callable(target):
        raise TypeError("target must be an nn.Module or callable")
    if parameter_count is not None and (type(parameter_count) is not int or parameter_count < 0):
        raise ValueError("parameter_count must be a nonnegative integer")
    if parameter_count is None:
        parameter_count = count_parameters(target) if isinstance(target, nn.Module) else 0

    # Turning off inference mode can re-enable gradients under an outer no_grad.
    with (
        torch.inference_mode(False),
        torch.no_grad(),
        prepare_forward(target, input_args, input_kwargs, device) as (forward, args, kwargs, _),
    ):
        # Preserve external non-MAC formulas. The mappings and validation
        # wrappers belong to this invocation, never the global registry.
        arithmetic = arithmetic_formulas()
        additions = {op: formula for op, formula in arithmetic.items() if op not in flop_registry}
        counter = FlopCounterMode(
            display=False,
            custom_mapping={
                **additions,
                torch.ops.aten.dot: dot_flops,
                torch.ops.aten.vdot: dot_flops,
                torch.ops.aten.mv: matrix_vector_flops,
                torch.ops.aten.addmv: biased_matrix_vector_flops,
                torch.ops.aten.addmv_: biased_matrix_vector_flops,
                torch.ops.aten.cudnn_convolution: cudnn_convolution_flops,
                torch.ops.aten._slow_conv2d_forward: slow_conv2d_flops,
            },
        )
        # Reuse the existing dense formulas for identical operand layouts.
        # Assign functions already wrapped by the registry directly.
        for alias, source in (
            (torch.ops.aten.addmm_, torch.ops.aten.addmm),
            (torch.ops.aten.baddbmm_, torch.ops.aten.baddbmm),
            (torch.ops.aten.addbmm, torch.ops.aten.baddbmm),
            (torch.ops.aten.addbmm_, torch.ops.aten.baddbmm),
            (
                torch.ops.aten._scaled_dot_product_flash_attention_for_cpu,
                torch.ops.aten._scaled_dot_product_flash_attention,
            ),
        ):
            counter.flop_registry[alias] = counter.flop_registry[source]
        counter.flop_registry[torch.ops.aten._efficient_attention_forward] = (
            efficient_attention_flops
        )
        macs = 0
        uncovered_flops: set[str] = set()

        def checked_formula(op: object, formula: Callable, added: bool) -> Callable:
            """Validate each invocation before aggregation and retain its dense MAC count."""
            name = str(op).replace(".", "::", 1)

            def record(*values: Any, out_val: Any = None, **options: Any) -> int:
                """Check the formula result and add separately defined non-MAC arithmetic."""
                nonlocal macs
                output_tensors = tensor_leaves(out_val)
                if added and any(t.is_complex() for t in tensor_leaves((values, options))):
                    uncovered_flops.add(name)
                    return 0
                if added and output_tensors and not output_tensors[0].is_floating_point():
                    if output_tensors[0].is_complex():
                        uncovered_flops.add(name)
                    with record_function(f"{_FORMULA_PREFIX}{name}"):
                        pass
                    return 0
                value = formula(*values, out_val=out_val, **options)
                if value is None and added:
                    uncovered_flops.add(name)
                    return 0
                is_mac = name in MAC_FORMULAS
                if type(value) is not int or value < 0 or (is_mac and value % 2):
                    expected = "nonnegative even" if is_mac else "nonnegative integer"
                    raise ValueError(f"{name}: expected a {expected} FLOP count")
                if (
                    name in ("aten::_scaled_mm", "aten::convolution_backward")
                    or (name.startswith("aten::") and "attention" in name)
                    or name in ("flex_attention", "flex_attention_backward")
                ):
                    # Native fused formulas omit arithmetic outside their
                    # matrix products, such as softmax or bias gradients.
                    uncovered_flops.add(name)
                if is_mac:
                    macs += value // 2
                    value += matrix_extra_flops(name, values, options, out_val)
                    if output_tensors and not output_tensors[0].is_floating_point():
                        if output_tensors[0].is_complex():
                            uncovered_flops.add(name)
                        # Count integer and complex MACs as logical products.
                        value = 0
                with record_function(f"{_FORMULA_PREFIX}{name}"):
                    pass
                return value

            return record

        counter.flop_registry = {
            op: checked_formula(op, formula, op in additions)
            for op, formula in counter.flop_registry.items()
        }
        with (
            profile(activities=[ProfilerActivity.CPU]) as trace,
            _counting_attention(),
            counter,
        ):
            forward(*args, **kwargs)
        mac_free = (
            MAC_FREE_OPS
            | FLOP_FREE_OPS
            | {str(op).replace("aten.", "aten::", 1) for op in arithmetic}
        )
        uncovered_macs, flop_gaps = _find_coverage_gaps(trace.events(), mac_free)
        uncovered_flops.update(flop_gaps)
        return ModelComplexity(
            macs=macs,
            params=parameter_count,
            flops=counter.get_total_flops(),
            uncovered_ops={
                "macs": tuple(sorted(uncovered_macs)),
                "flops": tuple(sorted(uncovered_flops)),
            },
        )
