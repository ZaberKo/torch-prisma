"""Eager and compiled inference latency on CPU, CUDA and Ascend."""

import gc
import math
import statistics
import time
from collections.abc import Callable, Iterable
from dataclasses import asdict, dataclass, field
from importlib import import_module
from typing import Any

import torch

from ._execution import accelerator_runtime, prepare_forward


@dataclass(frozen=True)
class LatencyStats:
    """Summarize forward latency in milliseconds.

    Attributes:
        median_ms: Median sample duration.
        mean_ms: Arithmetic mean of the sample durations.
        std_ms: Population standard deviation of the sample durations.
        p90_ms: Linearly interpolated 90th percentile.
        p99_ms: Linearly interpolated 99th percentile.
        min_ms: Shortest sample duration.
        max_ms: Longest sample duration.
        samples_ms: Sample durations in measurement order.
    """

    median_ms: float
    mean_ms: float
    std_ms: float
    p90_ms: float
    p99_ms: float
    min_ms: float
    max_ms: float
    samples_ms: tuple[float, ...] = field(repr=False)

    @classmethod
    def from_samples(cls, samples_ms: Iterable[float]) -> "LatencyStats":
        """Summarize timings in milliseconds.

        Args:
            samples_ms: A nonempty iterable of finite, nonnegative durations.

        Returns:
            Latency statistics and the original sample sequence.

        Raises:
            ValueError: The sequence is empty or contains an invalid duration.
        """
        samples = tuple(float(sample) for sample in samples_ms)
        if not samples or any(not math.isfinite(sample) or sample < 0 for sample in samples):
            raise ValueError("Latency samples must be nonempty, finite and nonnegative")
        quantiles = (
            statistics.quantiles(samples, n=100, method="inclusive") if len(samples) > 1 else ()
        )
        return cls(
            median_ms=statistics.median(samples),
            mean_ms=statistics.fmean(samples),
            std_ms=statistics.pstdev(samples),
            p90_ms=quantiles[89] if quantiles else samples[0],
            p99_ms=quantiles[98] if quantiles else samples[0],
            min_ms=min(samples),
            max_ms=max(samples),
            samples_ms=samples,
        )

    def to_dict(self) -> dict[str, Any]:
        """Return statistics and samples as JSON-ready data."""
        result = asdict(self)
        result["samples_ms"] = list(self.samples_ms)
        return result


def _compile_options(device: torch.device, options: dict[str, Any] | None) -> dict[str, Any]:
    """Select TorchAir for Ascend unless the caller supplied a backend."""
    options = dict(options or {})
    if device.type == "npu" and "backend" not in options:
        try:
            torchair = import_module("torchair")
        except ModuleNotFoundError as error:
            if error.name != "torchair":
                raise
            raise RuntimeError(
                "Compiled Ascend latency requires a compatible TorchAir installation"
            ) from error
        options["backend"] = torchair.get_npu_backend()
    return options


@torch.inference_mode()
def measure_module_latency(
    target: Callable[..., Any],
    input_args: Any,
    device: torch.device | str | None = None,
    input_kwargs: dict[str, Any] | None = None,
    *,
    repetitions: int = 20,
    warmup: int = 5,
    compile: bool = False,
    compile_kwargs: dict[str, Any] | None = None,
) -> LatencyStats:
    """Measure one forward per sample for the entire supplied batch.

    Args:
        target: An eager module or prepared callable. Modules are copied and
            evaluated in eval mode. Callables manage their own captured state.
        input_args: One positional input or a tuple/list of positional inputs.
        device: CPU, CUDA or Ascend NPU, inferred from model tensors, then inputs.
        input_kwargs: Optional keyword arguments.
        repetitions: Positive number of timed forwards.
        warmup: Number of untimed forwards after any compilation preparation.
        compile: Use `torch.compile`. Ascend selects TorchAir by default.
        compile_kwargs: Options forwarded to `torch.compile`, including `backend`.

    Returns:
        Timed forward durations and summary statistics in milliseconds.

    Raises:
        TypeError: The target is not callable.
        ValueError: The timing options are invalid or the measurement inputs
            cannot be prepared on the selected device.
        RuntimeError: The requested accelerator or default NPU backend is
            unavailable.
    """
    if type(repetitions) is not int or repetitions <= 0 or type(warmup) is not int or warmup < 0:
        raise ValueError("Require positive repetitions and nonnegative warmup")
    if type(compile) is not bool or (compile_kwargs is not None and not compile):
        raise ValueError("compile must be a boolean and compile_kwargs requires compile=True")

    with prepare_forward(target, input_args, input_kwargs, device) as (
        forward,
        args,
        kwargs,
        device,
    ):
        if compile:
            forward = torch.compile(forward, **_compile_options(device, compile_kwargs))
            forward(*args, **kwargs)
        for _ in range(warmup):
            forward(*args, **kwargs)

        runtime = accelerator_runtime(device)
        events = []
        if runtime is not None:
            events = [
                (runtime.Event(enable_timing=True), runtime.Event(enable_timing=True))
                for _ in range(repetitions)
            ]
            # Event handles may be allocated lazily by the first record.
            for start, end in events:
                start.record()
                end.record()
            runtime.synchronize(device)

        gc_enabled = gc.isenabled()
        gc.disable()
        try:
            if runtime is None:
                samples = []
                for _ in range(repetitions):
                    start_time = time.perf_counter()
                    forward(*args, **kwargs)
                    samples.append((time.perf_counter() - start_time) * 1000.0)
            else:
                for start, end in events:
                    start.record()
                    forward(*args, **kwargs)
                    end.record()
                runtime.synchronize(device)
                samples = [start.elapsed_time(end) for start, end in events]
        finally:
            if gc_enabled:
                gc.enable()
            else:
                gc.disable()

    return LatencyStats.from_samples(samples)
