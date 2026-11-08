"""Eager and compiled inference latency on CPU, CUDA and Ascend."""

import gc
import math
import statistics
import time
from collections.abc import Callable, Iterable, Iterator
from contextlib import AbstractContextManager, ExitStack, contextmanager
from dataclasses import asdict, dataclass, field
from importlib import import_module
from types import ModuleType
from typing import Any

import torch

from .execution import call_arguments

_COMPILE_ITERS = 3
_UNLIMITED_RECOMPILES = 2**31 - 1


@dataclass(frozen=True)
class LatencyStats:
    """Summarize forward latency in milliseconds.

    Attributes:
        median_ms: Median sample duration.
        mean_ms: Arithmetic mean of the sample durations.
        std_ms: Population standard deviation of the sample durations.
        p90_ms: 90th percentile selected from the measured samples.
        p99_ms: 99th percentile selected from the measured samples.
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
        ordered = sorted(samples)
        count = len(ordered)
        return cls(
            median_ms=statistics.median(samples),
            mean_ms=statistics.fmean(samples),
            std_ms=statistics.pstdev(samples),
            p90_ms=ordered[min(int(count * 0.90), count - 1)],
            p99_ms=ordered[min(int(count * 0.99), count - 1)],
            min_ms=ordered[0],
            max_ms=ordered[-1],
            samples_ms=samples,
        )

    def to_dict(self) -> dict[str, Any]:
        """Return statistics and samples as JSON-ready data."""
        result = asdict(self)
        result["samples_ms"] = list(self.samples_ms)
        return result


def _timing_device(requested: torch.device | str) -> tuple[torch.device, ModuleType | None]:
    """Resolve the device and runtime used to record timing events."""
    if isinstance(requested, str) and requested.split(":", 1)[0] == "npu":
        import_module("torch_npu")
    device = torch.device(requested)
    if device.type == "cpu":
        return device, None
    if device.type not in ("cuda", "npu"):
        raise ValueError(f"Unsupported timing device {device}. Expected CPU, CUDA or Ascend NPU")
    runtime = getattr(torch, device.type)
    if device.index is None:
        device = torch.device(device.type, runtime.current_device())
    return device, runtime


def _unlimited_recompile_patch() -> AbstractContextManager[None]:
    """Raise Dynamo's recompile caps during the compilation forwards."""
    config = torch._dynamo.config
    # PyTorch 2.6 uses the cache_size_limit names.
    overrides = {
        name if hasattr(config, name) else legacy: _UNLIMITED_RECOMPILES
        for name, legacy in (
            ("recompile_limit", "cache_size_limit"),
            ("accumulated_recompile_limit", "accumulated_cache_size_limit"),
        )
    }
    return config.patch(**overrides)


@contextmanager
def _fail_on_late_compile() -> Iterator[None]:
    """Reject compilation during warmup or timing."""
    with torch.compiler.set_stance("fail_on_recompile"):
        try:
            yield
        except RuntimeError as error:
            if not str(error).startswith("Detected recompile when torch.compile stance"):
                raise
            raise RuntimeError(
                f"PyTorch requested compilation after {_COMPILE_ITERS} preparation forwards. "
                "The compiled latency measurement would be unreliable. "
                "Set compile=False to measure eager latency."
            ) from error


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


def measure_module_latency(
    target: Callable[..., Any],
    input_args: Any,
    device: torch.device | str,
    input_kwargs: dict[str, Any] | None = None,
    *,
    repetitions: int = 20,
    warmup: int = 5,
    compile: bool = True,
    compile_kwargs: dict[str, Any] | None = None,
) -> LatencyStats:
    """Measure one forward per sample for the entire supplied batch.

    Prepare the model mode and device placement before calling this function.
    Compilation preparation, warmup, and sampling all run the forward.
    Changes made by those calls persist. Pass a separate model instance to
    preserve the original model state.

    Args:
        target: A prepared module or callable.
        input_args: One positional input or a tuple/list of positional inputs.
        device: Device used for timing. CUDA and Ascend use its current stream.
        input_kwargs: Optional keyword arguments.
        repetitions: Positive number of timed forwards.
        warmup: Number of untimed forwards after any compilation preparation.
        compile: Use `torch.compile` and run three preparation forwards.
            Pass `False` for eager timing. Ascend selects TorchAir.
        compile_kwargs: Options forwarded to `torch.compile`, including `backend`.

    Returns:
        Timed forward durations and summary statistics in milliseconds.

    Raises:
        ValueError: The timing options or timing device are invalid.
        RuntimeError: The requested accelerator or default NPU backend is
            unavailable, or the forward requires compilation after preparation.
    """
    if type(repetitions) is not int or repetitions <= 0 or type(warmup) is not int or warmup < 0:
        raise ValueError("Require positive repetitions and nonnegative warmup")
    if type(compile) is not bool or (compile_kwargs is not None and not compile):
        raise ValueError("compile must be a boolean and compile_kwargs requires compile=True")

    args, kwargs = call_arguments(input_args, input_kwargs)
    device, runtime = _timing_device(device)
    forward = target
    with torch.inference_mode(), ExitStack() as timing:
        if compile:
            with _unlimited_recompile_patch():
                forward = torch.compile(target, **_compile_options(device, compile_kwargs))
                # A forward can initialize state and invalidate the first graph.
                for _ in range(_COMPILE_ITERS):
                    forward(*args, **kwargs)
                if runtime is not None:
                    runtime.synchronize(device)
            timing.enter_context(_fail_on_late_compile())
        for _ in range(warmup):
            forward(*args, **kwargs)

        events = []
        if runtime is not None:
            stream = runtime.current_stream(device)
            events = [
                (runtime.Event(enable_timing=True), runtime.Event(enable_timing=True))
                for _ in range(repetitions)
            ]
            # Event handles may be allocated lazily by the first record.
            for start, end in events:
                start.record(stream)
                end.record(stream)
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
                    start.record(stream)
                    forward(*args, **kwargs)
                    end.record(stream)
                runtime.synchronize(device)
                samples = [start.elapsed_time(end) for start, end in events]
        finally:
            if gc_enabled:
                gc.enable()
            else:
                gc.disable()

    return LatencyStats.from_samples(samples)
