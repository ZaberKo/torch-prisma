import os
import sys
from types import SimpleNamespace
from unittest.mock import Mock

import pytest
import torch
from torch import nn

import torch_prisma.latency as latency
from torch_prisma import calculate_model_complexity, measure_module_latency


def test_ascend_compiler_and_event_wiring(monkeypatch):
    device = SimpleNamespace(type="npu")
    stream = object()
    operations = []
    forward_limits = []
    config = torch._dynamo.config
    limit_name = "recompile_limit" if hasattr(config, "recompile_limit") else "cache_size_limit"
    accumulated_limit_name = f"accumulated_{limit_name}"
    original_limits = getattr(config, limit_name), getattr(config, accumulated_limit_name)

    class Event:
        def __init__(self, *, enable_timing):
            assert enable_timing

        def record(self, selected_stream):
            assert selected_stream is stream
            operations.append("record")

        def elapsed_time(self, end):
            assert isinstance(end, Event)
            return 0.25

    def current_stream(selected):
        assert selected is device
        return stream

    def synchronize(selected):
        assert selected is device
        operations.append("sync")

    runtime = SimpleNamespace(Event=Event, current_stream=current_stream, synchronize=synchronize)

    def timing_device(requested):
        assert requested == "npu:0"
        return device, runtime

    def compile_fn(fn, *, backend, fullgraph):
        assert backend == "torchair-test" and fullgraph
        return fn

    def forward(x):
        operations.append("forward")
        forward_limits.append(
            (getattr(config, limit_name), getattr(config, accumulated_limit_name))
        )
        return x

    backend_factory = Mock(return_value="torchair-test")
    monkeypatch.setitem(sys.modules, "torchair", SimpleNamespace(get_npu_backend=backend_factory))
    monkeypatch.setattr(latency, "_timing_device", timing_device)
    monkeypatch.setattr(torch, "compile", compile_fn)
    latency._default_npu_backend.cache_clear()
    try:
        assert latency._compile_options(device, {"backend": "custom"}) == {"backend": "custom"}
        backend_factory.assert_not_called()
        result = measure_module_latency(
            forward,
            torch.ones(1),
            "npu:0",
            repetitions=2,
            warmup=1,
            compile_kwargs={"fullgraph": True},
        )
        assert latency._compile_options(device, None) == {"backend": "torchair-test"}
        backend_factory.assert_called_once_with()
    finally:
        latency._default_npu_backend.cache_clear()
    assert result.samples_ms == (0.25, 0.25)
    assert operations == (
        ["forward", "forward", "forward", "sync", "forward"]
        + ["record"] * 4
        + ["sync", "record", "forward", "record", "record", "forward", "record", "sync"]
    )
    raised_limits = (2**31 - 1, 2**31 - 1)
    assert forward_limits == [raised_limits] * 3 + [original_limits] * 3
    assert (getattr(config, limit_name), getattr(config, accumulated_limit_name)) == original_limits


@pytest.mark.cuda
@pytest.mark.skipif(os.environ.get("TORCH_PRISMA_TEST_DEVICE") != "cuda", reason="Requires CUDA")
def test_cuda_measurements():
    assert torch.cuda.is_available()
    device = torch.device("cuda", torch.cuda.current_device())
    model = nn.Linear(4, 3, device=device).eval()
    x = torch.ones(2, 4, device=device)
    stats = measure_module_latency(model, x, device, warmup=1, repetitions=3, compile=False)
    assert len(stats.samples_ms) == 3
    assert stats.median_ms > 0
    complexity = calculate_model_complexity(model, x)
    assert (complexity.params, complexity.macs, complexity.flops) == (15, 24, 54)


@pytest.mark.ascend
@pytest.mark.skipif(
    os.environ.get("TORCH_PRISMA_TEST_DEVICE") != "ascend", reason="Requires Ascend"
)
def test_ascend_measurements():
    __import__("torch_npu")
    assert torch.npu.is_available()
    device = torch.device("npu", torch.npu.current_device())
    model = nn.Linear(4, 3, device=device).eval()
    x = torch.ones(2, 4, device=device)
    for compile in (False, True):
        stats = measure_module_latency(model, x, device, warmup=2, repetitions=3, compile=compile)
        assert len(stats.samples_ms) == 3
        assert stats.median_ms > 0

    complexity = calculate_model_complexity(model, x)
    assert complexity.params == 15
    assert complexity.macs == 24 or complexity.uncovered_ops["macs"]
    assert complexity.flops == 54 or complexity.uncovered_ops["flops"]


@pytest.mark.ascend
@pytest.mark.skipif(
    os.environ.get("TORCH_PRISMA_TEST_ASCEND_INDUCTOR") != "1",
    reason="Requires Ascend with Triton Inductor",
)
def test_ascend_triton_inductor_measurement():
    __import__("torch_npu")
    device = torch.device("npu", torch.npu.current_device())
    x = torch.randn(8192, device=device)

    def forward(x):
        return torch.sin(x) + x * x

    stats = measure_module_latency(
        forward, x, device, warmup=2, repetitions=3, compile_kwargs={"backend": "inductor"}
    )
    assert len(stats.samples_ms) == 3
    assert stats.median_ms > 0
    torch.testing.assert_close(torch.compile(forward, backend="inductor")(x), forward(x))
