import os
import sys
from contextlib import contextmanager
from types import SimpleNamespace

import pytest
import torch
from torch import nn

import torch_prisma.latency as latency
from torch_prisma import calculate_model_complexity, measure_module_latency


def test_ascend_compiler_and_event_wiring(monkeypatch):
    device = SimpleNamespace(type="npu")
    operations = []

    class Event:
        def __init__(self, *, enable_timing):
            assert enable_timing

        def record(self):
            operations.append("record")

        def elapsed_time(self, end):
            assert isinstance(end, Event)
            return 0.25

    @contextmanager
    def prepare(target, args, kwargs, requested):
        assert requested == "npu:0"
        yield target, (args,), {}, device

    def synchronize(selected):
        assert selected is device
        operations.append("sync")

    def compile_fn(fn, *, backend, fullgraph):
        assert backend == "torchair-test" and fullgraph
        return fn

    def forward(x):
        operations.append("forward")
        return x

    monkeypatch.setitem(
        sys.modules, "torchair", SimpleNamespace(get_npu_backend=lambda: "torchair-test")
    )
    monkeypatch.setattr(latency, "prepare_forward", prepare)
    monkeypatch.setattr(
        latency,
        "accelerator_runtime",
        lambda _: SimpleNamespace(Event=Event, synchronize=synchronize),
    )
    monkeypatch.setattr(torch, "compile", compile_fn)
    result = measure_module_latency(
        forward,
        torch.ones(1),
        "npu:0",
        repetitions=2,
        warmup=1,
        compile=True,
        compile_kwargs={"fullgraph": True},
    )
    assert result.samples_ms == (0.25, 0.25)
    assert operations == (
        ["forward"] * 2
        + ["record"] * 4
        + ["sync", "record", "forward", "record", "record", "forward", "record", "sync"]
    )


@pytest.mark.cuda
@pytest.mark.skipif(os.environ.get("TORCH_PRISMA_TEST_DEVICE") != "cuda", reason="Requires CUDA")
def test_cuda_measurements():
    assert torch.cuda.is_available()
    device = torch.device("cuda", torch.cuda.current_device())
    model = nn.Linear(4, 3, device=device)
    x = torch.ones(2, 4)
    stats = measure_module_latency(model, x, warmup=1, repetitions=3)
    assert len(stats.samples_ms) == 3
    assert stats.median_ms > 0
    complexity = calculate_model_complexity(model, x)
    assert (complexity.params, complexity.macs, complexity.flops) == (15, 24, 54)
    with pytest.raises(ValueError, match="shared tensor storage"):
        measure_module_latency(lambda a, b: a + b, (x, x.view_as(x)), device)


@pytest.mark.ascend
@pytest.mark.skipif(
    os.environ.get("TORCH_PRISMA_TEST_DEVICE") != "ascend", reason="Requires Ascend"
)
def test_ascend_measurements():
    __import__("torch_npu")
    assert torch.npu.is_available()
    device = torch.device("npu", torch.npu.current_device())
    model = nn.Linear(4, 3, device=device)
    x = torch.ones(2, 4, device=device)
    for compile in (False, True):
        stats = measure_module_latency(model, x, warmup=2, repetitions=3, compile=compile)
        assert len(stats.samples_ms) == 3
        assert stats.median_ms > 0

    complexity = calculate_model_complexity(model, x)
    assert complexity.params == 15
    assert complexity.macs == 24 or complexity.uncovered_ops["macs"]
    assert complexity.flops == 54 or complexity.uncovered_ops["flops"]
