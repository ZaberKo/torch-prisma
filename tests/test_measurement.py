import json
from dataclasses import is_dataclass

import pytest
import torch
from torch import nn
from torch.nn import functional as F

from torch_prisma import (
    LatencyStats,
    ModelComplexity,
    OperationCounts,
    calculate_model_complexity,
    count_operations,
    count_parameters,
    measure_module_latency,
)


def test_parameter_count_and_linear_arithmetic():
    model = nn.Linear(4, 5)
    model.weight.requires_grad_(False)
    model.register_parameter("alias", model.weight)
    model.register_buffer("buffer", torch.zeros(10))
    with torch.inference_mode():
        result = calculate_model_complexity(model, torch.ones(2, 3, 4))
    assert is_dataclass(result)
    assert count_parameters(model) == 25
    assert result == ModelComplexity(macs=120, params=25, flops=270)
    assert json.loads(json.dumps(result.to_dict()))["uncovered_ops"] == {"macs": [], "flops": []}


def test_grouped_conv_bias_and_unknown_activation():
    model = nn.Sequential(nn.Conv2d(4, 6, 3, padding=1, groups=2), nn.GELU())
    result = calculate_model_complexity(model, torch.ones(2, 4, 5, 5))
    assert result.macs == 2 * 6 * 5 * 5 * 2 * 3 * 3
    assert result.flops == 2 * result.macs + 2 * 6 * 5 * 5
    assert result.params == 6 * 2 * 3 * 3 + 6
    assert result.uncovered_ops == {"macs": (), "flops": ("aten::gelu",)}


def test_count_operations_for_prepared_callable():
    weight = torch.ones(4, 2)

    def subnet(x, *, scale):
        return torch.mm(x, weight) * scale

    x = torch.ones(3, 4)
    assert count_operations(lambda: subnet(x, scale=2)) == OperationCounts(
        macs=24, flops=54, uncovered_ops={"macs": (), "flops": ()}
    )
    assert calculate_model_complexity(subnet, x, {"scale": 2}) == ModelComplexity(
        macs=24, params=0, flops=54
    )


def test_attention_counts_matrix_work():
    q = torch.randn(2, 3, 4, 8)
    k = torch.randn_like(q)
    v = torch.randn(2, 3, 4, 6)
    result = calculate_model_complexity(F.scaled_dot_product_attention, (q, k, v))
    assert result.macs == 2 * 3 * 4 * 4 * (8 + 6)
    assert result.uncovered_ops["macs"] == ()


@pytest.mark.parametrize("measure", [measure_module_latency, calculate_model_complexity])
def test_measurement_uses_the_supplied_model_mode_and_inputs(measure):
    class Model(nn.Module):
        def __init__(self):
            super().__init__()
            self.register_buffer("total", torch.zeros(1))

        def forward(self, x, *, nested):
            assert self.training
            assert x is nested["input"][0]
            self.total.add_(x)
            x.add_(1)
            return self.total

    model = Model()
    x = torch.ones(1)
    options = (
        {"device": "cpu", "repetitions": 1, "warmup": 0, "compile": False}
        if measure is measure_module_latency
        else {}
    )
    measure(model, x, input_kwargs={"nested": {"input": [x]}}, **options)
    assert model.training
    torch.testing.assert_close(model.total, torch.ones(1))
    torch.testing.assert_close(x, torch.full((1,), 2.0))


def test_latency_requires_an_explicit_timing_device():
    with pytest.raises(TypeError, match="device"):
        measure_module_latency(nn.Identity(), torch.ones(1))


def test_latency_stats_preserve_sample_order_and_define_percentiles():
    stats = LatencyStats.from_samples([4, 1, 3, 2])
    assert is_dataclass(stats)
    assert stats.samples_ms == (4, 1, 3, 2)
    assert stats.median_ms == stats.mean_ms == 2.5
    assert stats.std_ms == pytest.approx(1.25**0.5)
    assert stats.p90_ms == stats.p99_ms == 4
    assert (stats.min_ms, stats.max_ms) == (1, 4)
    assert stats.to_dict()["samples_ms"] == [4, 1, 3, 2]
    single = LatencyStats.from_samples([2])
    assert single.p90_ms == single.p99_ms == 2
    assert single.std_ms == 0


@pytest.mark.parametrize("samples", [[], [-1], [float("nan")], [float("inf")]])
def test_invalid_samples(samples):
    with pytest.raises(ValueError, match="Latency samples"):
        LatencyStats.from_samples(samples)


@pytest.mark.parametrize(
    "settings", [{"repetitions": 0}, {"warmup": -1}, {"compile": False, "compile_kwargs": {}}]
)
def test_invalid_timing_options(settings):
    with pytest.raises(ValueError):
        measure_module_latency(nn.Identity(), torch.ones(1), "cpu", **settings)


def test_cpu_latency_compiles_by_default():
    result = measure_module_latency(
        nn.Linear(4, 3),
        torch.ones(2, 4),
        "cpu",
        repetitions=3,
        warmup=1,
        compile_kwargs={"backend": "eager", "fullgraph": True},
    )
    assert len(result.samples_ms) == 3
    assert result.median_ms > 0


@pytest.mark.parametrize("warmup", [0, 1])
def test_latency_rejects_compilation_after_preparation(warmup):
    class Stateful(nn.Module):
        def __init__(self):
            super().__init__()
            self.step = 0

        def forward(self, x):
            self.step += 1
            return x + self.step

    with pytest.raises(RuntimeError, match=r"after 3 preparation forwards.*compile=False"):
        measure_module_latency(
            Stateful(),
            torch.ones(1),
            "cpu",
            repetitions=1,
            warmup=warmup,
            compile_kwargs={"backend": "eager", "dynamic": False},
        )
