import gc
import json
from dataclasses import is_dataclass

import pytest
import torch
from torch import nn
from torch.nn import functional as F

from torch_prisma import (
    LatencyStats,
    ModelComplexity,
    calculate_model_complexity,
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


def test_prepared_callable_and_active_parameter_count():
    weight = torch.ones(4, 2)

    def subnet(x, *, scale):
        return torch.mm(x, weight) * scale

    x = torch.ones(3, 4)
    kwargs = {"scale": 2}
    result = calculate_model_complexity(subnet, x, input_kwargs=kwargs, parameter_count=8)
    assert result == ModelComplexity(macs=24, params=8, flops=54)
    assert calculate_model_complexity(subnet, x, input_kwargs=kwargs).params == 0
    with pytest.raises(ValueError, match="parameter_count"):
        calculate_model_complexity(subnet, x, input_kwargs=kwargs, parameter_count=-1)


def test_attention_counts_matrix_work():
    q = torch.randn(2, 3, 4, 8)
    k = torch.randn_like(q)
    v = torch.randn(2, 3, 4, 6)
    result = calculate_model_complexity(F.scaled_dot_product_attention, (q, k, v))
    assert result.macs == 2 * 3 * 4 * 4 * (8 + 6)
    assert result.uncovered_ops["macs"] == ()


@pytest.mark.parametrize("measure", [measure_module_latency, calculate_model_complexity])
@pytest.mark.parametrize("fails", [False, True])
def test_module_inputs_buffers_and_rng_are_isolated(measure, fails):
    class Model(nn.Module):
        def __init__(self):
            super().__init__()
            self.register_buffer("buffer", torch.zeros(4))
            self.view = self.buffer.view(4)
            self.train()

        def forward(self, x, *, nested):
            assert not self.training
            assert x is nested["value"][0] is self.buffer
            x.add_(1)
            torch.testing.assert_close(self.view, x)
            torch.rand(())
            if fails:
                raise RuntimeError("forward failed")
            return x

    model = Model()
    original = model.buffer
    rng = torch.get_rng_state()
    fastpath = torch.backends.mha.get_fastpath_enabled()
    gc_state = gc.isenabled()
    call_kwargs = {"repetitions": 1, "warmup": 0} if measure is measure_module_latency else {}
    kwargs = {"nested": {"value": [original]}}
    if fails:
        with pytest.raises(RuntimeError, match="forward failed"):
            measure(model, original, input_kwargs=kwargs, **call_kwargs)
    else:
        measure(model, original, input_kwargs=kwargs, **call_kwargs)
    assert model.training and model.buffer is original
    assert torch.equal(original, torch.zeros_like(original))
    assert torch.equal(torch.get_rng_state(), rng)
    assert torch.backends.mha.get_fastpath_enabled() == fastpath
    assert gc.isenabled() == gc_state


def test_latency_stats_preserve_sample_order_and_define_percentiles():
    stats = LatencyStats.from_samples([4, 1, 3, 2])
    assert is_dataclass(stats)
    assert stats.samples_ms == (4, 1, 3, 2)
    assert stats.median_ms == stats.mean_ms == 2.5
    assert stats.std_ms == pytest.approx(1.25**0.5)
    assert stats.p90_ms == pytest.approx(3.7)
    assert stats.p99_ms == pytest.approx(3.97)
    assert (stats.min_ms, stats.max_ms) == (1, 4)
    assert stats.to_dict()["samples_ms"] == [4, 1, 3, 2]
    single = LatencyStats.from_samples([2])
    assert single.p90_ms == single.p99_ms == 2
    assert single.std_ms == 0


@pytest.mark.parametrize("samples", [[], [-1], [float("nan")], [float("inf")]])
def test_invalid_samples(samples):
    with pytest.raises(ValueError, match="Latency samples"):
        LatencyStats.from_samples(samples)


@pytest.mark.parametrize("settings", [{"repetitions": 0}, {"warmup": -1}, {"compile_kwargs": {}}])
def test_invalid_timing_options(settings):
    with pytest.raises(ValueError):
        measure_module_latency(nn.Identity(), torch.ones(1), **settings)


def test_cpu_compiled_latency():
    result = measure_module_latency(
        nn.Linear(4, 3),
        torch.ones(2, 4),
        repetitions=3,
        warmup=1,
        compile=True,
        compile_kwargs={"backend": "eager", "fullgraph": True},
    )
    assert len(result.samples_ms) == 3
    assert result.median_ms > 0
