"""Input preparation and state isolation shared by both measurements."""

import copy
from collections.abc import Callable, Iterator
from contextlib import contextmanager, nullcontext
from importlib import import_module
from threading import RLock
from types import ModuleType
from typing import Any

import torch
from torch import nn

# The profiler, default RNGs and measurement's backend switches share state.
_measurement_lock = RLock()


def tensor_leaves(value: Any) -> list[torch.Tensor]:
    """Find tensors inside nested tuples, lists and dictionaries."""
    tensors: list[torch.Tensor] = []
    seen: set[int] = set()

    def visit(item: Any) -> None:
        if isinstance(item, torch.Tensor):
            tensors.append(item)
        elif isinstance(item, (tuple, list, dict)) and id(item) not in seen:
            seen.add(id(item))
            children = item.values() if isinstance(item, dict) else item
            for child in children:
                visit(child)

    visit(value)
    return tensors


def _storage_key(tensor: torch.Tensor) -> tuple[torch.device, int] | None:
    """Identify allocated storage for dense, non-quantized tensors."""
    if tensor.layout == torch.strided and not tensor.is_quantized:
        storage = tensor.untyped_storage()
        if storage.nbytes():
            return tensor.device, storage.data_ptr()
    return None


def accelerator_runtime(device: torch.device) -> ModuleType | None:
    """Return the device module for an accelerator, or None for CPU."""
    if device.type in ("cuda", "npu"):
        return getattr(torch, device.type)
    return None


def _resolve_device(
    requested: torch.device | str | None, tensors: list[torch.Tensor]
) -> torch.device:
    """Resolve a supported device and select an explicit accelerator index."""
    if isinstance(requested, str) and requested.split(":", 1)[0] == "npu":
        # Older PyTorch versions recognize the device name only after registration.
        try:
            import_module("torch_npu")
        except ModuleNotFoundError as error:
            if error.name != "torch_npu":
                raise
            raise RuntimeError(
                "Ascend measurement requires a compatible torch_npu installation"
            ) from error
    device = torch.device(
        requested if requested is not None else tensors[0].device if tensors else "cpu"
    )
    if device.type == "cpu":
        return torch.device("cpu")
    runtime = accelerator_runtime(device)
    if runtime is None:
        raise ValueError(f"Unsupported device {device}. Expected CPU, CUDA or Ascend NPU")
    if not runtime.is_available():
        raise RuntimeError(f"{device.type.upper()} is unavailable")
    index = runtime.current_device() if device.index is None else device.index
    return torch.device(device.type, index)


def _copy_call(
    model: nn.Module | None,
    args: tuple[Any, ...],
    kwargs: dict[str, Any],
    device: torch.device,
) -> tuple[nn.Module | None, tuple[Any, ...], dict[str, Any]]:
    """Copy a model and its inputs together to preserve supported aliases."""
    inputs = tensor_leaves((args, kwargs))
    model_tensors = (
        tensor_leaves([vars(module) for module in model.modules()]) if model is not None else []
    )
    originals = {id(tensor): tensor for tensor in (*model_tensors, *inputs)}
    model_ids = {id(tensor) for tensor in model_tensors}
    memo: dict[int, Any] = {}
    conjugates = []

    with torch.inference_mode(False), torch.no_grad():
        try:
            for identity, tensor in originals.items():
                if tensor.device != device and identity not in model_ids:
                    # Validation below checks that transfer preserves aliasing.
                    memo[identity] = tensor.to(device).requires_grad_(tensor.requires_grad)
                elif (
                    type(tensor) in (torch.Tensor, nn.Parameter)
                    and tensor.is_conj()
                    and tensor.is_leaf
                ):
                    # Deepcopy otherwise resolves lazy conjugation and loses aliases.
                    copied = copy.deepcopy(tensor.detach().conj(), memo).conj()
                    if isinstance(tensor, nn.Parameter):
                        copied = nn.Parameter(copied, requires_grad=tensor.requires_grad)
                    else:
                        copied.requires_grad_(tensor.requires_grad)
                    memo[identity] = copied
                    conjugates.append((tensor, copied))
            for tensor, copied in conjugates:
                copied.__dict__ = copy.deepcopy(vars(tensor), memo)
                copied.grad = copy.deepcopy(tensor.grad, memo)
            copied_model, copied_args, copied_kwargs = copy.deepcopy((model, args, kwargs), memo)
        except Exception as error:
            raise ValueError("Cannot copy the model and inputs for isolated measurement") from error

    sources: dict[object, object] = {}
    copies: dict[object, object] = {}
    original_storages = {_storage_key(tensor) for tensor in originals.values()}
    for identity, original in originals.items():
        copied = memo.get(identity)
        if not isinstance(copied, torch.Tensor) or original is copied:
            raise ValueError("Measurement requires an independent copy of each tensor")
        if original.is_conj() != copied.is_conj() or original.is_neg() != copied.is_neg():
            raise ValueError("Measurement cannot preserve a lazy conjugate or negative tensor flag")
        source, destination = _storage_key(original), _storage_key(copied)
        if source is not None and (
            destination is None
            or destination in original_storages
            or sources.setdefault(source, destination) != destination
            or copies.setdefault(destination, source) != source
        ):
            raise ValueError(
                "Measurement cannot preserve shared tensor storage during copy or transfer"
            )
    return copied_model, copied_args, copied_kwargs


@contextmanager
def prepare_forward(
    target: Callable[..., Any],
    input_args: Any,
    input_kwargs: dict[str, Any] | None,
    requested_device: torch.device | str | None,
) -> Iterator[tuple[Callable[..., Any], tuple[Any, ...], dict[str, Any], torch.device]]:
    """Prepare an isolated module or a caller-managed callable on one device.

    Args:
        target: Module or callable to execute.
        input_args: One input or a tuple or list of positional arguments.
        input_kwargs: Optional keyword arguments.
        requested_device: Device override for the measurement.

    Yields:
        Prepared callable, positional arguments, keyword arguments, and
        resolved device. The context restores the default RNG state on exit.
    """
    if not callable(target):
        raise TypeError("target must be an nn.Module or callable")
    model = target if isinstance(target, nn.Module) else None
    args = tuple(input_args) if isinstance(input_args, (tuple, list)) else (input_args,)
    kwargs = {} if input_kwargs is None else input_kwargs
    registered = [*model.parameters(), *model.buffers()] if model is not None else []
    device = _resolve_device(requested_device, registered or tensor_leaves((args, kwargs)))
    if any(tensor.device != device for tensor in registered):
        raise ValueError("Place all model parameters and buffers on the measurement device first")

    runtime = accelerator_runtime(device)
    context = runtime.device(device) if runtime is not None else nullcontext()
    rng_devices = [device.index] if runtime is not None else []
    rng_type = device.type if runtime is not None else "cuda"
    with (
        _measurement_lock,
        context,
        torch.random.fork_rng(devices=rng_devices, device_type=rng_type),
    ):
        copied_model, args, kwargs = _copy_call(model, args, kwargs, device)
        if copied_model is not None:
            target = copied_model.eval()
        yield target, args, kwargs, device
