"""Argument and tensor helpers shared by the measurement APIs."""

from typing import Any

import torch
from torch.utils._pytree import tree_leaves


def call_arguments(
    input_args: Any, input_kwargs: dict[str, Any] | None
) -> tuple[tuple[Any, ...], dict[str, Any]]:
    """Normalize positional and keyword arguments for a forward call."""
    args = tuple(input_args) if isinstance(input_args, (tuple, list)) else (input_args,)
    return args, {} if input_kwargs is None else input_kwargs


def tensor_leaves(value: Any) -> list[torch.Tensor]:
    """Find the tensors in a nested value."""
    return [leaf for leaf in tree_leaves(value) if isinstance(leaf, torch.Tensor)]
