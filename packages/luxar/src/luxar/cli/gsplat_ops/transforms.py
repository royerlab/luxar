"""``luxar gsplat transforms`` command registration + compatibility wrappers."""

from __future__ import annotations

from typing import Any

import typer

from .transforms_commands import (
    additive_dataset as _additive_dataset_cmd,
)
from .transforms_commands import cull_dataset as _cull_dataset_cmd
from .transforms_commands import filter_dataset as _filter_dataset_cmd
from .transforms_commands import flatten_dataset as _flatten_dataset_cmd
from .transforms_commands import merge_datasets as _merge_datasets_cmd
from .transforms_commands import partition_dataset as _partition_dataset_cmd
from .transforms_commands import slice_dataset as _slice_dataset_cmd
from .transforms_commands import transform_dataset as _transform_dataset_cmd
from .transforms_parsing import parse_bbox as _parse_bbox_impl
from .transforms_parsing import parse_csv_floats as _parse_csv_floats_impl
from .transforms_parsing import parse_slices as _parse_slices_impl


def cull_dataset(*args: Any, **kwargs: Any) -> None:
    """Back-compat wrapper around the cull command function."""
    return _cull_dataset_cmd(*args, **kwargs)


def _parse_bbox(s: str, ndim: int) -> list[tuple[float, float]]:
    """Back-compat wrapper around shared bbox parsing helper."""
    return _parse_bbox_impl(s, ndim)


def filter_dataset(*args: Any, **kwargs: Any) -> None:
    """Back-compat wrapper around the filter command function."""
    return _filter_dataset_cmd(*args, **kwargs)


def partition_dataset(*args: Any, **kwargs: Any) -> None:
    """Back-compat wrapper around the partition command function."""
    return _partition_dataset_cmd(*args, **kwargs)


def flatten_dataset(*args: Any, **kwargs: Any) -> None:
    """Back-compat wrapper around the flatten command function."""
    return _flatten_dataset_cmd(*args, **kwargs)


def additive_dataset(*args: Any, **kwargs: Any) -> None:
    """Back-compat wrapper around the additive command function."""
    return _additive_dataset_cmd(*args, **kwargs)


def _parse_slices(s: str, ndim: int) -> list[slice]:
    """Back-compat wrapper around shared slice parsing helper."""
    return _parse_slices_impl(s, ndim)


def slice_dataset(*args: Any, **kwargs: Any) -> None:
    """Back-compat wrapper around the slice command function."""
    return _slice_dataset_cmd(*args, **kwargs)


def _parse_csv_floats(value: str, expected: int, name: str) -> list[float]:
    """Back-compat wrapper around shared CSV float parsing helper."""
    return _parse_csv_floats_impl(value, expected, name)


def transform_dataset(*args: Any, **kwargs: Any) -> None:
    """Back-compat wrapper around the transform command function."""
    return _transform_dataset_cmd(*args, **kwargs)


def merge_datasets(*args: Any, **kwargs: Any) -> None:
    """Back-compat wrapper around the merge command function."""
    return _merge_datasets_cmd(*args, **kwargs)


def register_transforms_commands(app: typer.Typer) -> None:
    """Register the transform/edit commands onto ``app_gsplat``."""
    app.command("cull")(_cull_dataset_cmd)
    app.command("filter")(_filter_dataset_cmd)
    app.command("partition")(_partition_dataset_cmd)
    app.command("flatten")(_flatten_dataset_cmd)
    app.command("additive")(_additive_dataset_cmd)
    app.command("slice")(_slice_dataset_cmd)
    app.command("transform")(_transform_dataset_cmd)
    app.command("merge")(_merge_datasets_cmd)
