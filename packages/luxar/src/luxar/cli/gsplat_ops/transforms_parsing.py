"""Shared parsing helpers for gsplat edit-style commands."""

from __future__ import annotations

import typer


def parse_bbox(s: str, ndim: int) -> list[tuple[float, float]]:
    """Parse ``'min0,max0,min1,max1,...'`` into ``[(min, max), ...]``."""
    parts = [float(x.strip()) for x in s.split(",")]
    if len(parts) != 2 * ndim:
        raise typer.BadParameter(
            f"bbox needs {2 * ndim} values for {ndim}D data, got {len(parts)}"
        )
    pairs = [(parts[2 * i], parts[2 * i + 1]) for i in range(ndim)]
    for i, (lo, hi) in enumerate(pairs):
        if lo > hi:
            raise typer.BadParameter(f"bbox dimension {i} has min ({lo}) > max ({hi})")
    return pairs


def parse_slices(s: str, ndim: int) -> list[slice]:
    """Parse numpy-style range string into a per-dimension slice list."""
    parts = [p.strip() for p in s.split(",")]
    if len(parts) != ndim:
        raise typer.BadParameter(
            f"Expected {ndim} ranges for {ndim}D data, got {len(parts)}"
        )
    slices = []
    for part in parts:
        if ":" not in part:
            raise typer.BadParameter(
                f"Invalid range '{part}': expected 'lo:hi', 'lo:', ':hi', or ':'"
            )
        lo_str, hi_str = part.split(":", 1)
        lo = float(lo_str.strip()) if lo_str.strip() else None
        hi = float(hi_str.strip()) if hi_str.strip() else None
        slices.append(slice(lo, hi))
    return slices


def parse_csv_floats(value: str, expected: int, name: str) -> list[float]:
    """Parse comma-separated float values and validate expected arity."""
    parts = [p.strip() for p in value.split(",")]
    if len(parts) != expected:
        raise typer.BadParameter(
            f"--{name} expects {expected} comma-separated values (one per dimension), "
            f"got {len(parts)}: '{value}'"
        )
    try:
        return [float(p) for p in parts]
    except ValueError as e:
        raise typer.BadParameter(f"--{name} values must be numbers: {e}") from e
