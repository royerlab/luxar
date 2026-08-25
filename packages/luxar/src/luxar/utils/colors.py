"""Color conversion and stacking helpers for Luxar demos."""

from __future__ import annotations

from typing import Any, NamedTuple, Optional, Sequence

import numpy as np


def hsv_to_rgb(h: np.ndarray, s: Any = 1.0, v: Any = 1.0) -> np.ndarray:
    """Vectorized HSV→RGB for arrays of hues (all in [0, 1]).

    A single shared implementation for the rainbow/hue-ramp colouring several
    demos each re-derived by hand. ``h`` is an array (or scalar); ``s``/``v`` may
    be scalars or broadcastable arrays.

    Returns:
        ``(..., 3)`` float32 RGB in [0, 1] with the same leading shape as ``h``.
    """
    h = np.asarray(h, dtype=np.float32)
    s = np.asarray(s, dtype=np.float32)
    v = np.asarray(v, dtype=np.float32)
    hp = (h % 1.0) * 6.0
    c = v * s
    x = c * (1.0 - np.abs(hp % 2.0 - 1.0))
    m = v - c
    z = np.zeros_like(hp)
    sector = np.floor(hp).astype(int) % 6
    r = np.select(
        [sector == 0, sector == 1, sector == 2, sector == 3, sector == 4, sector == 5],
        [c, x, z, z, x, c],
    )
    g = np.select(
        [sector == 0, sector == 1, sector == 2, sector == 3, sector == 4, sector == 5],
        [x, c, c, x, z, z],
    )
    b = np.select(
        [sector == 0, sector == 1, sector == 2, sector == 3, sector == 4, sector == 5],
        [z, z, x, c, c, x],
    )
    rgb = np.stack([r + m, g + m, b + m], axis=-1)
    return rgb.astype(np.float32)


class StackedColorings(NamedTuple):
    """Result of :func:`stack_colorings` — a point cloud replicated once per
    coloring scheme along a leading categorical axis."""

    positions: np.ndarray  # (N*K, D+1) float32; column 0 is the coloring index
    colors: np.ndarray  # (N*K, 3) float32
    labels: Optional[list[str]]  # (N*K,) hover labels, or None if any view lacks them
    categories: list[str]  # K coloring category names (for the `coloring` Dimension)
    keys: Optional[list[str]] = None  # (N*K,) machine-readable keys, or None


def stack_colorings(
    coords: np.ndarray,
    colorings: list[dict],
    keys: Optional[Sequence[str]] = None,
) -> StackedColorings:
    """Replicate a point cloud once per coloring scheme along a categorical axis.

    This is the render-proven pattern used by the census / peak-UMAP demos to
    give a single embedding several switchable color views: the N points are
    stacked K times (one block per coloring) and a leading ``coloring`` index
    column is prepended so a categorical ``coloring`` dimension can select which
    block (which colour scheme) is shown. Concentrating the stacking here keeps
    the per-point alignment of positions / colours / hover-labels correct and
    tested in one place instead of hand-rolled in each demo.

    Args:
        coords: ``(N, D)`` spatial coordinates (``D`` is usually 3).
        colorings: ordered list of dicts, one per color scheme. Each dict has:

            - ``"label"``: category name shown on the ``coloring`` dimension.
            - ``"colors"``: ``(N, 3)`` float RGB for this scheme.
            - ``"labels"`` (optional): ``(N,)`` per-point hover strings for this
              scheme. If ANY coloring omits labels, the combined ``labels`` is
              ``None`` (hover disabled) rather than misaligned.

        keys: optional ``(N,)`` machine-readable per-point strings for `link` /
            `copy` templates to substitute as ``{hover_key}`` (#1917). Passed
            once for the whole cloud rather than per coloring, because a point's
            IDENTITY does not change with the colour scheme — only its label
            does. Tiled K times here so it stays aligned with the stacked
            positions, which is the alignment this helper exists to own.

    Returns:
        A :class:`StackedColorings`. ``positions`` has shape ``(N*K, D+1)`` with
        column 0 = the coloring index; ``categories`` is the ordered label list
        for building ``Dimension("coloring", categories=...)``.
    """
    coords = np.asarray(coords, dtype=np.float32)
    n = len(coords)
    if not colorings:
        raise ValueError("stack_colorings requires at least one coloring")

    positions_blocks: list[np.ndarray] = []
    color_blocks: list[np.ndarray] = []
    label_blocks: list[list[str]] = []
    categories: list[str] = []
    have_labels = True

    for i, cg in enumerate(colorings):
        cols = np.asarray(cg["colors"], dtype=np.float32)
        if cols.shape != (n, 3):
            raise ValueError(
                f"coloring {i} ({cg.get('label')!r}) colors shape {cols.shape} "
                f"!= expected {(n, 3)}"
            )
        positions_blocks.append(
            np.column_stack([np.full(n, i, dtype=np.float32), coords])
        )
        color_blocks.append(cols)
        categories.append(str(cg["label"]))
        lbls = cg.get("labels")
        if lbls is None:
            have_labels = False
        else:
            if len(lbls) != n:
                raise ValueError(
                    f"coloring {i} ({cg.get('label')!r}) has {len(lbls)} labels "
                    f"!= {n} points"
                )
            label_blocks.append([str(x) for x in lbls])

    positions = np.vstack(positions_blocks).astype(np.float32)
    colors = np.vstack(color_blocks).astype(np.float32)
    labels: Optional[list[str]] = None
    if have_labels:
        labels = [x for block in label_blocks for x in block]

    stacked_keys: Optional[list[str]] = None
    if keys is not None:
        if len(keys) != n:
            raise ValueError(f"keys has {len(keys)} entries != {n} points")
        stacked_keys = [str(x) for x in keys] * len(colorings)

    return StackedColorings(positions, colors, labels, categories, stacked_keys)
