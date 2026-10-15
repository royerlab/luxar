"""Mapping a fitted store's splats onto the voxel grid they are scored on.

A fit's centers are not always voxel indices. Two things move them off the grid
of the volume that was fitted:

* **Physical output.** A fit given a ``voxel_size`` emits physical coordinates by
  default (``output_space="real"``): every center column and Cholesky row is
  multiplied by that axis's spacing. The store records the spacing as
  ``fitting/voxel_size``, so the mapping can be undone exactly.
* **Downscaled fitting.** A fit with ``--downscale`` optimised against a decimated
  copy and rescaled its splats back to the source grid. ``fitting/source_shape``
  and ``fitting/fitted_shape`` record both grids, so the integer factors are
  recoverable when a caller wants to score on the grid the fit actually saw.

Rendering against a voxel grid without undoing the first mapping compares
physical-unit splats with voxel indices — for a 0.2 µm lateral spacing the
splats land in the top-left fifth of the grid and the score is meaningless.
This module undoes both mappings, and is deliberately dumb: numpy only, the
caller supplies the stats it already loaded.
"""

from __future__ import annotations

import math
from typing import TYPE_CHECKING, Any, Mapping, Optional, Sequence

import numpy as np

if TYPE_CHECKING:
    from luxar.gsplats.gsplat_data import GSplatData

__all__ = [
    "FRAME_STAMP_KEYS",
    "drop_frame_stamp",
    "linear_part_is_identity",
    "agreed_voxel_size",
    "fit_downscale_factors",
    "fraction_inside_grid",
    "is_stacked_store",
    "spatial_extent_fraction",
    "map_to_grid",
    "parse_voxel_size",
    "physical_voxel_size_stamp",
    "source_region_start",
    "stored_voxel_size",
]


def _positive_finite_vector(value: Any, ndim: int) -> Optional[tuple[float, ...]]:
    """``value`` as an ``ndim``-vector of finite positive floats, else ``None``.

    A scalar is broadcast. Booleans are rejected (``True`` is an ``int``).
    """
    if isinstance(value, bool) or value is None:
        return None
    if isinstance(value, (int, float)):
        items: list[Any] = [value] * ndim
    elif isinstance(value, (list, tuple, np.ndarray)):
        items = list(value)
    else:
        return None
    if len(items) != ndim:
        return None
    out: list[float] = []
    for item in items:
        if isinstance(item, bool):
            return None
        try:
            number = float(item)
        except (TypeError, ValueError):
            return None
        if not math.isfinite(number) or number <= 0.0:
            return None
        out.append(number)
    return tuple(out)


#: The stamps that together say how the centers map to voxels. They describe
#: one frame, so a transform that changes the frame drops them TOGETHER.
FRAME_STAMP_KEYS = (
    "voxel_size",
    "voxel_unit",
    "voxel_size_source",
    "physical_coordinates",
)


def drop_frame_stamp(stats: Mapping[str, Any]) -> dict[str, Any]:
    """A copy of ``stats`` without frame stamps or config spacing fallback."""
    result = {k: v for k, v in stats.items() if k not in FRAME_STAMP_KEYS}
    config = result.get("config")
    if isinstance(config, Mapping) and "voxel_size" in config:
        result["config"] = {k: v for k, v in config.items() if k != "voxel_size"}
    return result


def linear_part_is_identity(matrix: Any, ndim: int) -> bool:
    """Whether a ``(d, d)`` / ``(d+1, d+1)`` transform is a pure translation."""
    m = np.asarray(matrix, dtype=np.float64)
    linear = m[:ndim, :ndim] if m.shape[0] > ndim else m
    return linear.shape == (ndim, ndim) and bool(np.allclose(linear, np.eye(ndim)))


def physical_voxel_size_stamp(
    voxel_size: Any, output_space: str, ndim: int
) -> dict[str, list[float]]:
    """The ``{"voxel_size": [...]}`` stamp for a fit that emitted physical centers.

    Empty when the fit stayed on the voxel grid (no spacing, or
    ``output_space="voxel"``), so a reader never mistakes a spacing the fit was
    merely told about for a scale it applied.
    """
    if output_space != "real":
        return {}
    vector = _positive_finite_vector(voxel_size, ndim)
    if vector is None or all(v == 1.0 for v in vector):
        return {}
    return {"voxel_size": [float(v) for v in vector]}


def agreed_voxel_size(
    stats_list: Sequence[Optional[Mapping[str, Any]]],
) -> Optional[list[float]]:
    """The ``voxel_size`` stamp every component that recorded one agrees on.

    For merges of tiles/parts of ONE fit: they all emit the same frame, so the
    stamp is carried when unanimous; components without one (empty tiles)
    abstain. Disagreement, or nobody stamping, returns ``None``.
    """
    stamps = [
        tuple(float(v) for v in value)
        for stats in stats_list
        if isinstance(value := (stats or {}).get("voxel_size"), (list, tuple))
    ]
    if stamps and all(stamp == stamps[0] for stamp in stamps[1:]):
        return list(stamps[0])
    return None


def stored_voxel_size(
    stats: Optional[Mapping[str, Any]], ndim: int
) -> Optional[tuple[float, ...]]:
    """The spacing a stored fit multiplied its centers by, or ``None``.

    The mapping applies when ``voxel_size`` is present and
    ``physical_coordinates`` is not ``False`` (a CLI fit records the spacing for
    provenance even when its centers stayed voxel indices, and says so with that
    flag). Falls back to a ``config`` block carrying ``voxel_size`` and an
    ``output_space`` other than ``"voxel"`` (externally authored stores).

    A stamp SHORTER than the centers is a spatial stamp on a stacked store (a
    timelapse merge appends its time/channel columns last): the trailing
    columns are padded with 1.0 and a note is printed. Any other length
    mismatch is refused with a warning rather than silently ignored.
    """
    if not stats:
        return None
    if stats.get("physical_coordinates") is False:
        return None
    raw = stats.get("voxel_size")
    if raw is None:
        config = stats.get("config")
        if (
            not isinstance(config, Mapping)
            or config.get("output_space", "real") != "real"
        ):
            return None
        raw = config.get("voxel_size")
        if raw is None:
            return None
    return _fit_stamp_to_rank(raw, ndim)


def _fit_stamp_to_rank(raw: Any, ndim: int) -> Optional[tuple[float, ...]]:
    """``raw`` as an ``ndim``-vector, padding trailing stacked columns with 1.0."""
    from arbol import aprint

    if isinstance(raw, (list, tuple, np.ndarray)) and 0 < len(raw) < ndim:
        spatial = _positive_finite_vector(raw, len(raw))
        if spatial is not None:
            aprint(
                f"voxel_size stamp has {len(raw)} values for {ndim} center columns: "
                f"treating the trailing {ndim - len(raw)} column(s) as stacked "
                "(time/channel) axes with unit spacing"
            )
            return spatial + (1.0,) * (ndim - len(raw))
    vector = _positive_finite_vector(raw, ndim)
    if vector is None:
        aprint(
            f"WARNING: ignoring the store's voxel_size stamp {raw!r}: it does not "
            f"describe {ndim} center columns; pass --voxel-size if the centers "
            "are physical"
        )
    return vector


def is_stacked_store(
    stats: Optional[Mapping[str, Any]],
    ndim: int,
    dimension_metadata: Any = None,
) -> bool:
    """Whether the centers carry stacked (time/channel) columns beyond the source grid.

    A stacked store (a ``batch-fit`` timelapse, ``merge --as-dimension``) has
    its spatial columns FIRST and the stacked ones LAST, while any
    ``source_shape`` it records is in source order (time first) or describes one
    component — so the source grid does not describe the centers. Detected from
    the root ``dimension_metadata`` naming a non-spatial column (both writers
    name the stacked axis, e.g. ``t``), or a ``source_shape`` whose rank differs
    from the centers'. Callers should still check that the centers fall inside a
    grid before trusting it (:func:`fraction_inside_grid`).
    """
    spatial_names = {"x", "y", "z"}
    if isinstance(dimension_metadata, (list, tuple)) and dimension_metadata:
        names = [
            str(item.get("name", "")).lower()
            for item in dimension_metadata
            if isinstance(item, Mapping)
        ]
        if any(name not in spatial_names for name in names):
            return True
    shape = (stats or {}).get("source_shape")
    return isinstance(shape, (list, tuple)) and len(shape) != ndim


def source_region_start(
    stats: Optional[Mapping[str, Any]], ndim: int
) -> Optional[tuple[int, ...]]:
    """Per-axis start of a ``--region`` crop fit (``fitting/source_region``), or ``None``.

    A crop fit records ``source_region`` as per-axis ``[start, stop]`` and its
    ``source_shape`` as the CROP's shape, while its centers stay in full-volume
    coordinates (start included). Rendering onto the crop's grid therefore
    subtracts the start (in voxels, after any physical spacing is divided out).
    ``None`` when absent (the fit covers the whole volume), malformed, or all
    zero.
    """
    region = (stats or {}).get("source_region")
    if not isinstance(region, (list, tuple)) or len(region) != ndim:
        return None
    starts: list[int] = []
    for span in region:
        if not isinstance(span, (list, tuple)) or len(span) != 2:
            return None
        start = span[0]
        if isinstance(start, bool) or not isinstance(start, (int, float)):
            return None
        starts.append(int(start))
    return tuple(starts) if any(starts) else None


def parse_voxel_size(text: str, ndim: int) -> tuple[float, ...]:
    """Parse a ``--voxel-size`` option (one value, or one per axis).

    Raises ``ValueError`` with a message naming the expected count.
    """
    try:
        values = [float(part) for part in text.split(",") if part.strip()]
    except ValueError as exc:
        raise ValueError(f"not a comma-separated list of numbers: {text!r}") from exc
    parsed = _positive_finite_vector(values[0] if len(values) == 1 else values, ndim)
    if parsed is None:
        raise ValueError(
            f"expected 1 or {ndim} finite positive values (one per center "
            f"column), got {text!r}"
        )
    return parsed


def _factor_for(source: int, fitted: int) -> Optional[int]:
    """The integer ``f`` with ``ceil(source / f) == fitted``, preferring the closest."""
    if source < 1 or fitted < 1 or fitted > source:
        return None
    guess = max(1, round(source / fitted))
    for candidate in (guess, guess - 1, guess + 1, guess - 2, guess + 2):
        if candidate >= 1 and -(-source // candidate) == fitted:
            return candidate
    return None


def fit_downscale_factors(
    stats: Optional[Mapping[str, Any]], ndim: int
) -> Optional[tuple[int, ...]]:
    """Per-axis ``--downscale`` factors recovered from the stored grids, or ``None``.

    ``fitted_shape`` is ``ceil(source_shape / f)`` per axis (the decimation is a
    strided slice), so each factor is the integer that reproduces it. ``None``
    when either grid is missing, the ranks disagree with the centers, or no
    integer reproduces a recorded axis; all-ones (no decimation) is also
    ``None``, so callers can treat "nothing to undo" uniformly.
    """
    if not stats:
        return None
    source = stats.get("source_shape")
    fitted = stats.get("fitted_shape")
    if not isinstance(source, (list, tuple)) or not isinstance(fitted, (list, tuple)):
        return None
    if len(source) != ndim or len(fitted) != ndim:
        return None
    factors: list[int] = []
    for s, f in zip(source, fitted, strict=True):
        factor = _factor_for(int(s), int(f))
        if factor is None:
            return None
        factors.append(factor)
    if all(factor == 1 for factor in factors):
        return None
    return tuple(factors)


def map_to_grid(
    data: "GSplatData",
    *,
    scale: Optional[Sequence[float]] = None,
    origin: Optional[Sequence[float]] = None,
) -> "GSplatData":
    """``data`` re-expressed on a voxel grid: ``centers / scale - origin``.

    The exact inverse of the physical (and downscale) conversion the fitter
    applies: centers divided per column, Cholesky row ``i`` divided by
    ``scale[i]``. Amplitudes are untouched — the renderer evaluates an
    unnormalised Gaussian, so a pure change of frame leaves every voxel's value
    the same. ``origin`` (in GRID units) then translates, for a render whose
    first voxel is not at index 0. Returns a flat render-only copy (centers,
    amplitudes, Cholesky, truncation radius); colours and stats do not ride
    along. With neither argument, ``data`` itself is returned.
    """
    if scale is None and origin is None:
        return data
    from luxar.gsplats.gsplat_data import GSplatData

    centers = np.asarray(data.centers, dtype=np.float32)
    cholesky = np.asarray(data.cholesky_factors, dtype=np.float32)
    ndim = centers.shape[1] if centers.ndim == 2 else len(scale or origin or ())
    if scale is not None:
        inv = 1.0 / np.asarray(scale, dtype=np.float64)
        centers = (centers * inv).astype(np.float32)
        tril = np.concatenate([[inv[i]] * (i + 1) for i in range(ndim)])
        cholesky = (cholesky * tril).astype(np.float32)
    if origin is not None:
        centers = (centers - np.asarray(origin, dtype=np.float32)).astype(np.float32)
    return GSplatData(
        centers=centers,
        amplitudes=np.asarray(data.amplitudes, dtype=np.float32),
        cholesky_factors=cholesky,
        truncation_radius=data.truncation_radius,
    )


def spatial_extent_fraction(centers: np.ndarray, shape: Sequence[int]) -> float:
    """The largest share of the grid extent the centers span on a spatial axis.

    Spatial = the first ``min(3, ndim)`` columns (stacked columns come last).
    A physical store with SUB-UNIT spacing and no stamp lands entirely inside the
    grid, crammed into a corner, which :func:`fraction_inside_grid` cannot see;
    a value well under 1 on every spatial axis exposes it. Returns 1.0 when the
    question has no answer (fewer than two centers, a one-voxel grid).
    """
    pts = np.asarray(centers)
    if pts.ndim != 2 or len(pts) < 2:
        return 1.0
    fractions = []
    for axis in range(min(3, pts.shape[1], len(shape))):
        span = float(shape[axis]) - 1.0
        if span <= 0:
            continue
        fractions.append(float(np.ptp(pts[:, axis])) / span)
    return max(fractions) if fractions else 1.0


def fraction_inside_grid(centers: np.ndarray, shape: Sequence[int]) -> float:
    """Share of ``centers`` inside the voxel grid ``[-0.5, n - 0.5)`` on every axis.

    A cheap sanity check before an expensive render: a store whose splats are in
    physical units but carry no spacing stamp lands mostly OUTSIDE (or crammed
    into a corner of) the reference grid, which this exposes as a low number.
    """
    pts = np.asarray(centers)
    if pts.ndim != 2 or len(pts) == 0:
        return 0.0
    upper = np.asarray(shape, dtype=np.float64) - 0.5
    inside = np.all((pts >= -0.5) & (pts < upper), axis=1)
    return float(np.count_nonzero(inside) / len(pts))
