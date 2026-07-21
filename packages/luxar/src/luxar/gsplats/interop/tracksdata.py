"""Bridge fitted Gaussian splats into `tracksdata` graphs.

Each Gaussian splat becomes a node in a `tracksdata` graph, carrying a binary
segmentation **mask** (the splat's ``n_sigma`` support), its **bounding box**,
amplitude, and per-axis position. This lets cells that were fitted as Gaussian
splats live in the Royer-lab tracking ecosystem — `tracksdata
<https://github.com/royerlab/tracksdata>`_ and the tools built on it (ultrack,
trackedit, inTRACKtive) — so they can be linked across time into lineages and
proofread.

Requires the optional dependency::

    pip install 'luxar[tracksdata]'

``tracksdata`` is imported lazily, so importing this module never fails when the
extra is absent; only the graph-building call raises a clear error.

History
-------
Prototyped by Jordão Bragantini (`@JoOkuma`) in PR #20 as a proof of concept that
was waiting on 4D (time + space) Gaussian-splat support. That support now exists,
and this module modernizes the idea to the current ``GSplatData`` API (split
lower-triangular Cholesky storage, no per-splat ``sharpness``) and replaces the
whole-frame rasterization with an efficient analytic per-splat bounding box.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

import numpy as np

from luxar.gsplats import GSplatData

if TYPE_CHECKING:  # pragma: no cover - typing only
    from tracksdata.graph import BaseGraph

__all__ = ["splat_mask_and_bbox", "gsplats_to_tracksdata_graph"]


def _unpack_tril(v: np.ndarray, d: int) -> np.ndarray:
    """Unpack row-major packed lower-triangular vectors to ``(N, d, d)`` matrices.

    Equivalent to :func:`luxar.gsplats.utils.trils.unpack_tril`, inlined here so
    this interop module stays importable with only NumPy + GSplatData (importing
    the ``gsplats.utils`` package eagerly pulls in torch via its device helpers,
    which this lightweight bridge should not require).
    """
    n = v.shape[0]
    out = np.zeros((n, d, d), dtype=v.dtype)
    rows, cols = np.tril_indices(d)
    out[:, rows, cols] = v
    return out


def splat_mask_and_bbox(
    center: np.ndarray,
    cholesky_factor: np.ndarray,
    frame_shape: tuple[int, ...],
    *,
    n_sigma: float = 2.0,
) -> tuple[np.ndarray, np.ndarray]:
    """Rasterize one splat's ``n_sigma`` support to a local boolean mask + bbox.

    The covariance is ``Sigma = L @ L.T`` for the lower-triangular Cholesky
    factor ``L = cholesky_factor``. A voxel ``x`` is inside the support iff its
    Mahalanobis distance ``||L^-1 (x - center)|| <= n_sigma``.

    Rather than evaluating that over the whole frame (``O(n_voxels)`` per splat),
    the axis-aligned bounding box of the ``n_sigma`` ellipsoid is computed
    analytically — the half-extent along axis ``k`` is
    ``n_sigma * sqrt(Sigma_kk) = n_sigma * ||L[k]||`` — and only that local box
    is rasterized, then clamped to ``frame_shape``.

    Parameters
    ----------
    center : np.ndarray
        Splat center, shape ``(d,)``, in voxel coordinates matching ``frame_shape``.
    cholesky_factor : np.ndarray
        Lower-triangular Cholesky factor of the covariance, shape ``(d, d)``.
    frame_shape : tuple[int, ...]
        Shape of the frame the masks live in, length ``d``.
    n_sigma : float
        Mahalanobis radius of the support (default 2.0).

    Returns
    -------
    bbox : np.ndarray
        ``[start_0, ..., start_{d-1}, stop_0, ..., stop_{d-1}]`` (int, half-open,
        clamped to ``frame_shape``).
    mask : np.ndarray
        Boolean array of shape ``stop - start`` (empty if the box is degenerate).
    """
    center = np.asarray(center, dtype=np.float64)
    L = np.asarray(cholesky_factor, dtype=np.float64)
    d = center.shape[0]
    if L.shape != (d, d):
        raise ValueError(f"cholesky_factor must be ({d}, {d}); got {L.shape}")
    if len(frame_shape) != d:
        raise ValueError(f"frame_shape must have length {d}; got {len(frame_shape)}")

    shape_arr = np.asarray(frame_shape, dtype=np.int64)
    half_extent = float(n_sigma) * np.linalg.norm(L, axis=1)  # sqrt(diag(Sigma))
    start = np.clip(np.floor(center - half_extent).astype(np.int64), 0, shape_arr)
    stop = np.clip(np.ceil(center + half_extent).astype(np.int64) + 1, 0, shape_arr)
    bbox = np.concatenate([start, stop])

    extent = stop - start
    if np.any(extent <= 0):
        return bbox, np.zeros(np.maximum(extent, 0), dtype=bool)

    axes = [np.arange(s, e) for s, e in zip(start, stop, strict=True)]
    grid = np.stack(np.meshgrid(*axes, indexing="ij"), axis=-1).astype(np.float64)
    centered = (grid - center).reshape(-1, d)
    # Mahalanobis whitening: z = L^-1 (x - center); inside iff ||z|| <= n_sigma.
    whitened = np.linalg.solve(L, centered.T).T
    sq_dist = np.square(whitened).sum(axis=-1).reshape(tuple(extent))
    mask = sq_dist <= float(n_sigma) ** 2
    return bbox, mask


def gsplats_to_tracksdata_graph(
    gsplats: GSplatData,
    frame_shape: tuple[int, ...],
    *,
    t: int = 0,
    n_sigma: float = 2.0,
    graph: "BaseGraph | None" = None,
    sort_by_amplitude: bool = True,
) -> "BaseGraph":
    """Insert every splat of ``gsplats`` as a node at time ``t`` in a tracksdata graph.

    Each node carries the default tracksdata attributes ``t``, ``mask``
    (:class:`tracksdata.nodes.Mask`) and ``bbox``, plus ``amplitude`` and the
    per-axis position (``z``/``y``/``x`` for the trailing dims). Call repeatedly
    with increasing ``t`` (and the same ``graph``) to build a time-lapse, then
    use tracksdata's edge operators to link nodes into lineages.

    Parameters
    ----------
    gsplats : GSplatData
        Fitted splats. ``gsplats.ndim`` must equal ``len(frame_shape)``.
    frame_shape : tuple[int, ...]
        Spatial shape the masks are rasterized into.
    t : int
        Timepoint to assign to all nodes added by this call.
    n_sigma : float
        Mask support radius passed to :func:`splat_mask_and_bbox`.
    graph : tracksdata.graph.BaseGraph | None
        Graph to add to; a new in-memory (RustWorkX) graph is created if ``None``.
    sort_by_amplitude : bool
        Add nodes in ascending alpha-effective amplitude (A·α) order so brighter
        splats paint last in a :class:`tracksdata.array.GraphArrayView` (matches
        the original POC). The ``amplitude`` node attribute is likewise A·α, so
        the ranking is meaningful for imported classical splats (whose raw
        amplitude is a constant 1, with opacity in the color alpha channel).

    Returns
    -------
    tracksdata.graph.BaseGraph
        The graph the nodes were added to.

    Raises
    ------
    ImportError
        If the optional ``tracksdata`` extra is not installed.
    """
    try:
        import polars as pl
        import tracksdata as td
    except ImportError as exc:  # pragma: no cover - exercised only without the extra
        raise ImportError(
            "gsplats_to_tracksdata_graph requires the optional 'tracksdata' extra: "
            "pip install 'luxar[tracksdata]'"
        ) from exc

    d = len(frame_shape)
    if gsplats.ndim != d:
        raise ValueError(
            f"gsplats.ndim ({gsplats.ndim}) must match len(frame_shape) ({d})"
        )

    keys = td.DEFAULT_ATTR_KEYS
    pos_keys = ["z", "y", "x"][-d:] if d <= 3 else [f"d{i}" for i in range(d)]

    from luxar.gsplats.utils.alpha import effective_amplitudes

    cholesky = _unpack_tril(np.asarray(gsplats.cholesky_factors, dtype=np.float64), d)
    centers = np.asarray(gsplats.centers, dtype=np.float64)
    # Alpha-effective amplitude (A·α when RGBA colors carry per-splat opacity):
    # every blending mode scales a splat's rendered contribution by α, so the
    # paint-order sort and the exported ``amplitude`` attribute must rank by
    # rendered energy. Raw amplitude is constant 1 for imported classical
    # splats (opacity rides in the color alpha channel — see
    # VOLUMETRIC_BLENDING_SPEC.md §5.4.1), which would make the sort a no-op;
    # identical to raw amplitude for fitted data (α = 1).
    amplitudes = np.asarray(effective_amplitudes(gsplats), dtype=np.float64)

    nodes: list[dict[str, Any]] = []
    for i in range(gsplats.n_splats):
        bbox, mask = splat_mask_and_bbox(
            centers[i], cholesky[i], frame_shape, n_sigma=n_sigma
        )
        node: dict[str, Any] = {
            keys.T: int(t),
            keys.MASK: td.nodes.Mask(mask=mask, bbox=bbox),
            keys.BBOX: bbox,
            "amplitude": float(amplitudes[i]),
        }
        for key, value in zip(pos_keys, centers[i], strict=True):
            node[key] = float(value)
        nodes.append(node)

    if sort_by_amplitude:
        nodes.sort(key=lambda n: n["amplitude"])

    if graph is None:
        graph = td.graph.InMemoryGraph()

    # Register the custom float columns with an explicit schema (tracksdata's
    # default keys t/mask/bbox are handled by the backend). Skip keys already
    # present so the function is safe to call repeatedly on the same graph.
    existing = set(graph.node_attr_keys())
    for key in ("amplitude", *pos_keys):
        if key not in existing:
            graph.add_node_attr_key(key, dtype=pl.Float64, default_value=0.0)

    graph.bulk_add_nodes(nodes)
    return graph
