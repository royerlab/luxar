"""Fit a volume according to a :class:`FitPlan` and merge the result.

Each box is fit on a halo-padded crop (so the optimiser has neighbour context),
then only the splats whose centre lands in the *core* box are kept (no
double-counting, no gaps). This is the validated, seam-free approach from the
investigation — far simpler than apodized blending and it held up visually.

This is a sibling of ``fit_tiled`` that consumes the planner's content-balanced,
budgeted boxes instead of a uniform grid. It does not touch ``fit_tiled`` /
``fit_tiled_parallel`` (it only fits each box via ``fit_gaussian_splats``).

The per-box body lives in :func:`_fit_one_box` so the **sequential** driver here
and the **parallel** subprocess worker (``fit_planned_parallel`` /
``gsplat fit --tiling content --plan-box``) share one source of truth for budget scaling, the
padded crop, and the keep-core mask — they can never drift.
"""

from __future__ import annotations

import gc
import time
from typing import TYPE_CHECKING, Any, Callable, Optional, Tuple

import numpy as np

from .spec import FitPlan, PlanBox

if TYPE_CHECKING:
    from luxar.gsplats.gsplat_data import GSplatData

ProgressCallback = Callable[[int, int, str], None]


def _padded_bounds(
    box: PlanBox, overlap: int, shape: Tuple[int, int, int]
) -> Tuple[int, int, int, int, int, int]:
    """Halo-padded crop bounds for ``box`` clamped to the volume ``shape``."""
    Z, Y, X = shape
    z0, z1, y0, y1, x0, x1 = box.box
    pad = int(overlap)
    return (
        max(0, z0 - pad),
        min(Z, z1 + pad),
        max(0, y0 - pad),
        min(Y, y1 + pad),
        max(0, x0 - pad),
        min(X, x1 + pad),
    )


def _scaled_budget(
    box: PlanBox, overlap: int, shape: Tuple[int, int, int], cap: int
) -> int:
    """Per-box seed budget, scaled up for the padded crop and capped.

    The plan budget sizes the *core*; the optimiser fits the padded crop, so we
    scale by the padded/core voxel ratio (capped at 4x to bound cost on thin
    slabs) and at the calibrated saturation ``cap`` so a box never fits past the
    over-saturated regime.
    """
    pz0, pz1, py0, py1, px0, px1 = _padded_bounds(box, overlap, shape)
    sub_size = (pz1 - pz0) * (py1 - py0) * (px1 - px0)
    core_vox = max(1, box.voxels)
    budget = int(min(box.budget * sub_size / core_vox, box.budget * 4))
    if cap > 0:
        budget = min(budget, cap)
    return budget


def _box_truncation_radius(fit_kwargs: "dict[str, Any]") -> float:
    """The truncation radius this box's fit config asks for.

    ``fit_gaussian_splats`` stamps ``truncation_radius=config.truncate`` on its
    result, so the fitted radius is simply the ``truncate`` fit kwarg — resolved
    from the kwargs here because the zero-budget early-out of
    :func:`_fit_one_box` has no fit result to read it off.
    """
    from luxar.typing_utils.constants import DEFAULT_TRUNCATION_RADIUS

    truncate = fit_kwargs.get("truncate")
    if truncate is None:
        return float(DEFAULT_TRUNCATION_RADIUS)
    return float(truncate)


def _fit_one_box(
    volume: np.ndarray,
    box: PlanBox,
    overlap: int,
    cap: int,
    **fit_kwargs: Any,
) -> GSplatData:
    """Fit a single plan box and return its core-kept splats in GLOBAL coords.

    Crops a halo-padded region, fits it with the (padded-scaled, capped) budget,
    translates the fitted centres back to global coordinates, and keeps only the
    splats whose centre lands in the half-open core box ``[z0:z1, y0:y1, x0:x1]``
    (so adjacent boxes tile the volume with no gaps and no double-counting).

    Returns the fitted :class:`~luxar.gsplats.gsplat_data.GSplatData`, filtered
    to the core-kept splats — possibly EMPTY (a box with non-positive scaled
    budget yields 0 splats), in which case it still carries the radius the fit
    config asked for. Shared by the sequential :func:`fit_planned` loop and the
    parallel per-box worker.

    The whole dataset, not the bare ``(centers, amplitudes, cholesky_factors)``
    arrays it used to hand back: every consumer rebuilt a ``GSplatData`` from
    those three arrays, which silently dropped the fit's ``truncation_radius``
    (a config asking for ``truncate: 3.5`` stored the 2.75 default) and its
    per-box ``stats``. A content result then refused to
    ``GSplatData.concatenate`` with a uniform-tiled one fitted from the same
    config ("Truncation radius mismatch") — issue #1637.
    """
    from luxar.gsplats._data.filtering import (
        _REGION_SCOPED_STATS_KEYS,
        drop_content_scoped_stats,
    )
    from luxar.gsplats.fit_gsplats import fit_gaussian_splats
    from luxar.gsplats.gsplat_data import GSplatData
    from luxar.gsplats.utils.trils import tril_size
    from luxar.io._compiler.gsplat_tree import json_safe_value

    # The content pipeline is voxel-space end to end: plan boxes, padded
    # bounds, and the keep-core mask below are all voxel coordinates. A
    # voxel_size/output_space pair leaking in from a YAML config would make
    # the inner fit return PHYSICAL centers, which would then get voxel-space
    # origin offsets added and be mis-filtered by the core mask — so strip
    # them here and warn (the default warning filter dedups per process).
    vs = fit_kwargs.pop("voxel_size", None)
    out_space = fit_kwargs.pop("output_space", None)
    if vs is not None and out_space != "voxel":
        import warnings

        warnings.warn(
            "voxel_size/output_space='real' are not supported with content-"
            "planned fitting; fitting in voxel space (voxel_size ignored).",
            UserWarning,
            stacklevel=2,
        )

    V = np.asarray(volume, dtype=np.float32)
    ndim = V.ndim
    pz0, pz1, py0, py1, px0, px1 = _padded_bounds(box, overlap, V.shape)
    z0, z1, y0, y1, x0, x1 = box.box

    budget = _scaled_budget(box, overlap, V.shape, cap)
    if budget <= 0:
        # An empty box still answers with the radius the config asked for, rather
        # than the default (#1637). Nothing on the current paths reads it: both
        # drivers skip zero-budget boxes, and a `--plan-box K` worker on such a
        # box writes an `.empty` marker instead of a store. It is here so an
        # in-process caller of `_fit_one_box` gets a consistent answer.
        return GSplatData(
            centers=np.zeros((0, ndim), np.float32),
            amplitudes=np.zeros((0,), np.float32),
            cholesky_factors=np.zeros((0, tril_size(ndim)), np.float32),
            truncation_radius=_box_truncation_radius(fit_kwargs),
        )

    sub = V[pz0:pz1, py0:py1, px0:px1]
    gd = fit_gaussian_splats(sub, seeds=budget, **fit_kwargs)
    c = np.asarray(gd.centers, dtype=np.float32) + np.array([pz0, py0, px0], np.float32)
    a = np.asarray(gd.amplitudes, dtype=np.float32)
    k = np.asarray(gd.cholesky_factors, dtype=np.float32)
    keep = (
        (c[:, 0] >= z0)
        & (c[:, 0] < z1)
        & (c[:, 1] >= y0)
        & (c[:, 1] < y1)
        & (c[:, 2] >= x0)
        & (c[:, 2] < x1)
    )
    # Slice FIRST, then drop `gd`: keeping the whole inner result alive just to
    # read slices out of it would defeat the per-box memory release below.
    # Defensive: `fit_gaussian_splats` produces no colors today, so this branch is
    # unreachable — kept so a future coloured fit is masked, not silently dropped.
    colors = None if gd.colors is None else np.asarray(gd.colors)[keep]
    box_stats = dict(gd.stats)
    # The napari capture buffers are per-optimisation scratch: the LEAF writer
    # stamps `lod_stats` RAW (io/_compiler/gsplat_tree.py), so under
    # `--napari-movie` these frame buffers would be handed to the attrs JSON
    # encoder — and holding the references would keep a box's frames (and the crop
    # shape they were rendered at) alive past the release below.
    box_stats.pop("movie_frames", None)
    box_stats.pop("movie_shape", None)
    # The fit's source/fitted grid stamps describe the PADDED CROP, and they become
    # each partition part's on-disk `lod_stats` — the provenance a reader asks for
    # the part's own source grid (`gsplat info`'s source block reads exactly this
    # key set, off a flat leaf's stats). Two independent things invalidate them: a
    # halo, which makes the crop strictly larger than the region this part
    # represents, and the core-keep mask, which is a spatial restriction like a
    # bbox crop. Not `_is_crop`, whose "did the mask remove anything" meaning
    # misses the halo: every splat can land in the core and the crop still be 18³
    # for a 12³ part.
    n_kept = int(np.count_nonzero(keep))
    padded_is_larger = (pz0, pz1, py0, py1, px0, px1) != (z0, z1, y0, y1, x0, x1)
    if padded_is_larger or n_kept < int(keep.size):
        for key in _REGION_SCOPED_STATS_KEYS:
            box_stats.pop(key, None)
        # ...and the MEASURED scores go with them, for the second reason the same
        # predicate covers: `psnr_db` / `ssim` / `final_*` were taken on the fit of
        # the PADDED crop, i.e. with the halo splats present and against a target
        # region larger than this part. Publishing them as the part's `lod_stats`
        # would quote a reconstruction score for a splat set the part does not hold
        # (#1600). Note the halo half is not a count question at all — every splat
        # can land in the core and the crop still be 18³ for a 12³ part — which is
        # why this rides the region predicate rather than `_is_crop`.
        drop_content_scoped_stats(box_stats)
    box_stats["n_splats"] = n_kept
    # ...and the same RAW `lod_stats` write is why non-finite values cannot ride
    # along either: zarr emits them as bare `Infinity`/`NaN` tokens that a strict
    # parser (the viewer's `JSON.parse`) refuses. A signal-free crop really does
    # fit to `psnr_db = inf`, and a content `batch-fit` reuses ONE box plan for
    # every (t, c), so a box with no signal at some timepoint is ordinary rather
    # than pathological. Filtered with the same helper the root `pipeline/` bucket
    # uses (`split_fitting_info`), which also coerces numpy scalars.
    safe_stats: dict[str, Any] = {}
    for key, value in box_stats.items():
        ok, converted = json_safe_value(value)
        if ok:
            safe_stats[key] = converted
    out = GSplatData(
        centers=c[keep],
        amplitudes=a[keep],
        cholesky_factors=k[keep],
        colors=colors,
        stats=safe_stats,
        truncation_radius=gd.truncation_radius,
    )

    # free the per-box working set before the next box / before the worker exits
    del gd
    gc.collect()
    device = fit_kwargs.get("device")
    if device and str(device).startswith("cuda"):
        import torch

        torch.cuda.empty_cache()
    return out


def fit_planned(
    volume: np.ndarray,
    plan: FitPlan,
    *,
    device: Optional[str] = None,
    verbose: bool = False,
    progress_callback: Optional[ProgressCallback] = None,
    partition: bool = False,
    recipe: Optional[str] = None,
    recipe_params: "Optional[Any]" = None,
    **fit_kwargs: Any,
) -> "Any":
    """Fit every box in ``plan`` and return the merged result.

    ``fit_kwargs`` are forwarded to ``fit_gaussian_splats`` per box (preset /
    n_iters / loss / cull_retention / ...). Each box's seed budget comes from the
    plan; near-empty boxes (budget 0) are skipped.

    ``fit_kwargs["floor"]`` must already be a CONCRETE level (or ``"none"``):
    every box crop is handed the same value, so a spec like ``auto``/``pNN``
    would be re-estimated against each crop and abutting core-kept boxes would
    subtract wildly different pedestals — visible brightness steps at box
    boundaries. The CLI resolves it once against the whole volume before calling
    here (``luxar.cli.gsplat_ops.fitting.fit_utils.resolve_shared_floor``). What
    is shared is the floor ARGUMENT, not the input: every box is still handed its
    own crop, and the subtraction is not bit-exact either, because the
    normalization floor is clamped up to a crop's own minimum
    (``image_min = max(level, min(crop))``) — so a box lying entirely above the
    pedestal subtracts its own minimum instead.

    With ``partition=True`` (the CLI default) the per-box splats are kept as a
    ``kind=partition`` tree — one part per box (boxes are core-disjoint, so this
    is exact) — for viewer frustum culling; a :class:`~luxar.gsplats.tree.GSplatNode`
    is returned. With ``partition=False`` the boxes are concatenated into a single
    flat :class:`GSplatData` leaf (``--flat``).

    Boxes are fit **sequentially**. For concurrent fitting on one GPU use
    :func:`luxar.gsplats.planner.fit_planned_parallel.fit_planned_parallel`
    (``gsplat fit --tiling content -j N``), which fits each box in its own
    subprocess and merges the same way.
    """
    from luxar.gsplats.gsplat_data import GSplatData

    V = np.asarray(volume, dtype=np.float32)
    if V.ndim != 3:
        raise ValueError(f"fit_planned expects a 3-D volume, got shape {V.shape}")
    pad = int(plan.overlap)
    fit_kwargs.setdefault("cull_retention", 0.999)
    fit_kwargs.setdefault("verbose", False)
    fit_kwargs["device"] = device

    # Per-box saturation cap (from the calibration): the halo inflation must not
    # push the fit past the K the calibration measured as over-saturated.
    cap = int(plan.density.get("saturation_cap", 0)) if plan.density else 0

    regions: list[GSplatData] = []  # one core-kept GSplatData per fit box
    # Which plan box each region came from. Boxes are skipped on three
    # independent conditions below, so position in `regions` is NOT the box
    # index — and the plan's split-plane tree is labelled by box index.
    region_boxes: list[int] = []
    n = len(plan.boxes)
    n_fit = 0
    t0 = time.perf_counter()
    for i, b in enumerate(plan.boxes):
        if b.budget <= 0:
            continue
        budget = _scaled_budget(b, pad, V.shape, cap)
        if budget <= 0:
            continue
        if progress_callback is not None:
            progress_callback(i, n, f"box {i + 1}/{n} budget={budget}")
        region = _fit_one_box(V, b, pad, cap, **fit_kwargs)
        n_fit += 1
        n_kept = int(region.n_splats)
        if n_kept > 0:
            # The fitted dataset itself, so its truncation_radius and per-box
            # stats reach the merge instead of being rebuilt away (#1637).
            regions.append(region)
            region_boxes.append(i)
        if verbose:
            from arbol import aprint

            aprint(f"  box {i + 1}/{n}: kept {n_kept:,} splats")

    elapsed = time.perf_counter() - t0
    if not regions:
        raise ValueError("fit_planned produced no splats (all boxes empty?)")

    if partition:
        # One part per box — boxes are core-disjoint, so this is an exact
        # spatial partition (viewer frustum-culls per part). Returns a tree node.
        # ``recipe`` gives each part its own LOD ladder/group as it is assembled.
        return GSplatData.partition_from_regions(
            regions,
            recipe=recipe,
            recipe_params=recipe_params,
            # The planner's own split planes: core-disjoint boxes have an EXACT
            # back-to-front order, and this is what carries it to the viewer.
            bsp_tree=plan.bsp_tree,
            region_labels=region_boxes,
        )

    # `concatenate` (what the uniform tiled path's `merge_tile_results` uses)
    # carries the boxes' shared truncation_radius through the merge — a manual
    # re-`GSplatData(...)` of the three arrays reset it to the default (#1637).
    # It REPLACES stats with its own summary, so the planned-fit keys go on after.
    merged = GSplatData.concatenate(regions)
    merged.stats.update(
        {
            "planned_fit": True,
            "n_boxes": n,
            "n_boxes_fit": n_fit,
            "overlap": pad,
            "volume_shape": list(V.shape),
            # Overwrite `concatenate`'s SUM of the boxes' own times with true
            # wall clock, as the uniform tiled merge does (`merge_tile_results`):
            # one key must not mean "summed fit time" here and "elapsed" there.
            "time_seconds": float(elapsed),
        }
    )
    return merged


__all__ = ["fit_planned"]
