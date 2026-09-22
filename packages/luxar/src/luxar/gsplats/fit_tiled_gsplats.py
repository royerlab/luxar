# fit_tiled_gsplats.py
"""Tiled fitting for large volumes that exceed GPU memory.

Splits a volume into overlapping tiles with cosine (Hann) apodization,
fits Gaussian splats independently per tile, and concatenates results.
The background floor is resolved once against the whole volume and
subtracted from each tile *before* apodization (floor subtraction and
windowing do not commute); on the floor-subtracted data the Hann
partition-of-unity property then ensures seamless blending without
post-merge pruning. When per-tile denoising is active, the level is resolved
on the DENOISED basis (``resolve_volume_floor_denoised``), because that is
the data it is subtracted from — matching what the non-tiled path, which
denoises the whole volume first, estimates (#1178). That match is exact for a
volume within the denoise probe's budget; above it, only a ``pNN`` floor is
corrected onto the denoised basis and the default ``auto`` keeps its raw-basis
level with a printed note.
The shared normalization range follows the same basis: its whole-volume
endpoints are shifted by the denoise-induced change measured on the bounded
shape-preserving probe, so tiles are not normalized against raw extremes they
never see.
"""

from __future__ import annotations

import time
from typing import Any, Optional, Sequence

import numpy as np
from arbol import aprint, asection

from luxar.gsplats.fit_basis import fit_image_min
from luxar.gsplats.fit_gsplats import fit_gaussian_splats
from luxar.gsplats.fitting.preprocessing import (
    NORM_RANGE_MIN_SPAN,
    _floor_spec_is_volume_derived,
    resolve_volume_floor_denoised,
    resolve_volume_norm_range_denoised,
)
from luxar.gsplats.fitting.results import stamp_voxels_per_splat
from luxar.gsplats.fitting.validation import _validate_floor
from luxar.gsplats.gsplat_data import GSplatData
from luxar.gsplats.merged_quality import stamp_merged_quality
from luxar.gsplats.tiling import (
    TileSpec,
    compute_tile_specs,
    cosine_window,
    grid_bsp_tree,
    resolve_grid_scale,
)

_TILE_SIGNAL_EPS = 1e-8


def tile_has_signal(tile_data: np.ndarray) -> bool:
    """Whether a prepared tile should enter the fitter rather than be skipped."""
    return not bool(tile_data.max() < _TILE_SIGNAL_EPS)


# A tile holding at least this fraction of the EQUAL mass share (1/N of the
# Hann-weighted above-floor intensity, N = non-empty tiles) is never budgeted
# below the same fraction of the equal seed share (K/N), before integer
# rounding. It is a backstop, not the allocation law: the mass weighting below
# already scales a dim tile's budget with its content, and at the default
# saturation exponent (0.44) the floor rarely binds.
MIN_TILE_SEED_SHARE_FRACTION = 0.25


def resolve_tile_intensity_scale(
    norm_range: "Sequence[float] | None", applied_floor: "float | None"
) -> float:
    """The intensity ceiling ABOVE the floor that occupancy weights measure against.

    A CLI/batch ``norm_range`` is expressed in raw input units while uniform
    tiles see floor-subtracted data, so the ceiling is shifted by the floor
    (the same basis :func:`_ensure_tile_norm_range` gives the tiles). With no
    shared range — each tile derives its own scale — the weights fall back to
    unit scale. The allocation is invariant to that common scale, but CLI and
    batch callers still resolve one shared range so the recorded weights retain
    a consistent physical meaning.
    """
    if norm_range is None:
        return 1.0
    scale = float(norm_range[1]) - float(applied_floor or 0.0)
    if not np.isfinite(scale) or scale <= _TILE_SIGNAL_EPS:
        return 1.0
    return scale


def occupancy_weight_from_mass(
    mass: float,
    hann_voxels: float,
    *,
    intensity_scale: float = 1.0,
    saturation_exponent: float = 1.0,
) -> float:
    """Weight one tile from its Hann-weighted above-floor intensity mass.

    ``hann_voxels * (mass / hann_voxels / intensity_scale) ** alpha``: the
    tile's Hann-weighted size times its saturated MEAN windowed intensity as a
    fraction of the shared ceiling. Homogeneous content therefore gets the same
    splat density in tiles of different sizes (the fraction cancels the size),
    while between equal-size tiles the budgets stand in the ratio of their
    masses to the power ``alpha``. A tile with no mass weighs zero and is
    skipped.
    """
    if mass <= 0.0 or hann_voxels <= 0.0:
        return 0.0
    fraction = mass / hann_voxels / max(float(intensity_scale), _TILE_SIGNAL_EPS)
    return float(hann_voxels * fraction ** float(saturation_exponent))


def tile_windowed_mass(tile_data: np.ndarray, window: np.ndarray) -> float:
    """Return one tile's Hann-weighted above-floor intensity mass."""
    windowed = tile_data * window
    if not tile_has_signal(windowed):
        return 0.0
    return float(np.sum(windowed, dtype=np.float64))


def tile_occupancy_weight(
    tile_data: np.ndarray,
    window: np.ndarray,
    *,
    intensity_scale: float = 1.0,
    saturation_exponent: float = 1.0,
) -> float:
    """Return Hann-weighted tile size scaled by saturated mean intensity.

    ``tile_data`` is floor-subtracted (background at zero), so its Hann-weighted
    sum is the tile's intensity MASS above the fitter's resolved floor. The
    mass weight replaces the fixed-threshold proxy. See
    :func:`occupancy_weight_from_mass` for the formula.
    """
    return occupancy_weight_from_mass(
        tile_windowed_mass(tile_data, window),
        float(np.sum(window, dtype=np.float64)),
        intensity_scale=intensity_scale,
        saturation_exponent=saturation_exponent,
    )


def floor_tile_seed_shares(
    weights: Sequence[float],
    masses: Sequence[float],
    *,
    min_share_fraction: float = MIN_TILE_SEED_SHARE_FRACTION,
) -> list[float]:
    """Raise weights so no tile of non-trivial mass falls below the floor share.

    With ``N`` non-empty tiles, a tile holding at least ``min_share_fraction /
    N`` of the total mass is pinned at ``min_share_fraction / N`` of the budget
    whenever its proportional share would be smaller; the remaining budget is
    split proportionally among the others (water-filling, repeated until no
    further tile drops below the floor, which is monotone and terminates within
    ``N`` passes). The returned weights sum to the input sum, so their
    magnitude stays in Hann-voxel units. Weights of zero stay zero; a tile
    holding less than the trigger share is never raised, so slivers of nearly
    nothing keep their small budgets.
    """
    n_positive = sum(weight > 0.0 for weight in weights)
    total_mass = float(sum(masses))
    total_weight = float(sum(weights))
    if n_positive <= 1 or total_mass <= 0.0 or total_weight <= 0.0:
        return [float(weight) for weight in weights]
    floor_share = float(min_share_fraction) / n_positive
    eligible = [
        weight > 0.0 and mass / total_mass >= floor_share
        for weight, mass in zip(weights, masses, strict=True)
    ]
    pinned = [False] * len(weights)
    shares = [0.0] * len(weights)
    for _ in range(len(weights)):
        free = [i for i, w in enumerate(weights) if w > 0.0 and not pinned[i]]
        free_weight = sum(weights[i] for i in free)
        free_budget = 1.0 - floor_share * sum(pinned)
        for i in free:
            shares[i] = free_budget * weights[i] / free_weight if free_weight else 0.0
        newly = [i for i in free if eligible[i] and shares[i] < floor_share]
        if not newly:
            break
        for i in newly:
            pinned[i] = True
            shares[i] = floor_share
    return [share * total_weight for share in shares]


def count_nonempty_tiles(
    volume: Any,
    specs: Sequence[TileSpec],
    applied_floor: "float | None",
) -> int:
    """Count tiles that survive floor subtraction and Hann apodization.

    The floor, window, and predicate mirror :func:`fit_tile`'s skip decision,
    but the scan deliberately does not replay optional per-tile denoising. If
    the scan finds no signal at all, use the geometric count as the safe divisor
    solely to avoid division by zero; no tile will be fitted, so the divisor is
    otherwise moot.
    """
    nonempty = 0
    floor = 0.0 if applied_floor is None else applied_floor
    for spec in specs:
        tile_data = np.asarray(volume[spec.slices], dtype=np.float32)
        if float(tile_data.max()) - floor < _TILE_SIGNAL_EPS:
            continue
        if applied_floor is not None:
            tile_data = np.clip(tile_data - applied_floor, 0.0, None)
        tile_data = tile_data * cosine_window(spec)
        if tile_has_signal(tile_data):
            nonempty += 1
    if nonempty > 0:
        return nonempty
    if specs:
        aprint(
            "Seed-budget scan found no non-empty tiles; using the full grid "
            "count as the divisor."
        )
    return len(specs)


def uniform_tile_occupancy_weights(
    volume: Any,
    specs: Sequence[TileSpec],
    applied_floor: "float | None",
    *,
    intensity_scale: float = 1.0,
    saturation_exponent: float,
    min_share_fraction: float = MIN_TILE_SEED_SHARE_FRACTION,
) -> list[float]:
    """Measure one grid's Hann-weighted above-floor intensity-mass weights.

    Each tile weighs ``hann_voxels * (mass / hann_voxels / intensity_scale) **
    saturation_exponent`` (:func:`occupancy_weight_from_mass`) over its
    floor-subtracted, Hann-windowed data, then
    :func:`floor_tile_seed_shares` applies the minimum-share backstop. An empty
    tile weighs zero. A grid with no signal anywhere returns equal weights, the
    safe divisor of :func:`count_nonempty_tiles`.
    """
    mass_weights: list[float] = []
    masses: list[float] = []
    for spec in specs:
        tile_data = np.asarray(volume[spec.slices], dtype=np.float32)
        if applied_floor is not None:
            tile_data = np.clip(tile_data - applied_floor, 0.0, None)
        window = cosine_window(spec)
        mass = tile_windowed_mass(tile_data, window)
        masses.append(mass)
        mass_weights.append(
            occupancy_weight_from_mass(
                mass,
                float(np.sum(window, dtype=np.float64)),
                intensity_scale=intensity_scale,
                saturation_exponent=saturation_exponent,
            )
        )
    if not any(weight > 0.0 for weight in mass_weights):
        return [1.0] * len(specs)
    return floor_tile_seed_shares(
        mass_weights, masses, min_share_fraction=min_share_fraction
    )


def _validate_tile_seed_counts(
    tile_seed_counts: "Optional[Sequence[int]]", n_tiles: int
) -> None:
    """Validate an optional exact seed-count handoff against a tile grid."""
    if tile_seed_counts is None:
        return
    if len(tile_seed_counts) != n_tiles:
        raise ValueError(
            "tile_seed_counts length must match the resolved tile grid "
            f"({len(tile_seed_counts)} != {n_tiles})"
        )
    if any(count < 0 for count in tile_seed_counts):
        raise ValueError("tile_seed_counts entries must be non-negative")


def _reject_seed_coordinate_array(seeds: Any) -> None:
    """Refuse explicit seed COORDINATES on a tile — they are whole-volume."""
    if isinstance(seeds, np.ndarray):
        raise ValueError(
            "Explicit seed coordinate arrays are not supported with tiled fitting. "
            "Use an integer (count per tile), float (compression ratio), or None (auto)."
        )


def _is_zero_seed_budget(seeds: Any) -> bool:
    """Whether ``seeds`` is the integer 0, i.e. "this tile was budgeted nothing"."""
    return isinstance(seeds, int) and not isinstance(seeds, bool) and seeds == 0


def _reject_nonpositive_shared_seed_budget(
    seeds: Any, tile_seed_counts: "Optional[Sequence[int]]"
) -> None:
    """Refuse a non-positive SHARED ``seeds``, which is invalid user input.

    ``fit_tile`` reads ``seeds == 0`` as "this tile was budgeted nothing, skip
    it", which is right for an apportioned count but would silently swallow a
    ``--seeds 0``. The two are only distinguishable HERE: a ``seeds`` handed to
    every tile is the user's number, while ``tile_seed_counts`` overrides it per
    tile. Raised with the fitter's own wording, which is what used to surface
    (see ``split_seeds_across_tiles``, which passes a non-positive K through
    UNCHANGED precisely so the error is not rounded away).
    """
    if tile_seed_counts is not None:
        return
    if isinstance(seeds, int) and not isinstance(seeds, bool) and seeds <= 0:
        raise ValueError("seeds as int must be positive")


def _tile_fit_kwargs(
    fit_kwargs: dict[str, Any],
    tile_seed_counts: "Optional[Sequence[int]]",
    tile_index: int,
) -> dict[str, Any]:
    """Return fit arguments with an exact count for one tile when provided."""
    if tile_seed_counts is None:
        return fit_kwargs
    tile_kwargs = dict(fit_kwargs)
    tile_kwargs["seeds"] = int(tile_seed_counts[tile_index])
    return tile_kwargs


def _zero_splat_tile(ndim: int) -> GSplatData:
    """The unstamped 0-splat payload of a tile that was never fitted.

    The single spelling behind every way a tile can contribute nothing —
    :func:`fit_tile`'s near-zero-signal skip, :func:`_skipped_tile_result`, and
    the ``.empty`` marker a parallel worker leaves behind
    (:func:`~luxar.gsplats.fit_tiled_parallel._empty_tile`) — so they can never
    drift apart in shape or stats.
    """
    from luxar.gsplats.utils.trils import tril_size

    return GSplatData(
        centers=np.zeros((0, ndim), dtype=np.float32),
        amplitudes=np.zeros((0,), dtype=np.float32),
        cholesky_factors=np.zeros((0, tril_size(ndim)), dtype=np.float32),
        stats={"time_seconds": 0.0, "skipped": True},
    )


def _skipped_tile_result(
    spec: TileSpec, ndim: int, applied_floor: "float | None"
) -> GSplatData:
    """The 0-splat result a tile contributes when it is never fitted.

    Identical in shape and stats to what :func:`fit_tile` returns for a tile
    windowed to near-zero signal, so a tile skipped for a zero seed budget is
    indistinguishable downstream: the merge stays positionally aligned with the
    grid (which is what labels the partition's BSP tree), ``splats_per_tile``
    keeps its zero, and the normalization block still records the level that
    would have been subtracted. This is the in-process counterpart of the
    ``.empty`` marker a ``--allow-empty-tile`` batch worker writes.

    :func:`fit_tile` returns this for a zero ``seeds`` budget itself, so the
    contract holds wherever a zero budget can arrive (a ``--tile-seed-count 0``
    worker as much as the in-process loop); the loop's own check is then just a
    short-circuit that skips extracting the tile at all.
    """
    result = _zero_splat_tile(ndim)
    result.stats["tile_index"] = spec.index
    result.stats["tile_grid_index"] = spec.grid_index
    result.stats["tile_origin"] = spec.origin
    _stamp_tile_normalization(result.stats, applied_floor)
    return result


def _cull_keeping_measured_scores(data: GSplatData, retention: float) -> GSplatData:
    """The fit's closing amplitude trim, keeping the score it just measured.

    ``cull`` drops inherited reconstruction scores, because a score describes the
    splat set it was taken on (#1600). This trim is the last step of the FIT
    rather than a rewrite of a published artifact, and re-scoring would cost a
    second full render of the volume — so the measurement is carried across, the
    way it always was. See
    :func:`luxar.gsplats._data.filtering.content_scoped_stats` for why a later
    ``gsplat cull`` gets no such exemption.
    """
    from luxar.gsplats._data.filtering import (
        measured_stats_snapshot,
        restore_measured_stats,
    )

    measured = measured_stats_snapshot(data)
    out = data.cull(method="cumulative", retention=retention)
    return restore_measured_stats(out, measured)


def _tile_norm_range(
    volume: Any,
    fit_kwargs: dict[str, Any],
    applied_floor: "float | None",
    verbose: bool = False,
) -> "tuple[float, float] | None":
    """Shared ``(image_min, image_max)`` for every tile of ``volume``.

    The top comes from :func:`resolve_volume_norm_range_denoised` (whole volume,
    shifted onto the denoised and post-floor basis). The bottom is pinned at
    **zero**, which is where the
    array each tile fitter actually sees starts: the floor subtraction clips at
    0 and the Hann window then tapers every overlapped face down to 0.

    Pinning matters whenever the applied floor sits below the volume minimum —
    ``--floor none``, a floor the guard refused, or an explicit level under the
    pedestal. A positive ``image_min`` would then subtract a constant from BOTH
    sides of an overlap: ``(V*w_A - m) + (V*w_B - m) = V - 2m`` against ``V - m``
    in the tile interior, i.e. exactly the box-shaped seam this is meant to
    remove — and it would clip the taper away entirely below ``m``.

    Returns ``None`` — meaning "no shared scale; let each tile derive its own,
    as it did before there was a shared one" — in the three cases where the
    measurement carries no usable scale:

    * A non-finite top. ``NaN`` compares False against everything, so it would
      otherwise sail through as a "valid" scale and turn every tile's normalized
      array into NaN, past the raw-tile NaN check that already ran. (A ``-inf``
      *low* is not a case: the low end is discarded by the zero pin, and a tile
      actually holding it is rejected by that same check.)
    * A top at or below zero. The sample saw no signal at all — an empty or
      fully-masked region of a large volume the bounded sample happened to land
      in, or background-subtracted data whose positive structure it missed.
      Sharing it would hand every tile ``intensity_range <= 0``: at exactly zero
      :func:`~luxar.gsplats.fitting.preprocessing._normalize_data` fills the tile
      with a uniform 0.5 and fits a fabricated flat field, and below zero it
      inverts the sign and zeroes the real signal. A tile that genuinely holds
      nothing is skipped by the near-zero guard in :func:`fit_tile` either way,
      so nothing is lost by declining here.
    * The applied floor reaches the sampled top, so the shifted top is nothing
      but :func:`resolve_volume_norm_range`'s degenerate epsilon. Reachable
      through an unguarded numeric ``--floor``: the single-tile CLI worker
      applies a parent-resolved level (``--floor-resolved``) verbatim on
      purpose, so that level still applies to a dim timepoint, and the bounded
      sample can under-report the max. A genuine user numeric is guarded here
      like at every other entry point. A tile that really is below the floor
      clips to zero and is skipped by the near-zero guard — but a tile holding
      the signal the sample missed would be normalized by ~1e-12 and come back
      ~1e12 times too dark, which is far worse than losing cross-tile
      comparability where the shared measurement is meaningless anyway.

    That last test is deliberately narrow: it catches the epsilon the shift
    manufactures, not every unhelpfully small range. A volume that is honestly
    this dim (a float stack topping out at 1e-13, floor or not) keeps its shared
    scale, because normalizing by its own true extent is exactly right.
    """
    _, hi = resolve_volume_norm_range_denoised(
        volume,
        fit_kwargs.get("norm_percentile", 0.0),
        denoise_h=fit_kwargs.get("_denoise_h"),
        denoise_params=fit_kwargs.get("_denoise_params"),
        subtract=applied_floor,
        probe_cache=fit_kwargs.get("_denoise_probe_cache"),
        verbose=verbose,
    )
    if not np.isfinite(hi):
        aprint(
            f"Whole-volume intensity range is not finite ({hi}) — tiles fall "
            "back to their own scale."
        )
        return None
    if hi <= 0.0:
        aprint(
            f"Whole-volume intensity range has no positive extent ({hi:g}) — "
            "tiles fall back to their own scale."
        )
        return None
    if applied_floor is not None and hi <= NORM_RANGE_MIN_SPAN:
        aprint(
            f"Whole-volume intensity range collapsed under the floor "
            f"({applied_floor:g}) — tiles fall back to their own scale."
        )
        return None
    return (0.0, hi)


def _ensure_tile_norm_range(
    volume: Any,
    fit_kwargs: dict[str, Any],
    applied_floor: "float | None",
    verbose: bool = False,
) -> None:
    """Resolve the shared range into ``fit_kwargs`` unless someone already has.

    The orchestrator resolves it once and passes it down, so a per-tile worker
    does not re-read the volume; a standalone :func:`fit_tile` has nobody to
    inherit from and resolves it itself. The answer may legitimately be ``None``
    — no usable shared scale, each tile derives its own (see
    :func:`_tile_norm_range`) — which a plain ``norm_range is None`` test cannot
    tell from "nobody has looked yet", hence the private already-resolved
    marker. It is popped rather than forwarded, like the other
    underscore-prefixed tile-internal keys.
    """
    already_resolved = fit_kwargs.pop("_norm_range_resolved", False)
    if already_resolved:
        return
    supplied = fit_kwargs.get("norm_range")
    if supplied is None:
        fit_kwargs["norm_range"] = _tile_norm_range(
            volume, fit_kwargs, applied_floor, verbose=verbose
        )
        return

    # A CLI/batch-supplied range is expressed in RAW input units. Uniform tiles
    # see floor-subtracted, apodized data, whose bottom is always zero; shift only
    # the top into that basis. Keep the no-ceiling semantics in _normalize_data.
    hi = float(supplied[1]) - float(applied_floor or 0.0)
    if (
        not np.isfinite(hi)
        or hi <= 0.0
        or (applied_floor is not None and hi <= NORM_RANGE_MIN_SPAN)
    ):
        aprint(
            f"Supplied normalization range collapses after shifting by the floor "
            f"({float(applied_floor or 0.0):g}) — tiles fall back to their own scale."
        )
        fit_kwargs["norm_range"] = None
    else:
        fit_kwargs["norm_range"] = (0.0, hi)


def fit_tile(
    volume: Any,
    spec: TileSpec,
    voxel_size: Optional[Sequence[float] | float] = None,
    output_space: str = "real",
    progressive: bool = False,
    max_splats_per_pass: int = 5000,
    psnr_patience: float = 0.5,
    max_passes: Optional[int] = None,
    tile_data: Optional[np.ndarray] = None,
    **fit_kwargs: Any,
) -> GSplatData:
    """Fit Gaussian splats on a single tile of a larger volume.

    Extracts the tile subvolume, optionally denoises it, subtracts the
    background floor (resolved against the *whole* volume, never the tile, and
    on the denoised basis when denoising is active), applies cosine apodization,
    fits splats, and translates centers to global volume coordinates. This is
    the atomic unit for tiled fitting — each call is independent and
    Slurm-ready.

    Parameters
    ----------
    volume : np.ndarray or zarr.Array
        Full volume (or lazy zarr array). Only the tile's slice is materialized
        into memory via ``volume[spec.slices]``.
    spec : TileSpec
        Tile specification from :func:`compute_tile_specs`. Contains the
        per-face overlap sizes used for cosine window construction.
    voxel_size : float or sequence of float, optional
        Physical voxel spacing. Passed through to the per-tile fitter (both
        :func:`fit_gaussian_splats` and the progressive fitter) and used for
        correct center translation when ``output_space="real"``.
    output_space : str, default "real"
        Coordinate space for output centers (``"real"`` or ``"voxel"``).
    progressive : bool, default False
        If True, use progressive fitting (multiple passes on residuals)
        instead of standard single-pass fitting. The passes are an
        optimization schedule: each tile still returns one flat splat set.
    max_splats_per_pass : int, default 5000
        Maximum splats per progressive pass (ignored if progressive=False).
    psnr_patience : float, default 0.5
        Stop progressive passes if ΔPSNR < this value in dB.
    max_passes : int, optional
        Maximum number of progressive passes (None = unlimited).
    **fit_kwargs
        All other keyword arguments forwarded to the fitting function.
        ``seeds`` here is **per tile**: an integer is the count for THIS tile
        alone, and **0 is a skip**, not an error — the caller has budgeted this
        tile nothing, so a 0-splat :func:`_skipped_tile_result` comes back
        instead of the fitter's "seeds as int must be positive". The CLI's
        ``--seeds`` is a whole-volume budget, divided before reaching this
        function: normally into one exact occupancy-weighted count per tile
        (``tile_seed_counts`` on :func:`fit_tiled`, ``--tile-seed-count`` on a
        worker, both from
        ``luxar.cli.gsplat_ops.fitting.fit_utils._weighted_uniform_seed_counts``),
        and by the equal-share fallback
        ``luxar.cli.gsplat_ops.fitting.fit_utils.split_seeds_across_tiles``
        where no weighting ran (a hand-run ``--tile k/M``). A direct Python
        caller does that division itself if it wants the same semantics.
        ``floor`` (default ``"auto"``) is intercepted here: a spec string is
        resolved once against the whole ``volume`` via
        :func:`~luxar.gsplats.fitting.preprocessing.resolve_volume_floor_denoised`
        (so independent tile workers agree on one level), with the
        "would erase all signal" guard applied. With ``_denoise_h`` /
        ``_denoise_params`` in ``fit_kwargs`` that resolution happens on the
        DENOISED basis — the tile is denoised before the level is subtracted, so
        a raw-basis level would remove a different pedestal than the non-tiled
        path does (#1178) — wherever that shift is measurable: always within the
        denoise probe's budget, and above it for a ``pNN`` spec only (see
        :func:`~luxar.gsplats.fitting.preprocessing.resolve_volume_floor_denoised`).
        A numeric value is taken at
        face value — the caller is expected to have guarded it (as
        :func:`fit_tiled` does with ``guard_numeric=True``; the single-tile CLI
        worker deliberately does NOT, so one level resolved by its parent
        applies unchanged to a dim timepoint). The level is subtracted from the
        tile — after any denoising, before apodization; the two do not commute
        the other way — and the inner fit then runs with ``floor="none"``
        and the applied level is recorded in ``result.stats["floor"]`` (with
        ``image_min``/``image_max`` shifted back into the input volume's units).

    Returns
    -------
    GSplatData
        Fit result with centers in global volume coordinates (a single
        flat LOD; ``progressive=True`` changes how each tile is optimized,
        not the structure of the result).

    Raises
    ------
    ValueError
        If ``seeds`` in fit_kwargs is an explicit np.ndarray (not supported
        with tiled fitting; use int, float, or None instead).
    """
    # Reject explicit seed arrays — they don't make sense per-tile
    seeds = fit_kwargs.get("seeds")
    _reject_seed_coordinate_array(seeds)

    # Background floor: resolve the spec against the WHOLE volume (never the
    # tile) so every tile — including independent --tile k/M workers —
    # subtracts one identical, deterministic pedestal. A per-tile estimate
    # would be meaningless on apodized data and would subtract signal on a
    # densely labelled tile. Only str specs are validated here: a numeric
    # floor reaching this point is by contract an already-resolved level
    # taken at face value (possibly negative for dark-frame-corrected data),
    # not a user spec — _validate_floor guards user input and would reject
    # a legitimate negative resolved level.
    floor_spec = fit_kwargs.pop("floor", "auto")
    probe_cache = fit_kwargs.setdefault("_denoise_probe_cache", {})
    if isinstance(floor_spec, str):
        _validate_floor(floor_spec)
    # Denoising is applied to the tile BELOW, before the level is subtracted, so
    # a volume-derived spec must be resolved on the DENOISED basis or this path
    # removes a different pedestal than `--tiling none` does (#1178). The
    # denoise keys are only PEEKED at here: they are popped further down, after
    # the tile has been extracted.
    # Deliberately no `verbose=`: this is the PER-TILE door, so anything logged
    # here is logged once per tile. The two callers that resolve a spec once per
    # RUN do the announcing — `fit_tiled` below, and the standalone
    # `fit_single_tile` worker, which resolves the level itself and hands this
    # function the number. Same log surface as before #1178.
    applied_floor = resolve_volume_floor_denoised(
        volume,
        floor_spec,
        denoise_h=fit_kwargs.get("_denoise_h"),
        denoise_params=fit_kwargs.get("_denoise_params"),
        probe_cache=probe_cache,
    )

    # A ZERO integer budget is not an error here, it is a SKIP. It is what the
    # occupancy-weighted allocator says for "no signal in this tile", and it
    # reaches this function two ways: `fit_tiled`'s loop (which short-circuits
    # before extracting the tile) and a `--tile-seed-count 0` worker, which has
    # no loop to guard it. Handing the 0 down would hit the fitter's "seeds as
    # int must be positive" and kill that worker, and `--allow-empty-tile` would
    # not rescue it — that only covers SAVING a 0-splat result. Returning the
    # placeholder makes the contract hold wherever the budget arrives.
    if _is_zero_seed_budget(seeds):
        return _skipped_tile_result(spec, len(spec.shape), applied_floor)

    # Resolve the INTENSITY SCALE against the whole volume too, for the same
    # reason the floor is: a tile normalized by its own extremes is fitted
    # under criteria (convergence tolerance, seeding and culling thresholds)
    # that are all absolute in the normalized [0, 1] range, so the same
    # physical structure is resolved to a different accuracy in each tile.
    # Only resolve it when the caller has not already done so — see
    # `_ensure_tile_norm_range`. A standalone worker resolving its own is
    # exactly where you want the resolved number logged, to confirm that every
    # worker of a `--tile k/M` run agreed on it.
    _ensure_tile_norm_range(
        volume,
        fit_kwargs,
        applied_floor,
        verbose=bool(fit_kwargs.get("verbose", False)),
    )

    # 1. Extract tile subvolume (materializes from zarr if needed). Batch workers
    # may receive the exact region already materialized so host RAM stays tile-local.
    tile_data = (
        np.asarray(volume[spec.slices], dtype=np.float32)
        if tile_data is None
        else np.asarray(tile_data, dtype=np.float32)
    )

    # 1b. Denoise tile (if requested via fit_kwargs)
    # Use pop to remove denoise keys before forwarding to fitting functions
    _denoise_h = fit_kwargs.pop("_denoise_h", None)
    _denoise_params = fit_kwargs.pop("_denoise_params", None)
    fit_kwargs.pop("_denoise_probe_cache", None)
    if _denoise_h is not None and _denoise_params is not None:
        from arbol import asection as _asection

        from luxar.gsplats.preprocessing.denoise_pipeline import denoise_volume_array

        with _asection(f"Denoising tile {spec.index} (h={_denoise_h:.4f})"):
            tile_data = denoise_volume_array(tile_data, h=_denoise_h, **_denoise_params)

    # Pop cull_retention — per-tile culling is disabled (fit_tiled culls the merged result)
    fit_kwargs.pop("cull_retention", None)

    # 1c. Subtract the floor from the RAW tile, BEFORE apodization. The two
    # do not commute: subtracting m after windowing turns a two-tile overlap
    # (w_A + w_B = 1) into V - 2m instead of V - m, and clip(..., 0) erases
    # signal wherever V*w < m. No per-tile "floor >= tile max" guard on
    # purpose: a tile entirely below the global floor legitimately becomes
    # empty (handled by the near-zero skip below).
    if applied_floor is not None:
        tile_data = np.clip(tile_data - applied_floor, 0.0, None)
    # The pedestal is already gone — the inner fits must not subtract again.
    fit_kwargs["floor"] = "none"

    # 2. Apply cosine apodization window
    window = cosine_window(spec)
    tile_data = tile_data * window

    # 3. Skip fitting if tile has negligible signal (e.g., windowed to near-zero)
    if not tile_has_signal(tile_data):
        result = _zero_splat_tile(tile_data.ndim)
    elif progressive:
        from luxar.gsplats.fit_progressive_gsplats import (
            fit_progressive_gaussian_splats,
        )

        # Use seeds as max_splats budget for progressive fitting (don't pop — shared dict)
        prog_max_splats = fit_kwargs.get("seeds", 5000)
        if isinstance(prog_max_splats, float):
            # Compression ratio — let progressive handle it as seed count
            prog_max_splats = max(100, int(prog_max_splats * np.prod(tile_data.shape)))
        elif prog_max_splats is None:
            prog_max_splats = 5000

        # Forward voxel_size/output_space so the progressive fitter converts
        # centers AND Cholesky factors to physical coordinates when requested,
        # matching the non-progressive branch (it pops both from kwargs).
        # Map n_iters → iters_per_pass (progressive uses its own param name)
        # Check iters_per_pass first (direct callers), then n_iters (CLI path)
        prog_iters = fit_kwargs.pop("iters_per_pass", None)
        if prog_iters is None:
            prog_iters = fit_kwargs.pop("n_iters", 1000)
        result = fit_progressive_gaussian_splats(
            tile_data,
            max_splats=prog_max_splats,
            max_splats_per_pass=max_splats_per_pass,
            iters_per_pass=prog_iters,
            psnr_patience=psnr_patience,
            max_passes=max_passes,
            voxel_size=voxel_size,
            output_space=output_space,
            cull_retention=None,  # Disable per-tile; fit_tiled culls the merged result
            **fit_kwargs,
        )
    else:
        result = fit_gaussian_splats(
            tile_data,
            voxel_size=voxel_size,
            output_space=output_space,
            cull_retention=None,  # Disable per-tile; fit_tiled culls the merged result
            **fit_kwargs,
        )

    # 4. Translate centers from tile-local to global coordinates
    if result.n_splats > 0:
        origin = np.array(spec.origin, dtype=np.float32)
        # Both branches now return centers in the caller's requested space, so
        # scale the tile-origin offset to match: physical when output_space is
        # "real" with a voxel_size, raw voxel-space otherwise.
        if output_space == "real" and voxel_size is not None:
            vs = np.broadcast_to(
                np.asarray(voxel_size, dtype=np.float32), (len(origin),)
            )
            offset = origin * vs
        else:
            offset = origin

        result = result.translate(offset)

    # Tag tile info in stats
    result.stats["tile_index"] = spec.index
    result.stats["tile_grid_index"] = spec.grid_index
    result.stats["tile_origin"] = spec.origin
    _stamp_tile_normalization(result.stats, applied_floor)

    return result


def _stamp_tile_normalization(
    stats: dict[str, Any], applied_floor: "float | None"
) -> None:
    """Record a tile's normalization provenance in the VOLUME's units (#1175).

    Under the one key name the format spec uses (``floor``, not the old
    tiling-only ``applied_floor``). The pedestal was removed from the tile
    before the fit and the inner fit then ran with ``floor="none"``, so its own
    stats claim ``floor: None`` and measured ``image_min``/``image_max`` on the
    already-subtracted tile. Shifting those back by the level makes a tiled
    record mean what a flat one does, where ``image_min`` IS the applied level.

    The merge carries the block up only if every tile agrees — which they do,
    since :func:`fit_tiled` resolves one level and one ``norm_range`` for the
    whole volume.
    """
    stats["floor"] = applied_floor
    if applied_floor is None:
        return
    for bound in ("image_min", "image_max"):
        value = stats.get(bound)
        if isinstance(value, (int, float)) and not isinstance(value, bool):
            stats[bound] = float(value) + float(applied_floor)


def _stamp_merge_normalization(
    target: dict[str, Any],
    sources: "Sequence[Any]",
    applied_floor: "float | None",
) -> None:
    """Record a MERGE's normalization provenance on ``target`` (#1175).

    ``target`` is the flat merge's ``stats`` or the partition root node's
    ``meta``; ``sources`` are the per-tile results. The bounds come from the
    tiles (they agree by construction — one whole-volume ``norm_range``), which
    is also the only source the subprocess merge paths have, since they hand
    :func:`merge_tile_results` no ``applied_floor``. A level this call DID apply
    is authoritative and overrides.

    Nothing is written when neither the caller nor the tiles know: ``floor:
    null`` asserts that no pedestal was removed, while an absent key reads as
    "this artifact does not know", and only the second is honest here. Today's
    tiled callers always know (``fit_tiled`` resolves one level for the whole
    volume, and a tile store records its own), so this is a guard against a tile
    written by an older luxar rather than a routine outcome.
    """
    # Deferred: luxar.gsplats.io imports GSplatData from this package's
    # __init__, which is still executing when this module is first imported.
    from luxar.gsplats.io.save_gsplats import agreed_normalization_stats

    target.update(agreed_normalization_stats([r.stats for r in sources]))
    if applied_floor is not None:
        target["floor"] = applied_floor


def _empty_merge(
    volume_shape: tuple[int, ...],
    sources: "Sequence[Any]",
    applied_floor: "float | None",
) -> GSplatData:
    """A 0-splat merged result that still records what was subtracted (#1175).

    Reached when there were no tile results at all, or when culling emptied
    every one of them. Both used to return ``stats={}`` unconditionally, so an
    empty merge could not say what its tiles had subtracted even when the level
    was known — indistinguishable from a path that never records anything.
    """
    from luxar.gsplats.utils.trils import tril_size

    ndim = len(volume_shape)
    stats: dict[str, Any] = {}
    _stamp_merge_normalization(stats, sources, applied_floor)
    return GSplatData(
        centers=np.zeros((0, ndim), dtype=np.float32),
        amplitudes=np.zeros((0,), dtype=np.float32),
        cholesky_factors=np.zeros((0, tril_size(ndim)), dtype=np.float32),
        stats=stats,
    )


def _score_merged_if_reference(
    merged: "GSplatData | Sequence[GSplatData]",
    volume: "Any | None",
    *,
    volume_shape: tuple[int, ...],
    grid_scale: "tuple[float, ...] | None",
    device: Optional[str],
    verbose: bool,
    image_min: Optional[float],
    stats: "dict[str, Any] | None" = None,
) -> None:
    """Score a flat merge or partition parts when a reference is available."""
    if volume is None:
        return
    stamp_merged_quality(
        merged,
        volume,
        volume_shape=volume_shape,
        grid_scale=grid_scale,
        device=device,
        verbose=verbose,
        image_min=image_min,
        stats=stats,
    )


def _tiled_source_grid_stats(
    volume_shape: tuple[int, ...],
    source_shape: Optional[Sequence[int]],
    source_dtype: Optional[str],
    source_itemsize: Optional[int],
    source_stored_bytes: Optional[int] = None,
) -> dict[str, Any]:
    """Source-grid stamps for a MERGED tiled fit.

    A tiled fit needs its own record because the per-tile stamps describe crops:
    each tile is handed a sub-volume and honestly records that, so no tile knows
    the grid the merged result stands for. (``GSplatData.concatenate`` drops tile
    stats rather than promoting one, which is the only reason the merged result
    was not already claiming a single tile's crop as its source.)

    ``fitted_shape`` is the whole volume, not a tile: the tiles cover it, so the
    grid the optimiser collectively saw is the volume itself.
    """
    from luxar.gsplats.fitting.validation import (
        _explicit_source_shape,
        _explicit_source_stored_bytes,
    )

    declared = _explicit_source_shape(source_shape)
    # Normalized through the same gate the single-volume fit uses: this becomes
    # a published denominator either way, and a bare `int()` here would take
    # "1000" and round 1.5 to 1 on the one producer that skipped the check.
    stored_bytes = _explicit_source_stored_bytes(source_stored_bytes)
    dtype, itemsize = source_dtype, source_itemsize
    if itemsize is None and dtype:
        # A caller holding only the dtype NAME (the parallel orchestrator, whose
        # volume lives in the worker subprocesses) still gets a byte count: no
        # size means no compression ratio at all in `info`.
        try:
            itemsize = int(np.dtype(dtype).itemsize)
        except TypeError:  # an unrecognized dtype name — record it without a size
            itemsize = None

    shape = [int(x) for x in (declared if declared else volume_shape)]
    voxels = int(np.prod(shape)) if shape else 0
    out: dict[str, Any] = {
        "source_shape": shape,
        "source_voxels": voxels,
    }
    if dtype:
        out["source_dtype"] = dtype
    out.update(
        {
            "fitted_shape": [int(x) for x in volume_shape],
            "fitted_voxels": int(np.prod(volume_shape)) if volume_shape else 0,
        }
    )
    if declared:
        out["source_declared"] = True
    if itemsize:
        out["source_bytes"] = voxels * int(itemsize)
    if stored_bytes:
        out["source_stored_bytes"] = stored_bytes
    return out


def _merged_fit_stats(
    results: Sequence[GSplatData],
    *,
    delivered_splats: int,
    volume_shape: tuple[int, ...],
    tile_size: int | Sequence[int],
    overlap: int | Sequence[int],
    num_tiles: int,
    progressive: bool,
    elapsed: float,
    source_shape: Optional[Sequence[int]],
    source_dtype: Optional[str],
    source_itemsize: Optional[int],
    source_stored_bytes: Optional[int],
) -> dict[str, Any]:
    """Build the common root stats for flat and partition tile merges."""
    stats = {
        "tiled_fitting": True,
        "progressive": progressive,
        "num_tiles": num_tiles,
        "tile_size": tile_size,
        "overlap": overlap,
        "volume_shape": volume_shape,
        "time_seconds": elapsed,
        "splats_per_tile": [result.n_splats for result in results],
        "n_splats": delivered_splats,
    }
    stats.update(
        _tiled_source_grid_stats(
            volume_shape,
            source_shape,
            source_dtype,
            source_itemsize,
            source_stored_bytes,
        )
    )
    stamp_voxels_per_splat(stats, delivered_splats)
    return stats


def fit_tiled(
    volume: Any,
    tile_size: int | Sequence[int] = 256,
    overlap: int | Sequence[int] = 32,
    voxel_size: Optional[Sequence[float] | float] = None,
    output_space: str = "real",
    verbose: bool = True,
    progressive: bool = False,
    max_splats_per_pass: int = 5000,
    psnr_patience: float = 0.5,
    max_passes: Optional[int] = None,
    cull_retention: float | None = 0.95,
    partition: bool = False,
    recipe: Optional[str] = None,
    recipe_params: "Optional[Any]" = None,
    fold_tile_slivers: bool = True,
    tile_seed_counts: "Optional[Sequence[int]]" = None,
    # Taken explicitly rather than left in **fit_kwargs: forwarded to the tiles,
    # they would declare the WHOLE acquisition as each crop's source. They
    # describe the merged result, so the merge is where they are applied.
    source_shape: Optional[Sequence[int]] = None,
    source_dtype: Optional[str] = None,
    source_stored_bytes: Optional[int] = None,
    **fit_kwargs: Any,
) -> "Any":
    """Fit Gaussian splats to a large volume using tiled decomposition.

    Splits the volume into overlapping tiles with cosine apodization
    (Hann window), fits each tile independently, and merges results.
    The background floor (``floor`` in ``fit_kwargs``, default ``"auto"``)
    is resolved once against the whole volume and subtracted from each tile
    before windowing; on the floor-subtracted data the Hann
    partition-of-unity property guarantees seamless blending. When per-tile
    denoising is active (``_denoise_h`` / ``_denoise_params``), the level is
    resolved on the denoised basis, matching the non-tiled path (#1178) — with
    the one documented exception that the default ``"auto"`` spec on a volume
    above the denoise probe's budget keeps its raw-basis level, since the
    histogram-mode shift is not measurable on a bounded crop.

    When ``progressive=True``, each tile is fitted in several residual
    passes; the progressive fitter flattens its passes before returning, so
    the tiles are concatenated exactly as in the single-pass case. (Tile
    results that do carry additive sub-LODs are merged level by level.)

    Parameters
    ----------
    volume : np.ndarray or zarr.Array
        Full volume. Can be a lazy zarr array for out-of-core processing —
        only one tile at a time is materialized in memory.
    tile_size : int or tuple of int, default 256
        Tile size per axis in voxels. Scalar is broadcast to all axes.
    overlap : int or tuple of int, default 32
        Overlap width per axis in voxels. Scalar is broadcast.
    voxel_size : float or sequence of float, optional
        Physical voxel spacing, forwarded to per-tile fitting.
    output_space : str, default "real"
        Coordinate space for output centers (``"real"`` or ``"voxel"``).
    verbose : bool, default True
        Print per-tile progress with arbol.
    progressive : bool, default False
        Optimize each tile in several residual passes. The fitter flattens those
        passes into one splat set before the tiles are merged.
    max_splats_per_pass : int, default 5000
        Maximum splats per progressive pass (ignored if progressive=False).
    psnr_patience : float, default 0.5
        Stop progressive passes if ΔPSNR < this value in dB.
    max_passes : int, optional
        Maximum number of progressive passes (None = unlimited).
    fold_tile_slivers : bool, default True
        Fold a trailing tile whose unique coverage is smaller than the overlap
        into its predecessor, instead of emitting an overlap-dominated sliver.

        **This CHANGES OUTPUT versus pre-#2838**, where the default was
        ``False``. There is now exactly ONE uniform grid: this function, the
        ``fit --tiling uniform`` sequential and ``-j N`` paths, the
        ``fit --tile k/M`` worker, and every ``batch-fit`` producer all build
        it. On ``(108, 1352, 532)`` at ``tile_size=512, overlap=32`` that is 3
        tiles, where the unfolded grid had 6 — so the same call on the same
        volume now fits different regions with different per-tile budgets, and
        the result differs from a pre-#2838 one by more than a ``content_hash``.
        Pass ``False`` to reproduce a historical grid.

        A folded tile also EXCEEDS ``tile_size``: it spans up to
        ``tile_size + overlap - 1`` voxels on the folded axis, so peak per-tile
        memory is that much above what ``tile_size`` alone suggests (256/32 ->
        287, 1.41x the voxels in 3D; 256/64 -> 319, 1.93x; 24/8 -> 31, 2.16x).
        Size ``tile_size`` for that worst case.
    tile_seed_counts : sequence of int, optional
        Exact per-tile integer seed counts in grid order. When provided, this
        overrides ``seeds`` for each tile; its length must match the resolved
        grid. This is the CLI handoff for occupancy-weighted whole-volume
        budgets. A ``0`` entry means "this tile holds no signal": the tile is
        NOT fitted and contributes a 0-splat placeholder, mirroring the
        ``--allow-empty-tile`` batch worker (the inner fitter rejects a
        non-positive integer ``seeds``).
    cull_retention : float or None, default=0.95
        Post-fit cumulative culling on the merged result.  Keeps the top
        splats that account for this fraction of total amplitude (0--1).
        Per-tile culling is disabled automatically; only the merged result
        is culled.  Set to ``None`` to disable.
    source_shape : sequence of int, optional
        Grid of the ACQUISITION, when ``volume`` is already a preprocessed copy
        of it — a caller that decimated before tiling must declare it, or the
        merged result records the working copy as its source and the compression
        ratio is quoted against a grid the data never had. ``None`` measures
        ``volume`` itself, which is right whenever nothing was preprocessed.
    source_dtype : str, optional
        Element type the volume was STORED in, for the same reason as in
        :func:`~luxar.gsplats.fit_gsplats.fit_gaussian_splats`. Applied to the
        MERGED result rather than forwarded to the tiles: a tile would use it to
        describe its own crop.
    **fit_kwargs
        All other keyword arguments forwarded to the per-tile fitting function
        (e.g. ``seeds``, ``n_iters``, ``preset``, ``device``,
        ``residual_pass_min_iters`` when ``progressive=True``).
        ``seeds`` is handed to EVERY tile as-is, so an integer here is a
        **per-tile** count, not a whole-volume budget: N tiles fit ~N x seeds
        splats — unless ``tile_seed_counts`` overrides it per tile, which is
        what the CLI normally passes. The CLI's ``--seeds`` IS a whole-volume
        budget, divided before this call into one exact occupancy-weighted
        count per tile
        (``luxar.cli.gsplat_ops.fitting.fit_utils._weighted_uniform_seed_counts``
        → ``tile_seed_counts``), falling back to the equal share
        ``luxar.cli.gsplat_ops.fitting.fit_utils.split_seeds_across_tiles``
        where no weighting ran; a direct Python caller that wants the same
        semantics divides itself.

    Returns
    -------
    GSplatData
        Merged result with all splats in global coordinates.
        Progressive fitting changes the optimization schedule, not the result's
        LOD structure. The merged reconstruction is also scored against the whole
        volume. Metrics land in ``stats`` for a flat result and in the root node's
        in-memory ``meta["fit_stats"]`` for a tree, ready for the CLI writer to
        persist at the store root.

    Notes
    -----
    **Merged quality metrics**: the per-tile scores describe crops of an
    apodized decomposition and do not compose, so the merged reconstruction is
    rendered once against ``volume`` and scored. Scoring materializes the whole
    volume, so separate host-reference and render-device peaks are bounded by
    half the memory actually free, each held under a 24 GiB ceiling. Concurrent
    local workers divide the default host allowance across the run and the
    default device allowance across the workers on their card.
    ``LUXAR_TILED_QUALITY_MAX_GB`` overrides both budgets (``0`` declines
    outright). Over budget, or on a failure, it says so even when
    ``verbose=False``. A partition is scored by rendering each surviving
    tile-part and summing the volumes in place, matching how the viewer composes
    the parts without flattening or copying the full splat set.

    **GPU utilization with progressive**: When ``progressive=True``, each
    per-pass fit uses fewer splats (``max_splats_per_pass``), which may
    under-saturate the GPU.  For batch/Slurm jobs, combine
    ``--progressive`` with ``--parallel`` to run multiple tiles concurrently
    on the same GPU and improve throughput.
    """
    # Propagate verbose to per-tile fit_gaussian_splats unless caller
    # explicitly provided a different value in fit_kwargs.
    fit_kwargs.setdefault("verbose", verbose)

    # Resolve the background floor ONCE for the whole run and hand every tile
    # the same concrete level (no tile re-scans, no per-tile drift). This is
    # where a USER-supplied spec becomes a level, so a numeric spec is guarded
    # against "floor >= max erases everything" too (matching the non-tiled
    # path, which warns and ignores such a floor).
    floor_spec = fit_kwargs.pop("floor", "auto")
    probe_cache = fit_kwargs.setdefault("_denoise_probe_cache", {})
    floor_already_resolved = fit_kwargs.pop("_floor_resolved", False)
    if floor_already_resolved:
        applied_floor = None if floor_spec == "none" else float(floor_spec)
    else:
        _validate_floor(floor_spec)
        # Every tile is denoised before the level is subtracted, so a
        # volume-derived spec is resolved on the DENOISED basis (#1178) —
        # otherwise this path removes a different pedestal than `--tiling none`
        # does on the same input. PEEK at the denoise keys: `fit_kwargs` is
        # forwarded to `fit_tile`, which pops them itself.
        applied_floor = resolve_volume_floor_denoised(
            volume,
            floor_spec,
            denoise_h=fit_kwargs.get("_denoise_h"),
            denoise_params=fit_kwargs.get("_denoise_params"),
            probe_cache=probe_cache,
            guard_numeric=True,
            # Log the DENOISED basis the level was resolved on (raw level + the
            # measured shift) — the one number the summary line below cannot show.
            # Gated on a VOLUME-DERIVED spec (`auto`/`pNN`) and on denoising actually
            # being active, so `--denoise` off, and any absolute level, print exactly
            # what they printed before #1178. `isinstance(str)` would not do: the CLI
            # hands `--floor 110` down as the STRING "110", which is an absolute the
            # summary line below already echoes.
            verbose=(
                verbose
                and _floor_spec_is_volume_derived(floor_spec)
                and fit_kwargs.get("_denoise_h") is not None
                and fit_kwargs.get("_denoise_params") is not None
            ),
        )
    if verbose and applied_floor is not None:
        aprint(
            f"Floor suppression: subtracting background level "
            f"{applied_floor:.6g} from every tile"
        )
    fit_kwargs["floor"] = applied_floor if applied_floor is not None else "none"

    # Same treatment for the intensity scale: resolve ONCE here so every tile
    # shares it, rather than letting each tile normalize by its own extremes.
    # `fit_tile` would resolve it per tile otherwise — identical result, but a
    # bounded volume read per tile instead of one. `_ensure_tile_norm_range`
    # consults the already-resolved marker FIRST, which is the only way to tell a
    # caller-DECLINED range (None, see `_tile_norm_range`) from "nobody has looked
    # yet" — an upstream decline re-measured here would re-read the volume and
    # re-print its own warning. The marker is then re-set for the per-tile
    # `fit_tile` calls below (it is popped, not forwarded).
    _ensure_tile_norm_range(volume, fit_kwargs, applied_floor, verbose=verbose)
    fit_kwargs["_norm_range_resolved"] = True

    volume_shape = tuple(volume.shape)
    specs = compute_tile_specs(
        volume_shape, tile_size, overlap, fold_slivers=fold_tile_slivers
    )
    _validate_tile_seed_counts(tile_seed_counts, len(specs))
    _reject_nonpositive_shared_seed_budget(fit_kwargs.get("seeds"), tile_seed_counts)

    results: list[GSplatData] = []
    t0 = time.perf_counter()

    with asection(
        f"Tiled fitting: {len(specs)} tiles, tile_size={tile_size}, overlap={overlap}"
    ):
        if verbose:
            aprint(f"Volume shape: {volume_shape}")
            aprint(f"Tile grid: {_grid_shape(specs, len(volume_shape))}")

        for spec in specs:
            label = f"Tile {spec.index + 1}/{len(specs)} grid={spec.grid_index}"
            if tile_seed_counts is not None and int(tile_seed_counts[spec.index]) == 0:
                # A zero budget is how the occupancy-weighted allocator says
                # "no signal here" — it never zeroes a tile that holds any.
                # Handing that 0 to `fit_tile` would reach the fitter's
                # "seeds as int must be positive" on any tile that does, and
                # this in-process loop has no per-tile `allow_empty_tile`
                # escape hatch the way a batch worker subprocess does.
                results.append(
                    _skipped_tile_result(spec, len(volume_shape), applied_floor)
                )
                if verbose:
                    aprint(f"{label}: zero seed budget, not fitted")
                continue
            with asection(label):
                tile_result = fit_tile(
                    volume,
                    spec,
                    voxel_size=voxel_size,
                    output_space=output_space,
                    progressive=progressive,
                    max_splats_per_pass=max_splats_per_pass,
                    psnr_patience=psnr_patience,
                    max_passes=max_passes,
                    **_tile_fit_kwargs(fit_kwargs, tile_seed_counts, spec.index),
                )
                n = tile_result.n_splats
                if verbose:
                    t_tile = tile_result.stats.get("time_seconds", 0)
                    aprint(f"{n:,} splats ({t_tile:.1f}s)")
                results.append(tile_result)

    elapsed = time.perf_counter() - t0
    # Resolved HERE, where the caller's array is still in hand: after the tiles
    # are fitted only their crops remain, and the merge cannot recover the
    # element type the volume was stored in.
    from luxar.gsplats.fitting.validation import _resolve_source_dtype

    merged_dtype, merged_itemsize = _resolve_source_dtype(volume, source_dtype)
    grid_scale = resolve_grid_scale(
        len(volume_shape),
        voxel_size=voxel_size,
        output_space=output_space,
    )
    merged = merge_tile_results(
        results,
        volume_shape=volume_shape,
        source_shape=source_shape,
        source_dtype=merged_dtype,
        source_itemsize=merged_itemsize,
        source_stored_bytes=source_stored_bytes,
        tile_size=tile_size,
        overlap=overlap,
        num_tiles=len(specs),
        progressive=progressive,
        cull_retention=cull_retention,
        elapsed=elapsed,
        verbose=verbose,
        partition=partition,
        recipe=recipe,
        recipe_params=recipe_params,
        applied_floor=applied_floor,
        # The grid above is in voxels; with a voxel_size and real-space output
        # every tile's splats were offset by `origin * voxel_size`, so the
        # partition's split planes need the same factor (#1587).
        grid_scale=grid_scale,
        volume=volume,
        device=fit_kwargs.get("device"),
        fold_tile_slivers=fold_tile_slivers,
    )
    return merged


def merge_tile_results(
    results: list[GSplatData],
    *,
    volume_shape: tuple[int, ...],
    tile_size: int | Sequence[int],
    overlap: int | Sequence[int],
    num_tiles: int,
    progressive: bool,
    cull_retention: float | None,
    elapsed: float,
    verbose: bool = True,
    partition: bool = False,
    recipe: Optional[str] = None,
    recipe_params: "Optional[Any]" = None,
    applied_floor: "float | None" = None,
    grid_scale: "tuple[float, ...] | None" = None,
    source_shape: Optional[Sequence[int]] = None,
    source_dtype: Optional[str] = None,
    source_itemsize: Optional[int] = None,
    source_stored_bytes: Optional[int] = None,
    volume: "Any | None" = None,
    device: Optional[str] = None,
    fold_tile_slivers: bool = True,
) -> "Any":
    """Merge per-tile fit results, preserving any additive ladders they carry.

    Shared by both the sequential :func:`fit_tiled` loop and the parallel
    orchestrator in ``fit_tiled_parallel``. Concatenates tile results, stamps
    tiled-fitting stats, and applies a single post-fit cumulative cull on the
    merged result.

    With ``partition=True`` (the CLI default) the tiles are kept as a
    ``kind=partition`` tree — one part per (Hann-apodized) tile, which sum
    correctly as additive parts — for viewer frustum culling; culling is applied
    per-tile and a :class:`~luxar.gsplats.tree.GSplatNode` is returned. With
    ``partition=False`` the tiles are concatenated into one flat leaf (``--flat``)
    and culled globally.

    Parameters
    ----------
    results : list of GSplatData
        Per-tile fit results, already translated to global coordinates. May
        be empty.
    volume_shape : tuple of int
        Full (possibly downscaled) volume shape, recorded in stats and used to
        build an empty result when ``results`` is empty.
    tile_size, overlap : int or sequence of int
        Tiling geometry, recorded in stats.
    num_tiles : int
        Number of tiles in the grid (``len(specs)``).
    progressive : bool
        Whether tiles were fit progressively, recorded in the merged stats.
    cull_retention : float or None
        Post-fit cumulative culling fraction on the merged result (0--1).
        ``None`` or outside (0, 1) disables culling.
    elapsed : float
        Wall-clock seconds for the fitting stage, recorded in stats.
    verbose : bool, default True
        Print a summary line via arbol.
    applied_floor : float or None, default None
        The background level subtracted from every tile (after any denoising)
        before apodization. Supplied by the sequential :func:`fit_tiled` path; the
        subprocess-based paths leave it ``None`` and the level is recovered from
        the tiles' own stats instead. Recorded as ``stats["floor"]`` on the flat
        merge and as the root node's ``meta["floor"]`` on the ``partition=True``
        path, which :func:`~luxar.gsplats.io.save_gsplats.write_gsplats_tree`
        promotes into the store's ``pipeline/`` group (#1175).
    grid_scale : tuple of float or None, default None
        Per-axis factor mapping the tile grid's VOXEL frame (``volume_shape``,
        ``tile_size``, ``overlap``) onto the frame ``results`` carry their
        centers in — the product of the ``--downscale`` factors and, when the
        fit emitted real-space output, the ``voxel_size`` (issue #1587). Build
        it with :func:`~luxar.gsplats.tiling.resolve_grid_scale`. Consumed only
        by the partition's split planes
        (:func:`~luxar.gsplats.tiling.grid_bsp_tree`), which would otherwise be
        a factor too small and would no longer separate the parts they label,
        and by merged-quality scoring to render back on the reference grid.
        ``None`` when the grid and the splats share one frame.
    fold_tile_slivers : bool, default True
        The uniform-grid geometry ``results`` were fitted on, replayed here to
        label the partition's split planes. Must match whatever built the tiles
        — every Luxar producer now folds (#2838), so the default agrees with
        them; pass ``False`` only to merge tiles fitted on a historical
        unfolded grid.
    source_shape, source_dtype, source_itemsize : optional
        What the merged result is a representation of, stamped by
        :func:`_tiled_source_grid_stats` — no tile can say, since each was handed
        a crop. ``source_shape`` is for a caller that decimated before tiling
        (``volume_shape`` is then the fitted grid, not the acquisition); the
        dtype/itemsize are the element type the volume was STORED in, without
        which no compression ratio can be quoted. A partition carries the merged
        block in its root ``meta["fit_stats"]``; :func:`save_fit_output` routes
        that block through the same fitting/pipeline split as a flat leaf.

    Returns
    -------
    GSplatData
        Merged result, preserving additive sub-LODs when the inputs carry them.
    """
    if len(results) == 0:
        return _empty_merge(volume_shape, results, applied_floor)

    # Partition: keep one part per tile (frustum culling). Apodized tiles sum
    # correctly as additive parts; cull each tile independently (the flat path's
    # single global cull has no meaning once tiles stay separate parts). Each
    # region's `.tree` preserves any additive ladder already carried by that tile.
    if partition:
        # Track each surviving region's TILE index alongside it: two independent
        # filters run below (empty tiles, then tiles a cull empties), so position
        # in `regions` is not the tile index the grid tree is labelled by.
        indexed = [(i, r) for i, r in enumerate(results) if r.n_splats > 0]
        if cull_retention is not None and 0 < cull_retention < 1.0:
            # Each tile carries the score its own fit measured, and this is that
            # fit's closing trim — so the measurement rides across it.
            indexed = [
                (i, _cull_keeping_measured_scores(r, cull_retention))
                for i, r in indexed
            ]
            indexed = [(i, r) for i, r in indexed if r.n_splats > 0]
        regions = [r for _, r in indexed]
        if not regions:
            return _empty_merge(volume_shape, results, applied_floor)
        # The grid this merge re-derives has to be the one the tiles were fitted
        # on, and `fold_tile_slivers` is the only thing that can make it differ.
        # An external caller that built an UNFOLDED grid itself and did not say
        # so now gets the folded default, i.e. fewer leaves than labelled
        # regions. `prune_serialized_bsp_tree` catches that only while the kept
        # labels are NOT a subset of the smaller tree's — once enough tiles come
        # back empty they are, and the planes are silently relabelled from the
        # wrong geometry. Check the recomputed grid too, the way
        # `merge_orchestrator._uniform_slot_bsp_tree` does.
        grid_specs = compute_tile_specs(
            volume_shape, tile_size, overlap, fold_slivers=fold_tile_slivers
        )
        node = GSplatData.partition_from_regions(
            regions,
            recipe=recipe,
            recipe_params=recipe_params,
            # Grid split planes so the viewer paints tile-parts far-side-first
            # instead of by centroid (#1555). APPROXIMATE here — apodized tiles
            # keep their overlap band, so neighbours genuinely share space and the
            # cut is the midplane of that band (see `grid_bsp_tree`). Only sound
            # while `results` is positionally aligned with the tile grid, which
            # both callers guarantee by inserting a 0-splat placeholder for a tile
            # that fit nothing; a length mismatch means that no longer holds, so
            # drop the tree rather than mislabel it.
            # `scale` lifts the (possibly downscaled) voxel grid into the
            # splats' own frame — see `grid_scale` above (#1587).
            bsp_tree=(
                grid_bsp_tree(grid_specs, scale=grid_scale)
                if len(results) == num_tiles and len(grid_specs) == num_tiles
                else None
            ),
            region_labels=[i for i, _ in indexed],
        )
        # A partition has no flat stats dict, so the block rides on the ROOT
        # node's meta and `save_fit_output` splits it into the store's root
        # fitting/config/provenance/pipeline groups.
        _stamp_merge_normalization(node.meta, regions, applied_floor)
        from luxar.gsplats.tree import total_splats

        fit_stats = _merged_fit_stats(
            results,
            delivered_splats=int(total_splats(node)),
            volume_shape=volume_shape,
            tile_size=tile_size,
            overlap=overlap,
            num_tiles=num_tiles,
            progressive=progressive,
            elapsed=elapsed,
            source_shape=source_shape,
            source_dtype=source_dtype,
            source_itemsize=source_itemsize,
            source_stored_bytes=source_stored_bytes,
        )
        _score_merged_if_reference(
            regions,
            volume,
            volume_shape=volume_shape,
            grid_scale=grid_scale,
            device=device,
            verbose=verbose,
            image_min=fit_image_min(node.meta),
            stats=fit_stats,
        )
        node.meta["fit_stats"] = fit_stats
        if verbose:
            lod_note = f", per-part recipe={recipe}" if recipe else ""
            aprint(
                f"Total: {sum(r.n_splats for r in regions):,} splats from "
                f"{len(regions)} tile-parts (partition{lod_note}) in {elapsed:.1f}s"
            )
        return node

    # Merge tile results, preserving additive ladders when present.
    has_lods = any(r.n_additive_sublods > 1 for r in results)
    if has_lods:
        merged = _merge_lods_across_tiles(results)
    else:
        merged = GSplatData.concatenate(results)

    if verbose:
        lod_info = f", {merged.n_additive_sublods} LODs" if has_lods else ""
        aprint(
            f"Total: {merged.n_splats:,} splats from {num_tiles} tiles "
            f"in {elapsed:.1f}s{lod_info}"
        )

    # Post-fit cumulative culling on merged result
    if cull_retention is not None and 0 < cull_retention < 1.0 and merged.n_splats > 0:
        n_before = merged.n_splats
        merged = _cull_keeping_measured_scores(merged, cull_retention)
        if verbose and merged.n_splats < n_before:
            aprint(
                f"Post-fit culling: {n_before} -> {merged.n_splats} splats "
                f"(retained {cull_retention * 100:.0f}% of amplitude)"
            )

    merged.stats.update(
        _merged_fit_stats(
            results,
            delivered_splats=merged.n_splats,
            volume_shape=volume_shape,
            tile_size=tile_size,
            overlap=overlap,
            num_tiles=num_tiles,
            progressive=progressive,
            elapsed=elapsed,
            source_shape=source_shape,
            source_dtype=source_dtype,
            source_itemsize=source_itemsize,
            source_stored_bytes=source_stored_bytes,
        )
    )
    _stamp_merge_normalization(merged.stats, results, applied_floor)

    _score_merged_if_reference(
        merged,
        volume,
        volume_shape=volume_shape,
        grid_scale=grid_scale,
        device=device,
        verbose=verbose,
        image_min=fit_image_min(merged.stats),
    )

    return merged


def _merge_lods_across_tiles(results: list[GSplatData]) -> GSplatData:
    """Merge LODs across tiles: LOD N = concat of all tiles' LOD N.

    Pads to the maximum LOD count — tiles that stopped early simply
    contribute nothing to higher LODs.
    """
    # Delegates to LOD-aware concatenate() which handles per-LOD merging
    # and mixed LOD counts automatically.
    return GSplatData.concatenate(results)


def _grid_shape(specs: list[TileSpec], ndim: int) -> tuple[int, ...]:
    """Compute the grid shape from tile specs."""
    if not specs:
        return tuple([0] * ndim)
    max_grid = [0] * ndim
    for spec in specs:
        for d in range(ndim):
            max_grid[d] = max(max_grid[d], spec.grid_index[d] + 1)
    return tuple(max_grid)
