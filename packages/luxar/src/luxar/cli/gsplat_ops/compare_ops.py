"""Implementation of ``luxar gsplat compare``.

The command renders a store back onto a voxel grid and scores it against a
reference volume. Getting a meaningful number out of that takes four
reconciliations, each of which used to be the user's problem:

* **Basis.** A render is background-relative (``V - image_min``); the reference
  is shifted onto that basis first (#1173). A ``transform --scale-intensity``
  edit (the store's ``intensity_scale``) is divided back out of the amplitudes,
  so the render is in the reference's units and a display rescale is not scored
  as reconstruction error.
* **Grid.** A physical-output fit's centers are in physical units; they are
  divided back by the stored ``voxel_size`` (or ``--voxel-size``) so they land on
  the reference's voxel grid. ``--at-fit-resolution`` additionally undoes a
  ``--downscale`` fit's rescale and decimates the reference the way the fit did.
* **Peak.** The headline PSNR uses a robust peak (reference p99.99 minus the
  floor) that one hot pixel cannot move; the classical full-range figure is
  reported beside it, labelled.
* **Foreground.** Foreground PSNR/SSIM are taken over a smoothed-Otsu mask by
  default, or over ``--mask`` / ``--fg-threshold``.
* **Region.** A ``--region`` crop fit is scored on the region it fitted when the
  reference is the full volume (``--whole-volume`` scores all of it), so the
  foreground mask is the one the fit stamped.
* **Noise.** The foreground PSNR is put next to the score a noise-free model
  would get against this noisy reference (``20 log10(peak / sigma)``, sigma
  measured in the foreground), with a one-line verdict.
"""

from __future__ import annotations

import contextlib
import json
import math
from dataclasses import dataclass, field
from pathlib import Path
from typing import TYPE_CHECKING, Any, Optional

import typer
from arbol import aprint, asection

from .._traceback import exit_with_error
from ..utils import store_size

if TYPE_CHECKING:
    import numpy as np

    from luxar.gsplats.gsplat_data import GSplatData

#: Printed when a store keeps ``image_min`` but records no ``intensity_range``.
_NO_AMPLITUDE_BASIS_HINT = (
    "the dataset records image_min but no amplitude basis (intensity_range): an "
    "amplitude edit that is not one global factor (an offset, a clamp, a "
    "per-splat reweight) or a merge of differently scaled fits dropped it, so "
    "the stored amplitudes are no longer in source units and the scores below "
    "include that difference. Score the store from before the edit."
)

#: Below this share of splat centers inside the reference grid, the command warns
#: that the splats are probably not in the reference's voxel frame.
_INSIDE_WARN_FRACTION = 0.5


@dataclass
class CompareOptions:
    """Every ``gsplat compare`` option, resolved from the command line."""

    gsplats_path: Path
    reference_path: Path
    shape: Optional[str] = None
    device: Optional[str] = None
    truncate: Optional[float] = None
    image_min: Optional[float] = None
    channel: Optional[int] = None
    timepoint: Optional[int] = None
    array_key: Optional[str] = None
    axes: Optional[str] = None
    mask: Optional[Path] = None
    fg_threshold: Optional[float] = None
    voxel_size: Optional[str] = None
    at_fit_resolution: bool = False
    whole_volume: bool = False
    output_json: Optional[Path] = None
    quiet: bool = False


@dataclass
class _ScoringGrid:
    """How the store's splats were mapped onto the grid that is scored."""

    voxel_size: Optional[tuple[float, ...]] = None
    voxel_size_source: Optional[str] = None
    downscale_factors: Optional[tuple[int, ...]] = None

    @property
    def scale(self) -> Optional[tuple[float, ...]]:
        """Center units per scored voxel (``voxel_size * factors``), or ``None``."""
        if self.voxel_size is None and self.downscale_factors is None:
            return None
        ndim = len(self.voxel_size or self.downscale_factors or ())
        vs = self.voxel_size or (1.0,) * ndim
        fs = self.downscale_factors or (1,) * ndim
        return tuple(float(v) * int(f) for v, f in zip(vs, fs, strict=True))

    def origin(
        self, region_start: Optional[tuple[int, ...]]
    ) -> Optional[tuple[float, ...]]:
        """A crop's start on the scored grid (source voxels / downscale factors)."""
        if region_start is None:
            return None
        fs = self.downscale_factors or (1,) * len(region_start)
        return tuple(s / f for s, f in zip(region_start, fs, strict=True))


@dataclass
class _Context:
    """What was compared, for the table and the JSON record."""

    n_splats: int = 0
    ndim: int = 0
    ref_shape: tuple[int, ...] = ()
    render_shape: tuple[int, ...] = ()
    image_min: Optional[float] = None
    #: The store's recorded ``intensity_scale`` the render was divided by.
    intensity_scale: float = 1.0
    grid: _ScoringGrid = field(default_factory=_ScoringGrid)
    foreground_method: str = "otsu_smoothed"
    dimension_metadata: Any = None
    foreground_threshold_raw: Optional[float] = None
    #: The reference's decoded size in its STORED element type (the loader hands
    #: back float32 whatever the file holds): the same ``source_bytes`` basis
    #: ``gsplat info`` and ``fit`` quote. Taken before any decimation.
    ref_decoded_bytes: int = 0
    #: The ``--region`` crop of the reference that is scored (source voxels), or
    #: ``None`` when the whole reference is.
    scored_region: Optional[tuple[tuple[int, int], ...]] = None
    #: The reference's shape as loaded, before any crop or decimation.
    loaded_shape: tuple[int, ...] = ()
    #: Region of a crop fit scored with ``--whole-volume`` (source voxels).
    fitted_region: Optional[tuple[tuple[int, int], ...]] = None


def silenced_when(quiet: bool) -> contextlib.AbstractContextManager[None]:
    """Silence all arbol output for the block when ``quiet``; else a no-op.

    Errors still reach stderr: ``exit_with_error`` falls back to it whenever arbol
    output is hidden.
    """
    if not quiet:
        return contextlib.nullcontext()
    from luxar.utils.verbosity import verbosity

    return verbosity("silent")


def validate_image_min_override(image_min: Optional[float]) -> None:
    """Refuse a ``--image-min`` no fit could have subtracted (usage error)."""
    if image_min is not None and (not math.isfinite(image_min) or image_min < 0.0):
        raise typer.BadParameter(
            "must be a finite, non-negative level", param_hint="--image-min"
        )


def load_gsplats_for_comparison(
    path: Path,
) -> tuple["GSplatData", int, int, Any]:
    """Materialize the tree selection that the renderer shows by default.

    Root stats come along on both paths so the caller can resolve the
    normalization basis (#1173) and the scoring grid. They remain unscrubbed
    because this temporary flat dataset is never persisted. The root
    ``dimension_metadata`` comes back too: it says whether the store is stacked.
    """
    from luxar.gsplats.gsplat_data import GSplatData
    from luxar.gsplats.io.load_gsplats import load_gsplat_node
    from luxar.gsplats.tree import iter_default_leaves, total_splats

    node, stats = load_gsplat_node(path, include_stats=True)
    n_leaves = sum(1 for _ in iter_default_leaves(node))
    data = GSplatData.from_default_selection(node, stats=stats)
    return data, n_leaves, total_splats(node), node.meta.get("dimension_metadata")


def resolve_voxel_size(
    stats: Any, ndim: int, override: Optional[str]
) -> tuple[Optional[tuple[float, ...]], Optional[str]]:
    """``(voxel_size, source)``: the ``--voxel-size`` override, else the stamp.

    An override of all ones disables the mapping explicitly (for a store whose
    stamp is known to be stale, e.g. after a geometric ``transform``).
    """
    from luxar.gsplats.scoring_grid import parse_voxel_size, stored_voxel_size

    if override is not None:
        try:
            parsed = parse_voxel_size(override, ndim)
        except ValueError as exc:
            raise typer.BadParameter(str(exc), param_hint="--voxel-size") from exc
        if all(v == 1.0 for v in parsed):
            return None, None
        return parsed, "--voxel-size"
    stored = stored_voxel_size(stats, ndim)
    return (stored, "dataset") if stored is not None else (None, None)


def _resolve_grid(data: "GSplatData", opts: CompareOptions) -> _ScoringGrid:
    from luxar.gsplats.scoring_grid import fit_downscale_factors

    voxel_size, source = resolve_voxel_size(data.stats, data.ndim, opts.voxel_size)
    grid = _ScoringGrid(voxel_size=voxel_size, voxel_size_source=source)
    if voxel_size is not None:
        aprint(
            f"Grid: physical splats mapped to voxels by voxel_size="
            f"{list(voxel_size)} (from {source})"
        )
    if opts.at_fit_resolution:
        grid.downscale_factors = fit_downscale_factors(data.stats, data.ndim)
        if grid.downscale_factors is None:
            aprint(
                "--at-fit-resolution: the store records no downscaled fitting grid "
                "(fitted_shape == source_shape, or not recorded); scoring at the "
                "source resolution"
            )
        else:
            aprint(
                f"Grid: scoring at the fit's own resolution "
                f"(downscale factors {list(grid.downscale_factors)})"
            )
    return grid


def _decoded_bytes(ref: "np.ndarray", info: dict[str, Any]) -> int:
    """``ref``'s decoded size in the element type its file STORED it as."""
    import numpy as np

    source_dtype = info.get("source_dtype")
    itemsize = np.dtype(source_dtype).itemsize if source_dtype else ref.dtype.itemsize
    return int(ref.size) * int(itemsize)


def _fitted_region_crop(
    ref_shape: tuple[int, ...], stats: Any, dimension_metadata: Any
) -> Optional[tuple[tuple[int, int], ...]]:
    """The ``--region`` a crop fit was fitted on, when ``ref_shape`` contains it.

    ``None`` unless the store records a ``fitting/source_region`` (and is not a
    stacked store, whose grids do not describe its centers), the reference is
    NOT already the crop, and the region lies inside the reference.
    """
    from luxar.gsplats.scoring_grid import is_stacked_store, source_region_bounds

    if is_stacked_store(stats, len(ref_shape), dimension_metadata):
        return None
    bounds = source_region_bounds(stats, len(ref_shape))
    if bounds is None:
        return None
    crop_shape = tuple(stop - start for start, stop in bounds)
    if crop_shape == tuple(ref_shape):
        return None
    if any(stop > n for (_, stop), n in zip(bounds, ref_shape, strict=True)):
        return None
    return bounds


def _format_region(bounds: tuple[tuple[int, int], ...]) -> str:
    return ",".join(f"{a}:{b}" for a, b in bounds)


def _load_reference(
    opts: CompareOptions, grid: _ScoringGrid, stats: Any, ctx: _Context
) -> tuple["np.ndarray", int]:
    """The reference volume, decimated like the fit under ``--at-fit-resolution``.

    A ``--region`` crop fit scored against its FULL source volume is cropped to
    the region it fitted first (unless ``--whole-volume``): outside the region
    the fit saw nothing, so scoring there counts every structure it was never
    asked to model as error, and the default foreground mask (an Otsu threshold
    of the whole reference) would differ from the one the fit stamped.

    Also returns its decoded size in the stored element type (before any crop
    or decimation), the source side of the compression ratio.

    The domain loader, not the CLI wrapper: its ImportError (missing tifffile)
    must reach ``exit_with_error`` so ``-q`` still reports it on stderr.
    """
    from luxar.io.volume import VolumeSelectionError, load_volume

    info: dict[str, Any] = {}
    try:
        ref = load_volume(
            opts.reference_path,
            channel=opts.channel,
            timepoint=opts.timepoint,
            array_key=opts.array_key,
            axes=opts.axes,
            info=info,
        )
    except VolumeSelectionError as exc:  # a usage error, as on fit and cal
        raise typer.BadParameter(str(exc)) from exc
    decoded_bytes = _decoded_bytes(ref, info)
    ctx.loaded_shape = tuple(ref.shape)
    region = _fitted_region_crop(ctx.loaded_shape, stats, ctx.dimension_metadata)
    if region is not None and not opts.whole_volume:
        ref = ref[tuple(slice(a, b) for a, b in region)]
        ctx.scored_region = region
        aprint(
            f"Region: scoring the fitted region {_format_region(region)} of the "
            f"{ctx.loaded_shape} reference (--whole-volume scores all of it)"
        )
    elif region is not None:
        ctx.fitted_region = region
    if grid.downscale_factors is not None and len(grid.downscale_factors) == ref.ndim:
        from luxar.gsplats.fitting.downscale import downscale_volume

        ref = downscale_volume(ref, grid.downscale_factors)
        aprint(f"Reference decimated like the fit: {ref.shape}")
    return ref, decoded_bytes


def _check_source_grid(
    stats: Any,
    ref_shape: tuple[int, ...],
    grid: _ScoringGrid,
    dimension_metadata: Any = None,
    cropped: bool = False,
) -> Optional[tuple[float, ...]]:
    """Check the reference against the fitted grid; return a crop's grid origin.

    Warns when the reference is not the grid the store says it was fitted on. A
    ``--region`` crop fit (``fitting/source_region``) keeps full-volume centers
    but records the CROP as its source grid: a reference of the crop's shape is
    scored with the crop start subtracted, a full-volume reference without it.
    A STACKED store's ``source_shape`` (time first, or one component) does not
    describe its centers' grid, so it is not checked at all.
    """
    from luxar.gsplats.scoring_grid import is_stacked_store, source_region_start

    if is_stacked_store(stats, len(ref_shape), dimension_metadata):
        return None
    start = source_region_start(stats, len(ref_shape))
    expected = stats.get("fitted_shape" if grid.downscale_factors else "source_shape")
    if not isinstance(expected, (list, tuple)) or len(expected) != len(ref_shape):
        return None
    if tuple(int(n) for n in expected) == tuple(ref_shape):
        origin = grid.origin(start)
        if start is not None and not cropped:
            aprint(f"Grid: crop fit, reference is the crop (start {list(start)})")
        return origin
    if start is None:
        aprint(
            f"WARNING: reference shape {tuple(ref_shape)} differs from the grid "
            f"the store was fitted on {tuple(int(n) for n in expected)}; scores "
            "assume the reference is the volume that was fitted"
        )
    else:
        aprint(
            f"Grid: crop fit (start {list(start)}) scored against a reference "
            "that is not the crop; using full-volume coordinates. Everything "
            "outside the fitted region counts as error"
        )
    return None


def reference_on_dataset_basis(
    reference: "np.ndarray", stats: Any, image_min: Optional[float]
) -> tuple["np.ndarray", Optional[float]]:
    """The reference on the fit's basis (shifted IN PLACE), and the level used."""
    from luxar.gsplats.fit_basis import (
        MISSING_BASIS_HINT,
        fit_image_min,
        reference_on_fit_basis,
    )

    resolved_min = float(image_min) if image_min is not None else fit_image_min(stats)
    if resolved_min is None:
        aprint(f"WARNING: {MISSING_BASIS_HINT}")
    elif resolved_min == 0.0:
        aprint("Basis: fit removed no background (image_min=0)")
    else:
        source = "--image-min" if image_min is not None else "dataset"
        reference = reference_on_fit_basis(reference, resolved_min, copy=False)
        aprint(
            f"Basis: reference shifted by image_min={resolved_min:.6g} "
            f"(from {source}); scores are background-relative, matching the render"
        )
    return reference, resolved_min


def _load_mask(
    opts: CompareOptions, grid: _ScoringGrid, shape: tuple[int, ...], ctx: _Context
) -> "np.ndarray":
    """The ``--mask`` volume (foreground = voxels > 0), on the scored grid.

    A mask of the full reference's shape is cropped with it to the fitted region.
    """
    from luxar.io.volume import load_volume

    assert opts.mask is not None
    mask = load_volume(opts.mask)
    if ctx.scored_region is not None and tuple(mask.shape) == ctx.loaded_shape:
        mask = mask[tuple(slice(a, b) for a, b in ctx.scored_region)]
    if grid.downscale_factors is not None and len(grid.downscale_factors) == mask.ndim:
        mask = mask[tuple(slice(None, None, f) for f in grid.downscale_factors)]
    if tuple(mask.shape) != tuple(shape):
        raise typer.BadParameter(
            f"mask shape {tuple(mask.shape)} does not match the scored reference "
            f"shape {tuple(shape)}",
            param_hint="--mask",
        )
    return mask


def _foreground_field(
    opts: CompareOptions, ref: "np.ndarray", grid: _ScoringGrid, ctx: _Context
) -> tuple[Optional["np.ndarray"], Optional[float]]:
    """``(field, threshold)``: foreground is ``field > threshold``.

    ``(None, None)`` selects the shared default (smoothed robust Otsu, see
    :func:`luxar.gsplats.metrics.default_foreground`) — the definition the fit
    summary stamps, computed on the scoring device. ``(None, thr)`` thresholds
    the reference itself.
    """
    if opts.mask is not None:
        ctx.foreground_method = "mask"
        aprint(f"Foreground: --mask {opts.mask.name} (voxels > 0)")
        return _load_mask(opts, grid, tuple(ref.shape), ctx), 0.0
    if opts.fg_threshold is not None:
        ctx.foreground_method = "threshold"
        ctx.foreground_threshold_raw = float(opts.fg_threshold)
        # The user states the level in the reference's own units; the field is
        # on the fit's basis, so the same shift applies.
        threshold = float(opts.fg_threshold) - (ctx.image_min or 0.0)
        aprint(f"Foreground: reference > {opts.fg_threshold:.6g} (--fg-threshold)")
        return None, threshold
    aprint("Foreground: smoothed Otsu (tent blur, histogram capped at p99.99)")
    return None, None


def _render_shape(
    opts: CompareOptions, ref_shape: tuple[int, ...], ndim: int
) -> tuple[int, ...]:
    """The comparison grid; mismatches are errors that reach stderr under ``-q``."""
    from luxar.cli.gsplat_config import parse_shape

    render_shape = parse_shape(opts.shape) if opts.shape is not None else ref_shape
    if opts.shape is not None and tuple(render_shape) != tuple(ref_shape):
        raise ValueError(
            f"--shape {tuple(render_shape)} does not match reference shape "
            f"{tuple(ref_shape)}; omit --shape to use the reference shape"
        )
    if len(render_shape) != ndim:
        raise ValueError(
            f"Dimension mismatch: gsplats are {ndim}D but the reference is "
            f"{len(render_shape)}D (check --axes/--channel/--timepoint)"
        )
    return tuple(render_shape)


#: Below this share of the grid extent spanned on EVERY spatial axis, the
#: centers are probably crammed into a corner (sub-unit spacing, no stamp).
_EXTENT_WARN_FRACTION = 0.5


def warn_if_outside_grid(
    centers: "np.ndarray",
    shape: tuple[int, ...],
    *,
    extent_shape: Optional[tuple[float, ...]] = None,
) -> None:
    """Warn when the splat centers do not sit in the voxel grid they render on.

    Two symptoms of a frame mismatch: most centers fall OUTSIDE the grid
    (physical units with spacing > 1, a stale stamp), or all of them fall inside
    a corner, spanning well under half the grid on every spatial axis (physical
    units with sub-unit spacing). ``extent_shape`` is the extent the splats are
    EXPECTED to span when it is smaller than the grid — a ``--region`` crop fit
    scored on its full source volume fills only its region, which is no corner.
    """
    from luxar.gsplats.scoring_grid import (
        fraction_inside_grid,
        spatial_extent_fraction,
    )

    advice = (
        "If the store is in physical units without a voxel_size stamp, pass "
        "--voxel-size; if its stamp is stale (the splats were moved after the "
        "fit), pass --voxel-size 1 to use the centers as they are. For compare, "
        "also check that --axes/--channel picked the fitted volume."
    )
    inside = fraction_inside_grid(centers, shape)
    if inside < _INSIDE_WARN_FRACTION:
        aprint(
            f"WARNING: only {inside:.0%} of splat centers fall inside the voxel "
            f"grid {shape}. {advice}"
        )
        return
    extent = spatial_extent_fraction(
        centers, shape if extent_shape is None else extent_shape
    )
    if extent < _EXTENT_WARN_FRACTION:
        aprint(
            f"WARNING: the splat centers span at most {extent:.0%} of the voxel "
            f"grid {shape} on any spatial axis (crammed into a corner?). {advice}"
        )


def _resolve_scored_foreground(
    ref_t: Any, field_t: Any, threshold: Optional[float]
) -> tuple[Any, float]:
    """``(field, threshold)`` with foreground = ``field > threshold``, resolved once.

    The caller's field (``--mask``), the reference itself (``--fg-threshold``),
    else the shared default (:func:`luxar.gsplats.metrics.default_foreground`).
    Resolved here rather than inside the metrics so the noise reference reuses
    the very mask that was scored.
    """
    from luxar.gsplats.metrics import default_foreground, otsu_threshold

    if field_t is None and threshold is None:
        return default_foreground(ref_t)
    if field_t is None:
        return ref_t, float(threshold)  # type: ignore[arg-type]
    if threshold is None:
        threshold = otsu_threshold(field_t)
    return field_t, float(threshold)


def noise_reference(
    ref: "np.ndarray", mask: "np.ndarray", metrics: dict[str, Any]
) -> dict[str, Any]:
    """The foreground PSNR a NOISE-FREE model would score against ``ref``.

    A perfect, noise-free reconstruction still differs from a noisy reference
    by the noise, so its foreground MSE is about ``sigma**2`` and its PSNR
    ``20 log10(peak / sigma)`` — the ceiling against which a foreground PSNR is
    judged. ``sigma`` is measured in the foreground
    (:func:`~luxar.gsplats.calibration.noise_floor.estimate_masked_noise`: shot
    noise is larger there than in the background); ``peak`` is the robust peak
    the headline uses. Returns the ``noise_*`` keys of the JSON record (empty
    when there is no foreground or no measurable sigma).

    An estimate, good to about a dB on oversampled microscopy (A549, CTC t000:
    20.9 dB, against 21.0 dB for raw-vs-lightly-blurred data). It assumes noise
    independent between neighbouring voxels, so correlated noise (a deconvolved
    or upsampled reference) reads low. Structure near the voxel scale reads as
    noise and inflates sigma. The estimator's scale check flags structure at
    roughly the voxel scale and coarser (the ceiling is then reported as
    unreliable, with no verdict), but it is BLIND to structure finer than ~0.7
    voxel in-plane (sparse undersampled puncta, hot pixels): its flat spectrum
    keeps the scale ratio near 1. A score above the ceiling is therefore never
    called definitively (see :func:`noise_verdict`).
    """
    from luxar.gsplats.calibration.noise_floor import estimate_masked_noise

    fg_psnr = float(metrics.get("foreground_psnr_robust_db", float("nan")))
    peak = float(metrics.get("robust_data_range", float("nan")))
    if not mask.any() or not math.isfinite(fg_psnr) or not peak > 0.0:
        return {}
    noise = estimate_masked_noise(ref, mask)
    sigma = noise.sigma
    if not math.isfinite(sigma):
        return {}
    out: dict[str, Any] = {
        "noise_sigma": sigma,
        "noise_sigma_method": "foreground Haar HH MAD (in-plane, intensity-binned)",
        "noise_scale_ratio": noise.scale_ratio,
        "noise_ceiling_reliable": noise.reliable,
    }
    if sigma <= 0.0:
        out["noise_verdict"] = "noise-free reference"
        return out
    ceiling = 20.0 * math.log10(peak / sigma)
    gap = fg_psnr - ceiling
    out["foreground_psnr_noise_ceiling_db"] = ceiling
    out["foreground_psnr_vs_noise_ceiling_db"] = gap
    out["model_error_over_noise_sigma"] = model_error_over_noise(gap)
    out["noise_verdict"] = (
        noise_verdict(gap, ceiling) if noise.reliable else _UNRELIABLE_VERDICT
    )
    return out


_UNRELIABLE_VERDICT = (
    "undetermined: the finest band is not white (voxel-scale structure or "
    "spatially correlated noise, e.g. a deconvolved, upsampled or "
    "--at-fit-resolution downscaled reference), so the ceiling is unreliable"
)

#: Above the ceiling by more than this the score is ambiguous: the model may be
#: reproducing noise, or the reference may have sub-voxel structure the scale
#: check cannot see and the estimate counts as noise. Never a definite call.
_ABOVE_CEILING_DB = 1.0
#: Above the ceiling by more than this, the ceiling itself is the likelier error
#: (sub-voxel structure put sigma 9-20x high in probes), so no verdict at all.
_UNDETERMINED_ABOVE_DB = 3.0
#: Within this many dB below the ceiling, a fit is "at the noise floor": the
#: model error is at most ~half the noise sigma (MSE = sigma^2 * 10^0.1).
_AT_FLOOR_DB = 1.0
#: Down to this many dB below, it is "near" it: the model error is at most the
#: noise sigma.
_NEAR_FLOOR_DB = 3.0
#: Above this ceiling the reference is clean enough that noise does not limit
#: the score (a 1000:1 peak-to-sigma ratio).
_NOISE_FREE_CEILING_DB = 60.0


def model_error_over_noise(gap_db: float) -> Optional[float]:
    """The model's own RMS error in units of the noise sigma, from the gap.

    Against a noisy reference the residual is noise plus model error, so
    ``MSE = sigma^2 + e^2`` and ``e / sigma = sqrt(10^(-gap/10) - 1)``: 0.5 at
    -1 dB, 1.0 at -3 dB (the error as large as the noise). ``None`` at or above
    the ceiling, where the residual is smaller than the noise.
    """
    if gap_db >= 0.0:
        return None
    return math.sqrt(10.0 ** (-gap_db / 10.0) - 1.0)


def noise_verdict(gap_db: float, ceiling_db: float) -> str:
    """One line on where a foreground PSNR sits against the noise ceiling."""
    if ceiling_db >= _NOISE_FREE_CEILING_DB:
        return (
            "the reference is nearly noise-free here: noise does not limit this score"
        )
    if gap_db > _UNDETERMINED_ABOVE_DB:
        return (
            f"undetermined: {gap_db:.1f} dB above the estimated ceiling, more than "
            "the estimate's error allows, so the reference likely has sub-voxel "
            "or voxel-scale structure the estimate counts as noise"
        )
    if gap_db > _ABOVE_CEILING_DB:
        return (
            f"above the estimated ceiling ({gap_db:.1f} dB): ambiguous — the "
            "reference may have sub-voxel or voxel-scale structure the estimate "
            "counts as noise, or the model may be reproducing noise"
        )
    ratio = model_error_over_noise(gap_db)
    if gap_db >= -_AT_FLOOR_DB:
        size = "below the noise" if ratio is None else f"~{ratio:.2f} x the noise sigma"
        return (
            f"at the noise floor: the model error is {size}; a better model "
            f"can gain at most ~{max(0.0, -gap_db):.1f} dB against this noisy "
            "reference"
        )
    if gap_db >= -_NEAR_FLOOR_DB:
        return (
            f"near the noise floor: {-gap_db:.1f} dB below the ceiling, the model "
            f"error is ~{ratio:.2f} x the noise sigma"
        )
    return (
        f"{-gap_db:.1f} dB below the noise ceiling: the model error is "
        f"~{ratio:.1f} x the noise sigma, so there is signal left to fit (more "
        "splats or iterations, or check the fit)"
    )


def _score(
    data: "GSplatData",
    ref: "np.ndarray",
    fg: tuple[Optional["np.ndarray"], Optional[float]],
    *,
    render_shape: tuple[int, ...],
    opts: CompareOptions,
    truncate: float,
) -> dict[str, Any]:
    import torch

    from luxar.gsplats.metrics import compute_quality_metrics
    from luxar.gsplats.rendering.volume_rendering import render_to_volume_tensor
    from luxar.gsplats.utils.device import resolve_torch_device

    dev = str(resolve_torch_device(opts.device))
    with torch.no_grad():
        with asection(f"Rendering on {dev}"):
            rendered_t = render_to_volume_tensor(
                data, shape=render_shape, device=dev, truncate=truncate
            )
            aprint(
                f"Rendered: {tuple(rendered_t.shape)}, range "
                f"[{rendered_t.min().item():.4f}, {rendered_t.max().item():.4f}]"
            )
        ref_t = torch.from_numpy(ref).to(rendered_t.device)
        field_np, user_threshold = fg
        field_t = None if field_np is None else torch.from_numpy(field_np).to(ref_t)
        field_t, threshold = _resolve_scored_foreground(ref_t, field_t, user_threshold)
        with asection("Computing metrics"):
            metrics = compute_quality_metrics(
                rendered_t,
                ref_t,
                foreground_field=field_t,
                foreground_threshold=threshold,
            )
        del rendered_t
        mask = (field_t > threshold).cpu().numpy()
        del field_t
    metrics.update(noise_reference(ref, mask, metrics))
    return metrics


def _undo_intensity_scale(data: "GSplatData", stats: Any) -> tuple["GSplatData", float]:
    """``data`` back in source units, and the recorded scale that was undone.

    The reference, its peak, the floor and the foreground threshold all stay in
    the reference's own units; only the render moves, so every metric keeps
    the meaning it has on an unedited store.
    """
    from luxar.gsplats.fit_basis import stored_intensity_scale
    from luxar.gsplats.scoring_grid import amplitudes_in_source_units

    scale = stored_intensity_scale(stats)
    if scale != 1.0:
        aprint(
            f"Basis: amplitudes divided by intensity_scale={scale:.6g} (a "
            "transform --scale-intensity / --normalize-intensity edit); scored in "
            "the source's units"
        )
    elif stats and "image_min" in stats and "intensity_range" not in stats:
        # A non-invertible amplitude edit (offset, clamp, per-splat reweight) or a
        # merge of fits on different scales drops the amplitude basis but keeps
        # the raw level, so the shift above still runs on amplitudes it no longer
        # describes.
        aprint(f"WARNING: {_NO_AMPLITUDE_BASIS_HINT}")
    return amplitudes_in_source_units(data, scale), scale


def _fit_region(ctx: _Context) -> Optional[tuple[tuple[int, int], ...]]:
    """The crop fit's region inside the loaded reference, scored or not."""
    return ctx.scored_region or ctx.fitted_region


def _source_bytes(ctx: _Context) -> int:
    """Decoded bytes of what the store encodes: the fitted region, else all."""
    region = _fit_region(ctx)
    if region is None or not ctx.loaded_shape:
        return ctx.ref_decoded_bytes
    share = math.prod(b - a for a, b in region) / max(1, math.prod(ctx.loaded_shape))
    return int(round(ctx.ref_decoded_bytes * share))


def _region_extent(ctx: _Context) -> Optional[tuple[float, ...]]:
    """The fitted region's extent on the scored grid, for ``--whole-volume``."""
    if ctx.fitted_region is None:
        return None
    fs = ctx.grid.downscale_factors or (1,) * len(ctx.fitted_region)
    return tuple((b - a) / f for (a, b), f in zip(ctx.fitted_region, fs, strict=True))


def _compare(opts: CompareOptions) -> tuple[dict[str, Any], _Context]:
    """Load, map, render and score; returns the metrics and what was compared."""
    from luxar.gsplats.scoring_grid import map_to_grid

    ctx = _Context()
    with asection("Loading gsplat dataset"):
        data, n_leaves, n_stored, ctx.dimension_metadata = load_gsplats_for_comparison(
            opts.gsplats_path
        )
        ctx.n_splats, ctx.ndim = data.n_splats, data.ndim
        aprint(f"Loaded {ctx.n_splats:,} splats ({ctx.ndim}D)")
        aprint(f"Materialized {n_leaves:,} default-rendered leaf/leaves")
        if n_stored > ctx.n_splats:
            aprint(
                f"Skipped {n_stored - ctx.n_splats:,} splats in coarse LOD levels; "
                "compression ratio covers the whole store"
            )
        ctx.grid = _resolve_grid(data, opts)
    truncate = opts.truncate if opts.truncate is not None else data.truncation_radius

    with asection("Loading reference volume"):
        ref, ctx.ref_decoded_bytes = _load_reference(
            opts, ctx.grid, data.stats or {}, ctx
        )
        ctx.ref_shape = tuple(ref.shape)
        origin = _check_source_grid(
            data.stats or {},
            ctx.ref_shape,
            ctx.grid,
            ctx.dimension_metadata,
            cropped=ctx.scored_region is not None,
        )
        ref, ctx.image_min = reference_on_dataset_basis(ref, data.stats, opts.image_min)
        fg = _foreground_field(opts, ref, ctx.grid, ctx)

    ctx.render_shape = _render_shape(opts, ctx.ref_shape, ctx.ndim)
    aprint(f"Comparison shape: {ctx.render_shape}")
    scored = map_to_grid(data, scale=ctx.grid.scale, origin=origin)
    scored, ctx.intensity_scale = _undo_intensity_scale(scored, data.stats)
    warn_if_outside_grid(
        scored.centers, ctx.render_shape, extent_shape=_region_extent(ctx)
    )
    metrics = _score(
        scored, ref, fg, render_shape=ctx.render_shape, opts=opts, truncate=truncate
    )
    # Compression ratio: the one definition `gsplat info` and `fit` quote —
    # decoded reference bytes over the store's bytes on disk. A crop fit only
    # encodes its region, so (as the fit's own stamp does) it is measured
    # against the region's share of the reference, under --whole-volume too.
    from luxar.gsplats.fitting.visualization import compression_ratio

    ratio = compression_ratio(_source_bytes(ctx), store_size(opts.gsplats_path))
    if ratio is not None:
        metrics["compression_ratio"] = ratio
    return metrics, ctx


_FOREGROUND_LABELS = {
    "otsu_smoothed": "smoothed Otsu > {thr:.4g}",
    "mask": "--mask voxels > 0",
    "threshold": "reference > {raw:.6g}",
}


def _fmt_db(value: float) -> str:
    return f"{value:.2f} dB" if math.isfinite(value) else str(value)


def print_quality_comparison(
    metrics: dict[str, Any], ctx: _Context, opts: CompareOptions
) -> None:
    """Render the ``gsplat compare`` results table."""
    from luxar.gsplats.metrics import ROBUST_PEAK_QUANTILE

    quantile = f"p{ROBUST_PEAK_QUANTILE * 100:g}"
    aprint("\n" + "=" * 60)
    aprint("QUALITY COMPARISON")
    aprint("=" * 60)
    aprint(f"\nReference:  {opts.reference_path.name} {ctx.ref_shape}")
    if ctx.scored_region is not None:
        aprint(
            f"Region:     fitted region {_format_region(ctx.scored_region)} of "
            f"{ctx.loaded_shape} (--whole-volume to score all of it)"
        )
    aprint(f"GSplats:    {opts.gsplats_path.name} ({ctx.n_splats:,} splats)")
    if ctx.grid.scale is not None:
        aprint(f"Grid:       splat centers divided by {list(ctx.grid.scale)}")
    aprint("")
    aprint(
        f"  PSNR:              {_fmt_db(metrics['psnr_robust_db'])}  "
        f"(robust peak = reference {quantile} - floor = "
        f"{metrics['robust_data_range']:.6g})"
    )
    aprint(
        f"  PSNR (full range): {_fmt_db(metrics['psnr_db'])}  "
        f"(peak = max - floor = {metrics['data_range']:.6g})"
    )
    if metrics["robust_data_range"] == metrics["data_range"]:
        aprint(
            f"  (the two peaks coincide: {quantile} is the maximum here — fewer "
            "than 10,000 voxels, or a sparse volume)"
        )
    label = _FOREGROUND_LABELS[ctx.foreground_method].format(
        thr=metrics["foreground_threshold"], raw=ctx.foreground_threshold_raw or 0.0
    )
    # The share is not decoration: on a 99%-empty volume the global PSNR above is
    # largely a score for reproducing the emptiness, and this says how little of
    # the volume the honest number was taken over.
    aprint(
        f"  PSNR foreground:   {_fmt_db(metrics['foreground_psnr_robust_db'])}  "
        f"(full range {_fmt_db(metrics['foreground_psnr_db'])}; over "
        f"{metrics['foreground_fraction'] * 100:.2f}% of voxels, {label})"
    )
    _print_noise_reference(metrics)
    aprint(f"  MSE:               {metrics['mse']:.6g}")
    aprint(f"  SSIM:              {metrics['ssim']:.4f}")
    aprint(f"  SSIM foreground:   {metrics['foreground_ssim']:.4f}")
    aprint(f"  Rel L2:            {metrics['rel_l2']:.6g}")
    aprint(f"  Max Abs Error:     {metrics['max_abs_error']:.6g}")
    if "compression_ratio" in metrics:
        from luxar.gsplats.fitting.visualization import format_compression_ratio

        of = " of the fitted region" if _fit_region(ctx) is not None else ""
        aprint(
            "  Compression:       "
            f"{format_compression_ratio(metrics['compression_ratio'])} vs raw "
            f"voxels{of}"
        )
    aprint("=" * 60)


def _print_noise_reference(metrics: dict[str, Any]) -> None:
    """The noise-ceiling lines under the foreground PSNR (when measurable)."""
    if "noise_verdict" not in metrics:
        return
    ceiling = metrics.get("foreground_psnr_noise_ceiling_db")
    if ceiling is not None:
        flag = "" if metrics.get("noise_ceiling_reliable") else "UNRELIABLE, "
        aprint(
            f"  Noise ceiling:     {_fmt_db(ceiling)}  ({flag}foreground PSNR of a "
            f"noise-free model vs this reference; noise sigma = "
            f"{metrics['noise_sigma']:.4g} in the foreground)"
        )
    aprint(f"  Verdict:           {metrics['noise_verdict']}")


def _json_record(
    metrics: dict[str, Any], ctx: _Context, opts: CompareOptions
) -> dict[str, Any]:
    from luxar.gsplats.metrics import ROBUST_PEAK_QUANTILE
    from luxar.typing_utils.json_safe import json_safe_value

    record: dict[str, Any] = {
        "gsplats": str(opts.gsplats_path),
        "reference": str(opts.reference_path),
        "n_splats": ctx.n_splats,
        "ndim": ctx.ndim,
        "shape": list(ctx.render_shape),
        # Which PSNR is the headline, and what its peak was: `psnr_db` keeps the
        # classical full-range meaning every stored fit uses.
        "psnr_headline": "psnr_robust_db",
        "robust_peak_quantile": ROBUST_PEAK_QUANTILE,
        "image_min": ctx.image_min,
        "voxel_size": list(ctx.grid.voxel_size) if ctx.grid.voxel_size else None,
        "scored_grid": "fit" if ctx.grid.downscale_factors else "source",
        "downscale_factors": (
            list(ctx.grid.downscale_factors) if ctx.grid.downscale_factors else None
        ),
        "foreground_method": ctx.foreground_method,
        "foreground_threshold_raw": ctx.foreground_threshold_raw,
        "scored_region": (
            [list(span) for span in ctx.scored_region] if ctx.scored_region else None
        ),
        "fitted_region": (
            [list(span) for span in region] if (region := _fit_region(ctx)) else None
        ),
        "whole_volume": opts.whole_volume,
    }
    if ctx.intensity_scale != 1.0:
        record["intensity_scale"] = ctx.intensity_scale
    for key, value in metrics.items():
        ok, converted = json_safe_value(value)
        # Non-finite floats (a perfect score is +inf) become null: valid JSON.
        record[key] = converted if ok else None
    return record


def run_compare(opts: CompareOptions) -> None:
    """Run ``gsplat compare`` end to end (``-q`` silences everything but errors)."""
    validate_image_min_override(opts.image_min)
    with silenced_when(opts.quiet):
        try:
            with asection("Quality Comparison"):
                metrics, ctx = _compare(opts)
            print_quality_comparison(metrics, ctx, opts)
            if opts.output_json is not None:
                with open(opts.output_json, "w") as f:
                    json.dump(_json_record(metrics, ctx, opts), f, indent=2)
                aprint(f"\nMetrics written to: {opts.output_json}")
        except (typer.Exit, typer.BadParameter):
            raise
        except Exception as e:  # noqa: BLE001 - CLI boundary: one line, exit 1
            exit_with_error(f"Error: {e}", e)
