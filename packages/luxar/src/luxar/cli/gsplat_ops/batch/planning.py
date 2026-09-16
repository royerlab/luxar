"""Scheduler-agnostic batch planning shared by ``batch-fit submit`` and ``run``.

``submit`` (Slurm) and ``run`` (local) share everything *up to and
including* building the :class:`BatchManifest` + its job list: dataset discovery
+ T/C slicing, uniform-tile / content-box decomposition (and the shared
``plan.json``), fit-arg assembly, and per-part merge-recipe validation.  Only the
*execution* differs (sbatch submit vs local subprocess pool).

This module owns that shared half so the two commands don't duplicate ~300 lines.
It lives in the CLI layer (it drives CLI helpers like ``load_volume`` /
``_resolve_density`` and raises :class:`typer.BadParameter`); the local engine
(:mod:`luxar.gsplats.batch.local_runner`) consumes only the returned manifest, so
there is no upward dependency from the gsplats package into the CLI.
"""

from __future__ import annotations

import dataclasses
import datetime
import math
from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING, Any, List, Optional, Sequence, Tuple

import typer
from arbol import aprint, asection

from luxar.cli.gsplat_ops.fitting.fit_utils import CONTENT_UNSUPPORTED_FIT_FLAGS
from luxar.core.group.partition import prune_serialized_bsp_tree
from luxar.gsplats.batch.manifest import BatchJob, BatchManifest, output_filename

if TYPE_CHECKING:  # pragma: no cover - typing only
    from luxar.io.ome_zarr import OMEZarrInfo

# ---------------------------------------------------------------------------
# Grouped option contracts (the shared surface both commands populate)
# ---------------------------------------------------------------------------


@dataclass
class FitConfig:
    """Per-tile fit parameters shared by both commands."""

    preset: str = "standard"
    seeds: Optional[str] = None
    iters: Optional[int] = None
    config: Optional[Path] = None
    floor: Optional[str] = "auto"
    physical: bool = False
    progressive: bool = False
    splats_per_pass: Optional[int] = None
    psnr_patience: Optional[float] = None
    max_passes: Optional[int] = None
    cull_retention: Optional[float] = None


@dataclass
class DenoiseConfig:
    """NLM denoise parameters (off by default)."""

    denoise: bool = False
    denoise_h: Optional[float] = None
    denoise_2d: bool = False
    patch_size: int = 3
    search_distance: int = 5
    backend: str = "auto"
    calibration_samples: int = 5
    preprocess: Optional[bool] = None


@dataclass
class ContentKnobs:
    """Content-adaptive box-plan knobs (used only when ``tiling == 'content'``)."""

    cal: Optional[Path] = None
    k_star_ref: Optional[int] = None
    n_features_ref: Optional[int] = None
    saturation_exponent: float = 0.44
    saturation_cap: Optional[int] = None
    feature_threshold: Optional[float] = None
    feature_metric: Optional[str] = None
    cell: int = 16
    target_features: Optional[int] = None
    min_leaf: int = 256
    max_leaf: int = 512
    plan_timepoint: Optional[int] = None
    plan_samples: int = 16


@dataclass
class MergeConfig:
    """Per-part LOD merge recipe + its knobs (validated by :func:`resolve_merge_recipe_args`)."""

    recipe: Optional[str] = None
    channel_colors: Optional[str] = None
    n_lods: Optional[int] = None
    additive_method: Optional[str] = None
    breakpoints: Optional[str] = None
    # Streaming sizing (--merge-target-ms trio): resolved at plan time into a
    # concrete `breakpoints="stream:<c>"` string, so the manifest schema is
    # unchanged and the merge job just re-parses the stored breakpoints.
    target_ms: Optional[float] = None
    bandwidth_mbps: Optional[float] = None
    bytes_per_splat: Optional[float] = None
    compression_factor: Optional[int] = None
    levels: Optional[int] = None
    substitutive_method: Optional[str] = None
    coarsen_dims: Optional[str] = None
    # Post-merge refinement of each per-tile coarse level. "volume" re-opens the
    # source this plan records and crops it per tile at merge time, so it is only
    # meaningful for a `levels` merge.
    refine: Optional[str] = None
    refine_iters: Optional[int] = None


@dataclass
class PlanResult:
    """Output of :func:`plan_batch`: the manifest + decomposition facts."""

    manifest: BatchManifest
    content_plan: Optional[Any]  # FitPlan in content mode, else None
    tile_voxels: int
    needs_tiling: bool


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _select_plan_timepoints(
    t_indices: list[int], plan_timepoint: Optional[int], plan_samples: int
) -> list[int]:
    """Pick which timepoints to scan when building the shared content box plan.

    A pinned ``plan_timepoint`` scans just that timepoint (it must be one of the
    selected ``t_indices``); otherwise return up to ``plan_samples`` evenly-spaced
    timepoints (endpoints included, deduplicated) to max-project — so the plan
    covers any region with signal at ANY timepoint rather than leaving holes where
    content moved over time. Raises :class:`typer.BadParameter` on invalid input.
    """
    if plan_timepoint is not None:
        if plan_timepoint not in t_indices:
            raise typer.BadParameter(
                f"--plan-timepoint {plan_timepoint} is not in the selected "
                f"timepoints {t_indices[0]}..{t_indices[-1]} "
                f"({len(t_indices)} total)."
            )
        return [plan_timepoint]
    if plan_samples < 1:
        raise typer.BadParameter(f"--plan-samples must be >= 1, got {plan_samples}.")
    if len(t_indices) <= plan_samples:
        return list(t_indices)
    import numpy as _np

    idx = _np.unique(
        _np.linspace(0, len(t_indices) - 1, plan_samples).round().astype(int)
    )
    return [t_indices[i] for i in idx]


def effective_floor_spec(fit: FitConfig) -> "str | float | None":
    """The floor spec a task would actually end up applying.

    ``--floor`` unset (``None``) does NOT mean "no floor": it means "let a
    ``floor:`` in ``--config``/the preset apply, else the ``fit_gaussian_splats``
    default (``auto``)". Plan-time resolution has to read that same chain, or an
    unset ``--floor`` (the batch default) would resolve the wrong spec — or
    silently override a config's own floor.
    """
    if fit.floor is not None:
        return fit.floor
    import yaml

    from luxar.cli.gsplat_config import PRESETS, load_fit_config

    try:
        merged = load_fit_config(
            preset=fit.preset if fit.preset in PRESETS else None,
            config_path=fit.config,
        )
    except (FileNotFoundError, ValueError, yaml.YAMLError):
        # A bad --config/preset is reported by the task's own load, as before.
        # ``yaml.YAMLError`` is NOT a ValueError, so a malformed --config would
        # otherwise escape planning as a raw parser traceback.
        return "auto"
    spec = merged.get("floor", "auto")
    if spec is None or isinstance(spec, (str, float, int)):
        return spec
    raise typer.BadParameter(
        f"config `floor:` must be a string or number, got {spec!r}"
    )


def effective_norm_percentile(fit: FitConfig) -> float:
    """The ``norm_percentile`` every task will inherit from preset/config."""
    from luxar.cli.gsplat_config import load_fit_config

    cfg = load_fit_config(preset=fit.preset, config_path=fit.config)
    try:
        value = float(cfg.get("norm_percentile", 0.0))
    except (TypeError, ValueError) as exc:
        raise typer.BadParameter(f"config `norm_percentile` is invalid: {exc}") from exc
    if not 0.0 <= value < 50.0:
        raise typer.BadParameter(
            f"config `norm_percentile` must be in [0, 50), got {value}"
        )
    return value


def effective_norm_range(fit: FitConfig) -> "Optional[Tuple[float, float]]":
    """A deliberate ``norm_range`` from preset/config, if one was supplied."""
    from luxar.cli.gsplat_config import load_fit_config
    from luxar.gsplats.fitting.validation import _validate_norm_range

    raw = load_fit_config(preset=fit.preset, config_path=fit.config).get("norm_range")
    if raw is None:
        return None
    try:
        _validate_norm_range(raw)
        return float(raw[0]), float(raw[1])
    except (IndexError, TypeError, ValueError) as exc:
        raise typer.BadParameter(f"config `norm_range` is invalid: {exc}") from exc


def _floor_axis_pins(
    axes_labels: List[str],
    channel_shape: Tuple[int, ...],
    *,
    timepoint: int,
    channel: int,
) -> "dict[int, int]":
    """``{axis: index}`` fixing one sampled ``(t, c)`` slice, by LABEL.

    Label-driven, not positional: ``load_volume``'s 4D positional heuristic reads
    a ``(T, Z, Y, X)`` store as ``CZYX`` and would index the TIME axis with the
    channel, silently measuring the floor on ``t=0`` whatever timepoint was asked
    for (#1174). The flat ``channel`` task index is decoded back into one index
    per channel-like axis, exactly as the task argv builders do. The vocabulary is
    :func:`luxar.io.ome_zarr.classify_axis_labels`, which goes by axis NAME, while
    NGFF discovery derived ``channel_shape`` by axis *type* first — so the two CAN
    disagree (an NGFF axis typed ``channel`` but named outside the vocabulary, say
    ``stain``, is channel-like to discovery and invisible here). Only the
    channel-side disagreement is caught here, as a COUNT mismatch against the
    discovered ``channel_shape``, and it raises :class:`ValueError`. A time-side
    one raises nothing — an axis typed ``time`` but named outside the vocabulary
    is spatial to the classifier, so no time pin is emitted and the caller's own
    spatial-shape check on the pinned view is what notices. Either outcome sends
    the caller to its eager fallback, which is a hedge, not a guarantee that the
    intended slice is recovered (see :func:`_pinned_slice_volume`).
    """
    from luxar.io.ome_zarr import classify_axis_labels
    from luxar.io.volume import decode_flat_channel_index

    time_axis, channel_axes, _ = classify_axis_labels(axes_labels)
    if len(channel_axes) != len(channel_shape):
        raise ValueError(
            f"axis labels {axes_labels} name {len(channel_axes)} channel-like "
            f"axes but the discovered channel shape is {channel_shape}"
        )
    pins: dict[int, int] = {}
    coords = decode_flat_channel_index(channel, tuple(channel_shape))
    for axis, coord in zip(channel_axes, coords, strict=True):
        pins[axis] = int(coord)
    if time_axis is not None:
        pins[time_axis] = int(timepoint)
    return pins


def _pinned_slice_volume(
    input_path: Path,
    *,
    channel: int,
    timepoint: int,
    array_key: Optional[str],
    axes: Optional[str],
    axes_labels: Optional[List[str]] = None,
    channel_shape: Tuple[int, ...] = (),
    spatial_shape: Optional[Tuple[int, ...]] = None,
) -> Any:
    """A LAZY view of one ``(t, c)`` sub-volume, when possible.

    :func:`~luxar.gsplats.fitting.preprocessing.resolve_volume_floor` reads only a
    bounded deterministic sample of whatever it is handed, so handing it a
    lazily-opened, axis-pinned zarr view keeps plan time metadata-cheap — a
    ``load_volume`` here ends in ``np.asarray(..., float32)`` and would
    materialize a whole multi-GB timepoint (on a login node, and again on every
    resume re-plan). Same construction as the merge's volume re-fit
    (:func:`luxar.gsplats.batch.merge_orchestrator._merge_refit_volume`).

    ``axes_labels`` are the labels DISCOVERED for this store (``ome_info.axes``),
    which exist whether or not the user passed ``--axes``. If they cannot be
    mapped onto the store (a label discovery classified differently — the NGFF
    parser goes by axis *type*, not name — or a pinned view whose remaining shape
    is not the discovered spatial shape), fall back to the eager
    :func:`load_volume`. That fallback still asks for the labels it discovered,
    so it slices the same ``(t, c)``; only if ``load_volume``'s own ``--axes``
    vocabulary rejects one of them does it drop to the positional heuristic,
    which may not honour ``timepoint`` at all. The fallback materializes the whole
    slice, so a caller that samples several of them pays for each in turn — it is
    a correctness hedge for stores whose labels do not map, not the normal path.
    """
    from luxar.io.volume import load_volume, open_volume_lazy, pin_volume_axes

    if axes_labels is not None:
        problem: Optional[str] = None
        pins: dict[int, int] = {}
        try:
            pins = _floor_axis_pins(
                axes_labels,
                tuple(channel_shape),
                timepoint=timepoint,
                channel=channel,
            )
            view = pin_volume_axes(open_volume_lazy(input_path, array_key), pins)
        except (ValueError, KeyError, TypeError) as exc:
            problem = str(exc)
        else:
            if spatial_shape is None or tuple(view.shape) == tuple(spatial_shape):
                return view
            singleton_pins = {
                axis: 0 for axis, size in enumerate(view.shape) if int(size) == 1
            }
            squeezed_view = pin_volume_axes(view, singleton_pins)
            if tuple(squeezed_view.shape) == tuple(spatial_shape):
                return squeezed_view
            problem = (
                f"pinning {pins} left shape {tuple(view.shape)}, not the "
                f"discovered spatial shape {tuple(spatial_shape)}"
            )
        aprint(
            f"Note: reading the (t={timepoint}, c={channel}) slice eagerly — a "
            f"lazy axis-pinned view could not be built ({problem})."
        )
        if axes is None:
            try:
                return load_volume(
                    input_path,
                    channel=channel,
                    timepoint=timepoint,
                    array_key=array_key,
                    axes=",".join(axes_labels),
                ).squeeze()
            except ValueError as exc:
                # load_volume's --axes vocabulary is narrower than discovery's
                # (no `view`/`angle`), so an exotic label lands here.
                aprint(
                    f"Note: the discovered axis labels {axes_labels} are not a "
                    f"usable --axes spec ({exc}); falling back to the positional "
                    f"heuristic, which may ignore the timepoint."
                )
    return load_volume(
        input_path,
        channel=channel,
        timepoint=timepoint,
        array_key=array_key,
        axes=axes,
    )


def _materialize(view: Any) -> Any:
    """Read a (possibly lazy, possibly axis-pinned) view as a float32 array."""
    import numpy as _np

    return _np.asarray(view[:], dtype=_np.float32)


def _bounded_exact_range(view: Any, max_block_voxels: int) -> Tuple[float, float]:
    """Return exact finite endpoints without materializing the whole view."""
    import itertools

    import numpy as np

    shape = tuple(int(size) for size in view.shape)
    if not shape or any(size <= 0 for size in shape):
        raise ValueError("volume is empty")
    block_shape = list(shape)
    while np.prod(block_shape, dtype=np.int64) > max_block_voxels:
        axis = max(range(len(block_shape)), key=block_shape.__getitem__)
        block_shape[axis] = max(1, block_shape[axis] // 2)
    starts = [range(0, size, block) for size, block in zip(shape, block_shape)]
    lo = float("inf")
    hi = float("-inf")
    for origin in itertools.product(*starts):
        selection = tuple(
            slice(start, min(start + block, size))
            for start, block, size in zip(origin, block_shape, shape)
        )
        block = np.asarray(view[selection], dtype=np.float32)
        if not bool(np.isfinite(block).all()):
            raise ValueError("volume contains non-finite values")
        lo = min(lo, float(block.min()))
        hi = max(hi, float(block.max()))
    return lo, hi


# Bounded, deterministic (t, c) sampling for the ONE global floor level (see
# :func:`resolve_batch_floor`). The two axes get INDEPENDENT caps, deliberately:
# a shared budget spent on channels first would leave a store with >= 5
# channel-like coordinates sampling a single timepoint, which is #1174's erase
# bug back again in the time direction. 4 timepoints x 4 channel-like
# coordinates = at most 16 slices, and the per-slice read budget is
# FLOOR_SAMPLE_BUDGET_VOXELS // n_pairs (2M voxels at the cap), so floor samples
# total at most one whole-volume budget. Deferred on-the-fly resolution also
# scans each sampled slice once for exact endpoints and reads bounded probe blocks.
FLOOR_SAMPLE_MAX_SLICES = 16
FLOOR_SAMPLE_MAX_TIMEPOINTS = 4
FLOOR_SAMPLE_MAX_CHANNELS = 4


def _should_defer_floor_resolution(
    mode: str,
    denoise: DenoiseConfig,
    floor_spec: "str | float | None",
    denoise_mode: Optional[str],
) -> bool:
    """Whether a uniform denoise plan must resolve its floor after planning."""
    from luxar.cli.gsplat_ops.fitting.fit_utils import floor_spec_needs_volume
    from luxar.gsplats.fitting.preprocessing import _floor_spec_is_percentile

    if mode != "uniform" or not denoise.denoise:
        return False
    if not floor_spec_needs_volume(floor_spec):
        return False
    # Content planning consumes the level while placing boxes, so deferring it
    # would require reordering the content plan itself.
    return denoise_mode == "preprocess" or _floor_spec_is_percentile(floor_spec)


def _resolve_planned_floor(
    *,
    deferred: bool,
    input_path: Path,
    fit: FitConfig,
    fit_args: dict[str, Any],
    n_timepoints: int,
    n_channels: int,
    array_key: Optional[str],
    axes: Optional[str],
    axes_labels: List[str],
    channel_shape: Tuple[int, ...],
    spatial_shape: Tuple[int, ...],
    sampled_slices: "Optional[List[Tuple[int, int, Any]]]" = None,
) -> tuple[Optional[float], Optional[float]]:
    if deferred:
        return None, None
    return _resolve_and_record_floor(
        input_path,
        fit,
        fit_args,
        n_timepoints=n_timepoints,
        n_channels=n_channels,
        array_key=array_key,
        axes=axes,
        axes_labels=axes_labels,
        channel_shape=channel_shape,
        spatial_shape=spatial_shape,
        sampled_slices=sampled_slices,
    )


def _evenly_spaced(n: int, k: int) -> List[int]:
    """``k`` evenly spaced indices in ``range(n)``, endpoints included, unique.

    Both callers pass ``k = min(n, <cap>)`` with ``n >= 1``, so ``k == 1`` only
    happens for a single-index axis and the endpoints are always among the
    returned indices — which is what makes a MINIMUM over the sampled slices a
    lower bound under a monotone drift (see :func:`_floor_sample_pairs`). There is
    deliberately no "one sample -> take the middle" rule here: for the (t, c)
    axes a middle sample is not representative of the run's dimmest pedestal, it
    just hides which end is.
    """
    if n <= 0 or k <= 0:
        return []
    if k >= n:
        return list(range(n))
    import numpy as _np

    return sorted({int(i) for i in _np.linspace(0, n - 1, k).round().astype(int)})


def _floor_sample_pairs(n_timepoints: int, n_channels: int) -> List[Tuple[int, int]]:
    """The ``(timepoint, channel)`` slices the global floor level is measured on.

    A pure, deterministic function of ``(n_timepoints, n_channels)``: the
    cartesian product of up to :data:`FLOOR_SAMPLE_MAX_TIMEPOINTS` evenly spaced
    timepoints and up to :data:`FLOOR_SAMPLE_MAX_CHANNELS` evenly spaced
    channel-like coordinates (flat index), i.e. at most
    :data:`FLOOR_SAMPLE_MAX_SLICES` slices, spanning the store's FULL extent.
    The two axes are capped INDEPENDENTLY so neither can starve the other:

    * ``n_t > 1`` always yields at least 2 timepoints, INCLUDING the endpoints
      ``t=0`` and ``t=T-1``. That is what makes a MINIMUM over the samples a true
      lower bound under a monotone pedestal drift (sensor warm-up, bleaching):
      the run's dimmest pedestal is then at one of the ends, and both ends are
      sampled. Allocating time FIRST also matters: spending a shared budget on
      channels first made a store with >= 5 channel-like coordinates sample a
      single timepoint, reinstating #1174's erase bug over time.
    * ``n_c > 1`` always yields at least 2 channel-like coordinates (endpoints
      included) — a dim channel is the other axis along which one global level
      can erase data.

    Deliberately independent of ``--timepoints``/``--channels``: the same store
    must resolve the same level whether the user selects ``0:50`` or ``0:100``,
    or a resumed/extended run would subtract a different pedestal than the tiles
    already on disk.

    RESIDUAL RISK — unavoidable with bounded sampling, and NOT eliminated by this
    rule: a slice that is *not* sampled and dimmer than every sampled one (a
    blank/bleached/bad frame inside a long movie; a channel beyond
    :data:`FLOOR_SAMPLE_MAX_CHANNELS`) can still sit below the resolved level,
    clip to all zeros, fit 0 splats and go silently missing from the merge. Pass
    ``--floor none``, or an explicit numeric ``--floor N`` low enough for the
    dimmest slice, when a particular slice must be guaranteed to survive.
    """
    n_c = max(1, int(n_channels))
    n_t = max(1, int(n_timepoints))
    times = _evenly_spaced(n_t, min(n_t, FLOOR_SAMPLE_MAX_TIMEPOINTS))
    channels = _evenly_spaced(n_c, min(n_c, FLOOR_SAMPLE_MAX_CHANNELS))
    return [(t, c) for c in channels for t in times]


def _sample_batch_slices(
    input_path: Path,
    *,
    n_timepoints: int,
    n_channels: int,
    array_key: Optional[str],
    axes: Optional[str],
    axes_labels: Optional[List[str]],
    channel_shape: Tuple[int, ...],
    spatial_shape: Optional[Tuple[int, ...]],
) -> "List[Tuple[int, int, Any]]":
    """Read bounded raw samples used to resolve one shared normalization range."""
    from luxar.gsplats.fitting.preprocessing import (
        FLOOR_SAMPLE_BUDGET_VOXELS,
        _sample_volume_for_floor,
    )

    pairs = _floor_sample_pairs(n_timepoints, n_channels)
    budget = max(1, int(FLOOR_SAMPLE_BUDGET_VOXELS) // len(pairs))
    sampled = []
    for timepoint, channel in pairs:
        view = _pinned_slice_volume(
            input_path,
            channel=channel,
            timepoint=timepoint,
            array_key=array_key,
            axes=axes,
            axes_labels=axes_labels,
            channel_shape=channel_shape,
            spatial_shape=spatial_shape,
        )
        sample = _sample_volume_for_floor(view, budget)
        if sample is not None and sample.size:
            sampled.append((timepoint, channel, sample))
    return sampled


def _floor_resolution_pairs(
    n_timepoints: int,
    n_channels: int,
    denoise_h_values: Optional[dict[int, float]],
) -> List[Tuple[int, int]]:
    """Choose full-store raw pairs or full-time calibrated-channel pairs."""
    if denoise_h_values is None:
        return _floor_sample_pairs(n_timepoints, n_channels)
    calibrated_channels = sorted(denoise_h_values)
    if not calibrated_channels:
        raise ValueError("denoised floor resolution has no calibrated channels")
    n_t = max(1, n_timepoints)
    times = _evenly_spaced(n_t, min(n_t, FLOOR_SAMPLE_MAX_TIMEPOINTS))
    channel_positions = _evenly_spaced(
        len(calibrated_channels),
        min(len(calibrated_channels), FLOOR_SAMPLE_MAX_CHANNELS),
    )
    channels = [calibrated_channels[index] for index in channel_positions]
    return [(timepoint, channel) for channel in channels for timepoint in times]


def _resolve_floor_slice(
    view: Any,
    floor_spec: "str | float | None",
    *,
    budget: int,
    timepoint: int,
    channel: int,
    denoise_h_values: Optional[dict[int, float]],
    denoise_params: Optional[dict[str, Any]],
) -> Tuple[Optional[float], Optional[float], bool]:
    """Return one slice's level, optional sampled max, and whether it was read."""
    from luxar.gsplats.fitting.preprocessing import (
        _sample_volume_for_floor,
        resolve_volume_floor,
        resolve_volume_floor_denoised,
    )

    if denoise_h_values is None:
        sample = _sample_volume_for_floor(view, budget)
        if sample is None or sample.size == 0:
            return None, None, False
        return resolve_volume_floor(sample, floor_spec), float(sample.max()), True

    params = dict(denoise_params or {})
    denoise_h = denoise_h_values.get(channel)
    if denoise_h is None:
        aprint(
            f"Note: t={timepoint}, c={channel} has no calibrated denoise h; "
            "resolving this slice on the raw basis."
        )
    try:
        params["norm_range"] = _bounded_exact_range(view, budget)
    except (TypeError, ValueError) as exc:
        aprint(
            f"Note: t={timepoint}, c={channel} exact normalization range could "
            f"not be read ({exc}); resolving this slice on the raw basis."
        )
        return resolve_volume_floor(view, floor_spec, sample_budget=budget), None, True
    return (
        resolve_volume_floor_denoised(
            view,
            floor_spec,
            denoise_h=denoise_h,
            denoise_params=params,
            sample_budget=budget,
        ),
        None,
        True,
    )


def resolve_batch_floor(
    input_path: Path,
    floor_spec: "str | float | None",
    *,
    n_timepoints: int = 1,
    n_channels: int = 1,
    array_key: Optional[str] = None,
    axes: Optional[str] = None,
    axes_labels: Optional[List[str]] = None,
    channel_shape: Tuple[int, ...] = (),
    spatial_shape: Optional[Tuple[int, ...]] = None,
    denoise_h_values: Optional[dict[int, float]] = None,
    denoise_params: Optional[dict[str, Any]] = None,
    sampled_slices: "Optional[List[Tuple[int, int, Any]]]" = None,
) -> "Tuple[Optional[float], Optional[str | float]]":
    """Resolve the batch's background floor ONCE, globally for the whole run.

    ``batch-fit`` uses **one global level for the whole timelapse**: it is
    resolved once here (at plan time or in the dependent post-denoise stage),
    recorded in the manifest (``floor_level``), and handed as a concrete number
    to every ``(t, c)`` task. Forwarding the *spec*
    instead would make each task re-estimate on its own timepoint — a
    time-varying pedestal, i.e. brightness flicker across the merged partition
    (issue #1174).

    One global level has to be safe for the dimmest **sampled** slice, not just
    for a typical one: the tile workers subtract it unguarded, so a level above
    some ``(t, c)``'s maximum clips that whole sub-volume to zero. Status and
    merge now reject an all-empty uniform slice, but the level is still biased
    low to avoid destructive over-subtraction in the first place. Under
    ``clip(V - level, 0)`` a too-LOW level is a recoverable under-subtraction
    while a too-HIGH one destroys signal, so the level is biased low:

    * the spec is resolved on a bounded, deterministic set of evenly spaced
      slices spanning the store's FULL time extent. Raw-basis resolution also
      spans the full channel extent; denoised resolution spans the selected,
      calibrated channels because NLM strength is channel-specific
      (:func:`_floor_sample_pairs`, at most :data:`FLOOR_SAMPLE_MAX_SLICES`) —
      each read through a lazy axis-pinned view with the whole-volume sample
      budget divided among them;
    * the global level is the **MINIMUM** of the per-slice levels, a lower bound
      on every SAMPLED slice's pedestal. Each per-slice level is itself already
      guarded below its own sampled max, so the minimum is below every sampled
      max — the invariant that makes the reduction unable to erase a sampled
      slice;
    * if any sampled slice resolves to "subtract nothing" (a zero level, or the
      "would erase all signal" guard refusing it), suppression is downgraded to
      none for the whole run, loudly, rather than erasing a slice.

    Because the sampled timepoints span the whole store rather than the selection,
    the same store resolves the same level for ``--timepoints 0:50`` and ``0:100``.

    RESIDUAL RISK: the bound holds for the sampled slices only. A dimmer
    NON-sampled slice (a blank/bleached frame between samples, a channel above
    the channel cap) can still be clipped to zero and vanish from the merge; see
    :func:`_floor_sample_pairs`. ``--floor none`` or an explicit numeric
    ``--floor N`` is the escape hatch.

    A volume is only opened when the spec is volume-derived (``auto`` / ``pNN`` —
    pass :func:`effective_floor_spec`, not the raw ``--floor``); ``none`` and a
    numeric spec are already concrete, so they cost no read and round-trip
    exactly as before. An explicitly disabled spec (``None``, e.g. ``floor:
    null`` in a YAML config) emits no ``--floor`` at all, leaving the task's own
    config to disable it.

    A NEGATIVE resolved level (dark-frame-corrected data) cannot be forwarded as
    a concrete ``--floor`` (neither the CLI nor ``fit_gaussian_splats`` accepts
    one), so the SPEC is forwarded and each ``(t, c)`` task resolves it itself —
    announced loudly, because pedestals may then differ across the run — and the
    manifest records ``floor_level=None`` ("not pinned").

    Returns ``(level, forward)``: the numeric level (``None`` when
    disabled/refused/unset) and the value to put in ``fit_args["floor"]``
    (``None`` = emit no ``--floor`` at all).
    """
    from luxar.cli.gsplat_ops.fitting.fit_utils import (
        floor_spec_needs_volume,
        resolve_shared_floor,
        validate_floor_spec,
    )
    from luxar.gsplats.fitting.preprocessing import (
        FLOOR_SAMPLE_BUDGET_VOXELS,
    )

    if floor_spec is None:
        return None, None
    # Validate BEFORE reading anything: a bad spec must not cost a volume read,
    # and must surface as a usage error rather than a traceback.
    validate_floor_spec(floor_spec)

    if not floor_spec_needs_volume(floor_spec):
        # Concrete already ("none" / a number): resolved without touching data,
        # so nothing is guarded here — exactly as before this resolution existed.
        return resolve_shared_floor(
            None, floor_spec, guard_numeric=False, scope="every (t, c) task"
        )

    pairs = _floor_resolution_pairs(n_timepoints, n_channels, denoise_h_values)
    budget = max(1, int(FLOOR_SAMPLE_BUDGET_VOXELS) // len(pairs))
    sampled_times = sorted({t for t, _ in pairs})
    sampled_channels = sorted({c for _, c in pairs})
    levels: List[Optional[float]] = []
    sampled_by_pair = (
        {(timepoint, channel): sample for timepoint, channel, sample in sampled_slices}
        if sampled_slices is not None and denoise_h_values is None
        else {}
    )
    with asection(
        f"Resolving background floor '{floor_spec}' (minimum over "
        f"{len(pairs)} slices; sampled T={sampled_times}, C={sampled_channels})"
    ):
        for t, c in pairs:
            view = sampled_by_pair.get((t, c))
            if view is None:
                view = _pinned_slice_volume(
                    input_path,
                    channel=c,
                    timepoint=t,
                    array_key=array_key,
                    axes=axes,
                    axes_labels=axes_labels,
                    channel_shape=channel_shape,
                    spatial_shape=spatial_shape,
                )
            level_here, sampled_max, processed = _resolve_floor_slice(
                view,
                floor_spec,
                budget=budget,
                timepoint=t,
                channel=c,
                denoise_h_values=denoise_h_values,
                denoise_params=denoise_params,
            )
            if not processed:
                continue
            levels.append(level_here)
            max_note = (
                "" if sampled_max is None else f" (sampled max {sampled_max:.6g})"
            )
            aprint(
                f"t={t}, c={c}: level "
                f"{'none' if level_here is None else format(level_here, '.6g')}"
                f"{max_note}"
            )

        level: Optional[float] = None
        if levels and all(item is not None for item in levels):
            # Every per-slice level is already guarded below ITS OWN sampled max,
            # so this minimum is below EVERY sampled max: the reduction cannot
            # erase a sampled slice, and no extra check is needed here.
            level = min(item for item in levels if item is not None)
        if not levels:
            aprint(
                f"⚠ Background floor '{floor_spec}': not one of the "
                f"{len(pairs)} sampled (t, c) slices could be read (all empty): "
                f"floor suppression is DISABLED for the WHOLE run — every "
                f"(t, c) task is given --floor none."
            )
            return None, "none"
        if level is None:
            aprint(
                f"⚠ Background floor '{floor_spec}' resolves to no suppression on "
                f"at least one sampled (t, c) slice (a zero level, or the 'would "
                f"erase all signal' guard refused it): floor suppression is "
                f"DISABLED for the WHOLE run — every (t, c) task is given "
                f"--floor none."
            )
            return None, "none"
        if level < 0.0:
            aprint(
                f"⚠ Background floor '{floor_spec}' resolves to a NEGATIVE level "
                f"({level:.6g}) (dark-frame-corrected data?), which cannot be "
                f"forwarded as a concrete --floor (neither --floor nor "
                f"fit_gaussian_splats accepts a negative value). The SPEC is "
                f"forwarded instead, so every (t, c) task resolves it on its own "
                f"sub-volume and pedestals may differ across the run; the "
                f"manifest records floor_level=None (not pinned). Pass "
                f"--floor none to disable suppression, or an explicit "
                f"non-negative --floor N to pin one level."
            )
            return level, floor_spec
        aprint(
            f"Floor suppression: every (t, c) task subtracts background level "
            f"{level:.6g} (the minimum over the sampled slices)"
        )
    return level, level


def _resolve_and_record_floor(
    input_path: Path,
    fit: FitConfig,
    fit_args: dict,
    *,
    n_timepoints: int,
    n_channels: int,
    array_key: Optional[str],
    axes: Optional[str],
    axes_labels: Optional[List[str]] = None,
    channel_shape: Tuple[int, ...] = (),
    spatial_shape: Optional[Tuple[int, ...]] = None,
    sampled_slices: "Optional[List[Tuple[int, int, Any]]]" = None,
) -> "Tuple[Optional[float], Optional[float]]":
    """:func:`resolve_batch_floor` + write the level into ``fit_args``/the manifest.

    Returns ``(level, recorded_level)``. ``recorded_level`` is the manifest's
    ``floor_level`` and is only set when the tasks really are handed a number:
    the ``"none"`` forward leaves nothing to subtract, and a forwarded SPEC (the
    negative-level case) means no level is pinned at all.
    """
    level, forward = resolve_batch_floor(
        input_path,
        effective_floor_spec(fit),
        n_timepoints=n_timepoints,
        n_channels=n_channels,
        array_key=array_key,
        axes=axes,
        axes_labels=axes_labels,
        channel_shape=channel_shape,
        spatial_shape=spatial_shape,
        sampled_slices=sampled_slices,
    )
    recorded: Optional[float] = None
    if forward is not None:
        fit_args["floor"] = str(forward)
        if not isinstance(forward, str):
            recorded = float(forward)
    return level, recorded


def resolve_batch_norm_range(
    input_path: Path,
    norm_percentile: float,
    *,
    n_timepoints: int = 1,
    n_channels: int = 1,
    array_key: Optional[str] = None,
    axes: Optional[str] = None,
    axes_labels: Optional[List[str]] = None,
    channel_shape: Tuple[int, ...] = (),
    spatial_shape: Optional[Tuple[int, ...]] = None,
    sampled_slices: "Optional[List[Tuple[int, int, Any]]]" = None,
) -> "Optional[Tuple[float, float]]":
    """Resolve one raw-input normalization range for the whole batch run."""
    import numpy as np

    from luxar.gsplats.fitting.preprocessing import (
        _norm_range_has_usable_span,
        resolve_volume_norm_range,
    )

    pairs = _floor_sample_pairs(n_timepoints, n_channels)
    sampled = sampled_slices
    if sampled is None:
        sampled = _sample_batch_slices(
            input_path,
            n_timepoints=n_timepoints,
            n_channels=n_channels,
            array_key=array_key,
            axes=axes,
            axes_labels=axes_labels,
            channel_shape=channel_shape,
            spatial_shape=spatial_shape,
        )
    with asection(
        f"Resolving normalization range over {len(pairs)} slices spanning "
        f"T={n_timepoints}, C={n_channels}"
    ):
        if not sampled:
            aprint(
                "No sampled voxels; each task resolves its own normalization "
                "range (no shared range pinned)."
            )
            return None
        norm_range = resolve_volume_norm_range(
            np.concatenate([sample for _, _, sample in sampled]),
            norm_percentile,
            verbose=False,
        )
        if not _norm_range_has_usable_span(norm_range):
            aprint(
                f"Sampled normalization range [{norm_range[0]:.6g}, "
                f"{norm_range[1]:.6g}] has no usable extent; each task resolves "
                "its own scale."
            )
            return None
        aprint(
            f"Every (t, c) task uses normalization range "
            f"[{norm_range[0]:.6g}, {norm_range[1]:.6g}]"
        )
        return norm_range


def _record_batch_norm_range(
    fit_args: dict[str, Any], norm_range: "Optional[Tuple[float, float]]"
) -> None:
    """Forward a resolved batch range when one is usable."""
    if norm_range is not None:
        fit_args["norm_range"] = f"{norm_range[0]:.17g},{norm_range[1]:.17g}"


def _resolve_planned_norm_range(
    *,
    input_path: Path,
    configured: "Optional[Tuple[float, float]]",
    denoise: bool,
    norm_percentile: float,
    n_timepoints: int,
    n_channels: int,
    array_key: Optional[str],
    axes: Optional[str],
    axes_labels: List[str],
    channel_shape: Tuple[int, ...],
    spatial_shape: Tuple[int, ...],
    sampled_slices: "List[Tuple[int, int, Any]]",
) -> "Optional[Tuple[float, float]]":
    """Resolve the batch range without putting raw sampled bounds on denoised data."""
    if configured is not None:
        return configured
    if denoise:
        aprint(
            "Denoising is enabled; each task resolves its normalization range "
            "on the data it fits."
        )
        return None
    return resolve_batch_norm_range(
        input_path,
        norm_percentile,
        n_timepoints=n_timepoints,
        n_channels=n_channels,
        array_key=array_key,
        axes=axes,
        axes_labels=axes_labels,
        channel_shape=channel_shape,
        spatial_shape=spatial_shape,
        sampled_slices=sampled_slices,
    )


def _load_scan_volume(
    input_path: Path,
    timepoints: List[int],
    *,
    channel: int,
    array_key: Optional[str],
    axes: Optional[str],
    axes_labels: Optional[List[str]] = None,
    channel_shape: Tuple[int, ...] = (),
    spatial_shape: Optional[Tuple[int, ...]] = None,
) -> Any:
    """Per-voxel MAX over ``timepoints`` of one channel — the content plan's basis.

    Max-projecting the sampled timepoints makes the box plan cover any region with
    signal at ANY of them (no holes where content moved over time). Each timepoint
    is materialized through the SAME label-driven pinned view the floor resolution
    uses (:func:`_pinned_slice_volume`): with no explicit ``--axes``,
    ``load_volume``'s 4D positional heuristic reads a ``(T, Z, Y, X)`` store as
    ``CZYX`` and ignores ``timepoint``, so every sample here used to be ``t=0``
    and ``--plan-samples`` silently no-oped (#1174).
    """
    import numpy as _np

    def _load(t: int) -> Any:
        return _materialize(
            _pinned_slice_volume(
                input_path,
                channel=channel,
                timepoint=t,
                array_key=array_key,
                axes=axes,
                axes_labels=axes_labels,
                channel_shape=channel_shape,
                spatial_shape=spatial_shape,
            )
        )

    vol = _load(timepoints[0])
    for t in timepoints[1:]:
        vol = _np.maximum(vol, _load(t))
    return vol


def _parse_slice(s: str, max_val: int) -> list[int]:
    """Parse a Python-style ``start:stop:step`` slice (or a single index)."""
    parts = s.split(":")
    if len(parts) == 1:
        return [int(parts[0])]
    start = int(parts[0]) if parts[0] else 0
    stop = int(parts[1]) if len(parts) > 1 and parts[1] else max_val
    step = int(parts[2]) if len(parts) > 2 and parts[2] else 1
    return list(range(start, stop, step))


def _record_merge_refine(args: dict, merge: MergeConfig) -> None:
    """Record the refine knobs into the stored manifest args, validated here.

    Plan time is the right place: the merge job would otherwise only discover a
    typo after every tile had been fitted. The stored KEYS ("refine",
    "refine-iters") are what the Slurm emitter turns back into CLI flags, so they
    must match `batch-fit merge`'s option names.
    """
    if merge.refine is None and merge.refine_iters is None:
        return
    from luxar.cli.gsplat_ops.recipe_shared import validate_refine

    norm = validate_refine(
        merge.refine,
        merge.refine_iters,
        flag="--merge-refine",
        iters_flag="--merge-refine-iters",
    )
    if merge.refine is not None:
        args["refine"] = norm
    if merge.refine_iters is not None:
        args["refine-iters"] = str(merge.refine_iters)


def _validate_merge_refine_source(
    merge: MergeConfig,
    axes_list: Optional[List[str]],
    n_timepoints: int,
    n_channels: int,
) -> None:
    """Refuse a planned merge-time volume re-fit this source cannot serve.

    The re-fit re-opens THIS input at merge time and pins or walks its non-spatial
    axes, which needs axis labels and a single channel. Checked HERE because the
    merge job runs only after every tile has been fitted — discovering it there
    costs the whole fit. (The merge front door re-checks the same rule: a manifest
    can predate this, or be written by hand.)

    Reads the NORMALISED mode, the way ``_record_merge_refine`` records it —
    ``--merge-refine " volume "`` is stored as ``volume`` and must not slip past
    this check on the strength of its whitespace.
    """
    if (merge.refine or "").strip() != "volume":
        return
    from luxar.gsplats.batch.merge_orchestrator import volume_refit_source_error

    problem = volume_refit_source_error(
        ",".join(axes_list) if axes_list else None, n_timepoints, n_channels
    )
    if problem:
        raise typer.BadParameter(f"--merge-refine volume: {problem}")


def _loader_task_axes(
    ndim: int, emits_channel: bool, emits_timepoint: bool
) -> "Tuple[Optional[int], Tuple[int, ...]]":
    """Which axes the POSITIONAL loader indexes away for one task, and as what.

    Returns ``(time_axis, channel_axes)``: the axis the loader hands the task's
    ``--timepoint`` to (or ``None``), and the axes it folds the flat ``--channel``
    index over, in decode order. Everything else is left for the volume. Read off
    :func:`luxar.io.volume._load_zarr_volume` branch by branch:

    * ``ndim >= 6``: ``arr[t, *decode_flat_channel_index(channel, shape[1:ndim-3])]``
      — axis 0 takes the timepoint, axes ``1 … ndim-4`` fold the channel index.
    * ``ndim == 5``: ``arr[t, c, :, :, :]``.
    * ``ndim == 4``: ``arr[channel]`` when the argv carries ``--channel``, else
      ``arr[timepoint]`` when it carries ``--timepoint``, else the whole array.
      At most ONE axis is consumed and channel wins — the one ndim where the
      branch depends on which flags :func:`~luxar.gsplats.batch.fit_command.build_task_fit_argv`
      emitted.
    * ``ndim <= 3``: ``np.array(arr)`` — nothing is consumed and both flags are
      ignored.
    """
    if ndim >= 6:
        return 0, tuple(range(1, ndim - 3))
    if ndim == 5:
        return 0, (1,)
    if ndim == 4:
        if emits_channel:
            return None, (0,)
        if emits_timepoint:
            return 0, ()
    return None, ()


def _worker_reproduces_the_plan(
    info: "OMEZarrInfo", emits_channel: bool, emits_timepoint: bool
) -> bool:
    """Whether a positionally-slicing worker lands on the plan's spatial volume.

    Asked of the decomposition discovery ITSELF used —
    ``time_axis``/``channel_indices``/``spatial_indices`` on
    :class:`~luxar.io.ome_zarr.OMEZarrInfo` — never of the axis LABELS. NGFF
    classifies by the ``type`` field, so re-deriving the roles from the names
    disagrees in both directions: a channel axis named ``stain`` looks spatial to
    a name rule (a false refusal of a canonical 5D store that always worked),
    while an axis typed ``view`` is spatial to the parser and channel-like to a
    name rule (a false pass, whose plan tiles a 4-D "spatial" shape the worker
    never sees).

    Two sets are compared, rather than a per-ndim table of whole layouts (which
    demanded an exact axis partition and so refused ``t,c,y,x`` at ``(5, 1,
    32, 32)`` — a 2D timelapse with a singleton channel — while admitting its
    ``T=1`` sibling):

    * what the loader CONSUMES for a task (:func:`_loader_task_axes`), and
    * what the plan FANS over, ``{time_axis} | channel_indices``, one task per
      combination.

    Reproducible when both of these hold:

    1. the two agree axis for axis, once singleton axes are dropped from both:
       the loader's time axis is the plan's (or neither varies), and its folded
       channel axes are the plan's channel axes in the same order. Dropping
       singletons is exact, not a fudge — a size-1 axis is indexed at 0 whatever
       the task index says, and ``decode_flat_channel_index`` gives it digit 0
       and a stride of 1, so it contributes nothing to either side. This is the
       clause that makes each task a DISTINCT volume. It also subsumes "no axis
       the plan calls spatial is indexed away": a consumed non-singleton axis
       outside the fan shows up on the loader's side of this comparison and on
       nothing on the plan's.
    2. the shape the loader hands back — the axes it did not consume — is the
       plan's ``spatial_shape``, both with size-1 axes dropped, because
       ``load_volume`` squeezes its positional result (``luxar/io/volume.py``,
       the ``np.squeeze`` in the ``axes is None`` branch). That squeeze is what
       lets a layout whose non-spatial axes are all singletons — ``z,y,x,c`` at
       ``(16, 32, 32, 1)``, ``t,c,y,x`` at ``(1, 1, 32, 32)`` — plan even though
       its axes do not lead.

    The plan now applies the same singleton-spatial-axis squeeze through
    :func:`_worker_spatial_shape`, so clause 2's squeezed-to-squeezed comparison
    is an equality between the shape the plan records and the worker emits.
    """
    shape = tuple(info.shape)
    ndim = len(shape)
    loader_t, loader_c = _loader_task_axes(ndim, emits_channel, emits_timepoint)

    def _varying(axis: Optional[int]) -> Optional[int]:
        return axis if axis is not None and shape[axis] > 1 else None

    # 1. same axes, same roles, same fold order (singletons excluded: they carry
    #    no index either way).
    if _varying(loader_t) != _varying(info.time_axis):
        return False
    if tuple(a for a in loader_c if shape[a] > 1) != tuple(
        a for a in info.channel_indices if shape[a] > 1
    ):
        return False

    # 2. what is left after the loader's indexing IS the planned volume.
    consumed = set(loader_c) | ({loader_t} if loader_t is not None else set())
    loaded = tuple(s for i, s in enumerate(shape) if i not in consumed)
    return tuple(s for s in loaded if s != 1) == tuple(
        s for s in info.spatial_shape if s != 1
    )


def _axes_spec_luxar_can_slice(
    labels: Sequence[str],
) -> "Tuple[Optional[str], Optional[str]]":
    """``(spec, None)`` when ``--axes <spec>`` would really run, else ``(None, why)``.

    Two conditions, both asked of :func:`luxar.io.volume._axis_kind` itself rather
    than of a copied word list, because that function IS the ``--axes``
    vocabulary — and it is deliberately narrower than discovery's (no
    ``view``/``angle``, and it raises rather than defaulting to spatial). Only its
    ``ValueError`` is caught, and only per label, so nothing else is swallowed.

    1. every label is in that vocabulary, else the spec is rejected outright;
    2. at most ONE time label. One ``--timepoint`` cannot address multiple time
       axes independently, and :func:`luxar.io.volume._apply_axes_spec` rejects
       such a spec. Multiple channel-like labels are valid: ``--channel`` is a
       flat row-major index decoded across all of them.
    """
    from luxar.io.volume import _axis_kind

    normalised = [str(label).strip().lower() for label in labels]
    kinds: List[str] = []
    unrecognised: List[str] = []
    for label in normalised:
        try:
            kinds.append(_axis_kind(label))
        except ValueError:
            unrecognised.append(label)
    if unrecognised:
        return None, (
            f"this store's own labels are not ones luxar can slice by "
            f"({', '.join(repr(u) for u in unrecognised)} unrecognised), so they "
            f"cannot be quoted back at you"
        )
    time_labels = [
        label for label, kind in zip(normalised, kinds, strict=True) if kind == "t"
    ]
    if len(time_labels) > 1:
        return None, (
            f"this store folds more than one time axis "
            f"({', '.join(repr(label) for label in time_labels)}), but one "
            "--timepoint cannot address them independently, so the discovered "
            "labels cannot be quoted back at you"
        )
    return ",".join(normalised), None


def _refuse_layout_the_workers_cannot_slice(
    axes_list: Optional[List[str]],
    info: "OMEZarrInfo",
    emits_channel: bool,
    emits_timepoint: bool,
) -> None:
    """Refuse a discovered layout the per-task worker cannot reproduce.

    A no-op when the user passed ``--axes`` (``axes_list``), which the manifest
    forwards to every worker so it slices by LABEL. Otherwise the manifest records
    no axes and every worker falls back to the POSITIONAL slicing in
    :func:`luxar.io.volume._load_zarr_volume`, whose per-task indexing
    :func:`_loader_task_axes` reads off branch by branch and
    :func:`_worker_reproduces_the_plan` compares against the plan's own fan.

    Discovery now reads real NGFF labels, so it can learn a layout — a ``(t, c,
    y, x)`` store, say — that the positional loader silently mis-slices: the plan
    fans over 5 timepoints × 3 channels while every worker prefers ``--channel``
    and ignores ``--timepoint``, so the merged partition is the same 3 volumes
    repeated. Nothing raises, so this does.

    The ``--axes`` escape hatch is only SUGGESTED when running it would really
    reproduce the plan (:func:`_axes_spec_luxar_can_slice`): the spec's
    vocabulary is narrower than discovery's, so labels such as ``stain`` and
    heuristic ``dim0…dimN`` cannot be quoted back. A second time axis is also
    unquotable because one ``--timepoint`` cannot address both independently.
    Every other suggested spec is preflighted against the worker vocabulary,
    including folded channel-like axes decoded by one flat ``--channel``.
    """
    if axes_list is not None:
        return
    if _worker_reproduces_the_plan(info, emits_channel, emits_timepoint):
        return

    labels = [str(a) for a in info.axes]
    spec, why = _axes_spec_luxar_can_slice(labels)
    if spec is not None:
        advice = f"Pass '--axes {spec}' to slice by label instead."
    else:
        reason = why or "this store's labels cannot be quoted back at you"
        advice = (
            f"{reason[0].upper()}{reason[1:]}: pass '--axes' with {len(labels)} "
            "labels of your own — time/t, channel/c/ch/camera/cam or z/y/x, one "
            "per dimension — saying what each axis means."
        )
    raise typer.BadParameter(
        f"the store's axes are '{','.join(labels)}' with shape "
        f"{tuple(info.shape)}, a layout batch-fit's per-task volume loader "
        f"cannot reproduce: without --axes it slices by POSITION — at most a "
        f"leading timepoint and the channel-like axes before the last three, "
        f"and at 3-D or less nothing at all. Planning "
        f"would fan out over axes the workers never see, producing duplicate "
        f"tiles. {advice}"
    )


def _validate_merge_refine_frame(
    merge: MergeConfig, grid_scale: Optional[List[float]]
) -> None:
    """Refuse a planned merge-time volume re-fit whose crops would be mis-framed.

    The second half of the same fail-fast pair as
    :func:`_validate_merge_refine_source`: that one asks whether the source's
    AXES can be mapped, this one whether its COORDINATES can. A per-part re-fit
    reads the part's ``bsp_tree`` cell as VOXEL INDICES into the source, so a
    ``grid_scale`` — for a batch run, a ``voxel_size`` in the run's ``--config``
    with the default ``output_space: real`` — puts every crop a factor off,
    exactly the reason ``gsplat fit`` refuses ``--refine volume`` under the same
    conditions
    (:func:`~luxar.cli.gsplat_ops.fitting.fit_utils.reject_rescaled_volume_refit`).
    A config ``downscale:`` is deliberately NOT a term in this factor: every
    batch task rescales its splats back to the FULL-resolution frame the planner
    tiled (see :func:`resolve_uniform_grid_scale`), so it cannot mis-frame a
    crop, and counting it would refuse a re-fit that works.
    Checked at plan time so a typo costs nothing; the merge front door checks it
    again off the same recorded factor.
    """
    if (merge.refine or "").strip() != "volume":
        return
    from luxar.gsplats.batch.merge_orchestrator import volume_refit_frame_error

    problem = volume_refit_frame_error(grid_scale)
    if problem:
        raise typer.BadParameter(f"--merge-refine volume: {problem}")


def validate_config_downscale(fit: FitConfig, ndim: int) -> Optional[Tuple[int, ...]]:
    """Normalize a ``downscale:`` in the run's fit config, or reject it as a usage error.

    ``batch-fit`` forwards its ``--config``/``--preset`` VERBATIM to every array
    task, so this key reaches the workers whether or not the planner looks at it.
    A MALFORMED value (``0``, ``2.5``, a wrong-length list) is therefore a
    plan-time usage error in every mode: the resolver that used to catch it no
    longer sees this key at all (see :func:`resolve_uniform_grid_scale`), and
    without this check it would reach every task as a raw traceback at fit time.

    Called EARLY — before any voxel is read and before anything is written —
    because it needs only the merged config and ``ndim``. Checking it alongside
    the decomposition instead made a malformed value in a CONTENT run surface
    only after the shared box plan had been scanned (a max-projection over up to
    ``--plan-samples`` timepoints, tens of GB on a real timelapse), the output
    directory created and ``plan.json`` written: measured, the "Plan: 6 boxes …
    → out-c/plan.json" line printed and the file existed before the refusal.

    Guarantees ONLY that the value is well-formed; whether the run's
    decomposition can actually complete with it is
    :func:`refuse_downscale_grid_mismatch`'s question, asked once the tile grid
    is known. Returns the per-axis factors, or ``None`` for an absent value and
    for a documented no-op (``downscale: 1``, ``[1, 1, 1]`` — all-ones, nothing
    is decimated). Raises :class:`typer.BadParameter` on a malformed one.
    """
    from luxar.cli.gsplat_config import load_fit_config
    from luxar.gsplats.fitting.downscale import normalize_downscale

    try:
        config = load_fit_config(preset=fit.preset, config_path=fit.config)
    except Exception as exc:
        raise typer.BadParameter(
            f"could not read the fit config for this run (--preset "
            f"{fit.preset!r}, --config {str(fit.config) if fit.config else None!r}): "
            f"{exc}"
        ) from exc
    raw = config.get("downscale")
    if raw is None:
        return None
    try:
        factors = normalize_downscale(raw, ndim)
    except (TypeError, ValueError) as exc:
        # TypeError as well as ValueError: a float (`downscale: 2.5`) is neither
        # an int nor iterable, so it never reaches the function's own checks.
        raise typer.BadParameter(
            f"config `downscale: {raw!r}` is invalid: {exc}"
        ) from exc
    return tuple(factors) if factors is not None else None


def refuse_downscale_grid_mismatch(
    factors: Optional[Tuple[int, ...]],
    *,
    mode: str,
    spatial: Tuple[int, ...],
    tile_size: int,
    tile_overlap: int,
    total_tasks: int,
) -> None:
    """Refuse a config ``downscale:`` whose workers would re-tile a DIFFERENT grid.

    Each uniform array task is a ``fit --tile k/M`` run that decimates its own
    volume and then recomputes the tile grid on the DECIMATED shape, while the
    planner tiled the FULL-RESOLUTION one (#1624). When those two grids disagree
    the run cannot complete: measured on a 32^3 store with ``--tile-size 12
    --overlap 2``, the plan builds 64 tiles per slot and a ``downscale: 2`` worker
    sees 8, so tiles 8..63 exit 1 with "tile index out of range"
    (``fit_single_tile``) and no merge ever happens.

    A decimating value is NOT fatal per se, so the disagreement is measured
    directly rather than inferred from a tile count: both grids are built with
    :func:`~luxar.gsplats.tiling.compute_tile_specs` (the planner's on ``spatial``,
    the worker's on the decimated shape, same ``tile_size``/``tile_overlap``) and
    they AGREE when they hold the same number of tiles and every worker origin,
    multiplied per axis by ``factors`` — which is exactly what
    ``rescale_and_save`` does to that tile's splats — is the corresponding planner
    origin. Three shapes are measured to complete correctly and pass that test
    (or skip it), instead of being refused:

    * a factor that decimates only axes holding a SINGLE tile at full resolution.
      Measured: spatial ``(40, 8, 8)`` with ``--tile-size 12 --overlap 0`` and
      ``downscale: [1, 2, 2]`` gives 4 planner tiles and 4 worker tiles on the
      decimated ``(40, 4, 4)``, and all four real ``fit --tile k/4`` workers exit
      0 with their splats in the correct full-resolution slabs; spatial
      ``(10, 24, 24)`` with ``--tile-size 12 --overlap 2`` and
      ``downscale: [2, 1, 1]`` gives 9 tiles on both sides with identical origins
      after the worker's rescale.
    * ``mode == "content"`` — a content task is ``fit --tiling content --plan
      plan.json --plan-box K``, which never re-tiles (it bounds-checks ``K``
      against the same plan JSON the planner wrote) and whose ``_fit_one_box``
      crops the FULL-resolution volume; ``finalize_results`` then rescales the
      crop's splats back to full resolution before the box origin is added, so
      the centers land in the same global range as without the key.
    * a single-tile plan — the worker's decimated volume still yields exactly one
      tile at the same origin, so ``--tile 0/1`` passes its bounds check and the
      splats come back at full resolution. (Not special-cased: it is the
      grid comparison's trivial pass.)

    ``factors`` is what :func:`validate_config_downscale` already normalized from
    the same config, so the value's well-formedness is settled before this runs
    and ``None`` (absent, or an all-ones no-op) returns silently.
    """
    if factors is None or mode == "content":
        return
    from luxar.gsplats.tiling import compute_tile_specs

    # `downscale_volume` decimates by strided slicing, so the worker's shape is
    # the CEIL of the division, and its grid is built on that.
    decimated_shape = tuple(-(-s // f) for s, f in zip(spatial, factors))
    planned = compute_tile_specs(spatial, tile_size, tile_overlap)
    worker = compute_tile_specs(decimated_shape, tile_size, tile_overlap)
    rescaled = [tuple(o * f for o, f in zip(spec.origin, factors)) for spec in worker]
    if len(worker) == len(planned) and all(
        got == tuple(want.origin) for got, want in zip(rescaled, planned)
    ):
        return  # the worker re-tiles the SAME grid: nothing to refuse
    if len(worker) < len(planned):
        # `ceil(s / f)` is monotone per axis, so a genuine disagreement with
        # fewer worker tiles is the out-of-range case — and this range is
        # non-empty exactly because the count is strictly smaller.
        detail = (
            f"seeing only {len(worker)} tiles — so each slot's tiles "
            f"{len(worker)}..{len(planned) - 1} exit with 'tile index out of "
            f"range' and no merge ever happens"
        )
    else:
        # Equal counts but different origins. `ceil(s / f) <= s` per axis makes
        # the worker's tile count monotone non-increasing, so nothing reaches
        # here today; the branch exists so the message can never assert an EMPTY
        # tile range, which is exactly how the first version of this gate went
        # wrong (it printed "tiles 4..3" for an anisotropic factor whose two
        # grids in fact agreed).
        detail = (
            f"seeing {len(worker)} tiles whose origins are not the planned ones "
            f"once rescaled — so each task fits and saves a different "
            f"sub-volume than the merge expects"
        )
    raise typer.BadParameter(
        f"config `downscale:` (per-axis factors {factors}) cannot be used with "
        f"this uniform batch: the plan tiles the full-resolution shape "
        f"{tuple(spatial)} into {len(planned)} tiles per slot ({total_tasks} "
        f"tasks in all), but every `fit --tile k/M` worker decimates its own "
        f"volume to {decimated_shape} first and re-tiles THAT, {detail}. Use "
        f"`gsplat fit -j N --downscale ...` (one fit whose tile grid is computed "
        f"AFTER the downscale), or decimate the store first and run batch-fit on "
        f"that. (`--tiling content` never re-tiles, and a factor that decimates "
        f"only single-tile axes leaves the grid unchanged, so both complete fine "
        f"with this key.)"
    )


def resolve_uniform_grid_scale(fit: FitConfig, ndim: int) -> Optional[List[float]]:
    """The frame factor to record on the manifest for a uniform batch (#1587).

    Resolved HERE, at plan time, because this is where the workers' merged fit
    config is in hand: every array task is a ``fit --tile k/M`` run with this
    run's ``--preset`` and (verbatim) its ``--config``, so a ``voxel_size`` with
    the default ``output_space: real`` in that config moves the tasks' splats out
    of the manifest's voxel tile grid. Doing it at MERGE time instead would mean
    re-reading a YAML that may have moved (planes silently dropped) or been
    replaced by an unrelated same-named file (planes silently wrong), and would
    put a ``luxar.cli`` import inside ``luxar.gsplats``.

    A config ``downscale:`` is deliberately NOT composed in, unlike in the
    single-fit ``fit --tiling uniform -j N --downscale`` parent that
    :func:`~luxar.gsplats.tiling.resolve_grid_scale`'s downscale term exists for
    (#1587). There the parent's grid IS built on the decimated shape, so a tile
    origin ``o`` lands at ``o * f``. In a BATCH run the planner tiles the
    FULL-resolution shape and every task rescales back to it — the ``--tile k/M``
    worker in ``rescale_and_save``, the ``--plan-box K`` worker in
    ``finalize_results`` — so the splats never leave the grid's frame and the
    term would simply be wrong. Measured: recording ``[2, 2, 2]`` for a
    single-tile batch left the split planes unchanged but falsely refused
    ``--merge-refine volume`` (:func:`_validate_merge_refine_frame`). Where a
    decimating value genuinely cannot complete — a uniform plan whose workers
    would re-tile a different grid — it is refused outright by
    :func:`refuse_downscale_grid_mismatch`, not recorded.

    Returns the per-axis factor, or ``None`` when the two frames already agree
    (the common case — recorded as an absent ``grid_scale``). Raises
    :class:`typer.BadParameter` when the config cannot be read or holds a frame
    the resolver refuses: at plan time the user is still here to fix it.
    """
    from luxar.cli.gsplat_config import load_fit_config
    from luxar.gsplats.tiling import resolve_grid_scale

    try:
        config = load_fit_config(preset=fit.preset, config_path=fit.config)
    except Exception as exc:
        raise typer.BadParameter(
            f"could not read the fit config for this run (--preset "
            f"{fit.preset!r}, --config {str(fit.config) if fit.config else None!r}): "
            f"{exc}"
        ) from exc
    try:
        scale = resolve_grid_scale(
            ndim,
            # No `downscale_factors=`: a batch task rescales its splats back to
            # the full-resolution frame this planner tiled, so the term does not
            # apply here (see the docstring). It stays load-bearing for the
            # single-fit `-j N` caller, whose grid IS the decimated one.
            voxel_size=config.get("voxel_size"),
            output_space=config.get("output_space", "real"),
        )
    except ValueError as exc:
        raise typer.BadParameter(
            f"the fit config for this run states a coordinate frame the tile "
            f"grid cannot be reconciled with: {exc}"
        ) from exc
    return list(scale) if scale is not None else None


def _physical_voxel_size(
    fit: FitConfig,
    ome_info: "OMEZarrInfo",
    axes_list: Optional[List[str]],
    ndim: int,
) -> "Tuple[Optional[List[float]], bool]":
    """Resolve ``--physical`` spacing and whether workers need an override."""
    if not fit.physical:
        return None, False

    from luxar.cli.gsplat_config import load_fit_config
    from luxar.gsplats.tiling import resolve_grid_scale

    config = load_fit_config(preset=fit.preset, config_path=fit.config)
    if config.get("output_space", "real") != "real":
        raise typer.BadParameter(
            "--physical requires output_space: real; remove the config override "
            "or omit --physical to keep voxel coordinates."
        )

    configured = config.get("voxel_size")
    discovered = ome_info.voxel_size
    if discovered is not None and axes_list is None:
        discovered = tuple(
            spacing
            for size, spacing in zip(ome_info.spatial_shape, discovered, strict=True)
            if size != 1
        )
    selected = configured if configured is not None else discovered
    if selected is None:
        raise typer.BadParameter(
            "--physical requested physical coordinates, but the selected OME-Zarr "
            "array has no usable spatial scale in coordinateTransformations and "
            "the fit config does not define voxel_size."
        )
    try:
        resolve_grid_scale(ndim, voxel_size=selected, output_space="real")
        values = (
            [float(selected)] * ndim
            if isinstance(selected, (int, float))
            else [float(value) for value in selected]
        )
    except (TypeError, ValueError) as exc:
        raise typer.BadParameter(f"--physical voxel size is invalid: {exc}") from exc
    return values, configured is None


def _restore_declared_voxel_size(
    input_path: Path,
    array_key: Optional[str],
    axes_list: Optional[List[str]],
    fit: FitConfig,
    ome_info: "OMEZarrInfo",
) -> None:
    """Recover NGFF spacing when an explicit axes view omits scale metadata."""
    if not fit.physical or axes_list is None or ome_info.voxel_size is not None:
        return
    import zarr

    from luxar._zarr_compat import close, open_store
    from luxar.io.ome_zarr import (
        _owner_ngff_attrs,
        _relative_key,
        _usable_ngff_for_selection,
        discover_ome_zarr_shape,
    )
    from luxar.io.volume import _select_zarr_array

    store = zarr.open(store=open_store(input_path, mode="r"), mode="r")
    try:
        array, key_path, owner, declares = _select_zarr_array(
            store, input_path, array_key
        )
        root_attrs = dict(getattr(store, "attrs", {}))
        owner_ngff = None
        owner_key = None
        if owner is not None and owner is not store:
            owner_attrs = dict(owner.attrs)
            owner_key = _relative_key(owner, key_path)
            owner_ngff, _ = _owner_ngff_attrs(owner, array, owner_attrs, declares)
        multiscales, _, _ = _usable_ngff_for_selection(
            root_attrs, owner_ngff, owner_key, key_path or None, len(array.shape)
        )
    finally:
        close(store)
    if multiscales is None:
        return

    declared_info = discover_ome_zarr_shape(input_path, array_key=array_key)
    if declared_info.spatial_indices == ome_info.spatial_indices:
        ome_info.voxel_size = declared_info.voxel_size


def _configure_physical_fit_args(
    fit_args: dict[str, str],
    fit: FitConfig,
    ome_info: "OMEZarrInfo",
    axes_list: Optional[List[str]],
    ndim: int,
) -> Optional[List[float]]:
    """Resolve physical spacing and add the worker-only override when needed."""
    voxel_size, inject = _physical_voxel_size(fit, ome_info, axes_list, ndim)
    if inject:
        assert voxel_size is not None
        fit_args["voxel-size"] = ",".join(str(value) for value in voxel_size)
    return voxel_size


def _configure_content_physical(
    fit_args: dict[str, str], fit: FitConfig, mode: str
) -> None:
    """Mark content workers only when physical output was explicitly requested."""
    if fit.physical and mode == "content":
        fit_args["physical-coordinates"] = ""


def _batch_grid_scale(
    fit: FitConfig,
    mode: str,
    physical_voxel_size: Optional[List[float]],
    ndim: int,
) -> Optional[List[float]]:
    """Resolve the coordinate frame recorded on the batch manifest."""
    if physical_voxel_size is not None:
        return physical_voxel_size
    return resolve_uniform_grid_scale(fit, ndim) if mode != "content" else None


def resolve_merge_recipe_args(
    merge: MergeConfig,
    *,
    merged_ndim: int = 4,
    merged_has_colors: bool = False,
    slice_count: int = 1,
    part_count: int = 1,
    resolve_target_ms: bool = True,
) -> dict:
    """Validate the per-part merge recipe + knobs into the manifest dict.

    Fail-fast (before any expensive planning / fitting): rejects an unsupported
    recipe, cross-recipe knobs, and merge knobs given WITHOUT a ``--merge-recipe``
    (previously silently dropped), and validates method/breakpoint spellings —
    mirroring ``fit --recipe`` and ``gsplat lod``. The streaming trio
    (``--merge-target-ms``/``--merge-bandwidth-mbps``/``--merge-bytes-per-splat``)
    is resolved HERE into a concrete ``breakpoints="stream:<c>"`` string (analytic
    bytes/splat for ``merged_ndim`` — no gsplat store exists yet at plan time), so
    the manifest schema is unchanged. ``merged_ndim`` defaults to 4 (3 spatial +
    the stacked-timepoint axis, the whole-timelapse norm); pass the value the
    merge itself will compute — ``len(spatial_shape) + (n_timepoints > 1)`` — so
    plan-time and merge-time size the SAME ladder for the same data.
    ``merged_has_colors`` marks a merge that will write per-splat colors (a
    multi-channel merge with channel colors). ``slice_count`` is the number of
    hidden time coordinates and ``part_count`` the spatial parts that render
    together; the stored per-part ``stream:<c>`` is scaled by their ratio.
    ``resolve_target_ms=False`` performs the same validation without resolving
    the size-dependent stream string, for the pre-discovery fail-fast pass.
    Returns ``{}`` when no recipe and no knobs are requested. Raises
    :class:`typer.BadParameter` on any problem.
    """
    if merge.recipe is None:
        # A merge knob without a recipe is a silent no-op — reject it loudly so
        # the user adds --merge-recipe (mirrors `batch-fit merge`'s runtime check).
        orphaned = [
            flag
            for flag, val in {
                "--merge-n-lods": merge.n_lods,
                "--merge-add-method": merge.additive_method,
                "--merge-breakpoints": merge.breakpoints,
                "--merge-target-ms": merge.target_ms,
                "--merge-bandwidth-mbps": merge.bandwidth_mbps,
                "--merge-bytes-per-splat": merge.bytes_per_splat,
                "--merge-compression-factor": merge.compression_factor,
                "--merge-levels": merge.levels,
                "--merge-subst-method": merge.substitutive_method,
                "--merge-refine": merge.refine,
                "--merge-refine-iters": merge.refine_iters,
                "--merge-coarsen-dims": merge.coarsen_dims,
            }.items()
            if val is not None
        ]
        if orphaned:
            raise typer.BadParameter(
                f"option(s) {', '.join(sorted(orphaned))} require a "
                f"--merge-recipe but none was given; pass --merge-recipe "
                f"stream|levels (without one the merge writes bare-leaf "
                f"parts, so these knobs would be ignored)."
            )
        return {}

    from luxar.gsplats.lod.recipes import LEGACY_RECIPE_NAMES, PER_PART_RECIPES

    if merge.recipe in LEGACY_RECIPE_NAMES:
        raise typer.BadParameter(
            f"recipe {merge.recipe!r} was renamed to "
            f"{LEGACY_RECIPE_NAMES[merge.recipe]!r}; use --merge-recipe "
            f"{LEGACY_RECIPE_NAMES[merge.recipe]}."
        )
    if merge.recipe not in PER_PART_RECIPES:
        raise typer.BadParameter(
            f"--merge-recipe {merge.recipe!r} is not supported; choose from "
            f"{', '.join(sorted(PER_PART_RECIPES))} (the composed recipes "
            "re-partition their input, but each tile is already one part)."
        )

    additive_only = {
        "--merge-n-lods": merge.n_lods,
        "--merge-add-method": merge.additive_method,
        "--merge-breakpoints": merge.breakpoints,
        "--merge-target-ms": merge.target_ms,
        "--merge-bandwidth-mbps": merge.bandwidth_mbps,
        "--merge-bytes-per-splat": merge.bytes_per_splat,
    }
    substitutive_only = {
        "--merge-compression-factor": merge.compression_factor,
        "--merge-levels": merge.levels,
        "--merge-subst-method": merge.substitutive_method,
        "--merge-refine": merge.refine,
        "--merge-refine-iters": merge.refine_iters,
        "--merge-coarsen-dims": merge.coarsen_dims,
    }
    irrelevant = substitutive_only if merge.recipe == "stream" else additive_only
    provided = [flag for flag, val in irrelevant.items() if val is not None]
    if provided:
        other = "levels" if merge.recipe == "stream" else "stream"
        raise typer.BadParameter(
            f"option(s) {', '.join(provided)} are not used by "
            f"--merge-recipe {merge.recipe} (they configure "
            f"--merge-recipe {other}). Remove them or switch recipe."
        )

    # Streaming trio → a concrete stream:<c> breakpoints string, resolved at
    # plan time (the stored string round-trips the manifest untouched).
    from luxar.cli.gsplat_ops.recipe_shared import validate_streaming_knobs

    validate_streaming_knobs(
        merge.target_ms,
        merge.bandwidth_mbps,
        merge.bytes_per_splat,
        merge.breakpoints,
        prefix="--merge-",
    )
    eff_breakpoints = merge.breakpoints
    if merge.target_ms is not None and resolve_target_ms:
        from luxar.cli.gsplat_ops.recipe_shared import (
            estimate_bytes_per_splat,
            resolve_streaming_breakpoints,
        )

        eff_breakpoints = resolve_streaming_breakpoints(
            merge.target_ms,
            merge.bandwidth_mbps,
            merge.bytes_per_splat,
            analytic_bps=estimate_bytes_per_splat(
                merged_ndim, has_colors=merged_has_colors
            ),
            slice_count=slice_count,
            part_count=part_count,
        )

    args: dict = {}
    if merge.n_lods is not None:
        args["n-lods"] = str(merge.n_lods)
    if merge.additive_method is not None:
        from luxar.cli.gsplat_ops.recipe_shared import VALID_ADDITIVE_METHODS

        am_norm = merge.additive_method.strip().replace("-", "_")
        if am_norm not in VALID_ADDITIVE_METHODS:
            raise typer.BadParameter(
                f"--merge-add-method must be one of "
                f"{list(VALID_ADDITIVE_METHODS)}; got {merge.additive_method!r}"
            )
        args["add-method"] = am_norm
    if eff_breakpoints is not None:
        from luxar.cli.gsplat_ops.recipe_shared import parse_lod_breakpoints

        parse_lod_breakpoints(eff_breakpoints)
        args["breakpoints"] = eff_breakpoints
    if merge.compression_factor is not None:
        args["compression-factor"] = str(merge.compression_factor)
    if merge.levels is not None:
        args["levels"] = str(merge.levels)
    _record_merge_refine(args, merge)
    if merge.substitutive_method is not None:
        from luxar.cli.gsplat_ops.recipe_shared import VALID_SUBSTITUTIVE_METHODS

        sm_norm = merge.substitutive_method.strip().replace("-", "_")
        if sm_norm not in VALID_SUBSTITUTIVE_METHODS:
            raise typer.BadParameter(
                f"--merge-subst-method must be one of "
                f"{list(VALID_SUBSTITUTIVE_METHODS)}; "
                f"got {merge.substitutive_method!r}"
            )
        args["subst-method"] = sm_norm
    if merge.coarsen_dims is not None:
        args["coarsen-dims"] = merge.coarsen_dims
    return args


def _assemble_fit_args(
    fit: FitConfig, denoise: DenoiseConfig
) -> Tuple[dict, Optional[str], Optional[str]]:
    """Build the ``fit_args`` dict + denoise mode from the fit/denoise configs.

    Returns ``(fit_args, denoise_mode, denoised_zarr_path)``. ``denoise_mode`` is
    set later against ``output_dir`` by the caller for preprocess mode; here it is
    resolved to ``'preprocess'`` / ``'on-the-fly'`` / ``None``.
    """
    fit_args: dict = {}
    if fit.seeds:
        fit_args["seeds"] = fit.seeds
    if fit.iters is not None:
        fit_args["iters"] = str(fit.iters)
    if fit.config:
        # Resolved like every other path on the manifest (input_path, output_dir,
        # plan_path, denoised_zarr_path): this string is handed to workers that
        # run from a Slurm job's own working directory, so a relative spelling
        # would resolve against the wrong CWD (or an unrelated same-named file).
        fit_args["config"] = str(fit.config.resolve())
    # NOTE: no "floor" here on purpose — `_resolve_and_record_floor` is the ONE
    # writer of that key (it forwards the resolved LEVEL, not the spec, #1174).
    if fit.progressive:
        fit_args["progressive"] = ""  # boolean flag, no value
    if fit.splats_per_pass is not None:
        fit_args["splats-per-pass"] = str(fit.splats_per_pass)
    if fit.psnr_patience is not None:
        fit_args["psnr-patience"] = str(fit.psnr_patience)
    if fit.max_passes is not None:
        fit_args["max-passes"] = str(fit.max_passes)
    if fit.cull_retention is not None:
        fit_args["cull-retention"] = str(fit.cull_retention)

    denoise_mode: Optional[str] = None
    if denoise.denoise:
        if denoise.preprocess is True:
            denoise_mode = "preprocess"
        else:
            # Default (and explicit --no-preprocess): denoise per-tile in each fit.
            denoise_mode = "on-the-fly"
        if denoise_mode == "on-the-fly":
            fit_args["denoise"] = ""
            if denoise.denoise_2d:
                fit_args["denoise-2d"] = ""
            if denoise.patch_size != 3:
                fit_args["denoise-patch-size"] = str(denoise.patch_size)
            if denoise.search_distance != 5:
                fit_args["denoise-search-distance"] = str(denoise.search_distance)
            if denoise.backend != "auto":
                fit_args["denoise-backend"] = denoise.backend
            # --denoise-h is injected at runtime from the calibrated h values.
    return fit_args, denoise_mode, None


def _validate_content_fit_flags(tiling: str, fit_args: dict) -> None:
    """Reject fit flags that content workers would silently ignore."""
    if tiling != "content":
        return
    unsupported = [
        flag
        for flag in CONTENT_UNSUPPORTED_FIT_FLAGS
        if flag.removeprefix("--") in fit_args
    ]
    if not unsupported:
        return
    advice = "Use --tiling uniform"
    if "--denoise" in unsupported:
        advice += ", or use --preprocess with batch-fit submit"
    raise typer.BadParameter(
        f"--tiling content with {', '.join(unsupported)} is not supported: "
        f"content workers ignore these options. {advice}."
    )


def _announce_seed_split(mode: str, fit_args: dict, n_tiles: int) -> None:
    """Announce how an integer ``--seeds`` budget divides across a task's tiles.

    Emitted ONCE at plan time as a lower bound: planning knows the geometric
    tile count but does not hold the task's array to count non-empty tiles. Both
    ``batch-fit run`` and ``batch-fit submit --dry-run`` therefore show what is
    guaranteed, while each ``--tile k/M`` task resolves the exact divisor. Task
    output does not reach this console: the local pool captures it and Slurm
    sends it to a log. Content plans ignore ``--seeds`` because per-box budgets
    come from the density model.
    """
    if mode == "content" or not fit_args.get("seeds"):
        return
    from luxar.cli.gsplat_config import parse_seeds
    from luxar.cli.gsplat_ops.fitting.fit_utils import announce_seed_split_lower_bound

    # batch does not parse --seeds itself (it forwards the string verbatim, and
    # each task's own `fit` validates it), so guard: a malformed value must fail
    # in the task as it always has, not break planning here over a notice.
    try:
        announce_seed_split_lower_bound(parse_seeds(fit_args["seeds"]), n_tiles)
    except ValueError:
        pass


def _worker_spatial_shape(
    spatial_shape: Tuple[int, ...], axes_list: Optional[List[str]]
) -> Tuple[int, ...]:
    """Return the spatial shape the task's volume loader will expose."""
    if axes_list is not None:
        return spatial_shape
    return tuple(size for size in spatial_shape if size != 1)


def _validate_content_spatial_shape(
    mode: str,
    spatial: Tuple[int, ...],
    discovered_spatial: Tuple[int, ...],
    axes_list: Optional[List[str]],
) -> None:
    """Reject content scans whose worker-visible volume is not 3-D."""
    if mode != "content" or len(spatial) == 3:
        return
    if axes_list is None and spatial != discovered_spatial:
        reason = (
            f"the store's spatial axes squeeze to {spatial} because the "
            "positional volume loader drops singleton dimensions"
        )
        alternative = "Use --tiling uniform, or pass --axes to keep the axis."
    elif axes_list is None:
        reason = f"the store's spatial shape {spatial} is already {len(spatial)}-D"
        alternative = "Use --tiling uniform, or provide a 3-D spatial array."
    else:
        reason = f"--axes resolves the store's spatial shape to {spatial}"
        alternative = "Use --tiling uniform, or provide a 3-D spatial array."
    raise typer.BadParameter(
        f"--tiling content requires exactly 3 spatial dimensions, but {reason}. "
        f"{alternative}"
    )


def _validate_worker_axes(axes_list: Optional[List[str]]) -> None:
    """Reject axis specs the per-task volume loader cannot apply."""
    if axes_list is None:
        return

    from luxar.io.volume import _axis_kind

    labels = [str(label).strip().lower() for label in axes_list]
    try:
        kinds = [_axis_kind(label) for label in labels]
    except ValueError as exc:
        raise typer.BadParameter(str(exc)) from exc
    if sum(kind == "t" for kind in kinds) > 1:
        raise typer.BadParameter(
            f"--axes {','.join(labels)!r} names more than one time axis; one "
            "--timepoint cannot index them independently."
        )


def _validate_merge_recipe_before_planning(
    merge: MergeConfig, merge_recipe_args: Optional[dict]
) -> None:
    """Run size-independent merge validation before discovery side effects."""
    if merge_recipe_args is None:
        resolve_merge_recipe_args(merge, resolve_target_ms=False)


def plan_batch(
    *,
    input_path: Path,
    output_dir: Path,
    tiling: str,
    tile_size: Optional[int],
    tile_overlap: int,
    axes_list: Optional[List[str]],
    array_key: Optional[str],
    timepoints_slice: Optional[str],
    channels_slice: Optional[str],
    fit: FitConfig,
    denoise: DenoiseConfig,
    content: ContentKnobs,
    merge: MergeConfig,
    merge_recipe_args: Optional[dict] = None,
    max_shape: Optional[List[int]] = None,
    throughput_table: Optional[list] = None,
    resolved_gpu: str = "unknown",
) -> PlanResult:
    """Discover, decompose, and build the manifest + job list (no execution).

    The scheduler-agnostic half of ``batch-fit``: stages shared by ``submit``
    (Slurm) and ``run`` (local). Slurm-only manifest fields (partition, packing,
    preemptible, …) are left at defaults for the caller to populate.

    Parameters
    ----------
    merge_recipe_args
        Pre-resolved manifest dict (tests). Default ``None`` resolves ``merge``
        via :func:`resolve_merge_recipe_args` after decomposition, once the
        planned part count is known. Size-independent validation runs before
        discovery; final sizing uses the true merged ndim, slice count, and
        planned part count, matching what ``batch-fit merge`` computes.
    max_shape
        GPU-profile "largest shape that fits" (uniform auto tile-size). ``None``
        in content mode, or locally without a benchmark profile — then an explicit
        ``tile_size`` is required for a multi-tile uniform fit.
    throughput_table
        GPU-profile throughput (for the wall-time estimate); ``None`` -> a flat
        default (irrelevant locally).
    resolved_gpu
        GPU name recorded in the manifest.
    """
    from luxar.cli.gsplat_config import (
        PRESETS,
        decode_flat_channel_index,
        discover_ome_zarr_shape,
    )
    from luxar.gsplats.batch.time_estimate import estimate_tile_wall_seconds
    from luxar.gsplats.tiling import compute_tile_specs

    # batch-fit fans a whole nD dataset across its axes — it needs a chunked,
    # randomly-addressable OME-Zarr store (so each task reads only its tile/box,
    # and a huge movie is never fully materialized). A flat .npy/.tiff/.h5 would
    # force a full in-RAM load per task. Fail fast with a clear pointer.
    if input_path.suffix.lower() in (".npy", ".npz", ".tif", ".tiff", ".h5", ".hdf5"):
        raise typer.BadParameter(
            f"batch-fit needs an OME-Zarr input (.zarr or .zarr.zip); got "
            f"'{input_path.name}'. Convert it to zarr first, or use `gsplat fit` "
            f"for a single {input_path.suffix.lower()} volume."
        )

    fit_args, denoise_mode, _ = _assemble_fit_args(fit, denoise)
    _validate_content_fit_flags(tiling, fit_args)
    _validate_merge_recipe_before_planning(merge, merge_recipe_args)
    denoised_zarr_path = None
    if denoise_mode == "preprocess":
        denoised_zarr_path = str(output_dir.resolve() / "denoised.zarr")

    # 2. Discover dataset shape + apply --timepoints/--channels slicing.
    _validate_worker_axes(axes_list)
    with asection("Discovering dataset shape"):
        ome_info = discover_ome_zarr_shape(
            input_path, axes_override=axes_list, array_key=array_key
        )
        _restore_declared_voxel_size(input_path, array_key, axes_list, fit, ome_info)
        n_t_full = ome_info.n_timepoints
        n_c_full = ome_info.n_channels
        # Positional workers call load_volume without --axes, whose legacy
        # post-processing squeezes every size-1 dimension. Plan the exact
        # spatial volume those workers fit so tile coordinates and the merge's
        # stacked-axis index use the same dimensionality.
        spatial = _worker_spatial_shape(ome_info.spatial_shape, axes_list)
        aprint(f"Axes: {ome_info.axes}")
        aprint(f"Shape: {ome_info.shape}")
        aprint(f"T={n_t_full}, C={n_c_full}, spatial={'x'.join(map(str, spatial))}")

        # Without --axes the manifest carries no axes, so the workers slice
        # POSITIONALLY. Refuse a discovered layout they would slice differently
        # from what this plan assumes, instead of emitting duplicate tiles. The
        # two flags mirror `build_task_fit_argv`'s emission rule, since the 4D
        # branch of the loader depends on which of them is present.
        _refuse_layout_the_workers_cannot_slice(
            axes_list,
            ome_info,
            emits_channel=n_c_full > 1 or bool(channels_slice),
            emits_timepoint=n_t_full > 1 or bool(timepoints_slice),
        )

        physical_voxel_size = _configure_physical_fit_args(
            fit_args, fit, ome_info, axes_list, len(spatial)
        )

        t_indices = (
            _parse_slice(timepoints_slice, n_t_full)
            if timepoints_slice
            else list(range(n_t_full))
        )
        c_indices = (
            _parse_slice(channels_slice, n_c_full)
            if channels_slice
            else list(range(n_c_full))
        )
        if not t_indices:
            raise ValueError("--timepoints selected no timepoints")
        if not c_indices:
            raise ValueError("--channels selected no channels")
        bad_t = [i for i in t_indices if i < 0 or i >= n_t_full]
        bad_c = [i for i in c_indices if i < 0 or i >= n_c_full]
        if bad_t:
            raise ValueError(
                f"--timepoints selected out-of-range indices {bad_t}; "
                f"valid range is 0..{n_t_full - 1}"
            )
        if bad_c:
            raise ValueError(
                f"--channels selected out-of-range flat channel indices {bad_c}; "
                f"valid range is 0..{n_c_full - 1}"
            )
        n_t = len(t_indices)
        n_c = len(c_indices)
        if timepoints_slice or channels_slice:
            aprint(f"Sliced: T={n_t} (of {n_t_full}), C={n_c} (of {n_c_full})")

    _validate_merge_refine_source(merge, axes_list, n_t, n_c)

    # 3. Decompose the spatial volume into the slots fanned across (t, c).
    mode = "content" if tiling == "content" else "uniform"
    _configure_content_physical(fit_args, fit, mode)
    _validate_content_spatial_shape(mode, spatial, ome_info.spatial_shape, axes_list)
    content_plan = None
    plan_path_str: Optional[str] = None
    total_voxels = math.prod(spatial)

    rep_c = c_indices[0]

    # A `downscale:` in the forwarded fit config is VALIDATED here, ahead of
    # everything expensive, because the value's well-formedness needs only the
    # merged config and len(spatial). Doing it alongside the decomposition below
    # made a malformed value in a content run surface only after the box plan had
    # been scanned and plan.json written. Whether the decomposition can complete
    # with a well-formed value is a separate, later question (see
    # `refuse_downscale_grid_mismatch`).
    downscale_factors = validate_config_downscale(fit, len(spatial))

    # The frame the tasks will emit in, resolved once and recorded on the
    # manifest for the merge (#1587). A content merge takes its split planes
    # from the shared plan's boxes; physical output scales that tree separately.
    # Ahead of the floor resolution below because it reads only the fit config:
    # a frame the merge cannot reconcile fails before any voxel is sampled.
    grid_scale = _batch_grid_scale(fit, mode, physical_voxel_size, len(spatial))
    _validate_merge_refine_frame(merge, grid_scale)

    # 3a. Background floor: ONE level for the whole timelapse, resolved here and
    # recorded in the manifest, so no task re-estimates its own (#1174). The
    # basis is a bounded set of evenly spaced (t, c) slices spanning the store's
    # FULL extent, reduced by MINIMUM — a level above some slice's maximum would
    # clip that whole sub-volume to zero and drop it silently from the merge, so
    # the level is made a lower bound on every SAMPLED slice's pedestal (bounded
    # sampling cannot bound an unsampled one). Raw resolution must not depend on
    # either selection; on-the-fly denoised resolution spans the full time extent
    # but follows the selected channels because NLM h is calibrated per channel
    # (see resolve_batch_floor).
    # Deliberately not the content plan's max-projection either, whose per-voxel
    # maximum biases the background mode upward relative to any single slice.
    from luxar.cli.gsplat_ops.fitting.fit_utils import (
        floor_spec_needs_volume,
        validate_floor_spec,
    )

    floor_spec = effective_floor_spec(fit)
    validate_floor_spec(floor_spec)
    floor_deferred = _should_defer_floor_resolution(
        mode, denoise, floor_spec, denoise_mode
    )
    configured_norm_range = effective_norm_range(fit)
    resolve_sampled_norm_range = configured_norm_range is None and not denoise.denoise
    sampled_slices = (
        _sample_batch_slices(
            input_path,
            n_timepoints=n_t_full,
            n_channels=n_c_full,
            array_key=array_key,
            axes=",".join(axes_list) if axes_list else None,
            axes_labels=list(ome_info.axes),
            channel_shape=tuple(ome_info.channel_shape),
            spatial_shape=tuple(spatial),
        )
        if (not floor_deferred and floor_spec_needs_volume(floor_spec))
        or resolve_sampled_norm_range
        else []
    )
    floor_level, recorded_floor_level = _resolve_planned_floor(
        deferred=floor_deferred,
        input_path=input_path,
        fit=fit,
        fit_args=fit_args,
        n_timepoints=n_t_full,
        n_channels=n_c_full,
        array_key=array_key,
        axes=",".join(axes_list) if axes_list else None,
        # The DISCOVERED labels pin the representative slice even without --axes.
        axes_labels=list(ome_info.axes),
        channel_shape=tuple(ome_info.channel_shape),
        spatial_shape=tuple(spatial),
        sampled_slices=sampled_slices,
    )
    norm_range = _resolve_planned_norm_range(
        input_path=input_path,
        configured=configured_norm_range,
        denoise=denoise.denoise,
        norm_percentile=effective_norm_percentile(fit),
        n_timepoints=n_t_full,
        n_channels=n_c_full,
        array_key=array_key,
        axes=",".join(axes_list) if axes_list else None,
        axes_labels=list(ome_info.axes),
        channel_shape=tuple(ome_info.channel_shape),
        spatial_shape=tuple(spatial),
        sampled_slices=sampled_slices,
    )
    _record_batch_norm_range(fit_args, norm_range)

    if mode == "content":
        import numpy as _np

        from luxar.cli.gsplat_ops.planner import _resolve_density
        from luxar.gsplats.planner import plan_volume
        from luxar.gsplats.planner.fit_planned_parallel import max_padded_box_voxels

        plan_t_samples = _select_plan_timepoints(
            t_indices, content.plan_timepoint, content.plan_samples
        )
        scan_desc = (
            f"t={plan_t_samples[0]}"
            if len(plan_t_samples) == 1
            else f"max-proj of {len(plan_t_samples)} timepoints"
        )
        with asection(f"Content plan (scan {scan_desc}, c={rep_c})"):
            rep_vol = _load_scan_volume(
                input_path,
                plan_t_samples,
                channel=rep_c,
                array_key=array_key,
                axes=",".join(axes_list) if axes_list else None,
                # The DISCOVERED labels, so each sampled timepoint is pinned by
                # label even with no --axes (the positional 4D heuristic reads a
                # (T, Z, Y, X) store as CZYX and would scan t=0 every time).
                axes_labels=list(ome_info.axes),
                channel_shape=tuple(ome_info.channel_shape),
                spatial_shape=tuple(spatial),
            )
            density = _resolve_density(
                content.cal,
                content.k_star_ref,
                content.n_features_ref,
                content.saturation_exponent,
                content.saturation_cap,
                content.feature_metric,
                content.feature_threshold,
            )
            if (
                content.feature_metric is not None
                and content.cal is not None
                and content.feature_metric != density.feature_method
            ):
                aprint(
                    f"⚠ --feature-metric '{content.feature_metric}' differs from the "
                    f"calibrated density.feature_method '{density.feature_method}' — "
                    "per-box budgets will be mis-scaled. Use matching metrics."
                )
            # Scan the floor-suppressed volume the boxes will fit: cal records
            # `density.feature_threshold` on floor-subtracted data, and every box
            # subtracts `floor_level`. Using that same level here (instead of
            # re-resolving against the max-projection, which biases the mode
            # upward) puts plan and fits on one basis.
            if floor_level is not None:
                rep_vol = _np.clip(rep_vol.astype(_np.float32) - floor_level, 0.0, None)
            content_plan = plan_volume(
                rep_vol,
                density,
                feature_method=(content.feature_metric or density.feature_method),
                cell=content.cell,
                target_features=content.target_features,
                min_leaf=content.min_leaf,
                max_leaf=content.max_leaf,
                overlap=tile_overlap,
            )
            kept_indices = [i for i, b in enumerate(content_plan.boxes) if b.budget > 0]
            kept_boxes = [content_plan.boxes[i] for i in kept_indices]
            if not kept_boxes:
                raise typer.BadParameter("content plan has no boxes with budget > 0")
            # Dropping zero-budget boxes RE-INDEXES the plan, and the split-plane
            # tree's leaf labels index the pre-drop list — renumber them in the
            # same step or every downstream consumer (each array task's
            # `--plan-box k`, and the merge's part order) reads the wrong box.
            content_plan = dataclasses.replace(
                content_plan,
                boxes=kept_boxes,
                bsp_tree=prune_serialized_bsp_tree(content_plan.bsp_tree, kept_indices),
            )
            output_dir.mkdir(parents=True, exist_ok=True)
            plan_path_obj = output_dir.resolve() / "plan.json"
            content_plan.to_json(plan_path_obj)
            plan_path_str = str(plan_path_obj)
            med, mx = content_plan.overlap_fraction()
            aprint(
                f"Plan: {content_plan.n_boxes} boxes, total budget "
                f"{content_plan.total_budget:,} splats, overlap median "
                f"{med:.0%} / max {mx:.0%} → {plan_path_obj}"
            )
        n_tiles = content_plan.n_boxes
        needs_tiling = n_tiles > 1
        tile_size_resolved = 0  # sentinel; content tasks omit --tile-size
        tile_voxels = max_padded_box_voxels(content_plan)
    else:
        auto_tile = tile_size is None
        if auto_tile:
            if max_shape is None:
                raise typer.BadParameter(
                    "uniform auto tile-size needs a GPU benchmark profile; "
                    "pass --tile-size explicitly (or use --tiling content)."
                )
            max_safe_voxels = math.prod(max_shape) if max_shape else 256**3
            if total_voxels <= max_safe_voxels:
                tile_size = max(spatial) + tile_overlap
            else:
                tile_edge = int(max_safe_voxels ** (1.0 / len(spatial)))
                tile_size = min(tile_edge, max(spatial))
        assert tile_size is not None
        tile_size_resolved = tile_size
        specs = compute_tile_specs(spatial, tile_size, tile_overlap)
        n_tiles = len(specs)
        needs_tiling = n_tiles > 1
        tile_voxels = tile_size ** len(spatial) if needs_tiling else total_voxels

    # Resolve after decomposition: each partition part expands the stored
    # stream:<c> independently, while only one hidden time slice is drawn.
    if merge_recipe_args is None:
        merge_recipe_args = resolve_merge_recipe_args(
            merge,
            merged_ndim=len(spatial) + (1 if n_t > 1 else 0),
            merged_has_colors=bool(merge.channel_colors) and n_c > 1,
            slice_count=max(1, n_t),
            part_count=max(1, n_tiles),
        )

    total_tasks = n_t * n_c * n_tiles

    # The already-validated `downscale:` is now checked against the decomposition
    # it has to agree with: a uniform worker re-tiles its own decimated volume, so
    # the two grids are compared directly and only a genuine disagreement is
    # refused. Still ahead of the manifest and of every task, so a fatal one costs
    # no submission.
    refuse_downscale_grid_mismatch(
        downscale_factors,
        mode=mode,
        spatial=tuple(spatial),
        tile_size=tile_size_resolved,
        tile_overlap=tile_overlap,
        total_tasks=total_tasks,
    )

    _announce_seed_split(mode, fit_args, n_tiles)

    preset_config = PRESETS.get(fit.preset, PRESETS["standard"])
    n_iters = fit.iters if fit.iters is not None else preset_config.get("n_iters", 3000)
    if throughput_table:
        est_seconds = estimate_tile_wall_seconds(tile_voxels, n_iters, throughput_table)
    else:
        est_seconds = 600.0

    colors_list = (
        [c.strip() for c in merge.channel_colors.split(",")]
        if merge.channel_colors
        else None
    )
    # NOTE: the uniform+per-part-LOD warning fires inside merge_batch_results
    # (the library boundary), so it is NOT emitted here — every caller (Slurm
    # merge, local run, direct API) gets it exactly once at merge time.

    manifest = BatchManifest(
        version=1,
        created=datetime.datetime.now(datetime.timezone.utc).isoformat(),
        input_path=str(input_path.resolve()),
        output_dir=str(output_dir.resolve()),
        array_key=array_key,
        axes=",".join(axes_list) if axes_list else None,
        n_timepoints=n_t,
        n_channels=n_c,
        channel_axes=ome_info.channel_axes,
        channel_shape=ome_info.channel_shape,
        spatial_shape=spatial,
        mode=mode,
        tile_size=tile_size_resolved,
        tile_overlap=tile_overlap,
        n_tiles=n_tiles,
        plan_path=plan_path_str,
        total_tasks=total_tasks,
        preset=fit.preset,
        fit_args=fit_args,
        floor_level=recorded_floor_level,
        floor_spec=floor_spec if floor_deferred else None,
        floor_deferred=floor_deferred,
        norm_range=norm_range,
        grid_scale=grid_scale,
        gpu_name=resolved_gpu,
        estimated_seconds_per_task=est_seconds,
        timepoint_indices=t_indices if timepoints_slice else None,
        channel_indices=c_indices if channels_slice else None,
        channel_colors=colors_list,
        merge_recipe=merge.recipe,
        merge_recipe_args=merge_recipe_args,
        denoise=denoise.denoise,
        denoise_2d=denoise.denoise_2d,
        denoise_h=denoise.denoise_h,
        denoise_patch_size=denoise.patch_size,
        denoise_search_distance=denoise.search_distance,
        denoise_backend=denoise.backend,
        denoise_mode=denoise_mode,
        denoised_zarr_path=denoised_zarr_path,
        calibration_samples=denoise.calibration_samples,
    )

    # Job list: store REAL dataset indices in filenames so status/merge/scripts
    # agree when --timepoints/--channels select non-contiguous values.
    jobs: list[BatchJob] = []
    t_width_base = max(t_indices) + 1
    c_width_base = max(c_indices) + 1
    for task_id in range(total_tasks):
        t_seq = task_id // (n_c * n_tiles)
        r = task_id % (n_c * n_tiles)
        c_seq = r // n_tiles
        k = r % n_tiles
        t_real = t_indices[t_seq]
        c_real = c_indices[c_seq]
        jobs.append(
            BatchJob(
                task_id=task_id,
                timepoint=t_real,
                channel=c_real,
                tile_index=k,
                output_filename=output_filename(
                    t_real,
                    c_real,
                    k,
                    t_width_base,
                    c_width_base,
                    n_tiles,
                    label="box" if mode == "content" else "tile",
                ),
                estimated_wall_seconds=est_seconds,
                channel_coords=decode_flat_channel_index(
                    c_real, ome_info.channel_shape
                ),
            )
        )
    manifest.jobs = jobs

    return PlanResult(
        manifest=manifest,
        content_plan=content_plan,
        tile_voxels=tile_voxels,
        needs_tiling=needs_tiling,
    )
