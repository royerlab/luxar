"""Post-batch merge orchestration: tiles -> per-(T,C) -> per-C -> final.

Two output shapes are supported:

* **partition** (default) — a ``kind=partition`` v3.0 file with one part per
  spatial tile. Built **tile-outer, streaming**: each part_K is assembled one
  tile-region at a time (its timepoints stacked + channels handled) and written
  straight to disk, so peak memory is one tile-region — never the whole volume.
  This preserves the spatial structure tiling targets (per-part frustum culling)
  and avoids the latent OOM of concatenating every tile into one flat leaf.
* **flat** (``--flat``) — the historical single-leaf 3-level fan-in
  (per-(T,C) concatenate → stack timepoints → merge channels). Reloads ALL tiles
  into memory; retained only for small scenes / backward parity.

Tiles are Hann-apodized (a partition of unity), so rendering them as separate
additive partition parts sums to the true signal exactly as the flat concat does
— no double-count — hence the partition is the safe default.
"""

from __future__ import annotations

from pathlib import Path
from typing import TYPE_CHECKING, Any, Dict, Iterator, List, Optional, Sequence, Tuple

from arbol import aprint, asection

from luxar.core.group.partition import prune_serialized_bsp_tree
from luxar.gsplats.batch.manifest import (
    BatchManifest,
    floor_erased_slices,
    output_filename,
)

if TYPE_CHECKING:
    from luxar.gsplats.gsplat_data import GSplatData
    from luxar.gsplats.lod.recipes import RecipeParams
    from luxar.gsplats.tree import GSplatNode


def merge_batch_results(
    manifest: BatchManifest,
    output_dir: Path,
    channel_colors: Optional[List[Tuple[float, float, float]]] = None,
    force: bool = False,
    verbose: bool = True,
    flat: bool = False,
    recipe: Optional[str] = None,
    recipe_params: "Optional[RecipeParams]" = None,
) -> Path:
    """Merge a completed batch job into a single ``.gsplats.zarr``.

    Args:
        manifest: Loaded batch manifest.
        output_dir: Batch output directory.
        channel_colors: Optional list of RGB tuples for channel coloring.
        force: Re-merge even if output already exists.
        verbose: Print progress.
        flat: Use the legacy single-leaf 3-level fan-in instead of the default
            streaming spatial partition. ``flat=True`` reloads ALL tiles into
            memory (the OOM the partition path avoids); use only for small scenes.
        recipe: Optional per-part LOD recipe applied to each spatial tile-part as
            it streams (one of :data:`PER_PART_RECIPES`: ``stream`` → the
            ``tiles`` topology, ``levels`` → ``adaptive``).
            ``None`` keeps the historical bare-leaf parts. Mutually exclusive with
            ``flat`` (flat has no parts to give a ladder to).
        recipe_params: Knobs for ``recipe`` (a :class:`RecipeParams`); ignored when
            ``recipe`` is ``None``. ``coarsen_dims`` defaults per part to the
            spatial dims only (the stacked-timepoint axis stays a hard barrier).

    Returns:
        Path to the final merged output.
    """
    if flat and recipe is not None:
        raise ValueError(
            "merge_batch_results: `flat` and `recipe` are mutually exclusive — "
            "the flat path produces a single leaf with no spatial parts to carry "
            "a per-part LOD ladder. Drop --flat to get a per-part LOD partition."
        )
    erased = floor_erased_slices(manifest, Path(output_dir) / "tiles")
    if erased:
        pairs = ", ".join(f"(t={t}, c={c})" for t, c in sorted(erased))
        raise RuntimeError(
            "Background floor suppression erased every spatial tile for "
            f"{pairs}; refusing to merge missing slices. Remove those slices' "
            ".empty markers and re-plan with a lower floor or --floor none."
        )
    if flat and manifest.mode == "content":
        raise ValueError(
            "merge_batch_results: `flat` is not supported for content-mode batches. "
            "Content fitting places boxes once from a representative timepoint, so a "
            "(timepoint, channel) slot can be legitimately empty across every box; "
            "the flat 3-level fan-in assumes a dense (t, c) grid and would crash on "
            "that gap (and reloads every box into memory, defeating content tiling "
            "on the large volumes it targets). Drop --flat to use the default "
            "streaming partition merge, which skips empty slots correctly — then, "
            "if you genuinely need a single flat leaf (e.g. to feed `gsplat lod "
            "--recipe overview`), run `luxar gsplat flatten` on the merged "
            "partition."
        )
    if recipe is not None:
        from luxar.gsplats.lod.recipes import PER_PART_RECIPES, canonical_recipe_name

        # Stored manifests may carry pre-rename spellings (additive/substitutive)
        # — translate silently; only CLI input gets the did-you-mean rejection.
        recipe = canonical_recipe_name(recipe)
        if recipe not in PER_PART_RECIPES:
            raise ValueError(
                f"merge_batch_results: per-part recipe {recipe!r} is not supported; "
                f"choose from {', '.join(PER_PART_RECIPES)}. The composed recipes "
                "(tiles/overview/adaptive) re-partition their input, but each "
                "tile is already one spatial part."
            )
        # Front-door: refine="volume" re-opens the source and crops it per tile,
        # so anything that makes the source unmappable must fail HERE, before any
        # output exists — the deep guard would otherwise fire mid-stream, after
        # parts were already written.
        if recipe_params is not None and recipe_params.refine == "volume":
            _validate_merge_volume_refit(manifest)
        # Per-part LOD on uniform (Hann-apodized) tiles only holds the halo
        # partition-of-unity at the finest level — warn here, the library boundary,
        # so the CLI, the Slurm merge job, and any direct API caller (e.g. the local
        # runner) all get it exactly once.
        from luxar.gsplats.lod.recipes import uniform_per_part_lod_warning

        _w = uniform_per_part_lod_warning(manifest.mode, recipe)
        if _w:
            aprint(f"⚠ {_w}")
    if flat:
        return _merge_flat(manifest, output_dir, channel_colors, force, verbose)
    return _merge_partition(
        manifest, output_dir, channel_colors, force, verbose, recipe, recipe_params
    )


def _tile_indices(manifest: BatchManifest) -> Tuple[List[int], List[int]]:
    """Resolve REAL (t, c) dataset indices used in tile filenames.

    When --timepoints/--channels slicing was used the filenames carry the REAL
    dataset indices (e.g. t0072 not t01), so derive them from the manifest.
    """
    n_t = manifest.n_timepoints
    n_c = manifest.n_channels
    t_indices = (
        manifest.timepoint_indices
        if manifest.timepoint_indices is not None
        else list(range(n_t))
    )
    c_indices = (
        manifest.channel_indices
        if manifest.channel_indices is not None
        else list(range(n_c))
    )
    return t_indices, c_indices


def _tile_path(
    tiles_dir: Path,
    t_real: int,
    c_real: int,
    k: int,
    t_max: int,
    c_max: int,
    n_k: int,
    label: str = "tile",
) -> "Optional[Path]":
    """Resolve a single spatial-slot output path (matches the sbatch naming).

    ``label`` is ``tile`` (uniform) or ``box`` (content) — it MUST match the
    label the fit array wrote, or the lookup misses every output. Returns
    ``None`` when the slot is legitimately empty: an array task that fit 0
    splats writes a sibling ``<path>.empty`` marker instead of a store (the
    same convention the local parallel path uses), so the merge skips it rather
    than treating it as failure. A genuinely missing output (no store, no
    marker = the task never completed) still raises.
    """
    fname = output_filename(t_real, c_real, k, t_max + 1, c_max + 1, n_k, label=label)
    tile_path = tiles_dir / fname
    if tile_path.exists():
        return tile_path
    if Path(f"{tile_path}.empty").exists():
        return None  # ran, legitimately produced 0 splats — skip this slot
    raise FileNotFoundError(
        f"Missing tile output: {tile_path}\n"
        f"Run `luxar gsplat batch-fit status {tiles_dir.parent}` "
        f"to check job status."
    )


def _source_dtype_for_tiles(
    tiles_dir: Path,
    t_indices: List[int],
    c_indices: List[int],
    n_tiles: int,
    label: str,
) -> Optional[str]:
    """Return the first non-empty tile's recorded source dtype, if present."""
    from luxar._zarr_compat import open_group

    for t_real in t_indices:
        for c_real in c_indices:
            for tile_index in range(n_tiles):
                tile_path = _tile_path(
                    tiles_dir,
                    t_real,
                    c_real,
                    tile_index,
                    max(t_indices),
                    max(c_indices),
                    n_tiles,
                    label,
                )
                if tile_path is None:
                    continue
                # Every batch task fits the same selected input array, so source
                # dtype is invariant across slots. On the streaming partition
                # path this also keeps an unfinished first slot failing before
                # any merge output exists.
                root = open_group(tile_path, mode="r")
                source_dtype = (
                    root["fitting"].attrs.get("source_dtype")
                    if "fitting" in root
                    else None
                )
                return source_dtype if isinstance(source_dtype, str) else None
    return None


def _merge_fitting_info(
    part_provenance: List[Dict[str, Any]], source_dtype: Optional[str]
) -> Dict[str, Any]:
    """Build root fitting metadata shared by batch merge writers."""
    fitting_info: Dict[str, Any] = {"part_provenance": part_provenance}
    if source_dtype is not None:
        fitting_info["source_dtype"] = source_dtype
    return fitting_info


def _copy_or_resave_tile(
    source: Path,
    destination: Path,
    dimension_metadata: "Optional[List[Dict[str, Any]]]",
) -> None:
    """Copy a tile byte-for-byte, then stamp optional final-root metadata."""
    import shutil

    if destination.exists():
        shutil.rmtree(destination)
    shutil.copytree(source, destination)
    if dimension_metadata is not None:
        from luxar._zarr_compat import consolidate as zc_consolidate
        from luxar._zarr_compat import open_group as zc_open_group
        from luxar.gsplats.io.save_gsplats import _stamp_content_hash

        root = zc_open_group(destination, mode="r+")
        root_attrs = _dimension_root_attrs(
            dimension_metadata, int(root["centers"].shape[1])
        )
        if root_attrs:
            root.attrs.update(root_attrs)
            _stamp_content_hash(root)
            zc_consolidate(root)


def _dimension_root_attrs(
    metadata: "Optional[List[Dict[str, Any]]]", ndim: int
) -> "Optional[Dict[str, Any]]":
    """Return aligned dimension metadata, or warn and omit a bad descriptor list."""
    if metadata is None:
        return None
    if len(metadata) != ndim:
        aprint(
            "  WARNING: manifest dimension metadata has "
            f"{len(metadata)} entries for {ndim} output columns; omitting it"
        )
        return None
    return {"dimension_metadata": metadata}


def _drop_mismatched_partition_dimension_attrs(
    root_attrs: "Optional[Dict[str, Any]]",
    metadata: "Optional[List[Dict[str, Any]]]",
    ndim: int,
) -> None:
    """Remove deferred partition metadata when a streamed part proves it wrong."""
    if not root_attrs or metadata is None or len(metadata) == ndim:
        return
    root_attrs.clear()
    aprint(
        "  WARNING: manifest dimension metadata does not match "
        f"the partition's {ndim} output columns; omitting it"
    )


# ════════════════════════════════════════════════════════════════════════
# Default: streaming spatial partition (tile-outer, O(1) memory per tile)
# ════════════════════════════════════════════════════════════════════════


def _build_part_for_tile(
    tiles_dir: Path,
    k: int,
    t_indices: List[int],
    c_indices: List[int],
    n_k: int,
    channel_colors: Optional[List[Tuple[float, float, float]]],
    label: str = "tile",
) -> "Optional[GSplatData]":
    """Assemble the full nD leaf-``GSplatData`` for spatial tile-region ``k``.

    Tile-outer reorder of the flat fan-in: within ONE spatial tile, stack that
    tile's timepoints (``combine_as_new_dimension`` -> +1 dim, 3D->4D) and apply
    channel handling (color merge if colors given, else concatenate). Only this
    one tile-region's splats are loaded — never the whole volume. Returns
    ``None`` for an empty/zero-splat tile (the caller skips it).
    """
    from luxar.gsplats.gsplat_data import GSplatData
    from luxar.gsplats.merged_quality import collect_part_provenance

    n_t = len(t_indices)

    # Per-channel: stack this tile's timepoints (Level-2 semantics, but scoped to
    # a single spatial tile so memory stays bounded to one tile-region).
    # ``kept_positions`` records which channel indices survived (a slot can be
    # empty in some channels but not others — common in content mode) so a
    # channel-colors merge subsets the colors to the surviving channels rather
    # than mismatching the fixed-length list.
    per_channel: List[GSplatData] = []
    kept_positions: List[int] = []
    for c_pos, c_real in enumerate(c_indices):
        # Build the timepoint stack, skipping slots that are legitimately empty
        # (`_tile_path` returns None) so a box absent at some timepoints still
        # stacks the timepoints where it has signal — at their real coords.
        tc_data: List[GSplatData] = []
        tc_values: List[float] = []
        for t_real in t_indices:
            tile_path = _tile_path(
                tiles_dir, t_real, c_real, k, max(t_indices), max(c_indices), n_k, label
            )
            if tile_path is None:
                continue
            tc_data.append(GSplatData.load(tile_path, include_stats=True))
            tc_values.append(float(t_real))
        if not tc_data:
            continue  # this (channel, slot) is empty at every timepoint
        if n_t > 1:
            timepoint_provenance = collect_part_provenance(
                tc_data,
                values=tc_values,
                fit_reference=None,
            )
            stacked = GSplatData.combine_as_new_dimension(
                tc_data,
                values=tc_values,
                sigma=0.0,
                part_provenance=timepoint_provenance,
            )
        else:
            stacked = tc_data[0]
        per_channel.append(stacked)
        kept_positions.append(c_pos)

    if not per_channel:
        return None  # empty at every (timepoint, channel) for this slot

    # Across channels (Level-3 semantics, scoped to this tile). Apply colors only
    # for a genuinely multi-channel dataset (`len(c_indices) > 1`) — matching the
    # flat path's `n_c > 1` gate, so a single-channel dataset keeps its fitted
    # colors instead of being tinted. Subset to the channels that actually survived
    # in THIS slot (some may be empty here); merge_with_channel_colors needs a
    # length match. A multi-channel dataset where only one channel survives in this
    # tile still tints that channel (its color), which is correct.
    channel_provenance = (
        collect_part_provenance(
            per_channel,
            values=[float(c_indices[position]) for position in kept_positions],
            fit_reference=None,
        )
        if len(c_indices) > 1
        else None
    )

    if channel_colors and len(c_indices) > 1:
        colors = [channel_colors[i] for i in kept_positions]
        part = GSplatData.merge_with_channel_colors(per_channel, colors)
    elif len(per_channel) == 1:
        part = per_channel[0]
    else:
        part = GSplatData.concatenate(per_channel)

    if channel_provenance is not None:
        part.stats["part_provenance"] = channel_provenance

    if part.n_splats == 0:
        return None
    return part


def _single_part_provenance(part: "GSplatData") -> List[Dict[str, Any]]:
    """Collect an attributable K=1 component record."""
    from luxar.gsplats.merged_quality import collect_part_provenance

    return collect_part_provenance([part], values=[0.0], fit_reference=None)


def _drop_root_quality(part: "GSplatData") -> None:
    """Keep component quality out of a bare merged root's scalar fields."""
    from luxar.gsplats._data.filtering import _CONTENT_SCOPED_STATS_KEYS

    for key in _CONTENT_SCOPED_STATS_KEYS:
        part.stats.pop(key, None)


def volume_refit_source_error(
    axes: "Optional[str]", n_timepoints: int, n_channels: int
) -> "Optional[str]":
    """Why a source cannot serve a per-tile volume re-fit — ``None`` if it can.

    Pure, and deliberately shared by two callers: the PLANNER runs it so a
    mistake costs nothing (the alternative is discovering it after every tile has
    been fitted), and the merge front door runs it again because a manifest can
    predate the check or be written by hand.

    A merged part carries the spatial dims plus, when more than one timepoint was
    selected, ONE stacked axis. Every other source axis has to be PINNED to the
    index this batch fitted (see :func:`_merge_refit_volume`), so what has to hold
    is that each such axis has a single index to pin:

    * the manifest must record ``axes``, since the merged parts put spatial dims
      first with the stacked axis LAST while the source is usually time-FIRST, and
      guessing that mapping wrong sends every re-fit at the wrong axis;
    * several selected channels cannot be pinned to one index — nor mapped onto
      the one stacked axis a merged part carries;
    * a channel index folded over SEVERAL source axes has no single axis to pin,
      and neither does a source with two time axes.
    """
    if not axes:
        return (
            "the source's axis labels are needed to map the stacked axis (merged "
            "parts put spatial dims first and the stacked axis LAST; a source is "
            "usually time-FIRST). This batch was planned without --axes, so "
            "re-plan with it, or merge without the re-fit."
        )
    labels = [a.strip().lower() for a in axes.split(",") if a.strip()]
    # Classify every label through the shared vocabulary, so an unrecognised one
    # is named here rather than surfacing from the loader further down.
    from luxar.io.volume import _axis_kind

    try:
        kinds = [_axis_kind(label, "--axes") for label in labels]
    except ValueError as exc:
        return str(exc)
    if not any(k == "s" for k in kinds):
        return (
            f"--axes {axes!r} names no spatial axis, so there is nothing to crop "
            "a tile from."
        )
    if n_channels and n_channels > 1:
        return (
            f"several channels were selected ({n_channels}): a merged part carries "
            "ONE stacked axis (the timepoints), so a channel cannot also map onto "
            "it — and there is no single channel index to pin the source's channel "
            "axis to. Merge without the re-fit, or fit one channel at a time."
        )
    if sum(1 for k in kinds if k == "c") > 1:
        return (
            f"--axes {axes!r} folds the channel index over more than one axis, so "
            "there is no single index to pin each of them to. Merge without the "
            "re-fit, or point --array-key at an array with one channel axis."
        )
    if sum(1 for k in kinds if k == "t") > 1:
        return f"--axes {axes!r} names more than one time axis."
    if n_timepoints > 1 and not any(k == "t" for k in kinds):
        return (
            f"{n_timepoints} timepoints were stacked but --axes {axes!r} names no "
            "time axis, so the stacked axis has nothing to map onto."
        )
    return None


def volume_refit_frame_error(
    grid_scale: "Optional[Sequence[float]]",
) -> "Optional[str]":
    """Why a mis-framed batch cannot serve a per-tile volume re-fit — ``None`` if it can.

    The coordinate half of :func:`volume_refit_source_error`, and the batch
    counterpart of ``gsplat fit``'s
    :func:`~luxar.cli.gsplat_ops.fitting.fit_utils.reject_rescaled_volume_refit`
    (#1587). A per-part re-fit crops the source to the part's own cell of the
    partition's ``bsp_tree`` and uses that cell as VOXEL INDICES into the
    volume, which only holds while the tile grid and the splats share a frame.
    A recorded ``grid_scale`` says they do not — for a planned run, a
    ``voxel_size`` (with the default ``output_space: real``) in the run's
    ``--config`` moved every task's splats off the grid. A ``downscale:`` there
    is NOT such a cause: every task rescales back to the full-resolution frame
    the planner tiled, so the planner records nothing for it (#1624). A
    hand-written manifest can still state any factor, which is why this stays a
    check on the recorded VALUE rather than on how it came about.

    Both signs fail, differently, and neither is caught downstream. With a
    factor above 1 each crop is too large and the re-fit's own frame heuristic
    turns it into a wasteful no-op; below 1 the crop shrinks to a WRONG region
    that the seed still sits inside, so no guard fires at all and the
    never-worse check judges the re-fit against that same wrong crop.

    Pure, and shared by the planner (so a typo costs nothing) and the merge
    front door (a manifest can predate the check, or be written by hand).
    """
    if not grid_scale:
        return None
    factors = [float(f) for f in grid_scale]
    if all(f == 1.0 for f in factors):
        return None
    return (
        f"this batch's tasks emitted their splats in a frame scaled by "
        f"{factors} relative to the tile grid (on a planned run, from a "
        "voxel_size with output_space='real' in the run's --config). Each "
        "part's crop of the volume is taken in VOXELS from that grid, so every "
        "crop would be a factor off — the same reason `gsplat fit` refuses "
        "--refine volume there. Merge with --recipe levels --refine l2, or "
        "with no refinement."
    )


def _validate_merge_volume_refit(manifest: "BatchManifest") -> None:
    """Refuse a merge-time volume re-fit the source cannot serve, before writing.

    The streaming merge was volume-free by design; it can now re-open the source
    and hand each tile-part its own crop (and each stacked timepoint its own
    slice). The path must still resolve, the axis facts must line up
    (:func:`volume_refit_source_error`), and the crops must be in the source's
    own voxel frame (:func:`volume_refit_frame_error`).
    """
    from pathlib import Path

    if not manifest.input_path:
        raise ValueError(
            "merge: refine='volume' needs the source volume, but the manifest "
            "records no input path. Merge without it, or run `gsplat lod "
            "--recipe levels --target <volume> --refine volume` on a flattened "
            "copy."
        )
    source = Path(manifest.input_path)
    if not source.exists():
        raise ValueError(
            f"merge: refine='volume' needs the source volume, but "
            f"{manifest.input_path} no longer exists. Merge without it, or "
            "re-run the merge from a machine that can see the source."
        )
    problem = volume_refit_source_error(
        manifest.axes, manifest.n_timepoints, manifest.n_channels
    ) or volume_refit_frame_error(manifest.grid_scale)
    if problem:
        raise ValueError(f"merge: refine='volume': {problem}")


def _merge_refit_volume(manifest: "BatchManifest") -> "tuple[Any, tuple]":
    """The lazily-opened source and the center-dim -> volume-axis map.

    Validated by :func:`_validate_merge_volume_refit`, so this only has to build
    what that proved possible. The source is never read whole: each tile-part
    slices its own timepoint and crop out of it.

    The re-opened source is the FULL array, so it still carries the axes this
    batch selected a single index of — a channel, and the time axis itself when
    only one timepoint was fitted (the merge then stacks nothing). A merged part
    has no center column for those, so they are PINNED to the index the fit used:
    lazily, because pre-slicing a zarr array materialises it.
    """
    from pathlib import Path

    from luxar.io.volume import (
        _axis_kind,
        open_volume_lazy,
        pin_volume_axes,
        volume_axes_from_spec,
    )

    volume = open_volume_lazy(Path(manifest.input_path), manifest.array_key)
    labels = [a.strip().lower() for a in (manifest.axes or "").split(",") if a.strip()]
    if len(labels) != len(volume.shape):
        raise ValueError(
            f"merge: refine='volume': --axes {manifest.axes!r} has {len(labels)} "
            f"labels but {Path(manifest.input_path).name} resolved to a "
            f"{len(volume.shape)}D array {tuple(volume.shape)}; the labels must "
            "describe the array the fit read."
        )
    kinds = [_axis_kind(label, "--axes") for label in labels]
    t_indices, c_indices = _tile_indices(manifest)
    # Validation rejects folded channel axes before this builder, so one flat
    # selected channel index can pin at most one source axis here.
    pins = {
        axis: (c_indices[0] if kind == "c" else t_indices[0])
        for axis, kind in enumerate(kinds)
        # The stacked axis is the one the re-fit WALKS (one slice per barrier
        # group), so it is the only non-spatial axis that must stay.
        if kind == "c" or (kind == "t" and manifest.n_timepoints <= 1)
    }
    volume = pin_volume_axes(volume, pins)
    kept = ",".join(label for i, label in enumerate(labels) if i not in pins)
    axes = volume_axes_from_spec(kept, len(volume.shape), flag="--axes")
    return volume, axes


def _slot_cells(
    slot_tree: Dict[str, Any], ndim: int
) -> "Dict[int, List[Tuple[float, float]]]":
    """Each slot's tile, keyed by slot index, for a per-tile volume crop.

    Derived from the split planes already reconstructed for the viewer's part
    ordering. Needs the part's ndim, so the caller resolves it on the FIRST part
    rather than guessing the stacked-axis arity before anything is loaded. An
    unusable tree yields no cells, which turns the per-tile re-fit off rather
    than cropping to a wrong box.
    """
    from luxar.core.group.partition import serialized_bsp_leaf_cells

    try:
        return serialized_bsp_leaf_cells(slot_tree, ndim)
    except (KeyError, TypeError, ValueError):
        return {}


def _cell_for_slot(
    slot_tree: "Optional[Dict[str, Any]]",
    cache: "Dict[int, List[Tuple[float, float]]]",
    slot: int,
    ndim: int,
) -> "Optional[List[Tuple[float, float]]]":
    """One slot's tile, resolving the whole set on first use.

    The cells need the part's ndim, which is only known once a part has been
    loaded, so they are filled in on the first part rather than guessed from the
    stacked-axis arity up front. ``cache`` is the caller's dict, kept across the
    stream so the tree is walked once.
    """
    if slot_tree and not cache:
        cache.update(_slot_cells(slot_tree, ndim))
    return cache.get(slot)


def _with_refit_source(
    recipe_params: "Optional[RecipeParams]", manifest: BatchManifest
) -> "Optional[RecipeParams]":
    """Attach the lazily-opened source volume and its axis map, for a volume re-fit.

    Done once per merge rather than per part: opening a zarr store is cheap but not
    free, and every part slices the SAME handle. ``_validate_merge_volume_refit``
    has already proved the source is readable and mappable, so this cannot fail
    mid-stream after parts were written.
    """
    if recipe_params is None or recipe_params.refine != "volume":
        return recipe_params
    import dataclasses

    volume, axes = _merge_refit_volume(manifest)
    return dataclasses.replace(recipe_params, volume=volume, volume_axes=axes)


def _default_part_coarsen_dims(ndim: int, n_timepoints: int) -> Tuple[int, ...]:
    """The dims a per-part ``levels`` recipe coarsens when none were requested.

    The spatial dims alone: when timepoints were stacked (``n_timepoints > 1``)
    the new axis is appended LAST (:meth:`GSplatData.embed_dimension`), so
    coarsening must not merge across it — it stays a hard barrier. With a single
    timepoint there is no such axis and the answer is every dim, which is the
    same reduction ``coarsen_dims=None`` performs.

    Spelled as a function because two sites need the SAME answer and they see
    the part from different distances: :func:`_finalize_part_node` reads
    ``part.ndim`` off the assembled tile it is about to reduce, while
    :func:`_recipe_pipeline_info` stamps the record before any tile is built and
    has to take the width from the manifest.
    """
    return tuple(range(ndim - (1 if n_timepoints > 1 else 0)))


def _finalize_part_node(
    part: "GSplatData",
    recipe: Optional[str],
    recipe_params: "Optional[RecipeParams]",
    n_timepoints: int,
    cell: "Optional[List[Tuple[float, float]]]" = None,
) -> "GSplatNode":
    """Turn one assembled tile-region ``GSplatData`` into its partition-child node.

    With no ``recipe`` this is just ``part.tree`` (a bare leaf — the historical
    behaviour). With a per-part recipe it builds that recipe ON the single
    tile-region (so the part becomes a leaf-with-ladder or a substitutive lod
    group), giving a ``kind=partition`` whose every child carries its own LOD.

    ``coarsen_dims`` (``substitutive`` only) defaults per part to
    :func:`_default_part_coarsen_dims`. An explicit ``coarsen_dims`` on
    ``recipe_params`` is honoured as-is.
    """
    if recipe is None:
        return part.tree

    import dataclasses

    from luxar.gsplats.fit_basis import fit_image_min
    from luxar.gsplats.lod.recipes import RecipeParams, build_part_lod

    params = recipe_params if recipe_params is not None else RecipeParams()
    if params.image_min is None:
        params = dataclasses.replace(params, image_min=fit_image_min(part.stats))
    if recipe == "levels" and params.coarsen_dims is None:
        params = dataclasses.replace(
            params, coarsen_dims=_default_part_coarsen_dims(part.ndim, n_timepoints)
        )
    # build_part_lod clamps LOD depth to the part's splat count (small tiles never
    # synthesise degenerate levels) — the exact per-part logic of tiles/adaptive.
    return build_part_lod(part.tree, recipe, params, cell=cell)


def _stamp_recipe_floor(
    part: "GSplatData", recipe: Optional[str], floor_stats: Dict[str, Any]
) -> None:
    if recipe is not None:
        part.stats.update(floor_stats)


def _effective_refine_iters(
    refine: str, refine_iters: "Optional[int]"
) -> "Optional[int]":
    """Resolve the refine_iters sentinel the way make_substitutive_lod does:
    None -> the engine's own config default (l2: 120, volume: 300); no refine
    -> None (nothing runs)."""
    if refine == "l2":
        if refine_iters is not None:
            return int(refine_iters)
        from luxar.gsplats.lod._substitutive.refine import L2RefineConfig

        return L2RefineConfig().iters
    if refine == "volume":
        if refine_iters is not None:
            return int(refine_iters)
        from luxar.gsplats.lod.volume_refit import VolumeRefitConfig

        return VolumeRefitConfig().iters
    return None


def _stamped_coarsen_dims(
    requested: "Optional[Sequence[int]]",
    default: "Optional[Sequence[int]]",
) -> Optional[List[int]]:
    """The ``coarsen_dims`` value the merged store publishes.

    An explicit request wins; otherwise the per-part default the caller
    resolved, and only if it could not be resolved does this fall back to
    ``None`` (no provenance). Never invents a width: a wrong explicit list is
    worse than an absent one, because the writer would act on it.

    The surviving list is spelled by
    :func:`~luxar.gsplats.lod.substitutive.resolved_merge_coarsen_dims` rather
    than re-derived here, so this third producer of the key cannot drift from
    the two that build the levels themselves (#1600). Only that resolver's
    normalising half runs: ``dims`` is already an explicit sequence by the time
    it gets there, so the ``ndim`` the resolver reserves for expanding a ``None``
    request is never read — that expansion is ``default``, which the caller
    resolved against the part width only the manifest can supply. Hence the
    ``None`` passed for it: a made-up width is the one thing this must not
    appear to assert, and the resolver raises rather than guessing if the two
    ever meet.
    """
    from luxar.gsplats.lod.substitutive import resolved_merge_coarsen_dims

    dims = requested if requested is not None else default
    return None if dims is None else resolved_merge_coarsen_dims(dims, None)


def _recipe_pipeline_info(
    recipe: Optional[str],
    recipe_params: "Optional[RecipeParams]",
    default_coarsen_dims: "Optional[Sequence[int]]" = None,
) -> Optional[Dict[str, Any]]:
    """Reduction/topology provenance for the merged store's ``pipeline/`` group.

    Mirrors the stats ``gsplat lod`` persists for the same reduction (the
    substitutive builder's out_stats keys / the additive-ladder knobs), plus
    ``per_part=True`` because the merge applies the recipe to each streamed
    tile-part rather than to the whole dataset. ``recipe=None`` → ``None``
    (bare-leaf parts write no ``pipeline/`` group, matching a plain fit).

    ``default_coarsen_dims`` is what the parts will coarsen over when
    ``recipe_params`` names no dims — :func:`_default_part_coarsen_dims` of the
    merged part width, which only the caller can compute (this runs before the
    first tile is assembled). Passing it makes the record EXPLICIT, which is the
    difference between a barrier and a guess: the writer derives the
    chunk-ordering barrier from this key's complement and reads a written
    ``null`` exactly as it reads an absent key, i.e. as no provenance at all
    (#1600). Left out — the caller could not establish a trustworthy width — the
    stamp stays ``None`` and the writer falls back to per-part auto-detection,
    which is the historical behaviour rather than a claim.
    """
    if recipe is None:
        return None
    from luxar.gsplats.lod.recipes import RecipeParams, canonical_recipe_name

    recipe = canonical_recipe_name(recipe)
    params = recipe_params if recipe_params is not None else RecipeParams()
    # ``recipe`` is the build instruction (a recipe, not a structural kind);
    # ``lod_kind`` is the underlying reduction MECHANISM, matching what the
    # standalone builders persist (make_substitutive_lod → "substitutive";
    # make_additive_lod → "additive"). Keeping them distinct: a recipe name
    # (stream/levels/…) must never masquerade as a lod_kind — before the recipe
    # rename the two coincided ("additive"/"substitutive"), which hid the mix-up.
    info: Dict[str, Any] = {
        "recipe": recipe,
        "lod_kind": "additive" if recipe == "stream" else "substitutive",
        "per_part": True,
    }
    if recipe == "stream":
        bp = params.breakpoints
        info.update(
            {
                "n_lods": int(params.n_lods),
                "method": str(params.additive_method),
                "breakpoints": bp if isinstance(bp, str) else list(bp),
            }
        )
    else:  # levels → per-part adaptive (a coarse↔fine lod group per tile)
        info.update(
            {
                "compression_factor": int(params.compression_factor),
                "levels": int(params.levels),
                "method": str(params.substitutive_method),
                "coverage_inflation": float(params.coverage_inflation),
                "conserve_mass": bool(params.conserve_mass),
                "refine": str(params.refine),
                # refine_iters=None is the sentinel for "engine default" —
                # record the value that actually runs (mirrors the resolution
                # in make_substitutive_lod), never int(None).
                "refine_iters": _effective_refine_iters(
                    str(params.refine), params.refine_iters
                ),
                "coarsen_dims": _stamped_coarsen_dims(
                    params.coarsen_dims, default_coarsen_dims
                ),
                "additive_ladders": bool(params.additive_ladders),
            }
        )
    return info


def _batch_floor_stats(manifest: BatchManifest) -> Dict[str, Any]:
    """The normalization block a batch merge can honestly claim (#1175).

    ``batch-fit`` resolves ONE background level at plan time and hands it to
    every ``(t, c)`` task, but the merge never recorded it: the streaming
    partition writer is fed a parts generator with no root node whose ``meta``
    block could be promoted. The manifest remains the authority for the shared
    level, so take it from there rather than inferring it from one tile.

    Only an unambiguous answer is written. ``floor_level`` is set exactly when
    the tasks were handed a concrete number; a ``None`` means one of three
    different things (suppression disabled, a negative level whose SPEC was
    forwarded for each task to re-resolve, or a manifest predating the field),
    and only the first is a claim this merge may make — which the recorded
    ``fit_args["floor"]`` spec distinguishes. Otherwise nothing is written,
    because an absent key reads as "unknown" while ``floor: null`` asserts that
    no pedestal was removed.
    """
    level = manifest.floor_level
    if level is not None:
        return {"floor": float(level)}
    spec = manifest.fit_args.get("floor")
    if isinstance(spec, str) and spec.strip().lower() == "none":
        return {"floor": None}
    return {}


def _manifest_part_coarsen_dims(
    manifest: BatchManifest,
) -> "Optional[Tuple[int, ...]]":
    """:func:`_default_part_coarsen_dims` for this batch's parts, or ``None``.

    Every part comes out of :func:`_build_part_for_tile` the same width — the
    fitted tile's dims, plus ONE if the timepoints were stacked — so the
    manifest settles it before a single tile is read. ``spatial_shape`` is the
    fitted tile's dim count; the explicit-``barrier_dims`` branch below already
    treats ``len(spatial_shape)`` as the stacked axis's index, which is the same
    assumption stated once here.

    ``()`` (a manifest predating the field, or one that never recorded it) is
    the one honest ``None``: guessing a width would publish a barrier over
    columns that may not exist.
    """
    if not manifest.spatial_shape:
        return None
    ndim = len(manifest.spatial_shape) + (1 if manifest.n_timepoints > 1 else 0)
    return _default_part_coarsen_dims(ndim, manifest.n_timepoints)


def _pipeline_info_with_floor(
    recipe: Optional[str],
    recipe_params: "Optional[RecipeParams]",
    floor_stats: Dict[str, Any],
    default_coarsen_dims: "Optional[Sequence[int]]" = None,
) -> Optional[Dict[str, Any]]:
    """:func:`_recipe_pipeline_info` plus the batch's floor block (#1175).

    Kept as its own function so the caller stays one assignment: the merge
    already carries the recipe's reduction provenance and the floor block into
    the same ``pipeline/`` group, and either can be empty.
    """
    merged = {
        **(_recipe_pipeline_info(recipe, recipe_params, default_coarsen_dims) or {}),
        **floor_stats,
    }
    return merged or None


def _slot_bsp_tree(
    manifest: BatchManifest, output_dir: Path, verbose: bool
) -> Optional[Dict[str, Any]]:
    """Split planes over the batch's spatial slots, labelled by SLOT index.

    Recovers the decomposition the fit array worked from, so the merged partition
    can tell the viewer how its parts stack up (#1555) instead of leaving it to
    guess from part centroids. Two modes, two sources:

    * ``content`` — the shared :class:`FitPlan` at ``manifest.plan_path`` carries
      the planner's own recursion, and slot ``k`` IS ``plan.boxes[k]`` (that is the
      ``--plan-box k`` contract each array task fits against). Exact: boxes are
      core-disjoint.
    * ``uniform`` — the tile grid is a deterministic function of the tiling
      geometry, so :func:`~luxar.gsplats.tiling.compute_tile_specs` reproduces it
      exactly, and ``TileSpec.index`` is the slot index. Approximate: apodized
      tiles keep their overlap band (see
      :func:`~luxar.gsplats.tiling.grid_bsp_tree`). That grid is in VOXELS,
      which is not always the frame the tasks' splats came back in, so it is
      scaled by the ``manifest.grid_scale`` the planner recorded (#1587).

    Returns ``None`` — the documented centroid fallback — when the source is
    missing or unusable: a plan written before this landed, an unreadable
    ``plan.json``, an incomplete manifest, a box count that disagrees with the
    plan (which would mislabel every leaf), or (uniform) a recorded
    ``grid_scale`` that does not fit the grid.
    """
    if manifest.mode == "content":
        return _content_slot_bsp_tree(manifest, output_dir, verbose)
    return _uniform_slot_bsp_tree(manifest, verbose)


def _content_slot_bsp_tree(
    manifest: BatchManifest, output_dir: Path, verbose: bool
) -> Optional[Dict[str, Any]]:
    """The shared ``FitPlan``'s own recursion, keyed by ``--plan-box`` slot."""
    if not manifest.plan_path:
        return None
    plan_path = Path(manifest.plan_path)
    if not plan_path.is_absolute():
        plan_path = output_dir / plan_path
    try:
        from luxar.gsplats.planner.spec import FitPlan

        plan = FitPlan.from_json(plan_path)
    except Exception as exc:  # unreadable / malformed / absent
        if verbose:
            aprint(f"  No split planes: could not read {plan_path} ({exc!r})")
        return None
    if plan.bsp_tree is None:
        return None
    if plan.n_boxes != manifest.n_tiles:
        # The plan and the array disagree on the slot space, so leaf labels would
        # not name the boxes the tasks actually fitted.
        if verbose:
            aprint(
                f"  No split planes: plan has {plan.n_boxes} boxes but the "
                f"manifest expects {manifest.n_tiles} slots"
            )
        return None
    if not manifest.grid_scale:
        return plan.bsp_tree
    try:
        import numpy as np

        from luxar.core.group.partition import map_serialized_bsp_tree

        return map_serialized_bsp_tree(
            plan.bsp_tree, linear=np.diag([float(v) for v in manifest.grid_scale])
        )
    except (TypeError, ValueError) as exc:
        # Only a hand-edited / corrupt manifest can reach this: the content
        # planner records a validated length-3 positive factor. Keep it loud for
        # the same reason as the uniform fallback below.
        aprint(
            f"  WARNING: no split planes — the manifest records a grid_scale "
            f"{manifest.grid_scale!r} that does not fit the content plan ({exc}). "
            "The partition is written WITHOUT split planes; the viewer falls "
            "back to ordering the parts by centroid, which can pop at the seams "
            "(#1555)."
        )
        return None


def _uniform_slot_bsp_tree(
    manifest: BatchManifest, verbose: bool
) -> Optional[Dict[str, Any]]:
    """The tile grid, recomputed from the manifest's tiling geometry.

    Scaled into the frame the fit tasks actually emitted in by
    ``manifest.grid_scale``, which the PLANNER resolved from the run's
    ``--preset``/``--config`` (#1587). Nothing is re-derived here: the config
    that produced it may have moved, been deleted, or been replaced by an
    unrelated same-named file by the time a Slurm run is merged. An absent
    ``grid_scale`` — the common case, and every manifest written before that
    field existed — means the grid and the splats share a frame.
    """
    if not manifest.spatial_shape:
        return None
    from luxar.gsplats.tiling import compute_tile_specs, grid_bsp_tree

    try:
        specs = compute_tile_specs(
            tuple(manifest.spatial_shape),
            manifest.tile_size,
            manifest.tile_overlap,
            fold_slivers=manifest.fold_tile_slivers,
        )
    except ValueError as exc:  # geometry the tiler rejects (e.g. overlap >= size)
        if verbose:
            aprint(f"  No split planes: cannot rebuild the tile grid ({exc!r})")
        return None
    if len(specs) != manifest.n_tiles:
        if verbose:
            aprint(
                f"  No split planes: recomputed grid has {len(specs)} tiles but "
                f"the manifest expects {manifest.n_tiles}"
            )
        return None
    try:
        return grid_bsp_tree(
            specs, scale=list(manifest.grid_scale) if manifest.grid_scale else None
        )
    except (ValueError, TypeError) as exc:
        # Only reachable from a hand-edited / corrupt manifest (the planner
        # writes a validated factor of the right length) — hence TypeError too,
        # for an entry that is not a number at all. Loud and NOT gated on
        # `verbose`, because the alternative — aborting a merge that ran after
        # every tile was fitted — is worse than the documented centroid
        # fallback.
        aprint(
            f"  WARNING: no split planes — the manifest records a grid_scale "
            f"{manifest.grid_scale!r} that does not fit this tile grid ({exc}). "
            "The partition is written WITHOUT split planes; the viewer falls "
            "back to ordering the parts by centroid, which can pop at the seams "
            "(#1555)."
        )
        return None


def _merge_partition(
    manifest: BatchManifest,
    output_dir: Path,
    channel_colors: Optional[List[Tuple[float, float, float]]],
    force: bool,
    verbose: bool,
    recipe: Optional[str] = None,
    recipe_params: "Optional[RecipeParams]" = None,
) -> Path:
    """Streaming tile-outer partition merge (the default, memory-safe path)."""
    from luxar.gsplats.io.save_gsplats import (
        resolve_amplitude_bits,
        write_partition_streaming,
    )
    from luxar.gsplats.merged_quality import collect_part_provenance

    tiles_dir = output_dir / "tiles"
    merged_dir = output_dir / "merged"
    merged_dir.mkdir(parents=True, exist_ok=True)
    final_path = merged_dir / "final.gsplats.zarr"

    if final_path.exists() and not force:
        if verbose:
            aprint("  Final output exists, skipping")
        return final_path

    n_k = manifest.n_tiles
    t_indices, c_indices = _tile_indices(manifest)
    # Slot label MUST match what the fit array wrote (uniform=tile, content=box).
    label = "box" if manifest.mode == "content" else "tile"
    # Reduction provenance for the pipeline/ group (None without a recipe),
    # plus the one background level every task subtracted (#1175) — the merge's
    # only chance to record it, since neither branch below has a root node whose
    # `meta` the tree writer could promote.
    floor_stats = _batch_floor_stats(manifest)
    pipeline_info = _pipeline_info_with_floor(
        recipe, recipe_params, floor_stats, _manifest_part_coarsen_dims(manifest)
    )
    root_attrs = (
        {"dimension_metadata": manifest.dimension_metadata}
        if manifest.dimension_metadata is not None
        else None
    )

    # Authoritative ordering barrier: when timepoints are stacked (n_timepoints
    # > 1) the merge appends them as the LAST axis (see _finalize_part_node /
    # _build_part_for_tile), and that stacked-time axis is a hard barrier. Pass
    # it explicitly so per-part chunk ordering groups by timepoint — the merge
    # KNOWS the barrier, so we must not leave it to the value-based auto-detect
    # (which false-negatives on sparse tiles or >max_cardinality timepoints).
    barrier_dims: Optional[List[int]] = None
    if manifest.n_timepoints > 1 and manifest.spatial_shape:
        barrier_dims = [len(manifest.spatial_shape)]  # last axis == stacked time

    # Single tile (K=1) → emit a bare leaf (or, with a recipe, a single lod
    # group / leaf-with-ladder), NOT a 1-part partition.
    if n_k == 1:
        with asection("Merging single tile-region (no partition wrapper)"):
            part = _build_part_for_tile(
                tiles_dir, 0, t_indices, c_indices, n_k, channel_colors, label
            )
            if part is None:
                raise ValueError("Single tile-region is empty — nothing to merge")
            source_dtype = part.stats.get("source_dtype")
            amplitude_bits = resolve_amplitude_bits("auto", source_dtype=source_dtype)
            root_attrs = _dimension_root_attrs(manifest.dimension_metadata, part.ndim)
            single_part_provenance = _single_part_provenance(part)
            if recipe is None:
                # Pass the authoritative stacked-time barrier here too (K==1,
                # no recipe) so a single-tile timelapse gets per-timepoint chunk
                # locality instead of relying on value-based auto-detect.
                # `save` derives pipeline_info from `stats`, so the floor block
                # goes in there rather than through the argument (#1175).
                _drop_root_quality(part)
                part.stats["part_provenance"] = single_part_provenance
                part.stats.update(floor_stats)
                part.save(
                    final_path,
                    amplitude_bits=amplitude_bits,
                    barrier_dims=barrier_dims,
                    root_attrs=root_attrs,
                )
                if verbose:
                    aprint(f"  Wrote bare leaf: {part.n_splats:,} splats, {part.ndim}D")
            else:
                from luxar.gsplats.io.save_gsplats import write_gsplats_tree

                part.stats.update(floor_stats)
                node = _finalize_part_node(
                    part,
                    recipe,
                    _with_refit_source(recipe_params, manifest),
                    manifest.n_timepoints,
                    # One part means no tiling: its "tile" is the whole volume.
                    cell=[(float("-inf"), float("inf"))] * part.ndim,
                )
                write_gsplats_tree(
                    final_path,
                    node,
                    fitting_info=_merge_fitting_info(
                        single_part_provenance, source_dtype
                    ),
                    amplitude_bits=amplitude_bits,
                    source_dtype=source_dtype,
                    pipeline_info=pipeline_info,
                    barrier_dims=barrier_dims,
                    root_attrs=root_attrs,
                )
                if verbose:
                    aprint(
                        f"  Wrote single {recipe} lod: "
                        f"{part.n_splats:,} splats, {part.ndim}D"
                    )
        return final_path

    # K > 1 → streaming partition, one part per spatial tile.
    source_dtype = _source_dtype_for_tiles(tiles_dir, t_indices, c_indices, n_k, label)
    amplitude_bits = resolve_amplitude_bits("auto", source_dtype=source_dtype)
    #
    # Split planes for the viewer's exact back-to-front part order (#1555). The
    # plan is not stored on the manifest itself: `content` mode points at the
    # shared FitPlan JSON every array task fits a box from, and `uniform` mode is
    # a deterministic function of the tiling geometry, so both are recoverable
    # here. Leaf labels are SLOT indices (k below) and get renumbered to written
    # part indices by the provider once the loop has skipped its empty slots.
    slot_tree = _slot_bsp_tree(manifest, output_dir, verbose)
    kept_slots: List[int] = []
    part_provenance: List[Dict[str, Any]] = []
    # A per-part volume re-fit crops the source to each tile. The split planes
    # above ARE those tiles, keyed by the same slot index the loop walks, so the
    # cells come from the tree already reconstructed for ordering.
    refit_params = _with_refit_source(recipe_params, manifest)
    slot_cells: "Dict[int, List[Tuple[float, float]]]" = {}

    def _parts() -> "Iterator[GSplatNode]":
        from luxar.gsplats.tree import GSplatLeaf, GSplatLodGroup

        kept = 0
        kept_slots.clear()
        part_provenance.clear()
        for k in range(n_k):
            part = _build_part_for_tile(
                tiles_dir, k, t_indices, c_indices, n_k, channel_colors, label
            )
            if part is None:
                if verbose:
                    aprint(f"  {label} {k}: empty, skipping")
                continue
            part_provenance.append(
                collect_part_provenance([part], values=[float(k)], fit_reference=None)[
                    0
                ]
            )
            part.stats.pop("part_provenance", None)
            _drop_mismatched_partition_dimension_attrs(
                root_attrs, manifest.dimension_metadata, part.ndim
            )
            _stamp_recipe_floor(part, recipe, floor_stats)
            # Each part is a single nD splat set → a matrix-shaped tree (a leaf,
            # or — with a per-part recipe — a leaf-with-ladder / substitutive lod
            # group). Hand the tree node straight to the streaming writer; it
            # writes part_<i>/ then releases the splats.
            node = _finalize_part_node(
                part,
                recipe,
                refit_params,
                manifest.n_timepoints,
                cell=_cell_for_slot(slot_tree, slot_cells, k, part.ndim),
            )
            assert isinstance(node, (GSplatLeaf, GSplatLodGroup))
            kept += 1
            kept_slots.append(k)
            if verbose:
                aprint(
                    f"  part {kept - 1} <- tile {k}: "
                    f"{part.n_splats:,} splats, {part.ndim}D"
                )
            yield node

    recipe_label = f" ({recipe} per part)" if recipe else ""
    with asection(f"Streaming spatial partition{recipe_label}: {n_k} tiles -> parts"):
        # NOTE: this path deliberately NEVER calls GSplatData.concatenate across
        # all tiles. Each part is built + (optionally LOD'd) + written + released
        # in turn, so peak memory is one tile-region (the OOM fix that motivates
        # tiling) — the per-part recipe operates on that one region only.
        n_written = write_partition_streaming(
            final_path,
            _parts,
            max_elements=0,
            amplitude_bits=amplitude_bits,
            source_dtype=source_dtype,
            fitting_info=lambda: _merge_fitting_info(
                list(part_provenance), source_dtype
            ),
            pipeline_info=pipeline_info,
            barrier_dims=barrier_dims,
            # Resolved after the stream, when `kept_slots` is complete.
            bsp_tree=lambda: prune_serialized_bsp_tree(slot_tree, kept_slots),
            root_attrs=root_attrs,
        )
        if verbose:
            aprint(f"  Wrote kind=partition with {n_written} parts{recipe_label}")

    return final_path


# ════════════════════════════════════════════════════════════════════════
# Legacy: flat single-leaf 3-level fan-in (--flat)
# ════════════════════════════════════════════════════════════════════════


def _merge_flat(
    manifest: BatchManifest,
    output_dir: Path,
    channel_colors: Optional[List[Tuple[float, float, float]]],
    force: bool,
    verbose: bool,
) -> Path:
    """Legacy 3-level fan-in producing a single flat leaf (reloads all tiles)."""
    from luxar.gsplats.gsplat_data import GSplatData

    tiles_dir = output_dir / "tiles"
    merged_dir = output_dir / "merged"
    merged_dir.mkdir(parents=True, exist_ok=True)

    n_t = manifest.n_timepoints
    n_c = manifest.n_channels
    n_k = manifest.n_tiles
    label = "box" if manifest.mode == "content" else "tile"

    t_indices, c_indices = _tile_indices(manifest)

    # ================================================================
    # Level 1: Merge tiles per (T, C)
    # ================================================================
    tc_paths: dict[tuple[int, int], Path] = {}

    t_w = max(2, len(str(max(t_indices))))
    c_w = max(2, len(str(max(c_indices))))

    with asection("Level 1: Merging tiles per (timepoint, channel)"):
        for t_seq, t_real in enumerate(t_indices):
            for c_seq, c_real in enumerate(c_indices):
                out_path = (
                    merged_dir / f"t{t_real:0{t_w}d}_c{c_real:0{c_w}d}.gsplats.zarr"
                )
                tc_paths[(t_seq, c_seq)] = out_path

                if out_path.exists() and not force:
                    if verbose:
                        aprint(f"  t={t_real} c={c_real}: exists, skipping")
                    continue

                # Skip legitimately-empty slots (`_tile_path` returns None).
                tile_files = []
                for k in range(n_k):
                    p = _tile_path(
                        tiles_dir,
                        t_real,
                        c_real,
                        k,
                        max(t_indices),
                        max(c_indices),
                        n_k,
                        label,
                    )
                    if p is not None:
                        tile_files.append(p)

                if not tile_files:
                    raise ValueError(
                        f"t={t_real} c={c_real}: all {n_k} {label}s empty — "
                        "nothing to merge for this (timepoint, channel)"
                    )
                if len(tile_files) == 1:
                    # A T=1 leaf can be the final output and needs metadata;
                    # a T>1 leaf is intermediate and can stay a direct copy.
                    _copy_or_resave_tile(
                        tile_files[0],
                        out_path,
                        manifest.dimension_metadata if n_t == 1 else None,
                    )
                else:
                    datasets = [
                        GSplatData.load(p, include_stats=True) for p in tile_files
                    ]
                    merged = GSplatData.concatenate(datasets)
                    merged.save(
                        out_path,
                        amplitude_bits="auto",
                        root_attrs=_dimension_root_attrs(
                            manifest.dimension_metadata, merged.ndim
                        )
                        if n_t == 1
                        else None,
                    )

                if verbose:
                    aprint(
                        f"  t={t_real} c={c_real}: merged {len(tile_files)} {label}s"
                    )

    # ================================================================
    # Level 2: Stack timepoints per channel (if T > 1)
    # ================================================================
    channel_paths: dict[int, Path] = {}

    if n_t > 1:
        with asection("Level 2: Stacking timepoints per channel"):
            for c_seq in range(n_c):
                out_path = merged_dir / f"c{c_seq:02d}_4d.gsplats.zarr"
                channel_paths[c_seq] = out_path

                if out_path.exists() and not force:
                    if verbose:
                        aprint(f"  c={c_seq}: exists, skipping")
                    continue

                tc_files = [tc_paths[(t_seq, c_seq)] for t_seq in range(n_t)]
                datasets = [GSplatData.load(p, include_stats=True) for p in tc_files]
                stacked = GSplatData.combine_as_new_dimension(
                    datasets,
                    values=[float(t) for t in t_indices],
                    sigma=0.0,
                )
                stacked.save(
                    out_path,
                    amplitude_bits="auto",
                    root_attrs=_dimension_root_attrs(
                        manifest.dimension_metadata, stacked.ndim
                    ),
                )

                if verbose:
                    aprint(
                        f"  c={c_seq}: stacked {n_t} timepoints "
                        f"-> {stacked.ndim}D ({stacked.n_splats:,} splats)"
                    )
    else:
        # Single timepoint — use Level 1 outputs directly
        for c_seq in range(n_c):
            channel_paths[c_seq] = tc_paths[(0, c_seq)]

    # ================================================================
    # Level 3: Merge channels (if C > 1 and colors provided)
    # ================================================================
    if n_c > 1 and channel_colors:
        with asection("Level 3: Merging channels with colors"):
            final_path = merged_dir / "final.gsplats.zarr"

            if final_path.exists() and not force:
                if verbose:
                    aprint("  Final output exists, skipping")
                return final_path

            ch_files = [channel_paths[c] for c in range(n_c)]
            datasets = [GSplatData.load(p, include_stats=True) for p in ch_files]
            final = GSplatData.merge_with_channel_colors(datasets, channel_colors)
            final.save(
                final_path,
                amplitude_bits="auto",
                root_attrs=_dimension_root_attrs(
                    manifest.dimension_metadata, final.ndim
                ),
            )

            if verbose:
                aprint(f"  Merged {n_c} channels -> {final.n_splats:,} splats")
            return final_path

    # No channel merge needed — pick the single-channel output or
    # the most "final" thing we have
    if n_c == 1:
        final_path = channel_paths[0]
    else:
        # Multiple channels but no colors — concatenate
        with asection("Level 3: Concatenating channels"):
            final_path = merged_dir / "final.gsplats.zarr"
            if not final_path.exists() or force:
                ch_files = [channel_paths[c] for c in range(n_c)]
                datasets = [GSplatData.load(p, include_stats=True) for p in ch_files]
                final = GSplatData.concatenate(datasets)
                final.save(
                    final_path,
                    amplitude_bits="auto",
                    root_attrs=_dimension_root_attrs(
                        manifest.dimension_metadata, final.ndim
                    ),
                )
                if verbose:
                    aprint(f"  Concatenated {n_c} channels")

    return final_path
