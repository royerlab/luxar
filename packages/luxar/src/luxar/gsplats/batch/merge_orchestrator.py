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
from typing import TYPE_CHECKING, Iterator, List, Optional, Tuple

from arbol import aprint, asection

from luxar.gsplats.batch.manifest import BatchManifest, output_filename

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
            it streams (one of :data:`PER_PART_RECIPES`: ``additive`` →
            ``partitioned`` topology, ``substitutive`` → ``mosaic``).
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
    if recipe is not None:
        from luxar.gsplats.lod.recipes import PER_PART_RECIPES

        if recipe not in PER_PART_RECIPES:
            raise ValueError(
                f"merge_batch_results: per-part recipe {recipe!r} is not supported; "
                f"choose from {', '.join(PER_PART_RECIPES)}. The composed recipes "
                "(partitioned/multiscale/mosaic) re-partition their input, but each "
                "tile is already one spatial part."
            )
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


def _tile_path(tiles_dir: Path, t_real: int, c_real: int, k: int, t_max: int, c_max: int, n_k: int) -> Path:
    """Resolve a single tile output path (matches sbatch output naming)."""
    fname = output_filename(t_real, c_real, k, t_max + 1, c_max + 1, n_k)
    tile_path = tiles_dir / fname
    if not tile_path.exists():
        raise FileNotFoundError(
            f"Missing tile output: {tile_path}\n"
            f"Run `luxar gsplat batch status {tiles_dir.parent}` "
            f"to check job status."
        )
    return tile_path


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
) -> "Optional[GSplatData]":
    """Assemble the full nD leaf-``GSplatData`` for spatial tile-region ``k``.

    Tile-outer reorder of the flat fan-in: within ONE spatial tile, stack that
    tile's timepoints (``combine_as_new_dimension`` -> +1 dim, 3D->4D) and apply
    channel handling (color merge if colors given, else concatenate). Only this
    one tile-region's splats are loaded — never the whole volume. Returns
    ``None`` for an empty/zero-splat tile (the caller skips it).
    """
    from luxar.gsplats.gsplat_data import GSplatData

    n_t = len(t_indices)
    n_c = len(c_indices)

    # Per-channel: stack this tile's timepoints (Level-2 semantics, but scoped to
    # a single spatial tile so memory stays bounded to one tile-region).
    per_channel: List[GSplatData] = []
    for c_real in c_indices:
        tc_data: List[GSplatData] = []
        for t_real in t_indices:
            tile_path = _tile_path(
                tiles_dir, t_real, c_real, k, max(t_indices), max(c_indices), n_k
            )
            tc_data.append(GSplatData.load(tile_path))
        if n_t > 1:
            stacked = GSplatData.combine_as_new_dimension(
                tc_data,
                values=[float(t) for t in t_indices],
                sigma=0.0,
            )
        else:
            stacked = tc_data[0]
        per_channel.append(stacked)

    # Across channels (Level-3 semantics, scoped to this tile).
    if n_c == 1:
        part = per_channel[0]
    elif channel_colors:
        part = GSplatData.merge_with_channel_colors(per_channel, channel_colors)
    else:
        part = GSplatData.concatenate(per_channel)

    if part.n_splats == 0:
        return None
    return part


def _finalize_part_node(
    part: "GSplatData",
    recipe: Optional[str],
    recipe_params: "Optional[RecipeParams]",
    n_timepoints: int,
) -> "GSplatNode":
    """Turn one assembled tile-region ``GSplatData`` into its partition-child node.

    With no ``recipe`` this is just ``part.tree`` (a bare leaf — the historical
    behaviour). With a per-part recipe it builds that recipe ON the single
    tile-region (so the part becomes a leaf-with-ladder or a substitutive lod
    group), giving a ``kind=partition`` whose every child carries its own LOD.

    ``coarsen_dims`` (``substitutive`` only) defaults per part to the spatial
    dims alone: when timepoints were stacked (``n_timepoints > 1``) the new axis is
    appended LAST (:meth:`GSplatData.embed_dimension`), so coarsening must not
    merge across it — it stays a hard barrier. An explicit ``coarsen_dims`` on
    ``recipe_params`` is honoured as-is.
    """
    if recipe is None:
        return part.tree

    import dataclasses

    from luxar.gsplats.lod.recipes import RecipeParams, build_part_lod

    params = recipe_params if recipe_params is not None else RecipeParams()
    if recipe == "substitutive" and params.coarsen_dims is None:
        # Stacked-timepoint axis (the last column) is a barrier; coarsen the rest.
        n_spatial = part.ndim - (1 if n_timepoints > 1 else 0)
        if n_spatial < part.ndim:
            params = dataclasses.replace(
                params, coarsen_dims=tuple(range(n_spatial))
            )
    # build_part_lod clamps LOD depth to the part's splat count (small tiles never
    # synthesise degenerate levels) — the exact per-part logic of partitioned/mosaic.
    return build_part_lod(part.tree, recipe, params)


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
    from luxar.gsplats.io.save_gsplats import write_partition_streaming

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

    # Single tile (K=1) → emit a bare leaf (or, with a recipe, a single lod
    # group / leaf-with-ladder), NOT a 1-part partition.
    if n_k == 1:
        with asection("Merging single tile-region (no partition wrapper)"):
            part = _build_part_for_tile(
                tiles_dir, 0, t_indices, c_indices, n_k, channel_colors
            )
            if part is None:
                raise ValueError("Single tile-region is empty — nothing to merge")
            if recipe is None:
                part.save(final_path)
                if verbose:
                    aprint(
                        f"  Wrote bare leaf: {part.n_splats:,} splats, {part.ndim}D"
                    )
            else:
                from luxar.gsplats.io.save_gsplats import write_gsplats_tree

                node = _finalize_part_node(
                    part, recipe, recipe_params, manifest.n_timepoints
                )
                write_gsplats_tree(final_path, node)
                if verbose:
                    aprint(
                        f"  Wrote single {recipe} lod: "
                        f"{part.n_splats:,} splats, {part.ndim}D"
                    )
        return final_path

    # K > 1 → streaming partition, one part per spatial tile.
    def _parts() -> "Iterator[GSplatNode]":
        from luxar.gsplats.tree import GSplatLeaf, GSplatLodGroup

        kept = 0
        for k in range(n_k):
            part = _build_part_for_tile(
                tiles_dir, k, t_indices, c_indices, n_k, channel_colors
            )
            if part is None:
                if verbose:
                    aprint(f"  tile {k}: empty, skipping")
                continue
            # Each part is a single nD splat set → a matrix-shaped tree (a leaf,
            # or — with a per-part recipe — a leaf-with-ladder / substitutive lod
            # group). Hand the tree node straight to the streaming writer; it
            # writes part_<i>/ then releases the splats.
            node = _finalize_part_node(
                part, recipe, recipe_params, manifest.n_timepoints
            )
            assert isinstance(node, (GSplatLeaf, GSplatLodGroup))
            kept += 1
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
        )
        if verbose:
            aprint(
                f"  Wrote kind=partition with {n_written} parts{recipe_label}"
            )

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

                tile_files = []
                for k in range(n_k):
                    tile_files.append(
                        _tile_path(
                            tiles_dir,
                            t_real,
                            c_real,
                            k,
                            max(t_indices),
                            max(c_indices),
                            n_k,
                        )
                    )

                if n_k == 1:
                    # Single tile — just copy/symlink
                    import shutil

                    if out_path.exists():
                        shutil.rmtree(out_path)
                    shutil.copytree(tile_files[0], out_path)
                else:
                    datasets = [GSplatData.load(p) for p in tile_files]
                    merged = GSplatData.concatenate(datasets)
                    merged.save(out_path)

                if verbose:
                    aprint(f"  t={t_real} c={c_real}: merged {n_k} tiles")

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
                datasets = [GSplatData.load(p) for p in tc_files]
                stacked = GSplatData.combine_as_new_dimension(
                    datasets,
                    values=[float(t) for t in t_indices],
                    sigma=0.0,
                )
                stacked.save(out_path)

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
            datasets = [GSplatData.load(p) for p in ch_files]
            final = GSplatData.merge_with_channel_colors(datasets, channel_colors)
            final.save(final_path)

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
                datasets = [GSplatData.load(p) for p in ch_files]
                final = GSplatData.concatenate(datasets)
                final.save(final_path)
                if verbose:
                    aprint(f"  Concatenated {n_c} channels")

    return final_path
