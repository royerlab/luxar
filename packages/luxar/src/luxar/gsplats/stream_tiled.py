"""Bounded-memory store output for a uniform tiled fit."""

from __future__ import annotations

import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable, Iterator, Sequence, cast

from luxar.core.group.partition import prune_serialized_bsp_tree
from luxar.gsplats.fit_basis import fit_image_min
from luxar.gsplats.fitting.scale_population import (
    finish_population_stats,
    sum_population_counts,
    sum_run_history,
)
from luxar.gsplats.gsplat_data import GSplatData
from luxar.gsplats.io.save_gsplats import split_fitting_info, write_partition_streaming
from luxar.gsplats.merged_quality import stamp_merged_quality
from luxar.gsplats.tiling import compute_tile_specs, grid_bsp_tree

@dataclass
class _TileSummary:
    n_splats: int
    stats: dict[str, Any]


def stream_tile_results(
    output_path: Path,
    load_tile: Callable[[int], GSplatData],
    *,
    num_tiles: int,
    volume_shape: tuple[int, ...],
    tile_size: int | Sequence[int],
    overlap: int | Sequence[int],
    progressive: bool,
    cull_retention: float | None,
    elapsed: float,
    source_shape: Sequence[int] | None = None,
    source_dtype: str | None = None,
    source_itemsize: int | None = None,
    source_stored_bytes: int | None = None,
    applied_floor: float | None = None,
    floor_strategy: str | None = None,
    tile_amplitude_floor: float | None = None,
    amplitude_pruned: int | None = None,
    grid_scale: tuple[float, ...] | None = None,
    volume: Any | None = None,
    device: str | None = None,
    recipe: str | None = None,
    recipe_params: Any | None = None,
    fold_tile_slivers: bool = True,
    root_attrs: dict[str, Any] | None = None,
    verbose: bool = True,
) -> int:
    """Write tile parts in grid order, retaining only their small stats blocks.

    ``load_tile`` may read a worker store or a sequential fit's staging store.
    It is called again for quality scoring; both passes hold at most one tile.
    """
    from luxar.gsplats.batch.merge_orchestrator import _finalize_part_node
    from luxar.gsplats.fit_tiled_gsplats import (
        _cull_keeping_measured_scores,
        _merged_fit_stats,
        _report_merged_min_extent,
        _stamp_merge_normalization,
    )

    specs = compute_tile_specs(
        volume_shape, tile_size, overlap, fold_slivers=fold_tile_slivers
    )
    if len(specs) != num_tiles:
        raise ValueError("tile count does not match the uniform grid")
    slot_tree = grid_bsp_tree(specs, scale=grid_scale)
    before: list[_TileSummary] = []
    kept: list[_TileSummary] = []
    kept_slots: list[int] = []
    pipeline: dict[str, Any] = {}
    config: dict[str, Any] = {}
    provenance: dict[str, Any] = {}
    total = 0

    def _cull(tile: GSplatData) -> GSplatData:
        if cull_retention is not None and 0 < cull_retention < 1 and tile.n_splats:
            return _cull_keeping_measured_scores(tile, cull_retention)
        return tile

    def _parts() -> Iterator[Any]:
        nonlocal total
        from luxar.core.group.partition import serialized_bsp_leaf_cells

        cells = serialized_bsp_leaf_cells(slot_tree, len(volume_shape))
        for index in range(num_tiles):
            tile = load_tile(index)
            before.append(_TileSummary(tile.n_splats, dict(tile.stats)))
            tile = _cull(tile)
            if not tile.n_splats:
                continue
            finish_population_stats(tile.stats)
            kept.append(_TileSummary(tile.n_splats, dict(tile.stats)))
            kept_slots.append(index)
            total += tile.n_splats
            cell = cells.get(index)
            yield _finalize_part_node(
                tile, recipe, recipe_params, 1, cell=cell
            )

    def _fitting_info() -> dict[str, Any]:
        stats = _merged_fit_stats(
            cast(Sequence[GSplatData], before),
            delivered_splats=total,
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
            provenance_results=cast(Sequence[GSplatData], kept),
            tile_indices=kept_slots,
            tile_specs=specs,
        )
        sum_population_counts([part.stats for part in kept], stats, total)
        sum_run_history([tile.stats for tile in before], stats)
        _stamp_merge_normalization(
            stats, cast(Sequence[GSplatData], kept), applied_floor,
            floor_strategy=floor_strategy,
        )
        if tile_amplitude_floor is not None:
            stats["tile_amplitude_floor"] = tile_amplitude_floor
            counts = [tile.stats.get("tile_amplitude_pruned") for tile in before]
            exact_count = (
                amplitude_pruned
                if amplitude_pruned is not None
                else sum(int(count) for count in counts)
                if all(count is not None for count in counts)
                else None
            )
            if exact_count is not None:
                stats["tile_amplitude_pruned"] = exact_count
        if volume is not None:
            stamp_merged_quality(
                (_cull(load_tile(i)) for i in kept_slots),
                volume,
                volume_shape=volume_shape,
                grid_scale=grid_scale,
                device=device,
                verbose=verbose,
                image_min=fit_image_min(stats),
                stats=stats,
                total_splats=total,
            )
        _report_merged_min_extent(stats, verbose=verbose)
        fitting, fit_config, prov, pipe = split_fitting_info(stats)
        config.update(fit_config or {})
        provenance.update(prov or {})
        pipeline.update(pipe or {})
        return fitting or {}

    started = time.perf_counter()
    write_partition_streaming(
        output_path,
        _parts,
        max_elements=0,
        amplitude_bits="auto",
        source_dtype=source_dtype,
        fitting_info=_fitting_info,
        fitting_config=config,
        provenance_info=provenance,
        pipeline_info=pipeline,
        bsp_tree=lambda: prune_serialized_bsp_tree(slot_tree, kept_slots),
        root_attrs=root_attrs,
    )
    if verbose:
        from arbol import aprint

        aprint(
            f"Wrote {total:,} splats in {len(kept_slots)} tile-parts "
            f"({time.perf_counter() - started:.1f}s merge)"
        )
    return total
