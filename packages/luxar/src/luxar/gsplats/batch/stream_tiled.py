"""Bounded-memory store output for a uniform tiled fit."""

from __future__ import annotations

import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable, Iterator, Sequence, cast

import numpy as np
from arbol import aprint

from luxar.gsplats.batch.merge_orchestrator import (
    _prune_slot_tree,
    _slot_cells,
    _translate_slot_tree,
)
from luxar.gsplats.fit_basis import fit_image_min
from luxar.gsplats.fitting.scale_population import (
    finish_population_stats,
    sum_population_counts,
    sum_run_history,
)
from luxar.gsplats.gsplat_data import GSplatData
from luxar.gsplats.io.save_gsplats import (
    split_fitting_info,
    write_gsplats_tree,
    write_partition_streaming,
)
from luxar.gsplats.merged_quality import stamp_merged_quality
from luxar.gsplats.tiling import compute_tile_specs, grid_bsp_tree


@dataclass
class _TileSummary:
    """Only the per-tile fields needed after its splat arrays are released."""

    n_splats: int
    stats: dict[str, Any]


@dataclass
class _StreamOptions:
    """Merge and writer settings shared by sequential and parallel tile sources."""

    num_tiles: int
    volume_shape: tuple[int, ...]
    tile_size: int | Sequence[int]
    overlap: int | Sequence[int]
    progressive: bool
    cull_retention: float | None
    elapsed: float
    source_shape: Sequence[int] | None = None
    source_dtype: str | None = None
    source_itemsize: int | None = None
    source_stored_bytes: int | None = None
    applied_floor: float | None = None
    floor_strategy: str | None = None
    tile_amplitude_floor: float | None = None
    amplitude_pruned: int | None = None
    grid_scale: tuple[float, ...] | None = None
    volume: Any | None = None
    device: str | None = None
    recipe: str | None = None
    recipe_params: Any | None = None
    fold_tile_slivers: bool = True
    root_attrs: dict[str, Any] | None = None
    frame: Any | None = None
    verbose: bool = True


def _translate_node(node: Any, offset: np.ndarray) -> Any:
    """Move a streamed part without importing the CLI's frame adapter."""
    from dataclasses import replace

    from luxar.gsplats.tree import map_leaves

    def _leaf(leaf: Any) -> Any:
        """Translate one leaf while preserving its fitted metadata."""

        moved = GSplatData.from_tree(leaf).translate(offset).tree
        return replace(moved, meta=dict(leaf.meta))

    return map_leaves(node, _leaf)


def _single_tile_store(
    path: Path, load_tile: Callable[[int], GSplatData], o: _StreamOptions
) -> int:
    """A one-tile partition is a bare leaf, as on the in-memory path."""
    from luxar.gsplats.fit_tiled_gsplats import (
        merge_tile_results,
        stamp_tile_amplitude_floor,
    )
    from luxar.gsplats.tree import total_splats

    node = merge_tile_results(
        [load_tile(0)],
        volume_shape=o.volume_shape,
        tile_size=o.tile_size,
        overlap=o.overlap,
        num_tiles=1,
        progressive=o.progressive,
        cull_retention=o.cull_retention,
        elapsed=o.elapsed,
        verbose=o.verbose,
        partition=True,
        recipe=o.recipe,
        recipe_params=o.recipe_params,
        applied_floor=o.applied_floor,
        floor_strategy=o.floor_strategy,
        grid_scale=o.grid_scale,
        source_shape=o.source_shape,
        source_dtype=o.source_dtype,
        source_itemsize=o.source_itemsize,
        source_stored_bytes=o.source_stored_bytes,
        volume=o.volume,
        device=o.device,
        fold_tile_slivers=o.fold_tile_slivers,
    )
    stamp_tile_amplitude_floor(node, o.tile_amplitude_floor, o.amplitude_pruned)
    if o.frame is not None:
        offset = o.frame.offset()
        if offset is not None:
            node = _translate_node(node, offset)
        node.meta.setdefault("fit_stats", {}).update(o.frame.stamps())
    fitting, config, provenance, pipeline = split_fitting_info(
        node.meta.get("fit_stats", {})
    )
    write_gsplats_tree(
        path,
        node,
        amplitude_bits="auto",
        source_dtype=o.source_dtype,
        fitting_info=fitting,
        fitting_config=config,
        provenance_info=provenance,
        pipeline_info=pipeline,
        root_attrs=o.root_attrs,
    )
    return int(total_splats(node))


@dataclass
class _PartitionWriter:
    """Feed ordered tile parts to the batch writer and collect root metadata."""

    path: Path
    load_tile: Callable[[int], GSplatData]
    options: _StreamOptions
    before: list[_TileSummary] = field(default_factory=list)
    kept: list[_TileSummary] = field(default_factory=list)
    kept_slots: list[int] = field(default_factory=list)
    fitting_config: dict[str, Any] = field(default_factory=dict)
    provenance: dict[str, Any] = field(default_factory=dict)
    pipeline: dict[str, Any] = field(default_factory=dict)
    total: int = 0

    def __post_init__(self) -> None:
        """Resolve the uniform tile grid and its painter-order split tree."""

        o = self.options
        self.specs = compute_tile_specs(
            o.volume_shape,
            o.tile_size,
            o.overlap,
            fold_slivers=o.fold_tile_slivers,
        )
        if len(self.specs) != o.num_tiles:
            raise ValueError("tile count does not match the uniform grid")
        self.slot_tree = grid_bsp_tree(self.specs, scale=o.grid_scale)

    def _cull(self, tile: GSplatData) -> GSplatData:
        """Apply the same measured-score-preserving cull as the memory merge."""

        o = self.options
        if o.cull_retention is not None and 0 < o.cull_retention < 1 and tile.n_splats:
            from luxar.gsplats.fit_tiled_gsplats import _cull_keeping_measured_scores

            return _cull_keeping_measured_scores(tile, o.cull_retention)
        return tile

    def parts(self) -> Iterator[Any]:
        """Yield one finished nonempty part at a time to the store writer."""

        from luxar.gsplats.batch.merge_orchestrator import _finalize_part_node

        o = self.options
        cells = (
            _slot_cells(self.slot_tree, len(o.volume_shape)) if self.slot_tree else {}
        )
        for index in range(o.num_tiles):
            tile = self.load_tile(index)
            self.before.append(_TileSummary(tile.n_splats, dict(tile.stats)))
            tile = self._cull(tile)
            if not tile.n_splats:
                continue
            finish_population_stats(tile.stats)
            self.kept.append(_TileSummary(tile.n_splats, dict(tile.stats)))
            self.kept_slots.append(index)
            self.total += tile.n_splats
            node = _finalize_part_node(
                tile, o.recipe, o.recipe_params, 1, cell=cells.get(index)
            )
            offset = o.frame.offset() if o.frame is not None else None
            if offset is not None:
                node = _translate_node(node, offset)
            yield node

    def _floor_stats(self, stats: dict[str, Any]) -> None:
        """Sum exact floor counts, including counts from empty tile markers."""

        o = self.options
        if o.tile_amplitude_floor is None:
            return
        stats["tile_amplitude_floor"] = o.tile_amplitude_floor
        counts = [tile.stats.get("tile_amplitude_pruned") for tile in self.before]
        exact_count = o.amplitude_pruned
        if exact_count is None and all(count is not None for count in counts):
            exact_count = sum(int(count) for count in counts if count is not None)
        if exact_count is not None:
            stats["tile_amplitude_pruned"] = exact_count

    def _frame_stats(self, stats: dict[str, Any]) -> None:
        """Stamp the CLI crop and physical-coordinate frame on the root."""

        frame = self.options.frame
        if frame is None:
            return
        stats.update(frame.stamps())
        if not stats.get("source_shape") and frame.source_shape is not None:
            stats["source_shape"] = list(frame.source_shape)

    def _quality(self, stats: dict[str, Any]) -> None:
        """Score the whole reconstruction by loading one kept tile at a time."""

        o = self.options
        if o.volume is None:
            return
        stamp_merged_quality(
            (self._cull(self.load_tile(i)) for i in self.kept_slots),
            o.volume,
            volume_shape=o.volume_shape,
            grid_scale=o.grid_scale,
            device=o.device,
            verbose=o.verbose,
            image_min=fit_image_min(stats),
            stats=stats,
            total_splats=self.total,
        )

    def fitting_info(self) -> dict[str, Any]:
        """Build merged fitting, provenance, and pipeline metadata after writing."""

        from luxar.gsplats.fit_tiled_gsplats import (
            _merged_fit_stats,
            _report_merged_min_extent,
            _stamp_merge_normalization,
        )

        o = self.options
        stats = _merged_fit_stats(
            cast(Sequence[GSplatData], self.before),
            delivered_splats=self.total,
            volume_shape=o.volume_shape,
            tile_size=o.tile_size,
            overlap=o.overlap,
            num_tiles=o.num_tiles,
            progressive=o.progressive,
            elapsed=o.elapsed,
            source_shape=o.source_shape,
            source_dtype=o.source_dtype,
            source_itemsize=o.source_itemsize,
            source_stored_bytes=o.source_stored_bytes,
            provenance_results=cast(Sequence[GSplatData], self.kept),
            tile_indices=self.kept_slots,
            tile_specs=self.specs,
        )
        sum_population_counts([part.stats for part in self.kept], stats, self.total)
        sum_run_history([tile.stats for tile in self.before], stats)
        _stamp_merge_normalization(
            stats,
            cast(Sequence[GSplatData], self.kept),
            o.applied_floor,
            floor_strategy=o.floor_strategy,
        )
        self._floor_stats(stats)
        self._frame_stats(stats)
        self._quality(stats)
        _report_merged_min_extent(stats, verbose=o.verbose)
        fitting, fit_config, prov, pipe = split_fitting_info(stats)
        self.fitting_config.update(fit_config or {})
        self.provenance.update(prov or {})
        self.pipeline.update(pipe or {})
        return fitting or {}

    def bsp_tree(self) -> dict[str, Any] | None:
        """Prune empty slots and move split planes into the output frame."""

        tree = _prune_slot_tree(self.slot_tree, self.kept_slots)
        frame = self.options.frame
        offset = frame.offset() if frame is not None else None
        if tree is None or offset is None:
            return tree
        return _translate_slot_tree(tree, offset)

    def write(self) -> int:
        """Write the partition atomically and return its total splat count."""

        o = self.options
        started = time.perf_counter()
        write_partition_streaming(
            self.path,
            self.parts,
            max_elements=0,
            amplitude_bits="auto" if o.source_dtype is not None else 16,
            source_dtype=o.source_dtype,
            fitting_info=self.fitting_info,
            fitting_config=self.fitting_config,
            provenance_info=self.provenance,
            pipeline_info=self.pipeline,
            bsp_tree=self.bsp_tree,
            root_attrs=o.root_attrs,
        )
        if o.verbose:
            aprint(
                f"Wrote {self.total:,} splats in {len(self.kept_slots)} tile-parts "
                f"({time.perf_counter() - started:.1f}s merge)"
            )
        return self.total


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
    frame: Any | None = None,
    verbose: bool = True,
) -> int:
    """Write tile parts in grid order, retaining only their small stats blocks.

    ``load_tile`` may read a worker store or a sequential fit's staging store.
    It is called again for quality scoring; both passes hold at most one tile.
    """
    options = _StreamOptions(
        num_tiles=num_tiles,
        volume_shape=volume_shape,
        tile_size=tile_size,
        overlap=overlap,
        progressive=progressive,
        cull_retention=cull_retention,
        elapsed=elapsed,
        source_shape=source_shape,
        source_dtype=source_dtype,
        source_itemsize=source_itemsize,
        source_stored_bytes=source_stored_bytes,
        applied_floor=applied_floor,
        floor_strategy=floor_strategy,
        tile_amplitude_floor=tile_amplitude_floor,
        amplitude_pruned=amplitude_pruned,
        grid_scale=grid_scale,
        volume=volume,
        device=device,
        recipe=recipe,
        recipe_params=recipe_params,
        fold_tile_slivers=fold_tile_slivers,
        root_attrs=root_attrs,
        frame=frame,
        verbose=verbose,
    )
    if num_tiles == 1:
        return _single_tile_store(output_path, load_tile, options)
    return _PartitionWriter(output_path, load_tile, options).write()
