"""Implementation helpers for partition/flatten gsplat edit commands."""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Iterator, Literal, Optional, Sequence

import numpy as np
import typer
from arbol import aprint, asection

from luxar.io._ordering.compound import (
    _DEFAULT_BARRIER_MAX_CARDINALITY,
    _barrier_axis_qualifies,
    _rounded_barrier_values,
)

from ..encoding import _resolve_encoding_mode


def _default_leaf_paths(group: Any, path: str = "") -> list[str]:
    kind = group.attrs.get("kind")
    if kind == "partition":
        count = sum(1 for name in group if str(name).startswith("part_"))
        paths: list[str] = []
        for index in range(count):
            child = f"part_{index}"
            child_path = f"{path}/{child}" if path else child
            paths.extend(_default_leaf_paths(group[child], child_path))
        return paths
    if kind == "lod":
        count = sum(1 for name in group if str(name).startswith("child_"))
        if count == 0:
            return []
        # On-disk children are coarsest-first; the data-model default is the
        # finest child, not the viewer's unrelated ``default_level`` load hint.
        child = f"child_{count - 1}"
        child_path = f"{path}/{child}" if path else child
        return _default_leaf_paths(group[child], child_path)
    return [path]


def _contains_partition_group(group: Any) -> bool:
    kind = group.attrs.get("kind")
    if kind == "partition":
        return True
    if kind != "lod":
        return False
    count = sum(1 for name in group if str(name).startswith("child_"))
    return any(
        _contains_partition_group(group[f"child_{index}"]) for index in range(count)
    )


def _leaf_splat_groups(root: Any, leaf_path: str) -> list[Any]:
    leaf = root[leaf_path] if leaf_path else root
    count = int(leaf.attrs.get("n_additive_sublods", 1))
    if count == 1 and "centers" in leaf:
        return [leaf]
    return [leaf[f"additive_{index}"] for index in range(count)]


def _order_leaf_paths(root: Any, leaf_paths: Sequence[str]) -> list[str]:
    from luxar.io.ordering import sort_splats_spatial

    if len(leaf_paths) <= 1:
        return list(leaf_paths)
    centroids = []
    for leaf_path in leaf_paths:
        group = root[leaf_path] if leaf_path else root
        bounds = group.attrs.get("center_bounds")
        if bounds is None:
            bounds = group.attrs.get("position_bounds")
        if bounds is None:
            raise ValueError(f"leaf {leaf_path or '/'} has no center/position bounds")
        centroids.append(
            (
                np.asarray(bounds["min"], dtype=np.float32)
                + np.asarray(bounds["max"], dtype=np.float32)
            )
            * 0.5
        )
    order, _ = sort_splats_spatial(np.stack(centroids), method="hilbert")
    return [leaf_paths[int(index)] for index in order]


def _iter_flatten_splat_sets(root: Any, leaf_paths: Sequence[str]) -> Iterator[Any]:
    from luxar.gsplats.tree import GSplatLeaf
    from luxar.io._compiler.gsplat_tree import read_gsplat_node

    for leaf_path in leaf_paths:
        group = root[leaf_path] if leaf_path else root
        leaf = read_gsplat_node(group, root)
        if not isinstance(leaf, GSplatLeaf):
            raise ValueError(f"default leaf path {leaf_path or '/'} is not a leaf")
        yield from (sublod for sublod in leaf.additive_sublods if sublod.n_splats > 0)


def _resolve_barrier_dims(
    splat_groups: Sequence[Any], stats: dict[str, Any]
) -> Optional[tuple[int, ...]]:
    if not splat_groups:
        return None

    coarsen_dims = stats.get("coarsen_dims")
    if coarsen_dims is not None:
        ndim = int(splat_groups[0].attrs["ndim"])
        coarsen_set = {int(dim) for dim in coarsen_dims}
        return tuple(dim for dim in range(ndim) if dim not in coarsen_set)

    stamped: set[tuple[int, ...]] = set()
    for group in splat_groups:
        if "slice_dims" not in group.attrs:
            return None
        stamped.add(tuple(int(dim) for dim in group.attrs["slice_dims"]))
    if len(stamped) != 1:
        return None
    resolved = next(iter(stamped))
    ndim = int(splat_groups[0].attrs["ndim"])
    return None if not resolved and ndim > 3 else resolved


@dataclass
class _BarrierAxisAccumulator:
    max_cardinality: int
    rounded_values: set[int] = field(default_factory=set)
    exact_values: Optional[set[float]] = field(default_factory=set)
    counts_by_group: Optional[list[dict[float, int]]] = field(default_factory=list)
    current_counts: dict[float, int] = field(default_factory=dict)
    active: bool = True

    def add(self, values: np.ndarray) -> None:
        if not self.active:
            return
        rounded = _rounded_barrier_values(values)
        if rounded is None:
            self._disable()
            return
        self.rounded_values.update(int(value) for value in np.unique(rounded))
        if len(self.rounded_values) > self.max_cardinality:
            self._disable()
            return
        if self.counts_by_group is None:
            return
        unique, counts = np.unique(values, return_counts=True)
        assert self.exact_values is not None
        self.exact_values.update(float(value) for value in unique)
        if len(self.exact_values) > self.max_cardinality:
            self.exact_values = None
            self.counts_by_group = None
            self.current_counts.clear()
            return
        for value, count in zip(unique, counts):
            key = float(value)
            self.current_counts[key] = self.current_counts.get(key, 0) + int(count)

    def finish_group(self) -> None:
        if self.counts_by_group is not None:
            self.counts_by_group.append(self.current_counts)
            self.current_counts = {}

    def qualifies(self, n_splats: int) -> bool:
        return self.active and _barrier_axis_qualifies(
            n_splats, len(self.rounded_values), self.max_cardinality
        )

    def runs(self) -> Optional[list[tuple[np.ndarray, np.ndarray]]]:
        if self.counts_by_group is None:
            return None
        runs = []
        for group_counts in self.counts_by_group:
            ordered = sorted(group_counts)
            runs.append(
                (
                    np.asarray(ordered, dtype=np.float32).reshape(-1, 1),
                    np.asarray(
                        [group_counts[value] for value in ordered], dtype=np.int64
                    ),
                )
            )
        return runs

    def _disable(self) -> None:
        self.active = False
        self.rounded_values.clear()
        self.exact_values = None
        self.counts_by_group = None
        self.current_counts.clear()


def _detect_barrier_dims_across_groups(
    splat_groups: Sequence[Any],
    root: Any,
    decoder: Any,
    max_cardinality: int = _DEFAULT_BARRIER_MAX_CARDINALITY,
) -> tuple[tuple[int, ...], Optional[list[tuple[np.ndarray, np.ndarray]]]]:
    ndim = int(splat_groups[0].attrs["ndim"])
    accumulators = [_BarrierAxisAccumulator(max_cardinality) for _ in range(ndim)]
    n_splats = 0
    for group in splat_groups:
        centers = np.asarray(decoder.decode(group["centers"], root))
        n_splats += len(centers)
        for start in range(0, len(centers), 65_536):
            block = centers[start : start + 65_536]
            for dim, accumulator in enumerate(accumulators):
                accumulator.add(block[:, dim])
        for accumulator in accumulators:
            accumulator.finish_group()
        del centers
    barrier_dims = tuple(
        dim
        for dim, accumulator in enumerate(accumulators)
        if accumulator.qualifies(n_splats)
    )
    if len(barrier_dims) != 1:
        return barrier_dims, None
    runs = accumulators[barrier_dims[0]].runs()
    if runs is None:
        raise ValueError(
            f"auto-detected barrier axis {barrier_dims[0]} exceeds the "
            f"{max_cardinality}-value streaming limit; coarsen_dims is absent"
        )
    return barrier_dims, runs


def _stored_color_layout(group: Any) -> tuple[Optional[np.dtype[Any]], int]:
    if "colors" not in group:
        return None, 0
    colors = group["colors"]
    encoding = colors.attrs.get("encoding", {})
    original_shape = encoding.get("original_shape")
    if original_shape is not None and len(original_shape) > 1:
        color_channels = int(original_shape[1])
    elif len(colors.shape) > 1:
        color_channels = int(colors.shape[1])
    else:
        raise ValueError("stored colors have no logical channel dimension")
    return np.dtype(encoding.get("original_dtype", colors.dtype)), color_channels


def _collect_streaming_metadata(
    splat_groups: Sequence[Any],
    root: Any,
    decoder: Any,
    barrier_dims: Sequence[int],
    barrier_runs: Optional[Sequence[tuple[np.ndarray, np.ndarray]]] = None,
    coarsen_dims: Optional[Sequence[int]] = None,
) -> list[Any]:
    from luxar.gsplats.io.save_gsplats import StreamingSplatSetMetadata
    from luxar.typing_utils.constants import DEFAULT_TRUNCATION_RADIUS

    metadata = []
    barrier_values_seen: set[tuple[float, ...]] = set()
    n_splats = 0
    if barrier_runs is not None and len(barrier_runs) != len(splat_groups):
        raise ValueError("barrier runs do not match streamed splat groups")
    for index, group in enumerate(splat_groups):
        color_dtype, color_channels = _stored_color_layout(group)
        barrier_values = None
        barrier_counts = None
        if barrier_dims:
            if barrier_runs is not None:
                barrier_values, barrier_counts = barrier_runs[index]
            else:
                centers = decoder.decode(group["centers"], root)
                barrier_values, barrier_counts = _barrier_runs(
                    centers,
                    barrier_dims,
                    coarsen_dims=coarsen_dims,
                )
                del centers
            barrier_values_seen.update(
                tuple(float(component) for component in value)
                for value in barrier_values
            )
            if len(barrier_values_seen) > _DEFAULT_BARRIER_MAX_CARDINALITY:
                _raise_barrier_cardinality_error(
                    barrier_dims,
                    coarsen_dims,
                    len(barrier_values_seen),
                    n_splats + int(group.attrs["n_splats"]),
                    _DEFAULT_BARRIER_MAX_CARDINALITY,
                )
        n_splats += int(group.attrs["n_splats"])
        metadata.append(
            StreamingSplatSetMetadata(
                n_splats=int(group.attrs["n_splats"]),
                ndim=int(group.attrs["ndim"]),
                truncation_radius=float(
                    group.attrs.get("truncation_radius", DEFAULT_TRUNCATION_RADIUS)
                ),
                color_channels=color_channels,
                color_dtype=color_dtype,
                barrier_values=barrier_values,
                barrier_counts=barrier_counts,
            )
        )
    if barrier_dims and not _barrier_axis_qualifies(
        n_splats,
        len(barrier_values_seen),
        _DEFAULT_BARRIER_MAX_CARDINALITY,
    ):
        _raise_barrier_cardinality_error(
            barrier_dims,
            coarsen_dims,
            len(barrier_values_seen),
            n_splats,
            _DEFAULT_BARRIER_MAX_CARDINALITY,
        )
    return metadata


def _barrier_runs(
    centers: Any,
    barrier_dims: Sequence[int],
    *,
    max_cardinality: int = _DEFAULT_BARRIER_MAX_CARDINALITY,
    coarsen_dims: Optional[Sequence[int]] = None,
) -> tuple[np.ndarray, np.ndarray]:
    counts: dict[tuple[float, ...], int] = {}
    for start in range(0, len(centers), 65_536):
        block = np.asarray(centers[start : start + 65_536])[:, barrier_dims]
        values, block_counts = np.unique(block, axis=0, return_counts=True)
        for value, count in zip(values, block_counts):
            key = tuple(float(component) for component in value)
            counts[key] = counts.get(key, 0) + int(count)
            if len(counts) > max_cardinality:
                _raise_barrier_cardinality_error(
                    barrier_dims,
                    coarsen_dims,
                    len(counts),
                    len(centers),
                    max_cardinality,
                )
    ordered = sorted(counts)
    return (
        np.asarray(ordered, dtype=np.float32),
        np.asarray([counts[key] for key in ordered], dtype=np.int64),
    )


def _raise_barrier_cardinality_error(
    barrier_dims: Sequence[int],
    coarsen_dims: Optional[Sequence[int]],
    n_unique: int,
    n_splats: int,
    max_cardinality: int,
) -> None:
    source = (
        f"coarsen_dims {list(coarsen_dims)}"
        if coarsen_dims is not None
        else "persisted slice_dims"
    )
    raise ValueError(
        f"barrier axes {list(barrier_dims)} derived from {source} have {n_unique} "
        f"distinct value tuples across {n_splats} splats; categorical barriers must "
        f"have at most {max_cardinality} values and at least four splats per value"
    )


def run_partition_dataset(
    *,
    input_path: Path,
    output_path: Path,
    max_elements: Optional[int],
    parts: Optional[int],
    rule: Literal["median", "midpoint", "sah"],
    encoding_mode: Literal["auto", "precision", "memory"],
    compress: Optional[Literal["zip", "tar.gz"]],
) -> None:
    """Run partition command implementation."""
    try:
        import math

        from luxar.cli.gsplat_ops.loading import load_matrix_gsplats
        from luxar.gsplats.gsplat_data import stats_after_structure_change
        from luxar.gsplats.io.load_gsplats import read_authored_appearance
        from luxar.gsplats.io.save_gsplats import split_fitting_info, write_gsplats_tree
        from luxar.gsplats.tree import iter_leaves

        if max_elements is None and parts is None:
            aprint("❌ Error: specify --max-elements N (or --parts N)")
            raise typer.Exit(1)
        if max_elements is not None and parts is not None:
            aprint("❌ Error: --max-elements and --parts are mutually exclusive")
            raise typer.Exit(1)

        encoding_mode_obj = _resolve_encoding_mode(encoding_mode)

        with asection(f"Partitioning: {input_path.name}"):
            with asection("Loading dataset"):
                data = load_matrix_gsplats(
                    input_path,
                    include_stats=True,
                    command="partition",
                )
                aprint(f"Loaded {data.n_splats:,} splats ({data.ndim}D)")

            # --parts N → target ~N parts via ceil(n / N).
            resolved_max = (
                max_elements
                if max_elements is not None
                else max(1, math.ceil(data.n_splats / int(parts)))  # type: ignore[arg-type]
            )

            with asection("Spatial BSP partition"):
                partition_node = data.to_spatial_partition(
                    max_elements=resolved_max, rule=rule
                )
                part_sizes = [leaf.n_splats for leaf in iter_leaves(partition_node)]
                aprint(
                    f"Produced {len(part_sizes)} spatial parts "
                    f"(rule={rule}, max_elements={resolved_max:,}): sizes={part_sizes}"
                )

            with asection(f"Saving to {output_path.name}"):
                # Thread the loaded provenance through the tree writer (it used to
                # be silently dropped, so partitioning a fit erased `psnr_db`,
                # `fitter_name` and the source grid alike). A BSP partition is
                # content-PRESERVING — the same splats, regrouped — so the measured
                # scores stay true of it and no content scrub applies. Its
                # STRUCTURE is not: the output is a `kind=partition` of bare
                # leaves, so the input's LOD topology record is false of it.
                fitting, config, provenance, pipeline = split_fitting_info(
                    stats_after_structure_change(data.stats), include_fitting_info=True
                )
                write_gsplats_tree(
                    output_path,
                    partition_node,
                    encoding_mode=encoding_mode_obj,
                    compress=compress,
                    fitting_info=fitting,
                    fitting_config=config,
                    provenance_info=provenance,
                    pipeline_info=pipeline,
                    root_attrs=read_authored_appearance(input_path),
                )
                aprint(
                    f"  Saved kind=partition file: {output_path} "
                    f"({data.n_splats:,} splats in {len(part_sizes)} parts)"
                )

    except typer.Exit:
        raise
    except Exception as e:
        aprint(f"❌ Error: {e}")
        import traceback

        traceback.print_exc()
        raise typer.Exit(1)


def run_flatten_dataset(
    *,
    input_path: Path,
    output_path: Path,
    encoding_mode: Literal["auto", "precision", "memory"],
    compress: Optional[Literal["zip", "tar.gz"]],
    overwrite: bool,
) -> None:
    """Run flatten command implementation."""
    try:
        import shutil

        from luxar._zarr_compat import open_group as zc_open_group
        from luxar.encoding import ArrayDecoder
        from luxar.gsplats.gsplat_data import stats_after_structure_change
        from luxar.gsplats.io._archive import resolve_store_path
        from luxar.gsplats.io.load_gsplats import (
            read_authored_appearance,
            read_gsplat_root_stats,
        )
        from luxar.gsplats.io.save_gsplats import (
            split_fitting_info,
            write_flat_leaf_streaming,
        )

        if output_path.exists() and not overwrite:
            aprint(f"❌ Error: {output_path} exists; pass --overwrite to replace it.")
            raise typer.Exit(1)

        encoding_mode_obj = _resolve_encoding_mode(encoding_mode)
        if encoding_mode_obj.value != "precision":
            aprint(
                "❌ Error: streaming flatten currently requires "
                "--encoding precision; auto/memory need whole-array analysis."
            )
            raise typer.Exit(1)

        with asection(f"Flattening: {input_path.name}"):
            zarr_path, temp_dir = resolve_store_path(input_path)
            try:
                root = zc_open_group(str(zarr_path), mode="r")
                stats = read_gsplat_root_stats(root)
                leaf_paths = _default_leaf_paths(root)
                if not leaf_paths:
                    aprint("❌ Error: input tree has no leaves")
                    raise typer.Exit(1)

                leaf_paths = _order_leaf_paths(root, leaf_paths)

                splat_groups = [
                    group
                    for leaf_path in leaf_paths
                    for group in _leaf_splat_groups(root, leaf_path)
                ]
                barrier_dims = _resolve_barrier_dims(splat_groups, stats)
                decoder = ArrayDecoder()
                barrier_runs = None
                if barrier_dims is None:
                    barrier_dims, barrier_runs = _detect_barrier_dims_across_groups(
                        splat_groups, root, decoder
                    )
                splat_set_metadata = _collect_streaming_metadata(
                    splat_groups,
                    root,
                    decoder,
                    barrier_dims,
                    barrier_runs,
                    coarsen_dims=stats.get("coarsen_dims"),
                )

                def splat_sets() -> Iterator[Any]:
                    yield from _iter_flatten_splat_sets(root, leaf_paths)

                carried_stats = stats_after_structure_change(stats)
                if _contains_partition_group(root):
                    carried_stats.pop("part_provenance", None)
                fitting, config, provenance, pipeline = split_fitting_info(
                    carried_stats, include_fitting_info=True
                )

                with asection(f"Saving to {output_path.name}"):
                    n_splats = write_flat_leaf_streaming(
                        output_path,
                        splat_sets,
                        splat_set_metadata=splat_set_metadata,
                        encoding_mode=encoding_mode_obj,
                        compress=compress,
                        fitting_info=fitting,
                        fitting_config=config,
                        provenance_info=provenance,
                        pipeline_info=pipeline,
                        root_attrs=read_authored_appearance(input_path),
                        barrier_dims=barrier_dims,
                    )
                    aprint(f"  Saved flat file: {output_path} ({n_splats:,} splats)")
            finally:
                if temp_dir is not None and temp_dir.exists():
                    shutil.rmtree(temp_dir, ignore_errors=True)

    except typer.Exit:
        raise
    except Exception as e:
        aprint(f"❌ Error: {e}")
        import traceback

        traceback.print_exc()
        raise typer.Exit(1)
