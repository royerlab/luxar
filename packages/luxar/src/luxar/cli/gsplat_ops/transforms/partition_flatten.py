"""Implementation helpers for partition/flatten gsplat edit commands."""

from __future__ import annotations

from pathlib import Path
from typing import TYPE_CHECKING, Any, Dict, Iterator, Literal, Optional

import typer
from arbol import aprint, asection

from ..encoding import _resolve_encoding_mode

if TYPE_CHECKING:
    from luxar.gsplats.tree import GSplatNode


def _contains_partition(node: "GSplatNode") -> bool:
    from luxar.gsplats.tree import GSplatLodGroup, GSplatPartition

    if isinstance(node, GSplatPartition):
        return True
    if isinstance(node, GSplatLodGroup):
        return any(_contains_partition(child) for child in node.children)
    return False


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
        child = f"child_{count - 1}"
        child_path = f"{path}/{child}" if path else child
        return _default_leaf_paths(group[child], child_path)
    return [path]


def _root_stats(root: Any) -> Dict[str, Any]:
    stats: Dict[str, Any] = {}
    if "fitting" in root:
        stats.update(dict(root["fitting"].attrs))
    if "pipeline" in root:
        for key, value in root["pipeline"].attrs.items():
            stats.setdefault(key, value)
    if "provenance" in root:
        stats["provenance"] = dict(root["provenance"].attrs)
    stats["format_version"] = root.attrs.get("format_version")
    stats["timestamp"] = root.attrs.get("timestamp")
    stats["luxar_gsplats_version"] = root.attrs.get("luxar_gsplats_version")
    if "description" in root.attrs:
        stats["description"] = root.attrs["description"]
    return stats


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

        import numpy as np

        from luxar._zarr_compat import open_group as zc_open_group
        from luxar.gsplats.gsplat_data import stats_after_structure_change
        from luxar.gsplats.io._archive import resolve_store_path
        from luxar.gsplats.io.load_gsplats import (
            read_authored_appearance,
        )
        from luxar.gsplats.io.save_gsplats import (
            split_fitting_info,
            write_flat_leaf_streaming,
        )
        from luxar.io._compiler.gsplat_tree import read_gsplat_node
        from luxar.io.ordering import sort_splats_spatial

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
                leaf_paths = _default_leaf_paths(root)
                if not leaf_paths:
                    aprint("❌ Error: input tree has no leaves")
                    raise typer.Exit(1)

                centroids = []
                for leaf_path in leaf_paths:
                    group = root[leaf_path] if leaf_path else root
                    bounds = group.attrs.get("center_bounds")
                    if bounds is None:
                        bounds = group.attrs.get("position_bounds")
                    if bounds is None:
                        raise ValueError(
                            f"leaf {leaf_path or '/'} has no center/position bounds"
                        )
                    centroids.append(
                        (
                            np.asarray(bounds["min"], dtype=np.float32)
                            + np.asarray(bounds["max"], dtype=np.float32)
                        )
                        * 0.5
                    )
                if len(leaf_paths) > 1:
                    order, _ = sort_splats_spatial(
                        np.stack(centroids), method="hilbert"
                    )
                    leaf_paths = [leaf_paths[int(index)] for index in order]

                slice_dims = {
                    tuple(
                        int(dim)
                        for dim in (
                            root[path].attrs.get("slice_dims", [])
                            if path
                            else root.attrs.get("slice_dims", [])
                        )
                    )
                    for path in leaf_paths
                }
                barrier_dims = next(iter(slice_dims)) if len(slice_dims) == 1 else None
                stats = _root_stats(root)

                def splat_sets() -> Iterator[Any]:
                    for leaf_path in leaf_paths:
                        group = root[leaf_path] if leaf_path else root
                        leaf = read_gsplat_node(group, root)
                        for sublod in leaf.additive_sublods:
                            if sublod.n_splats > 0:
                                yield sublod

                carried_stats = stats_after_structure_change(stats)
                if root.attrs.get("kind") == "partition":
                    carried_stats.pop("part_provenance", None)
                fitting, config, provenance, pipeline = split_fitting_info(
                    carried_stats, include_fitting_info=True
                )

                with asection(f"Saving to {output_path.name}"):
                    n_splats = write_flat_leaf_streaming(
                        output_path,
                        splat_sets,
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
