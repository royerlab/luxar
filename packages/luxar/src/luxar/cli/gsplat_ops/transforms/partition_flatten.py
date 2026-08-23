"""Implementation helpers for partition/flatten gsplat edit commands."""

from __future__ import annotations

from pathlib import Path
from typing import Literal, Optional

import typer
from arbol import aprint, asection

from ..encoding import _resolve_encoding_mode


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
        from luxar.gsplats.gsplat_data import (
            GSplatData,
            stats_after_structure_change,
        )
        from luxar.gsplats.io.load_gsplats import (
            load_gsplat_node,
            read_authored_appearance,
        )
        from luxar.gsplats.tree import iter_default_leaves

        if output_path.exists() and not overwrite:
            aprint(f"❌ Error: {output_path} exists; pass --overwrite to replace it.")
            raise typer.Exit(1)

        encoding_mode_obj = _resolve_encoding_mode(encoding_mode)

        with asection(f"Flattening: {input_path.name}"):
            with asection("Loading tree"):
                node, stats = load_gsplat_node(input_path, include_stats=True)

            leaves = list(iter_default_leaves(node))
            if not leaves:
                aprint("❌ Error: input tree has no leaves")
                raise typer.Exit(1)

            flat = GSplatData.from_default_selection(node).flattened()
            flat = GSplatData.from_additive_sublods(
                list(flat.additive_sublods),
                stats=stats_after_structure_change(stats),
            )
            aprint(
                f"Flattened {len(leaves)} leaf/leaves → {flat.n_splats:,} splats "
                f"({flat.ndim}D, single matrix-shaped leaf)"
            )

            with asection(f"Saving to {output_path.name}"):
                if output_path.exists() and overwrite:
                    import shutil

                    if output_path.is_dir():
                        shutil.rmtree(output_path)
                    else:
                        output_path.unlink()
                flat.save(
                    output_path,
                    encoding_mode=encoding_mode_obj,
                    compress=compress,
                    root_attrs=read_authored_appearance(input_path),
                )
                aprint(f"  Saved flat file: {output_path} ({flat.n_splats:,} splats)")

    except typer.Exit:
        raise
    except Exception as e:
        aprint(f"❌ Error: {e}")
        import traceback

        traceback.print_exc()
        raise typer.Exit(1)
