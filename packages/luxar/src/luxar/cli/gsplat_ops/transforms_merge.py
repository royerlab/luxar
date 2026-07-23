"""Implementation helper for gsplat merge command."""

from __future__ import annotations

from pathlib import Path
from typing import Literal, Optional

import typer
from arbol import aprint, asection

from .encoding import _resolve_encoding_mode


def run_merge_datasets(
    *,
    inputs: list[Path],
    output_path: Path,
    as_dimension: bool,
    values: Optional[str],
    sigma: float,
    channel_colors: Optional[str],
    compress: Optional[Literal["zip", "tar.gz"]],
    encoding: Literal["auto", "precision", "memory"],
) -> None:
    """Run merge command implementation."""
    try:
        from luxar.cli.gsplat_config import parse_hex_color
        from luxar.gsplats.gsplat_data import GSplatData

        if len(inputs) < 2:
            aprint("Error: At least 2 input datasets required for merge")
            raise typer.Exit(1)

        if as_dimension and channel_colors:
            aprint("Error: --as-dimension and --channel-colors are mutually exclusive")
            raise typer.Exit(1)

        with asection(f"Merging {len(inputs)} datasets"):
            datasets: list[GSplatData] = []
            total_splats = 0
            for inp in inputs:
                with asection(f"Loading {inp.name}"):
                    ds = GSplatData.load(inp, include_stats=False)
                    aprint(f"{ds.n_splats:,} splats ({ds.ndim}D)")
                    datasets.append(ds)
                    total_splats += ds.n_splats
            aprint(f"Total input splats: {total_splats:,}")

            if channel_colors:
                color_strs = [c.strip() for c in channel_colors.split(",")]
                if len(color_strs) != len(datasets):
                    aprint(
                        f"Error: {len(color_strs)} colors but {len(datasets)} datasets"
                    )
                    raise typer.Exit(1)
                colors = [parse_hex_color(c) for c in color_strs]
                with asection("Merging with channel colors"):
                    merged = GSplatData.merge_with_channel_colors(datasets, colors)

            elif as_dimension:
                if values is not None:
                    dim_values: list[float] = [
                        float(v.strip()) for v in values.split(",")
                    ]
                    if len(dim_values) != len(datasets):
                        aprint(
                            f"Error: {len(dim_values)} values but "
                            f"{len(datasets)} datasets"
                        )
                        raise typer.Exit(1)
                else:
                    dim_values = [float(i) for i in range(len(datasets))]
                with asection(f"Stacking along new dimension (sigma={sigma})"):
                    aprint(f"  Values: {dim_values}")
                    merged = GSplatData.combine_as_new_dimension(
                        datasets, values=dim_values, sigma=sigma
                    )
                    aprint(f"  Result: {merged.ndim}D ({merged.n_splats:,} splats)")

            else:
                with asection("Concatenating"):
                    merged = GSplatData.concatenate(datasets)

            with asection(f"Saving to {output_path.name}"):
                # Color SDR/HDR is auto-detected by the writer.
                merged.save(
                    output_path,
                    encoding_mode=_resolve_encoding_mode(encoding),
                    compress=compress,
                )
                aprint(f"Saved {merged.n_splats:,} splats ({merged.ndim}D)")

        aprint(f"\nDone: {merged.n_splats:,} splats merged")

    except typer.Exit:
        raise
    except Exception as e:
        # (No partial-output caveat needed: the writer streams into a temp
        # sibling and atomically swaps into place, so a mid-write failure
        # leaves any prior store untouched and no partial output behind.)
        aprint(f"Error: {e}")
        import traceback

        traceback.print_exc()
        raise typer.Exit(1)
