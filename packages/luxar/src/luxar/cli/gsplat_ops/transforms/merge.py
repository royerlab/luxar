"""Implementation helper for gsplat merge command."""

from __future__ import annotations

from pathlib import Path
from typing import Literal, Optional

import typer
from arbol import aprint, asection

from ..._traceback import exit_with_error
from ..encoding import _resolve_encoding_mode
from ..recipe_shared import carried_appearance_from_inputs


def _mode_invalidated_appearance(
    *, channel_colors: bool, colors_manufactured: bool
) -> dict[str, str]:
    """Appearance keys THIS MERGE invalidates, mapped to the reason.

    Distinct from a disagreement: these are dropped even when every input
    agrees perfectly, because the merge itself makes the value untrue of the
    output. Handed to
    :func:`~luxar.gsplats.io.load_gsplats.agreed_authored_appearance` as
    ``exclude=``, which warns the reason back to the user for each key an input
    actually authored (a writer-manufactured value does not count, so an
    exclusion nobody would have exercised stays quiet).

    Both cases are about ``colormap``, and both about the merge having
    MANUFACTURED per-splat RGB the palette does not describe. They are kept
    separate because they are not the same predicate: ``--channel-colors``
    always bakes an RGB (even when every input already had colors, so nothing
    was "manufactured"), while the white-fill in ``GSplatData.concatenate``
    fires on the mixed case of ANY mode. Either way the output HAS colors, and
    the viewer makes an ancestor palette override per-splat RGB unconditionally
    (``data/attrs-composer.ts`` → ``materials/gsplat/shader-tsl.ts``), so a
    carried colormap would render the RGB inputs through someone's scalar ramp.
    The writer cannot rescue it either: ``apply_gsplat_group_attrs`` gates its
    manufactured ``colormap="gray"`` on ``not has_colors``, so on a colored
    output a carried palette is the ONLY colormap present.

    Note what is NOT here: ``--as-dimension`` does not invalidate
    ``nd_transform``. The stated reason used to be that the mode changes the
    dimension set, but ``combine_as_new_dimension`` APPENDS the new axis LAST
    and ``nd_transform`` is keyed by dimension name, so no existing entry's
    name OR index moves and the new axis simply has no entry — the correct
    identity default. Measured: an entry that validates against the inputs'
    dimensions validates unchanged against the stacked set, and even the
    viewer's positionally SYNTHESIZED names for a standalone ``.gsplats.zarr``
    (``X``/``Y``/``Z``/``dim3``/… — a detached root carries no
    ``scene_dimensions``) are stable, the new axis becoming ``dim4``. So the
    stacked output carries it exactly like a plain concatenating merge does.
    """
    exclude: dict[str, str] = {}
    if channel_colors:
        exclude["colormap"] = (
            "--channel-colors bakes a per-channel RGB, so an input's colormap "
            "no longer describes what is rendered"
        )
    elif colors_manufactured:
        exclude["colormap"] = (
            "the merge gave the colorless input(s) a white fill so the output "
            "could have per-splat RGB, and in the viewer a root palette "
            "overrides per-splat RGB — a carried colormap would render the "
            "colored input through it"
        )
    return exclude


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
        from luxar.cli.gsplat_ops.loading import load_matrix_gsplats
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
                    ds = load_matrix_gsplats(
                        inp,
                        include_stats=False,
                        command="merge",
                    )
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

            # The merge owns the STRUCTURE, not the look: without this the
            # writer's own defaults take over and every authored value on every
            # input is lost (#1600). Unlike a one-input rewrite, "the"
            # appearance has to be AGREED — see `agreed_authored_appearance`.
            #
            # Read off the MERGED RESULT, not off the flags: `concatenate`
            # white-fills the inputs that had no colors when the set is MIXED,
            # so a plain merge (and `--as-dimension`) can manufacture a colored
            # output too — which is exactly the case a flag-keyed exclusion
            # missed. The call sits after the merge for this reason.
            root_attrs = carried_appearance_from_inputs(
                inputs,
                exclude=_mode_invalidated_appearance(
                    channel_colors=bool(channel_colors),
                    colors_manufactured=(
                        merged.colors is not None
                        and any(d.colors is None for d in datasets)
                    ),
                ),
                input_has_colors=[d.colors is not None for d in datasets],
                output_has_colors=merged.colors is not None,
            )

            with asection(f"Saving to {output_path.name}"):
                # Color SDR/HDR is auto-detected by the writer.
                merged.save(
                    output_path,
                    encoding_mode=_resolve_encoding_mode(encoding),
                    compress=compress,
                    root_attrs=root_attrs,
                )
                aprint(f"Saved {merged.n_splats:,} splats ({merged.ndim}D)")

        aprint(f"\nDone: {merged.n_splats:,} splats merged")

    except typer.Exit:
        raise
    except Exception as e:
        # (No partial-output caveat needed: the writer streams into a temp
        # sibling and atomically swaps into place, so a mid-write failure
        # leaves any prior store untouched and no partial output behind.)
        exit_with_error(f"Error: {e}", e)
