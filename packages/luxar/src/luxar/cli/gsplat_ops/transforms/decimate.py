"""Implementation for the ``luxar gsplat decimate`` command."""

from __future__ import annotations

from pathlib import Path
from typing import Literal, Optional, Union

import typer
from arbol import aprint, asection

from ..encoding import _resolve_encoding_mode
from ..recipe_shared import VALID_ADDITIVE_METHODS


def run_decimate_dataset(
    *,
    input_path: Path,
    output_path: Path,
    target: Optional[int],
    fraction: Optional[float],
    method: Literal["auto", "merge", "prefix"],
    prefix_method: str,
    device: str,
    seed: Optional[int],
    coarsen_dims: Optional[str],
    encoding_mode: Literal["auto", "precision", "memory"],
    compress: Optional[Literal["zip", "tar.gz"]],
) -> None:
    """Reduce a dataset to a target splat count and write a flat result."""
    try:
        from luxar.gsplats.gsplat_data import GSplatData
        from luxar.gsplats.io.save_gsplats import write_gsplats_tree
        from luxar.gsplats.lod.decimate import decimate

        if (target is None) == (fraction is None):
            aprint(
                "❌ Error: give exactly one of --target N (absolute splat count) "
                "or --fraction F (share of the input, 0 < F <= 1)"
            )
            raise typer.Exit(1)

        # Validate the ordering name here rather than letting it reach
        # `compute_additive_order` — same pattern (and same message) the
        # `additive` / `lod` commands use, so a typo is a BadParameter naming
        # the valid set instead of a stack trace from three layers down.
        prefix_norm = (prefix_method or "auto").strip().replace("-", "_")
        if prefix_norm not in VALID_ADDITIVE_METHODS:
            raise typer.BadParameter(
                f"--prefix-method must be one of {list(VALID_ADDITIVE_METHODS)}; "
                f"got {prefix_method!r}"
            )

        encoding_mode_obj = _resolve_encoding_mode(encoding_mode)
        dims = (
            [int(x) for x in coarsen_dims.split(",")]
            if coarsen_dims is not None
            else None
        )

        with asection(f"Decimating: {input_path.name}"):
            with asection("Loading dataset"):
                data = GSplatData.load(input_path, include_stats=False)
                aprint(f"Loaded {data.n_splats:,} splats ({data.ndim}D)")

            request: Union[int, float] = (
                int(target) if target is not None else float(fraction)  # type: ignore[arg-type]
            )
            reduced = decimate(
                data,
                target=request,
                method=method,
                prefix_method=prefix_norm,  # type: ignore[arg-type]
                device=device,
                seed=seed,
                coarsen_dims=dims,
                verbose=True,
            )

            with asection(f"Saving to {output_path.name}"):
                write_gsplats_tree(
                    output_path,
                    reduced.tree,
                    encoding_mode=encoding_mode_obj,
                    compress=compress,
                )
                kept = 100.0 * reduced.n_splats / max(data.n_splats, 1)
                aprint(
                    f"✅ {data.n_splats:,} → {reduced.n_splats:,} splats "
                    f"({kept:.1f}% kept)"
                )

    except typer.Exit:
        raise
    except Exception as e:  # noqa: BLE001 - surfaced to the user as a CLI error
        aprint(f"❌ Decimation failed: {e}")
        raise typer.Exit(1) from e
