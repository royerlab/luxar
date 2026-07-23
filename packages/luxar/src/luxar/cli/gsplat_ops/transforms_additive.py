"""Implementation helper for gsplat additive command."""

from __future__ import annotations

from pathlib import Path
from typing import Literal, Optional

import typer
from arbol import aprint, asection

from .encoding import _resolve_encoding_mode


def run_additive_dataset(
    *,
    input_path: Path,
    output_path: Path,
    n_lods: Optional[int],
    method: Optional[str],
    breakpoints: Optional[str],
    target_ms: Optional[float],
    bandwidth_mbps: Optional[float],
    bytes_per_splat: Optional[float],
    encoding_mode: Literal["auto", "precision", "memory"],
    compress: Optional[Literal["zip", "tar.gz"]],
    overwrite: bool,
) -> None:
    """Run additive command implementation."""
    try:
        from dataclasses import replace

        from luxar.cli.lod import (
            VALID_ADDITIVE_METHODS,
            detect_store_encoding,
            estimate_bytes_per_splat,
            measure_store_bytes,
            parse_lod_breakpoints,
            resolve_streaming_breakpoints,
            validate_streaming_knobs,
        )
        from luxar.gsplats.gsplat_data import GSplatData
        from luxar.gsplats.io.load_gsplats import load_gsplat_node
        from luxar.gsplats.io.save_gsplats import split_fitting_info, write_gsplats_tree
        from luxar.gsplats.lod.additive import (
            clamp_counts_breakpoints,
            make_additive_lod,
            validate_counts_breakpoints,
        )
        from luxar.gsplats.tree import (
            GSplatLeaf,
            GSplatNode,
            iter_leaves,
            map_leaves,
            node_ndim,
        )

        # ── usage validation (mirrors `gsplat lod`) ──
        method_norm = (method or "auto").strip().replace("-", "_")
        if method_norm not in VALID_ADDITIVE_METHODS:
            raise typer.BadParameter(
                f"--method must be one of {list(VALID_ADDITIVE_METHODS)}; "
                f"got {method!r}"
            )
        validate_streaming_knobs(
            target_ms, bandwidth_mbps, bytes_per_splat, breakpoints
        )
        bp = parse_lod_breakpoints(breakpoints or "equal-count")
        eff_n_lods = n_lods if n_lods is not None else 4
        encoding_mode_obj = _resolve_encoding_mode(encoding_mode)

        if output_path.exists() and not overwrite:
            aprint(f"❌ Error: {output_path} exists; pass --overwrite to replace it.")
            raise typer.Exit(1)

        with asection(f"Additive laddering: {input_path.name}"):
            with asection("Loading tree"):
                node, stats = load_gsplat_node(input_path, include_stats=True)
            leaves = list(iter_leaves(node))
            n_leaves = len(leaves)
            total_stored = sum(leaf.n_splats for leaf in leaves)
            if n_leaves == 0 or total_stored == 0:
                aprint("❌ Error: input tree has no splats to ladder")
                raise typer.Exit(1)
            has_colors = any(
                sub.colors is not None for lf in leaves for sub in lf.additive_sublods
            )
            aprint(
                f"Loaded {total_stored:,} stored splats across {n_leaves} "
                f"leaf/leaves ({node_ndim(node)}D)"
            )

            # Explicit counts: are clamped PER LEAF below (parts/levels differ
            # in N), but the spec must still fit the dataset as a whole — a
            # largest count exceeding the union N is a typo and aborts loudly
            # (mirrors a direct whole-dataset `lod --recipe stream` build).
            try:
                validate_counts_breakpoints(bp, total_stored)
            except ValueError as e:
                raise typer.BadParameter(str(e)) from e

            # ── streaming breakpoints from --target-ms (measured B/splat) ──
            if target_ms is not None:
                store_bytes = measure_store_bytes(input_path)
                measured = store_bytes / total_stored if store_bytes > 0 else None
                # Mirror `gsplat lod`: an explicit non-default --encoding
                # re-encodes the output, so measured INPUT bytes misstate the
                # on-wire output cost — size against the analytic estimate
                # for the target encoding instead.
                if measured is not None and encoding_mode != "auto":
                    input_encoding = detect_store_encoding(input_path)
                    if input_encoding != encoding_mode:
                        aprint(
                            f"--encoding {encoding_mode} re-encodes the output "
                            f"(input store looks "
                            f"{input_encoding or 'unknown'}-encoded); sizing "
                            f"--target-ms from the analytic {encoding_mode} "
                            f"estimate instead of the measured input bytes"
                        )
                        measured = None
                bp = resolve_streaming_breakpoints(
                    target_ms,
                    bandwidth_mbps,
                    bytes_per_splat,
                    measured_bps=measured,
                    analytic_bps=estimate_bytes_per_splat(
                        node_ndim(node), has_colors, encoding=encoding_mode
                    ),
                )

            with asection(f"Laddering {n_leaves} leaf/leaves"):

                def _ladder_leaf(leaf: "GSplatLeaf") -> "GSplatNode":
                    # Rebuild from the leaf's flattened union (an existing
                    # ladder is discarded and recomputed). Explicit counts are
                    # clamped to THIS leaf's size (parts/levels differ in N).
                    gd = GSplatData.from_tree(leaf)
                    n = gd.n_splats
                    laddered = make_additive_lod(
                        gd,
                        n_lods=max(1, min(eff_n_lods, n)) if n else 1,
                        method=method_norm,  # type: ignore[arg-type]
                        breakpoints=clamp_counts_breakpoints(bp, n),
                    )
                    new_leaf = laddered.tree
                    # Merge meta, keeping the FRESHLY-computed ladder stats
                    # (lod_n_lods/lod_cutpoints/lod_breakpoints_kind) — a blind
                    # `meta=dict(leaf.meta)` would restore the SOURCE leaf's
                    # stale ladder stats when re-laddering. Source-only keys
                    # (e.g. a stamped `coverage_fraction`) are preserved;
                    # per-key, `stats` merges so source-only stat entries
                    # survive but ladder keys take the fresh values.
                    merged = {**leaf.meta, **new_leaf.meta}
                    src_stats = leaf.meta.get("stats")
                    new_stats = new_leaf.meta.get("stats")
                    if isinstance(src_stats, dict) and isinstance(new_stats, dict):
                        merged["stats"] = {**src_stats, **new_stats}
                    return replace(new_leaf, meta=merged)

                result = map_leaves(node, _ladder_leaf)
                ladder_sizes = sorted(
                    {len(lf.additive_sublods) for lf in iter_leaves(result)}
                )
                aprint(f"Ladders built: {ladder_sizes} additive level(s) per leaf")

            with asection(f"Saving to {output_path.name}"):
                if output_path.exists() and overwrite:
                    import shutil

                    if output_path.is_dir():
                        shutil.rmtree(output_path)
                    else:
                        output_path.unlink()
                fitting_info, fitting_config, provenance_info, pipeline_info = (
                    split_fitting_info(stats or {}, include_fitting_info=True)
                )
                write_gsplats_tree(
                    output_path,
                    result,
                    encoding_mode=encoding_mode_obj,
                    compress=compress,
                    fitting_info=fitting_info,
                    fitting_config=fitting_config,
                    provenance_info=provenance_info,
                    pipeline_info=pipeline_info,
                )
                aprint(
                    f"  Saved laddered tree: {output_path} "
                    f"({total_stored:,} splats, {n_leaves} leaf/leaves)"
                )

    except typer.Exit:
        raise
    except Exception as e:
        aprint(f"❌ Error: {e}")
        import traceback

        traceback.print_exc()
        raise typer.Exit(1)
