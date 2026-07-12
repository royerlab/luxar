"""Implementation helper for gsplat transform command."""

from __future__ import annotations

from pathlib import Path
from typing import Callable, Literal, Optional

import typer
from arbol import aprint, asection

from ..utils import format_memory_size
from .encoding import _resolve_encoding_mode
from .transforms_parsing import parse_csv_floats


def run_transform_dataset(
    *,
    input_path: Path,
    output_path: Path,
    scale_factors: Optional[str],
    translate_offset: Optional[str],
    rotate_x_deg: Optional[float],
    rotate_y_deg: Optional[float],
    rotate_z_deg: Optional[float],
    center: bool,
    scale_intensity_factor: Optional[float],
    normalize_intensity: Optional[float],
    encoding_mode: Literal["auto", "precision", "memory"],
    compress: Optional[Literal["zip", "tar.gz"]],
) -> None:
    """Run transform command implementation."""
    try:
        from dataclasses import replace

        import numpy as np

        from luxar.gsplats.gsplat_data import GSplatData
        from luxar.gsplats.io.load_gsplats import load_gsplat_node
        from luxar.gsplats.io.save_gsplats import split_fitting_info, write_gsplats_tree
        from luxar.gsplats.tree import (
            amplitude_weighted_centroid,
            center_bounds,
            global_amplitude_max,
            is_matrix_shaped,
            map_leaves,
            node_ndim,
            nondegenerate_axes,
            total_splats,
        )
        from luxar.gsplats.utils.spatial_axes import spatial_only_shift

        encoding_mode_obj = _resolve_encoding_mode(encoding_mode)

        # Check that at least one transform is requested
        has_transform = any(
            [
                scale_factors,
                translate_offset,
                rotate_x_deg is not None,
                rotate_y_deg is not None,
                rotate_z_deg is not None,
                center,
                scale_intensity_factor is not None,
                normalize_intensity is not None,
            ]
        )
        if not has_transform:
            aprint("❌ No transforms specified. Use --help to see available options.")
            raise typer.Exit(1)

        with asection(f"Transforming: {input_path.name}"):
            # Load the raw node tree so partitions / nested trees are preserved.
            # A matrix-shaped tree (a leaf, or a lod group of leaves) flattens to a
            # GSplatData exactly as `GSplatData.load` would; a kind=partition (or
            # otherwise nested) tree is transformed leaf-by-leaf, keeping its shape.
            with asection("Loading dataset"):
                node, stats = load_gsplat_node(input_path, include_stats=True)
                d = node_ndim(node)
                matrix_shaped = is_matrix_shaped(node)
                shape_desc = "leaf/matrix" if matrix_shaped else type(node).__name__
                aprint(f"Loaded {total_splats(node):,} splats ({d}D, {shape_desc})")

            transforms_applied: list[str] = []

            # ── Parse the geometry transforms once (shared by both code paths) ──
            scale_matrix = None
            if scale_factors is not None:
                factors = parse_csv_floats(scale_factors, d, "scale")
                scale_matrix = np.diag(factors)
                transforms_applied.append(f"scale({scale_factors})")

            rot_matrix = None
            has_rotation = any(
                r is not None for r in [rotate_x_deg, rotate_y_deg, rotate_z_deg]
            )
            if has_rotation:
                # nD convention: the last 3 dims are spatial (XYZ); any preceding
                # dims (e.g. time) are left unrotated.
                if d < 3:
                    aprint(
                        f"❌ Rotation requires at least 3 spatial dimensions, got {d}D data"
                    )
                    raise typer.Exit(1)
                rot3 = np.eye(3, dtype=np.float64)
                if rotate_x_deg is not None:
                    rad = np.radians(rotate_x_deg)
                    c, s = np.cos(rad), np.sin(rad)
                    rot3 = np.array([[1, 0, 0], [0, c, -s], [0, s, c]]) @ rot3
                    transforms_applied.append(f"rotate_x({rotate_x_deg}°)")
                if rotate_y_deg is not None:
                    rad = np.radians(rotate_y_deg)
                    c, s = np.cos(rad), np.sin(rad)
                    rot3 = np.array([[c, 0, s], [0, 1, 0], [-s, 0, c]]) @ rot3
                    transforms_applied.append(f"rotate_y({rotate_y_deg}°)")
                if rotate_z_deg is not None:
                    rad = np.radians(rotate_z_deg)
                    c, s = np.cos(rad), np.sin(rad)
                    rot3 = np.array([[c, -s, 0], [s, c, 0], [0, 0, 1]]) @ rot3
                    transforms_applied.append(f"rotate_z({rotate_z_deg}°)")
                rot_matrix = np.eye(d, dtype=np.float64)
                rot_matrix[d - 3 :, d - 3 :] = rot3

            translate_vec = None
            if translate_offset is not None:
                offsets = parse_csv_floats(translate_offset, d, "translate")
                translate_vec = np.array(offsets, dtype=np.float64)
                transforms_applied.append(f"translate({translate_offset})")

            if center:
                transforms_applied.append("center")
            if scale_intensity_factor is not None:
                transforms_applied.append(f"scale_intensity({scale_intensity_factor})")
            if normalize_intensity is not None:
                transforms_applied.append(f"normalize({normalize_intensity})")

            if matrix_shaped:
                # ── Flat path: a leaf / matrix tree → the GSplatData methods ──
                data = GSplatData.from_tree(node, stats=stats)
                if scale_matrix is not None:
                    with asection("Applying scale"):
                        aprint(f"Scale factors: {list(np.diagonal(scale_matrix))}")
                        data = data.transform(scale_matrix)
                if rot_matrix is not None:
                    with asection("Applying rotation"):
                        data = data.transform(rot_matrix)
                if translate_vec is not None:
                    with asection("Applying translation"):
                        aprint(f"Translation: {list(translate_vec)}")
                        data = data.translate(translate_vec)
                if center:
                    with asection("Centering at centroid"):
                        data = data.center_at_centroid()
                        aprint("Centered at amplitude-weighted centroid")
                if scale_intensity_factor is not None:
                    with asection("Scaling intensity"):
                        aprint(f"Intensity scale factor: {scale_intensity_factor}")
                        data = data.scale_intensity(scale_intensity_factor)
                if normalize_intensity is not None:
                    with asection("Normalizing intensity"):
                        current_max = float(data.amplitudes.max())
                        aprint(
                            f"Current max: {current_max:.4f} → "
                            f"target max: {normalize_intensity}"
                        )
                        data = data.normalize_intensity(normalize_intensity)
                result_node = data.tree
            else:
                # ── Tree-walking path: kind=partition / nested (structure kept) ──
                aprint(
                    "Tree-structured input — transforming each part in place "
                    "(structure preserved)."
                )

                from luxar.gsplats.tree import GSplatLeaf, GSplatNode

                def _leaf_op(
                    op: "Callable[[GSplatData], GSplatData]",
                ) -> "Callable[[GSplatLeaf], GSplatNode]":
                    def _fn(leaf: "GSplatLeaf") -> "GSplatNode":
                        # GSplatData.transform/translate/... rebuild a fresh leaf with
                        # empty meta; restore the source leaf's provenance verbatim.
                        # The coverage_fraction threshold is scrubbed AFTER all
                        # transforms (from leaf AND group nodes) — see below.
                        new_leaf = op(GSplatData.from_tree(leaf)).tree
                        return replace(new_leaf, meta=dict(leaf.meta))

                    return _fn

                if scale_matrix is not None:
                    with asection("Applying scale"):
                        aprint(f"Scale factors: {list(np.diagonal(scale_matrix))}")
                        node = map_leaves(
                            node, _leaf_op(lambda gd: gd.transform(scale_matrix))
                        )
                if rot_matrix is not None:
                    with asection("Applying rotation"):
                        node = map_leaves(
                            node, _leaf_op(lambda gd: gd.transform(rot_matrix))
                        )
                if translate_vec is not None:
                    with asection("Applying translation"):
                        aprint(f"Translation: {list(translate_vec)}")
                        node = map_leaves(
                            node, _leaf_op(lambda gd: gd.translate(translate_vec))
                        )
                if center:
                    with asection("Centering at centroid"):
                        centroid = amplitude_weighted_centroid(node)
                        if centroid is not None:
                            # Re-origin only the spatial axes; leave a categorical
                            # (zero-variance) time/channel axis in place — matching
                            # GSplatData.center_at_centroid.
                            shift = spatial_only_shift(
                                centroid, nondegenerate_axes(node)
                            )
                            node = map_leaves(
                                node, _leaf_op(lambda gd: gd.translate(-shift))
                            )
                            aprint(
                                "Centered spatial axes at the global "
                                "amplitude-weighted centroid"
                            )
                if scale_intensity_factor is not None:
                    with asection("Scaling intensity"):
                        aprint(f"Intensity scale factor: {scale_intensity_factor}")
                        node = map_leaves(
                            node,
                            _leaf_op(
                                lambda gd: gd.scale_intensity(scale_intensity_factor)
                            ),
                        )
                if normalize_intensity is not None:
                    with asection("Normalizing intensity"):
                        current_max = global_amplitude_max(node)
                        aprint(
                            f"Current global max: {current_max:.4f} → "
                            f"target max: {normalize_intensity}"
                        )
                        if current_max > 0:
                            factor = normalize_intensity / current_max
                            node = map_leaves(
                                node, _leaf_op(lambda gd: gd.scale_intensity(factor))
                            )
                # Scrub the coverage_fraction LOD-switch threshold from EVERY node
                # (leaves AND group nodes — a multiscale partition child, a mosaic
                # per-part lod group) after a geometry transform so the writer
                # re-derives it. coverage_fraction is a per-level COUNT ratio, hence
                # invariant to scale/rotate/translate/center — so this re-derives the
                # identical value; it is kept as a safety net for transforms that also
                # re-ladder and change per-level counts. Intensity-only transforms
                # leave it intact regardless.
                geometry_changed = (
                    scale_matrix is not None
                    or rot_matrix is not None
                    or translate_vec is not None
                    or center
                )
                if geometry_changed:
                    from luxar.gsplats.tree import without_meta_key

                    node = without_meta_key(node, "coverage_fraction")
                result_node = node

            # Summary
            aprint(f"\nTransforms applied: {' → '.join(transforms_applied)}")

            # Print new bounding box (from the result tree's center bounds)
            with asection("Result bounding box"):
                bounds = center_bounds(result_node)
                if bounds is not None:
                    lo_all, hi_all = bounds
                    for i in range(d):
                        lo, hi = float(lo_all[i]), float(hi_all[i])
                        aprint(f"  Dim {i}: [{lo:.4f}, {hi:.4f}]  range: {hi - lo:.4f}")

            # Save (color SDR/HDR is auto-detected by the writer). The flat path
            # keeps GSplatData.save (carries fitting provenance, unchanged); the
            # tree path uses the v3.0 tree writer so a partition stays a partition
            # on disk — threading the loaded stats through split_fitting_info so
            # the fitting/ / provenance/ / pipeline/ groups round-trip exactly
            # like the flat path (they used to be silently stripped here).
            with asection(f"Saving to {output_path.name}"):
                if matrix_shaped:
                    data.save(
                        output_path,
                        encoding_mode=encoding_mode_obj,
                        include_fitting_info=True,
                        compress=compress,
                    )
                else:
                    fitting_info, fitting_config, provenance_info, pipeline_info = (
                        split_fitting_info(stats or {}, include_fitting_info=True)
                    )
                    write_gsplats_tree(
                        output_path,
                        result_node,
                        encoding_mode=encoding_mode_obj,
                        compress=compress,
                        fitting_info=fitting_info,
                        fitting_config=fitting_config,
                        provenance_info=provenance_info,
                        pipeline_info=pipeline_info,
                    )
                aprint(f"Saved: {output_path}")

                if output_path.exists():
                    aprint(f"  Size: {format_memory_size(output_path.stat().st_size)}")

    except typer.Exit:
        raise
    except Exception as e:
        aprint(f"❌ Error: {e}")
        import traceback

        traceback.print_exc()
        raise typer.Exit(1)
