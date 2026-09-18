"""Implementation helper for gsplat transform command."""

from __future__ import annotations

import re
from pathlib import Path
from typing import TYPE_CHECKING, Callable, Literal, Optional

import typer
from arbol import aprint, asection

from ..._traceback import exit_with_error
from ...utils import format_memory_size
from ..encoding import _resolve_encoding_mode
from .parsing import parse_csv_floats

if TYPE_CHECKING:  # pragma: no cover - typing only
    import numpy as np

    from luxar.gsplats.gsplat_data import GSplatData
    from luxar.gsplats.tree import GSplatNode


def _composed_affine(
    ndim: int,
    scale_matrix: "Optional[np.ndarray]",
    rot_matrix: "Optional[np.ndarray]",
    translate_vec: "Optional[np.ndarray]",
    center_shift: "Optional[np.ndarray]",
) -> "tuple[np.ndarray, np.ndarray]":
    """The single affine equivalent to the geometry transforms, in applied order.

    ``gsplat transform`` applies scale, then rotation, then translation, then a
    centroid re-origin — each as its own pass over the leaves. Composing them
    once, here, lets a partition's split planes take the SAME motion in one step
    (``p -> linear @ p + offset``) instead of threading four incremental updates
    through the leaf loop.
    """
    import numpy as np

    linear = np.eye(ndim)
    offset = np.zeros(ndim)
    for matrix in (scale_matrix, rot_matrix):
        if matrix is not None:
            linear = matrix @ linear
            offset = matrix @ offset
    if translate_vec is not None:
        offset = offset + np.asarray(translate_vec, dtype=float)
    if center_shift is not None:
        offset = offset - np.asarray(center_shift, dtype=float)
    return linear, offset


def _map_partition_planes(
    node: "GSplatNode", linear: "np.ndarray", shift: "np.ndarray"
) -> "GSplatNode":
    """Carry every partition's split planes through the composed geometry affine.

    A ``bsp_tree``'s ``split`` is a coordinate in the centers' own space, so a
    transform that moves centers invalidates it. Preserving one verbatim would be
    worse than having none: a stale plane still produces a valid-looking part
    permutation, so the viewer's back-to-front order degrades SILENTLY instead of
    falling back to its documented centroid heuristic.

    Translation, per-axis scale and quarter-turn rotations map cleanly (a mirror
    also swaps each node's halves). An arbitrary rotation shears the cells out of
    axis-alignment, which the serialized format cannot express — that tree is
    dropped, and the resulting ordering downgrade is announced rather than left
    for someone to discover in the viewer.
    """
    from dataclasses import replace

    from luxar.core.group.partition import map_serialized_bsp_tree
    from luxar.gsplats.tree import GSplatLodGroup, GSplatPartition

    dropped = 0

    def walk(current: "GSplatNode") -> "GSplatNode":
        nonlocal dropped
        if isinstance(current, GSplatPartition):
            mapped = map_serialized_bsp_tree(current.bsp_tree, linear, shift)
            if current.bsp_tree is not None and mapped is None:
                dropped += 1
            return replace(
                current,
                children=[walk(c) for c in current.children],
                bsp_tree=mapped,
            )
        if isinstance(current, GSplatLodGroup):
            return replace(current, children=[walk(c) for c in current.children])
        return current

    result = walk(node)
    if dropped:
        aprint(
            f"⚠️ Dropped the split planes of {dropped} partition(s): this transform "
            "does not map axis-aligned cells to axis-aligned cells (an arbitrary "
            "rotation). Parts will be ordered by centroid instead of exactly; "
            "re-partition after rotating to restore exact ordering."
        )
    return result


class _IntensityChangeRecorder:
    """Wrap the tree path's amplitude edits so the ROOT scrub keys off the VALUES.

    An intensity edit rewrites the amplitudes, so the fit's measured reconstruction
    scores no longer describe this artifact (PSNR/MSE are absolute-error metrics —
    a global x0.5 changes them outright, #1600). The per-leaf ops scrub their own
    ``GSplatData``, but the tree path writes the ROOT ``fitting/`` group from the
    ``stats`` dict loaded off disk, so it needs the same pass — otherwise a
    partition would stay exempt from a rule the flat path enforces.

    The predicate has to be the SAME one, though: the ``GSplatData`` methods scrub
    on :func:`~luxar.gsplats.gsplat_data.amplitudes_changed`, so gating this on the
    flag's mere presence made ``transform --scale-intensity 1.0`` destroy a
    partition's scores while the flat path (correctly) kept them — and
    ``--normalize-intensity`` on an all-zero store scrubbed without running a
    single leaf op. So every amplitude edit reports through here, and the root is
    scrubbed only if at least one leaf really moved.

    Geometry-only transforms keep the scores: the splat set is identical and only
    its frame moved (the splats are what the score describes — the format spec's
    reproducibility argument is weaker here, since ``transform`` records no scale
    factor).
    """

    def __init__(self) -> None:
        self.changed = False

    def watching(
        self, op: "Callable[[GSplatData], GSplatData]"
    ) -> "Callable[[GSplatData], GSplatData]":
        """``op``, with "did the amplitudes actually move" recorded on the side."""
        from luxar.gsplats.gsplat_data import amplitudes_changed

        def _fn(data: "GSplatData") -> "GSplatData":
            out = op(data)
            if amplitudes_changed(data.amplitudes, out.amplitudes):
                self.changed = True
            return out

        return _fn

    def scrub_root(self, stats: "Optional[dict]") -> None:
        """Drop the measured scores from the root ``stats``, if anything moved."""
        from luxar.gsplats.gsplat_data import drop_content_scoped_stats

        if self.changed and stats:
            drop_content_scoped_stats(stats)


def _restamp_tree_after_intensity(
    node: "GSplatNode", source: "GSplatNode", *, changed: bool
) -> "GSplatNode":
    if not changed:
        return node
    from luxar.gsplats.lod.restamp import refresh_reduction_lod_tree

    return refresh_reduction_lod_tree(node, source)


def run_transform_dataset(
    *,
    input_path: Path,
    output_path: Path,
    scale_factors: Optional[str],
    translate_offset: Optional[str],
    rotate_x_deg: Optional[float],
    rotate_y_deg: Optional[float],
    rotate_z_deg: Optional[float],
    spatial_dims: Optional[str] = None,
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

        # ── Validate the --spatial-dims FORMAT before loading anything: a typo
        # should not cost a multi-GB store load. Only the range check (needs
        # the dataset's d) stays after the load. ──
        has_rotation = any(
            r is not None for r in [rotate_x_deg, rotate_y_deg, rotate_z_deg]
        )
        if spatial_dims is not None and not has_rotation:
            aprint(
                "❌ --spatial-dims only affects --rotate-x/--rotate-y/"
                "--rotate-z; add a rotation or remove --spatial-dims."
            )
            raise typer.Exit(1)
        explicit_rot_axes: Optional[list[int]] = None
        if spatial_dims is not None:
            # No empty-token filtering: '0,,1,2' and '0,1,2,' are rejected
            # (the empty token is not an integer) rather than silently fixed.
            tokens = [t.strip() for t in spatial_dims.split(",")]
            explicit_rot_axes = []
            for token in tokens:
                try:
                    # The regex gate rejects the syntax; int() can still refuse
                    # an absurdly long digit run (CPython's int-str conversion
                    # limit, sys.get_int_max_str_digits) — same clean error.
                    if re.fullmatch(r"[0-9]+", token) is None:
                        raise ValueError(token)
                    explicit_rot_axes.append(int(token))
                except ValueError:
                    aprint(
                        f"❌ Invalid --spatial-dims '{spatial_dims}': "
                        f"'{token}' is not a valid axis index "
                        f"(use unsigned ASCII digits 0-9)"
                    )
                    raise typer.Exit(1) from None
            if len(explicit_rot_axes) != 3:
                aprint(
                    f"❌ --spatial-dims must list exactly 3 axis indices, "
                    f"got {len(explicit_rot_axes)}: '{spatial_dims}'"
                )
                raise typer.Exit(1)
            if len(set(explicit_rot_axes)) != 3:
                aprint(f"❌ --spatial-dims axes must be distinct, got '{spatial_dims}'")
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
            if has_rotation:
                # nD convention: by default the FIRST 3 center columns are
                # spatial (X/Y/Z) — matching the partitioner
                # (positions[:, :3]) and the stacking convention
                # (embed_dimension / merge --as-dimension append time/channel
                # LAST); every other dim is left unrotated. --spatial-dims
                # overrides which three center dims the rotation acts on
                # (e.g. a direct nD fit whose leading axis is time); its
                # listed order assigns the rotation frame's X/Y/Z roles.
                if d < 3:
                    aprint(
                        f"❌ Rotation requires at least 3 spatial dimensions, got {d}D data"
                    )
                    raise typer.Exit(1)
                if explicit_rot_axes is not None:
                    rot_axes = explicit_rot_axes
                    out_of_range = [a for a in rot_axes if not 0 <= a < d]
                    if out_of_range:
                        aprint(
                            f"❌ --spatial-dims axis {out_of_range[0]} is out of "
                            f"range for {d}D data (valid indices: 0..{d - 1})"
                        )
                        raise typer.Exit(1)
                else:
                    rot_axes = [0, 1, 2]
                    if d > 3:
                        # The default is a guess on >3D data — say exactly what
                        # is rotated and how to override it. When more than
                        # three axes carry real extent this likely is a direct
                        # nD fit whose dims 0,1,2 need not be spatial, so warn
                        # more sharply (but keep it a warning: a legitimate
                        # stacked dataset may have a continuous stacked axis).
                        unrotated = [i for i in range(d) if i not in rot_axes]
                        # fallback=False so all-degenerate data (no axis above
                        # eps) reports 0 real-extent axes and takes the milder
                        # branch, instead of the all-axes fallback masquerading
                        # as "every axis carries extent — a direct nD fit".
                        n_extent = len(nondegenerate_axes(node, fallback=False))
                        if n_extent > 3:
                            aprint(
                                f"⚠️ {d}D data has {n_extent} axes with real "
                                f"extent — this looks like a direct nD fit, "
                                f"where dims 0, 1, 2 may not be the spatial "
                                f"ones. Defaulting to rotating dims 0, 1, 2 "
                                f"and leaving dims {unrotated} unrotated; "
                                f"pass --spatial-dims i,j,k to pick the three "
                                f"spatial axes explicitly."
                            )
                        else:
                            aprint(
                                f"⚠️ {d}D data: rotating dims 0, 1, 2 (assumed "
                                f"spatial X/Y/Z) and leaving dims {unrotated} "
                                f"unrotated — pass --spatial-dims to override."
                            )
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
                axes_arr = np.asarray(rot_axes, dtype=int)
                rot_matrix[np.ix_(axes_arr, axes_arr)] = rot3

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

                from luxar.gsplats.tree import GSplatLeaf

                # Centroid shift applied by --center, captured for the split-plane
                # remap below (it is only known once the centroid is measured).
                center_shift: "Optional[np.ndarray]" = None
                intensity = _IntensityChangeRecorder()
                source_node = node

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
                            center_shift = np.asarray(shift, dtype=float)
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
                                intensity.watching(
                                    lambda gd: gd.scale_intensity(
                                        scale_intensity_factor
                                    )
                                )
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
                                node,
                                _leaf_op(
                                    intensity.watching(
                                        lambda gd: gd.scale_intensity(factor)
                                    )
                                ),
                            )
                intensity.scrub_root(stats)
                node = _restamp_tree_after_intensity(
                    node, source_node, changed=intensity.changed
                )
                # Scrub the coverage_fraction LOD-switch threshold from EVERY node
                # (leaves AND group nodes — an overview partition child, an adaptive
                # per-part lod group) after a geometry transform so the writer
                # re-derives it. coverage_fraction is derived from the ladder's LENGTH
                # and its topology, not from geometry, hence invariant to
                # scale/rotate/translate/center — so this re-derives the identical
                # value; it is kept as a safety net for transforms that also
                # re-ladder and change the number of levels. Intensity-only transforms
                # leave it intact regardless. The re-derivation is TOPOLOGY-AWARE
                # (``write_gsplat_node`` picks ``partitioned_coverage_fractions`` for a
                # partition-bound ladder), so scrubbing an ``adaptive``/``overview``
                # store restores its fills-screen anchor rather than downgrading it to
                # the whole-object one.
                geometry_changed = (
                    scale_matrix is not None
                    or rot_matrix is not None
                    or translate_vec is not None
                    or center
                )
                from luxar.gsplats.tree import without_meta_key

                node = without_meta_key(node, "dimension_metadata")
                if geometry_changed:
                    node = without_meta_key(node, "coverage_fraction")
                    # The selector names the UNITS of the thresholds just
                    # scrubbed, so it goes with them: the writer re-derives in
                    # screen-area units and re-stamps that mode. Keeping a
                    # legacy store's "coverage" stamp here would mislabel the
                    # fresh area values.
                    node = without_meta_key(node, "selector")
                    linear, offset = _composed_affine(
                        d, scale_matrix, rot_matrix, translate_vec, center_shift
                    )
                    node = _map_partition_planes(node, linear, offset)
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
            from luxar.gsplats.io.load_gsplats import read_authored_appearance

            with asection(f"Saving to {output_path.name}"):
                if matrix_shaped:
                    data.save(
                        output_path,
                        encoding_mode=encoding_mode_obj,
                        amplitude_bits="auto",
                        include_fitting_info=True,
                        compress=compress,
                        root_attrs=read_authored_appearance(input_path),
                    )
                else:
                    fitting_info, fitting_config, provenance_info, pipeline_info = (
                        split_fitting_info(stats or {}, include_fitting_info=True)
                    )
                    write_gsplats_tree(
                        output_path,
                        result_node,
                        encoding_mode=encoding_mode_obj,
                        amplitude_bits="auto",
                        source_dtype=(stats or {}).get("source_dtype"),
                        compress=compress,
                        fitting_info=fitting_info,
                        fitting_config=fitting_config,
                        provenance_info=provenance_info,
                        pipeline_info=pipeline_info,
                        root_attrs=read_authored_appearance(input_path),
                    )
                aprint(f"Saved: {output_path}")

                if output_path.exists():
                    aprint(f"  Size: {format_memory_size(output_path.stat().st_size)}")

    except typer.Exit:
        raise
    except Exception as e:
        exit_with_error(f"❌ Error: {e}", e)
