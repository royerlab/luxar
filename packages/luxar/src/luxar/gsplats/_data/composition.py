"""Dataset-composition mixin for ``GSplatData``.

Combining several datasets (concatenate / new-dimension stacking / per-channel
color merge), promoting one into a higher-dimensional space, and assembling
``kind=partition`` tree nodes.
"""

from __future__ import annotations

import warnings
from typing import TYPE_CHECKING, Any, Dict, List, Literal, Optional, cast

import numpy as np

from .base import _concat_additive_levels, _GSplatDataOps

if TYPE_CHECKING:
    from luxar.gsplats.gsplat_data import GSplatData, SubstitutiveLevel
    from luxar.gsplats.tree import GSplatNode, GSplatPartition


class CompositionMixin(_GSplatDataOps):
    """Merge / partition / embed — the multi-dataset composition family."""

    @classmethod
    def concatenate(cls, datasets: list["GSplatData"]) -> "GSplatData":
        """Concatenate multiple GSplatData objects into one.

        All datasets must share the same dimensionality, truncation radius,
        and number of substitutive levels. The full 2-D LOD matrix is
        preserved: merging is done per ``(substitutive, additive)`` cell, so
        concatenating pyramids yields a pyramid (no level is silently
        dropped). To merge across a mismatched substitutive hierarchy,
        ``flattened()`` the inputs first.

        Colors: if all have colors, concatenate; if all None, None;
        if mixed, fill missing with white (1,1,1).

        Args:
            datasets: List of GSplatData (same ndim, truncation_radius, and
                n_substitutive required).

        Returns:
            New GSplatData with all splats concatenated per LOD cell.

        Raises:
            ValueError: On empty input list, or mismatched ndim /
                truncation_radius / n_substitutive across datasets.
        """
        from luxar.gsplats.gsplat_data import SubstitutiveLevel

        make = cast("type[GSplatData]", cls)

        if len(datasets) == 0:
            raise ValueError("At least one GSplatData is required")

        # Filter out empty datasets to avoid shape mismatch in np.concatenate
        non_empty = [d for d in datasets if d.n_splats > 0]
        if len(non_empty) == 0:
            # All empty: return a fresh empty instance (never alias an input,
            # per the immutability contract).
            from luxar.gsplats.utils.trils import tril_size

            d0 = datasets[0]
            d = d0.ndim
            return make(
                centers=np.empty((0, d), dtype=np.float32),
                amplitudes=np.empty(0, dtype=np.float32),
                cholesky_factors=np.empty(
                    (0, tril_size(d) if d > 0 else 0), dtype=np.float32
                ),
                colors=None,
                stats=dict(d0.stats),
                truncation_radius=d0.truncation_radius,
            )

        ndim = non_empty[0].ndim
        tr = non_empty[0].truncation_radius
        n_sub = non_empty[0].n_substitutive
        for i, ds in enumerate(non_empty[1:], start=1):
            if ds.ndim != ndim:
                raise ValueError(
                    f"Dimensionality mismatch: dataset 0 has {ndim}D, "
                    f"dataset {i} has {ds.ndim}D"
                )
            if ds.truncation_radius != tr:
                raise ValueError(
                    f"Truncation radius mismatch: dataset 0 has {tr}, "
                    f"dataset {i} has {ds.truncation_radius}. "
                    f"Cannot concatenate datasets fitted with different truncation radii."
                )
            if ds.n_substitutive != n_sub:
                raise ValueError(
                    f"Substitutive-level count mismatch: dataset 0 has "
                    f"{n_sub}, dataset {i} has {ds.n_substitutive}. "
                    f"concatenate() requires a uniform substitutive hierarchy; "
                    f"flatten() the inputs first to merge mismatched pyramids."
                )

        merged_stats: Dict[str, Any] = {
            "concatenated_from": len(datasets),
            "splats_per_source": [d.n_splats for d in datasets],
        }
        total_time = sum(d.stats.get("time_seconds", 0) for d in non_empty)
        if total_time > 0:
            merged_stats["time_seconds"] = total_time

        # Multi-substitutive path: merge per (substitutive, additive) cell so
        # the full pyramid survives.
        if n_sub > 1:
            # Read each source's finest-first level list ONCE (the property
            # reconstructs the whole matrix per call); index per level below so
            # the merge stays O(L·D) rather than O(L²·D).
            levels_per_source = [d.substitutive_levels for d in non_empty]
            template = levels_per_source[0]
            sub_levels: List["SubstitutiveLevel"] = []
            for s in range(n_sub):
                views_s = [
                    d._view_of_level(levels_per_source[di][s])
                    for di, d in enumerate(non_empty)
                ]
                ref = template[s]
                sub_levels.append(
                    SubstitutiveLevel(
                        additive_sublods=_concat_additive_levels(views_s),
                        compression_factor=ref.compression_factor,
                        parent_method=ref.parent_method,
                        level_index=ref.level_index,
                        stats={**ref.stats, "n_sources": len(non_empty)},
                    )
                )
            return cls.from_substitutive_levels(
                sub_levels,
                stats=merged_stats,
            )

        # Single substitutive level: merge its additive ladder.
        merged_lods = _concat_additive_levels(non_empty)
        if len(merged_lods) > 1:
            return make(additive_sublods=merged_lods, stats=merged_stats)

        only = merged_lods[0]
        return make(
            centers=only.centers,
            amplitudes=only.amplitudes,
            cholesky_factors=only.cholesky_factors,
            colors=only.colors,
            stats=merged_stats,
            truncation_radius=only.truncation_radius,
        )

    @classmethod
    def combine_as_new_dimension(
        cls,
        datasets: "list[GSplatData]",
        values: "np.ndarray | list[float] | None" = None,
        sigma: float = 0.0,
    ) -> "GSplatData":
        """Combine datasets by embedding each into a new dimension, then concatenating.

        Each dataset is promoted from D-dimensional to (D+1)-dimensional by
        appending a coordinate in the new dimension, then all are concatenated
        into a single dataset.

        This is useful for combining per-timepoint 3D fits into a single 4D
        dataset, per-slice 2D fits into 3D, or any similar stacking operation.

        Args:
            datasets: List of GSplatData, all with the same ndim.
            values: Coordinate for each dataset in the new dimension.
                If None, uses 0.0, 1.0, 2.0, ... (one per dataset).
                If scalar-per-dataset, all splats in that dataset get the same
                coordinate.  Can also be a list of per-splat arrays if different
                splats within a dataset need different coordinates.
            sigma: Standard deviation in the new dimension.
                Use 0.0 for discrete dimensions (e.g., time frames) where
                splats should not extend across the new axis.
                Use a positive value for continuous dimensions where splats
                should have Gaussian extent.

        Returns:
            Single GSplatData with ndim+1 dimensions containing all splats.

        Raises:
            ValueError: If datasets is empty, lengths mismatch, or ndims differ.

        Example:
            >>> # Combine 3D timepoints into 4D
            >>> combined = GSplatData.combine_as_new_dimension(
            ...     [t0_3d, t1_3d, t2_3d], sigma=0.0
            ... )
            >>> combined.ndim  # 4
            >>> combined.n_splats  # sum of all timepoints
        """
        if not datasets:
            raise ValueError("At least one GSplatData is required")

        if values is None:
            values = [float(i) for i in range(len(datasets))]
        elif hasattr(values, "__len__"):
            values = list(values)
        else:
            raise TypeError(
                f"values must be a list/array or None, got {type(values).__name__}"
            )

        if len(values) != len(datasets):
            raise ValueError(
                f"Number of values ({len(values)}) must match "
                f"number of datasets ({len(datasets)})"
            )

        embedded = [
            ds.embed_dimension(val, sigma=sigma) for ds, val in zip(datasets, values)
        ]
        return cls.concatenate(embedded)

    def to_spatial_partition(
        self,
        *,
        max_elements: int,
        rule: Literal["median", "midpoint", "sah"] = "median",
    ) -> "GSplatPartition":
        """Spatially partition the splats into a ``kind=partition`` tree node.

        Recursively BSP-splits the splat **centers** so each part holds at most
        ``max_elements`` splats, using the shared splitters in
        :mod:`luxar.core.group.partition` (the same machinery the scene uses).
        Returns a :class:`~luxar.gsplats.tree.GSplatPartition` (a tree node, not
        a ``GSplatData`` — a partition has no flat-matrix equivalent); write it
        with ``write_gsplats_tree`` (one self-contained ``kind=partition`` file)
        or embed it in a scene. Each part gets its own ``position_bounds`` at
        write time so the viewer can frustum-cull per part.

        A multi-LOD input is flattened to its default substitutive level first
        (BSP partitions a single splat set), matching :meth:`partition`.
        """
        from luxar.core.group.partition import spatial_bsp_tree
        from luxar.gsplats.gsplat_data import AdditiveSubLOD
        from luxar.gsplats.tree import GSplatLeaf, GSplatPartition

        if max_elements < 1:
            raise ValueError(f"max_elements must be >= 1, got {max_elements}")
        if rule not in ("median", "midpoint", "sah"):
            raise ValueError(
                f"rule must be 'median', 'midpoint', or 'sah'; got {rule!r}"
            )

        src: "GSplatData" = cast("GSplatData", self)
        if self.n_substitutive > 1 or self.n_additive_sublods > 1:
            warnings.warn(
                "to_spatial_partition() flattens LOD structure: input has "
                f"n_substitutive={self.n_substitutive}, "
                f"n_additive_sublods={self.n_additive_sublods}; coarser "
                "substitutive levels and the additive ladder are collapsed "
                "into a single level before partitioning.",
                UserWarning,
                stacklevel=2,
            )
            src = self.flattened()

        centers = np.asarray(src.centers)
        # Build the BSP TREE (retains split planes), then read its leaves in
        # left-first DFS order as the parts. The serialized tree rides along on
        # the partition so the viewer can order parts back-to-front exactly
        # (see GSplatPartition.bsp_tree); leaf part index k == child index k.
        tree = spatial_bsp_tree(centers, max_elements, rule=rule)
        parts = [leaf.indices for leaf in tree.leaves()]
        children: List["GSplatNode"] = []
        for idx in parts:
            assert idx is not None
            children.append(
                GSplatLeaf(
                    additive_sublods=[
                        AdditiveSubLOD(
                            centers=src.centers[idx],
                            amplitudes=src.amplitudes[idx],
                            cholesky_factors=src.cholesky_factors[idx],
                            colors=src.colors[idx] if src.colors is not None else None,
                            truncation_radius=src.truncation_radius,
                        )
                    ]
                )
            )
        return GSplatPartition(
            children=children,
            max_elements=max_elements,
            bsp_tree=tree.to_serializable(),
        )

    @staticmethod
    def partition_from_regions(
        regions: "List[GSplatData]",
        *,
        recipe: "Optional[str]" = None,
        recipe_params: "Optional[Any]" = None,
    ) -> "GSplatNode":
        """Assemble a ``kind=partition`` tree from pre-decomposed spatial regions.

        Unlike :meth:`to_spatial_partition` (which BSP-splits a flat splat set),
        this keeps the **given** spatial decomposition: each region becomes one
        partition part, preserving the exact tile/box boundaries the fitter
        already produced. Used by tiled / content-aware fitting, where the
        regions are the per-tile (apodized) or per-box (core-kept) splats — both
        sum correctly as additive partition parts, so the partitioned render
        equals the flat concatenation with no double-count.

        With ``recipe`` (one of
        :data:`~luxar.gsplats.lod.recipes.PER_PART_RECIPES`: ``stream`` →
        ``tiles`` topology, ``levels`` → ``adaptive``) each part is
        given its OWN LOD via :func:`~luxar.gsplats.lod.recipes.build_part_lod`
        (clamped to the part's splat count), so the output is a partition whose
        every child carries a ladder/lod-group — the fit-time equivalent of a
        per-part ``gsplat lod`` pass (which cannot run on a partition). Without a
        recipe each part is a bare leaf (the historical behaviour).

        Empty regions (0 splats) are dropped. With a single non-empty region the
        bare part node is returned (no 1-part partition wrapper); with none,
        raises. Returns a tree node (write with ``write_gsplats_tree`` or embed
        in a scene) — a partition has no flat-matrix ``GSplatData`` equivalent.
        """
        from luxar.gsplats.tree import GSplatPartition

        nonempty = [r for r in regions if r.n_splats > 0]
        if not nonempty:
            raise ValueError("partition_from_regions: all regions are empty")

        def _part_node(region: "GSplatData") -> "GSplatNode":
            if recipe is None:
                return region.tree
            from luxar.gsplats.lod.recipes import RecipeParams, build_part_lod

            params = recipe_params if recipe_params is not None else RecipeParams()
            return build_part_lod(region.tree, recipe, params)

        if len(nonempty) == 1:
            return _part_node(nonempty[0])  # single part -> bare part node
        return GSplatPartition(children=[_part_node(r) for r in nonempty])

    def embed_dimension(
        self,
        values: "np.ndarray | float",
        sigma: float = 0.0,
    ) -> "GSplatData":
        """Add a new dimension to the splat data.

        Appends a column to centers and embeds Cholesky factors into
        the higher-dimensional space.

        Args:
            values: Coordinate for the new dimension. Scalar (same for all)
                or (N,) array (per-splat).
            sigma: Standard deviation in the new dimension (default 0.0
                for discrete dimensions like time).

        Returns:
            New GSplatData with ndim+1 dimensions.

        Example:
            >>> data_4d = data_3d.embed_dimension(5.0, sigma=0.0)
            >>> data_4d = data_3d.embed_dimension(time_values, sigma=0.5)
        """
        from luxar.gsplats.gsplat_data import AdditiveSubLOD, GSplatData
        from luxar.gsplats.utils.trils import embed_cholesky_packed

        # A 0-d numpy array is semantically a scalar; unwrap it so the
        # np.isscalar() branches below treat it as the broadcast coordinate it
        # represents (rather than a malformed per-splat array of shape ()).
        if isinstance(values, np.ndarray) and values.ndim == 0:
            values = values.item()

        n = self.n_splats
        d = self.ndim

        # Multi-substitutive: embed every level and rebuild the pyramid. A scalar
        # coordinate broadcasts cleanly to all levels; a per-splat array is sized
        # to the finest level only and cannot map to coarser levels, so reject it
        # (mirrors with_colors) rather than silently collapsing the ladder. The
        # scalar path is the one the merge pipeline (combine_as_new_dimension)
        # exercises on pyramid inputs.
        if self.n_substitutive > 1:
            if not np.isscalar(values):
                raise ValueError(
                    "embed_dimension with a per-splat values array is not "
                    "supported on a multi-substitutive pyramid (each level has a "
                    "different splat count). Pass a scalar coordinate to broadcast "
                    "across all levels, or operate per level via at_substitutive()."
                )
            return self._map_substitutive(
                lambda lvl: lvl.embed_dimension(values, sigma)
            )

        # Multi-LOD path: embed each LOD independently
        if self.n_additive_sublods > 1:
            is_scalar = np.isscalar(values)
            values_arr: Optional[np.ndarray] = None
            if not is_scalar:
                values_arr = np.asarray(values, dtype=self.centers.dtype)
                if values_arr.shape != (n,):
                    raise ValueError(
                        f"values shape {values_arr.shape} doesn't match splat count ({n},)"
                    )
            dim_mapping = list(range(d))
            fill_sigma = {d: sigma}

            def _embed_lod(
                lod: "AdditiveSubLOD", offset: int, nl: int
            ) -> "AdditiveSubLOD":
                if is_scalar:
                    lod_col = np.full((nl, 1), values, dtype=lod.centers.dtype)
                else:
                    assert values_arr is not None
                    lod_col = values_arr[offset : offset + nl].reshape(nl, 1)
                lod_centers = np.concatenate([lod.centers, lod_col], axis=1)
                lod_cholesky = embed_cholesky_packed(
                    lod.cholesky_factors, d, d + 1, dim_mapping, fill_sigma
                )
                return AdditiveSubLOD(
                    centers=lod_centers,
                    amplitudes=lod.amplitudes,
                    cholesky_factors=lod_cholesky,
                    colors=lod.colors,
                    stats=dict(lod.stats),
                    truncation_radius=lod.truncation_radius,
                )

            return self._map_additive(_embed_lod)

        # Single-LOD fast path (unchanged)
        if np.isscalar(values):
            new_col = np.full((n, 1), values, dtype=self.centers.dtype)
        else:
            values = np.asarray(values, dtype=self.centers.dtype)
            if values.shape != (n,):
                raise ValueError(
                    f"values shape {values.shape} doesn't match splat count ({n},)"
                )
            new_col = values.reshape(n, 1)

        new_centers = np.concatenate([self.centers, new_col], axis=1)
        new_cholesky = embed_cholesky_packed(
            self.cholesky_factors,
            d_src=d,
            d_dst=d + 1,
            dim_mapping=list(range(d)),
            fill_sigma={d: sigma},
        )

        return GSplatData(
            centers=new_centers,
            amplitudes=self.amplitudes,
            cholesky_factors=new_cholesky,
            colors=self.colors,
            stats=dict(self.stats),
            truncation_radius=self.truncation_radius,
        )

    @classmethod
    def merge_with_channel_colors(
        cls,
        gsplats_per_channel: list["GSplatData"],
        channel_colors: list[tuple[float, float, float]],
    ) -> "GSplatData":
        """Merge multiple GSplatData objects, assigning a fixed color per channel.

        This is useful for multi-channel visualization where each channel was
        fitted separately and should be displayed with a distinct color.

        Args:
            gsplats_per_channel: List of GSplatData objects, one per channel.
                All must have the same dimensionality.
            channel_colors: List of RGB color tuples (one per channel).
                Each tuple should have values in [0, 1] range, e.g., (1.0, 0.0, 0.5).

        Returns:
            New GSplatData with all splats merged and colors assigned.

        Raises:
            ValueError: If lists have different lengths or dimensionalities don't match.

        Example:
            >>> # Fit each channel separately
            >>> gsplats_ch0 = fit_gaussian_splats(volume_ch0, ...)
            >>> gsplats_ch1 = fit_gaussian_splats(volume_ch1, ...)
            >>>
            >>> # Merge with magenta for ch0, cyan for ch1
            >>> merged = GSplatData.merge_with_channel_colors(
            ...     [gsplats_ch0, gsplats_ch1],
            ...     channel_colors=[(1.0, 0.0, 0.5), (0.0, 1.0, 0.5)],
            ... )
            >>>
            >>> # Add to scene
            >>> scene.add_gsplats_from_data("multichannel", merged)
        """
        from luxar.gsplats.gsplat_data import AdditiveSubLOD, SubstitutiveLevel

        make = cast("type[GSplatData]", cls)

        if len(gsplats_per_channel) != len(channel_colors):
            raise ValueError(
                f"Number of GSplatData objects ({len(gsplats_per_channel)}) must match "
                f"number of colors ({len(channel_colors)})"
            )

        if len(gsplats_per_channel) == 0:
            raise ValueError("At least one GSplatData object is required")

        # Validate all have same dimensionality and truncation radius
        ndim = gsplats_per_channel[0].ndim
        tr = gsplats_per_channel[0].truncation_radius
        for i, gsplat in enumerate(gsplats_per_channel[1:], start=1):
            if gsplat.ndim != ndim:
                raise ValueError(
                    f"Dimensionality mismatch: channel 0 has {ndim}D, "
                    f"channel {i} has {gsplat.ndim}D"
                )
            if gsplat.truncation_radius != tr:
                raise ValueError(
                    f"Truncation radius mismatch: channel 0 has {tr}, "
                    f"channel {i} has {gsplat.truncation_radius}. "
                    f"Cannot merge datasets fitted with different truncation radii."
                )

        # Merge stats (basic aggregation)
        merged_stats: Dict[str, Any] = {
            "merged_from_channels": len(gsplats_per_channel),
            "splats_per_channel": [len(g.amplitudes) for g in gsplats_per_channel],
        }
        total_time = sum(g.stats.get("time_seconds", 0) for g in gsplats_per_channel)
        if total_time > 0:
            merged_stats["time_seconds"] = total_time

        # Multi-substitutive: merge per substitutive level and rebuild the
        # pyramid (mirrors concatenate / the transform ops), never silently
        # collapsing to the finest level. Reachable via `luxar gsplat merge
        # --channel-colors` on kind=lod inputs. Each level merges the channels
        # that HAVE that level (parallel to the additive max_lods path below).
        max_sub = max(g.n_substitutive for g in gsplats_per_channel)
        if max_sub > 1:
            new_levels: List["SubstitutiveLevel"] = []
            for s in range(max_sub):
                parts = [
                    (g.at_substitutive(s), color)
                    for g, color in zip(gsplats_per_channel, channel_colors)
                    if s < g.n_substitutive
                ]
                merged_level = cls.merge_with_channel_colors(
                    [view for view, _ in parts], [color for _, color in parts]
                )
                template = parts[0][0].substitutive_levels[0]
                new_levels.append(
                    SubstitutiveLevel(
                        additive_sublods=merged_level.substitutive_levels[
                            0
                        ].additive_sublods,
                        compression_factor=template.compression_factor,
                        parent_method=template.parent_method,
                        level_index=template.level_index,
                        stats=dict(template.stats),
                    )
                )
            return cls.from_substitutive_levels(new_levels, stats=merged_stats)

        # Multi-LOD path: per-LOD channel color assignment
        max_lods = max(g.n_additive_sublods for g in gsplats_per_channel)
        if max_lods > 1:
            merged_lods = []
            for level in range(max_lods):
                level_parts = [
                    (g.additive_sublod(level), color)
                    for g, color in zip(gsplats_per_channel, channel_colors)
                    if level < g.n_additive_sublods
                ]
                centers = np.concatenate(
                    [lod.centers for lod, _ in level_parts], axis=0
                )
                amplitudes = np.concatenate([lod.amplitudes for lod, _ in level_parts])
                cholesky = np.concatenate(
                    [lod.cholesky_factors for lod, _ in level_parts], axis=0
                )
                colors = np.concatenate(
                    [
                        np.tile(np.array(c, dtype=np.float32), (lod.n_splats, 1))
                        for lod, c in level_parts
                    ],
                    axis=0,
                )
                merged_lods.append(
                    AdditiveSubLOD(
                        centers=centers,
                        amplitudes=amplitudes,
                        cholesky_factors=cholesky,
                        colors=colors,
                        stats={"lod_level": level, "n_channels": len(level_parts)},
                        truncation_radius=level_parts[0][0].truncation_radius,
                    )
                )
            return make(additive_sublods=merged_lods, stats=merged_stats)

        # Single-LOD fast path (unchanged)
        all_centers = np.concatenate([g.centers for g in gsplats_per_channel], axis=0)
        all_amplitudes = np.concatenate(
            [g.amplitudes for g in gsplats_per_channel], axis=0
        )
        all_cholesky = np.concatenate(
            [g.cholesky_factors for g in gsplats_per_channel], axis=0
        )
        color_arrays = []
        for gsplat, color in zip(gsplats_per_channel, channel_colors):
            channel_color_array = np.tile(
                np.array(color, dtype=np.float32), (gsplat.n_splats, 1)
            )
            color_arrays.append(channel_color_array)
        all_colors = np.concatenate(color_arrays, axis=0)

        return make(
            centers=all_centers,
            amplitudes=all_amplitudes,
            cholesky_factors=all_cholesky,
            colors=all_colors,
            stats=merged_stats,
            truncation_radius=gsplats_per_channel[0].truncation_radius,
        )
