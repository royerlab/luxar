"""Dataset-composition mixin for ``GSplatData``.

Combining several datasets (concatenate / new-dimension stacking / per-channel
color merge), promoting one into a higher-dimensional space, and assembling
``kind=partition`` tree nodes.
"""

from __future__ import annotations

import warnings
from copy import deepcopy
from typing import TYPE_CHECKING, Any, Dict, List, Literal, Optional, Sequence, cast

import numpy as np

from .base import _concat_additive_levels, _GSplatDataOps
from .filtering import _refresh_reduction_lod_stats_if_needed

if TYPE_CHECKING:
    from luxar.gsplats.gsplat_data import GSplatData, SubstitutiveLevel
    from luxar.gsplats.tree import GSplatNode, GSplatPartition


def _concatenate_label_channel(items: Sequence[Any]) -> tuple[Any, Any]:
    """Concatenate compatible categorical channels, or fail loudly."""
    presence = {item.label_ids is not None for item in items}
    if len(presence) > 1:
        raise ValueError(
            "cannot merge label_ids when only some inputs carry the channel"
        )
    if not presence or presence == {False}:
        return None, None
    vocabularies = [item.label_vocabulary for item in items]
    if any(vocabulary != vocabularies[0] for vocabulary in vocabularies[1:]):
        raise ValueError(
            "cannot merge label_ids with different label_vocabulary values"
        )
    return np.concatenate([item.label_ids for item in items]), vocabularies[0]


def _aggregate_part_source_stats(
    part_provenance: Sequence[Dict[str, Any]],
) -> Dict[str, Any]:
    """Source fields that describe the whole stack without averaging quality."""
    fittings = [part["fitting"] for part in part_provenance]
    aggregate: Dict[str, Any] = {}

    shapes = [fit.get("source_shape") for fit in fittings]
    if shapes and all(
        shape == shapes[0] and isinstance(shape, list) for shape in shapes
    ):
        aggregate["source_shape"] = [len(shapes), *shapes[0]]

    dtypes = [fit.get("source_dtype") for fit in fittings]
    if (
        dtypes
        and isinstance(dtypes[0], str)
        and all(dtype == dtypes[0] for dtype in dtypes)
    ):
        aggregate["source_dtype"] = dtypes[0]

    declared = [fit.get("source_declared") for fit in fittings]
    if (
        "source_shape" in aggregate
        and declared
        and isinstance(declared[0], bool)
        and all(value == declared[0] for value in declared)
    ):
        aggregate["source_declared"] = declared[0]

    if "source_shape" in aggregate:
        for key in ("source_voxels", "source_bytes", "source_stored_bytes"):
            values = [fit.get(key) for fit in fittings]
            if values and all(
                isinstance(value, int) and not isinstance(value, bool)
                for value in values
            ):
                aggregate[key] = sum(values)
    return aggregate


def _elide_repeated_part_source_stats(
    part_provenance: list[Dict[str, Any]], aggregate: Dict[str, Any]
) -> None:
    """Remove component source fields recoverable from a unanimous stack root."""
    fittings = [part["fitting"] for part in part_provenance]
    for key in ("source_shape", "source_dtype", "source_declared"):
        if key in aggregate:
            for fitting in fittings:
                fitting.pop(key, None)
    for key in ("source_voxels", "source_bytes", "source_stored_bytes"):
        values = [fitting.get(key) for fitting in fittings]
        if key in aggregate and values and all(value == values[0] for value in values):
            for fitting in fittings:
                fitting.pop(key, None)


def _validated_part_provenance(
    part_provenance: Optional[Sequence[Dict[str, Any]]],
    datasets: Sequence["GSplatData"],
    values: Sequence[float | np.ndarray],
) -> Optional[list[Dict[str, Any]]]:
    """Copy and validate caller-supplied records against the stacked parts."""
    if part_provenance is None:
        return None
    if len(part_provenance) != len(datasets):
        raise ValueError(
            f"part_provenance has {len(part_provenance)} entries for "
            f"{len(datasets)} datasets"
        )
    records = deepcopy(list(part_provenance))
    for index, (record, value) in enumerate(zip(records, values)):
        if not isinstance(record, dict):
            raise TypeError(f"part_provenance[{index}] must be a dict")
        if isinstance(value, np.ndarray) or record.get("coordinate") != value:
            raise ValueError(
                f"part_provenance[{index}].coordinate must equal the scalar "
                f"values[{index}]"
            )
        if not isinstance(record.get("fitting"), dict):
            raise ValueError(f"part_provenance[{index}].fitting must be a dictionary")
    return records


class CompositionMixin(_GSplatDataOps):
    """Merge / partition / embed — the multi-dataset composition family."""

    @classmethod
    def from_default_selection(
        cls,
        node: "GSplatNode",
        *,
        stats: Optional[Dict[str, Any]] = None,
    ) -> "GSplatData":
        """Materialize the tree selection rendered by default.

        Matrix-shaped nodes are preserved verbatim, including their additive
        and substitutive ladders. Nested trees become one flat dataset containing
        every partition part and only each LOD group's default (finest) child.
        Call ``flattened()`` on the result when the caller requires one rung.
        """
        from luxar.gsplats.tree import is_matrix_shaped, iter_default_leaves

        make = cast("type[GSplatData]", cls)
        if is_matrix_shaped(node):
            return make.from_tree(node, stats=stats)

        parts = [make.from_tree(leaf).flattened() for leaf in iter_default_leaves(node)]
        if not parts:
            raise ValueError("GSplat tree has no default-rendered leaves")
        flat = make.concatenate(parts)
        return make.from_additive_sublods(list(flat.additive_sublods), stats=stats)

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
            from luxar.gsplats.io.save_gsplats import agreed_normalization_stats
            from luxar.gsplats.utils.trils import tril_size

            d0 = datasets[0]
            d = d0.ndim
            # The normalization block obeys the same unanimity rule here as on
            # the non-empty path below (#1175): copying d0's stats wholesale
            # would promote the FIRST input's pedestal onto a merge whose other
            # inputs may have removed a different one.
            empty_stats: Dict[str, Any] = {
                "concatenated_from": len(datasets),
                "splats_per_source": [0 for _ in datasets],
            }
            empty_stats.update(agreed_normalization_stats([x.stats for x in datasets]))
            return make(
                centers=np.empty((0, d), dtype=np.float32),
                amplitudes=np.empty(0, dtype=np.float32),
                cholesky_factors=np.empty(
                    (0, tril_size(d) if d > 0 else 0), dtype=np.float32
                ),
                colors=None,
                stats=empty_stats,
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
                    f"call flattened() on the inputs first to merge mismatched "
                    f"pyramids."
                )

        merged_stats: Dict[str, Any] = {
            "concatenated_from": len(datasets),
            "splats_per_source": [d.n_splats for d in datasets],
        }
        # Normalization provenance (#1175). A fresh stats dict used to drop the
        # background level every input had removed, so a tiled merge shipped no
        # record of its own pedestal. Carried only when every input that records
        # a key AGREES on it — see `agreed_normalization_stats`; concatenating
        # two unrelated fits legitimately has no single answer, and the merged
        # result then says nothing rather than claiming the first input's.
        from luxar.gsplats.io.save_gsplats import agreed_normalization_stats

        merged_stats.update(agreed_normalization_stats([d.stats for d in datasets]))
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
            label_ids=only.label_ids,
            label_vocabulary=only.label_vocabulary,
            stats=merged_stats,
            truncation_radius=only.truncation_radius,
        )

    @classmethod
    def combine_as_new_dimension(
        cls,
        datasets: "list[GSplatData]",
        values: "np.ndarray | Sequence[float | np.ndarray] | None" = None,
        sigma: float = 0.0,
        *,
        part_provenance: Optional[Sequence[Dict[str, Any]]] = None,
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
            part_provenance: Optional caller-supplied component-fit records, one
                entry per dataset in the same order as ``values``. This requires
                one scalar coordinate per dataset.

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

        safe_part_provenance = _validated_part_provenance(
            part_provenance, datasets, values
        )

        embedded = [
            ds.embed_dimension(val, sigma=sigma) for ds, val in zip(datasets, values)
        ]
        combined = cls.concatenate(embedded)
        if safe_part_provenance is not None:
            aggregate = _aggregate_part_source_stats(safe_part_provenance)
            _elide_repeated_part_source_stats(safe_part_provenance, aggregate)
            combined.stats["part_provenance"] = safe_part_provenance
            combined.stats.update(aggregate)
        return combined

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
                            label_ids=(
                                src.label_ids[idx]
                                if src.label_ids is not None
                                else None
                            ),
                            label_vocabulary=src.label_vocabulary,
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
        bsp_tree: "Optional[Dict[str, Any]]" = None,
        region_labels: "Optional[Sequence[int]]" = None,
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

        ``bsp_tree`` is the decomposition's serialized split planes — the
        producer's, since this method is handed a decomposition rather than
        computing one (contrast :meth:`to_spatial_partition`, which splits and so
        knows its own planes). Supplying it is what lets the viewer order the
        parts back-to-front EXACTLY instead of guessing from part centroids, which
        is not a valid painter's order and pops at the seams as the camera orbits
        (#1555). Its leaf labels are read in ``region_labels`` space (default: the
        positions of ``regions``), and it is pruned to the regions that survived
        the empty filter — so a caller passes the labels of the regions it is
        handing over and does not have to pre-compensate for drops itself.
        """
        from luxar.core.group.partition import prune_serialized_bsp_tree
        from luxar.gsplats.tree import GSplatPartition

        labels = (
            list(range(len(regions)))
            if region_labels is None
            else [int(label) for label in region_labels]
        )
        if len(labels) != len(regions):
            raise ValueError(
                f"partition_from_regions: region_labels has {len(labels)} entries "
                f"for {len(regions)} regions"
            )
        if any(b <= a for a, b in zip(labels, labels[1:])):
            # Children are written in the order given, but the tree's leaves are
            # renumbered by ASCENDING label — the two only agree when the labels
            # ascend. Out of order (or duplicated) they would silently attach each
            # leaf to the wrong part.
            raise ValueError(
                "partition_from_regions: region_labels must be strictly "
                f"increasing (children keep the order given, while the split-plane "
                f"tree is renumbered by ascending label); got {labels}"
            )
        kept = [(label, r) for label, r in zip(labels, regions) if r.n_splats > 0]
        if not kept:
            raise ValueError("partition_from_regions: all regions are empty")

        # A per-part volume re-fit crops the volume to the part's own tile, so it
        # needs the tile. The split planes carried alongside these regions are
        # exactly that, keyed by the SAME labels, so read the cells once here
        # rather than making every caller reconstruct the decomposition.
        cells: "Dict[int, List[Any]]" = {}
        if bsp_tree and recipe is not None and regions:
            from luxar.core.group.partition import serialized_bsp_leaf_cells

            try:
                cells = serialized_bsp_leaf_cells(bsp_tree, regions[0].ndim)
            except (KeyError, TypeError, ValueError):
                cells = {}

        def _part_node(label: int, region: "GSplatData") -> "GSplatNode":
            if recipe is None:
                return region.tree
            from dataclasses import replace

            from luxar.gsplats.fit_basis import fit_image_min
            from luxar.gsplats.lod.recipes import RecipeParams, build_part_lod

            params = recipe_params if recipe_params is not None else RecipeParams()
            if params.image_min is None:
                params = replace(params, image_min=fit_image_min(region.stats))
            return build_part_lod(
                region.tree, recipe, params, cell=cells.get(int(label))
            )

        if len(kept) == 1:
            # Single part -> bare part node. Nothing to order, so the tree (which
            # would prune to a lone leaf) is deliberately dropped with the wrapper.
            return _part_node(*kept[0])
        return GSplatPartition(
            children=[_part_node(label, r) for label, r in kept],
            bsp_tree=prune_serialized_bsp_tree(bsp_tree, [label for label, _ in kept]),
        )

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
        from luxar.gsplats.gsplat_data import AdditiveSubLOD
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
            scalar_value = cast(float, values)
            out = self._map_substitutive(
                lambda lvl: lvl.embed_dimension(scalar_value, sigma)
            )
            return _refresh_reduction_lod_stats_if_needed(out, self)

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

        def _embed_lod(lod: "AdditiveSubLOD", offset: int, nl: int) -> "AdditiveSubLOD":
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
                label_ids=lod.label_ids,
                label_vocabulary=lod.label_vocabulary,
                stats=dict(lod.stats),
                truncation_radius=lod.truncation_radius,
            )

        out = self._map_additive(_embed_lod)
        return _refresh_reduction_lod_stats_if_needed(out, self)

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
                label_ids, label_vocabulary = _concatenate_label_channel(
                    [lod for lod, _ in level_parts]
                )
                merged_lods.append(
                    AdditiveSubLOD(
                        centers=centers,
                        amplitudes=amplitudes,
                        cholesky_factors=cholesky,
                        colors=colors,
                        label_ids=label_ids,
                        label_vocabulary=label_vocabulary,
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
        all_label_ids, label_vocabulary = _concatenate_label_channel(
            gsplats_per_channel
        )

        return make(
            centers=all_centers,
            amplitudes=all_amplitudes,
            cholesky_factors=all_cholesky,
            colors=all_colors,
            label_ids=all_label_ids,
            label_vocabulary=label_vocabulary,
            stats=merged_stats,
            truncation_radius=gsplats_per_channel[0].truncation_radius,
        )
