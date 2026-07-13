"""Disk I/O adapter mixin for ``GSplatData`` (save / load)."""

from __future__ import annotations

from pathlib import Path
from typing import TYPE_CHECKING, Any, Literal, Optional, Sequence

from .base import _GSplatDataOps

if TYPE_CHECKING:
    from luxar.encoding import EncodingMode
    from luxar.gsplats.gsplat_data import GSplatData


#: Sentinel for ``GSplatData.save(compressor=...)`` distinguishing "not specified"
#: (→ default Blosc) from an explicit ``compressor=None`` (→ no compression). A
#: plain ``None`` default would conflate the two and make uncompressed output
#: impossible (the bug that produced blosc-bitshuffle fixtures zarrita can't read).
_USE_DEFAULT_COMPRESSOR = object()


class IOAdapterMixin(_GSplatDataOps):
    """``GSplatData.save`` / ``GSplatData.load`` — .gsplats.zarr I/O."""

    def save(
        self,
        path: str | Path,
        ordering: Literal["morton", "hilbert", "none"] = "hilbert",
        encoding_mode: Optional["EncodingMode"] = None,
        include_fitting_info: bool = True,
        include_provenance: bool = False,
        description: Optional[str] = None,
        compress: Optional[Literal["zip", "tar.gz"]] = None,
        compressor: Any = _USE_DEFAULT_COMPRESSOR,
        zip_deflate: bool = False,
        barrier_dims: Optional[Sequence[int]] = None,
    ) -> None:
        """Save splats to .gsplats.zarr format.

        Args:
            path: Output path (should end with .gsplats.zarr or .gsplats.zarr.zip/.tar.gz if compress is used)
            ordering: Spatial ordering method ("morton", "hilbert", or "none")
            encoding_mode: Encoding mode (AUTO, PRECISION, or MEMORY), defaults to AUTO
            include_fitting_info: Whether to include fitting statistics
            include_provenance: Whether to include provenance info from stats
            description: Optional user description
            compress: Optional compression format ("zip" or "tar.gz"). Creates compressed archive.
            zip_deflate: Use DEFLATE compression for the outer zip (default: STORED).
                Useful when metadata overhead matters, e.g. for Git LFS storage.
            barrier_dims: Explicit categorical/barrier center columns for chunk
                ordering (e.g. a stacked-time axis). ``None`` (default) derives
                the barrier from the ``coarsen_dims`` complement in stats, else
                per-leaf auto-detect — see ``write_gsplats_tree``.

        For a multi-substitutive dataset the per-level ``coverage_fraction`` LOD
        switch thresholds (``sqrt(N_i/N_finest)``) are derived automatically — the
        viewer anchors the finest at fills-screen via the live viewport, so there
        is no per-dataset threshold knob. See
        ``core.group.lod.group.coverage_fractions``.

        Colors are written via the shared COLOR helper, which auto-detects SDR vs
        HDR (values > 1) — there is no explicit ``color_mode`` knob.

        Example:
            >>> result = fit_gaussian_splats(image, n_iters=1000)
            >>> result.save("fitted.gsplats.zarr", encoding_mode=EncodingMode.MEMORY)
            >>> # With compression for storage/git-lfs
            >>> result.save("fitted.gsplats.zarr.zip", compress="zip")
        """
        from luxar.encoding import EncodingMode
        from luxar.gsplats.io.save_gsplats import split_fitting_info, write_gsplats_tree
        from luxar.io.reader import DEFAULT_COMP

        # Use AUTO as default
        if encoding_mode is None:
            encoding_mode = EncodingMode.AUTO

        # Use Blosc(zstd) by default; an EXPLICIT compressor=None disables
        # compression (e.g. for raw, zarrita-readable cross-language fixtures).
        # Only the sentinel "not specified" coerces to the default.
        if compressor is _USE_DEFAULT_COMPRESSOR:
            compressor = DEFAULT_COMP

        # Extract fitting/provenance/pipeline groups from stats (single-sourced).
        fitting_info, fitting_config, provenance_info, pipeline_info = (
            split_fitting_info(
                self.stats,
                include_fitting_info=include_fitting_info,
                include_provenance=include_provenance,
            )
        )

        # One authoring path: serialize this dataset's node tree to the current
        # format (v3.2) via the shared walker (the same machinery the scene
        # compiler uses for leaves).
        # Multi-substitutive → a kind=lod group whose per-level coverage_fraction is
        # derived here (sqrt(N_i/N_finest)); a single level is a bare leaf.
        from luxar.gsplats.tree import tree_from_substitutive_levels

        tree = tree_from_substitutive_levels(self.substitutive_levels)
        write_gsplats_tree(
            path,
            tree,
            ordering=ordering,
            encoding_mode=encoding_mode,
            fitting_info=fitting_info,
            fitting_config=fitting_config,
            provenance_info=provenance_info,
            pipeline_info=pipeline_info,
            description=description,
            compress=compress,
            compressor=compressor,
            zip_deflate=zip_deflate,
            barrier_dims=barrier_dims,
        )

    @classmethod
    def load(
        cls,
        path: str | Path,
        include_stats: bool = False,
    ) -> "GSplatData":
        """Load splats from .gsplats.zarr format.

        Args:
            path: Path to .gsplats.zarr directory
            include_stats: Whether to include fitting/provenance metadata

        Returns:
            GSplatData with decoded arrays

        Example:
            >>> data = GSplatData.load("fitted.gsplats.zarr")
            >>> aprint(data.centers.shape)
        """
        from luxar.gsplats.io.load_gsplats import load_gsplats

        return load_gsplats(path, include_stats=include_stats)
