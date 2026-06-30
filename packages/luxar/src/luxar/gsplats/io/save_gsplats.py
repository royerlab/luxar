"""Save Gaussian splats to the ``.gsplats.zarr`` node-tree format.

A standalone ``.gsplats.zarr`` is a **detached scene-node subtree** written by the
single shared authoring path
(:func:`luxar.io._compiler.gsplat_tree.write_gsplat_node`) — the same machinery the
scene compiler uses for its gsplats leaves, so there is no parallel writer and
standalone leaves are byte-identical to scene leaves.

This module is the thin standalone wrapper: it builds the zarr store (handling
optional ``.zip`` / ``.tar.gz`` compression), writes the self-identifying root
header (``format_version`` = :data:`FORMAT_VERSION`), hands
the node tree to the shared walker, attaches optional ``fitting/`` / ``provenance/``
groups, and consolidates metadata.
"""

from __future__ import annotations

import datetime
import shutil
import tempfile
from pathlib import Path
from typing import TYPE_CHECKING, Any, Callable, Dict, Iterator, List, Literal, Optional

import numpy as np
import zarr
from zarr.storage import DirectoryStore

if TYPE_CHECKING:
    from luxar.gsplats.tree import GSplatNode

from luxar.encoding import EncodingMode
from luxar.io._compiler.gsplat_tree import (
    make_dataset_ctx,
    make_ordering_ctx,
    write_gsplat_node,
)
from luxar.io.reader import DEFAULT_COMP
from luxar.utils.paths import normalize_zarr_path

# Get luxar.gsplats version
try:
    from luxar.gsplats import (  # type: ignore[attr-defined]
        __version__ as GSPLATS_VERSION,
    )
except ImportError:
    GSPLATS_VERSION: str = "unknown"  # type: ignore[no-redef]

#: On-disk format version for the node-tree ``.gsplats.zarr`` layout.
#: v3.1 splits the Cholesky factors into ``cholesky_factors_diag`` (N, d) +
#: ``cholesky_factors_offdiag`` (N, k-d) so each can be encoded independently;
#: v3.0 stored a single packed ``cholesky_factors`` array.
FORMAT_VERSION = "3.1"

#: Node-tree format versions the readers accept. v3.0 is still read
#: transparently: the loaders fall back to the single packed Cholesky array
#: when the split (``cholesky_factors_diag``) is absent.
SUPPORTED_FORMAT_VERSIONS = ("3.0", "3.1")

#: ``stats`` keys lifted into the ``fitting/`` group on save (quality metrics,
#: culling/filtering provenance). Single-sourced here so every writer (``GSplatData.save``
#: and the ``lod --recipe`` CLI) selects the same fields.
_FITTING_INFO_KEYS = (
    "time_seconds",
    "iterations",
    "converged",
    "early_stopped",
    "best_iteration",
    "final_loss",
    "final_max_abs_error",
    "final_rel_l2",
    "n_splats",
    "n_culled",
    "fitter_name",
    "fitter_version",
    "timestamp",
    "culled",
    "culling_method",
    "n_original",
    "n_removed",
    "amplitude_retention",
    "filtered",
    "filter_criteria",
    "truncate",
    "psnr_db",
    "ssim",
    "mse",
)


def split_fitting_info(
    stats: Optional[Dict[str, Any]],
    *,
    include_fitting_info: bool = True,
    include_provenance: bool = False,
) -> tuple[
    Optional[Dict[str, Any]], Optional[Dict[str, Any]], Optional[Dict[str, Any]]
]:
    """Split a ``stats`` dict into ``(fitting_info, fitting_config, provenance_info)``.

    Mirrors the extraction done by :meth:`GSplatData.save` so the standalone
    node-tree writers (e.g. the ``lod --recipe`` composed recipes, which have no
    flat ``GSplatData``) attach the same ``fitting/`` / ``provenance/`` groups.
    """
    if not stats:
        return None, None, None
    fitting_info: Optional[Dict[str, Any]] = None
    fitting_config: Optional[Dict[str, Any]] = None
    provenance_info: Optional[Dict[str, Any]] = None
    if include_fitting_info:
        fitting_info = {k: v for k, v in stats.items() if k in _FITTING_INFO_KEYS}
        if "config" in stats:
            fitting_config = stats["config"]
    if include_provenance and "provenance" in stats:
        provenance_info = stats["provenance"]
    return fitting_info, fitting_config, provenance_info


def _resolve_zarr_path(
    path: Path, compress: Optional[str]
) -> tuple[Optional[Path], Path]:
    """Return ``(temp_dir, zarr_path)`` — a temp dir when compressing, else None."""
    if not compress:
        return None, path
    temp_dir = Path(tempfile.mkdtemp(prefix="luxar_gsplat_save_"))
    zarr_name = path.name
    for suffix in (".zip", ".tar.gz", ".gz"):
        if zarr_name.endswith(suffix):
            zarr_name = zarr_name[: -len(suffix)]
            break
    # Enforce the canonical standalone suffix on the inner store name (shared
    # with the scene compiler's ``.luxar.zarr`` normalization).
    zarr_name = normalize_zarr_path(zarr_name, ".gsplats.zarr").name
    return temp_dir, temp_dir / zarr_name


def _compress_zarr(
    zarr_path: Path,
    out_path: Path,
    compress: str,
    zip_deflate: bool,
    temp_dir: Optional[Path],
) -> None:
    """Compress a written zarr directory into ``out_path`` and clean up temp."""
    try:
        import tarfile
        import zipfile

        if compress == "zip":
            zip_method = zipfile.ZIP_DEFLATED if zip_deflate else zipfile.ZIP_STORED
            with zipfile.ZipFile(out_path, "w", zip_method) as zipf:
                for file_path in zarr_path.rglob("*"):
                    if file_path.is_file():
                        zipf.write(file_path, file_path.relative_to(zarr_path.parent))
        elif compress == "tar.gz":
            with tarfile.open(out_path, "w:gz") as tarf:
                tarf.add(zarr_path, arcname=zarr_path.name)
        else:
            raise ValueError(f"Unsupported compression format: {compress!r}")
    finally:
        if temp_dir is not None and temp_dir.exists():
            shutil.rmtree(temp_dir, ignore_errors=True)


def write_gsplats_tree(
    path: str | Path,
    node: Any,  # luxar.gsplats.tree.GSplatNode
    *,
    ordering: Literal["morton", "hilbert", "none"] = "hilbert",
    encoding_mode: EncodingMode = EncodingMode.AUTO,
    fitting_info: Optional[Dict[str, Any]] = None,
    fitting_config: Optional[Dict[str, Any]] = None,
    provenance_info: Optional[Dict[str, Any]] = None,
    description: Optional[str] = None,
    compress: Optional[Literal["zip", "tar.gz"]] = None,
    compressor: Optional[Any] = DEFAULT_COMP,
    zip_deflate: bool = False,
) -> None:
    """Write a :class:`~luxar.gsplats.tree.GSplatNode` subtree as ``.gsplats.zarr``.

    The node *is* the file root: the shared walker stamps the root group with the
    node's own attrs (``type``/``kind`` + ``position_bounds``), and this wrapper
    adds the self-identifying header (``format_version`` = :data:`FORMAT_VERSION`)
    plus optional ``fitting/`` / ``provenance/``.
    """
    path = Path(path)
    temp_dir, zarr_path = _resolve_zarr_path(path, compress)

    store = DirectoryStore(str(zarr_path))
    root = zarr.group(store=store, overwrite=True)

    dataset_ctx = make_dataset_ctx(encoding_mode, compressor=compressor)
    ordering_ctx = make_ordering_ctx(ordering)
    write_gsplat_node(
        root, node, dataset_ctx=dataset_ctx, ordering_ctx=ordering_ctx, store=root
    )

    # Self-identifying v3.0 header (the node's own type/kind/position_bounds attrs
    # were written onto root by the walker; these header keys are disjoint).
    root.attrs["format_version"] = FORMAT_VERSION
    root.attrs["format_type"] = "gsplats_zarr"
    root.attrs["timestamp"] = datetime.datetime.now(datetime.timezone.utc).isoformat()
    root.attrs["luxar_gsplats_version"] = GSPLATS_VERSION
    # A standalone file opened directly (?src=…gsplats.zarr) is the whole layer,
    # so expose the root in the viewer's Layers panel (the scene-embed graft uses
    # the scene builders instead and does not carry this root attr).
    root.attrs.setdefault("layer", True)
    if description:
        root.attrs["description"] = description

    if fitting_info is not None:
        # `n_splats` denotes the count of splats in THIS artifact (see
        # fitting/results.py). Stats are inherited from the source fit, so for
        # count-changing operations (substitutive / multiscale LOD synthesise
        # extra representative splats) the inherited value is stale and would
        # contradict the file's own leaf arrays. Correct it to the true total.
        if "n_splats" in fitting_info:
            from luxar.gsplats.tree import total_splats

            fitting_info = {**fitting_info, "n_splats": int(total_splats(node))}
        fitting_group = root.create_group("fitting")
        fitting_group.attrs.update(fitting_info)
        if fitting_config is not None:
            fitting_group.create_group("config").attrs.update(fitting_config)
    if provenance_info is not None:
        root.create_group("provenance").attrs.update(provenance_info)

    zarr.consolidate_metadata(store)

    if compress:
        _compress_zarr(zarr_path, path, compress, zip_deflate, temp_dir)


def write_partition_streaming(
    path: str | Path,
    part_nodes: Callable[[], Iterator["GSplatNode"]],
    *,
    max_elements: int = 0,
    ordering: Literal["morton", "hilbert", "none"] = "hilbert",
    encoding_mode: EncodingMode = EncodingMode.AUTO,
    fitting_info: Optional[Dict[str, Any]] = None,
    fitting_config: Optional[Dict[str, Any]] = None,
    provenance_info: Optional[Dict[str, Any]] = None,
    description: Optional[str] = None,
    compressor: Optional[Any] = DEFAULT_COMP,
) -> int:
    """Write a ``kind=partition`` file part-by-part, holding ≤1 part in memory.

    This is the **streaming** sibling of :func:`write_gsplats_tree`: rather than
    take a whole in-memory :class:`~luxar.gsplats.tree.GSplatPartition` (which
    would materialize every part at once — the OOM the tiled-batch merge must
    avoid), it pulls each part subtree one at a time from ``part_nodes()`` and
    writes it straight into ``part_<i>/`` via the shared leaf/node walker
    (:func:`~luxar.io._compiler.gsplat_tree.write_gsplat_node`), so the on-disk
    bytes are identical to the standalone partition writer's. Only the producer's
    single yielded subtree is resident at any moment.

    The root group is stamped with the same ``kind=partition`` attrs the standalone
    writer emits (``type``/``kind``/``display_type``/``max_elements`` + a union
    ``position_bounds``), plus the v3.0 self-identifying header. Each part carries
    a ``child_index`` for napari-style sibling ordering — matching
    :func:`~luxar.io._compiler.gsplat_tree.write_gsplat_node`'s partition branch.

    The producer is responsible for skipping empty tile-regions (it must yield
    only non-empty subtrees). Compression is intentionally not supported here
    (the streaming use case writes a directory store); use
    :func:`write_gsplats_tree` for a compressed standalone partition.

    Returns the number of parts actually written.
    """
    from luxar.io._compiler.gsplat_tree import (
        _union_bounds,
        make_dataset_ctx,
        make_ordering_ctx,
        write_gsplat_node,
    )

    path = Path(path)
    store = DirectoryStore(str(path))
    root = zarr.group(store=store, overwrite=True)

    dataset_ctx = make_dataset_ctx(encoding_mode, compressor=compressor)
    ordering_ctx = make_ordering_ctx(ordering)

    child_bounds: List[Dict[str, List[float]]] = []
    n_written = 0
    for node in part_nodes():
        part_group = root.require_group(f"part_{n_written}")
        cmeta = write_gsplat_node(
            part_group,
            node,
            dataset_ctx=dataset_ctx,
            ordering_ctx=ordering_ctx,
            store=root,
            # Insertion order for the viewer's sibling sort — matches the
            # standalone partition writer (prevents part_10 < part_2 reorder).
            attrs={"child_index": n_written},
        )
        if "position_bounds" in cmeta:
            child_bounds.append(cmeta["position_bounds"])
        n_written += 1

    if n_written == 0:
        raise ValueError("write_partition_streaming: no non-empty parts to write")

    # Root partition attrs — byte-identical to the GSplatPartition branch of
    # write_gsplat_node (type/kind/display_type/max_elements/position_bounds).
    root.attrs["type"] = "group"
    root.attrs["kind"] = "partition"
    root.attrs["display_type"] = "gsplats"
    root.attrs["max_elements"] = int(max_elements)
    bounds = _union_bounds(child_bounds)
    if bounds is not None:
        root.attrs["position_bounds"] = bounds

    # Self-identifying v3.0 header (disjoint from the node's structural attrs).
    root.attrs["format_version"] = FORMAT_VERSION
    root.attrs["format_type"] = "gsplats_zarr"
    root.attrs["timestamp"] = datetime.datetime.now(datetime.timezone.utc).isoformat()
    root.attrs["luxar_gsplats_version"] = GSPLATS_VERSION
    root.attrs.setdefault("layer", True)
    if description:
        root.attrs["description"] = description

    if fitting_info is not None:
        fitting_group = root.create_group("fitting")
        fitting_group.attrs.update(fitting_info)
        if fitting_config is not None:
            fitting_group.create_group("config").attrs.update(fitting_config)
    if provenance_info is not None:
        root.create_group("provenance").attrs.update(provenance_info)

    zarr.consolidate_metadata(store)
    return n_written


def save_gsplats(
    path: str | Path,
    centers: np.ndarray,
    amplitudes: np.ndarray,
    cholesky_factors: np.ndarray,
    colors: Optional[np.ndarray] = None,
    ordering: Literal["morton", "hilbert", "none"] = "hilbert",
    encoding_mode: EncodingMode = EncodingMode.AUTO,
    fitting_info: Optional[Dict[str, Any]] = None,
    fitting_config: Optional[Dict[str, Any]] = None,
    provenance_info: Optional[Dict[str, Any]] = None,
    description: Optional[str] = None,
    compress: Optional[Literal["zip", "tar.gz"]] = None,
    compressor: Optional[Any] = DEFAULT_COMP,
    zip_deflate: bool = False,
    truncation_radius: float = 3.0,
) -> None:
    """Save a single Gaussian-splat set to ``.gsplats.zarr`` (a leaf node).

    Thin convenience wrapper: builds a single-leaf :class:`GSplatNode` and hands it
    to :func:`write_gsplats_tree`. Colors are written via the shared COLOR helper,
    which auto-detects SDR vs HDR (values > 1) — there is no explicit ``color_mode``
    knob; amplitudes use the canonical POSITIVE_SCALAR encoding.

    Empty input (``n_splats == 0``) raises — the shared writer validates against
    empty splat sets, matching the scene writer's no-empty policy.
    """
    from luxar.gsplats.gsplat_data import AdditiveSubLOD
    from luxar.gsplats.tree import GSplatLeaf

    leaf = GSplatLeaf(
        additive_sublods=[
            AdditiveSubLOD(
                centers=centers,
                amplitudes=amplitudes,
                cholesky_factors=cholesky_factors,
                colors=colors,
                truncation_radius=truncation_radius,
            )
        ]
    )
    write_gsplats_tree(
        path,
        leaf,
        ordering=ordering,
        encoding_mode=encoding_mode,
        fitting_info=fitting_info,
        fitting_config=fitting_config,
        provenance_info=provenance_info,
        description=description,
        compress=compress,
        compressor=compressor,
        zip_deflate=zip_deflate,
    )
