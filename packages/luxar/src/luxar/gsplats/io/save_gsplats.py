"""Save Gaussian splats to the ``.gsplats.zarr`` node-tree format.

A standalone ``.gsplats.zarr`` is a **detached scene-node subtree** written by the
single shared authoring path
(:func:`luxar.io._compiler.gsplat_tree.write_gsplat_node`) — the same machinery the
scene compiler uses for its gsplats leaves, so there is no parallel writer and
standalone leaves are byte-identical to scene leaves.

This module is the thin standalone wrapper: it builds the zarr store (handling
optional ``.zip`` / ``.tar.gz`` compression), writes the self-identifying root
header (``format_version`` = :data:`FORMAT_VERSION`), hands
the node tree to the shared walker, attaches optional ``fitting/`` /
``provenance/`` / ``pipeline/`` groups (``pipeline/`` = reduction/topology
stats; see :func:`split_fitting_info`), and consolidates metadata.
"""

from __future__ import annotations

import datetime
import os
import shutil
import tempfile
import uuid
from pathlib import Path
from typing import (
    TYPE_CHECKING,
    Any,
    Callable,
    Dict,
    Iterator,
    List,
    Literal,
    Optional,
    Sequence,
)

import numpy as np
import zarr
from zarr.storage import DirectoryStore

if TYPE_CHECKING:
    from luxar.gsplats.tree import GSplatNode

from luxar.encoding import EncodingMode
from luxar.io._compiler.gsplat_tree import (
    json_safe_value,
    make_dataset_ctx,
    make_ordering_ctx,
    write_gsplat_node,
)
from luxar.io.reader import DEFAULT_COMP
from luxar.typing_utils._format_contract import (
    GSPLATS_FORMAT_VERSION,
    SUPPORTED_GSPLATS_VERSIONS,
)
from luxar.utils.arbol_warnings import arbol_warnings
from luxar.utils.paths import normalize_zarr_path


# Get luxar.gsplats version — the subpackage has no own __version__, so fall
# back to the installed luxar distribution version (resolves under editable
# installs too); "unknown" only when the package metadata itself is missing.
def _resolve_gsplats_version() -> str:
    try:
        from luxar.gsplats import (  # type: ignore[attr-defined]
            __version__,
        )

        return str(__version__)
    except ImportError:
        try:
            from importlib.metadata import version

            return version("luxar")
        except Exception:
            return "unknown"


GSPLATS_VERSION: str = _resolve_gsplats_version()

#: On-disk format version for the node-tree ``.gsplats.zarr`` layout.
#: v3.3 introduces the optional ``luxar_delta_v1`` zarr filter on quantized
#: code arrays (coordinates / Cholesky halves / amplitudes / colors): columnar per-chunk
#: delta+zigzag, probe-gated at encode time (12-16% smaller stores, lossless).
#: The codec auto-registers via the numcodecs ``numcodecs.codecs`` entry
#: point whenever luxar is installed (importing ``luxar.encoding`` also
#: registers it eagerly); v3.3 stores without the filter are byte-identical
#: to v3.2.
#: v3.2 renames the ``kind=lod`` selector attrs: the group ``selector`` value
#: ``pixel_size`` → ``coverage`` and the per-child ``min_pixel_size`` (absolute
#: pixels) → ``coverage_fraction`` (viewport-relative ``sqrt(N_i/N_finest)`` in
#: [0,1], strictly ascending coarsest→finest, finest == 1.0).
#: v3.1 splits the Cholesky factors into ``cholesky_factors_diag`` (N, d) +
#: ``cholesky_factors_offdiag`` (N, k-d) so each can be encoded independently;
#: v3.0 stored a single packed ``cholesky_factors`` array.
#: Single-sourced from ``format-contract/contract.yaml``.
FORMAT_VERSION = GSPLATS_FORMAT_VERSION

#: Node-tree format versions the readers accept. v3.0 / v3.1 are still read
#: transparently: the loaders fall back to the single packed Cholesky array
#: when the split (``cholesky_factors_diag``) is absent, and a re-save derives
#: fresh ``coverage_fraction`` thresholds (the legacy ``min_pixel_size`` lod
#: attrs are ignored on read). The web viewer auto-adapts the legacy lod attrs
#: too, but warns — upgrade old stores with ``luxar gsplat migrate-format``.
#: Single-sourced from ``format-contract/contract.yaml``.
SUPPORTED_FORMAT_VERSIONS = SUPPORTED_GSPLATS_VERSIONS

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


#: ``stats`` keys stamped by the loader from the store header, never persisted
#: as pipeline stats (they would shadow the real header on the next save).
_HEADER_STATS_KEYS = (
    "format_version",
    "luxar_gsplats_version",
    "timestamp",
    "description",
)

#: Fit-runtime scratch keys that live in ``stats`` but are NOT persistable
#: reduction/topology provenance: napari-movie capture buffers (``movie_frames``
#: is set to ``None`` on *every* default fit — see ``fitting/results.py`` — so
#: without this exclusion every plain fit would emit a spurious ``pipeline/``
#: group, and a captured movie would leak the input volume shape). Kept out of
#: ``pipeline_info`` so a plain fit writes no ``pipeline/`` group at all.
_PIPELINE_EXCLUDE_KEYS = (
    "movie_frames",
    "movie_shape",
)


def split_fitting_info(
    stats: Optional[Dict[str, Any]],
    *,
    include_fitting_info: bool = True,
    include_provenance: bool = False,
) -> tuple[
    Optional[Dict[str, Any]],
    Optional[Dict[str, Any]],
    Optional[Dict[str, Any]],
    Optional[Dict[str, Any]],
]:
    """Split a ``stats`` dict into ``(fitting_info, fitting_config,
    provenance_info, pipeline_info)``.

    Mirrors the extraction done by :meth:`GSplatData.save` so the standalone
    node-tree writers (e.g. the ``lod --recipe`` composed recipes, which have no
    flat ``GSplatData``) attach the same ``fitting/`` / ``provenance/`` /
    ``pipeline/`` groups.

    ``pipeline_info`` is everything the first three buckets do NOT consume —
    the reduction/topology provenance (``lod_kind``, ``method``,
    ``compression_factor``, ``coverage_inflation``, ``refine``, ...) that was
    historically dropped on save (a silent lossy round-trip). Header keys the
    loader stamps itself (:data:`_HEADER_STATS_KEYS`), private ``_``-prefixed
    scratch keys, and non-JSON-serializable values are excluded.
    """
    if not stats:
        return None, None, None, None
    fitting_info: Optional[Dict[str, Any]] = None
    fitting_config: Optional[Dict[str, Any]] = None
    provenance_info: Optional[Dict[str, Any]] = None
    if include_fitting_info:
        fitting_info = {k: v for k, v in stats.items() if k in _FITTING_INFO_KEYS}
        if "config" in stats:
            fitting_config = stats["config"]
    if include_provenance and "provenance" in stats:
        provenance_info = stats["provenance"]
    pipeline_info: Dict[str, Any] = {}
    for key, value in stats.items():
        if (
            key in _FITTING_INFO_KEYS
            or key in _HEADER_STATS_KEYS
            or key in _PIPELINE_EXCLUDE_KEYS
            or key in ("config", "provenance")
            or key.startswith("_")
        ):
            continue
        ok, converted = json_safe_value(value)
        if ok:
            pipeline_info[key] = converted
    return fitting_info, fitting_config, provenance_info, pipeline_info or None


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


def _tmp_sibling(dest: Path) -> Path:
    """A hidden temp sibling of ``dest`` — same parent, so ``os.replace`` is a
    same-filesystem rename (never a copy). Dot-prefixed so glob discovery
    (``part_*``/``*.gsplats.zarr`` scans, batch status candidates) skips it;
    pid+uuid suffix so a stale sibling from a killed run can't collide."""
    return dest.parent / f".{dest.name}.tmp-{os.getpid()}-{uuid.uuid4().hex[:8]}"


def _atomic_finalize(tmp: Path, dest: Path) -> None:
    """Replace ``dest`` with the fully-written ``tmp``.

    Write-to-temp-then-swap is what makes the store writers crash-safe: a
    mid-write failure leaves ``dest`` (the prior good store, if any) untouched
    instead of destroyed-then-partially-rewritten. For single files the
    ``os.replace`` is atomic. For directories the swap is TRASH-FIRST: the
    old ``dest`` is renamed aside (atomic), ``tmp`` renamed in (atomic), and
    only then is the old copy deleted — so there is no instant, even under
    SIGKILL, at which ``dest`` is absent while a prior good store existed.
    A kill between the two renames leaves the old store recoverable at the
    ``.trash-*`` sibling; a kill after leaves at worst a stale trash dir.
    """
    if dest.is_dir():
        trash = dest.parent / f".{dest.name}.trash-{os.getpid()}-{uuid.uuid4().hex[:8]}"
        os.replace(str(dest), str(trash))
        try:
            os.replace(str(tmp), str(dest))
        except BaseException:
            # Restore the prior good store before propagating.
            os.replace(str(trash), str(dest))
            raise
        shutil.rmtree(trash, ignore_errors=True)
        return
    if dest.exists():
        dest.unlink()
    os.replace(str(tmp), str(dest))


def _compress_zarr(
    zarr_path: Path,
    out_path: Path,
    compress: str,
    zip_deflate: bool,
    temp_dir: Optional[Path],
) -> None:
    """Compress a written zarr directory into ``out_path`` and clean up temp.

    The archive streams into a temp sibling and is atomically renamed into
    place on success — a mid-compression failure leaves no partial archive
    at ``out_path`` (and any prior file there survives).
    """
    out_tmp = _tmp_sibling(out_path)
    try:
        import tarfile
        import zipfile

        if compress == "zip":
            zip_method = zipfile.ZIP_DEFLATED if zip_deflate else zipfile.ZIP_STORED
            with zipfile.ZipFile(out_tmp, "w", zip_method) as zipf:
                for file_path in zarr_path.rglob("*"):
                    if file_path.is_file():
                        zipf.write(file_path, file_path.relative_to(zarr_path.parent))
        elif compress == "tar.gz":
            with tarfile.open(out_tmp, "w:gz") as tarf:
                tarf.add(zarr_path, arcname=zarr_path.name)
        else:
            raise ValueError(f"Unsupported compression format: {compress!r}")
        _atomic_finalize(out_tmp, out_path)
    except BaseException:
        out_tmp.unlink(missing_ok=True)
        raise
    finally:
        if temp_dir is not None and temp_dir.exists():
            shutil.rmtree(temp_dir, ignore_errors=True)


def _stamp_content_hash(root: zarr.Group) -> str:
    """Stamp a root ``content_hash`` for viewer-side cache invalidation.

    The web viewer's persistent (OPFS) cache validates a dataset by re-fetching
    the root ``.zattrs`` and comparing ``content_hash``; without one the cache
    is trusted indefinitely and a regenerated file at the same URL serves stale
    data. The scene compiler stamps a hash at finalize
    (:func:`luxar.io._compiler.finalize.hashing.compute_content_hashes`), but
    that helper reads every array's full data — prohibitive for multi-GB splat
    stores — so this variant hashes the **metadata tree only**: per-group sorted
    attrs plus each array's ``(name, shape, dtype)``. The root ``timestamp``
    attr (microsecond ISO, rewritten on every save) rides along in the attrs
    walk, so every re-save yields a distinct hash even when the structure is
    unchanged — which is exactly the token cache invalidation needs.
    """
    import json

    import xxhash

    def hash_group(group: zarr.Group) -> str:
        hasher = xxhash.xxh64()
        for name in sorted(group.array_keys()):
            arr = group[name]
            hasher.update(
                f"{name}:{tuple(arr.shape)}:{arr.dtype}".encode()  # metadata only
            )
        attrs = {k: v for k, v in dict(group.attrs).items() if k != "content_hash"}
        hasher.update(json.dumps(attrs, sort_keys=True, default=str).encode())
        for name in sorted(group.group_keys()):
            hasher.update(hash_group(group[name]).encode())
        return hasher.hexdigest()

    content_hash = hash_group(root)
    root.attrs["content_hash"] = content_hash
    return content_hash


def _barrier_from_coarsen_dims(
    pipeline_info: Optional[Dict[str, Any]],
    node: Any,
) -> Optional[Sequence[int]]:
    """Derive ordering barrier axes from persisted ``coarsen_dims``.

    The LOD reducer coarsens (merges) over ``coarsen_dims`` and treats the rest
    as hard grouping barriers (time/channel). The ordering barrier is exactly
    that complement over the node's ``ndim``.

    Returns ``None`` only when no ``coarsen_dims`` provenance is available (→
    per-leaf auto-detection). When provenance IS present the complement is
    authoritative and returned verbatim — INCLUDING an empty list, which means
    "coarsen everything, no barrier" (pure spatial ordering). Returning ``None``
    there would wrongly fall through to auto-detect and could re-introduce a
    barrier the producer explicitly ruled out.
    """
    if not pipeline_info:
        return None
    coarsen = pipeline_info.get("coarsen_dims")
    if coarsen is None:
        return None
    try:
        ndim = int(node.ndim)
    except (AttributeError, TypeError):
        return None
    coarsen_set = {int(d) for d in coarsen}
    # Empty complement (coarsen-all) is an explicit no-barrier → return [] not None.
    return [d for d in range(ndim) if d not in coarsen_set]


def write_gsplats_tree(
    path: str | Path,
    node: Any,  # luxar.gsplats.tree.GSplatNode
    *,
    ordering: Literal["morton", "hilbert", "none"] = "hilbert",
    encoding_mode: EncodingMode = EncodingMode.AUTO,
    fitting_info: Optional[Dict[str, Any]] = None,
    fitting_config: Optional[Dict[str, Any]] = None,
    provenance_info: Optional[Dict[str, Any]] = None,
    pipeline_info: Optional[Dict[str, Any]] = None,
    description: Optional[str] = None,
    compress: Optional[Literal["zip", "tar.gz"]] = None,
    compressor: Optional[Any] = DEFAULT_COMP,
    zip_deflate: bool = False,
    barrier_dims: Optional[Sequence[int]] = None,
) -> None:
    """Write a :class:`~luxar.gsplats.tree.GSplatNode` subtree as ``.gsplats.zarr``.

    The node *is* the file root: the shared walker stamps the root group with the
    node's own attrs (``type``/``kind`` + ``position_bounds``), and this wrapper
    adds the self-identifying header (``format_version`` = :data:`FORMAT_VERSION`)
    plus optional ``fitting/`` / ``provenance/`` / ``pipeline/`` groups
    (``pipeline/`` carries the reduction/topology stats — see
    :func:`split_fitting_info`).

    ``barrier_dims`` names categorical/barrier center columns (time, channel) so
    chunk ordering groups by them first and per-slice reads stay local. When
    ``None`` it is derived from ``pipeline_info["coarsen_dims"]`` (barrier =
    complement) if present; failing that each leaf auto-detects from its centers.
    """
    path = Path(path)
    temp_dir, zarr_path = _resolve_zarr_path(path, compress)
    if not compress:
        # Crash-safety: write into a hidden temp sibling and atomically swap
        # into place at the end. The old in-place ``overwrite=True`` cleared a
        # pre-existing store at ``path`` BEFORE writing, so a mid-write crash
        # destroyed the prior good copy and left a partial store behind (which
        # existence-gated consumers like the batch-merge resume then treated
        # as complete).
        zarr_path = _tmp_sibling(path)

    store = DirectoryStore(str(zarr_path))
    root = zarr.group(store=store, overwrite=True)
    try:
        # Barrier axes for ordering: explicit arg wins; else the LOD reduction
        # barrier (complement of the persisted coarsen_dims); else per-leaf
        # auto-detect (barrier_dims stays None → detect_barrier_dims per leaf).
        if barrier_dims is None:
            barrier_dims = _barrier_from_coarsen_dims(pipeline_info, node)

        dataset_ctx = make_dataset_ctx(encoding_mode, compressor=compressor)
        ordering_ctx = make_ordering_ctx(ordering)
        write_gsplat_node(
            root,
            node,
            dataset_ctx=dataset_ctx,
            ordering_ctx=ordering_ctx,
            store=root,
            barrier_dims=barrier_dims,
        )

        # Self-identifying v3.0 header (the node's own type/kind/position_bounds attrs
        # were written onto root by the walker; these header keys are disjoint).
        root.attrs["format_version"] = FORMAT_VERSION
        root.attrs["format_type"] = "gsplats_zarr"
        root.attrs["timestamp"] = datetime.datetime.now(
            datetime.timezone.utc
        ).isoformat()
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
        if pipeline_info:
            # Reduction/topology stats (lod_kind, method, compression_factor,
            # coverage_inflation, refine, ...) — everything split_fitting_info's
            # other buckets do not consume. Optional group: absent for plain fits.
            root.create_group("pipeline").attrs.update(pipeline_info)

        # Stamp BEFORE consolidating so the hash lands in ``.zmetadata`` too.
        _stamp_content_hash(root)
        zarr.consolidate_metadata(store)

        if compress:
            _compress_zarr(zarr_path, path, compress, zip_deflate, temp_dir)
        else:
            _atomic_finalize(zarr_path, path)
    except BaseException:
        # Never leave a temp sibling behind; the destination (the prior good
        # store, if any) is untouched by construction.
        if temp_dir is not None:
            shutil.rmtree(temp_dir, ignore_errors=True)
        else:
            shutil.rmtree(zarr_path, ignore_errors=True)
        raise


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
    pipeline_info: Optional[Dict[str, Any]] = None,
    description: Optional[str] = None,
    compressor: Optional[Any] = DEFAULT_COMP,
    barrier_dims: Optional[Sequence[int]] = None,
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
    (The standalone branch additionally writes an optional ``bsp_tree`` split-plane
    record when the parts came from a single BSP; a streamed grid/content merge has
    no single tree, so this writer intentionally omits it and the viewer falls back
    to a per-part centroid order.)

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
    # Crash-safety: stream into a hidden temp sibling and atomically swap into
    # place after the final consolidate. This writer can run for a long time
    # (one part per tile of a whole timelapse); the old in-place
    # ``overwrite=True`` destroyed a pre-existing output before the first part
    # landed, and a crashed merge left a partial store that the batch-merge
    # resume (existence-gated) then treated as complete.
    tmp = _tmp_sibling(path)
    store = DirectoryStore(str(tmp))
    root = zarr.group(store=store, overwrite=True)
    try:
        dataset_ctx = make_dataset_ctx(encoding_mode, compressor=compressor)
        ordering_ctx = make_ordering_ctx(ordering)

        child_bounds: List[Dict[str, List[float]]] = []
        n_written = 0
        for node in part_nodes():
            part_group = root.require_group(f"part_{n_written}")
            # Barrier for ordering: explicit arg wins; else derive from this part's
            # coarsen_dims provenance; else per-leaf auto-detect. Computed per part
            # since parts are streamed one at a time (each has its own ndim).
            part_barrier = (
                barrier_dims
                if barrier_dims is not None
                else _barrier_from_coarsen_dims(pipeline_info, node)
            )
            cmeta = write_gsplat_node(
                part_group,
                node,
                dataset_ctx=dataset_ctx,
                ordering_ctx=ordering_ctx,
                store=root,
                # Insertion order for the viewer's sibling sort — matches the
                # standalone partition writer (prevents part_10 < part_2 reorder).
                attrs={"child_index": n_written},
                barrier_dims=part_barrier,
            )
            if "position_bounds" in cmeta:
                child_bounds.append(cmeta["position_bounds"])
            n_written += 1

        if n_written == 0:
            raise ValueError("write_partition_streaming: no non-empty parts to write")

        # Root partition attrs — same set the GSplatPartition branch of
        # write_gsplat_node emits (type/kind/display_type/max_elements/position_bounds).
        # That branch may ALSO write an optional bsp_tree; a streamed merge has no
        # single BSP tree, so this writer omits it (viewer falls back to centroids).
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
        root.attrs["timestamp"] = datetime.datetime.now(
            datetime.timezone.utc
        ).isoformat()
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
        if pipeline_info:
            root.create_group("pipeline").attrs.update(pipeline_info)

        # Stamp BEFORE consolidating so the hash lands in ``.zmetadata`` too.
        _stamp_content_hash(root)
        zarr.consolidate_metadata(store)
    except BaseException:
        # Includes the n_written == 0 ValueError above — no partial root is
        # left behind either way; the destination stays untouched.
        shutil.rmtree(tmp, ignore_errors=True)
        raise
    _atomic_finalize(tmp, path)
    return n_written


@arbol_warnings()
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
