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
    cast,
)

import numpy as np
import zarr

from luxar._zarr_compat import consolidate, create_root_group, open_store

if TYPE_CHECKING:
    from luxar.gsplats.gsplat_data import AdditiveSubLOD
    from luxar.gsplats.tree import GSplatNode

from luxar.core.group.compositing import WRITER_STAMPED_APPEARANCE_DEFAULTS
from luxar.encoding import EncodingMode
from luxar.io._compiler.finalize.amplitude_window import (
    harmonize_gsplat_amplitude_windows,
)
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
from luxar.typing_utils.constants import DEFAULT_TRUNCATION_RADIUS
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
#: v3.4 adds the ``selector`` value ``screen-area`` (what every derived ladder
#: now stamps): per-child ``coverage_fraction`` becomes a literal screen-area
#: fraction (occupancy halving — whole-object finest 0.5, partition tile 1.0).
#: Stores carrying ``selector="coverage"`` keep the legacy diagonal units and
#: are read/round-tripped unchanged.
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
    # Foreground PSNR, plus the two numbers needed to interpret it. Global PSNR
    # on a sparse volume is dominated by background, so it is not the figure to
    # publish alone; the threshold and the foreground share say what the score
    # was actually taken over, which a bare dB value cannot.
    "foreground_psnr_db",
    "foreground_threshold",
    "foreground_fraction",
    # What the splats represent. These describe the fit's INPUT, so they belong
    # beside the fit's other statistics rather than in the pipeline bucket that
    # catches everything else. Without them a stored dataset cannot say how much
    # it compressed: the source grid appears nowhere else on disk, and it is not
    # recoverable from the producing script either, because the fitted grid is
    # derived at run time from downscale factors and voxel spacing.
    "source_shape",
    # Whether that grid was DECLARED by the caller rather than measured from the
    # array handed in. It belongs beside the grid it qualifies: dropped from this
    # list it lands in `pipeline/` instead, and a reader looking at
    # `fitting/source_shape` would have no way to tell a stated denominator from
    # a measured one — the whole point of stamping it.
    "source_declared",
    "source_dtype",
    "source_voxels",
    "source_bytes",
    "source_stored_bytes",
    # Per-coordinate component-fit provenance for a stacked axis. This remains a
    # list rather than being averaged into one quality claim for the union.
    "part_provenance",
    "fitted_shape",
    "fitted_voxels",
    "occupancy",
    "voxels_per_splat",
)


#: ``stats`` keys stamped by the loader from the store header, never persisted
#: as pipeline stats (they would shadow the real header on the next save).
_HEADER_STATS_KEYS = (
    "format_version",
    "luxar_gsplats_version",
    "timestamp",
    "description",
)

#: The normalization-provenance block: what a fit did to the intensity scale
#: before optimising. ONE key name (``floor``) and ONE location (the root
#: ``pipeline/`` group) on every writer path — see
#: ``docs/specs/GSPLATS_ZARR_FORMAT.md`` §"Pipeline Group Attributes" (#1175).
#:
#: The values are in the INPUT VOLUME's own units: ``floor`` is the background
#: level subtracted before fitting (``None``/``null`` when suppression was
#: disabled or refused), ``image_min`` / ``image_max`` are the normalization
#: bounds and ``intensity_range`` their span. A tiled/progressive path removes
#: the pedestal OUTSIDE the fitter and then fits with ``floor="none"``, so it
#: must shift its inner bounds back into those units before recording them —
#: otherwise ``image_min`` would mean the tile's post-subtraction minimum on one
#: path and the applied level on another.
NORMALIZATION_STATS_KEYS = ("floor", "image_min", "image_max", "intensity_range")


def agreed_normalization_stats(
    stats_list: "Sequence[Optional[Dict[str, Any]]]",
) -> Dict[str, Any]:
    """The :data:`NORMALIZATION_STATS_KEYS` block the inputs UNANIMOUSLY agree on.

    Merging several fits (tiles of one volume, boxes of one plan, arbitrary
    datasets) must not invent a normalization record. Inputs can legitimately
    disagree — two independently fitted volumes have two different pedestals —
    and promoting the first one's block would silently mislabel every other
    input. So a key is carried only when every input that HAS an opinion agrees;
    on any disagreement it is dropped, and the merged artifact simply says
    nothing rather than something false.

    An input that does not carry a key casts no vote: an empty/skipped tile
    records no bounds at all, and vetoing on that would erase the block for the
    whole merge. At least one input must carry the key for it to appear.
    """
    agreed: Dict[str, Any] = {}
    for key in NORMALIZATION_STATS_KEYS:
        values = [s[key] for s in stats_list if s is not None and key in s]
        if not values:
            continue
        first = values[0]
        if all(v == first for v in values):
            agreed[key] = first
    return agreed


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

    ``fitting_info`` is held to the same JSON rule: a non-finite metric (``nan``
    foreground PSNR on a volume with no foreground, ``+inf`` global PSNR on an
    exact reconstruction) is DROPPED rather than written, because zarr emits it
    as a bare ``NaN`` / ``Infinity`` token that invalidates the entire metadata
    document for a strict parser.
    """
    if not stats:
        return None, None, None, None
    fitting_info: Optional[Dict[str, Any]] = None
    fitting_config: Optional[Dict[str, Any]] = None
    provenance_info: Optional[Dict[str, Any]] = None
    if include_fitting_info:
        fitting_info = {k: v for k, v in stats.items() if k in _FITTING_INFO_KEYS}
        # Same JSON discipline the ``pipeline_info`` loop below applies, and for
        # the same reason: zarr serializes a non-finite float as a bare ``NaN`` /
        # ``Infinity`` token, which is not JSON. A single such value anywhere in
        # the tree makes the WHOLE metadata document unreadable to a strict
        # parser (the viewer's ``JSON.parse``, jq, any non-Python reader) — and
        # quality metrics reach non-finite on ordinary inputs: a constant or
        # signal-free volume has no foreground, so ``foreground_psnr_db`` is
        # ``nan`` and ``psnr_db`` is ``+inf``. Drop those keys rather than
        # corrupt the store; ``foreground_fraction: 0.0`` still says why.
        _, fitting_info = json_safe_value(fitting_info)
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
    attrs plus each array's ``(name, shape, chunks, shards, dtype, codec ids,
    own attrs)`` and each child group's NAME alongside its digest. The root
    ``timestamp`` attr (microsecond ISO, rewritten on every save) rides along in
    the attrs walk, so every re-save yields a distinct hash even when the
    structure is unchanged — which is exactly the token cache invalidation needs.

    Chunk/shard layout, the codec ids and the per-array attrs are part of that
    identity for the same reasons the compiler-side hash folds them in — see
    :func:`luxar.io._compiler.finalize.hashing._storage_identity`, which is where
    that argument lives. Both digests should agree on what a store's identity IS,
    and this one carries an extra obligation: it is what the three IN-PLACE
    re-stampers (``gsplat annotate-quality``, ``gsplat doctor --fix``, and
    ``luxar restamp-lod``) write, and they leave the per-save ``timestamp``
    untouched. None of them can change layout today — every mutation on those
    paths is attrs-only — so what moves their digest is the changed attrs, as it
    already did. The fold is here so
    that a future in-place RE-LAYOUT tool cannot re-stamp a store to its input's
    digest. The two remain separate digests over different serializations (this one
    an f-string over metadata only; the compiler's a sorted-key JSON dict plus the
    decoded values), and nothing compares them with each other.
    """
    import json

    import xxhash

    from luxar.io._compiler.finalize.hashing import codec_ids

    def hash_group(group: zarr.Group) -> str:
        hasher = xxhash.xxh64()
        for name in sorted(group.array_keys()):
            arr = group[name]
            # Metadata only — never `arr[:]`. See `_storage_identity` for why
            # `shards` needs no defensive access and why the codec ids are
            # derived format-agnostically.
            shards = arr.shards
            arr_attrs = json.dumps(dict(arr.attrs), sort_keys=True, default=str)
            # A SHARDED array's whole pipeline nests inside the single top-level
            # `ShardingCodec`, so the ids alone read `["sharding_indexed"]` and
            # say nothing about the inner codecs or the shard index. Expand it,
            # exactly as `_storage_identity` does.
            pipeline = (
                [codec.to_dict() for codec in arr.metadata.codecs]
                if shards is not None
                else None
            )
            hasher.update(
                f"{name}:{tuple(arr.shape)}:{tuple(arr.chunks)}:"
                f"{tuple(shards) if shards is not None else None}:{arr.dtype}:"
                f"{codec_ids(arr.metadata)}:{pipeline}:{arr_attrs}".encode()
            )
        attrs = {k: v for k, v in dict(group.attrs).items() if k != "content_hash"}
        hasher.update(json.dumps(attrs, sort_keys=True, default=str).encode())
        # The child's NAME, not just its digest: a node's own digest does not
        # carry its name, so hashing digests alone left a renamed child group
        # invisible to every ancestor.
        for name in sorted(group.group_keys()):
            hasher.update(f"{name}:{hash_group(group[name])}".encode())
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


def _with_root_normalization(
    pipeline_info: Optional[Dict[str, Any]], node: Any
) -> Optional[Dict[str, Any]]:
    """Fold the root node's normalization block into ``pipeline_info`` (#1175).

    A ``kind=partition`` (or any tree) result has nowhere to put fit stats — the
    writer takes them from a flat leaf's ``stats`` — which is how the background
    level a tiled/content fit removed used to be lost on save. The producing path
    stamps it onto the ROOT node's ``meta`` instead and this promotes it into the
    store's ``pipeline/`` group, so a partition and a flat leaf answer the same
    question with the same key in the same place. Deliberately NOT added to
    ``_NODE_META_ATTR_KEYS``: a second on-disk home for one fact is the very
    inconsistency #1175 is about.
    """
    meta = getattr(node, "meta", None)
    if not meta:
        return pipeline_info
    merged = dict(pipeline_info) if pipeline_info else {}
    for key in NORMALIZATION_STATS_KEYS:
        if key not in meta or key in merged:
            continue
        ok, converted = json_safe_value(meta[key])
        if ok:
            merged[key] = converted
    return merged or None


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
    root_attrs: Optional[Dict[str, Any]] = None,
) -> None:
    """Write a :class:`~luxar.gsplats.tree.GSplatNode` subtree as ``.gsplats.zarr``.

    The node *is* the file root: the shared walker stamps the root group with the
    node's own attrs (``type``/``kind`` + ``position_bounds``), and this wrapper
    adds the self-identifying header (``format_version`` = :data:`FORMAT_VERSION`)
    plus optional ``fitting/`` / ``provenance/`` / ``pipeline/`` groups
    (``pipeline/`` carries the reduction/topology stats — see
    :func:`split_fitting_info`).

    ``root_attrs`` seeds the root group through the walker's LOWEST-precedence
    caller-attrs channel, so structural attrs and a node's own ``meta`` still
    win. A structure-only rebuild uses it to carry the SOURCE root's authored
    appearance (see :data:`~luxar.core.group.compositing.
    AUTHORED_APPEARANCE_ATTRS`) onto the result, which would otherwise be
    silently dropped — the reduction builds fresh nodes that know nothing about
    the input's appearance.

    ``barrier_dims`` names categorical/barrier center columns (time, channel) so
    chunk ordering groups by them first and per-slice reads stay local. When
    ``None`` it is derived from ``pipeline_info["coarsen_dims"]`` (barrier =
    complement) if present; failing that each leaf auto-detects from its centers.

    A tree has no flat ``stats`` dict for :func:`split_fitting_info` to route, so
    the :data:`NORMALIZATION_STATS_KEYS` block rides on the ROOT node's ``meta``
    and is promoted here into ``pipeline/`` — the one location the format spec
    names for it (#1175). An explicit ``pipeline_info`` entry wins.
    """
    path = Path(path)
    pipeline_info = _with_root_normalization(pipeline_info, node)
    temp_dir, zarr_path = _resolve_zarr_path(path, compress)
    if not compress:
        # Crash-safety: write into a hidden temp sibling and atomically swap
        # into place at the end. The old in-place ``overwrite=True`` cleared a
        # pre-existing store at ``path`` BEFORE writing, so a mid-write crash
        # destroyed the prior good copy and left a partial store behind (which
        # existence-gated consumers like the batch-merge resume then treated
        # as complete).
        zarr_path = _tmp_sibling(path)

    store = open_store(zarr_path, mode="w")
    root = create_root_group(store, overwrite=True)
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
            attrs=dict(root_attrs) if root_attrs else None,
            warn_on_missing_tone_mapping=False,
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
        # the scene builders instead and does not carry this root attr). The value
        # is single-sourced with the READER that has to recognise it as a stamp
        # rather than as an authored choice (`agreed_authored_appearance`).
        root.attrs.setdefault("layer", WRITER_STAMPED_APPEARANCE_DEFAULTS["layer"])
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

        # One colormap window per gsplat structure (#1691) — before the hash so
        # the stamp covers the corrected attrs.
        harmonize_gsplat_amplitude_windows(root)

        # Stamp BEFORE consolidating so the hash lands in ``.zmetadata`` too.
        _stamp_content_hash(root)
        consolidate(root)

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


def _stamp_optional_bsp_tree(
    root: Any, provider: Optional[Callable[[], Optional[Dict[str, Any]]]]
) -> None:
    """Write the root's optional ``bsp_tree`` attr from a (possibly absent) provider.

    Both "no provider" and "provider returned nothing" mean the same thing to a
    reader — no split planes, fall back to a centroid order — so they collapse
    here rather than at the call site.
    """
    if provider is None:
        return
    tree = provider()
    if tree is not None:
        root.attrs["bsp_tree"] = tree


def _resolve_metadata_provider(
    value: Optional[Dict[str, Any] | Callable[[], Optional[Dict[str, Any]]]],
) -> Optional[Dict[str, Any]]:
    return value() if callable(value) else value


def write_partition_streaming(
    path: str | Path,
    part_nodes: Callable[[], Iterator["GSplatNode"]],
    *,
    max_elements: int = 0,
    ordering: Literal["morton", "hilbert", "none"] = "hilbert",
    encoding_mode: EncodingMode = EncodingMode.AUTO,
    fitting_info: Optional[
        Dict[str, Any] | Callable[[], Optional[Dict[str, Any]]]
    ] = None,
    fitting_config: Optional[Dict[str, Any]] = None,
    provenance_info: Optional[Dict[str, Any]] = None,
    pipeline_info: Optional[Dict[str, Any]] = None,
    description: Optional[str] = None,
    compressor: Optional[Any] = DEFAULT_COMP,
    barrier_dims: Optional[Sequence[int]] = None,
    bsp_tree: Optional[Callable[[], Optional[Dict[str, Any]]]] = None,
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
    ``fitting_info`` may be a value or a provider; a provider is called once after
    the part loop, allowing metadata to describe the parts that actually survived.
    ``bsp_tree`` is likewise a provider, and whatever it returns is written as the
    root's optional split-plane attr (the same one the standalone branch writes).
    A callable is needed because the
    surviving part set — the thing the tree's leaf labels must be renumbered
    against — is only known once the producer has finished skipping empty regions,
    so the caller prunes inside the provider. Omit it, or return ``None``, and the
    viewer falls back to a per-part centroid order, which is not a valid painter's
    order and pops at the seams under order-dependent blending (#1555).

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
    store = open_store(tmp, mode="w")
    root = create_root_group(store, overwrite=True)
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
                warn_on_missing_tone_mapping=False,
                # This writer's ROOT is stamped kind=partition below, so every
                # part_<i> is under a partition by construction — exactly what the
                # GSplatPartition branch of write_gsplat_node passes. Without it a
                # meta-less per-part lod ladder would fall back to the whole-object
                # anchor here while the standalone writer gave the partitioned one.
                under_partition=True,
            )
            if "position_bounds" in cmeta:
                child_bounds.append(cmeta["position_bounds"])
            n_written += 1

        if n_written == 0:
            raise ValueError("write_partition_streaming: no non-empty parts to write")

        # Root partition attrs — same set the GSplatPartition branch of
        # write_gsplat_node emits (type/kind/display_type/max_elements/position_bounds
        # + the optional bsp_tree).
        root.attrs["type"] = "group"
        root.attrs["kind"] = "partition"
        root.attrs["display_type"] = "gsplats"
        root.attrs["max_elements"] = int(max_elements)
        bounds = _union_bounds(child_bounds)
        if bounds is not None:
            root.attrs["position_bounds"] = bounds
        # Resolved only now: the provider needs the surviving part set, which the
        # loop above has just finished determining.
        _stamp_optional_bsp_tree(root, bsp_tree)

        # Self-identifying v3.0 header (disjoint from the node's structural attrs).
        root.attrs["format_version"] = FORMAT_VERSION
        root.attrs["format_type"] = "gsplats_zarr"
        root.attrs["timestamp"] = datetime.datetime.now(
            datetime.timezone.utc
        ).isoformat()
        root.attrs["luxar_gsplats_version"] = GSPLATS_VERSION
        root.attrs.setdefault("layer", WRITER_STAMPED_APPEARANCE_DEFAULTS["layer"])
        if description:
            root.attrs["description"] = description

        resolved_fitting_info = _resolve_metadata_provider(fitting_info)
        if resolved_fitting_info is not None:
            fitting_group = root.create_group("fitting")
            fitting_group.attrs.update(resolved_fitting_info)
            if fitting_config is not None:
                fitting_group.create_group("config").attrs.update(fitting_config)
        if provenance_info is not None:
            root.create_group("provenance").attrs.update(provenance_info)
        if pipeline_info:
            root.create_group("pipeline").attrs.update(pipeline_info)

        # One colormap window per gsplat structure (#1691) — before the hash so
        # the stamp covers the corrected attrs.
        harmonize_gsplat_amplitude_windows(root)

        # Stamp BEFORE consolidating so the hash lands in ``.zmetadata`` too.
        _stamp_content_hash(root)
        consolidate(root)
    except BaseException:
        # Includes the n_written == 0 ValueError above — no partial root is
        # left behind either way; the destination stays untouched.
        shutil.rmtree(tmp, ignore_errors=True)
        raise
    _atomic_finalize(tmp, path)
    return n_written


def write_flat_leaf_streaming(
    path: str | Path,
    splat_sets: Callable[[], Iterator["AdditiveSubLOD"]],
    *,
    encoding_mode: EncodingMode = EncodingMode.PRECISION,
    fitting_info: Optional[Dict[str, Any]] = None,
    fitting_config: Optional[Dict[str, Any]] = None,
    provenance_info: Optional[Dict[str, Any]] = None,
    pipeline_info: Optional[Dict[str, Any]] = None,
    description: Optional[str] = None,
    compressor: Optional[Any] = DEFAULT_COMP,
    barrier_dims: Optional[Sequence[int]] = None,
    root_attrs: Optional[Dict[str, Any]] = None,
    compress: Optional[Literal["zip", "tar.gz"]] = None,
    zip_deflate: bool = True,
) -> int:
    """Write one flat leaf from a replayable stream, holding one set in memory.

    The decoded source sets are spatially ordered independently and concatenated
    into disk-backed staging arrays. The final leaf therefore has a piecewise
    Hilbert order without the whole-array permutation required by the regular
    writer. The canonical array writer still owns the on-disk encoding, chunk
    layout, metadata, appearance defaults, and format stamps.

    ``splat_sets`` is consumed twice: once to validate the flat-shape contract
    and size the staging arrays, then once to populate them. Callers must return
    a fresh iterator each time. Streaming currently requires precision encoding;
    auto/memory need whole-array encoding analysis and would defeat the memory
    bound this entry point exists to provide.
    """
    if encoding_mode != EncodingMode.PRECISION:
        raise ValueError(
            "write_flat_leaf_streaming requires encoding_mode='precision'; "
            "auto/memory encoding needs whole-array analysis"
        )

    from luxar.gsplats.gsplat_data import widen_colors_to_rgba
    from luxar.gsplats.utils.trils import tril_size
    from luxar.io._compiler.gsplat_assembly import (
        apply_gsplat_group_attrs,
        apply_gsplat_spatial_ordering,
        write_gsplat_arrays,
    )
    from luxar.io._ordering.gsplats import compute_chunk_bounds_gsplats

    n_splats = 0
    ndim: Optional[int] = None
    truncation_radius: Optional[float] = None
    color_channels = 0
    color_dtypes: set[np.dtype[Any]] = set()
    any_colors = False
    any_missing_colors = False
    for sublod in splat_sets():
        if ndim is None:
            ndim = sublod.ndim
            truncation_radius = float(sublod.truncation_radius)
        elif sublod.ndim != ndim:
            raise ValueError(
                f"streamed splat sets must share one dimensionality; got {ndim}D "
                f"and {sublod.ndim}D"
            )
        elif float(sublod.truncation_radius) != truncation_radius:
            raise ValueError(
                "streamed splat sets must share one truncation_radius; got "
                f"{truncation_radius} and {sublod.truncation_radius}"
            )
        n_splats += sublod.n_splats
        if sublod.colors is None:
            any_missing_colors = True
        else:
            any_colors = True
            color_channels = max(color_channels, int(sublod.colors.shape[1]))
            color_dtypes.add(sublod.colors.dtype)

    if ndim is None or n_splats == 0:
        raise ValueError("write_flat_leaf_streaming: no non-empty splat sets to write")
    assert truncation_radius is not None

    path = Path(path)
    temp_dir, zarr_path = _resolve_zarr_path(path, compress)
    if temp_dir is None:
        zarr_path = _tmp_sibling(path)
    stage_parent = temp_dir if temp_dir is not None else path.parent

    try:
        with tempfile.TemporaryDirectory(
            prefix="luxar_gsplat_flatten_", dir=stage_parent
        ) as stage:
            stage_path = Path(stage)
            centers = np.memmap(
                stage_path / "centers.dat",
                mode="w+",
                dtype=np.float32,
                shape=(n_splats, ndim),
            )
            amplitudes = np.memmap(
                stage_path / "amplitudes.dat",
                mode="w+",
                dtype=np.float32,
                shape=(n_splats,),
            )
            cholesky = np.memmap(
                stage_path / "cholesky.dat",
                mode="w+",
                dtype=np.float32,
                shape=(n_splats, tril_size(ndim)),
            )
            preserve_integer_colors = (
                not any_missing_colors
                and len(color_dtypes) == 1
                and np.issubdtype(next(iter(color_dtypes)), np.integer)
            )
            color_dtype = (
                next(iter(color_dtypes)) if preserve_integer_colors else np.float32
            )
            colors = (
                np.memmap(
                    stage_path / "colors.dat",
                    mode="w+",
                    dtype=color_dtype,
                    shape=(n_splats, color_channels),
                )
                if any_colors
                else None
            )

            dataset_ctx = make_dataset_ctx(encoding_mode, compressor=compressor)
            ordering_ctx = make_ordering_ctx("hilbert")
            offset = 0
            ordering_template: Optional[Dict[str, Any]] = None
            for sublod in splat_sets():
                ordered = apply_gsplat_spatial_ordering(
                    sublod.centers,
                    sublod.amplitudes,
                    sublod.cholesky_factors,
                    sublod.colors,
                    sublod.n_splats,
                    sublod.ndim,
                    False,
                    ordering_ctx,
                    coverage_sigma=float(sublod.truncation_radius),
                    barrier_dims=barrier_dims,
                    dataset_ctx=dataset_ctx,
                )
                part_centers, part_amplitudes, part_cholesky, part_colors, meta, _ = (
                    ordered
                )
                stop = offset + sublod.n_splats
                centers[offset:stop] = part_centers
                amplitudes[offset:stop] = part_amplitudes
                cholesky[offset:stop] = part_cholesky
                if colors is not None:
                    if part_colors is None:
                        normalized = np.ones(
                            (sublod.n_splats, color_channels), dtype=np.float32
                        )
                    else:
                        normalized = np.asarray(part_colors)
                        if not preserve_integer_colors:
                            if np.issubdtype(normalized.dtype, np.integer):
                                integer_max = np.iinfo(cast(Any, normalized.dtype)).max
                                normalized = normalized.astype(np.float32) / np.float32(
                                    integer_max
                                )
                            else:
                                normalized = normalized.astype(np.float32, copy=False)
                        if color_channels == 4 and normalized.shape[1] == 3:
                            normalized = widen_colors_to_rgba(normalized)
                    colors[offset:stop] = normalized
                if meta is not None:
                    if ordering_template is None:
                        ordering_template = meta
                    elif meta.get("slice_dims") != ordering_template.get(
                        "slice_dims"
                    ) or meta.get("ordering_dims") != ordering_template.get(
                        "ordering_dims"
                    ):
                        raise ValueError(
                            "streamed splat sets resolved inconsistent ordering axes"
                        )
                offset = stop
                del sublod, part_centers, part_amplitudes, part_cholesky, part_colors
                if colors is not None:
                    del normalized

            assert ordering_template is not None
            centers.flush()
            amplitudes.flush()
            cholesky.flush()
            if colors is not None:
                colors.flush()

            chunk_size = int(ordering_template["chunk_size"])
            ordering_dims = list(ordering_template.get("ordering_dims", []))
            ordering_data = {
                "ordering": "hilbert",
                "ordering_min": np.min(centers[:, ordering_dims], axis=0).tolist(),
                "ordering_max": np.max(centers[:, ordering_dims], axis=0).tolist(),
                "ordering_bits_per_dim": ordering_template["ordering_bits_per_dim"],
                "chunk_size": chunk_size,
                "slice_dims": list(ordering_template.get("slice_dims", [])),
                "ordering_dims": ordering_dims,
                "chunk_bounds": compute_chunk_bounds_gsplats(
                    centers,
                    cholesky,
                    chunk_size,
                    coverage_sigma=float(truncation_radius),
                    slice_dims=ordering_template.get("slice_dims", []),
                ),
            }

            store = open_store(zarr_path, mode="w")
            root = create_root_group(store, overwrite=True)
            cholesky_is_uniform = bool(np.all(cholesky == cholesky[0]))
            metadata = write_gsplat_arrays(
                root,
                centers,
                amplitudes,
                cholesky,
                colors,
                n_splats,
                ndim,
                cholesky_is_uniform,
                ordering_data,
                None,
                dataset_ctx,
            )
            attrs = dict(root_attrs or {})
            attrs.setdefault("truncation_radius", float(truncation_radius))
            apply_gsplat_group_attrs(
                root,
                metadata,
                attrs,
                root,
                scene_tone_mapping=None,
                lut_tone_mapping_warned=False,
                warn_on_missing_tone_mapping=False,
            )
            root.attrs["format_version"] = FORMAT_VERSION
            root.attrs["format_type"] = "gsplats_zarr"
            root.attrs["timestamp"] = datetime.datetime.now(
                datetime.timezone.utc
            ).isoformat()
            root.attrs["luxar_gsplats_version"] = GSPLATS_VERSION
            root.attrs.setdefault("layer", WRITER_STAMPED_APPEARANCE_DEFAULTS["layer"])
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
            harmonize_gsplat_amplitude_windows(root)
            _stamp_content_hash(root)
            consolidate(root)

        if compress:
            _compress_zarr(zarr_path, path, compress, zip_deflate, temp_dir)
        else:
            _atomic_finalize(zarr_path, path)
        return n_splats
    except BaseException:
        if temp_dir is not None:
            shutil.rmtree(temp_dir, ignore_errors=True)
        else:
            shutil.rmtree(zarr_path, ignore_errors=True)
        raise


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
    truncation_radius: float = DEFAULT_TRUNCATION_RADIUS,
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
