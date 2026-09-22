"""Save Gaussian splats to the ``.gsplats.zarr`` node-tree format.

A standalone ``.gsplats.zarr`` is a **detached scene-node subtree** written by the
single shared authoring path
(:func:`luxar.io._compiler.gsplat_tree.write_gsplat_node`) — the same machinery the
scene compiler uses for its gsplats leaves, so there is no parallel writer and
standalone leaves are byte-identical to scene leaves.

This module provides the standalone tree writer plus streaming partition and
flat-leaf entry points. It builds the zarr store (handling optional ``.zip`` /
``.tar.gz`` compression), writes the self-identifying root header
(``format_version`` = :data:`FORMAT_VERSION`), attaches optional ``fitting/`` /
``provenance/`` / ``pipeline/`` groups (``pipeline/`` = reduction/topology
stats; see :func:`split_fitting_info`), and consolidates metadata.
"""

from __future__ import annotations

import datetime
import os
import shutil
import tempfile
import uuid
from dataclasses import dataclass
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
    make_dataset_ctx,
    make_ordering_ctx,
    write_gsplat_node,
)
from luxar.io.reader import DEFAULT_COMP
from luxar.typing_utils._format_contract import (
    GSPLATS_FORMAT_VERSION,
    SUPPORTED_GSPLATS_VERSIONS,
    OrderingMethodName,
)
from luxar.typing_utils.constants import DEFAULT_TRUNCATION_RADIUS
from luxar.typing_utils.json_safe import json_safe_value
from luxar.utils.arbol_warnings import arbol_warnings
from luxar.utils.paths import normalize_zarr_path

_MAX_STREAMING_BARRIER_RUNS = 1_000_000


def resolve_amplitude_bits(
    amplitude_bits: Literal["auto", 8, 16],
    fitting_info: Optional[Dict[str, Any]] = None,
    *,
    source_dtype: Optional[str] = None,
) -> Literal[8, 16]:
    """Resolve the AUTO amplitude tier from source dtype metadata."""
    if amplitude_bits in (8, 16):
        return amplitude_bits
    if amplitude_bits != "auto":
        raise ValueError("amplitude_bits must be 'auto', 8, or 16")

    if source_dtype is None and fitting_info:
        source_dtype = fitting_info.get("source_dtype")
    try:
        dtype = np.dtype(source_dtype)
    except (TypeError, ValueError):
        return 16
    return 8 if dtype.kind in "iu" and dtype.itemsize == 1 else 16


@dataclass(frozen=True)
class StreamingSplatSetMetadata:
    """Metadata needed to size a streamed flat-leaf write without decoding it."""

    n_splats: int
    ndim: int
    truncation_radius: float
    color_channels: int = 0
    color_dtype: Optional[np.dtype[Any]] = None
    label_dtype: Optional[np.dtype[Any]] = None
    label_vocabulary: Optional[Dict[int, str]] = None
    barrier_values: Optional[np.ndarray] = None
    barrier_counts: Optional[np.ndarray] = None


@dataclass(frozen=True)
class _StreamingLayout:
    n_splats: int
    ndim: int
    truncation_radius: float
    color_channels: int
    color_dtype: np.dtype[Any]
    any_colors: bool
    preserve_integer_colors: bool
    label_dtype: Optional[np.dtype[Any]]
    label_vocabulary: Optional[Dict[int, str]]


@dataclass
class _StreamingWriteState:
    offset: int = 0
    ordering_template: Optional[Dict[str, Any]] = None
    ordering_min: Optional[np.ndarray] = None
    ordering_max: Optional[np.ndarray] = None
    cholesky_reference: Optional[np.ndarray] = None
    cholesky_is_uniform: bool = True


def _resolve_streaming_labels(
    metadata: Sequence[StreamingSplatSetMetadata],
) -> tuple[Optional[np.dtype[Any]], Optional[Dict[int, str]]]:
    present = [item for item in metadata if item.label_dtype is not None]
    if present and len(present) != len(metadata):
        raise ValueError(
            "cannot stream label_ids when only some splat sets carry the channel"
        )
    if not present:
        return None, None
    if any(item.label_vocabulary is None for item in present):
        raise ValueError("streamed label_ids require label_vocabulary")
    vocabulary = present[0].label_vocabulary
    if any(item.label_vocabulary != vocabulary for item in present[1:]):
        raise ValueError(
            "cannot stream label_ids with different label_vocabulary values"
        )
    return np.result_type(*(item.label_dtype for item in present)), vocabulary


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
    "configured_iterations",
    "dynamic_ops_enabled",
    "dynamic_ops_step_every",
    "dynamic_ops_k_max_residuals",
    "dynamic_ops_relocation_events",
    "dynamic_ops_unique_splats_relocated",
    "scale_diagnostic_tolerance_vox",
    "fit_init_sigma_vox",
    "fit_init_sigma_diag_vox",
    "fit_init_marginal_sigma_diag_vox_min",
    "fit_init_marginal_sigma_diag_vox_median",
    "fit_init_marginal_sigma_diag_vox_max",
    "relocation_init_sigma_vox",
    "sigma_min_diag_vox",
    "splats_near_fit_init_sigma_count",
    "splats_near_fit_init_sigma_fraction",
    "splats_near_relocation_init_sigma_count",
    "splats_near_relocation_init_sigma_fraction",
    "splats_near_sigma_min_count",
    "splats_near_sigma_min_fraction",
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
    and this one carries an extra obligation: it is what the four IN-PLACE
    re-stampers (single-tile batch merge, ``gsplat annotate-quality``,
    ``gsplat doctor --fix``, and ``luxar restamp-lod``) write, and they leave
    the per-save ``timestamp``
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

    from luxar.io._compiler.finalize.hashing import (
        _canonicalized_group_attrs,
        codec_ids,
    )

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
        attrs = {
            k: v
            for k, v in _canonicalized_group_attrs(group).items()
            if k != "content_hash"
        }
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
    ordering: OrderingMethodName = "hilbert",
    encoding_mode: EncodingMode = EncodingMode.AUTO,
    amplitude_bits: Literal["auto", 8, 16] = 16,
    source_dtype: Optional[str] = None,
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

    ``amplitude_bits="auto"`` resolves from the explicit ``source_dtype`` when
    provided, otherwise from ``fitting_info["source_dtype"]``. The explicit
    argument keeps encoding independent of whether fitting metadata is persisted.
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

        dataset_ctx = make_dataset_ctx(
            encoding_mode,
            compressor=compressor,
            positive_scalar_bits=resolve_amplitude_bits(
                amplitude_bits, fitting_info, source_dtype=source_dtype
            ),
        )
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

        artifact_n_splats = None
        if fitting_info is not None and "n_splats" in fitting_info:
            from luxar.gsplats.tree import total_splats

            artifact_n_splats = int(total_splats(node))
        _finalize_gsplat_root(
            root,
            artifact_n_splats=artifact_n_splats,
            fitting_info=fitting_info,
            fitting_config=fitting_config,
            provenance_info=provenance_info,
            pipeline_info=pipeline_info,
            description=description,
        )

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


def _finalize_gsplat_root(
    root: zarr.Group,
    *,
    artifact_n_splats: Optional[int],
    fitting_info: Optional[
        Dict[str, Any] | Callable[[], Optional[Dict[str, Any]]]
    ] = None,
    fitting_config: Optional[Dict[str, Any]] = None,
    provenance_info: Optional[Dict[str, Any]] = None,
    pipeline_info: Optional[Dict[str, Any]] = None,
    description: Optional[str] = None,
) -> None:
    """Attach root metadata, then hash and consolidate in contract order.

    Appearance harmonization must precede the content hash, and the hash must
    precede consolidation so corrected attrs and the hash itself land in the
    consolidated metadata used by every standalone writer.
    """
    # Self-identifying v3.0 header (disjoint from the node's structural attrs).
    root.attrs["format_version"] = FORMAT_VERSION
    root.attrs["format_type"] = "gsplats_zarr"
    root.attrs["timestamp"] = datetime.datetime.now(datetime.timezone.utc).isoformat()
    root.attrs["luxar_gsplats_version"] = GSPLATS_VERSION
    # A standalone file opened directly is the whole layer, so expose its root
    # in the Layers panel. Single-source the value with the reader that must
    # distinguish this writer stamp from an authored choice.
    root.attrs.setdefault("layer", WRITER_STAMPED_APPEARANCE_DEFAULTS["layer"])
    if description:
        root.attrs["description"] = description

    resolved_fitting_info = _resolve_metadata_provider(fitting_info)
    if resolved_fitting_info is not None:
        # `n_splats` is the count in THIS artifact (see fitting/results.py).
        # Count-changing LOD inherits a stale source-fit value unless corrected.
        if artifact_n_splats is not None and "n_splats" in resolved_fitting_info:
            resolved_fitting_info = {
                **resolved_fitting_info,
                "n_splats": artifact_n_splats,
            }
        fitting_group = root.create_group("fitting")
        fitting_group.attrs.update(resolved_fitting_info)
        if fitting_config is not None:
            fitting_group.create_group("config").attrs.update(fitting_config)
    if provenance_info is not None:
        root.create_group("provenance").attrs.update(provenance_info)
    if pipeline_info:
        # Reduction/topology stats: everything split_fitting_info's other
        # buckets do not consume. Optional group: absent for plain fits.
        root.create_group("pipeline").attrs.update(pipeline_info)

    # One colormap window per gsplat structure (#1691) — before the hash so
    # the stamp covers the corrected attrs.
    harmonize_gsplat_amplitude_windows(root)

    # Stamp BEFORE consolidating so the hash lands in `.zmetadata` too.
    _stamp_content_hash(root)
    consolidate(root)


def _resolve_streaming_layout(
    metadata: Sequence[StreamingSplatSetMetadata],
) -> _StreamingLayout:
    n_splats = 0
    ndim: Optional[int] = None
    truncation_radius: Optional[float] = None
    color_channels = 0
    color_dtypes: set[np.dtype[Any]] = set()
    any_colors = False
    any_missing_colors = False
    for item in metadata:
        if ndim is None:
            ndim = item.ndim
            truncation_radius = float(item.truncation_radius)
        elif item.ndim != ndim:
            raise ValueError(
                f"streamed splat sets must share one dimensionality; got {ndim}D "
                f"and {item.ndim}D"
            )
        elif float(item.truncation_radius) != truncation_radius:
            raise ValueError(
                "streamed splat sets must share one truncation_radius; got "
                f"{truncation_radius} and {item.truncation_radius}"
            )
        n_splats += item.n_splats
        if item.color_dtype is None:
            any_missing_colors = True
        else:
            any_colors = True
            color_channels = max(color_channels, item.color_channels)
            color_dtypes.add(item.color_dtype)
    if ndim is None or truncation_radius is None or n_splats == 0:
        raise ValueError("write_flat_leaf_streaming: no non-empty splat sets to write")
    preserve_integer_colors = (
        not any_missing_colors
        and len(color_dtypes) == 1
        and np.issubdtype(next(iter(color_dtypes)), np.integer)
    )
    color_dtype = (
        next(iter(color_dtypes)) if preserve_integer_colors else np.dtype(np.float32)
    )
    label_dtype, label_vocabulary = _resolve_streaming_labels(metadata)
    return _StreamingLayout(
        n_splats=n_splats,
        ndim=ndim,
        truncation_radius=truncation_radius,
        color_channels=color_channels,
        color_dtype=color_dtype,
        any_colors=any_colors,
        preserve_integer_colors=preserve_integer_colors,
        label_dtype=label_dtype,
        label_vocabulary=label_vocabulary,
    )


def _resolve_streaming_barrier_offsets(
    metadata: Sequence[StreamingSplatSetMetadata], barrier_dims: Sequence[int]
) -> Optional[dict[tuple[float, ...], int]]:
    """Map exact barrier tuples, ordered by sorted barrier dimensions, to offsets."""
    if not barrier_dims:
        return None
    totals: dict[tuple[float, ...], int] = {}
    for item in metadata:
        values = item.barrier_values
        counts = item.barrier_counts
        if values is None or counts is None:
            raise ValueError(
                "barrier_values and barrier_counts are required when barrier_dims "
                "are non-empty"
            )
        if values.ndim != 2 or values.shape[1] != len(barrier_dims):
            raise ValueError("streamed barrier values do not match barrier_dims")
        if counts.shape != (len(values),):
            raise ValueError("streamed barrier counts do not match barrier values")
        if int(np.sum(counts)) != item.n_splats:
            raise ValueError("streamed barrier counts do not match n_splats")
        for value, count in zip(values, counts):
            key = tuple(float(component) for component in value)
            _add_streaming_barrier_total(
                totals, key, int(count), _MAX_STREAMING_BARRIER_RUNS
            )
    offsets = {}
    next_offset = 0
    for key in sorted(totals):
        offsets[key] = next_offset
        next_offset += totals[key]
    return offsets


def _add_streaming_barrier_total(
    totals: dict[tuple[float, ...], int],
    key: tuple[float, ...],
    count: int,
    max_runs: int,
) -> None:
    """Accumulate one barrier run while enforcing the dictionary size bound."""
    totals[key] = totals.get(key, 0) + count
    if len(totals) > max_runs:
        raise ValueError(
            "streamed barrier values exceed the allocation limit of "
            f"{max_runs} exact value tuples"
        )


def _normalize_stream_colors(
    colors: Optional[np.ndarray],
    *,
    n_splats: int,
    color_channels: int,
    preserve_integer: bool,
) -> np.ndarray:
    from luxar.gsplats.gsplat_data import widen_colors_to_rgba

    if colors is None:
        return np.ones((n_splats, color_channels), dtype=np.float32)
    normalized = np.asarray(colors)
    if not preserve_integer:
        if np.issubdtype(normalized.dtype, np.integer):
            integer_max = np.iinfo(cast(Any, normalized.dtype)).max
            normalized = normalized.astype(np.float32) / np.float32(integer_max)
        else:
            normalized = normalized.astype(np.float32, copy=False)
    if color_channels == 4 and normalized.shape[1] == 3:
        normalized = widen_colors_to_rgba(normalized)
    return normalized


def _stream_destinations(
    metadata: StreamingSplatSetMetadata,
    barrier_offsets: Optional[dict[tuple[float, ...], int]],
    offset: int,
) -> list[tuple[int, int, int]]:
    """Map sorted source runs to barrier-major output ranges.

    Metadata run order must match ``_compound_sort``'s lexsort key order.
    """
    if barrier_offsets is None:
        return [(offset, 0, metadata.n_splats)]
    assert metadata.barrier_values is not None
    assert metadata.barrier_counts is not None
    destinations = []
    source_start = 0
    for value, count in zip(metadata.barrier_values, metadata.barrier_counts):
        key = tuple(float(component) for component in value)
        destination_start = barrier_offsets[key]
        source_stop = source_start + int(count)
        destinations.append((destination_start, source_start, source_stop))
        barrier_offsets[key] += int(count)
        source_start = source_stop
    if source_start != metadata.n_splats:
        raise ValueError("streamed barrier runs do not cover the splat set")
    return destinations


def _cholesky_is_uniform(cholesky: np.ndarray, reference: np.ndarray) -> bool:
    return all(
        np.all(cholesky[start : start + 65_536] == reference)
        for start in range(0, len(cholesky), 65_536)
    )


def _write_streamed_splat_set(
    sublod: "AdditiveSubLOD",
    metadata: StreamingSplatSetMetadata,
    *,
    state: _StreamingWriteState,
    barrier_dims: Sequence[int],
    barrier_offsets: Optional[dict[tuple[float, ...], int]],
    layout: _StreamingLayout,
    centers: np.ndarray,
    amplitudes: np.ndarray,
    cholesky: np.ndarray,
    colors: Optional[np.ndarray],
    label_ids: Optional[np.ndarray],
) -> None:
    from luxar.gsplats.gsplat_data import validate_label_channel
    from luxar.io.ordering import sort_splats_spatial

    if (
        sublod.n_splats != metadata.n_splats
        or sublod.ndim != metadata.ndim
        or float(sublod.truncation_radius) != metadata.truncation_radius
        or (sublod.label_ids is None) != (metadata.label_dtype is None)
        or sublod.label_vocabulary != metadata.label_vocabulary
    ):
        raise ValueError("streamed splat set does not match its metadata")
    validate_label_channel(sublod.label_ids, sublod.label_vocabulary, sublod.n_splats)
    sort_indices, ordering = sort_splats_spatial(
        sublod.centers,
        method="hilbert",
        slice_dims=barrier_dims,
    )
    part_centers = sublod.centers[sort_indices]
    part_amplitudes = sublod.amplitudes[sort_indices]
    part_cholesky = sublod.cholesky_factors[sort_indices]
    part_colors = sublod.colors[sort_indices] if sublod.colors is not None else None
    part_label_ids = (
        sublod.label_ids[sort_indices] if sublod.label_ids is not None else None
    )
    normalized = None
    if colors is not None:
        normalized = _normalize_stream_colors(
            part_colors,
            n_splats=sublod.n_splats,
            color_channels=layout.color_channels,
            preserve_integer=layout.preserve_integer_colors,
        )
    for destination_start, source_start, source_stop in _stream_destinations(
        metadata, barrier_offsets, state.offset
    ):
        destination_stop = destination_start + source_stop - source_start
        centers[destination_start:destination_stop] = part_centers[
            source_start:source_stop
        ]
        amplitudes[destination_start:destination_stop] = part_amplitudes[
            source_start:source_stop
        ]
        cholesky[destination_start:destination_stop] = part_cholesky[
            source_start:source_stop
        ]
        if colors is not None and normalized is not None:
            colors[destination_start:destination_stop] = normalized[
                source_start:source_stop
            ]
        if label_ids is not None and part_label_ids is not None:
            label_ids[destination_start:destination_stop] = part_label_ids[
                source_start:source_stop
            ]

    if state.ordering_template is None:
        state.ordering_template = ordering
        ordering_dims = list(ordering.get("ordering_dims", []))
        state.ordering_min = np.full(len(ordering_dims), np.inf)
        state.ordering_max = np.full(len(ordering_dims), -np.inf)
    ordering_dims = list(state.ordering_template.get("ordering_dims", []))
    assert state.ordering_min is not None and state.ordering_max is not None
    for index, dim in enumerate(ordering_dims):
        state.ordering_min[index] = min(
            state.ordering_min[index], float(np.min(part_centers[:, dim]))
        )
        state.ordering_max[index] = max(
            state.ordering_max[index], float(np.max(part_centers[:, dim]))
        )

    if state.cholesky_reference is None:
        state.cholesky_reference = part_cholesky[0].copy()
    if state.cholesky_is_uniform:
        state.cholesky_is_uniform = _cholesky_is_uniform(
            part_cholesky, state.cholesky_reference
        )
    state.offset += sublod.n_splats


def _next_streamed_splat_set(stream: Iterator["AdditiveSubLOD"]) -> "AdditiveSubLOD":
    try:
        return next(stream)
    except StopIteration as exc:
        raise ValueError("splat_sets ended before splat_set_metadata") from exc


def _require_stream_exhausted(stream: Iterator["AdditiveSubLOD"]) -> None:
    try:
        next(stream)
    except StopIteration:
        return
    raise ValueError("splat_sets yielded more entries than its metadata")


def write_partition_streaming(
    path: str | Path,
    part_nodes: Callable[[], Iterator["GSplatNode"]],
    *,
    max_elements: int = 0,
    ordering: OrderingMethodName = "hilbert",
    encoding_mode: EncodingMode = EncodingMode.AUTO,
    amplitude_bits: Literal["auto", 8, 16] = 16,
    source_dtype: Optional[str] = None,
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
    root_attrs: Optional[Dict[str, Any]] = None,
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
    ``amplitude_bits="auto"`` resolves from ``source_dtype`` before writing; when
    fitting metadata is a provider, the explicit dtype is required because the
    provider is intentionally not called until all parts have been written.
    ``bsp_tree`` is likewise a provider, and whatever it returns is written as the
    root's optional split-plane attr (the same one the standalone branch writes).
    A callable is needed because the
    surviving part set — the thing the tree's leaf labels must be renumbered
    against — is only known once the producer has finished skipping empty regions,
    so the caller prunes inside the provider. Omit it, or return ``None``, and the
    viewer falls back to a per-part centroid order, which is not a valid painter's
    order and pops at the seams under order-dependent blending (#1555).

    ``root_attrs`` seeds optional root metadata before the structural partition
    attrs are stamped, so caller metadata cannot override the node kind.

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
    if amplitude_bits == "auto" and callable(fitting_info) and source_dtype is None:
        raise ValueError(
            "source_dtype is required when amplitude_bits='auto' and "
            "fitting_info is callable"
        )
    static_fitting_info = fitting_info if isinstance(fitting_info, dict) else None
    resolved_amplitude_bits = resolve_amplitude_bits(
        amplitude_bits, static_fitting_info, source_dtype=source_dtype
    )
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
        dataset_ctx = make_dataset_ctx(
            encoding_mode,
            compressor=compressor,
            positive_scalar_bits=resolved_amplitude_bits,
        )
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
        if root_attrs:
            root.attrs.update(root_attrs)
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

        _finalize_gsplat_root(
            root,
            artifact_n_splats=None,
            fitting_info=fitting_info,
            fitting_config=fitting_config,
            provenance_info=provenance_info,
            pipeline_info=pipeline_info,
            description=description,
        )
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
    splat_set_metadata: Sequence[StreamingSplatSetMetadata],
    encoding_mode: EncodingMode = EncodingMode.PRECISION,
    fitting_info: Optional[Dict[str, Any]] = None,
    fitting_config: Optional[Dict[str, Any]] = None,
    provenance_info: Optional[Dict[str, Any]] = None,
    pipeline_info: Optional[Dict[str, Any]] = None,
    description: Optional[str] = None,
    compressor: Optional[Any] = DEFAULT_COMP,
    barrier_dims: Sequence[int],
    root_attrs: Optional[Dict[str, Any]] = None,
    compress: Optional[Literal["zip", "tar.gz"]] = None,
    zip_deflate: bool = False,
) -> int:
    """Write one flat leaf from a stream, holding one decoded leaf in memory.

    The decoded source sets are spatially ordered independently and written into
    disk-backed staging arrays. Barrier values are grouped globally while each
    value retains a piecewise Hilbert order, avoiding the whole-array permutation
    required by the regular writer. The canonical array writer still owns the
    on-disk encoding, chunk layout, metadata, appearance defaults, and format
    stamps.

    ``splat_set_metadata`` sizes the staging arrays without decoding source data;
    ``splat_sets`` is consumed exactly once to populate them. Streaming currently
    requires precision encoding; auto/memory need whole-array encoding analysis
    and would defeat the memory bound this entry point exists to provide.
    """
    if encoding_mode != EncodingMode.PRECISION:
        raise ValueError(
            "write_flat_leaf_streaming requires encoding_mode='precision'; "
            "auto/memory encoding needs whole-array analysis"
        )

    from luxar.gsplats.utils.trils import tril_size
    from luxar.io._compiler.gsplat_assembly import (
        apply_gsplat_group_attrs,
        resolve_gsplat_chunk_size,
        write_gsplat_arrays,
    )
    from luxar.io._ordering.gsplats import compute_chunk_bounds_gsplats

    layout = _resolve_streaming_layout(splat_set_metadata)
    n_splats = layout.n_splats
    ndim = layout.ndim
    truncation_radius = layout.truncation_radius

    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temp_dir, zarr_path = _resolve_zarr_path(path, compress)
    if temp_dir is None:
        zarr_path = _tmp_sibling(path)
    stage_parent = path.parent

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
            colors = (
                np.memmap(
                    stage_path / "colors.dat",
                    mode="w+",
                    dtype=layout.color_dtype,
                    shape=(n_splats, layout.color_channels),
                )
                if layout.any_colors
                else None
            )
            label_ids = (
                np.memmap(
                    stage_path / "label_ids.dat",
                    mode="w+",
                    dtype=layout.label_dtype,
                    shape=(n_splats,),
                )
                if layout.label_dtype is not None
                else None
            )

            dataset_ctx = make_dataset_ctx(encoding_mode, compressor=compressor)
            chunk_size = resolve_gsplat_chunk_size(n_splats, ndim)
            resolved_slice_dims = list(barrier_dims)
            barrier_offsets = _resolve_streaming_barrier_offsets(
                splat_set_metadata, resolved_slice_dims
            )

            state = _StreamingWriteState()
            stream = iter(splat_sets())
            for stream_metadata in splat_set_metadata:
                sublod = _next_streamed_splat_set(stream)
                _write_streamed_splat_set(
                    sublod,
                    stream_metadata,
                    state=state,
                    barrier_dims=resolved_slice_dims,
                    barrier_offsets=barrier_offsets,
                    layout=layout,
                    centers=centers,
                    amplitudes=amplitudes,
                    cholesky=cholesky,
                    colors=colors,
                    label_ids=label_ids,
                )

            _require_stream_exhausted(stream)

            assert state.ordering_template is not None
            assert state.ordering_min is not None and state.ordering_max is not None
            centers.flush()
            amplitudes.flush()
            cholesky.flush()
            if colors is not None:
                colors.flush()
            if label_ids is not None:
                label_ids.flush()

            chunk_bounds = compute_chunk_bounds_gsplats(
                centers,
                cholesky,
                chunk_size,
                coverage_sigma=float(truncation_radius),
                slice_dims=resolved_slice_dims,
            )

            ordering_dims = list(state.ordering_template.get("ordering_dims", []))
            ordering_data = {
                "ordering": state.ordering_template["ordering"],
                "ordering_min": state.ordering_min.tolist(),
                "ordering_max": state.ordering_max.tolist(),
                "ordering_bits_per_dim": state.ordering_template[
                    "ordering_bits_per_dim"
                ],
                "chunk_size": chunk_size,
                "slice_dims": list(resolved_slice_dims),
                "ordering_dims": ordering_dims,
                "chunk_bounds": chunk_bounds,
            }

            store = open_store(zarr_path, mode="w")
            root = create_root_group(store, overwrite=True)
            metadata = write_gsplat_arrays(
                root,
                centers,
                amplitudes,
                cholesky,
                colors,
                label_ids,
                layout.label_vocabulary,
                n_splats,
                ndim,
                state.cholesky_is_uniform,
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
            _finalize_gsplat_root(
                root,
                artifact_n_splats=n_splats,
                fitting_info=fitting_info,
                fitting_config=fitting_config,
                provenance_info=provenance_info,
                pipeline_info=pipeline_info,
                description=description,
            )

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
    label_ids: Optional[np.ndarray] = None,
    label_vocabulary: Optional[Dict[int, str]] = None,
    ordering: OrderingMethodName = "hilbert",
    encoding_mode: EncodingMode = EncodingMode.AUTO,
    fitting_info: Optional[Dict[str, Any]] = None,
    fitting_config: Optional[Dict[str, Any]] = None,
    provenance_info: Optional[Dict[str, Any]] = None,
    description: Optional[str] = None,
    compress: Optional[Literal["zip", "tar.gz"]] = None,
    compressor: Optional[Any] = DEFAULT_COMP,
    zip_deflate: bool = False,
    truncation_radius: float = DEFAULT_TRUNCATION_RADIUS,
    amplitude_bits: Literal["auto", 8, 16] = 16,
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
                label_ids=label_ids,
                label_vocabulary=label_vocabulary,
                truncation_radius=truncation_radius,
            )
        ]
    )
    write_gsplats_tree(
        path,
        leaf,
        ordering=ordering,
        encoding_mode=encoding_mode,
        amplitude_bits=amplitude_bits,
        fitting_info=fitting_info,
        fitting_config=fitting_config,
        provenance_info=provenance_info,
        description=description,
        compress=compress,
        compressor=compressor,
        zip_deflate=zip_deflate,
    )
