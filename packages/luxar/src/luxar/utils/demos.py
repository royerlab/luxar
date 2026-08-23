"""luxar.demos – Demo scene generators and precomputed-data helpers for Luxar.

This module provides functions to generate various demo scenes for testing
and demonstration purposes, plus helpers for loading precomputed GSplat data
from Git LFS (shipped with the package) or a local cache.
"""

from __future__ import annotations

import hashlib
import pickle
import re
import shutil
import sys
import zipfile
import zlib
from contextlib import nullcontext
from pathlib import Path, PurePosixPath
from typing import Any, Callable, Final, NamedTuple, Optional, Union, overload

import numpy as np
from arbol import aprint, asection

from .._zarr_compat import read_node_attrs
from ..core.dimensions import Dimension, Dimensions
from ..typing_utils.aliases import PathLike
from ..typing_utils.config import check_dataset_size_warning
from .process import run_child_process


def _validate_zip_member_path(member: str) -> PurePosixPath:
    """Validate a zip member path before reading it from a bundle.

    Zip files always use POSIX-style separators. Reject absolute paths,
    parent-directory traversal, and backslashes to avoid platform-specific
    traversal surprises when bundles are created on Windows.
    """
    if "\\" in member:
        raise ValueError(f"Unsafe zip path with backslash separator: {member!r}")

    path = PurePosixPath(member)
    if path.is_absolute() or any(part == ".." for part in path.parts):
        raise ValueError(f"Unsafe zip path: {member!r}")
    if not path.parts or path.name in ("", "."):
        raise ValueError(f"Invalid zip path: {member!r}")
    return path


def _safe_extract_zip_member(
    zf: zipfile.ZipFile,
    member: str,
    destination: Path,
    *,
    target_name: str | None = None,
) -> Path:
    """Extract one validated zip member under ``destination``.

    The member's archive path is validated, and the final output path is
    resolved to ensure it remains inside ``destination``. ``target_name`` can
    be used to flatten bundle members into the cache root.
    """
    member_path = _validate_zip_member_path(member)
    output_name = target_name if target_name is not None else member_path.as_posix()
    output_path = destination / output_name
    destination_resolved = destination.resolve()
    output_resolved = output_path.resolve()

    try:
        output_resolved.relative_to(destination_resolved)
    except ValueError as exc:
        raise ValueError(f"Unsafe extraction target: {output_name!r}") from exc

    output_resolved.parent.mkdir(parents=True, exist_ok=True)
    with zf.open(member, "r") as src, output_resolved.open("wb") as dst:
        shutil.copyfileobj(src, dst)
    return output_resolved


def detect_device(verbose: bool = True) -> str:
    """Auto-detect the best available compute device (cuda > mps > cpu).

    Args:
        verbose: If True, print detected device via arbol.

    Returns:
        Device string: 'cuda', 'mps', or 'cpu'.
    """
    from luxar.gsplats.utils.device import resolve_torch_device

    device = str(resolve_torch_device())
    if verbose:
        aprint(
            {
                "cuda": "Using CUDA device",
                "mps": "Using MPS device (Metal acceleration)",
                "cpu": "Using CPU device",
            }.get(device, f"Using device: {device}")
        )
    return device


def warn_if_no_cuda_gpu() -> None:
    """Print a warning if no CUDA GPU is available.

    GSplat demos require significant GPU compute for fitting.  Running on
    CPU is orders of magnitude slower and generally impractical for
    production runs.  This function prints a prominent warning so users
    understand the hardware requirements before waiting hours for a CPU run.
    """
    try:
        import torch  # noqa: F401  # check PyTorch is importable

        from luxar.gsplats.utils.device import is_mps_available

        if torch.cuda.is_available():
            return  # All good
        device = "MPS" if is_mps_available() else "CPU"
    except ImportError:
        device = "CPU (PyTorch not installed)"

    aprint("")
    aprint("=" * 70)
    aprint("WARNING: No CUDA GPU detected — running on " + device)
    aprint("=" * 70)
    aprint("GSplat demos require a CUDA GPU for practical performance.")
    aprint("Without one, fitting can take hours instead of minutes.")
    if device.startswith("MPS"):
        aprint("MPS (Apple Metal) provides some acceleration but is much")
        aprint("slower than CUDA for Gaussian splatting workloads.")
    aprint("")
    aprint("Options:")
    aprint("  - Use a machine with an NVIDIA GPU (CUDA)")
    aprint("  - Run the default path (shipped precomputed data) instead of --recompute")
    aprint("  - Use --serve-only if a scene was already generated")
    aprint("=" * 70)
    aprint("")


def print_data_provenance(
    *, title: str, source: str, license: str, url: str, note: str = ""
) -> None:
    """Print a dataset provenance/licence notice before a runtime download.

    Demos that fetch third-party data at runtime print this first, so the user
    sees the source and licence terms of what is about to be downloaded (Luxar
    itself redistributes none of it). See the DATA SOURCE & CITATION docstring
    block each demo also carries.
    """
    aprint("")
    aprint("─" * 70)
    aprint(f"  Dataset: {title}")
    aprint(f"  Source:  {source}")
    aprint(f"  License: {license}")
    aprint(f"  URL:     {url}")
    if note:
        aprint(f"  Note:    {note}")
    aprint("─" * 70)
    aprint("")


# =============================================================================
# Precomputed Data Helpers
# =============================================================================

# Directory containing precomputed data shipped with the package (via Git LFS)
_DEMOS_DATA_DIR = Path(__file__).resolve().parent.parent / "demos" / "data"

# Default user-level cache
_DEFAULT_CACHE_ROOT = Path.home() / ".cache" / "luxar"


class BundleMemberNotFound(FileNotFoundError):
    """A requested frame is not inside an otherwise perfectly good bundle.

    The bundle itself resolved, verified and opened; only the per-frame member
    names missed. That is a routable absence rather than a fault, because the
    names are DERIVED from the caller's own parameters — NEXRAD's per-frame
    cache names carry its ``--dbz-floor`` / ``--splats`` / ``--grid-m``, so any
    non-default value legitimately asks for members the shipped bundle cannot
    contain, and recomputing is the correct answer.

    It is the bundle-side counterpart of
    :class:`~luxar.utils.data_fetch.DatasetUnavailable` (#1618): both mean "the
    bytes are not obtainable", so a demo may answer either with its own rebuild,
    while every other ``FileNotFoundError`` around a fetch stays a fault that
    must propagate.
    """


def _cache_is_stale(cache_file: Path, source_file: Path) -> bool:
    """True if ``cache_file`` should be refreshed from ``source_file``.

    Stale when the cache is missing, its size differs from the source, or the
    source is newer (``shutil.copy2`` preserves mtime, so a re-migrated /
    re-checked-out source carries a newer one). This makes the demo cache
    self-healing across a packaged-data re-migration (e.g. the gsplats v2.0 ->
    v3.0 cutover) instead of pinning the first-seen copy forever. If the source
    is absent (e.g. an unpulled LFS file) the existing cache is kept.
    """
    if not cache_file.exists():
        return True
    if not source_file.exists():
        return False
    cs, ss = cache_file.stat(), source_file.stat()
    return cs.st_size != ss.st_size or ss.st_mtime > cs.st_mtime + 1e-6


def is_lfs_pointer(path: Path) -> bool:
    """Check whether *path* is an unpulled Git LFS pointer file.

    LFS pointers are small text files (< 1 KB) whose first line is
    ``version https://git-lfs.github.com/spec/v1``.
    """
    if not path.exists():
        return False
    if path.stat().st_size > 1024:
        return False
    try:
        with open(path, "r") as f:
            first_line = f.readline()
        return first_line.startswith("version https://git-lfs.github.com/spec/v1")
    except (UnicodeDecodeError, OSError):
        return False


def _unshippable_reason(demo_name: str) -> Optional[str]:
    """Why *demo_name*'s data is not shipped, or ``None`` if it should be.

    The manifest is the single source of truth for dataset disposition, so this
    asks it rather than keeping a second list that could drift: a
    ``local-compute`` / ``regenerate`` bucket means the data is deliberately
    absent (we may not redistribute it, or it is cheap to rebuild). Any failure
    to answer is reported as "shippable", which preserves the previous
    behaviour — a missing file then raises the ordinary git-lfs error rather
    than being silently excused.
    """
    try:
        from .data_fetch import dataset_spec, load_manifest

        spec = dataset_spec(demo_name, load_manifest())
    except (ImportError, KeyError, OSError, ValueError):
        # DatasetNotFound subclasses KeyError and a malformed manifest raises
        # ValueError, so the handler needs no imported name — which is what
        # lets it cover a failure of the import above too (naming
        # DatasetNotFound here would raise NameError from the handler instead).
        return None
    if spec.get("bucket") not in ("local-compute", "regenerate"):
        return None
    return str(spec.get("reason") or spec.get("strategy") or "not redistributable")


def _validate_lfs_files(paths: list[Path]) -> None:
    """Raise a helpful error if any *paths* are missing or are LFS pointers."""
    missing = [p for p in paths if not p.exists()]
    pointers = [p for p in paths if p.exists() and is_lfs_pointer(p)]

    if missing:
        names = ", ".join(p.name for p in missing)
        raise FileNotFoundError(
            f"Precomputed data files not found: {names}\n"
            "Run 'git lfs pull' to download the data files.\n"
            "Alternatively, use --recompute to fit from scratch (requires GPU)."
        )
    if pointers:
        names = ", ".join(p.name for p in pointers)
        raise FileNotFoundError(
            f"Precomputed data files are Git LFS pointers (not pulled): {names}\n"
            "Run 'git lfs pull' to download the actual data files.\n"
            "Alternatively, use --recompute to fit from scratch (requires GPU)."
        )


def parse_demo_flags() -> dict:
    """Parse common GSplat demo command-line flags from ``sys.argv``.

    Returns a dict with keys: ``recompute``, ``no_serve``, ``serve_only``,
    ``keep_stale``.
    """
    return {
        "recompute": "--recompute" in sys.argv,
        "no_serve": "--no-serve" in sys.argv,
        "serve_only": "--serve-only" in sys.argv,
        "keep_stale": "--keep-stale" in sys.argv,
    }


#: Scene-root attr holding the fingerprint of the builder that wrote the scene.
BUILDER_FINGERPRINT_ATTR: Final[str] = "builder_fingerprint"


def demo_source_fingerprint(module_file: Union[str, Path]) -> str:
    """Short content hash of a demo module's source, for staleness checks.

    Call as ``demo_source_fingerprint(__file__)``. The hash covers that demo
    module's SOURCE TEXT, so an edit to the file produces a different
    fingerprint, while re-running an unchanged demo produces the same one.

    Args:
        module_file: Path to the demo module (normally ``__file__``).

    Returns:
        16 hex characters, or ``""`` if the source cannot be read (in which case
        :func:`scene_is_current` degrades to a plain existence check rather than
        rebuilding a large scene on every run).
    """
    try:
        source = Path(module_file).read_bytes()
    except OSError:
        return ""
    return hashlib.sha256(source).hexdigest()[:16]


def scene_is_current(
    output_path: Path,
    fingerprint: str,
    *,
    recompute: bool = False,
    keep_stale: bool = False,
) -> bool:
    """True if the scene at ``output_path`` can be reused as-is.

    Demos cache their built scene and, historically, reused it whenever the
    path merely EXISTED. That let a scene built by an older version of the demo
    be served forever: #1957 was reported against an ocean-currents scene whose
    missing streamlines had been fixed three weeks earlier, because the fix
    never rebuilt the stale store on disk.

    So a scene is current only when it exists AND was written by this exact
    builder. A scene from before fingerprinting carries no attr and is treated
    as stale — one rebuild, then it stamps itself.

    This gates SCENE ASSEMBLY only. Downloads, gsplat fits and precomputed
    bundles keep their own caches under ``~/.cache/luxar``, so a source edit
    costs a scene rebuild and never a re-download or a re-fit.

    Args:
        output_path: The ``.luxar.zarr`` the demo would write.
        fingerprint: This build's :func:`demo_source_fingerprint`.
        recompute: The demo's ``--recompute`` flag; forces a rebuild.
        keep_stale: The demo's ``--keep-stale`` flag; reuse whatever is on disk
            even when the builder changed. An escape hatch for an expensive
            scene the caller knows is good enough.

    Returns:
        True to reuse the existing scene, False to rebuild.
    """
    if not output_path.exists():
        return False
    if recompute:
        return False
    if keep_stale:
        return True
    if not fingerprint:
        # Unreadable source: no basis to call it stale, and rebuilding a large
        # scene on a bad guess is worse than serving the one on disk.
        return True

    attrs = read_node_attrs(output_path) or {}
    stored = attrs.get(BUILDER_FINGERPRINT_ATTR)
    if stored == fingerprint:
        return True

    aprint(
        f"Demo source changed since this scene was built "
        f"({stored or 'unstamped'} -> {fingerprint}); rebuilding. "
        f"Pass --keep-stale to reuse it instead."
    )
    return False


def _flag_token(name: str) -> str:
    """``--name`` for a flag called ``name``, tolerating pre-written dashes.

    :func:`parse_int_arg` and :func:`parse_path_arg` prepend the ``--``
    themselves, so a caller that passes ``"--points"`` used to make them search
    for ``----points`` — a flag that never matches, silently ignored, every run
    at the default. Normalising here removes that silent no-op for both helpers.
    """
    return f"--{name.lstrip('-')}"


@overload
def parse_int_arg(name: str, default: int, argv: Optional[list[str]] = ...) -> int: ...
@overload
def parse_int_arg(
    name: str, default: None, argv: Optional[list[str]] = ...
) -> Optional[int]: ...
def parse_int_arg(
    name: str, default: Optional[int], argv: Optional[list[str]] = None
) -> Optional[int]:
    """Parse an integer ``--name=VALUE`` or ``--name VALUE`` flag from argv.

    A tiny shared replacement for the ad-hoc ``sys.argv`` scanning every demo
    re-implements (``--points``, ``--sample``, ``--grid``, ``--frames``,
    ``--resolution``, …). Accepts both ``--name=8000`` and ``--name 8000``.
    Returns ``default`` when the flag is absent. The first occurrence is
    decisive, even if malformed: a malformed/unparseable value (e.g.
    ``--points=abc``) warns and returns ``default`` rather than raising — the
    shared helpers never abort a run over a mistyped flag.
    """
    args = list(sys.argv if argv is None else argv)
    flag = _flag_token(name)
    for i, arg in enumerate(args):
        raw: Optional[str] = None
        if arg.startswith(flag + "="):
            raw = arg.split("=", 1)[1]
        elif arg == flag and i + 1 < len(args):
            raw = args[i + 1]
        if raw is not None:
            try:
                return int(raw)
            except ValueError:
                aprint(f"Ignoring malformed {flag}={raw!r}; using {default}")
                return default
    return default


def parse_path_arg(name: str, argv: Optional[list[str]] = None) -> Optional[Path]:
    """Parse a path ``--name=PATH`` or ``--name PATH`` flag from argv.

    Sibling of :func:`parse_int_arg` for path-valued flags (``--cache-dir``,
    ``--data``). Expands a leading ``~``. Returns ``None`` when the flag is
    absent. An empty value (``--data=``) is not treated as a hit — the scan
    skips it and keeps looking, so a lone ``--data=`` reads as absent instead of
    resolving to the current directory (and a later non-empty occurrence wins).

    In the space form the next token must not itself look like an option:
    ``--data --no-tsp`` is a missing value, not a path named ``--no-tsp``, so it
    warns and keeps scanning rather than handing the demo a bogus file to open.
    The ``=`` form stays literal (``--data=--odd`` really does mean that path).
    """
    args = list(sys.argv if argv is None else argv)
    flag = _flag_token(name)
    for i, arg in enumerate(args):
        raw: Optional[str] = None
        if arg.startswith(flag + "="):
            raw = arg.split("=", 1)[1]
        elif arg == flag and i + 1 < len(args):
            if args[i + 1].startswith("--"):
                aprint(f"Ignoring {flag}: followed by {args[i + 1]!r}, not a path")
                continue
            raw = args[i + 1]
        if raw:
            return Path(raw).expanduser()
    return None


# =============================================================================
# Generic cache helpers (downloads + computed results under ~/.cache/luxar)
# =============================================================================
#
# The ``load_precomputed_*`` helpers above cover LFS-shipped gsplat data. These
# two cover the other two demo needs — downloading a remote file once, and
# caching an expensive computed result (a UMAP embedding, a fitted field) — so
# demos stop hand-rolling ``Path.home() / ".cache" / ...`` logic each time. Both
# namespace under ``~/.cache/luxar/<name>/``.


def _safe_cache_key(key: str) -> str:
    """Filesystem-safe slug for a cache key (keeps it readable)."""
    return re.sub(r"[^A-Za-z0-9_.-]+", "_", key).strip("_") or "default"


def cached_download(
    url: str,
    name: str,
    filename: Optional[str] = None,
    *,
    sha256: Optional[str] = None,
    expected_size: Optional[int] = None,
    verbose: bool = True,
) -> Path:
    """Download ``url`` once into ``~/.cache/luxar/<name>/<filename>``.

    Reuses :func:`luxar.utils.download.robust_download` /
    :func:`download_with_checksum` (retry, resume, checksum), but adds the
    skip-if-already-present behaviour a cache needs: a complete cached file is
    returned without touching the network.

    Args:
        url: Source URL.
        name: Cache namespace (the demo name), e.g. ``"earthquakes"``.
        filename: Destination basename; inferred from the URL when omitted.
        sha256: Optional expected SHA-256 (verified on download; a matching
            cached file is trusted without re-download).
        expected_size: Optional expected size in bytes (skip-if-matches).

    Returns:
        Path to the cached file.
    """
    from .download import (
        download_with_checksum,
        quarantine_file,
        robust_download,
        verify_file_checksum,
    )

    cache_dir = _DEFAULT_CACHE_ROOT / name
    cache_dir.mkdir(parents=True, exist_ok=True)
    if not filename:
        filename = url.split("?")[0].rstrip("/").rsplit("/", 1)[-1] or "download.bin"
    dest = cache_dir / filename

    # Skip-if-present. A file that is genuinely WRONG (an LFS pointer stub, or a
    # sha256 mismatch) is quarantined here. But a mere size mismatch is NOT a
    # corruption signal — `expected_size` is only a skip-if-matches hint, and it
    # can be a stale/wrong client-side guess (e.g. an API-reported byte count for
    # a `Content-Encoding: gzip` response whose decoded on-disk size exceeds it).
    # So a file that does not match `expected_size` — whether LONGER or SHORTER —
    # is left in place for robust_download to reconcile against the TRUE remote
    # size. A stale file at the destination is never resumed from (in-progress
    # bytes stage in a sibling `.part` file): robust_download re-fetches into the
    # `.part` and atomically replaces the destination only once the download is
    # complete and size-verified. Quarantining an oversized-but-complete file here
    # would re-download it every launch forever, since the re-fetched bytes are
    # still larger than the stale guess. `expected_size` must never destroy a
    # complete cached file.
    if dest.exists():
        if is_lfs_pointer(dest):
            # A pointer stub is not data — and it is exactly the ~130 bytes that
            # robust_download would otherwise happily resume from.
            quarantine_file(dest, reason="unpulled git-LFS pointer", verbose=verbose)
        elif sha256 is not None:
            if verify_file_checksum(dest, None, sha256, verbose=verbose):
                if verbose:
                    aprint(f"✓ Cached (checksum ok): {dest}")
                return dest
            quarantine_file(dest, reason="sha256 mismatch", verbose=verbose)
        elif expected_size is not None:
            size = dest.stat().st_size
            if size == expected_size:
                if verbose:
                    aprint(f"✓ Cached: {dest}")
                return dest
            # size != expected_size (LONGER or SHORTER): leave it in place and let
            # robust_download reconcile against the TRUE remote size — it
            # re-fetches into a sibling `.part` and atomically replaces the
            # destination on success (a stale destination is never resumed from).
            # Never quarantine here.
        else:
            if verbose:
                aprint(f"✓ Cached: {dest}")
            return dest

    if sha256 is not None:
        return download_with_checksum(
            url, dest, expected_sha256=sha256, expected_size=expected_size
        )
    return robust_download(url, dest, expected_size=expected_size)


def cache_computed(
    name: str,
    key: str,
    compute_fn: Callable[[], Any],
    *,
    version: int = 1,
    recompute: bool = False,
    verbose: bool = True,
    cache_dir: Optional[Path] = None,
) -> Any:
    """Cache the result of ``compute_fn()`` under ``~/.cache/luxar/<name>/``.

    For expensive deterministic results (UMAP embeddings, fitted vector fields).
    The on-disk file is keyed by ``<key>_v<version>`` — bump ``version`` (or fold
    the inputs/params into ``key``) whenever the computation's inputs change, so a
    stale cache is never silently reused. A truncated/corrupt cache file is
    quarantined (``.corrupt``) and recomputed rather than crashing the demo.

    Args:
        name: Cache namespace (the demo name).
        key: Stable identifier for this result (include the sample size / params
            that affect the output, e.g. ``f"umap3d_n{n}_feat{feat_hash}"``).
        compute_fn: Zero-arg callable producing the (picklable) result.
        version: Schema/logic version; bump to invalidate all prior caches.
        recompute: If True, ignore any cached file and recompute.
        cache_dir: Explicit cache directory, used verbatim instead of
            ``~/.cache/luxar/<name>/``; ``name`` is then unused. Without it, the
            location is derived from the cache NAMESPACE, which a demo's
            ``--cache-dir`` flag cannot reach — the same limitation
            :func:`cached_download` has, since it too derives
            ``~/.cache/luxar/<name>/<filename>`` from the namespace.
            ``demo_caida_as_topology`` takes such a flag and must put its derived
            bundles beside the raw downloads they came from, wherever the user
            pointed it.

    Returns:
        The cached or freshly computed result.
    """
    cache_dir = _DEFAULT_CACHE_ROOT / name if cache_dir is None else Path(cache_dir)
    cache_dir.mkdir(parents=True, exist_ok=True)
    cache_file = cache_dir / f"{_safe_cache_key(key)}_v{version}.pkl"

    if not recompute and cache_file.exists():
        try:
            with open(cache_file, "rb") as f:
                result = pickle.load(f)
            if verbose:
                aprint(f"✓ Loaded cached result: {cache_file.name}")
            return result
        except Exception as exc:  # truncated / incompatible pickle
            from .download import quarantine_file

            quarantine_file(
                cache_file, reason=f"unreadable pickle ({exc})", verbose=True
            )

    result = compute_fn()

    # Atomic write so an interrupted run never leaves a truncated cache.
    tmp = cache_file.with_suffix(".pkl.tmp")
    with open(tmp, "wb") as f:
        pickle.dump(result, f, protocol=pickle.HIGHEST_PROTOCOL)
    tmp.replace(cache_file)
    if verbose:
        aprint(f"✓ Cached result: {cache_file.name}")
    return result


def require_local_data(path: Union[str, Path], hint: Optional[str] = None) -> Path:
    """Return ``path`` if it is real local data, else raise a helpful error.

    Guards the local-data demos (LFS-tracked parquet/npz) so an unpulled Git LFS
    pointer raises the clear "run git lfs pull" message instead of a cryptic
    downstream parse error. Wraps :func:`_validate_lfs_files`.
    """
    p = Path(path)
    _validate_lfs_files([p])
    if hint and not p.exists():  # pragma: no cover - _validate_lfs_files raised
        raise FileNotFoundError(hint)
    return p


def hsv_to_rgb(h: np.ndarray, s: Any = 1.0, v: Any = 1.0) -> np.ndarray:
    """Vectorized HSV→RGB for arrays of hues (all in [0, 1]).

    A single shared implementation for the rainbow/hue-ramp colouring several
    demos each re-derived by hand. ``h`` is an array (or scalar); ``s``/``v`` may
    be scalars or broadcastable arrays.

    Returns:
        ``(..., 3)`` float32 RGB in [0, 1] with the same leading shape as ``h``.
    """
    h = np.asarray(h, dtype=np.float32)
    s = np.asarray(s, dtype=np.float32)
    v = np.asarray(v, dtype=np.float32)
    hp = (h % 1.0) * 6.0
    c = v * s
    x = c * (1.0 - np.abs(hp % 2.0 - 1.0))
    m = v - c
    z = np.zeros_like(hp)
    sector = np.floor(hp).astype(int) % 6
    r = np.select(
        [sector == 0, sector == 1, sector == 2, sector == 3, sector == 4, sector == 5],
        [c, x, z, z, x, c],
    )
    g = np.select(
        [sector == 0, sector == 1, sector == 2, sector == 3, sector == 4, sector == 5],
        [x, c, c, x, z, z],
    )
    b = np.select(
        [sector == 0, sector == 1, sector == 2, sector == 3, sector == 4, sector == 5],
        [z, z, x, c, c, x],
    )
    rgb = np.stack([r + m, g + m, b + m], axis=-1)
    return rgb.astype(np.float32)


class StackedColorings(NamedTuple):
    """Result of :func:`stack_colorings` — a point cloud replicated once per
    coloring scheme along a leading categorical axis."""

    positions: np.ndarray  # (N*K, D+1) float32; column 0 is the coloring index
    colors: np.ndarray  # (N*K, 3) float32
    labels: Optional[list[str]]  # (N*K,) hover labels, or None if any view lacks them
    categories: list[str]  # K coloring category names (for the `coloring` Dimension)


def stack_colorings(
    coords: np.ndarray,
    colorings: list[dict],
) -> StackedColorings:
    """Replicate a point cloud once per coloring scheme along a categorical axis.

    This is the render-proven pattern used by the census / peak-UMAP demos to
    give a single embedding several switchable color views: the N points are
    stacked K times (one block per coloring) and a leading ``coloring`` index
    column is prepended so a categorical ``coloring`` dimension can select which
    block (which colour scheme) is shown. Concentrating the stacking here keeps
    the per-point alignment of positions / colours / hover-labels correct and
    tested in one place instead of hand-rolled in each demo.

    Args:
        coords: ``(N, D)`` spatial coordinates (``D`` is usually 3).
        colorings: ordered list of dicts, one per color scheme. Each dict has:

            - ``"label"``: category name shown on the ``coloring`` dimension.
            - ``"colors"``: ``(N, 3)`` float RGB for this scheme.
            - ``"labels"`` (optional): ``(N,)`` per-point hover strings for this
              scheme. If ANY coloring omits labels, the combined ``labels`` is
              ``None`` (hover disabled) rather than misaligned.

    Returns:
        A :class:`StackedColorings`. ``positions`` has shape ``(N*K, D+1)`` with
        column 0 = the coloring index; ``categories`` is the ordered label list
        for building ``Dimension("coloring", categories=...)``.
    """
    coords = np.asarray(coords, dtype=np.float32)
    n = len(coords)
    if not colorings:
        raise ValueError("stack_colorings requires at least one coloring")

    positions_blocks: list[np.ndarray] = []
    color_blocks: list[np.ndarray] = []
    label_blocks: list[list[str]] = []
    categories: list[str] = []
    have_labels = True

    for i, cg in enumerate(colorings):
        cols = np.asarray(cg["colors"], dtype=np.float32)
        if cols.shape != (n, 3):
            raise ValueError(
                f"coloring {i} ({cg.get('label')!r}) colors shape {cols.shape} "
                f"!= expected {(n, 3)}"
            )
        positions_blocks.append(
            np.column_stack([np.full(n, i, dtype=np.float32), coords])
        )
        color_blocks.append(cols)
        categories.append(str(cg["label"]))
        lbls = cg.get("labels")
        if lbls is None:
            have_labels = False
        else:
            if len(lbls) != n:
                raise ValueError(
                    f"coloring {i} ({cg.get('label')!r}) has {len(lbls)} labels "
                    f"!= {n} points"
                )
            label_blocks.append([str(x) for x in lbls])

    positions = np.vstack(positions_blocks).astype(np.float32)
    colors = np.vstack(color_blocks).astype(np.float32)
    labels: Optional[list[str]] = None
    if have_labels:
        labels = [x for block in label_blocks for x in block]
    return StackedColorings(positions, colors, labels, categories)


# |center| at or above this has no int64 voxel index (the cast would overflow),
# so such a row is excluded from the agreement rather than cast.
_VOXEL_KEY_LIMIT = 2.0**62


def voxel_sampled_payload_agreement(
    centers: np.ndarray,
    payload: np.ndarray,
    *,
    min_pairs: int = 1024,
) -> Optional[float]:
    """Fraction of same-voxel splat pairs that carry an identical payload row.

    Several demos ship a per-splat payload (an organ label, a sampled RGB) as a
    SEPARATE sidecar file, indexed positionally against a ``.gsplats.zarr`` fit.
    Nothing in either file records the correspondence, so a sidecar written in a
    different splat order than the store — ``GSplatData.save`` applies a spatial
    ordering, so the stored order is NOT the in-memory one — loads silently and
    renders plausible nonsense. This is the cheap check that catches it.

    The invariant it tests: the payload was produced by NEAREST-VOXEL sampling of
    a volume at the splat centers (``np.round`` the center, index the volume), so
    two splats whose centers round to the SAME voxel necessarily read the SAME
    value. Aligned data satisfies that exactly (agreement 1.0); a permuted
    sidecar pairs each voxel with unrelated rows and scores at the chance level
    of the payload's own value distribution.

    HOW STRONG THE VERDICT IS depends on the payload's own value diversity, not
    on this function: a shuffled sidecar still scores at that payload's chance
    level ``Σ p_v²`` (measured on the two shipped payloads: 0.027 for the CT's
    117 organ labels, 1.4e-05 for the Visible Human's sampled uint8 RGB), so a
    payload that is NEARLY CONSTANT scores near 1.0 however badly it is
    permuted. A caller whose payload has little diversity must not rely on this
    check. NaN is likewise invisible to it: ``NaN == NaN`` is False, so a
    float payload using NaN as "no data" scores ~0 even when perfectly aligned
    (neither Luxar caller can hit that — int labels and uint8-derived colours).

    PRECONDITION — ``centers`` must be in the voxel coordinates the payload was
    sampled in, i.e. straight off ``GSplatData.load``, before any centring,
    scaling or other transform. Recentred centers round to different voxels and
    the collision structure the test relies on is lost.

    Args:
        centers: ``(N, D)`` splat centers. EVERY column takes part in the voxel
            key: a stacked/nD fit puts the spatial dims first and the stacked
            axis LAST, so keying on three columns alone would fold every
            timepoint of a voxel together and reject an aligned sidecar. A row
            that has no integer voxel — a non-finite center, or a magnitude at
            or above ``_VOXEL_KEY_LIMIT`` — is excluded (it cannot be judged).
        payload: ``(N,)`` or ``(N, C)`` per-splat values sampled at those centers.
        min_pairs: Minimum number of same-voxel pairs required to return a
            verdict. Clamped to at least 1: with zero pairs there is nothing to
            divide by, so "no evidence" must stay ``None`` rather than raise
            ``ZeroDivisionError``.

    Returns:
        The agreement fraction in ``[0, 1]``, or ``None`` when fewer than
        ``min_pairs`` same-voxel pairs exist — too little evidence to judge, which
        a caller must treat as "unverifiable", NOT as a failure.

    Raises:
        ValueError: if ``centers`` and ``payload`` have different lengths, or
            ``centers`` is not 2-D, or ``centers`` has no columns.
    """
    centers = np.asarray(centers)
    payload = np.asarray(payload)
    if len(centers) != len(payload):
        raise ValueError(
            f"centers and payload length mismatch: {len(centers)} != {len(payload)}"
        )
    if centers.ndim != 2:
        raise ValueError(f"centers must be 2-D (N, D), got shape {centers.shape}")
    if centers.shape[1] == 0:
        # No columns is no voxel key at all. Rejected explicitly because the
        # lexsort below raises a bare `TypeError: need sequence of keys with
        # len > 0` there, which reads as an internal bug rather than as the
        # caller's malformed input.
        raise ValueError(f"centers must have at least one column, got {centers.shape}")
    # A caller-supplied floor of 0 would let a pair-free input reach the final
    # division; one pair is the least that can be judged.
    min_pairs = max(int(min_pairs), 1)

    # A center that has no int64 voxel — NaN, ±inf, or a magnitude that overflows
    # the cast — must be excluded BEFORE the cast below: `astype(np.int64)` warns
    # bare on such a row ("invalid value encountered in cast", fatal under
    # `-W error`) and collapses every one of them onto ONE sentinel voxel,
    # inventing collisions between splats that share nothing. Dropping them costs
    # nothing: they are unjudgeable, and finite in-range data is unaffected. This
    # test itself is warning-free (`isfinite`/`abs` on a float array are total),
    # so it needs no `errstate` of its own.
    judgeable = np.isfinite(centers) & (np.abs(centers) < _VOXEL_KEY_LIMIT)
    keep = np.flatnonzero(judgeable.all(axis=1))

    # Sort by voxel index so same-voxel splats become adjacent (O(N log N)).
    # `voxels.T[::-1]` makes column 0 the primary lexsort key, for any D.
    voxels = np.rint(centers[keep]).astype(np.int64)
    sort = np.lexsort(voxels.T[::-1])
    voxels = voxels[sort]
    order = keep[sort]
    collides = np.flatnonzero((voxels[1:] == voxels[:-1]).all(axis=1))
    n_pairs = int(collides.size)
    if n_pairs < min_pairs:
        return None

    # Index only the colliding pairs — the payload can be wide and long.
    a = payload[order[collides]]
    b = payload[order[collides + 1]]
    equal = a == b if a.ndim == 1 else (a == b).all(axis=tuple(range(1, a.ndim)))
    return float(np.count_nonzero(equal)) / float(n_pairs)


def load_precomputed_gsplats(
    demo_name: str,
    file_names: list[str],
    *,
    recompute: bool = False,
) -> list | None:
    """Load precomputed GSplat data from Git LFS / local cache.

    Resolution order (per file):
      1. If ``recompute`` is True, return ``None`` immediately.
      2. Check the local cache ``~/.cache/luxar/<demo_name>/``.
      3. Copy from ``demos/data/<demo_name>/`` (LFS) to local cache.
      4. Load from local cache.

    Args:
        demo_name: Subdirectory name under ``demos/data/`` (e.g. ``"tribolium"``).
        file_names: File basenames to load (e.g. ``["tribolium.gsplats.zarr.zip"]``).
        recompute: If True, skip precomputed data entirely and return None.

    Returns:
        List of :class:`GSplatData` in the same order as *file_names*,
        or ``None`` when the caller should recompute.
    """
    from ..gsplats.gsplat_data import GSplatData

    if recompute:
        return None

    cache_dir = _DEFAULT_CACHE_ROOT / demo_name
    lfs_dir = _DEMOS_DATA_DIR / demo_name

    with asection(f"Loading precomputed GSplats ({demo_name})"):
        # Ensure cache dir exists
        cache_dir.mkdir(parents=True, exist_ok=True)

        # Copy any missing OR STALE files from the LFS source to the cache.
        # Staleness matters: when the packaged source is re-migrated (e.g. the
        # v2.0 -> v3.0 format cutover) the cache must refresh, else demos load
        # an outdated cached copy and fail against the v3.0-only reader. We treat
        # the cache as stale when its size differs or the source is newer
        # (``shutil.copy2`` preserves mtime, so a fresh source has a newer one).
        for fname in file_names:
            cache_file = cache_dir / fname
            lfs_file = lfs_dir / fname
            if _cache_is_stale(cache_file, lfs_file):
                # A dataset we are not allowed to redistribute is ABSENT ON
                # PURPOSE, so "run git lfs pull" would send the caller after a
                # file that no longer exists in the repository and never will.
                # Returning None instead routes the demo to its own
                # fetch-the-raw-source-and-rebuild path, which is how these
                # datasets are meant to ship.
                if not lfs_file.exists() and not cache_file.exists():
                    reason = _unshippable_reason(demo_name)
                    if reason is not None:
                        aprint(f"{demo_name} is not redistributable: {reason}")
                        aprint("Rebuilding it locally from the original source…")
                        return None
                _validate_lfs_files([lfs_file])
                aprint(f"Copying {fname} from package data to cache")
                shutil.copy2(lfs_file, cache_file)

        # Load all
        results = []
        for fname in file_names:
            cache_file = cache_dir / fname
            gsplats = GSplatData.load(cache_file, include_stats=False)
            aprint(f"Loaded {fname}: {len(gsplats.amplitudes):,} splats")
            results.append(gsplats)

        return results


def load_precomputed_bundle(
    demo_name: str,
    bundle_name: str,
    file_names: list[str],
    *,
    recompute: bool = False,
) -> list | None:
    """Load precomputed GSplat data from a bundled zip archive in Git LFS.

    For timelapse demos where many per-frame ``.gsplats.zarr.zip`` files are
    bundled into a single outer ``.zip`` stored via Git LFS.

    Args:
        demo_name: Subdirectory name under ``demos/data/`` (e.g. ``"zebrafish"``).
        bundle_name: Filename of the outer bundle zip (e.g. ``"zebrafish.gsplats.zarr.zip"``).
        file_names: Basenames of per-frame files *inside* the bundle to load,
            in the desired order.
        recompute: If True, return None.

    Returns:
        List of :class:`GSplatData`, or ``None`` when the caller should recompute.
    """
    if recompute:
        return None

    cache_dir = _DEFAULT_CACHE_ROOT / demo_name
    bundle_path = _DEMOS_DATA_DIR / demo_name / bundle_name

    with asection(f"Loading precomputed GSplats bundle ({demo_name})"):
        return _extract_bundle_and_load(
            bundle_path, bundle_name, cache_dir, file_names, validate_lfs=True
        )


def load_dataset_bundle(
    name: str,
    bundle_name: str,
    file_names: list[str],
    *,
    recompute: bool = False,
    cache_root: Optional[Path] = None,
    manifest: Optional[dict] = None,
    verbose: bool = True,
) -> list | None:
    """Manifest-driven counterpart of :func:`load_precomputed_bundle`.

    Same contract -- a list of ``GSplatData`` in the requested order, or ``None``
    when the caller must build the data itself -- but the OUTER bundle is resolved
    through :func:`luxar.utils.data_fetch.ensure_dataset`, so it is checksum-
    verified against the manifest (cache -> in-repo git-LFS -> Zenodo) instead of
    copied unverified out of the working tree.

    Bundle members are deliberately NOT pinned individually: the manifest
    addresses the bundle, which is the unit that is downloaded, and verifying it
    covers everything inside. Extraction then reuses the same safe-member and
    staleness logic as the in-repo path, so a re-migrated bundle still refreshes
    its extracted frames rather than pinning the first extraction — and here the
    staleness key is the manifest **digest** rather than the in-repo path's
    ``(size, mtime)`` guess, since a re-upload that happened to preserve both
    would otherwise keep serving the previous extraction.

    Args:
        name: Manifest dataset key (e.g. ``"gsplats_zebrafish"``).
        bundle_name: Basename of the outer bundle zip, a manifest file entry.
        file_names: Basenames of per-frame files *inside* the bundle, in order.
        recompute: Return ``None`` immediately (mirrors ``--recompute``).
        cache_root: Override the cache root (tests).
        manifest: Pre-loaded manifest (tests).
        verbose: Print progress.

    Returns:
        ``list[GSplatData]``, or ``None`` when the caller should build the data.
    """
    if recompute:
        return None

    # Lazy, like `_unshippable_reason`'s: data_fetch reads this module's cache
    # root, so a module-level import here would close the loop.
    from .data_fetch import (
        LocalComputeDataset,
        dataset_spec,
        ensure_dataset,
        resolve_variant,
    )

    ctx = asection(f"Loading GSplats bundle ({name})") if verbose else nullcontext()
    with ctx:
        try:
            paths = ensure_dataset(
                name,
                recompute=False,
                cache_root=cache_root,
                manifest=manifest,
                verbose=verbose,
            )
        except LocalComputeDataset as exc:
            aprint(f"⚠️  {exc}")
            return None

        by_name = {p.name: p for p in paths}
        if bundle_name not in by_name:
            raise FileNotFoundError(
                f"{bundle_name!r} is not a manifest file of dataset {name!r}; "
                f"it lists {sorted(by_name)}"
            )
        bundle_path = by_name[bundle_name]
        # The digest ensure_dataset just verified is the exact staleness key for
        # the extracted frames. None when the manifest entry carries no sha256
        # (a pending-upload row), which falls back to the (size, mtime) stamp.
        files, _ = resolve_variant(name, dataset_spec(name, manifest), None)
        sha = next(
            (e.get("sha256") for e in files if e.get("name") == bundle_name), None
        )
        # ensure_dataset already verified the sha256, so an LFS-pointer check
        # would be checking the wrong thing about an already-trusted file.
        return _extract_bundle_and_load(
            bundle_path,
            bundle_name,
            bundle_path.parent,
            file_names,
            validate_lfs=False,
            stamp=f"sha256:{sha}" if sha else None,
            verbose=verbose,
        )


def _bundle_stamp(bundle_path: Path) -> str:
    """``"<size>:<mtime>"`` for *bundle_path*, or ``""`` when it does not exist.

    Empty is the "cannot vouch for the source" value: callers must then neither
    trust nor write a stamp, so a bundle that appears later still triggers a
    fresh extraction rather than inheriting an earlier run's verdict.
    """
    if not bundle_path.exists():
        return ""
    bs = bundle_path.stat()
    return f"{bs.st_size}:{int(bs.st_mtime)}"


def _frames_needing_extraction(
    bundle_stamp: str,
    stamp_file: Path,
    cache_dir: Path,
    file_names: list[str],
) -> list[str]:
    """Which of *file_names* must be (re-)extracted from the bundle.

    Re-extract when a frame is simply absent OR when the bundle source changed
    (e.g. a v2.0 -> v3.0 re-migration). Keying the stamp on the bundle's identity
    (its manifest digest, or failing that its (size, mtime)) makes the extracted
    cache self-healing instead of pinning the
    first-seen extraction — otherwise demos keep loading stale frames that fail
    against the v3.0-only reader.
    """
    stamp_ok = (
        bundle_stamp != ""
        and stamp_file.exists()
        and stamp_file.read_text() == bundle_stamp
    )
    if bundle_stamp and not stamp_ok:
        # Bundle differs from the cached extraction → re-extract all frames.
        return list(file_names)
    return [f for f in file_names if not (cache_dir / f).exists()]


def _bundle_member_for(
    safe_members: list[str], fname: str, bundle_name: str
) -> tuple[str, Path]:
    """Locate the archive member holding *fname*, as ``(member, requested_path)``.

    Frames may sit at the top level of the bundle or inside a directory, so the
    match is on BASENAME — but only after each candidate survives
    :func:`_validate_zip_member_path`, since an archive is free to name ``../``.
    An unsafe member is skipped rather than rejected outright: it must not be
    able to shadow the legitimate frame sitting further down the list.
    """
    requested_path = _validate_zip_member_path(fname)
    for member in safe_members:
        try:
            member_path = _validate_zip_member_path(member)
        except ValueError:
            continue
        if member_path.name == requested_path.name:
            return member, requested_path
    raise BundleMemberNotFound(
        f"{fname} not found in bundle {bundle_name}. Available: {safe_members[:5]}..."
    )


def _extract_bundle_and_load(
    bundle_path: Path,
    bundle_name: str,
    cache_dir: Path,
    file_names: list[str],
    *,
    validate_lfs: bool,
    stamp: Optional[str] = None,
    verbose: bool = True,
) -> list:
    """Extract the requested members of *bundle_path* into *cache_dir* and load them.

    Shared by the in-repo and manifest-driven bundle loaders: the member matching
    is security-sensitive (an archive may name ``../``) and the staleness stamp is
    what stops a re-migrated bundle serving stale frames, so both paths must use
    the same copy rather than a lookalike.

    *stamp* overrides the staleness key. The manifest-driven caller passes the
    verified sha256, which identifies the bundle exactly; without one the key
    falls back to the bundle's ``(size, mtime)``.
    """
    from ..gsplats.gsplat_data import GSplatData

    cache_dir.mkdir(parents=True, exist_ok=True)

    stamp_file = cache_dir / f".{bundle_name}.stamp"
    bundle_stamp = stamp if stamp is not None else _bundle_stamp(bundle_path)
    missing = _frames_needing_extraction(
        bundle_stamp, stamp_file, cache_dir, file_names
    )

    if missing:
        if validate_lfs:
            _validate_lfs_files([bundle_path])
        if verbose:
            aprint(f"Extracting {len(missing)} files from {bundle_name}")
        with zipfile.ZipFile(bundle_path, "r") as zf:
            safe_members = [m for m in zf.namelist() if not zf.getinfo(m).is_dir()]
            for fname in missing:
                member, requested_path = _bundle_member_for(
                    safe_members, fname, bundle_name
                )
                _safe_extract_zip_member(
                    zf, member, cache_dir, target_name=requested_path.as_posix()
                )
        # Record the bundle stamp so a later run with the SAME bundle skips
        # re-extraction, but a re-migrated bundle (new digest) refreshes.
        if bundle_stamp:
            stamp_file.write_text(bundle_stamp)

    results = []
    for fname in file_names:
        gsplats = GSplatData.load(cache_dir / fname, include_stats=False)
        if verbose:
            aprint(f"Loaded {fname}: {len(gsplats.amplitudes):,} splats")
        results.append(gsplats)

    return results


# Derived demo port ranges. Deliberately DISJOINT from the bare `luxar serve`
# defaults (8000 data / 5173 viewer): a demo must never contend with a manually
# started server, and two different demos must never share a URL — a browser
# tab left over from demo A would otherwise silently front demo B's server
# later (the "I started demo B and got demo A" trap). 499 slots (prime, so
# stems spread well); `pick_port` inside `serve` still resolves the rare
# same-slot collision by shifting up with a warning. Note the guarantee is at
# the PAIR level: two demos may still land on the same DATA port (15 such
# pairs among today's bundled outputs) and merely shift, which is harmless —
# the wrong-scene trap needs BOTH ports to match, since the viewer URL carries
# its own `?src=` data URL.
_DEMO_DATA_PORT_BASE = 8001
_DEMO_VIEWER_PORT_BASE = 5200
_DEMO_PORT_SLOTS = 499


def demo_ports(output_path: Union[str, Path]) -> tuple[int, int]:
    """Stable per-dataset ``(data_port, viewer_port)`` pair for demo serving.

    Derived from the dataset's file name, so re-running the same demo lands on
    the same URL (an old tab for that demo stays valid after a reload) while
    different demos get different URLs. The two slots are drawn from
    INDEPENDENT hash digits: with one shared slot, ~1/6 of the bundled demo
    names collided pairwise (birthday at 499 slots), and an identical
    ``(data, viewer)`` pair reproduces the very same-URL stale-tab trap this
    derivation exists to prevent. Independent slots square the pair space
    (~249k), making a full-pair collision vanishingly rare.

    The dataset NAME is the identity: two datasets that share a file name in
    different directories deliberately share a pair (that is what makes a
    demo's URL stable across output directories), so re-running the same demo
    from two workspaces still hands the second one a shifted port.
    """
    digest = zlib.crc32(Path(output_path).name.encode("utf-8"))
    data_slot = digest % _DEMO_PORT_SLOTS
    viewer_slot = (digest // _DEMO_PORT_SLOTS) % _DEMO_PORT_SLOTS
    return _DEMO_DATA_PORT_BASE + data_slot, _DEMO_VIEWER_PORT_BASE + viewer_slot


def _serve_command(
    output_path: Union[str, Path],
    open_browser: bool,
    serve_args: Optional[list[str]],
) -> list[str]:
    """Build the ``luxar serve`` argv for a demo, with derived default ports.

    The derived pair is appended only when ``serve_args`` doesn't already pin
    the respective option, so a demo passing an explicit ``--port`` /
    ``--viewer-port`` keeps full control.
    """
    args = list(serve_args or [])
    cmd = [sys.executable, "-m", "luxar", "serve", str(output_path), "--viewer"]
    cmd.extend(args)
    data_port, viewer_port = demo_ports(output_path)
    # `serve` also spells the data port `-p` (separate or attached, `-p 9` /
    # `-p9`), and Click silently keeps the LAST occurrence of a repeated
    # option — so missing a pinned spelling here would OVERRIDE the demo's
    # explicit choice with the derived port. No other serve option starts
    # with `-p`.
    data_pinned = any(
        a == "--port"
        or a.startswith("--port=")
        or (a.startswith("-p") and not a.startswith("--"))
        for a in args
    )
    if not data_pinned:
        cmd.extend(["--port", str(data_port)])
    if not any(a == "--viewer-port" or a.startswith("--viewer-port=") for a in args):
        cmd.extend(["--viewer-port", str(viewer_port)])
    if open_browser:
        cmd.append("--open")
    return cmd


def launch_viewer(
    output_path: Union[str, Path],
    open_browser: bool = True,
    serve_args: Optional[list[str]] = None,
) -> None:
    """Launch the Luxar viewer to display a dataset.

    This function uses sys.executable to ensure it works regardless of how
    the demo script was launched (hatch run, hatch shell, conda, etc.).

    Each dataset serves on its own stable derived port pair (see
    :func:`demo_ports`) rather than everyone contending for 8000/5173.

    Args:
        output_path: Path to the .zarr dataset to view
        open_browser: Whether to automatically open a browser window
        serve_args: Extra arguments forwarded verbatim to ``luxar serve`` — e.g.
            ``["--profile", "3g"]`` for the network-simulation demo. These let a
            demo drive server-side behaviour (bandwidth throttling, latency,
            packet loss) that ``serve`` already supports. An explicit
            ``--port`` / ``--viewer-port`` here overrides the derived pair.

    Raises:
        SystemExit: If the viewer fails to launch
    """
    cmd = _serve_command(output_path, open_browser, serve_args)

    # isolate_group=False: keep the `luxar serve` child in this process's group
    # so an ancestor's group-kill (from `luxar demo run`) still cascades to it.
    # When the demo is run directly (`python -m luxar.demos.demo_X`), the
    # helper's per-PID teardown still cleanly kills a hung server on Ctrl-C.
    code = run_child_process(cmd, label="Luxar viewer", isolate_group=False)
    if code in (0, 130):  # clean exit or user Ctrl-C
        return
    # Genuine failure. returncode 1 usually means luxar/module import failure;
    # anything else typically means the viewer isn't built.
    if code == 1:
        aprint("\n❌ Error running luxar (exit 1).")
        aprint("Make sure luxar is installed in your Python environment.")
        aprint("If using hatch: run this demo with 'hatch run python <demo.py>'")
    else:
        aprint(f"\n❌ Error: luxar serve exited with code {code}.")
        aprint("Make sure the viewer is built: cd packages/luxar-viewer && pnpm build")
    sys.exit(1)


def create_lorenz_attractor(
    store_path: PathLike,
    n_points: int = 10_000,
    seed: Optional[int] = None,
) -> None:
    """Create a demo scene with a Lorenz attractor visualization.

    This creates a beautiful butterfly-shaped 3D structure with colors
    that transition smoothly over time, demonstrating the Luxar scene format
    with an aesthetically pleasing mathematical visualization.

    Args:
        store_path: Path to the Zarr store to create
        n_points: Number of points to generate along the attractor
        seed: Random seed for reproducible results

    Example:
        >>> from luxar.utils import create_lorenz_attractor
        >>> create_lorenz_attractor('demo.zarr', n_points=50000)
    """
    aprint(f"Creating Lorenz attractor demo scene with {n_points:,} points.")

    # Check for performance warnings
    warning = check_dataset_size_warning(n_points)
    if warning:
        aprint(f"Performance warning: {warning}")

    # Reuse the demo's own integrator so this fixture and `luxar demo run
    # lorenz` trace the very same trajectory (the two scenes still differ in
    # radii, brightness and overlays).
    # Imported here (not at module top) to break the import cycle
    # demos.demo_lorenz -> luxar.demos -> utils.demos.
    from ..demos.demo_lorenz import lorenz_trajectory

    positions = lorenz_trajectory(n_points, seed=seed)
    # Cycle through hues twice for smooth time-based colour transitions.
    colors = hsv_to_rgb((np.linspace(0, 1, n_points) * 2.0) % 1.0)

    # Create scene.
    # Imported here (not at module top) to break the import cycle
    # io.compiler -> luxar.utils -> utils.demos -> io.compiler.
    from ..io.compiler import LuxarZarrCompiler

    with LuxarZarrCompiler(store_path) as compiler:
        # Define 3D dimensions
        dims = Dimensions(
            [
                Dimension("x", unit="units", display=True),
                Dimension("y", unit="units", display=True),
                Dimension("z", unit="units", display=True),
            ]
        )

        compiler.create_scene(dimensions=dims)

        # Write the attractor data
        # NOTE: radii must be scaled by same factor as positions!
        compiler.write_points(
            "LorenzAttractor",
            positions * 100.0 - 50.0,  # Scale up for better visibility
            colors=colors,
            radii=2.0,  # Uniform radius for visibility
            opacity=0.9,
            blending_mode="additive",
        )

    aprint(f"✓ Lorenz attractor demo scene created at {store_path}")


def create_random_spheres(
    store_path: PathLike,
    n_spheres: int = 100,
    points_per_sphere: int = 1000,
    seed: Optional[int] = None,
) -> None:
    """Create a demo scene with random colored spheres.

    Args:
        store_path: Path to the Zarr store to create
        n_spheres: Number of spheres to generate
        points_per_sphere: Points per sphere
        seed: Random seed for reproducible results
    """
    rng = np.random.default_rng(seed)

    aprint(f"Creating random spheres demo with {n_spheres} spheres")

    # Imported here (not at module top) to break the import cycle
    # io.compiler -> luxar.utils -> utils.demos -> io.compiler.
    from ..io.compiler import LuxarZarrCompiler

    with LuxarZarrCompiler(store_path) as compiler:
        from luxar import Dimensions

        compiler.create_scene(dimensions=Dimensions.default_3d())

        for i in range(n_spheres):
            # Random center position
            center = rng.uniform(-10, 10, 3)

            # Generate points on sphere surface
            phi = rng.uniform(0, 2 * np.pi, points_per_sphere)
            costheta = rng.uniform(-1, 1, points_per_sphere)
            u = rng.uniform(0, 1, points_per_sphere)

            theta = np.arccos(costheta)

            radius = rng.uniform(0.5, 2.0)
            r = radius * u ** (1 / 3)

            x = r * np.sin(theta) * np.cos(phi) + center[0]
            y = r * np.sin(theta) * np.sin(phi) + center[1]
            z = r * np.cos(theta) + center[2]

            positions = np.column_stack([x, y, z]).astype(np.float32)

            # Random HDR color for each sphere - use tuple for scalar passthrough
            color_arr = rng.uniform(0.5, 2.0, 3)
            color_tuple = tuple(float(c) for c in color_arr)  # Convert to tuple

            compiler.write_points(
                f"sphere_{i:03d}",
                positions,
                colors=color_tuple,  # Single color for whole sphere (tuple for broadcasting)
                radii=0.05,  # Scalar passthrough
                opacity=0.8,
            )

    aprint(f"✓ Random spheres demo created at {store_path}")


def create_time_series_demo(
    store_path: PathLike,
    n_timepoints: int = 10,
    n_points_per_time: int = 1000,
    seed: Optional[int] = None,
) -> None:
    """Create a 4D time series demo scene.

    Args:
        store_path: Path to the Zarr store to create
        n_timepoints: Number of time points
        n_points_per_time: Points per time step
        seed: Random seed
    """
    rng = np.random.default_rng(seed)

    aprint(f"Creating 4D time series demo with {n_timepoints} time points")

    # Ensure at least 2 timepoints for valid dimension range
    if n_timepoints < 2:
        n_timepoints = 2
        aprint("  Note: Minimum 2 time points required, using 2")

    # Create 4D dimensions
    dims = Dimensions(
        [
            Dimension("x", unit="um", display=True),
            Dimension("y", unit="um", display=True),
            Dimension("z", unit="um", display=True),
            Dimension(
                "time",
                unit="s",
                display=False,
                discrete=True,
                range=(0, n_timepoints - 1),
            ),
        ]
    )

    # Imported here (not at module top) to break the import cycle
    # io.compiler -> luxar.utils -> utils.demos -> io.compiler.
    from ..io.compiler import LuxarZarrCompiler

    with LuxarZarrCompiler(store_path) as compiler:
        compiler.create_scene(dimensions=dims)

        # Generate all time points
        all_positions = []
        all_colors = []

        for t in range(n_timepoints):
            # Generate expanding sphere over time
            radius = 1.0 + t * 0.5

            # Random points in sphere
            phi = rng.uniform(0, 2 * np.pi, n_points_per_time)
            costheta = rng.uniform(-1, 1, n_points_per_time)
            u = rng.uniform(0, 1, n_points_per_time)

            theta = np.arccos(costheta)
            r = radius * u ** (1 / 3)

            x = r * np.sin(theta) * np.cos(phi)
            y = r * np.sin(theta) * np.sin(phi)
            z = r * np.cos(theta)

            # Add time dimension
            time_coord = np.full(n_points_per_time, t, dtype=np.float32)

            positions_4d = np.column_stack([x, y, z, time_coord]).astype(np.float32)
            all_positions.append(positions_4d)

            # Color changes over time
            color = np.array(
                [1.0 - t / n_timepoints, 0.5, t / n_timepoints], dtype=np.float32
            )
            colors = np.tile(color, (n_points_per_time, 1))
            all_colors.append(colors)

        # Concatenate all time points
        positions_array = np.vstack(all_positions)
        colors_array = np.vstack(all_colors)

        # Write as single 4D dataset
        compiler.write_points(
            "time_series",
            positions_array,
            colors=colors_array,
            radii=0.1,  # Scalar passthrough
        )

    aprint(f"✓ Time series demo created at {store_path}")
