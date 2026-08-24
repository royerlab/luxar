"""Download and computed-value cache helpers for Luxar demos."""

from __future__ import annotations

import pickle
import re
from pathlib import Path
from typing import Any, Callable, Optional

from arbol import aprint

from .lfs import is_lfs_pointer

# Default user-level cache
_DEFAULT_CACHE_ROOT = Path.home() / ".cache" / "luxar"


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
