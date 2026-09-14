"""Precomputed GSplat and bundle loading helpers."""

from __future__ import annotations

import shutil
import zipfile
from contextlib import nullcontext
from pathlib import Path, PurePosixPath
from typing import Optional

from arbol import aprint, asection

from ..downloads.zip_safety import _safe_extract_zip_member, _validate_zip_member_path
from .cache import _DEFAULT_CACHE_ROOT, _cache_is_stale
from .lfs import _DEMOS_DATA_DIR, _unshippable_reason, _validate_lfs_files


class BundleMemberNotFound(FileNotFoundError):
    """A requested frame is not inside an otherwise perfectly good bundle.

    The bundle itself resolved, verified and opened; only the per-frame member
    names missed. That is a routable absence rather than a fault, because the
    names are DERIVED from the caller's own parameters — NEXRAD's per-frame
    cache names carry its ``--dbz-floor`` / ``--splats`` / ``--grid-m``, so any
    non-default value legitimately asks for members the shipped bundle cannot
    contain, and recomputing is the correct answer.

    It is the bundle-side counterpart of
    :class:`~luxar.demos._support.datasets.data_fetch.DatasetUnavailable` (#1618): both mean "the
    bytes are not obtainable", so a demo may answer either with its own rebuild,
    while every other ``FileNotFoundError`` around a fetch stays a fault that
    must propagate.
    """


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
    from ....gsplats.gsplat_data import GSplatData

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
    through :func:`luxar.demos.ensure_dataset`, so it is checksum-
    verified against the manifest (cache -> in-repo git-LFS -> Zenodo) instead of
    copied unverified out of the working tree.

    Bundle members are deliberately NOT pinned individually: the manifest
    addresses the bundle, which is the unit that is downloaded, and verifying it
    covers everything inside. Extraction then reuses the same safe-member and
    staleness logic as the in-repo path, so a re-migrated bundle still refreshes
    its extracted frames rather than pinning the first extraction — and here the
    staleness key is the exact **digest** :func:`ensure_dataset` verified rather
    than the in-repo path's ``(size, mtime)`` guess.

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
    from .data_fetch import LocalComputeDataset, ensure_dataset

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
        stamp = f"sha256:{paths.input_digests[bundle_name]}"
        # ensure_dataset already verified the sha256, so an LFS-pointer check
        # would be checking the wrong thing about an already-trusted file.
        return _extract_bundle_and_load(
            bundle_path,
            bundle_name,
            bundle_path.parent,
            file_names,
            validate_lfs=False,
            stamp=stamp,
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
) -> tuple[str, PurePosixPath]:
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
    exact digest verified by the resolver.
    """
    from ....gsplats.gsplat_data import GSplatData

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
