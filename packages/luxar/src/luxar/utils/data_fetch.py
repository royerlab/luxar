"""Manifest-driven demo-dataset fetch (R17: retire git-LFS heavy data → Zenodo).

``demos/data_manifest.json`` is the single source of truth for how every demo
dataset is obtained (see :mod:`scripts.gen_data_manifest`). This module reads it
and resolves a dataset's files into the local cache, so demos can migrate off
in-repo git-LFS to fetch-on-demand from Zenodo without changing their own logic.

The manifest is a *packaged resource* — it ships inside the wheel and is always
present. It deliberately does NOT live under ``demos/data/``: that whole tree is
excluded from the wheel and sdist (~450 MB of git-LFS payload), so a manifest
kept there would be missing for exactly the installed users this module serves.

Resolution order for a ``zenodo`` dataset (per file). The manifest sha256 is the
authority at every step — a copy that fails it is quarantined, never returned:
    1. Local cache ``~/.cache/luxar/<dataset>/<file>``, if it verifies.
    2. In-repo git-LFS copy ``demos/data/<dir>/<file>``, where ``<dir>`` is the
       manifest ``dir`` field (EMPTY for top-level datasets — the file then lives
       directly under ``demos/data/``). Copied atomically into the cache and then
       verified. This is the fallback that keeps demos working *during* the
       migration, until a dataset's Zenodo URL is populated.
    3. Download from the dataset's Zenodo record (checksum-verified), if the
       record has a resolvable URL.
    4. Otherwise a clear error (data neither cached, in-repo, nor hosted yet).

``local-compute`` and ``regenerate`` datasets are NOT fetched here — they raise
:class:`LocalComputeDataset`, signalling the caller to run its own
fetch-raw-and-build path (these are the datasets we cannot redistribute, plus the
cheap CPU-rebuild ones).
"""

from __future__ import annotations

import json
from functools import lru_cache
from pathlib import Path
from types import TracebackType
from typing import Any, Optional

from arbol import aprint, asection

# Reuse the cache root, packaged-data dir and LFS-pointer probe so this module
# and load_precomputed_gsplats share one notion of "the cache".
# NOTE: _cache_is_stale is deliberately NOT imported. Its (size, mtime) test is
# blind to an in-place corruption of unchanged length — precisely how a
# checksum-failing cache entry used to survive step 1 and be handed back.
from .demos import _DEFAULT_CACHE_ROOT, _DEMOS_DATA_DIR, is_lfs_pointer

#: Packaged manifest, resolved the same way ``demos._DEMOS_DATA_DIR`` is: this
#: module lives in ``luxar/utils/`` and the manifest ships in ``luxar/demos/``.
#: Anchored to ``__file__`` rather than derived from ``_DEMOS_DATA_DIR`` so it
#: stays reachable once R17 step 4 removes the data tree.
MANIFEST_PATH = Path(__file__).resolve().parent.parent / "demos" / "data_manifest.json"

#: A decoded JSON object out of the manifest — the manifest itself, a dataset
#: spec, a Zenodo record, or a single file entry. The values are heterogeneous
#: JSON (str / int / list / nested object), so ``Any`` is the honest element type.
Manifest = dict[str, Any]

# Buckets that this module does NOT fetch (caller builds them locally).
_LOCAL_BUCKETS = {"local-compute", "regenerate"}


class DatasetNotFound(KeyError):
    """The dataset name is not present in the manifest."""


class LocalComputeDataset(RuntimeError):
    """Dataset is not hosted — the demo must fetch its raw source and build it.

    Raised for ``local-compute`` (not redistributable) and ``regenerate`` (cheap
    CPU rebuild) datasets. Carries the manifest ``strategy``/``reason`` text.
    """

    def __init__(self, name: str, spec: Manifest) -> None:
        self.name = name
        self.spec = spec
        reason = spec.get("reason", "")
        strategy = spec.get("strategy", "")
        msg = f"Dataset {name!r} (bucket={spec.get('bucket')!r}) is not hosted."
        if reason:
            msg += f" Reason: {reason}"
        if strategy:
            msg += f" Build it locally: {strategy}"
        super().__init__(msg)


@lru_cache(maxsize=1)
def load_manifest(path: Optional[str] = None) -> Manifest:
    """Load and cache the demo-data manifest (JSON)."""
    p = Path(path) if path else MANIFEST_PATH
    try:
        with open(p, "r") as f:
            manifest: Manifest = json.load(f)
            return manifest
    except FileNotFoundError:
        raise FileNotFoundError(
            f"Demo-data manifest not found at {p}. It is a packaged resource that "
            "ships inside luxar/demos/, so a missing file means a broken install "
            "or a packaging exclude that swallowed it — see "
            "test_manifest_is_shippable_in_the_wheel_and_sdist."
        ) from None


def dataset_spec(name: str, manifest: Optional[Manifest] = None) -> Manifest:
    """Return the manifest entry for *name* (raises :class:`DatasetNotFound`)."""
    m = manifest or load_manifest()
    try:
        spec: Manifest = m["datasets"][name]
        return spec
    except KeyError:
        raise DatasetNotFound(
            f"{name!r} not in manifest ({sorted(m['datasets'])[:6]}…)"
        ) from None


def zenodo_file_url(record: Manifest, filename: str) -> Optional[str]:
    """Build a Zenodo file-download URL for *filename* in *record*, or None.

    Prefers an explicit ``base_url``; otherwise derives the standard
    ``.../records/<id>/files/<name>?download=1`` form from ``zenodo_record``.
    Returns None when the record has neither (i.e. not uploaded yet), which keeps
    the fetch path dormant and demos on the in-repo fallback.
    """
    base = record.get("base_url")
    if base:
        return f"{base.rstrip('/')}/{filename}?download=1"
    rec = record.get("zenodo_record")
    if rec:
        return f"https://zenodo.org/records/{rec}/files/{filename}?download=1"
    return None


def resolve_variant(
    name: str, spec: Manifest, variant: Optional[str]
) -> tuple[list[Manifest], Optional[str]]:
    """Return ``(files, variant_name)`` for a dataset, honouring size variants.

    A dataset with a ``variants`` map (e.g. h2afva's light ``51tp`` default vs
    the full ``253tp``) selects the requested variant, or the one flagged
    ``default`` when *variant* is None. A dataset without variants uses its flat
    ``files`` list and rejects a variant request.
    """
    variants = spec.get("variants")
    if not variants:
        if variant is not None:
            raise ValueError(
                f"Dataset {name!r} has no variants (requested {variant!r})."
            )
        return spec.get("files") or [], None
    if variant is None:
        defaults = [v for v, meta in variants.items() if meta.get("default")]
        variant = defaults[0] if defaults else next(iter(variants))
    if variant not in variants:
        raise ValueError(
            f"Unknown variant {variant!r} for {name!r}; available: {sorted(variants)}"
        )
    return variants[variant].get("files") or [], variant


def ensure_dataset(
    name: str,
    *,
    variant: Optional[str] = None,
    recompute: bool = False,
    cache_root: Optional[Path] = None,
    manifest: Optional[Manifest] = None,
    verbose: bool = True,
) -> list[Path]:
    """Ensure a ``zenodo`` dataset's files are present locally; return their paths.

    Args:
        name: Dataset key in the manifest (e.g. ``"gsplats_kidney"``).
        variant: For datasets with size variants (e.g. h2afva), which to fetch;
            defaults to the variant flagged ``default`` (the lighter one). An
            error for a dataset without variants, or an unknown variant name.
        recompute: If True, raise :class:`LocalComputeDataset` for *any* dataset
            so the caller takes its own build path (mirrors the demos' ``--recompute``).
        cache_root: Override the cache root (tests). Defaults to ``~/.cache/luxar``.
        manifest: Pre-loaded manifest (tests); defaults to the packaged one.

    Returns:
        Cache paths of the dataset files, in manifest order.

    Raises:
        DatasetNotFound: unknown dataset.
        LocalComputeDataset: dataset is local-compute/regenerate (or recompute=True).
        FileNotFoundError: data is neither cached, in-repo, nor hosted yet.
    """
    m = manifest or load_manifest()
    spec = dataset_spec(name, m)
    bucket = spec.get("bucket")

    if recompute or bucket in _LOCAL_BUCKETS:
        raise LocalComputeDataset(name, spec)
    if bucket != "zenodo":
        raise ValueError(f"Dataset {name!r} has unsupported bucket {bucket!r}")

    files, variant_name = resolve_variant(name, spec, variant)
    if not files:
        detail = f" variant {variant_name!r}" if variant_name else ""
        raise FileNotFoundError(
            f"Dataset {name!r}{detail} has no files listed in the manifest yet "
            "(pending upload). Nothing to fetch."
        )

    root = Path(cache_root) if cache_root else _DEFAULT_CACHE_ROOT
    # Variant files live in their own cache subdir so a light and full variant
    # of the same dataset never collide.
    cache_dir = (root / name / variant_name) if variant_name else (root / name)
    cache_dir.mkdir(parents=True, exist_ok=True)
    # The in-repo LFS layout comes from the manifest ``dir`` (the effective
    # subdir under demos/data): "" means the file lives at the top level of
    # demos/data, otherwise it is a ``<dir>/`` subdir. Older/synthetic
    # manifests that omit ``dir`` fall back to the dataset name (backward
    # compatible). The cache still namespaces by name/variant above regardless.
    subdir = spec.get("dir", name)
    parts = [p for p in (subdir, variant_name) if p]
    lfs_dir = _DEMOS_DATA_DIR.joinpath(*parts)
    record = m["records"].get(spec.get("record", ""), {})

    resolved: list[Path] = []
    label = f"{name}:{variant_name}" if variant_name else name
    with asection(f"Ensuring dataset ({label})") if verbose else _null_ctx():
        for entry in files:
            fname = entry["name"]
            sha = entry.get("sha256")
            dest = cache_dir / fname
            resolved.append(_ensure_one(dest, fname, sha, lfs_dir, record, verbose))
    return resolved


def _ensure_one(
    dest: Path,
    fname: str,
    sha: Optional[str],
    lfs_dir: Path,
    record: Manifest,
    verbose: bool,
) -> Path:
    """Resolve one file: cache → in-repo LFS → Zenodo, checksum-authoritative.

    The manifest sha256 is the only authority at every step:

    * a cached file that fails it is QUARANTINED and never reused. The previous
      version fell through to step 2 instead, where a (size, mtime) staleness
      test could not see an in-place corruption and returned the bad file;
    * a copy taken from the in-repo git-LFS tree is verified AFTER copying, so
      the checksum we already hold is actually used rather than carried around;
    * the Zenodo leg is never handed a pre-existing file, because
      ``robust_download`` RESUMES onto whatever bytes sit at the destination
      (issue #731) — appending a fresh download to stale garbage.

    Quarantining in step 1 is what establishes that last invariant: below it,
    ``dest`` does not exist.
    """
    from .atomic_copy import atomic_copy_file
    from .download import download_with_checksum, quarantine_file, verify_file_checksum

    def _verified(path: Path) -> bool:
        """True only on a POSITIVE checksum match.

        ``verify_file_checksum`` returns True vacuously when given no expected
        hash, so ``sha is None`` is excluded here rather than left to a
        short-circuit that a later edit could drop.
        """
        return (
            sha is not None and path.is_file() and verify_file_checksum(path, None, sha)
        )

    # ── 1. Local cache ──────────────────────────────────────────────────────
    if dest.exists():
        if not dest.is_file():
            raise IsADirectoryError(
                f"Cache entry {dest} exists but is not a regular file; remove it "
                "(or run 'luxar demo clear') and retry."
            )
        if sha is None and not is_lfs_pointer(dest):
            # Unverifiable: a pending-upload entry, or a manifest predating the
            # checksum. Reuse it — but say so. Silence is how corruption lives.
            if verbose:
                aprint(f"✓ Cached (UNVERIFIED — no sha256 in manifest): {fname}")
            return dest
        if _verified(dest):
            if verbose:
                aprint(f"✓ Cached (sha256 verified): {fname}")
            return dest
        quarantine_file(
            dest,
            reason=(
                "unpulled git-LFS pointer"
                if is_lfs_pointer(dest)
                else "does not match the manifest sha256 (corrupt, or superseded "
                "by a data update)"
            ),
            verbose=verbose,
        )

    # INVARIANT from here on: `dest` does not exist.

    # ── 2. In-repo git-LFS copy (the migration fallback) ────────────────────
    lfs_file = lfs_dir / fname
    inrepo_is_bad = False
    if lfs_file.is_file() and not is_lfs_pointer(lfs_file):
        if verbose:
            aprint(f"Copying {fname} from packaged data to cache")
        # Atomic: a Ctrl-C mid-copy must not leave a truncated file under the
        # canonical name for the next run to quarantine and re-fetch.
        atomic_copy_file(lfs_file, dest)
        if sha is None or _verified(dest):
            return dest
        # The copy is wrong. Hash the SOURCE to apportion blame — this second
        # pass only ever runs on this failure path.
        source_ok = _verified(lfs_file)
        quarantine_file(dest, reason="sha256 mismatch after copy", verbose=verbose)
        if source_ok:
            raise RuntimeError(
                f"{fname} was copied from {lfs_file} but the copy failed its "
                "sha256 while the source passed — the cache write is damaged "
                "(disk full? failing media?). The bad copy was quarantined; "
                "re-downloading would not help."
            )
        # A bad packaged copy is a repo/manifest problem. Never rename anything
        # under demos/data: it is git-tracked, so a .corrupt file there would
        # dirty the tree and break the manifest/disk consistency test.
        inrepo_is_bad = True
        aprint(
            f"⚠️  In-repo copy of {fname} does not match the manifest sha256 "
            f"({lfs_file}). Re-pull it ('git lfs pull'), or regenerate the "
            "manifest (make gen-data-manifest) if the data changed."
        )

    # ── 3. Zenodo (only once the record URL is populated) ───────────────────
    url = zenodo_file_url(record, fname)
    if url:
        if verbose:
            aprint(f"↓ Fetching {fname} from Zenodo")
        # `dest` is absent by the invariant above, so robust_download starts at
        # byte 0 instead of resuming onto (appending to) a stale file.
        return download_with_checksum(url, dest, expected_sha256=sha)

    if inrepo_is_bad:
        raise FileNotFoundError(
            f"{fname} is present in-repo but fails its manifest sha256, and the "
            "dataset has no Zenodo URL yet — there is no good copy to fall back "
            "to. Re-pull the LFS object, or regenerate the manifest if the data "
            "was intentionally updated."
        )
    raise FileNotFoundError(
        f"{fname} is not cached (any cached copy failed its checksum and was "
        "quarantined), not present in-repo (git lfs pull), and its Zenodo record "
        "URL is not set yet in the demo-data manifest."
    )


class _null_ctx:
    """No-op context manager for the non-verbose path."""

    def __enter__(self) -> "_null_ctx":
        return self

    def __exit__(
        self,
        exc_type: Optional[type[BaseException]],
        exc: Optional[BaseException],
        tb: Optional[TracebackType],
    ) -> None:
        # Returning None (not False) so mypy knows exceptions are never swallowed.
        return None


#: Suffixes the gsplat loader understands. A dataset may legitimately carry
#: sidecars (gsplats_ct_totalsegmentator ships ct_atlas_labels.npz next to its
#: fit), so the default selection filters on these rather than loading every
#: file the manifest lists.
_GSPLAT_SUFFIXES = (".gsplats.zarr.zip", ".gsplats.zarr")


def load_dataset_gsplats(
    name: str,
    file_names: Optional[list[str]] = None,
    *,
    variant: Optional[str] = None,
    recompute: bool = False,
    cache_root: Optional[Path] = None,
    manifest: Optional[Manifest] = None,
    verbose: bool = True,
) -> Optional[list[Any]]:
    """Manifest-driven stand-in for :func:`luxar.utils.demos.load_precomputed_gsplats`.

    Same contract as the helper it is meant to replace — a list of ``GSplatData``
    in the requested order, or ``None`` when the caller must build the data
    itself — but the files are resolved through :func:`ensure_dataset`
    (checksum-verified cache → in-repo git-LFS → Zenodo) instead of an unverified
    ``shutil.copy2``. Migrating a demo is then a one-line swap.

    ``None`` is returned when:

    * ``recompute`` is True (mirrors the demos' ``--recompute`` flag), or
    * the dataset is not hosted (``local-compute`` / ``regenerate``): the
      manifest's reason/strategy is printed and the caller's existing
      "fit from scratch" branch takes over.

    Everything else raises — an unknown dataset, an unknown variant, a file that
    is neither cached nor in-repo nor hosted, a checksum that will not verify.
    Those are faults a demo must not silently route around.

    Args:
        name: Manifest dataset key (e.g. ``"gsplats_kidney"``).
        file_names: Explicit basenames, in load order. Must all be manifest
            entries of this dataset. Defaults to every ``*.gsplats.zarr[.zip]``
            file, in manifest order.
        variant: Size variant; see :func:`ensure_dataset`.
        recompute: Return ``None`` immediately.
        cache_root: Override the cache root (tests).
        manifest: Pre-loaded manifest (tests).
        verbose: Print progress.

    Returns:
        ``list[GSplatData]``, or ``None`` when the caller should build the data.

    Deliberately NOT supported:
        * **Bundle datasets.** ``gsplats_zebrafish`` / ``gsplats_celegans`` ship
          one outer zip holding many per-frame files; the manifest addresses the
          bundle, not its members. Those demos stay on
          :func:`~luxar.utils.demos.load_precomputed_bundle`.
        * **Runtime-computed file lists** that are not manifest entries (e.g.
          zebrafish's subsampled frame names). A *subset* of the manifest's own
          files is fine; anything else raises rather than fetching the wrong data.
        * **Non-gsplat payloads.** Use :func:`ensure_dataset`, which returns paths
          and assumes nothing about the format.
        * **Datasets whose bucket is not ``zenodo``.** ``gsplats_tribolium``,
          ``gsplats_acto3d_heart``, ``gsplats_tng_cosmic_web`` and
          ``milky_way_gaia_3m`` still ship files in-repo but are marked
          ``local-compute``, so this returns ``None`` for them and the demo would
          fit from scratch on a GPU instead of loading the file that is sitting
          right there. Migrate a demo only once its dataset is ``zenodo``.
    """
    if recompute:
        return None

    from ..gsplats.gsplat_data import GSplatData

    try:
        paths = ensure_dataset(
            name,
            variant=variant,
            recompute=False,
            cache_root=cache_root,
            manifest=manifest,
            verbose=verbose,
        )
    except LocalComputeDataset as exc:
        # Not hosted → the caller's own build path. Say why, loudly: this is the
        # difference between an instant demo and twenty minutes of GPU time.
        aprint(f"⚠️  {exc}")
        return None

    if file_names is not None:
        by_name = {p.name: p for p in paths}
        missing = [f for f in file_names if f not in by_name]
        if missing:
            raise FileNotFoundError(
                f"Dataset {name!r} does not list {missing} in the manifest "
                f"(available: {sorted(by_name)}). A runtime-computed file list is "
                "not supported — see load_dataset_gsplats's docstring."
            )
        selected = [by_name[f] for f in file_names]
    else:
        selected = [p for p in paths if p.name.endswith(_GSPLAT_SUFFIXES)]
        if not selected:
            raise FileNotFoundError(
                f"Dataset {name!r} lists no *.gsplats.zarr[.zip] file "
                f"({[p.name for p in paths]}). Use ensure_dataset() for "
                "non-gsplat payloads."
            )

    results: list[Any] = []
    with asection(f"Loading gsplats ({name})") if verbose else _null_ctx():
        for path in selected:
            gsplats = GSplatData.load(path, include_stats=False)
            if verbose:
                aprint(f"Loaded {path.name}: {len(gsplats.amplitudes):,} splats")
            results.append(gsplats)
    return results
