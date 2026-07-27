"""Manifest-driven demo-dataset fetch (R17: retire git-LFS heavy data → Zenodo).

``demos/data/manifest.json`` is the single source of truth for how every demo
dataset is obtained (see :mod:`scripts.gen_data_manifest`). This module reads it
and resolves a dataset's files into the local cache, so demos can migrate off
in-repo git-LFS to fetch-on-demand from Zenodo without changing their own logic.

Resolution order for a ``zenodo`` dataset (per file):
    1. Local cache ``~/.cache/luxar/<dataset>/<file>`` (checksum-verified if known).
    2. In-repo git-LFS copy ``demos/data/<dataset>/<file>`` (copied into cache).
       This is the fallback that keeps demos working *during* the migration and
       until a dataset's Zenodo URL is populated.
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
import shutil
from functools import lru_cache
from pathlib import Path
from typing import Optional

from arbol import aprint, asection

# Reuse the cache root, packaged-data dir, and LFS/staleness helpers so this
# module and load_precomputed_gsplats share one notion of "the cache".
from ..utils.demos import (
    _DEFAULT_CACHE_ROOT,
    _DEMOS_DATA_DIR,
    _cache_is_stale,
    is_lfs_pointer,
)

MANIFEST_PATH = _DEMOS_DATA_DIR / "manifest.json"

# Buckets that this module does NOT fetch (caller builds them locally).
_LOCAL_BUCKETS = {"local-compute", "regenerate"}


class DatasetNotFound(KeyError):
    """The dataset name is not present in the manifest."""


class LocalComputeDataset(RuntimeError):
    """Dataset is not hosted — the demo must fetch its raw source and build it.

    Raised for ``local-compute`` (not redistributable) and ``regenerate`` (cheap
    CPU rebuild) datasets. Carries the manifest ``strategy``/``reason`` text.
    """

    def __init__(self, name: str, spec: dict):
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
def load_manifest(path: Optional[str] = None) -> dict:
    """Load and cache the demo-data manifest (JSON)."""
    p = Path(path) if path else MANIFEST_PATH
    with open(p, "r") as f:
        return json.load(f)


def dataset_spec(name: str, manifest: Optional[dict] = None) -> dict:
    """Return the manifest entry for *name* (raises :class:`DatasetNotFound`)."""
    m = manifest or load_manifest()
    try:
        return m["datasets"][name]
    except KeyError:
        raise DatasetNotFound(
            f"{name!r} not in manifest ({sorted(m['datasets'])[:6]}…)"
        ) from None


def zenodo_file_url(record: dict, filename: str) -> Optional[str]:
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
    name: str, spec: dict, variant: Optional[str]
) -> tuple[list[dict], Optional[str]]:
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
    manifest: Optional[dict] = None,
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
    lfs_subdir = f"{name}/{variant_name}" if variant_name else name
    lfs_dir = _DEMOS_DATA_DIR / lfs_subdir
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
    record: dict,
    verbose: bool,
) -> Path:
    """Resolve a single file: cache → in-repo LFS → Zenodo, in that order."""
    from ..utils.download import download_with_checksum, verify_file_checksum

    # 1. Cache hit (checksum-verified when we know it).
    if dest.exists() and not is_lfs_pointer(dest):
        if sha is None or verify_file_checksum(dest, None, sha):
            if verbose:
                aprint(f"✓ Cached: {fname}")
            return dest

    # 2. In-repo git-LFS copy (the migration fallback).
    lfs_file = lfs_dir / fname
    if lfs_file.exists() and not is_lfs_pointer(lfs_file):
        if _cache_is_stale(dest, lfs_file):
            if verbose:
                aprint(f"Copying {fname} from packaged data to cache")
            shutil.copy2(lfs_file, dest)
        return dest

    # 3. Zenodo (only once the record URL is populated).
    url = zenodo_file_url(record, fname)
    if url:
        if verbose:
            aprint(f"↓ Fetching {fname} from Zenodo")
        return download_with_checksum(url, dest, expected_sha256=sha)

    raise FileNotFoundError(
        f"{fname} is not cached, not present in-repo (git lfs pull), and its "
        "Zenodo record URL is not set yet in manifest.json."
    )


class _null_ctx:
    """No-op context manager for the non-verbose path."""

    def __enter__(self):
        return self

    def __exit__(self, *exc):
        return False
