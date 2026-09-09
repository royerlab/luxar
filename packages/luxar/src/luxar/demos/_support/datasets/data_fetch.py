"""Manifest-driven demo-dataset fetch (R17: retire git-LFS heavy data → Zenodo).

``demos/data_manifest.json`` is the single source of truth for how every demo
dataset is obtained (see :mod:`scripts.gen_data_manifest`). This module reads it
and resolves a dataset's files into the local cache, so demos can migrate off
in-repo git-LFS to fetch-on-demand from Zenodo without changing their own logic.

The manifest is a *packaged resource* — it ships inside the wheel and is always
present. It deliberately does NOT live under ``demos/data/``: that whole tree is
excluded from the wheel and sdist (~450 MB of git-LFS payload), so a manifest
kept there would be missing for exactly the installed users this module serves.

Current ``zenodo`` entries carry one checksum contract: ``sha256`` and ``bytes``
describe what the record serves. The optional ``hosted_sha256`` and
``hosted_bytes`` fields remain supported for legacy manifests where an in-repo
copy differed from the record copy; in that shape the hosted fields are
authoritative for downloads.

Resolution order for a ``zenodo`` dataset (per file). A checksum is the authority
at every step. Bytes matching neither live contract are quarantined unless they
match the most recent ``superseded_sha256`` and no source can replace them; the
download leg remains strict on the record digest:

    1. Local cache ``~/.cache/luxar/<dataset>/<file>``, if it verifies.
    2. Legacy in-repo git-LFS copy ``demos/data/<dir>/<file>``, where ``<dir>``
       is the manifest ``dir`` field. Current ``zenodo`` entries have no such
       payload; the only retained demo payload is the ``regenerate``-bucket Dip-C
       file, outside this fetch path.
    3. Download from the dataset's Zenodo record (checksum-verified), if the
       record has a resolvable URL.
    4. Otherwise a clear error (data neither cached, in-repo, nor hosted yet).

``local-compute`` and ``regenerate`` datasets are NOT fetched here — they raise
:class:`LocalComputeDataset`, signalling the caller to run its own
fetch-raw-and-build path (these are the datasets we cannot redistribute, plus the
cheap CPU-rebuild ones).

A demo that builds its own stand-in for a hosted file (a local GPU refit, when
the record is unpublished and the git-LFS object was never pulled) must NOT store
it at ``<dataset>/<file>``: that path belongs to the manifest, and step 1 above
quarantines anything sitting there that matches neither pinned digest — which a
local fit never does. :func:`local_fit_path` gives such an artifact its own
namespace, ``<dataset>/local/<file>``, which the fetch never looks at (#1618).
"""

from __future__ import annotations

import copy
import hashlib
import json
from collections.abc import Collection, Sequence
from functools import lru_cache
from pathlib import Path
from types import TracebackType
from typing import Any, Generic, Optional, TypeVar

from arbol import aprint, asection

# Reuse the cache root, packaged-data dir and LFS-pointer probe so this module
# and load_precomputed_gsplats share one notion of "the cache".
# NOTE: _cache_is_stale is deliberately NOT imported. Its (size, mtime) test is
# blind to an in-place corruption of unchanged length — precisely how a
# checksum-failing cache entry used to survive step 1 and be handed back.
from .cache import _DEFAULT_CACHE_ROOT
from .lfs import _DEMOS_DATA_DIR, is_lfs_pointer

#: Packaged manifest, resolved the same way ``lfs._DEMOS_DATA_DIR`` is: this
#: module lives below ``luxar/demos/_support/`` and the manifest ships in
#: ``luxar/demos/``.
#: Anchored to ``__file__`` rather than derived from ``_DEMOS_DATA_DIR`` so it
#: stays reachable once R17 step 4 removes the data tree.
MANIFEST_PATH = Path(__file__).resolve().parents[2] / "data_manifest.json"

#: A decoded JSON object out of the manifest — the manifest itself, a dataset
#: spec, a Zenodo record, or a single file entry. The values are heterogeneous
#: JSON (str / int / list / nested object), so ``Any`` is the honest element type.
Manifest = dict[str, Any]
_T = TypeVar("_T")


class ResolvedDataset(list[_T], Generic[_T]):
    """List-compatible resolved values plus their verified input digests."""

    def __init__(self, values: list[_T], input_digests: dict[str, str]) -> None:
        super().__init__(values)
        self.input_digests = dict(sorted(input_digests.items()))


# Buckets that this module does NOT fetch (caller builds them locally).
_LOCAL_BUCKETS = {"local-compute", "regenerate"}


class DatasetNotFound(KeyError):
    """The dataset name is not present in the manifest."""


class DatasetUnavailable(FileNotFoundError):
    """A hosted dataset's bytes cannot be obtained from anywhere — yet.

    The narrow, *routable* half of the ``FileNotFoundError`` surface: the
    manifest lists no files for the entry yet (pending upload), or none of the
    three sources holds a good copy and the record has no URL. Nothing is
    broken; the data simply is not there, so a demo may legitimately fall back
    to computing its own stand-in.

    Every OTHER ``FileNotFoundError`` out of this module is a fault a demo must
    NOT route around — an unknown file name requested of
    :func:`load_dataset_gsplats`, a dataset that lists no gsplat file at all, a
    missing packaged manifest (broken install), an in-repo copy that matches
    neither pinned digest with no hosted fallback. Those stay plain
    ``FileNotFoundError``, so ``except DatasetUnavailable`` lets them through
    instead of disguising them as a routine multi-minute refit.

    Subclasses ``FileNotFoundError``, so a caller that does not care about the
    distinction keeps working.
    """


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
def _load_manifest_cached(path: Optional[str] = None) -> Manifest:
    """Parse the manifest for the most recent path (internal, shared instance)."""
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


def load_manifest(path: Optional[str] = None) -> Manifest:
    """Load the demo-data manifest (JSON).

    Parsing is cached (the most recently requested path), but each call returns
    an independent deep copy: the shared cached dict must never be handed out
    directly, or a caller that mutates the result (or a nested ``dataset_spec``)
    would poison every later read process-wide.
    """
    return copy.deepcopy(_load_manifest_cached(path))


def clear_manifest_cache() -> None:
    """Drop the memoised manifest parse.

    Needed by anything that rewrites a manifest on disk and then reads it back
    (a generator script, a test): the parse is cached, so without this the stale
    copy would be re-served for the rest of the process.
    """
    _load_manifest_cached.cache_clear()


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

    A record that carries ids but is not yet ``published`` returns None for the
    DERIVED form. The ids are reserved and final from deposition time, so they are
    recorded well before the record goes public -- but a file URL into an
    unpublished draft 404s for everyone, and turning a clean "not hosted yet" into
    an HTTP error would be a worse story for the one caller who has no in-repo
    copy. ``published`` therefore gates the URL, not the presence of an id.

    It does NOT gate an explicit ``base_url``, which says "the files are HERE"
    about somewhere other than the record the flag describes. That ordering is
    what the migration runbook's rehearsal needs: point a record's ``base_url``
    at sandbox.zenodo.org and fetch through ``ensure_dataset`` *while the
    production record stays an unpublished draft*. Gating it the other way round
    would force ``published: true`` onto a record that is still a draft -- a lie
    the audit script would then report as LIVE.

    Only an explicit ``published: false`` gates: a record that omits the field
    is treated as reachable, which is what keeps a hand-rolled record (a test
    fixture, or a ``base_url`` pointed at a one-off mirror) working without it.
    The generator always emits the flag, so every shipped record has one --
    ``test_every_shipped_record_agrees_with_its_published_flag`` holds that both
    ways.
    """
    base = record.get("base_url")
    if base:
        return f"{base.rstrip('/')}/{filename}?download=1"
    if record.get("published") is False:
        return None
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


def declared_file_names(
    name: str,
    *,
    variant: Optional[str] = None,
    manifest: Optional[Manifest] = None,
) -> set[str]:
    """Return the file names declared by the resolved dataset variant."""
    spec = dataset_spec(name, manifest)
    files, _variant_name = resolve_variant(name, spec, variant)
    return {entry["name"] for entry in files}


def _select_files(
    name: str,
    files: list[Manifest],
    file_names: Optional[Collection[str]],
    variant_name: Optional[str],
) -> list[Manifest]:
    """Select exact manifest entries without hiding mistakes or splitting groups.

    Unknown names raise so a typo cannot silently resolve the wrong subset.
    Positional sidecars are indexed against their partner, so selecting only
    part of a group could pair unrelated rows and corrupt their alignment.
    """
    if file_names is None:
        return files

    requested = set(file_names)
    available = {entry["name"] for entry in files}
    unknown = requested - available
    if unknown:
        detail = f" variant {variant_name!r}" if variant_name else ""
        raise ValueError(
            f"Requested files are not declared by dataset {name!r}{detail}: "
            f"{sorted(unknown)}"
        )

    groups: dict[str, set[str]] = {}
    for entry in files:
        group = entry.get("positional_pair")
        if group:
            groups.setdefault(group, set()).add(entry["name"])
    for group, members in groups.items():
        selected = requested & members
        if selected and selected != members:
            raise ValueError(
                f"Requested files split positional pair {group!r} in dataset "
                f"{name!r}; select every member: {sorted(members)}"
            )

    return [entry for entry in files if entry["name"] in requested]


def ensure_dataset(
    name: str,
    *,
    variant: Optional[str] = None,
    file_names: Optional[Collection[str]] = None,
    recompute: bool = False,
    cache_root: Optional[Path] = None,
    manifest: Optional[Manifest] = None,
    verbose: bool = True,
) -> ResolvedDataset[Path]:
    """Ensure a ``zenodo`` dataset's files are present locally; return their paths.

    Args:
        name: Dataset key in the manifest (e.g. ``"gsplats_kidney"``).
        variant: For datasets with size variants (e.g. h2afva), which to fetch;
            defaults to the variant flagged ``default`` (the lighter one). An
            error for a dataset without variants, or an unknown variant name.
        file_names: Optional exact set of manifest file names to resolve. The
            result remains in manifest order. Unknown names and selections that
            split a declared ``positional_pair`` are rejected. An empty
            selection returns an empty list without creating a cache directory.
            A non-empty selection against a dataset with no declared files is
            rejected as unknown; query :func:`declared_file_names` first when
            that absence should remain routable as :class:`DatasetUnavailable`.
        recompute: If True, raise :class:`LocalComputeDataset` for *any* dataset
            so the caller takes its own build path (mirrors the demos' ``--recompute``).
        cache_root: Override the cache root (tests). Defaults to ``~/.cache/luxar``.
        manifest: Pre-loaded manifest (tests); defaults to the packaged one.

    Returns:
        Cache paths in manifest order, with a canonical ``input_digests`` map.

    Raises:
        DatasetNotFound: unknown dataset.
        LocalComputeDataset: dataset is local-compute/regenerate (or recompute=True).
        ValueError: unsupported bucket or variant, an unknown selected file name,
            or a selection that splits a declared ``positional_pair``.
        DatasetUnavailable: data is neither cached, in-repo, nor hosted yet, or
            a declared ``positional_pair`` cannot resolve every member to the
            same generation — the conditions a caller may route around by
            building its own copy.
        FileNotFoundError: a fault, not an absence — a missing packaged manifest,
            or an in-repo copy that matches neither pinned digest with no hosted
            fallback.
            :class:`DatasetUnavailable` subclasses this, so catch the subclass
            when you mean "not there yet".
    """
    m = manifest or load_manifest()
    spec = dataset_spec(name, m)
    bucket = spec.get("bucket")

    if recompute or bucket in _LOCAL_BUCKETS:
        raise LocalComputeDataset(name, spec)
    if bucket != "zenodo":
        raise ValueError(f"Dataset {name!r} has unsupported bucket {bucket!r}")

    files, variant_name = resolve_variant(name, spec, variant)
    files = _select_files(name, files, file_names, variant_name)
    if file_names is not None and not files:
        return ResolvedDataset([], {})
    if not files:
        detail = f" variant {variant_name!r}" if variant_name else ""
        raise DatasetUnavailable(
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
    label = f"{name}:{variant_name}" if variant_name else name
    positional_fallbacks = _positional_superseded_fallbacks(
        files, cache_dir, lfs_dir, record, dataset_label=label
    )

    resolved: list[tuple[Path, str]] = []
    with asection(f"Ensuring dataset ({label})") if verbose else _null_ctx():
        for entry in files:
            fname = entry["name"]
            dest = cache_dir / fname
            if fname in positional_fallbacks:
                resolved.append((dest, positional_fallbacks[fname]))
                continue
            resolved.append(
                _ensure_one(
                    dest,
                    fname,
                    entry.get("sha256"),
                    lfs_dir,
                    record,
                    verbose,
                    hosted_sha=entry.get("hosted_sha256"),
                    superseded=tuple(entry.get("superseded_sha256") or ()),
                )
            )
    result = ResolvedDataset(
        [path for path, _ in resolved],
        {path.name: digest for path, digest in resolved},
    )
    from ..runtime.provenance import _record_input_digests

    _record_input_digests(result.input_digests)
    return result


def _matches(path: Path, expected: Optional[str], verbose: bool) -> bool:
    """True only on a POSITIVE checksum match.

    ``verify_file_checksum`` returns True vacuously when given no expected hash,
    so a missing digest is excluded here rather than left to a short-circuit that
    a later edit could drop.
    """
    from ..downloads.download import verify_file_checksum

    return (
        expected is not None
        and path.is_file()
        and verify_file_checksum(path, None, expected, verbose=verbose)
    )


def _contract_candidates(
    sha: Optional[str], hosted_sha: Optional[str], superseded: Sequence[str] = ()
) -> list[tuple[str, str]]:
    """Acceptable digests in priority order, de-duplicated.

    A digest repeated across roles is checked once and reported under the
    strongest role that names it, so a pin that is simultaneously current and
    listed as superseded never reads as out of date.

    Only the LAST superseded entry is offered, even though the manifest keeps the
    full history: the list's effect IS how far back "acceptable" reaches, and the
    oldest entry is the likeliest to be genuinely wrong. Anything older than one
    generation degrades to a build failure — loud and recoverable — rather than
    to a silently stale artifact.
    """
    candidates: list[tuple[str, str]] = []
    for digest, kind in (
        (hosted_sha, "hosted"),
        (sha, "local"),
        *((d, "superseded") for d in list(superseded or ())[-1:]),
    ):
        if digest and all(digest != known for known, _ in candidates):
            candidates.append((digest, kind))
    return candidates


def _accepted_contract(
    path: Path,
    sha: Optional[str],
    hosted_sha: Optional[str],
    verbose: bool,
    superseded: Sequence[str] = (),
) -> Optional[tuple[str, str]]:
    """Which contract *path* satisfies as ``(kind, digest)``, or None.

    Hosted is tried first so the canonical answer is the one reported when both
    would match — which is every case where the two pins agree.

    ``"superseded"`` is a deliberately WEAKER verdict: those are digests this
    project pinned in an EARLIER generation, so bytes matching one are known-good
    data that is merely out of date, not corruption. That distinction is the whole
    point of recording them — without it a re-pinned dataset with no fetch route
    is indistinguishable from a corrupt one, and the only safe response to
    ambiguity is to quarantine, which destroys the last copy in existence. Only
    :func:`_resolve_from_cache` may act on this verdict, and only as a last
    resort.
    """
    candidates = _contract_candidates(sha, hosted_sha, superseded)
    if not candidates:
        return None
    if len(candidates) == 1:
        digest, kind = candidates[0]
        return (kind, digest) if _matches(path, digest, verbose) else None
    if not path.is_file():
        return None

    return _verdict_from_one_pass(path, candidates, verbose)


def _has_current_source(lfs_dir: Path, record: Manifest, fname: str) -> bool:
    """Whether the current generation can be obtained without cache fallback."""
    lfs_file = lfs_dir / fname
    has_repo_copy = lfs_file.is_file() and not is_lfs_pointer(lfs_file)
    return has_repo_copy or zenodo_file_url(record, fname) is not None


def _source_remedy(lfs_dir: Path, fname: str) -> str:
    """Explain how the caller can make the current generation obtainable."""
    if (lfs_dir / fname).exists():
        return "In a source checkout, run `git lfs pull`."
    if not _DEMOS_DATA_DIR.exists():
        return (
            "This installed package ships no demo payloads. If this archive has "
            "an in-repo copy, use a source checkout and run `git lfs pull`."
        )
    return "This archive is hosted-only and has no in-repo Git LFS copy."


def _positional_superseded_fallbacks(
    files: list[Manifest],
    cache_dir: Path,
    lfs_dir: Path,
    record: Manifest,
    *,
    dataset_label: str,
) -> dict[str, str]:
    """Return members that may reuse one complete superseded generation.

    The ordinary resolver decides fallback eligibility per file. Positionally
    indexed sidecars cannot: if one member has no declared current source and
    must stay on its previous generation, every member must have that same
    generation in cache. Otherwise returning the files would silently pair
    unrelated rows. A declared source that later fails to download or verify is
    handled by the ordinary per-file resolver; this preflight does not prove
    that current bytes are obtainable.
    """
    groups: dict[str, list[Manifest]] = {}
    for entry in files:
        group = entry.get("positional_pair")
        if group:
            groups.setdefault(group, []).append(entry)

    fallbacks: dict[str, str] = {}
    for group, entries in groups.items():
        verdicts: dict[str, Optional[tuple[str, str]]] = {}

        def cached_verdict(entry: Manifest) -> Optional[tuple[str, str]]:
            fname = entry["name"]
            if fname not in verdicts:
                verdicts[fname] = _accepted_contract(
                    cache_dir / fname,
                    entry.get("sha256"),
                    entry.get("hosted_sha256"),
                    False,
                    tuple(entry.get("superseded_sha256") or ()),
                )
            return verdicts[fname]

        # Keep the source check first: source checkouts then avoid hashing a
        # cached superseded payload that the ordinary resolver will refresh.
        forced = any(
            not _has_current_source(lfs_dir, record, entry["name"])
            and (cached_verdict(entry) or (None, None))[0] == "superseded"
            for entry in entries
        )
        if not forced:
            continue

        history_lengths = {
            len(entry.get("superseded_sha256") or ()) for entry in entries
        }
        all_superseded = all(
            (cached_verdict(entry) or (None, None))[0] == "superseded"
            for entry in entries
        )
        source_remedies = " ".join(
            f"{entry['name']}: {_source_remedy(lfs_dir, entry['name'])}"
            for entry in entries
        )
        if not all_superseded or len(history_lengths) != 1:
            states = ", ".join(
                f"{entry['name']}="
                f"{(cached_verdict(entry) or ('missing/corrupt', ''))[0]}"
                for entry in entries
            )
            raise DatasetUnavailable(
                f"Cannot assemble dataset {dataset_label!r} positional pair {group!r}: "
                "one member requires a superseded cache fallback, but the cached "
                f"members are not the same generation ({states}). Obtain every "
                "current member or restore the complete prior generation. "
                f"Source remedies: {source_remedies}"
            )

        fallbacks.update(
            {
                entry["name"]: verdict[1]
                for entry in entries
                if (verdict := cached_verdict(entry)) is not None
            }
        )
        aprint(
            f"⚠️  Using SUPERSEDED positional pair {group!r} for dataset "
            f"{dataset_label!r}: every member matches the same previously pinned "
            "generation, and at least one current member is unavailable. The pair "
            f"is out of date, not corrupt. Source remedies: {source_remedies}"
        )
    return fallbacks


def _verdict_from_one_pass(
    path: Path, candidates: list[tuple[str, str]], verbose: bool
) -> Optional[tuple[str, str]]:
    """Hash *path* ONCE and report which candidate it matches, if any.

    Used when more than one digest is acceptable, so the file is not read once
    per candidate.
    """
    with asection(f"Verifying {path.name}") if verbose else _null_ctx():
        if verbose:
            aprint("Computing SHA256...")
        state = hashlib.sha256()
        with open(path, "rb") as file:
            for chunk in iter(lambda: file.read(8192 * 128), b""):
                state.update(chunk)
        actual = state.hexdigest()

        for digest, kind in candidates:
            if actual == digest:
                if verbose:
                    aprint(f"✓ SHA256 verified ({kind}): {actual}")
                return kind, actual

        if verbose:
            local_label = (
                "in-repo:"
                if any(kind == "hosted" for _, kind in candidates)
                else "record:"
            )
            aprint("❌ SHA256 mismatch!")
            label = {
                "hosted": "hosted:",
                "local": local_label,
                "superseded": "superseded:",
            }
            for digest, kind in candidates:
                aprint(f"   Expected {label[kind]:<12s}{digest}")
            aprint(f"   {'Actual:':<21s}{actual}")
        return None


def _resolve_from_cache(
    dest: Path,
    fname: str,
    sha: Optional[str],
    hosted_sha: Optional[str],
    unverifiable: bool,
    verbose: bool,
    superseded: Sequence[str] = (),
    irreplaceable: bool = False,
    source_remedy: str = "",
) -> Optional[tuple[Path, str]]:
    """Step 1: reuse the cached copy, or quarantine it and return None.

    Returning None carries the invariant the rest of ``_ensure_one`` depends on:
    ``dest`` does not exist, so nothing wrong can be trusted or promoted in its
    place (``robust_download`` would otherwise resume onto stale bytes).
    """
    from ..downloads.download import quarantine_file

    if not dest.exists():
        return None
    if not dest.is_file():
        raise IsADirectoryError(
            f"Cache entry {dest} exists but is not a regular file; remove it "
            "(or run 'luxar demo cache clear') and retry."
        )
    if unverifiable and not is_lfs_pointer(dest):
        # Unverifiable: a pending-upload entry, or a manifest predating the
        # checksum. Reuse it — but say so. Silence is how corruption lives.
        if verbose:
            aprint(f"✓ Cached (UNVERIFIED — no sha256 in manifest): {fname}")
        return dest, _file_sha256(dest)
    accepted = _accepted_contract(dest, sha, hosted_sha, verbose, superseded)
    if accepted is not None and accepted[0] == "superseded":
        # Known-good bytes from an earlier generation. Keep them ONLY when there
        # is nothing better to be had: quarantining here would destroy the last
        # copy in existence and leave the dataset unobtainable, which is how a
        # re-pin bricked a hosted-only demo (see the changelog). When any route
        # to the current bytes exists we fall through and take it instead.
        if irreplaceable:
            aprint(
                f"⚠️  Using a SUPERSEDED copy of {fname}: it matches a digest this "
                "project pinned previously, and no current copy is immediately "
                f"available. {source_remedy} The record is not published. It is "
                "out of date, not corrupt."
            )
            return dest, accepted[1]
        if verbose:
            aprint(f"↻ Cached copy of {fname} is superseded; fetching the current one")
    elif accepted is not None:
        _report_contract(
            accepted[0], "✓ Cached (sha256 verified):", fname, sha, hosted_sha, verbose
        )
        return dest, accepted[1]
    quarantine_file(
        dest,
        reason=(
            "unpulled git-LFS pointer"
            if is_lfs_pointer(dest)
            else (
                "matches neither the in-repo nor the hosted sha256 (corrupt, "
                "or superseded by a data update)"
                if hosted_sha
                else "matches neither the record nor a superseded sha256 (corrupt, "
                "or superseded by a data update)"
            )
        ),
        verbose=verbose,
    )
    return None


def _report_contract(
    kind: str,
    prefix: str,
    fname: str,
    sha: Optional[str],
    hosted_sha: Optional[str],
    verbose: bool,
) -> None:
    """One line about which contract was satisfied, when it is worth saying.

    Only a DIVERGENCE is worth a remark: a hosted pin equal to the local one is
    the majority case, and narrating it would train people to ignore the notice.
    """
    if not verbose:
        return
    divergent = hosted_sha is not None and sha is not None and hosted_sha != sha
    if kind == "local" and divergent:
        aprint(
            f"{prefix} {fname} matches the in-repo sha256, but the record "
            "hosts a newer build (hosted_sha256 differs)"
        )
    else:
        aprint(f"{prefix} {fname}")


def _file_sha256(path: Path) -> str:
    """Return the sha256 of ``path``."""
    state = hashlib.sha256()
    with open(path, "rb") as file:
        for chunk in iter(lambda: file.read(8192 * 128), b""):
            state.update(chunk)
    return state.hexdigest()


def _ensure_one(
    dest: Path,
    fname: str,
    sha: Optional[str],
    lfs_dir: Path,
    record: Manifest,
    verbose: bool,
    hosted_sha: Optional[str] = None,
    superseded: Sequence[str] = (),
) -> tuple[Path, str]:
    """Resolve one file: cache → in-repo LFS → Zenodo, checksum-authoritative.

    Current manifests use one contract: ``sha`` (manifest ``sha256``) describes
    the RECORD copy. ``hosted_sha`` (``hosted_sha256``) is optional legacy input
    for manifests that still distinguish an in-repo copy from the record copy;
    when present, it remains authoritative for downloads.

    Bytes already in hand prefer the two live contracts in order — hosted
    (canonical, silent), then local (usable, with a one-line notice that the
    record holds something newer). A cached copy matching the most recent
    ``superseded_sha256`` is a third, weaker verdict: it is reused only when no
    in-repo copy or download route can replace it, and is reported as out of
    date. Both legs that can supply live bytes have to agree on the live
    contracts: the cache slot is keyed on
    ``(dataset, variant, basename)`` and nothing else, with no record of which
    source filled it, so a cache leg stricter than the leg that wrote it would
    quarantine its own copy and re-make it on every single run.

    Only the DOWNLOAD leg is strict, and strictly on the record digest
    (``hosted_sha`` for a legacy dual contract, otherwise ``sha``): bytes
    arriving from the record must be the record's bytes.

    The in-repo preference remains for legacy dual-contract manifests, but no
    current ``zenodo`` dataset has an in-repo payload; non-``zenodo`` datasets
    raise :class:`LocalComputeDataset` before reaching this helper. Current
    manifests therefore resolve through the checksum-verified cache or record.

    The checksum authority at every step:

    * a cached file matching neither live contract is QUARANTINED unless it
      matches the newest superseded digest and is irreplaceable. The previous
      version fell through to step 2 instead, where a (size, mtime) staleness
      test could not see an in-place corruption and returned the bad file;
    * a copy taken from the in-repo git-LFS tree is verified AFTER copying, so
      the checksum we already hold is actually used rather than carried around;
    * the Zenodo leg is handed no stale file at ``dest``: ``robust_download``
      now stages every download to a sibling ``.part`` and atomically promotes
      it only once complete, so it never resumes onto (or appends to) a stale
      file sitting at the destination (issues #731/#732).

    Quarantining in step 1 keeps that clean regardless: below it, ``dest`` does
    not exist, so nothing wrong can be trusted or promoted in its place.
    """
    from ....utils.atomic_copy import atomic_copy_file
    from ..downloads.download import download_with_checksum, quarantine_file

    # The record's bytes if we know them, else the only digest we have.
    download_sha = hosted_sha or sha
    unverifiable = sha is None and hosted_sha is None

    # ── 1. Local cache ──────────────────────────────────────────────────────
    # Decide UP FRONT whether the current bytes are obtainable at all, because
    # the cache leg's quarantine is irreversible and must not fire on a file it
    # cannot replace. This is the ordering bug that bricked a hosted-only demo:
    # the old flow quarantined first and discovered "no source" afterwards.
    # Same shape as #854, where a quarantine on a stale EXPECTATION destroyed a
    # good file and looped; there the fix was to stop quarantining, and here the
    # superseded list is what lets us tell "out of date" from "corrupt".
    lfs_file = lfs_dir / fname
    has_repo_copy = lfs_file.is_file() and not is_lfs_pointer(lfs_file)
    url = zenodo_file_url(record, fname)
    irreplaceable = not _has_current_source(lfs_dir, record, fname)
    source_remedy = _source_remedy(lfs_dir, fname)

    cached = _resolve_from_cache(
        dest,
        fname,
        sha,
        hosted_sha,
        unverifiable,
        verbose,
        superseded=superseded,
        irreplaceable=irreplaceable,
        source_remedy=source_remedy,
    )
    if cached is not None:
        return cached

    # INVARIANT from here on: `dest` does not exist.

    # ── 2. In-repo git-LFS copy (the migration fallback) ────────────────────
    inrepo_is_bad = False
    if has_repo_copy:
        if verbose:
            aprint(f"Copying {fname} from packaged data to cache")
        # Atomic: a Ctrl-C mid-copy must not leave a truncated file under the
        # canonical name for the next run to quarantine and re-fetch.
        atomic_copy_file(lfs_file, dest)
        if unverifiable:
            return dest, _file_sha256(dest)
        accepted = _accepted_contract(dest, sha, hosted_sha, verbose)
        if accepted is not None:
            _report_contract(
                accepted[0],
                "✓ Copied from packaged data:",
                fname,
                sha,
                hosted_sha,
                verbose,
            )
            return dest, accepted[1]
        # The copy is wrong. Hash the SOURCE to apportion blame — this second
        # pass only ever runs on this failure path.
        source_ok = _accepted_contract(lfs_file, sha, hosted_sha, verbose) is not None
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
            f"⚠️  In-repo copy of {fname} matches neither the in-repo nor the "
            f"hosted sha256 ({lfs_file}). Re-pull it ('git lfs pull'), or "
            "regenerate the manifest (make gen-data-manifest) if the data "
            "changed."
        )

    # ── 3. Zenodo (only once the record URL is populated) ───────────────────
    if url:
        if verbose:
            aprint(f"↓ Fetching {fname} from Zenodo")
        # `dest` is absent by the invariant above, so robust_download starts at
        # byte 0 instead of resuming onto (appending to) a stale file.
        # Strictly the hosted digest: bytes from the record must be the
        # record's bytes, whatever the in-repo copy happens to be.
        downloaded = download_with_checksum(url, dest, expected_sha256=download_sha)
        return downloaded, download_sha or _file_sha256(downloaded)

    if inrepo_is_bad:
        # Deliberately NOT `DatasetUnavailable`: the bytes are RIGHT THERE and
        # wrong. That is a broken checkout or a stale manifest, and a demo that
        # swallowed it would present a repo fault as a routine refit.
        raise FileNotFoundError(
            f"{fname} is present in-repo but matches neither its in-repo nor its "
            "hosted sha256, and the dataset has no Zenodo URL yet — there is no "
            "good copy to fall back to. Re-pull the LFS object, or regenerate the "
            "manifest if the data was intentionally updated."
        )
    raise DatasetUnavailable(
        f"{fname} is not cached (any cached copy failed its checksum and was "
        "quarantined), not available from the in-repo Git LFS copy, and the "
        "demo-data manifest builds no Zenodo URL for it yet — its record has no "
        f"id, or is still an unpublished draft. {source_remedy} Publish the "
        "record and populate its manifest "
        "URL, or rerun the demo with --recompute when that demo provides a build "
        "path."
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


#: Subdirectory, inside a dataset's cache dir, holding artifacts the machine
#: computed for itself rather than obtained from the manifest.
LOCAL_FIT_DIRNAME = "local"


def local_fit_path(
    name: str,
    filename: str,
    *,
    variant: Optional[str] = None,
    cache_root: Optional[Path] = None,
) -> Path:
    """Where a demo's OWN locally computed stand-in for a hosted file belongs.

    Returns ``<cache_root>/<name>/local/<filename>`` (plus the variant subdir
    when one is given, mirroring :func:`ensure_dataset`'s layout).

    The split exists because ``<name>/<filename>`` — with no ``local/`` in it —
    is the path :func:`ensure_dataset` resolves for that manifest entry, and step
    1 of :func:`_ensure_one` treats whatever it finds there as a candidate copy
    of the manifest file: it hashes it against both pinned digests and
    QUARANTINES it when neither matches. A local fit is a different artifact that
    happens to answer the same need, so it can never match either digest. Storing
    one under the manifest name therefore guarantees it is destroyed by the next
    fetch, and the demo refits from scratch on every single launch (#1618/#1672).

    Nothing under ``local/`` is ever hashed, quarantined or overwritten by the
    fetch — the cache dir is shared, the two namespaces are not.

    Args:
        name: Manifest dataset key, i.e. the cache-dir name (``"gsplats_dapi"``).
        filename: Basename of the computed artifact. Deliberately allowed to be
            the manifest's own file name: reusing it documents what the local
            artifact stands in for, and is now safe.
        variant: Size variant, for a dataset that has them; see
            :func:`ensure_dataset`. ``None`` (every dataset that needs this
            today) puts the file directly under ``<name>/local/``.
        cache_root: Override the cache root (tests). Defaults to
            ``~/.cache/luxar``.

    Raises:
        ValueError: if *variant* is ``"local"``, which is the one name that
            would put a manifest destination and this namespace back on top of
            each other. ``test_no_shipped_variant_is_named_local`` holds the
            shipped manifest to it too, so the check can only fire on a hand
            rolled call.
    """
    if variant == LOCAL_FIT_DIRNAME:
        raise ValueError(
            f"A variant named {LOCAL_FIT_DIRNAME!r} would collide with the "
            "local-fit namespace: ensure_dataset caches a variant's files at "
            f"<name>/<variant>/<file>, i.e. <name>/{LOCAL_FIT_DIRNAME}/<file> "
            "— the very directory this namespace exists to keep out of its "
            "reach. Rename the variant."
        )
    root = Path(cache_root) if cache_root else _DEFAULT_CACHE_ROOT
    parts = [p for p in (name, variant, LOCAL_FIT_DIRNAME) if p]
    return root.joinpath(*parts, filename)


def load_local_fit_gsplats(
    name: str,
    file_names: list[str],
    *,
    variant: Optional[str] = None,
    cache_root: Optional[Path] = None,
    verbose: bool = True,
) -> Optional[list[Any]]:
    """Load a previous run's own local fit, or ``None`` if the caller must build it.

    Same return contract as :func:`load_dataset_gsplats` — a list of
    ``GSplatData`` in the requested order, or ``None`` meaning "build it
    yourself" — but it reads the :func:`local_fit_path` namespace instead of the
    manifest. A demo consults it AFTER the manifest fetch comes up empty and
    BEFORE it refits, which is what makes the "one-time" refit actually one-time.

    ``None`` is returned when any requested file is missing (a partial set is not
    a usable answer: the caller refits, and the fit rewrites all of them), and
    also when one of them fails to load.

    A broken local file does NOT raise. Unlike the manifest cache, these bytes
    have no checksum, no remote to re-fetch from and no second copy — the only
    recovery is the refit the caller is already able to do, so raising would
    strand a demo on rubble it can heal itself. It is reported loudly (⚠️, with
    the path and the error) rather than silently: a fit that keeps re-running is
    the bug this whole namespace exists to fix, so a machine that has quietly
    started refitting every launch must be able to see why. The bad file is left
    in place for inspection; the refit overwrites it.

    The ONE exception is a store that is structurally unloadable — a
    ``kind=partition`` / non-leaf lod tree, which has no flat ``GSplatData``
    form at all. See :func:`load_local_fit_gsplats_at`.

    An empty ``file_names`` raises: ``[]`` is neither a loaded set nor "rebuild
    it", and returning it would break the ``if fits is not None: fits[0]`` shape
    every caller uses.
    """
    paths = [
        local_fit_path(name, f, variant=variant, cache_root=cache_root)
        for f in file_names
    ]
    return load_local_fit_gsplats_at(paths, label=name, verbose=verbose)


#: Reported when a local fit is present but unreadable — corrupt bytes, a
#: truncated zip, a half-written store. The refit is the recovery.
_LOCAL_FIT_UNREADABLE = (
    "⚠️  Local fit {path} could not be loaded ({exc}). Rebuilding it from "
    "scratch; delete the file if the rebuild keeps happening."
)


def load_local_fit_gsplats_at(
    paths: list[Path],
    *,
    label: str = "local fit",
    verbose: bool = True,
) -> Optional[list[Any]]:
    """:func:`load_local_fit_gsplats` for paths the caller already holds.

    The name-based form re-derives its paths from the cache root, which is the
    right default but is NOT the same object as a demo's module-level
    ``LOCAL_FIT`` constant. A demo that publishes such a constant — and writes
    its refit through it — must READ through it too, or the two halves of its
    local door can be pointed at different files (a redirected constant that the
    read side silently ignores; #1618 review finding A). Those demos call this.

    Args:
        paths: The artifacts to load, in the order they should be returned.
        label: What to call this set in log output (usually the dataset name).
        verbose: Print progress.

    Returns:
        ``list[GSplatData]``, or ``None`` when the caller should rebuild.

    Raises:
        ValueError: *paths* is empty, or the store is multi-part
            (``kind=partition`` or a lod group with non-leaf children) and
            therefore has no flat ``GSplatData`` form. The latter is
            deliberately NOT swallowed into a rebuild: the rebuild would write
            the same unloadable shape, so the caller would refit on every launch
            — exactly the #1618 symptom this namespace exists to end. It is a
            recipe/loader mismatch in the demo, and only a code change fixes it.
            :func:`load_dataset_gsplats` translates the same error.
    """
    from ....gsplats.gsplat_data import GSplatData

    if not paths:
        # `[]` would otherwise sail through as a successful load of nothing, and
        # the contract here is "a list, or None meaning rebuild" — a caller that
        # tested `is not None` and indexed [0] would get an IndexError instead
        # of taking its rebuild branch.
        raise ValueError(
            f"load_local_fit_gsplats_at({label!r}) was given no paths; asking for "
            "zero artifacts is a caller bug, not an empty local fit."
        )

    missing = [p for p in paths if not p.exists()]
    if missing:
        if verbose and len(missing) < len(paths):
            aprint(
                f"Local fit for {label!r} is incomplete "
                f"({len(paths) - len(missing)}/{len(paths)} files) — rebuilding."
            )
        return None

    results: list[Any] = []
    with asection(f"Loading local fit ({label})") if verbose else _null_ctx():
        for path in paths:
            try:
                gsplats = GSplatData.load(path, include_stats=False)
            except ValueError as exc:
                if "matrix-shaped" not in str(exc):
                    aprint(_LOCAL_FIT_UNREADABLE.format(path=path, exc=repr(exc)))
                    return None
                raise ValueError(
                    f"{path} is a multi-part gsplats store (a partition, or a lod "
                    "group with non-leaf children), which has no flat GSplatData "
                    f"form and cannot be loaded by this helper: {exc}\n"
                    "Rebuilding it would write the same shape again and refit on "
                    "EVERY launch, so this is raised rather than routed around. "
                    "Either save the local fit with a flat recipe, or keep the "
                    "PATHS and hand them to Group.add_gsplats_from_file(), which "
                    "grafts a multi-part subtree whole."
                ) from exc
            except Exception as exc:  # noqa: BLE001 — see the docstring
                aprint(_LOCAL_FIT_UNREADABLE.format(path=path, exc=repr(exc)))
                return None
            if verbose:
                aprint(f"Loaded {path.name}: {len(gsplats.amplitudes):,} splats")
            results.append(gsplats)
    return results


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
) -> Optional[ResolvedDataset[Any]]:
    """Manifest-driven stand-in for :func:`luxar.demos.load_precomputed_gsplats`.

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

    :class:`DatasetUnavailable` is the one routable absence: the data is neither
    cached, in-repo, nor hosted yet, so the caller may build its own copy.
    Everything else raises — an unknown dataset, an unknown variant, or a
    checksum that will not verify. Those are faults a demo must not silently
    route around.

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
        * **Bundle datasets.** ``gsplats_celegans`` ships one outer zip holding
          many per-frame files; the manifest addresses the bundle, not its
          members. That demo stays on
          :func:`~luxar.demos.load_precomputed_bundle`. (``gsplats_zebrafish``
          was one until it moved to a single stacked 4D archive, which this
          function serves.)
        * **Runtime-computed file lists** that are not manifest entries. A
          *subset* of the manifest's own files is fine; anything else raises
          rather than fetching the wrong data.
        * **Non-gsplat payloads.** Use :func:`ensure_dataset`, which returns paths
          and assumes nothing about the format.
        * **Multi-part artifacts.** A ``kind=partition`` store — what
          :func:`luxar.demos._lod_policy.save_with_lod` writes for the
          ``adaptive`` recipe, and what ``gsplat lod --recipe
          tiles|overview|adaptive`` writes — has no flat ``GSplatData`` form and
          is meant to be GRAFTED whole. Fetch the paths with
          :func:`ensure_dataset` and hand them to
          :meth:`~luxar.core.group.Group.add_gsplats_from_file`. This function
          says so rather than letting the shape error surface bare.
        * **Datasets whose bucket is not ``zenodo``.** ``gsplats_tribolium``,
          ``gsplats_acto3d_heart``, ``gsplats_tng_cosmic_web`` and
          ``milky_way_gaia_3m`` are marked ``local-compute`` and no longer ship
          files in-repo, so this returns ``None`` for them and the demo takes its
          own build path: a GPU refit for the three gsplat ones, and for Gaia an
          opt-in ``--build-catalog`` query of the ESA archive (or a copy the user
          already placed in the cache). Migrate a demo only once its dataset is
          ``zenodo``.
    """
    if recompute:
        return None

    from ....gsplats.gsplat_data import GSplatData

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
    input_digests = {path.name: paths.input_digests[path.name] for path in selected}
    with asection(f"Loading gsplats ({name})") if verbose else _null_ctx():
        for path in selected:
            try:
                gsplats = GSplatData.load(path, include_stats=False)
            except ValueError as exc:
                if "matrix-shaped" not in str(exc):
                    raise
                # A partition / multi-part LOD store. The bare shape error names
                # an internal method and leaves the caller nowhere to go, and
                # this is a mistake a demo makes by CHANGING its cache recipe —
                # so point at the entry point that does handle one.
                raise ValueError(
                    f"{path.name} is a multi-part gsplats store (a partition, or "
                    "a lod group with non-leaf children), which has no flat "
                    "GSplatData form and cannot be loaded by "
                    f"load_dataset_gsplats: {exc}\n"
                    "Graft it instead: take the paths from ensure_dataset("
                    f"{name!r}) and pass each to Group.add_gsplats_from_file(), "
                    "which loads a multi-part subtree whole."
                ) from exc
            if verbose:
                aprint(f"Loaded {path.name}: {len(gsplats.amplitudes):,} splats")
            results.append(gsplats)
    return ResolvedDataset(results, input_digests)
