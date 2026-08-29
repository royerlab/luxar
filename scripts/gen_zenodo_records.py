#!/usr/bin/env python3
"""Generate the Zenodo record descriptions from the manifest and the archives.

The three demo records describe several dozen files between them, each with a
splat count, a compression figure, a reconstruction quality and a licence. Typed
by hand that drifts from the data within one refit, and a record is the one place
where a stale number is published rather than merely wrong.

So the text is derived: dataset disposition and licensing come from
``data_manifest.json``, and the per-dataset characteristics come from
``scripts/demo_archive_characteristics.json`` — measured out of each archive's
own ``fitting/`` stamps by ``--refresh`` and committed. A figure that is not
stamped is reported as absent rather than guessed at.

Measurement is separate from rendering because the archives are LEAVING the
repository. Reading them at render time meant the text could only be generated
on a machine holding ~400 MB of demo data, and on a partial checkout it reported
"no PSNR" for archives whose hosted copies are stamped — publishing an absent
figure for data that has one. Digest-backed archive measurements carry a
``measured_sha256`` so ``--check`` can say when the bytes read by ``--refresh`` are
no longer pinned, which is the drift a refit causes. Source-derived figures may
retain their own provenance note after a later pinned archive read supplies that
digest.

``quality_note`` is internal provenance and is never rendered. ``quality_caveat``
is reader-facing text rendered next to an absent figure. ``unmeasured_reason`` is
written by ``--refresh`` when a local file exists but is not the pinned artifact
and no committed measurement survives that rejected read.

This script NEVER talks to Zenodo. It writes markdown for a human to paste into
a draft, and publication stays a manual act.

    python scripts/gen_zenodo_records.py                 # all records to stdout
    python scripts/gen_zenodo_records.py --record cc-by  # just one
    python scripts/gen_zenodo_records.py --outdir docs/zenodo/
    python scripts/gen_zenodo_records.py --check         # report gaps, exit 1
    python scripts/gen_zenodo_records.py --refresh       # re-measure the archives
"""

from __future__ import annotations

import argparse
import io
import json
import sys
import zipfile
from collections.abc import Iterator
from pathlib import Path
from typing import Any, Optional

REPO_ROOT = Path(__file__).resolve().parent.parent
MANIFEST = REPO_ROOT / "packages/luxar/src/luxar/demos/data_manifest.json"
DATA_DIR = REPO_ROOT / "packages/luxar/src/luxar/demos/data"
CACHE_DIR = Path.home() / ".cache/luxar"
#: Measured characteristics, committed so the record text does not depend on
#: holding 400 MB of archives. See `load_characteristics`.
CHARACTERISTICS = REPO_ROOT / "scripts/demo_archive_characteristics.json"

_ABSENT = "—"


# ---------------------------------------------------------------------------
# Reading an archive's own stamps
# ---------------------------------------------------------------------------


def _attrs(zf: zipfile.ZipFile, root: str, node: str = "") -> dict[str, Any]:
    """Attrs of a group inside a zipped store, zarr v2 or v3."""
    for candidate in (f"{root}{node}.zattrs", f"{root}{node}zarr.json"):
        try:
            raw = json.loads(zf.read(candidate))
        except KeyError:
            continue
        except (ValueError, UnicodeDecodeError):
            # Present but not JSON: not a store this script can describe. The
            # alternative is a traceback halfway through a record, which is the
            # one failure mode the corrupt-archive handling exists to avoid.
            continue
        attrs = raw.get("attributes", raw) if candidate.endswith("zarr.json") else raw
        return attrs if isinstance(attrs, dict) else {}
    return {}


def _as_int(value: Any) -> Optional[int]:
    """*value* as an int, or ``None`` for anything that is not one."""
    try:
        return int(value)
    except (TypeError, ValueError):
        return None


def _finite(value: Any) -> bool:
    """True for a real measurement: a number, not a bool, not nan/inf."""
    return (
        isinstance(value, (int, float))
        and not isinstance(value, bool)
        and value == value
        and value not in (float("inf"), float("-inf"))
    )


def _describe_topology(root_attrs: dict[str, Any], groups: set[str]) -> str:
    """Name the LOD topology from what is actually on disk.

    Reads the structural ``kind`` the writer stamped rather than the ``recipe``
    provenance, because ``kind`` is what the viewer acts on -- a mismatch
    between the two is exactly the sort of thing worth seeing in the record.
    """
    kind = root_attrs.get("kind")
    rungs = _as_int(root_attrs.get("n_additive_sublods"))
    if kind == "partition":
        parts = len({g.split("/")[0] for g in groups if g.startswith("part")})
        has_levels = any("/child_" in g for g in groups)
        detail = f"{parts} spatial tiles"
        if has_levels:
            detail += ", each with its own detail levels"
        return detail
    if kind == "lod":
        levels = len({g.split("/")[0] for g in groups if g.startswith("child")})
        laddered = any("/additive_" in g for g in groups)
        detail = f"{levels} coarse-to-fine levels"
        if laddered:
            detail += ", each progressively streamed"
        return detail
    if rungs and rungs > 1:
        return f"progressive ladder, {rungs} steps"
    return "single level"


def _read_archive(path: Path) -> Optional[dict[str, Any]]:
    """Characteristics of one archive, or ``None`` if it cannot be read.

    Handles the timelapse BUNDLES (a zip of per-frame zips): a naive read finds
    no store at all in those and would report every field as absent.
    """
    try:
        with zipfile.ZipFile(path) as zf:
            names = zf.namelist()
            if not names:
                return None
            inner = sorted(n for n in names if n.endswith(".gsplats.zarr.zip"))
            if inner:
                return _read_bundle(zf, inner)
            return _read_store(zf)
    except (zipfile.BadZipFile, OSError):
        return None


def _read_bundle(zf: zipfile.ZipFile, inner: list[str]) -> Optional[dict[str, Any]]:
    """Aggregate a bundle over ALL its frames rather than describing the first.

    Every figure in a record is a statement about the *archive*, so it has to
    cover the whole archive: the first frame's splat count understates a
    400-frame bundle 400-fold, and its ``source_bytes`` divided by the whole
    bundle's stored size would price one frame's compression as the bundle's.
    Counts are therefore summed -- only when every frame carries one, since a
    partial sum published as a total is the same falsehood -- and the qualities,
    which are inherently per frame, are reported as the range across frames.
    """
    frames = []
    for name in inner:
        try:
            with zipfile.ZipFile(io.BytesIO(zf.read(name))) as frame:
                info = _read_store(frame)
        except (zipfile.BadZipFile, OSError, KeyError):
            info = None
        if info is not None:
            frames.append(info)
    if not frames:
        return None
    complete = len(frames) == len(inner)
    out = dict(frames[0])
    out["frames"] = len(inner)
    out["n_splats"] = _total(frames, "n_splats") if complete else None
    out["source_bytes"] = _total(frames, "source_bytes") if complete else None
    out["psnr_db"] = _span(frames, "psnr_db") if complete else None
    out["foreground_psnr_db"] = (
        _span(frames, "foreground_psnr_db") if complete else None
    )
    return out


def _total(frames: list[dict[str, Any]], key: str) -> Optional[int]:
    """Sum of *key* over frames, or ``None`` unless every frame carries it."""
    values = [f.get(key) for f in frames]
    if any(not isinstance(v, int) or isinstance(v, bool) for v in values):
        return None
    return sum(values)  # type: ignore[arg-type]


def _span(frames: list[dict[str, Any]], key: str) -> Optional[tuple[float, float]]:
    """``(lo, hi)`` of *key* over frames, or ``None`` unless all measured it."""
    values = [f.get(key) for f in frames]
    if any(not _finite(v) for v in values):
        return None
    return (min(values), max(values))  # type: ignore[type-var]


def _read_part_provenance(value: Any, *, root_kind: Any) -> Optional[dict[str, Any]]:
    """Aggregate component fits without mistaking nested parts for frames."""
    if not isinstance(value, list) or not value:
        return None
    fittings: list[dict[str, Any]] = []
    quotable = True
    for part in value:
        if not isinstance(part, dict) or not isinstance(part.get("fitting"), dict):
            return None
        fittings.append(part["fitting"])
        reference = part.get("fit_reference")
        quotable &= (
            isinstance(reference, dict) and reference.get("kind") == "acquisition"
        )
    nested = any(
        isinstance(fitting.get("part_provenance"), list) for fitting in fittings
    )
    return {
        "frames": None if root_kind == "partition" or nested else len(value),
        "quality_quotable": quotable,
        "source_bytes": (
            None if root_kind == "partition" else _total(fittings, "source_bytes")
        ),
        "psnr_db": _span(fittings, "psnr_db") if quotable else None,
        "foreground_psnr_db": (
            _span(fittings, "foreground_psnr_db") if quotable else None
        ),
    }


def _immediate_children(groups: set[str], path: str) -> list[str]:
    """Group paths exactly one level below *path* (``""`` for the root)."""
    depth = 0 if not path else path.count("/") + 1
    prefix = (path + "/") if path else ""
    return sorted(
        g for g in groups if g and g.startswith(prefix) and g.count("/") == depth
    )


def _finest_elements(
    zf: zipfile.ZipFile, root: str, groups: set[str], path: str = ""
) -> Optional[int]:
    """Element count of the FINEST representation, walking the tree's semantics.

    A tree's root carries no ``n_splats`` — the counts live on the groups — and
    the three group families combine differently, so a naive sum is wrong by a
    lot: over ct_atlas it gives 2,574,354 against a true 647,083.

    * ``part_N``     disjoint spatial tiles      -> SUM
    * ``child_N``    substitutive LOD levels     -> MAX (they REPLACE each other)
    * ``additive_N`` disjoint streaming chunks   -> SUM to their own parent, so
                                                   the parent's stamp wins

    Verified against three archives whose counts were established independently:
    ct_atlas 647,083, cmu1_ch0 8,823,953, dapi 7,740.
    """
    attrs = _attrs(zf, root, (path + "/") if path else "")
    kids = _immediate_children(groups, path)
    kind = attrs.get("kind")
    if kind == "partition":
        vals = [
            _finest_elements(zf, root, groups, k)
            for k in kids
            if k.rsplit("/", 1)[-1].startswith("part_")
        ]
        return sum(vals) if vals and all(v is not None for v in vals) else None
    if kind == "lod":
        vals = [
            _finest_elements(zf, root, groups, k)
            for k in kids
            if k.rsplit("/", 1)[-1].startswith("child_")
        ]
        return max((v for v in vals if v is not None), default=None)
    own = _as_int(attrs.get("n_splats"))
    if own is not None:
        return own
    vals = [
        _as_int(_attrs(zf, root, k + "/").get("n_splats"))
        for k in kids
        if k.rsplit("/", 1)[-1].startswith("additive_")
    ]
    return sum(vals) if vals and all(v is not None for v in vals) else None


def _store_root(zf: zipfile.ZipFile, names: list[str]) -> Optional[tuple[str, dict]]:
    """Find the candidate root whose attributes declare a gsplats store.

    Root-level stores and conventional ``*.gsplats.zarr/`` wrappers are common,
    but valid archives can use another wrapper name or include unrelated
    top-level members. Inspect the zip root and every top-level directory rather
    than treating the first member's prefix as authoritative.
    """
    candidates = [""]
    candidates += sorted({n.split("/")[0] + "/" for n in names if "/" in n})
    for root in candidates:
        attrs = _attrs(zf, root)
        self_identified = attrs.get("format_type") == "gsplats_zarr"
        conventional_root = not root or root.rstrip("/").endswith(".gsplats.zarr")
        if self_identified or conventional_root and attrs.get("type") == "gsplats":
            return root, attrs
    return None


def _read_store(zf: zipfile.ZipFile) -> Optional[dict[str, Any]]:
    names = zf.namelist()
    if not names:
        return None
    located = _store_root(zf, names)
    if located is None:
        return None
    root, root_attrs = located
    # The format check that used to live here is now `_store_root`'s search
    # condition: an .npz is also a zip and a point-cloud .luxar.zarr is also a
    # zarr store, and on either of those no candidate root declares the gsplats
    # format, so `_store_root` returns None above. Both would otherwise parse
    # "successfully" with every field defaulted, which reads as a claim ("single
    # level", no PSNR) rather than as "this is not a splat fit".
    fit = _attrs(zf, root, "fitting/")
    part_info = _read_part_provenance(
        fit.get("part_provenance"), root_kind=root_attrs.get("kind")
    )
    n_splats = _as_int(root_attrs.get("n_splats"))
    groups = {
        n[len(root) :].rsplit("/", 1)[0]
        for n in names
        if n.endswith((("/.zgroup"), "/zarr.json"))
    }
    return {
        # A tree root has no count of its own; derive it from the groups rather
        # than publishing a dash for an archive that plainly knows its size.
        "n_splats": n_splats
        if n_splats is not None
        else _finest_elements(zf, root, groups),
        "ndim": root_attrs.get("ndim"),
        "format_version": root_attrs.get("format_version"),
        "topology": _describe_topology(root_attrs, groups),
        "psnr_db": fit.get("psnr_db")
        if fit.get("psnr_db") is not None
        else (part_info or {}).get("psnr_db"),
        "foreground_psnr_db": fit.get("foreground_psnr_db")
        if fit.get("foreground_psnr_db") is not None
        else (part_info or {}).get("foreground_psnr_db"),
        "foreground_fraction": fit.get("foreground_fraction"),
        "source_shape": fit.get("source_shape"),
        "source_dtype": fit.get("source_dtype"),
        "source_bytes": fit.get("source_bytes")
        if fit.get("source_bytes") is not None
        else (part_info or {}).get("source_bytes"),
        "frames": (part_info or {}).get("frames"),
        "quality_quotable": (part_info or {}).get("quality_quotable"),
    }


def _files_of(entry: dict[str, Any]) -> list[tuple[str, dict[str, Any]]]:
    """``(variant, file spec)`` per declared file; variant ``""`` when there are none.

    The variant name has to travel with the file: a variant's archives sit one
    level deeper on disk, exactly as ``ensure_dataset`` resolves them. Reading
    only ``entry["files"]`` makes a variant-only dataset -- h2afva, whose record
    is the timelapse -- render as "no files uploaded yet" with its whole file
    table missing.
    """
    files = entry.get("files") or []
    if files:
        return [("", spec) for spec in files]
    return [
        (name, spec)
        for name, variant in (entry.get("variants") or {}).items()
        for spec in (variant.get("files") or [])
    ]


def _locate(
    dataset: str,
    entry: dict[str, Any],
    variant: str,
    file_name: str,
    extra_root: Optional[Path] = None,
) -> Iterator[Path]:
    """Yield archives from *extra_root*, the repo copy, then the local cache.

    The roots namespace differently, as ``ensure_dataset`` does: in-repo by the
    manifest ``dir``, the cache by the DATASET NAME, and both by the variant.

    *extra_root* is searched FIRST and is namespaced like the cache. It exists
    because the repo and cache copies are the PRE-REFIT generation for every
    dataset the refit campaign touched — measuring those gives figures that are
    stale or absent (8 of 27 local archives carry a PSNR, none carry a foreground
    PSNR) while the artifacts the records actually serve are fully stamped. Point
    it at the staging tree holding the uploaded generation.
    """
    roots = [(extra_root, dataset)] if extra_root else []
    roots += [(DATA_DIR, entry.get("dir", dataset)), (CACHE_DIR, dataset)]
    for base, subdir in roots:
        parts = [p for p in (subdir, variant, file_name) if p]
        candidate = base.joinpath(*parts)
        if candidate.exists():
            yield candidate


# ---------------------------------------------------------------------------
# Measured characteristics, committed
#
# Reading the archives directly was the original design and it does not survive
# contact with the migration. The figures come out of each archive's `fitting/`
# stamps, so they can only be read where the bytes are — and the bytes are
# leaving: `demos/data/` is being emptied onto Zenodo. On a machine holding a
# partial set the generator reported "no PSNR" for archives whose HOSTED copies
# are stamped (ct_atlas 43.0/30.6 dB among them), which is the failure this whole
# tool exists to prevent, in its own output.
#
# So measurement is separated from rendering. `--refresh` reads whatever archives
# are present and records what it measured; rendering reads the committed record.
# It is small text, so it survives payload removal, needs no network and no LFS
# content, and the next refit re-stamps it as part of the upload step.
# ---------------------------------------------------------------------------


def _char_key(dataset: str, variant: str, file_name: str) -> str:
    """Stable identity for one archive: ``dataset[/variant]/file``."""
    return "/".join(p for p in (dataset, variant, file_name) if p)


def load_characteristics() -> dict[str, Any]:
    """The committed measurements, or an empty map when none exist yet."""
    if not CHARACTERISTICS.exists():
        return {}
    payload = json.loads(CHARACTERISTICS.read_text())
    if not isinstance(payload, dict) or payload.get("schema_version") != 1:
        raise ValueError(f"unsupported characteristics schema in {CHARACTERISTICS}")
    archives = payload.get("archives", {})
    if not isinstance(archives, dict):
        raise ValueError(f"invalid archives map in {CHARACTERISTICS}")
    return {key: value for key, value in archives.items() if isinstance(value, dict)}


def hosted_size(spec: dict[str, Any]) -> Optional[int]:
    """The size of the copy the RECORD serves, not the one this repo ships.

    ``bytes`` describes the in-repo copy and ``hosted_bytes`` the hosted one; they
    diverge for every refitted dataset. A record's own table must quote the size
    of the file a reader will download, so the hosted value wins where it exists.
    """
    return spec.get("hosted_bytes") or spec.get("bytes")


def _pinned_digest(spec: dict[str, Any]) -> Optional[str]:
    """The digest the manifest expects for the copy a record serves.

    Prefers ``hosted_sha256`` where the hosted artifact and the in-repo copy have
    diverged; falls back to ``sha256``, which described both before they did.
    """
    return spec.get("hosted_sha256") or spec.get("sha256")


def _sha256_of(path: Path) -> str:
    import hashlib

    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _select_pinned_location(
    candidates: Iterator[Path], pinned_digest: Optional[str]
) -> tuple[Optional[Path], Optional[str]]:
    """Prefer pinned bytes, falling back to the first existing candidate."""
    fallback: tuple[Optional[Path], Optional[str]] = (None, None)
    for path in candidates:
        digest = _sha256_of(path)
        if fallback[0] is None:
            fallback = (path, digest)
        if digest == pinned_digest:
            return path, digest
    return fallback


def _retain_preferred_measurements(
    measured: dict[str, Any],
    existing: dict[str, Any],
    pinned_digests: dict[str, Optional[str]],
) -> tuple[int, int]:
    """Blank unpinned reads and retain stronger committed measurements.

    Staged provenance outranks repo/cache even for identical bytes, avoiding
    sidecar churn when a later local refresh sees the same pinned archive.
    Source-derived figures survive a stampless pinned read, while the read still
    supplies archive metadata and provenance that the committed row lacks.
    """
    rank = {"staged": 2, "repo": 1, "cache": 1}
    measurement_fields = (
        "n_splats",
        "ndim",
        "format_version",
        "topology",
        "psnr_db",
        "foreground_psnr_db",
        "foreground_fraction",
        "source_shape",
        "source_dtype",
        "source_bytes",
        "frames",
    )
    recovered_fields = (
        "psnr_db",
        "foreground_psnr_db",
        "foreground_fraction",
        "source_shape",
        "source_dtype",
        "source_bytes",
        "frames",
    )
    retained = 0
    rejected = 0
    for key, new_entry in list(measured.items()):
        old_entry = existing.get(key)
        pinned_digest = pinned_digests.get(key)
        if new_entry.get("measured_sha256") != pinned_digest:
            rejected += 1
            if old_entry is None:
                measured[key] = {
                    "n_splats": None,
                    "ndim": None,
                    "format_version": None,
                    "topology": None,
                    "psnr_db": None,
                    "foreground_psnr_db": None,
                    "foreground_fraction": None,
                    "source_shape": None,
                    "source_dtype": None,
                    "source_bytes": None,
                    "frames": None,
                    "measured_from": None,
                    "measured_sha256": None,
                    "unmeasured_reason": "unpinned-local-copy",
                }
            else:
                measured[key] = (
                    old_entry
                    if any(
                        old_entry.get(field) is not None for field in measurement_fields
                    )
                    else {
                        **old_entry,
                        "unmeasured_reason": "unpinned-local-copy",
                    }
                )
        elif old_entry is not None and old_entry.get("measured_sha256") is None:
            recovered = {
                field: old_entry[field]
                for field in recovered_fields
                if old_entry.get(field) is not None and new_entry.get(field) is None
            }
            if recovered:
                measured[key] = {**new_entry, **recovered}
                retained += 1
        elif (
            old_entry is not None
            and old_entry.get("measured_sha256") == pinned_digest
            and rank.get(new_entry.get("measured_from"), 0)
            < rank.get(old_entry.get("measured_from"), 0)
        ):
            measured[key] = old_entry
            retained += 1
    return retained, rejected


def refresh_characteristics(
    manifest: dict[str, Any], extra_root: Optional[Path] = None
) -> tuple[int, int, int, int]:
    """Re-measure archives; returns (read, retained, rejected, preserved).

    PRESERVES entries whose archive is not on this machine, for the same reason
    ``gen_data_manifest`` preserves committed file lists: a refresh run from a
    partial checkout would otherwise silently delete the measurements for every
    dataset it cannot see, and a partial checkout is the normal case now.
    """
    existing = load_characteristics()
    measured: dict[str, Any] = {}
    pinned_digests: dict[str, Optional[str]] = {}
    seen: set[str] = set()
    for dataset, entry in sorted(manifest["datasets"].items()):
        if entry.get("bucket") != "zenodo":
            continue
        for variant, spec in _files_of(entry):
            key = _char_key(dataset, variant, spec["name"])
            seen.add(key)
            pinned_digests[key] = _pinned_digest(spec)
            path, measured_sha256 = _select_pinned_location(
                _locate(dataset, entry, variant, spec["name"], extra_root),
                pinned_digests[key],
            )
            info = _read_archive(path) if path else None
            if info is None:
                continue
            if extra_root and path.is_relative_to(extra_root):
                root = "staged"
            elif path.is_relative_to(DATA_DIR):
                root = "repo"
            else:
                root = "cache"
            measured[key] = {
                **info,
                # Which copy was read, and what it hashed to. Without the digest a
                # stale measurement is indistinguishable from a current one, which
                # is precisely the state the hand-edited descriptions were in.
                "measured_from": root,
                "measured_sha256": measured_sha256,
                **{
                    field: existing[key][field]
                    for field in ("quality_note", "quality_caveat")
                    if key in existing and field in existing[key]
                },
            }

    read = len(measured)
    retained, rejected = _retain_preferred_measurements(
        measured, existing, pinned_digests
    )
    preserved = {k: v for k, v in existing.items() if k in seen and k not in measured}
    archives = dict(sorted({**preserved, **measured}.items()))
    CHARACTERISTICS.write_text(
        json.dumps(
            {
                "schema_version": 1,
                "description": (
                    "Measured characteristics of the demo gsplat archives, read "
                    "from each archive's own fitting/ stamps by "
                    "scripts/gen_zenodo_records.py --refresh. Committed so the "
                    "record text does not require holding the archives, which are "
                    "hosted on Zenodo rather than in this repository. "
                    "measured_sha256 records WHICH archive bytes refresh read; "
                    "--check reports any non-null digest that no longer matches "
                    "the manifest pin. A null digest means the figures were "
                    "recovered from the stated source rather than archive bytes; "
                    "a later pinned archive read may supply the digest while "
                    "retaining those source-derived figures. "
                    "quality_note is internal provenance; quality_caveat is "
                    "published beside an absent figure. unmeasured_reason records "
                    "why refresh deliberately withheld measurements."
                ),
                "archives": archives,
            },
            indent=2,
        )
        + "\n"
    )
    return read, retained, rejected, len(preserved)


def _stale_characteristics(manifest: dict[str, Any]) -> list[str]:
    """Measurements taken from bytes the manifest no longer pins.

    This is the drift guard the hand-written descriptions never had: a figure is
    only trustworthy if it was measured from the artifact the record actually
    serves, and a refit changes that artifact without touching this file.
    """
    chars = load_characteristics()
    stale = []
    for dataset, entry in sorted(manifest["datasets"].items()):
        if entry.get("bucket") != "zenodo":
            continue
        for variant, spec in _files_of(entry):
            key = _char_key(dataset, variant, spec["name"])
            got = chars.get(key)
            if not got:
                continue
            measured, pinned = got.get("measured_sha256"), _pinned_digest(spec)
            if measured and pinned and measured != pinned:
                stale.append(
                    f"  {key}: measured from {measured[:12]}… but the manifest "
                    f"pins {pinned[:12]}… — re-run --refresh against the "
                    "current archive"
                )
    return stale


# ---------------------------------------------------------------------------
# Formatting
# ---------------------------------------------------------------------------


def _mib(n: Optional[int]) -> str:
    if not n:
        return _ABSENT
    for unit, size in (("GiB", 1 << 30), ("MiB", 1 << 20), ("KiB", 1 << 10)):
        if n >= size:
            return f"{n / size:.1f} {unit}"
    return f"{n} B"


def _ratio(numerator: Optional[int], denominator: Optional[int]) -> str:
    if (
        not (_finite(numerator) and _finite(denominator))
        or not numerator
        or not denominator
    ):
        return _ABSENT
    return f"{numerator / denominator:.0f}:1"


def _db(value: Any) -> str:
    """A dB figure, a ``lo–hi`` range for a per-frame one, or absent."""
    if isinstance(value, (list, tuple)) and len(value) == 2:
        lo, hi = value
        if not (_finite(lo) and _finite(hi)):
            return _ABSENT
        return f"{lo:.1f}" if f"{lo:.1f}" == f"{hi:.1f}" else f"{lo:.1f}–{hi:.1f}"
    if not _finite(value):
        return _ABSENT
    return f"{value:.1f}"


def _dataset_rows(
    dataset: str, entry: dict[str, Any], chars: Optional[dict[str, Any]] = None
) -> list[dict[str, Any]]:
    """Rows for one dataset, from the committed measurements.

    The committed record WINS over a local archive, deliberately: a record
    describes the artifact it serves, and the local copy may be a pre-refit
    generation or a local scratch fit. Reading an archive is the fallback for
    something not measured yet, so a fresh dataset still renders before its first
    ``--refresh``.
    """
    chars = load_characteristics() if chars is None else chars
    rows = []
    for variant, spec in _files_of(entry):
        info = chars.get(_char_key(dataset, variant, spec["name"]))
        # An entry that `refresh` wrote is authoritative INCLUDING its nulls: an
        # all-null one is its verdict that the local copy's bytes are not the
        # pinned artifact, so reading that copy would publish figures describing
        # a generation the record does not serve. Such an entry always carries
        # `measured_sha256`, which is what distinguishes it from a hand-written
        # note. A note is prose about an archive, not a finding about its bytes,
        # so the archive still supplies whatever the note does not state --
        # otherwise documenting why one figure is missing silently deletes the
        # rest, and the only way to explain a gap is to widen it.
        if info is not None and "measured_sha256" not in info:
            pinned = _pinned_digest(spec)
            path, digest = _select_pinned_location(
                _locate(dataset, entry, variant, spec["name"]),
                pinned,
            )
            if digest != pinned:
                path = None
            read = _read_archive(path) if path else None
            if read is not None:
                info = {**read, **{k: v for k, v in info.items() if v is not None}}
        elif info is None:
            path = next(_locate(dataset, entry, variant, spec["name"]), None)
            info = _read_archive(path) if path else None
        stored = hosted_size(spec)
        name = f"{variant}/{spec['name']}" if variant else spec["name"]
        if info and info.get("frames"):
            name += f" ({info['frames']} frames)"
        rows.append(
            {
                "is_gsplat": info is not None,
                # A declared fit whose bytes are not on THIS machine (an unpulled
                # LFS file, an archive not fetched yet) must not be mistaken for
                # "this is not a fit": it belongs in the splat table with its
                # figures absent, and `--check` has to say it went unexamined.
                "is_fit": spec["name"].endswith(".gsplats.zarr.zip"),
                "file": name,
                "splats": f"{info['n_splats']:,}"
                if info and isinstance(info.get("n_splats"), int)
                else _ABSENT,
                "size": _mib(stored),
                "topology": info.get("topology") or _ABSENT if info else _ABSENT,
                "psnr": _db(info.get("psnr_db")) if info else _ABSENT,
                "fg_psnr": _db(info.get("foreground_psnr_db")) if info else _ABSENT,
                "vs_raw": _ratio(info.get("source_bytes"), stored) if info else _ABSENT,
                # Lets `--check` look the row's own sidecar entry back up, so it
                # can distinguish an unmeasured archive from one deliberately
                # left unmeasured because the local bytes are not the pinned ones.
                "char_key": _char_key(dataset, variant, spec["name"]),
                # A published, reader-facing reason for an absent figure. Kept
                # separate from `quality_note`, which is internal provenance and
                # is never rendered -- see the module docstring.
                "caveat": (info or {}).get("quality_caveat"),
                "quality_quotable": info.get("quality_quotable") if info else None,
            }
        )
    return rows


def _acquisition_line(entry: dict[str, Any], total_stored: Optional[int]) -> str:
    acq = entry.get("acquisition")
    if not acq:
        return ""
    description = acq.get("description", "")
    if not acq.get("comparable", False):
        return (
            f"- Fitted from {description}. No compression ratio against the stored "
            f"source is quoted: {acq.get('reason', 'not comparable')}.\n"
        )
    stored = acq.get("stored_bytes")
    if not stored:
        return f"- Fitted from {description}.\n"
    if not total_stored:
        # No single total to divide by: size VARIANTS are alternative downloads
        # of the same data, so summing them would price the same frames twice.
        return f"- Fitted from {description} ({_mib(stored)} stored).\n"
    return (
        f"- Fitted from {description} ({_mib(stored)} stored). "
        f"All files here total {_mib(total_stored)}, i.e. "
        f"**{_ratio(stored, total_stored)} against the source as downloaded**.\n"
    )


def render_record(key: str, manifest: dict[str, Any]) -> str:
    record = manifest["records"][key]
    datasets = {
        name: entry
        for name, entry in manifest["datasets"].items()
        if entry.get("record") == key and entry.get("bucket") == "zenodo"
    }
    out: list[str] = []
    out.append(f"# {record['title']}\n")
    if not record.get("published", False):
        out.append(
            "<!-- DRAFT. This record is unpublished; publication is a manual "
            "step taken by the maintainer. -->\n"
        )
    out.append(
        f"\nLicence: **{record['license']}** · Reserved DOI: `{record['zenodo_doi']}`\n"
    )
    out.append(
        "\nGaussian-splat and point-cloud scenes for the "
        "[Luxar](https://github.com/royerlab/luxar) viewer. Each archive is a "
        "fitted representation of a public dataset, published so that a scene "
        "can be opened in seconds rather than refitted on a GPU. Luxar fetches "
        "these on demand — `luxar demo run <name>` — and verifies every file "
        "against a recorded checksum.\n"
    )
    out.append(
        "\nThe licence above is the record's single field; each dataset's own "
        "terms are listed below and may be more permissive (several are public "
        "domain or CC0).\n"
    )
    chars = load_characteristics()

    for name, entry in sorted(datasets.items()):
        rows = _dataset_rows(name, entry, chars)
        variants = entry.get("variants") or {}
        total = (
            None
            if variants
            else sum(hosted_size(f) or 0 for f in entry.get("files", []))
        )
        out.append(f"\n## `{name}`\n")
        out.append(f"\n{entry.get('source', '')}\n")
        out.append(f"\n- Licence: **{entry.get('license', _ABSENT)}**\n")
        out.append(f"- Attribution: {entry.get('attribution', _ABSENT)}\n")
        out.append(_acquisition_line(entry, total))
        for vname, variant in variants.items():
            default = " (default)" if variant.get("default") else ""
            out.append(f"- Variant `{vname}`{default}: {variant.get('note', '')}\n")
        if not rows:
            out.append("\n_No files uploaded yet._\n")
            continue
        # Only a splat fit has splats, levels and a reconstruction quality. The
        # point-cloud and tabular datasets get a plain file list; a table of
        # dashes would imply those figures exist and were merely not measured.
        # A fit whose archive is not readable here still belongs in the splat
        # table -- demoting it to the plain list would state the opposite.
        if any(row["is_gsplat"] or row["is_fit"] for row in rows):
            out.append(
                "\n| File | Splats | Size | Detail levels | PSNR (dB) | "
                "Foreground PSNR (dB) | vs raw voxels |\n"
                "|---|---:|---:|---|---:|---:|---:|\n"
            )
            for row in rows:
                out.append(
                    f"| `{row['file']}` | {row['splats']} | {row['size']} | "
                    f"{row['topology']} | {row['psnr']} | {row['fg_psnr']} | "
                    f"{row['vs_raw']} |\n"
                )
            # An em dash in a quality column means "not stated", which a reader
            # cannot distinguish from "not measurable" or from evasion. Where the
            # sidecar records a reason meant for publication, say it here rather
            # than leaving the blank to speak for itself.
            for row in rows:
                if row.get("caveat"):
                    out.append(f"\n- `{row['file']}`: {row['caveat']}\n")
        else:
            out.append("\n| File | Size |\n|---|---:|\n")
            for row in rows:
                out.append(f"| `{row['file']}` | {row['size']} |\n")

    out.append(
        "\n---\n\n**Reading the quality columns.** PSNR is measured over the "
        "whole volume and foreground PSNR only over voxels above the source's "
        "Otsu threshold. On sparse data — most light-sheet and tomography — the "
        "global figure is largely a score for reproducing empty space, and the "
        "foreground column is the one that says whether the signal survived the "
        "fit. Both are reported; neither alone is the answer.\n"
    )
    out.append(
        "\n**Reading the compression column.** `vs raw voxels` compares the "
        "archive against the source grid held as uncompressed samples. Where a "
        "dataset's stored source covers the same data as its archives, the "
        "ratio against that download is given above the table instead — a "
        "smaller and more honest number, since the source is itself compressed.\n"
    )
    return "".join(out)


def _why_absent(row: dict[str, Any], chars: dict[str, Any]) -> str:
    """Name the cause when the sidecar already records one.

    ``refresh`` stamps ``unmeasured_reason`` when it rejects a local copy whose
    bytes are NOT the pinned artifact and no committed measurements can be
    retained. The figures are absent on purpose, because measuring that copy
    would describe a generation the record does not serve. That is a different
    job from an unmeasured archive -- fetch the hosted copy, rather than go and
    measure -- and printing the two identically invites someone to "fix" the
    generator into publishing the very numbers the marker withholds.
    """
    info = chars.get(row.get("char_key", ""))
    if not info or info.get("unmeasured_reason") != "unpinned-local-copy":
        return ""
    return (
        " (a local copy is present but its bytes are not the pinned artifact, "
        "so it was deliberately not measured; fetch the hosted copy)"
    )


def _gaps(manifest: dict[str, Any]) -> tuple[list[str], list[str]]:
    """Figures a record would print as absent, and the rows nothing was read for."""
    problems: list[str] = []
    unread: list[str] = []
    chars = load_characteristics()
    for name, entry in sorted(manifest["datasets"].items()):
        if entry.get("bucket") != "zenodo":
            continue
        for row in _dataset_rows(name, entry, chars):
            # Companion sidecars (label maps, colour arrays) and the tabular /
            # point-cloud datasets are not fits, so they owe no splat count or
            # reconstruction quality. Demanding one would make this list
            # permanently non-empty and therefore useless as a work list.
            if not row["is_gsplat"]:
                # A declared fit that is simply not on this machine was never
                # examined. Dropping it silently would let the count below read
                # as completeness while a dozen rows went unchecked.
                if row["is_fit"]:
                    unread.append(f"{name}/{row['file']}")
                continue
            expected = [
                ("splats", row["splats"]),
                ("compression", row["vs_raw"]),
            ]
            if row["quality_quotable"] is not False:
                expected[1:1] = [
                    ("PSNR", row["psnr"]),
                    ("foreground PSNR", row["fg_psnr"]),
                ]
            missing = [label for label, value in expected if value == _ABSENT]
            if missing:
                problems.append(
                    f"{name}/{row['file']}: no {', '.join(missing)}"
                    + _why_absent(row, chars)
                )
        if not _files_of(entry):
            problems.append(f"{name}: no files uploaded")
    return problems, unread


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--record", help="only this record key (cc-by, cc-by-sa, h2afva)")
    ap.add_argument("--outdir", type=Path, help="write <key>.md files here")
    ap.add_argument(
        "--check",
        action="store_true",
        help="list characteristics that are not yet stamped, and exit 1 if any",
    )
    ap.add_argument(
        "--archives-root",
        type=Path,
        help="with --refresh, search this tree BEFORE the repo copy and the cache "
        "(namespaced <dataset>/[<variant>/]<file>). Use it to measure the "
        "uploaded generation rather than the pre-refit copies on this machine.",
    )
    ap.add_argument(
        "--refresh",
        action="store_true",
        help="re-measure every archive present on this machine and rewrite "
        "scripts/demo_archive_characteristics.json (preserves entries whose "
        "archive is absent here)",
    )
    args = ap.parse_args()

    manifest = json.loads(MANIFEST.read_text())
    if args.refresh:
        read, retained, rejected, preserved = refresh_characteristics(
            manifest, args.archives_root
        )
        print(
            f"read {read} archive(s) here, skipped {rejected} read(s) taken from "
            f"bytes the manifest does not pin, kept {retained} committed "
            f"measurement(s) that outrank the local copy, preserved {preserved} "
            f"not on this machine -> {CHARACTERISTICS.relative_to(REPO_ROOT)}"
        )
        return 0
    if args.check:
        return _run_check(manifest)
    return _run_render(manifest, args)


def _run_check(manifest: dict[str, Any]) -> int:
    """``--check``: report figures a record would print as absent, or as stale."""
    problems, unread = _gaps(manifest)
    for problem in problems:
        print(problem)
    print(f"\n{len(problems)} archive(s) would publish an incomplete row.")
    stale = _stale_characteristics(manifest)
    if stale:
        # A stale figure is worse than an absent one: absent prints as "—",
        # stale prints as a number that is simply wrong.
        print(
            f"\n{len(stale)} measurement(s) were taken from bytes the "
            "manifest no longer pins:"
        )
        for line in stale:
            print(line)
    if unread:
        # Saying so is the point: without it the count above reads as a
        # clean bill of health on a machine that holds none of the data.
        print(
            f"{len(unread)} declared archive(s) are not on this machine, so "
            "nothing was read for them (`git lfs pull`, or fetch the demo):"
        )
        for item in unread:
            print(f"  {item}")
    return 1 if problems or stale else 0


def _run_render(manifest: dict[str, Any], args: argparse.Namespace) -> int:
    """Render one record or all of them, to stdout or to ``--outdir``."""
    keys = [args.record] if args.record else list(manifest["records"])
    for key in keys:
        if key not in manifest["records"]:
            print(f"unknown record {key!r}", file=sys.stderr)
            return 2
        text = render_record(key, manifest)
        if args.outdir:
            args.outdir.mkdir(parents=True, exist_ok=True)
            (args.outdir / f"{key}.md").write_text(text)
            print(f"wrote {args.outdir / f'{key}.md'}")
        else:
            print(text)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
