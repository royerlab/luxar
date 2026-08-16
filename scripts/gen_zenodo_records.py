#!/usr/bin/env python3
"""Generate the Zenodo record descriptions from the manifest and the archives.

The three demo records describe several dozen files between them, each with a
splat count, a compression figure, a reconstruction quality and a licence. Typed
by hand that drifts from the data within one refit, and a record is the one place
where a stale number is published rather than merely wrong.

So the text is derived: dataset disposition and licensing come from
``data_manifest.json``, and the per-dataset characteristics are read out of the
archives' own ``fitting/`` stamps. A figure that is not stamped is reported as
absent rather than guessed at.

This script NEVER talks to Zenodo. It writes markdown for a human to paste into
a draft, and publication stays a manual act.

    python scripts/gen_zenodo_records.py                 # all records to stdout
    python scripts/gen_zenodo_records.py --record cc-by  # just one
    python scripts/gen_zenodo_records.py --outdir docs/zenodo/
    python scripts/gen_zenodo_records.py --check         # report gaps, exit 1
"""

from __future__ import annotations

import argparse
import io
import json
import sys
import zipfile
from pathlib import Path
from typing import Any, Optional

REPO_ROOT = Path(__file__).resolve().parent.parent
MANIFEST = REPO_ROOT / "packages/luxar/src/luxar/demos/data_manifest.json"
DATA_DIR = REPO_ROOT / "packages/luxar/src/luxar/demos/data"
CACHE_DIR = Path.home() / ".cache/luxar"

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
        attrs = raw.get("attributes", raw) if candidate.endswith("zarr.json") else raw
        return attrs if isinstance(attrs, dict) else {}
    return {}


def _describe_topology(root_attrs: dict[str, Any], groups: set[str]) -> str:
    """Name the LOD topology from what is actually on disk.

    Reads the structural ``kind`` the writer stamped rather than the ``recipe``
    provenance, because ``kind`` is what the viewer acts on -- a mismatch
    between the two is exactly the sort of thing worth seeing in the record.
    """
    kind = root_attrs.get("kind")
    rungs = root_attrs.get("n_additive_sublods")
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
    if rungs and int(rungs) > 1:
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
            inner = [n for n in names if n.endswith(".gsplats.zarr.zip")]
            if inner:
                with zipfile.ZipFile(io.BytesIO(zf.read(inner[0]))) as frame:
                    info = _read_store(frame)
                if info is not None:
                    info["frames"] = len(inner)
                return info
            return _read_store(zf)
    except (zipfile.BadZipFile, OSError):
        return None


def _read_store(zf: zipfile.ZipFile) -> Optional[dict[str, Any]]:
    names = zf.namelist()
    if not names:
        return None
    root = names[0].split("/")[0] + "/"
    root_attrs = _attrs(zf, root)
    # An .npz is also a zip, and a point-cloud .luxar.zarr is also a zarr store.
    # Without this check both parse "successfully" and every field comes back
    # defaulted -- which reads as a claim ("single level", no PSNR) rather than
    # as "this is not a splat fit". Publishing that would be publishing a
    # falsehood, so identify the format before describing it.
    if root_attrs.get("format_type") != "gsplats_zarr" and (
        root_attrs.get("type") != "gsplats"
    ):
        return None
    fit = _attrs(zf, root, "fitting/")
    groups = {
        n[len(root) :].rsplit("/", 1)[0]
        for n in names
        if n.endswith((("/.zgroup"), "/zarr.json"))
    }
    return {
        "n_splats": root_attrs.get("n_splats"),
        "ndim": root_attrs.get("ndim"),
        "format_version": root_attrs.get("format_version"),
        "topology": _describe_topology(root_attrs, groups),
        "psnr_db": fit.get("psnr_db"),
        "foreground_psnr_db": fit.get("foreground_psnr_db"),
        "foreground_fraction": fit.get("foreground_fraction"),
        "source_shape": fit.get("source_shape"),
        "source_dtype": fit.get("source_dtype"),
        "source_bytes": fit.get("source_bytes"),
        "frames": None,
    }


def _locate(dataset: str, entry: dict[str, Any], file_name: str) -> Optional[Path]:
    """Find an archive in the repo copy or the local cache."""
    subdir = entry.get("dir", dataset)
    for base in (DATA_DIR, CACHE_DIR):
        candidate = (base / subdir / file_name) if subdir else (base / file_name)
        if candidate.exists():
            return candidate
    return None


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
    if not numerator or not denominator:
        return _ABSENT
    return f"{numerator / denominator:.0f}:1"


def _db(value: Any) -> str:
    if not isinstance(value, (int, float)):
        return _ABSENT
    if value != value or value in (float("inf"), float("-inf")):  # nan / inf
        return _ABSENT
    return f"{value:.1f}"


def _dataset_rows(dataset: str, entry: dict[str, Any]) -> list[dict[str, str]]:
    rows = []
    for spec in entry.get("files", []):
        path = _locate(dataset, entry, spec["name"])
        info = _read_archive(path) if path else None
        stored = spec.get("bytes")
        name = spec["name"]
        if info and info.get("frames"):
            name += f" ({info['frames']} frames)"
        rows.append(
            {
                "is_gsplat": info is not None,
                "file": name,
                "splats": f"{info['n_splats']:,}"
                if info and isinstance(info.get("n_splats"), int)
                else _ABSENT,
                "size": _mib(stored),
                "topology": info["topology"] if info else _ABSENT,
                "psnr": _db(info.get("psnr_db")) if info else _ABSENT,
                "fg_psnr": _db(info.get("foreground_psnr_db")) if info else _ABSENT,
                "vs_raw": _ratio(info.get("source_bytes"), stored) if info else _ABSENT,
            }
        )
    return rows


def _acquisition_line(entry: dict[str, Any], total_stored: int) -> str:
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

    for name, entry in sorted(datasets.items()):
        rows = _dataset_rows(name, entry)
        total = sum(f.get("bytes", 0) for f in entry.get("files", []))
        out.append(f"\n## `{name}`\n")
        out.append(f"\n{entry.get('source', '')}\n")
        out.append(f"\n- Licence: **{entry.get('license', _ABSENT)}**\n")
        out.append(f"- Attribution: {entry.get('attribution', _ABSENT)}\n")
        out.append(_acquisition_line(entry, total))
        if not rows:
            out.append("\n_No files uploaded yet._\n")
            continue
        # Only a splat fit has splats, levels and a reconstruction quality. The
        # point-cloud and tabular datasets get a plain file list; a table of
        # dashes would imply those figures exist and were merely not measured.
        if any(row["is_gsplat"] for row in rows):
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


def _gaps(manifest: dict[str, Any]) -> list[str]:
    """Figures a record would currently have to print as absent."""
    problems = []
    for name, entry in sorted(manifest["datasets"].items()):
        if entry.get("bucket") != "zenodo":
            continue
        for row in _dataset_rows(name, entry):
            # Companion sidecars (label maps, colour arrays) and the tabular /
            # point-cloud datasets are not fits, so they owe no splat count or
            # reconstruction quality. Demanding one would make this list
            # permanently non-empty and therefore useless as a work list.
            if not row["is_gsplat"]:
                continue
            missing = [
                label
                for label, value in (
                    ("splats", row["splats"]),
                    ("PSNR", row["psnr"]),
                    ("foreground PSNR", row["fg_psnr"]),
                    ("compression", row["vs_raw"]),
                )
                if value == _ABSENT
            ]
            if missing:
                problems.append(f"{name}/{row['file']}: no {', '.join(missing)}")
        if not entry.get("files"):
            problems.append(f"{name}: no files uploaded")
    return problems


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--record", help="only this record key (cc-by, cc-by-sa, h2afva)")
    ap.add_argument("--outdir", type=Path, help="write <key>.md files here")
    ap.add_argument(
        "--check",
        action="store_true",
        help="list characteristics that are not yet stamped, and exit 1 if any",
    )
    args = ap.parse_args()

    manifest = json.loads(MANIFEST.read_text())
    if args.check:
        problems = _gaps(manifest)
        for problem in problems:
            print(problem)
        print(f"\n{len(problems)} archive(s) would publish an incomplete row.")
        return 1 if problems else 0

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
