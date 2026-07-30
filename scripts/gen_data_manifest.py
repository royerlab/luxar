#!/usr/bin/env python3
"""Generate ``demos/data_manifest.json`` — the single source of truth for demo
dataset disposition (R17: retire git-LFS heavy datasets → Zenodo).

The manifest lives one level ABOVE ``demos/data/`` on purpose: everything under
that directory is excluded from the wheel and the sdist (it is ~450 MB of
git-LFS payload), so a manifest kept inside it would never reach an installed
user — the very audience fetch-on-demand exists for.

The manifest records, per dataset:
  * ``bucket``  — how the data is obtained:
      - ``zenodo``        : redistributable derived product; fetched from a Zenodo
                            record on first run (falls back to the in-repo LFS copy
                            until the Zenodo URL is populated).
      - ``local-compute`` : NOT redistributable — the demo fetches the raw source
                            and fits/builds locally, caching under ~/.cache/luxar.
      - ``regenerate``    : cheap to rebuild client-side (no GPU); not hosted.
  * ``license`` — the *true* per-dataset license (may be stricter/looser than the
                  Zenodo record's single license field; that is why records are
                  grouped by license family).
  * ``record``  — (zenodo only) which Zenodo record groups this dataset.
  * ``files``   — basenames + sha256 (git-LFS oid) + byte size, so a fetched copy
                  is checksum-verified.
  * ``dir``     — the in-repo subdir relative to ``demos/data/`` (empty string =
                  top level; the in-repo LFS fallback in data_fetch honours it).

Curated metadata (license/source/citation) lives here; checksums come from the data
tree itself, so the manifest stays reproducible on any checkout — see
:func:`_checksum` for why neither ``git`` nor the ``git-lfs`` binary is needed.
Re-run after any dataset re-encode:

    python scripts/gen_data_manifest.py            # write the manifest
    python scripts/gen_data_manifest.py --check     # verify it is up to date (CI)
    python scripts/gen_data_manifest.py --prune     # let an empty checkout win

A dataset the checkout cannot see keeps whatever the committed manifest says, so
regenerating from a partial checkout (or after R17 step 4 removes the payload from
the repo) does not wipe its checksums. ``--prune`` is the explicit opt-out.
"""

from __future__ import annotations

import argparse
import difflib
import hashlib
import json
import sys
from pathlib import Path
from typing import Optional

REPO_ROOT = Path(__file__).resolve().parent.parent
DATA_DIR = REPO_ROOT / "packages/luxar/src/luxar/demos/data"
MANIFEST = REPO_ROOT / "packages/luxar/src/luxar/demos/data_manifest.json"

# Zenodo records, grouped by license family. IDs/DOIs/base_url are null until the
# depositions are created and files uploaded (R17 step 2); once set, the fetch
# helper pulls from Zenodo instead of the in-repo LFS copy.
RECORDS = {
    "cc-by": {
        "title": "Luxar demo datasets (CC-BY / CC0 / public domain)",
        "license": "cc-by-4.0",
        "zenodo_concept_doi": None,
        "zenodo_record": None,
        "base_url": None,
    },
    "cc-by-sa": {
        "title": "Luxar demo datasets (CC-BY-SA)",
        "license": "cc-by-sa-4.0",
        "zenodo_concept_doi": None,
        "zenodo_record": None,
        "base_url": None,
    },
    "h2afva": {
        "title": "h2afva zebrafish histone light-sheet timelapse (Gaussian splats)",
        "license": "cc-by-4.0",
        "zenodo_concept_doi": None,
        "zenodo_record": None,
        "base_url": None,
    },
}

# Curated per-dataset metadata. `dir` is the demos/data subdir (or "" for
# top-level files). Files are filled in from the on-disk dir + LFS checksums.
DATASETS: dict[str, dict] = {
    # ---- Bucket 2: redistributable → Zenodo -------------------------------
    "gsplats_kidney": dict(
        bucket="zenodo",
        record="cc-by",
        license="cc0-1.0",
        source="scikit-image (data.kidney)",
        attribution="scikit-image sample data (CC0).",
    ),
    "gsplats_cells3d": dict(
        bucket="zenodo",
        record="cc-by",
        license="cc0-1.0",
        source="scikit-image (data.cells3d) — Allen Institute for Cell Science",
        attribution="scikit-image sample data.",
    ),
    "gsplats_cmu1_pathology": dict(
        bucket="zenodo",
        record="cc-by",
        license="cc0-1.0",
        source="OpenSlide test data (Aperio CMU-1.svs)",
        attribution="OpenSlide CMU-1 (CC0 1.0 public domain).",
    ),
    "gsplats_cryoem_virus": dict(
        bucket="zenodo",
        record="cc-by",
        license="cc0-1.0",
        source="EMDB EMD-5384",
        attribution="EMDB is public domain / CC0. Map: EMD-5384.",
    ),
    "gsplats_ct_totalsegmentator": dict(
        bucket="zenodo",
        record="cc-by",
        license="cc-by-4.0",
        source="TotalSegmentator (Wasserthal et al. 2023)",
        attribution="TotalSegmentator (Wasserthal et al. 2023; CC BY 4.0).",
    ),
    "gsplats_milkyway_dust": dict(
        bucket="zenodo",
        record="cc-by",
        license="cc-by-4.0",
        source="Zenodo 3993082 (3D dust map)",
        attribution="doi:10.5281/zenodo.3993082 (CC BY 4.0).",
    ),
    "desi_galaxies": dict(
        bucket="zenodo",
        record="cc-by",
        license="cc-by-4.0",
        source="DESI DR1 LSS catalogs",
        attribution="DESI DR1 (arXiv:2503.14745; CC BY 4.0).",
    ),
    "gsplats_celegans": dict(
        bucket="zenodo",
        record="cc-by",
        license="cc-by-4.0",
        source="Zenodo 6460303 (mskcc_confocal C. elegans)",
        attribution="Santella, Kovacevic, Bao, Hirsch — doi:10.5281/zenodo.6460303 (CC BY 4.0).",
    ),
    "gsplats_visible_human_head": dict(
        bucket="zenodo",
        record="cc-by",
        license="pd-nlm",
        source="NLM Visible Human Project (Male head, PNG)",
        attribution="U.S. NLM Visible Human Project (public domain; acknowledge NLM, no endorsement implied).",
    ),
    "gsplats_multichannel": dict(
        bucket="zenodo",
        record="cc-by",
        license="cc-by-4.0",
        source="IDR idr0062 (image 6001240; Nessys, Blin et al. 2019)",
        attribution="IDR idr0062 (Blin et al., PLOS Biol 2019; CC BY 4.0).",
    ),
    "gsplats_dapi": dict(
        bucket="zenodo",
        record="cc-by",
        license="cc-by-4.0",
        source="IDR idr0062 (image 6001240; Nessys, Blin et al. 2019)",
        attribution="IDR idr0062 (Blin et al., PLOS Biol 2019; CC BY 4.0).",
    ),
    "census_umap_1m": dict(
        bucket="zenodo",
        record="cc-by",
        license="cc-by-4.0",
        dir="",
        source="CZ CELLxGENE Census (our computed 3D UMAP coords)",
        attribution="Derived UMAP coordinates over CZ CELLxGENE Census (CC BY 4.0).",
    ),
    "3d_umap_coords_human": dict(
        bucket="zenodo",
        record="cc-by",
        license="cc-by-4.0",
        dir="",
        source="Human multiome peak UMAP (our computed coords)",
        attribution="Derived UMAP coordinates (CC BY 4.0 upstream).",
    ),
    "3d_umap_coords_mouse": dict(
        bucket="zenodo",
        record="cc-by",
        license="cc-by-4.0",
        dir="",
        source="Mouse multiome peak UMAP (our computed coords)",
        attribution="Derived UMAP coordinates (CC BY 4.0 upstream).",
    ),
    # -- CC BY-SA (derived product must be relicensed CC BY-SA 4.0) ---------
    "gsplats_opencell_map4": dict(
        bucket="zenodo",
        record="cc-by-sa",
        license="cc-by-sa-4.0",
        source="OpenCell (CZ Biohub) MAP4",
        attribution="OpenCell / CZ Biohub — Cho et al., Science 2022, doi:10.1126/science.abi6983 (CC BY-SA 4.0).",
    ),
    "gsplats_zebrafish": dict(
        bucket="zenodo",
        record="cc-by-sa",
        license="cc-by-sa-4.0",
        source="Zenodo 1211599 (zebrafish light-sheet)",
        attribution="Pia Aanstad — doi:10.5281/zenodo.1211599 (CC BY-SA 4.0).",
    ),
    # -- NEW heavy timelapses computed on obsidian (not yet in LFS) ----------
    "gsplats_4d_neuromast_2ch": dict(
        bucket="zenodo",
        record="cc-by",
        license="cc-by-4.0",
        source="Neuromast 2-channel light-sheet timelapse (iSIM)",
        attribution="Adrian Jacobo (CZ Biohub SF); used with permission (CC BY 4.0).",
        pending_upload=True,
    ),
    "h2afva": dict(
        bucket="zenodo",
        record="h2afva",
        license="cc-by-4.0",
        source="h2afva zebrafish histone light-sheet timelapse (Royer lab)",
        attribution="Royer lab, CZ Biohub SF (CC BY 4.0).",
        pending_upload=True,
        # Two size variants in one record: the demo fetches the light default,
        # the full timelapse is opt-in (avoids a 16 GB first-run download over
        # Zenodo's best-effort, unguaranteed bandwidth). Files filled at upload.
        variants={
            "51tp": dict(
                default=True,
                approx_bytes=2_900_000_000,
                note="51-timepoint refit — lighter default for the demo.",
            ),
            "253tp": dict(
                default=False,
                approx_bytes=16_000_000_000,
                note="Full 253-timepoint timelapse — opt-in (large download).",
            ),
        },
    ),
    # ---- Bucket 3: NOT redistributable → fetch raw + compute locally -------
    "milky_way_gaia_3m": dict(
        bucket="local-compute",
        redistribute=False,
        license="cc-by-nc-3.0-igo",
        dir="",
        source="ESA Gaia DR3 archive",
        reason="CC BY-NC (non-commercial) — incompatible with a cleanly-reusable host.",
        strategy="Query the ESA Gaia archive and build the point cloud client-side (no GPU); cache locally. Requires the ESA/Gaia/DPAC acknowledgement.",
    ),
    "gsplats_tng_cosmic_web": dict(
        bucket="local-compute",
        redistribute=False,
        license="none",
        source="IllustrisTNG (tng-project.org)",
        reason="Access-gated (account + API key); no redistribution license.",
        strategy="Fetch particle data after TNG registration + fit locally (GPU).",
    ),
    "gsplats_acto3d_heart": dict(
        bucket="local-compute",
        redistribute=False,
        license="none",
        source="Acto3D sample data (github.com/Acto3D/Acto3D)",
        reason="Repo MIT covers software only; sample data unlicensed (all rights reserved).",
        strategy="Fetch raw from the Acto3D source + fit locally (GPU).",
    ),
    "gsplats_tribolium": dict(
        bucket="local-compute",
        redistribute=False,
        license="conflict",
        source="Cell Tracking Challenge / Zenodo 5270323",
        reason="CTC origin forbids cloning; Zenodo re-host CC-BY is an authority conflict.",
        strategy="Fetch raw from the Cell Tracking Challenge + fit locally (GPU), or seek CTC permission.",
    ),
    # ---- Bucket 1: regenerate client-side (no GPU); not hosted -------------
    "dipc_genome": dict(
        bucket="regenerate",
        redistribute=True,
        license="cc-by-4.0",
        source="GEO GSE117876 (Dip-C GM12878)",
        strategy="Small GEO download + CPU polyline build; auto-download+rebuild fallback already in the demo.",
    ),
}


_LFS_POINTER_V1 = "version https://git-lfs.github.com/spec/v1"


def _pointer_checksum(path: Path) -> Optional[dict]:
    """``{sha256, bytes}`` read out of an unpulled git-LFS pointer, else None.

    A pointer is a <1 KB text stub::

        version https://git-lfs.github.com/spec/v1
        oid sha256:<hex>
        size <bytes>
    """
    if path.stat().st_size > 1024:
        return None
    try:
        text = path.read_text(encoding="utf-8")
    except (UnicodeDecodeError, OSError):
        return None
    if not text.startswith(_LFS_POINTER_V1):
        return None
    oid: Optional[str] = None
    size: Optional[int] = None
    for line in text.splitlines():
        if line.startswith("oid sha256:"):
            oid = line.split(":", 1)[1].strip()
        elif line.startswith("size "):
            size = int(line.split(" ", 1)[1].strip())
    return {"sha256": oid, "bytes": size} if oid and size is not None else None


def _checksum(path: Path) -> dict:
    """``{sha256, bytes}`` for a data file, whether pulled or still a pointer.

    Needs neither ``git``, the ``git-lfs`` binary, nor a git work tree — which is
    exactly what CI has, since CI deliberately checks out without ``lfs: true``
    (see the checkout note in .github/workflows/ci.yml).

    For a *pulled* LFS file the LFS oid IS the sha256 of the content, so hashing
    the bytes reproduces what ``git lfs ls-files --json`` would report (~0.2 s for
    the whole ~450 MB tree). For an unpulled one the pointer already carries both
    numbers. Either way the output is identical, so the manifest is reproducible.
    """
    pointer = _pointer_checksum(path)
    if pointer is not None:
        return pointer
    digest = hashlib.sha256()
    with open(path, "rb") as fh:
        for chunk in iter(lambda: fh.read(1 << 20), b""):
            digest.update(chunk)
    return {"sha256": digest.hexdigest(), "bytes": path.stat().st_size}


def _keep(p: Path) -> bool:
    # Real data files only — skip dotfiles (e.g. .gitkeep).
    return p.is_file() and not p.name.startswith(".")


def _files_in(base: Path, glob: str, *, prune: bool) -> Optional[list[dict]]:
    """On-disk entries matching *glob* under *base*, or None when unknowable.

    Returns None — meaning "keep whatever the committed manifest says" — when
    *base* is absent or holds no data files. Without this, running the generator
    after R17 step 4 (``git rm`` the LFS payload), or simply from a partial
    checkout, would emit empty ``files`` lists and silently WIPE every checksum
    in the manifest. ``--prune`` opts out and lets the empty checkout win.
    """
    if not base.is_dir():
        return [] if prune else None
    found = sorted(p for p in base.glob(glob) if _keep(p))
    if not found and not prune:
        return None
    return [{"name": p.name, **_checksum(p)} for p in found]


def _prev_files(prev: dict, name: str, variant: Optional[str] = None) -> list[dict]:
    """The committed manifest's file list for a dataset (or one of its variants)."""
    d = prev.get("datasets", {}).get(name, {})
    if variant is not None:
        return d.get("variants", {}).get(variant, {}).get("files", []) or []
    return d.get("files", []) or []


def _files_for(name: str, spec: dict, prev: dict, *, prune: bool) -> list[dict]:
    """A (non-variant) dataset's files: disk when visible, else the committed list.

    ``pending_upload`` datasets (computed elsewhere, not yet in the repo) have
    nothing on disk to read, so the committed list is authoritative — that is what
    lets R17 step 2 fill in their files without the next regeneration wiping them.
    """
    committed = _prev_files(prev, name)
    if spec.get("pending_upload"):
        return [] if prune else committed
    subdir = spec.get("dir", name)  # default: dir named after the dataset
    found = (
        _files_in(DATA_DIR / subdir, "*", prune=prune)
        if subdir
        else _files_in(DATA_DIR, f"{name}.*", prune=prune)  # top-level single file
    )
    return committed if found is None else found


def _variants_for(name: str, spec: dict, prev: dict, *, prune: bool) -> dict:
    """Build the ``variants`` map, each carrying its own file list.

    Variant files live under ``demos/data/<dir>/<variant>/`` — ``<dir>`` is the
    dataset's ``dir`` (default: its name; ``""`` puts variants at the top level).
    Empty until the dataset is uploaded. Exactly one variant should be marked
    ``default``.
    """
    out = {}
    for vname, vmeta in spec["variants"].items():
        v = dict(vmeta)
        committed = _prev_files(prev, name, vname)
        if spec.get("pending_upload"):
            v["files"] = [] if prune else committed
        else:
            subdir = spec.get("dir", name)
            base = DATA_DIR.joinpath(*[p for p in (subdir, vname) if p])
            found = _files_in(base, "*", prune=prune)
            v["files"] = committed if found is None else found
        out[vname] = v
    return out


def build(prev: Optional[dict] = None, *, prune: bool = False) -> dict:
    prev = prev or {}
    datasets = {}
    for name, spec in DATASETS.items():
        d = {k: v for k, v in spec.items() if k not in ("dir", "variants")}
        # Emit the effective in-repo subdir explicitly so the packaged manifest
        # carries layout info: "" for top-level datasets, the dataset name
        # otherwise. The in-repo LFS fallback in data_fetch honours this.
        d["dir"] = spec.get("dir", name)
        if "variants" in spec:
            d["variants"] = _variants_for(name, spec, prev, prune=prune)
        else:
            d["files"] = _files_for(name, spec, prev, prune=prune)
        datasets[name] = d
    return {
        "schema_version": 1,
        "description": (
            "Single source of truth for Luxar demo dataset disposition (R17). "
            "bucket: zenodo=fetch-on-demand redistributable; "
            "local-compute=fetch raw + build locally (not redistributable); "
            "regenerate=cheap CPU rebuild (not hosted). See scripts/gen_data_manifest.py."
        ),
        "records": RECORDS,
        "datasets": datasets,
    }


def main() -> int:
    ap = argparse.ArgumentParser(description="Generate the demo-data manifest.")
    ap.add_argument(
        "--check",
        action="store_true",
        help="verify the committed manifest is current (no writes); exit 1 on drift",
    )
    ap.add_argument(
        "--prune",
        action="store_true",
        help="let the checkout be authoritative even for datasets it cannot see, "
        "emptying their file lists. Only correct in a fully `git lfs pull`-ed "
        "checkout; without it, entries for absent datasets are preserved.",
    )
    args = ap.parse_args()

    try:
        previous = json.loads(MANIFEST.read_text()) if MANIFEST.exists() else {}
    except (OSError, json.JSONDecodeError) as exc:
        print(f"error: cannot read {MANIFEST}: {exc}", file=sys.stderr)
        return 2

    manifest = build(previous, prune=args.prune)
    text = json.dumps(manifest, indent=2) + "\n"

    if args.check:
        current = MANIFEST.read_text() if MANIFEST.exists() else ""
        if current != text:
            sys.stderr.writelines(
                difflib.unified_diff(
                    current.splitlines(True),
                    text.splitlines(True),
                    fromfile=f"a/{MANIFEST.relative_to(REPO_ROOT)}",
                    tofile="b/(generated)",
                )
            )
            print(
                "\ndemo-data manifest is stale — run `make gen-data-manifest` "
                "(or `hatch run gen-data-manifest`) and commit the result.",
                file=sys.stderr,
            )
            return 1
        print("demo-data manifest is up to date.")
        return 0

    MANIFEST.write_text(text)
    print(
        f"Wrote {MANIFEST.relative_to(REPO_ROOT)} "
        f"({len(manifest['datasets'])} datasets)"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
