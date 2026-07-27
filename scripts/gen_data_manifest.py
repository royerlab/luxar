#!/usr/bin/env python3
"""Generate ``demos/data/manifest.json`` — the single source of truth for demo
dataset disposition (R17: retire git-LFS heavy datasets → Zenodo).

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

Curated metadata (license/source/citation) lives here; checksums are merged from
``git lfs ls-files --json`` so the manifest stays reproducible. Re-run after any
dataset re-encode:

    python scripts/gen_data_manifest.py            # write the manifest
    python scripts/gen_data_manifest.py --check     # verify it is up to date (CI)
"""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
DATA_DIR = REPO_ROOT / "packages/luxar/src/luxar/demos/data"
MANIFEST = DATA_DIR / "manifest.json"
DATA_PREFIX = "packages/luxar/src/luxar/demos/data/"

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


def _lfs_checksums() -> dict[str, dict]:
    """Map demos/data-relative path → {sha256, bytes} from git-LFS oids."""
    out = subprocess.check_output(
        ["git", "lfs", "ls-files", "--json"], cwd=REPO_ROOT, text=True
    )
    result: dict[str, dict] = {}
    for f in json.loads(out).get("files", []):
        name = f["name"]
        if not name.startswith(DATA_PREFIX):
            continue
        rel = name[len(DATA_PREFIX) :]
        if f.get("oid_type") == "sha256" and f.get("oid"):
            result[rel] = {"sha256": f["oid"], "bytes": int(f["size"])}
    return result


def _keep(p: Path) -> bool:
    # Real data files only — skip dotfiles (e.g. .gitkeep) and the manifest.
    return p.is_file() and not p.name.startswith(".") and p.name != "manifest.json"


def _files_in(
    base: Path, prefix: str, checksums: dict[str, dict], glob: str
) -> list[dict]:
    """List files matching *glob* under *base*, joined with their LFS checksums.

    *prefix* is the demos/data-relative directory prefix used to look up
    checksums (``""`` for top-level files).
    """
    files = []
    for p in sorted(base.glob(glob)):
        if not _keep(p):
            continue
        rel = f"{prefix}/{p.name}" if prefix else p.name
        entry = {"name": p.name}
        entry.update(checksums.get(rel, {"sha256": None, "bytes": None}))
        files.append(entry)
    return files


def _files_for(name: str, spec: dict, checksums: dict[str, dict]) -> list[dict]:
    """List a (non-variant) dataset's on-disk files with their checksums.

    ``pending_upload`` datasets (computed on obsidian, not yet in the repo) have
    no local files; their file list is left empty to be filled at upload time.
    """
    if spec.get("pending_upload"):
        return []
    subdir = spec.get("dir", name)  # default: dir named after the dataset
    if subdir:
        return _files_in(DATA_DIR / subdir, subdir, checksums, "*")
    # top-level single file: match "<name>.*"
    return _files_in(DATA_DIR, "", checksums, f"{name}.*")


def _variants_for(name: str, spec: dict, checksums: dict[str, dict]) -> dict:
    """Build the ``variants`` map, each carrying its own file list.

    Variant files live under ``demos/data/<name>/<variant>/`` (empty until the
    dataset is uploaded). Exactly one variant should be marked ``default``.
    """
    out = {}
    for vname, vmeta in spec["variants"].items():
        v = dict(vmeta)
        if spec.get("pending_upload"):
            v["files"] = []
        else:
            v["files"] = _files_in(
                DATA_DIR / name / vname, f"{name}/{vname}", checksums, "*"
            )
        out[vname] = v
    return out


def build() -> dict:
    checksums = _lfs_checksums()
    datasets = {}
    for name, spec in DATASETS.items():
        d = {k: v for k, v in spec.items() if k not in ("dir", "variants")}
        if "variants" in spec:
            d["variants"] = _variants_for(name, spec, checksums)
        else:
            d["files"] = _files_for(name, spec, checksums)
        datasets[name] = d
    return {
        "schema_version": 1,
        "description": (
            "Single source of truth for Luxar demo dataset disposition (R17). "
            "bucket: zenodo=fetch-on-demand redistributable; "
            "local-compute=fetch raw + build locally (not redistributable); "
            "regenerate=cheap CPU rebuild (not hosted). See scripts/gen_data_manifest.py."
        ),
        "cache_root": "~/.cache/luxar",
        "records": RECORDS,
        "datasets": datasets,
    }


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--check", action="store_true", help="verify manifest is current")
    args = ap.parse_args()
    manifest = build()
    text = json.dumps(manifest, indent=2) + "\n"
    if args.check:
        current = MANIFEST.read_text() if MANIFEST.exists() else ""
        if current != text:
            print(
                "manifest.json is out of date — run: python scripts/gen_data_manifest.py"
            )
            return 1
        print("manifest.json is up to date.")
        return 0
    MANIFEST.write_text(text)
    print(f"Wrote {MANIFEST} ({len(manifest['datasets'])} datasets)")
    return 0


if __name__ == "__main__":
    sys.exit(main())
