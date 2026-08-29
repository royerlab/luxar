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
  * ``acquisition`` — what the dataset as a whole was fitted FROM, and either its
                  stored size (``comparable: true`` + ``stored_bytes``) or the
                  ``reason`` no honest ratio exists. Dataset-level rather than
                  per-archive because a dataset is often several archives (one
                  per channel) fitted from a single file. Gated by
                  ``demos/tests/test_manifest_acquisition.py``.
  * ``files``   — basenames + sha256 (git-LFS oid) + byte size, so a fetched copy
                  is checksum-verified.
  * ``dir``     — the in-repo subdir relative to ``demos/data/`` (empty string =
                  top level; the in-repo LFS fallback in data_fetch honours it).

Curated metadata (license/source/citation) lives here; the ``sha256`` comes from the data
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
import re
import sys
from pathlib import Path
from typing import Optional

REPO_ROOT = Path(__file__).resolve().parent.parent
DATA_DIR = REPO_ROOT / "packages/luxar/src/luxar/demos/data"
MANIFEST = REPO_ROOT / "packages/luxar/src/luxar/demos/data_manifest.json"

# Zenodo records, grouped by license family.
#
# The record ids and DOIs below are REAL and final: Zenodo reserves a DOI at
# deposition time and the deposition id becomes the record id on publication, so
# `https://zenodo.org/records/<id>/...` is already the right URL. What is not yet
# true is that the records are PUBLIC — all three are still unsubmitted drafts,
# and a file URL into a draft 404s for everyone.
#
# `zenodo_doi` is that reserved DOI, i.e. the VERSION DOI of this deposition —
# NOT the version-independent concept DOI, which is a different identifier Zenodo
# mints on publication and which cannot be known from the deposition id.
#
# Hence `published`: while it is false the fetch helper builds no URL at all, so
# the Zenodo leg stays dormant exactly as it did when the ids were null, and
# demos keep resolving cache -> in-repo LFS. Flipping the three flags at
# publication time is what activates fetching, and it is the only edit needed.
# Recording the ids now (rather than at publish time) means the manifest, the
# record descriptions and the reserved DOIs cannot drift apart in the meantime.
#
# Titles are kept in step with the live record titles on purpose: they are what a
# `luxar demo` user is pointed at, and the h2afva one in particular used to
# describe the 51tp cut as a "subset", which its own record text contradicts.
RECORDS = {
    "cc-by": {
        "title": "Luxar demo datasets: permissively licensed (CC-BY, CC0, public domain)",
        "license": "cc-by-4.0",
        "zenodo_doi": "10.5281/zenodo.21912280",
        "zenodo_record": "21912280",
        "base_url": None,
        "published": False,
    },
    "cc-by-sa": {
        "title": "Luxar demo datasets: ShareAlike (CC BY-SA 4.0)",
        "license": "cc-by-sa-4.0",
        "zenodo_doi": "10.5281/zenodo.21912282",
        "zenodo_record": "21912282",
        "base_url": None,
        "published": False,
    },
    "h2afva": {
        "title": (
            "Zebrafish embryogenesis, histone-labelled nuclei: 253-timepoint "
            "light-sheet timelapse as Gaussian splats (with a lighter "
            "51-timepoint fit)"
        ),
        "license": "cc-by-4.0",
        "zenodo_doi": "10.5281/zenodo.21912284",
        "zenodo_record": "21912284",
        "base_url": None,
        "published": False,
    },
    # Its OWN record rather than a 33rd file on `cc-by`, decided by Loic on
    # 2026-08-26. The deciding argument was attribution granularity: on a mixed
    # record "Keller, DataCollector" is one name among four attached to nothing
    # in particular, while here he is the data collector *of this recording*.
    #
    # The shipped single frame in `cc-by` IS frame 150 of this same recording,
    # so one recording spans two DOIs. That is deliberate and handled by a
    # cross-reference in both descriptions rather than by co-location — moving
    # the existing file would break its pins.
    #
    # PUBLICATION GATE, beyond the standing never-publish rule: this record
    # additionally waits on a conversation with Philipp J. Keller, whose imaging
    # it is, and which as of 2026-08-26 had not happened. Loic confirmed that and
    # chose the sequencing: upload to the draft, gate the publish. An upload here
    # implies no consent to publish. Recorded in the repo on purpose — Zenodo's
    # `notes` and `description` are both PUBLISHED metadata, so an internal
    # process gate written there would ship with the record.
    # The timelapse dataset below points at this record; publishing remains
    # gated on the conversation above even after the archive is uploaded.
    "droso-timelapse": {
        "title": (
            "Drosophila melanogaster embryogenesis: a 500-timepoint "
            "light-sheet timelapse as Gaussian splats"
        ),
        "license": "cc-by-4.0",
        "zenodo_doi": "10.5281/zenodo.22118695",
        "zenodo_record": "22118695",
        "base_url": None,
        "published": False,
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
        acquisition=dict(
            description="the scikit-image `kidney` sample, all 3 channels of which are fitted",
            comparable=True,
            stored_bytes=None,
        ),
    ),
    "gsplats_cells3d": dict(
        bucket="zenodo",
        record="cc-by",
        license="cc0-1.0",
        source="scikit-image (data.cells3d) — Allen Institute for Cell Science",
        attribution="scikit-image sample data.",
        acquisition=dict(
            description="the scikit-image `cells3d` sample, both channels of which are fitted",
            comparable=True,
            stored_bytes=None,
        ),
    ),
    "gsplats_cmu1_pathology": dict(
        bucket="zenodo",
        record="cc-by",
        license="cc0-1.0",
        source="OpenSlide test data (Aperio CMU-1.svs)",
        attribution="OpenSlide CMU-1 (CC0 1.0 public domain).",
        acquisition=dict(
            description="Aperio CMU-1.svs, a ~169 MB multi-resolution slide pyramid",
            comparable=False,
            reason=(
                "the file stores every pyramid level and the fit reads one, then "
                "resizes it, so its size would price the levels that were never "
                "fitted rather than the compression of the one that was"
            ),
        ),
    ),
    "gsplats_cryoem_virus": dict(
        bucket="zenodo",
        record="cc-by",
        license="cc0-1.0",
        source="EMDB EMD-5384",
        attribution="EMDB is public domain / CC0. Map: EMD-5384.",
        acquisition=dict(
            description="EMD-5384 `emd_5384.map.gz` -- the whole map, as downloaded",
            comparable=True,
            stored_bytes=None,
        ),
    ),
    "gsplats_ct_totalsegmentator": dict(
        bucket="zenodo",
        record="cc-by",
        license="cc-by-4.0",
        positional_pairs={
            "ct_atlas": (
                "ct_atlas.gsplats.zarr.zip",
                "ct_atlas_labels.npz",
            )
        },
        source="TotalSegmentator (Wasserthal et al. 2023)",
        attribution="TotalSegmentator (Wasserthal et al. 2023; CC BY 4.0).",
        acquisition=dict(
            description="a TotalSegmentator subset, repacked locally into an npz",
            comparable=False,
            reason=(
                "the download is a multi-scan subset and the demo repacks selected "
                "scans into its own npz, so neither the download nor the repack is "
                "the same array the fit represents"
            ),
        ),
    ),
    "gsplats_milkyway_dust": dict(
        bucket="zenodo",
        record="cc-by",
        license="cc-by-4.0",
        source="Zenodo 3993082 (3D dust map)",
        attribution="doi:10.5281/zenodo.3993082 (CC BY 4.0).",
        acquisition=dict(
            description=(
                "the `mean` array inside Zenodo 3993082 `mean_std.h5`, measured "
                "with its own HDF5 storage size"
            ),
            comparable=True,
            stored_bytes=None,
            note=(
                "the FILE is 2.4 GB but holds mean AND std, and the fit reads only "
                "the mean, so the file size is not the denominator -- h5py's "
                "per-dataset get_storage_size() is"
            ),
        ),
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
        acquisition=dict(
            description="the Zenodo TIFF timelapse, every timepoint of which is fitted",
            comparable=True,
            stored_bytes=None,
        ),
    ),
    "gsplats_visible_human_head": dict(
        bucket="zenodo",
        record="cc-by",
        license="pd-nlm",
        positional_pairs={
            "vh_head": (
                "vh_head.gsplats.zarr.zip",
                "vh_head_colors.npz",
            )
        },
        source="NLM Visible Human Project (Male head, PNG)",
        attribution="U.S. NLM Visible Human Project (public domain; acknowledge NLM, no endorsement implied).",
        acquisition=dict(
            description="377 RGB cryosection photographs (Visible Human Project)",
            comparable=False,
            reason=(
                "the fit converts colour photographs to a single greyscale volume "
                "and crops it to content, so a ratio against the stored photographs "
                "would price a colour conversion, not compression"
            ),
        ),
    ),
    "gsplats_nexrad_supercell": dict(
        bucket="zenodo",
        record="cc-by",
        license="pd-noaa",
        source="NOAA NEXRAD Level II, KTLX (Twin Lakes, OK), 2013-05-31 21Z - 06-01 03Z",
        # NOAA NODD asks for attribution, forbids implying endorsement, and
        # forbids presenting MODIFIED data as original NOAA data — hence the
        # explicit derived-product sentence.
        attribution=(
            "U.S. NOAA/NWS NEXRAD Level II via NOAA Open Data Dissemination "
            "(public domain, 17 U.S.C. 105; acknowledge NOAA, no endorsement "
            "implied). Derived product: regridded and Gaussian-fitted, not "
            "original NOAA data."
        ),
        acquisition=dict(
            description="82 NEXRAD Level II scans (~800 MB gzipped)",
            comparable=False,
            reason=(
                "the scans are polar sweeps carrying several moments and elevation "
                "angles, of which the demo re-grids ONE moment inside one box, so "
                "the figure would price dropping the other moments and the "
                "polar-to-Cartesian resampling"
            ),
        ),
    ),
    "gsplats_multichannel": dict(
        bucket="zenodo",
        record="cc-by",
        license="cc-by-4.0",
        source="IDR idr0062 (image 6001240; Nessys, Blin et al. 2019)",
        attribution="IDR idr0062 (Blin et al., PLOS Biol 2019; CC BY 4.0).",
        acquisition=dict(
            description="IDR 6001240 OME-Zarr, the two fitted channels' own stored chunks",
            comparable=True,
            stored_bytes=None,
        ),
    ),
    "gsplats_dapi": dict(
        bucket="zenodo",
        record="cc-by",
        license="cc-by-4.0",
        source="IDR idr0062 (image 6001240; Nessys, Blin et al. 2019)",
        attribution="IDR idr0062 (Blin et al., PLOS Biol 2019; CC BY 4.0).",
        acquisition=dict(
            description="IDR 6001240 OME-Zarr, the fitted channel's own stored chunks",
            comparable=True,
            stored_bytes=None,
        ),
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
        acquisition=dict(
            description="the OpenCell MAP4 TIFF, both channels of which are fitted",
            comparable=True,
            stored_bytes=None,
        ),
    ),
    "gsplats_flylight_mcfo_63x": dict(
        bucket="zenodo",
        record="cc-by",
        license="cc-by-4.0",
        source=(
            "Janelia FlyLight Gen1 MCFO, line VT019012, slide 20140423_20_D5, "
            "63x confocal (stitched unaligned_stack.h5j)"
        ),
        attribution=(
            "Janelia FlyLight Project Team, HHMI Janelia Research Campus — "
            "Meissner et al., eLife 12:e80660 (2023), doi:10.7554/eLife.80660; "
            "MCFO method: Nern, Pfeiffer & Rubin, PNAS 112(22) (2015), "
            "doi:10.1073/pnas.1506763112; VT driver lines: Tirian & Dickson, "
            "bioRxiv 198648 (2017), doi:10.1101/198648. Imagery CC BY 4.0."
        ),
    ),
    "gsplats_zebrafish": dict(
        bucket="zenodo",
        record="cc-by-sa",
        license="cc-by-sa-4.0",
        source="Zenodo 1211599 (zebrafish gastrulation, confocal timelapse)",
        attribution="Pia Aanstad — doi:10.5281/zenodo.1211599 (CC BY-SA 4.0).",
        acquisition=dict(
            description=(
                "the whole LSM timelapse: 151 timepoints of 44 x 512 x 512 uint8, "
                "every one of which is fitted"
            ),
            comparable=True,
            # 151 * 44 * 512 * 512 voxels at one byte each. The .lsm FILE is
            # 2,080,484,264 bytes, but that includes an embedded RGB thumbnail
            # series and the LSM metadata, neither of which is fitted.
            stored_bytes=1_741_684_736,
        ),
    ),
    # -- Heavy timelapses computed on obsidian -------------------------------
    "gsplats_4d_neuromast_2ch": dict(
        bucket="zenodo",
        record="cc-by",
        license="cc-by-4.0",
        source="Neuromast 2-channel light-sheet timelapse (iSIM)",
        attribution="Adrian Jacobo (CZ Biohub SF); used with permission (CC BY 4.0).",
        # Permission CONFIRMED by the author 2026-08-12; both channels uploaded to
        # the cc-by record and pinned below (md5 verified against Zenodo). The
        # record is still a DRAFT, so RECORDS["cc-by"] carries its id but
        # `published: False`, and the fetch leg stays dormant until that flips —
        # the pins are what publication turns on.
    ),
    "gsplats_cell_tracking": dict(
        bucket="zenodo",
        record="cc-by",
        license="cc0-1.0",
        source=(
            "Kaggle competition 'Biohub - Cell Tracking During Development' "
            "(zebrafish light-sheet crops + GEFF ground-truth lineages)"
        ),
        attribution=(
            "Biohub cell-tracking challenge data (CC0); imaging by the Royer "
            "group, CZ Biohub SF. Derived product: per-crop 4D Gaussian-splat "
            "fits with substitutive LOD."
        ),
        # Computed on obsidian, awaiting upload to the cc-by record. Redistributable
        # as a DERIVED product because the source is CC0 — and since the raw
        # competition data is behind an authenticated endpoint, hosting the fits is
        # what makes the demo runnable with no Kaggle credentials and no GPU at all.
        #
        # ONE file set, at the full 100 timepoints: ~700 MB (measured — 77.3 MB per
        # crop x 9). No lighter variant, deliberately. The demo exists to show the
        # whole timelapse, a decimated one would undercut that, and 700 MB is modest
        # for this catalogue (the celegans demo pulls 26 GB). The derived product is
        # far smaller than either the ~4 GB of raw crops or the ~450 MB of
        # per-timepoint fit cache it is built from, because a fitted splat costs
        # about 12 bytes once encoded and zipped.
        acquisition=dict(
            description=(
                "one competition training crop per fitted archive: an OME-Zarr "
                "store of T=100, Z=64, Y=256, X=256 uint16 (838.9 MB of raw "
                "voxels each), fitted timepoint by timepoint at its native grid"
            ),
            comparable=True,
            # Measured when the crops are next in hand: they sit behind the
            # authenticated competition endpoint (~450 MB of chunks per crop),
            # so opening them to size the chunk-compressed store is a 4 GB
            # download rather than a local probe.
            stored_bytes=None,
        ),
        pending_upload=True,
    ),
    "h2afva": dict(
        bucket="zenodo",
        record="h2afva",
        license="cc-by-4.0",
        source="h2afva zebrafish histone light-sheet timelapse (Royer lab)",
        attribution="Royer lab, CZ Biohub SF (CC BY 4.0).",
        acquisition=dict(
            description=(
                "the full 253-timepoint h2afva light-sheet timelapse; the 51tp "
                "variant is every fifth frame from the same fit"
            ),
            comparable=True,
            stored_bytes=None,
        ),
        # Both variants are uploaded and pinned, so there is no `pending_upload`
        # here. The 51tp file that used to be a SUPERSEDED build — pre-isotropic
        # (z extent 405 raw voxels instead of 1620), no substitutive LOD levels,
        # format 3.2 — has been replaced.
        #
        # The 51tp variant is now a strided SLICE of the 253tp fit rather than an
        # independent fit: every 5th frame (source frames 0, 5, ... 250),
        # renumbered to a dense 0..50.
        # The renumbering is load-bearing, not cosmetic — it puts the stacked axis
        # back on a regular grid so the encoder's gridded-axis snap stores it
        # EXACTLY. The old build's frame values were quantization-smeared off
        # their integers, which left four of every five slider positions
        # rendering an empty scene. Because it is a slice, the two variants now
        # agree splat-for-splat on the frames they share, where the previous
        # independent fits differed (2.50M vs 2.38M per timepoint at the finest
        # level). Shipped as the default because pulling ~1.1 GB is far easier
        # than ~9.3 GB over Zenodo's best-effort bandwidth.
        variants={
            "51tp": dict(
                default=True,
                approx_bytes=1_115_714_088,
                note="51-timepoint fit (every 5th frame) — lighter default for the demo.",
            ),
            "253tp": dict(
                default=False,
                approx_bytes=9_253_211_541,
                note="Full 253-timepoint timelapse — opt-in (large download).",
            ),
        },
    ),
    # The whole 500-timepoint recording, of which the single-frame entry below
    # is frame 150. Its OWN record (see RECORDS["droso-timelapse"]) rather than a
    # 33rd file on cc-by, so Keller is credited as the data collector of THIS
    # recording rather than as one name among four on a mixed record.
    "gsplats_4d_drosophila_embryogenesis": dict(
        bucket="zenodo",
        record="droso-timelapse",
        license="cc-by-4.0",
        source="Drosophila His2Av::mRFP1 embryo, 500-timepoint SiMView light-sheet timelapse (Royer/Keller)",
        attribution="Royer & Keller labs — Royer et al., Nat. Biotechnol. 34, 1267-1278 (2016), doi:10.1038/nbt.3708 (CC BY 4.0).",
        acquisition=dict(
            # A clean 1:1 denominator, unusually: the fit consumed the WHOLE
            # recording — all 500 timepoints, full spatial extent, the single
            # channel — with no downscale and no crop. So the dataset as a whole
            # really is comparable to the acquisition as a whole, which is what
            # this block is for.
            description=(
                "the whole 500-timepoint SiMView recording "
                "(500 x 108 x 1352 x 532 uint16), every timepoint fitted at full "
                "resolution"
            ),
            comparable=True,
            stored_bytes=19_333_771_628,  # DrosophilaHistone.zarr.zip as stored
        ),
    ),
    "gsplats_3d_drosophila_gastrulation": dict(
        bucket="zenodo",
        record="cc-by",
        license="cc-by-4.0",
        source="Drosophila His2Av::mRFP1 embryo, SiMView light-sheet (Royer/Keller)",
        attribution="Royer & Keller labs — Royer et al., Nat. Biotechnol. 34, 1267-1278 (2016), doi:10.1038/nbt.3708 (CC BY 4.0).",
    ),
    "gsplats_3d_h2afva_stack": dict(
        bucket="zenodo",
        # The general cc-by record, NOT the h2afva timelapse one: this is a
        # single 24 MB stack that ships like every other demo dataset, and
        # record grouping cannot be changed after publication.
        record="cc-by",
        license="cc-by-4.0",
        source="h2afva zebrafish histone light-sheet, single stack (Royer lab)",
        attribution="Royer lab, CZ Biohub SF (CC BY 4.0).",
    ),
    "gsplats_3d_h2afva_decimation": dict(
        bucket="zenodo",
        record="cc-by",  # derived from the single stack; ships with it
        license="cc-by-4.0",
        source="h2afva single stack at four decimation levels (Royer lab)",
        attribution="Royer lab, CZ Biohub SF (CC BY 4.0).",
    ),
    # ---- Bucket 3: NOT redistributable → fetch raw + compute locally -------
    "milky_way_gaia_3m": dict(
        bucket="local-compute",
        redistribute=False,
        license="cc-by-nc-3.0-igo",
        dir="",
        source="ESA Gaia DR3 archive",
        reason="CC BY-NC (non-commercial) — incompatible with a cleanly-reusable host.",
        strategy="Build on demand with `luxar demo run gaia_milky_way -- --build-catalog` (queries the ESA Gaia archive, no GPU) into ~/.cache/luxar/, or place a copy there. Requires the ESA/Gaia/DPAC acknowledgement.",
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
    "gsplats_flylight_mcfo": dict(
        bucket="local-compute",
        # Unlike the other local-compute rows this one IS redistributable —
        # it is local-compute only because the cc-by Zenodo record does not
        # exist yet. Promote it to bucket="zenodo", record="cc-by" once it
        # does; nothing else about the demo has to change.
        redistribute=True,
        license="cc-by-4.0",
        source="FISBe v1.0 (Zenodo 10875063) / Janelia FlyLight Gen1 MCFO",
        attribution=(
            "FISBe (Mais et al., CVPR 2024; doi:10.5281/zenodo.10875063, "
            "CC BY 4.0). Imagery from the FlyLight Project Team, Janelia "
            "Research Campus, HHMI; cite Meissner et al. eLife 2023 "
            "12:e80660 and Tirian & Dickson 2017 for the VT line."
        ),
        reason="CC BY 4.0 and redistributable, but no Zenodo record is published yet.",
        strategy=(
            "Range-extract one sample (~415 MB) from the 7.1 GB Zenodo "
            "archive + fit locally (GPU). The archive is never fetched whole."
        ),
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
# A git-LFS oid is a lowercase hex sha256. Anything else can never match
# hashlib's ``hexdigest()`` (the comparison downstream is case-sensitive), so an
# entry built from it would be permanently unverifiable — reject it instead.
_LFS_OID_RE = re.compile(r"[0-9a-f]{64}")
_LFS_SIZE_RE = re.compile(r"[0-9]+")


def _pointer_checksum(path: Path) -> Optional[dict]:
    """``{sha256, bytes}`` read out of an unpulled git-LFS pointer.

    Returns None for anything that is not a pointer stub (a pulled data file);
    raises ValueError for a file that carries the v1 header but no usable
    (oid, size) — a corrupt pointer must fail loudly, not be hashed as data.

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
        # A malformed value leaves oid/size unset, so it lands on the
        # corrupt-pointer error below instead of aborting with a bare,
        # unattributed ValueError (or, worse, shipping a bogus entry).
        if line.startswith("oid sha256:"):
            value = line.split(":", 1)[1].strip()
            if _LFS_OID_RE.fullmatch(value):
                oid = value
        elif line.startswith("size "):
            value = line.split(" ", 1)[1].strip()
            if _LFS_SIZE_RE.fullmatch(value):
                size = int(value)
    if oid is not None and size is not None:
        return {"sha256": oid, "bytes": size}
    # A file carrying the v1 header but no usable (oid, size) is a corrupt
    # pointer. Returning None here would hand it to _checksum, which hashes the
    # ~130-byte stub and silently ships a plausible-but-bogus {sha256, bytes};
    # fail loudly and name the file instead.
    raise ValueError(f"corrupt git-LFS pointer (missing or malformed oid/size): {path}")


def _checksum(path: Path) -> dict:
    """``{sha256, bytes}`` for a data file, whether pulled or still a pointer.

    Needs neither ``git``, the ``git-lfs`` binary, nor a git work tree — which is
    exactly what CI has, since CI deliberately checks out without ``lfs: true``
    (see the checkout note in .github/workflows/ci.yml).

    This describes the copy IN THIS REPO, and only ever that: what a Zenodo
    record serves has no source on disk, so ``hosted_sha256`` is carried forward
    from the committed manifest instead (see ``_carry_hosted``).

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


# Local scratch that must never enter the shipped manifest: quarantined copies
# (``.corrupt``), interrupted downloads (``.part`` plus its ``.part.validator``
# If-Range resume sidecar), and generic temporaries.
_SCRATCH_SUFFIXES = (".corrupt", ".part", ".part.validator", ".tmp")


def _keep(p: Path) -> bool:
    # Real data files only — skip dotfiles (e.g. .gitkeep) and local scratch
    # (quarantined/partial/temp copies) that ensure_dataset could never resolve.
    if not p.is_file() or p.name.startswith("."):
        return False
    return not p.name.endswith(_SCRATCH_SUFFIXES)


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


#: Per-file keys that describe the HOSTED artifact rather than the in-repo copy.
#:
#: Both must be carried, and for the same reason: `bytes` is as ambiguous as
#: `sha256` was once the two copies differ. Keeping the digest and size paired
#: preserves a complete description of the hosted artifact for consumers that
#: need to compare it with the in-repo copy.
#:
#: Named explicitly rather than matched as a `hosted_*` prefix, so a typo'd key
#: is dropped loudly by the drift gate instead of carried forever.
_HOSTED_KEYS = ("hosted_sha256", "hosted_bytes")

#: Digests this project pinned in an EARLIER generation, newest last.
#:
#: When an in-repo pin changes, regeneration records the outgoing digest while
#: the old bytes are still knowable. A hosted-only re-pin has no bytes on disk to
#: compare, so its outgoing digest must be appended by hand as part of the edit.
#: Without that history, `data_fetch` cannot tell "out of date" from "corrupt"
#: and must quarantine the cache. That is what #1734 did to
#: `gsplats_3d_drosophila_gastrulation`.
#:
#: Unbounded and ordered newest-last: a flat list of 64-char hashes is a few
#: hundred bytes even after many re-pins. Reverting a pin moves that digest to
#: the end so the runtime's one-generation fallback remains correct.
_SUPERSEDED_KEY = "superseded_sha256"


def _carry_hosted(found: list[dict], committed: list[dict]) -> list[dict]:
    """Re-attach hosted keys, and record a changed pin as superseded.

    The generator can only ever compute the digest of the copy IN THIS REPO —
    ``_checksum`` reads the git-LFS pointer's oid or hashes the bytes. What the
    Zenodo record serves has no source on disk at all, so ``hosted_sha256`` is
    carried forward (or edited in by hand, or written by the upload tool) and
    must survive a regeneration that rebuilds every entry from scratch.

    Without this, any dataset whose data dir happens to be VISIBLE would silently
    lose its hosted pin on the next ``make gen-data-manifest`` — while datasets
    with no dir on disk kept theirs, because those keep the committed list whole.
    A field that survives in some rows and evaporates in others is worse than one
    that never worked.

    Absent stays absent: the key is only added when the committed entry had one,
    which is what keeps a regeneration byte-identical for the datasets that have
    no hosted pin.
    """
    prev = {e["name"]: e for e in committed if e.get("name")}
    out = []
    for entry in found:
        old_entry = prev.get(entry.get("name"))
        if old_entry is None:
            out.append(entry)
            continue
        merged = dict(entry)
        for key in _HOSTED_KEYS:
            if old_entry.get(key) is not None:
                merged[key] = old_entry[key]
        # Carry the history forward, and EXTEND it when this regeneration changes
        # the pin: the outgoing digest is what every existing cache holds, and it
        # is the only thing that later distinguishes superseded from corrupt.
        history = list(old_entry.get(_SUPERSEDED_KEY) or ())
        outgoing = old_entry.get("sha256")
        if outgoing and merged.get("sha256") and outgoing != merged["sha256"]:
            if outgoing in history:
                history.remove(outgoing)
            history.append(outgoing)
        if history:
            merged[_SUPERSEDED_KEY] = history
        out.append(merged)
    return out


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
    return committed if found is None else _carry_hosted(found, committed)


def _declare_positional_pairs(
    files: list[dict], groups: dict[str, tuple[str, ...]]
) -> list[dict]:
    """Stamp positionally indexed files with their shared generation group."""
    if not groups:
        return files
    by_name = {entry["name"]: entry for entry in files}
    declared: set[str] = set()
    for group, members in groups.items():
        member_names = set(members)
        if len(member_names) < 2:
            raise ValueError(f"positional pair {group!r} must name at least two files")
        present = member_names.intersection(by_name)
        if not present:
            continue
        missing = member_names - by_name.keys()
        if missing:
            raise ValueError(
                f"positional pair {group!r} names missing files: {sorted(missing)}"
            )
        overlap = declared.intersection(members)
        if overlap:
            raise ValueError(
                f"files belong to more than one positional pair: {sorted(overlap)}"
            )
        history_lengths = {
            len(by_name[member].get(_SUPERSEDED_KEY) or ()) for member in members
        }
        if len(history_lengths) != 1:
            raise ValueError(
                f"positional pair {group!r} did not move atomically: member "
                "history depths differ"
            )
        for member in members:
            by_name[member]["positional_pair"] = group
        declared.update(members)
    return files


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
            v["files"] = committed if found is None else _carry_hosted(found, committed)
        out[vname] = v
    return out


def build(prev: Optional[dict] = None, *, prune: bool = False) -> dict:
    prev = prev or {}
    datasets = {}
    for name, spec in DATASETS.items():
        if "variants" in spec and spec.get("positional_pairs"):
            raise ValueError(
                f"dataset {name!r} cannot declare both variants and positional_pairs"
            )
        d = {
            k: v
            for k, v in spec.items()
            if k not in ("dir", "variants", "positional_pairs")
        }
        # Emit the effective in-repo subdir explicitly so the packaged manifest
        # carries layout info: "" for top-level datasets, the dataset name
        # otherwise. The in-repo LFS fallback in data_fetch honours this.
        d["dir"] = spec.get("dir", name)
        if "variants" in spec:
            d["variants"] = _variants_for(name, spec, prev, prune=prune)
        else:
            d["files"] = _declare_positional_pairs(
                _files_for(name, spec, prev, prune=prune),
                spec.get("positional_pairs", {}),
            )
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
