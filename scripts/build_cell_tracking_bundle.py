#!/usr/bin/env python3
"""Build the uploadable artifacts for the ``gsplats_cell_tracking`` dataset.

The cell-tracking demo's raw source sits behind an authenticated Kaggle
competition endpoint, so the redistributable thing is the **derived product**:
per crop, the finished 4D Gaussian-splat volume (LOD ladder included) plus the
track geometry. Once those files are published, the demo runs with no credentials
and no GPU.

This reads the demo's own per-timepoint fit cache, runs the demo's own combine →
LOD → tracks pipeline, and writes one pair of files per crop, plus the ``files``
block to paste into the committed ``data_manifest.json`` entry (name + sha256 +
bytes, which is what makes a fetched copy verifiable).

    # every crop the fit cache holds, at all 100 timepoints (what is hosted)
    python scripts/build_cell_tracking_bundle.py

    # one crop only, to exercise the hosted path without building 0.7 GB
    python scripts/build_cell_tracking_bundle.py --crops 6bba_09961292

Output lands in ``--out`` (default ``delme/cell_tracking_bundle/``), deliberately
NOT under ``demos/data/``: these files are for Zenodo, and the in-repo tree is
what R17 exists to empty.
"""

from __future__ import annotations

import argparse
import hashlib
import importlib.util
import json
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
DEMO_PATH = (
    REPO_ROOT
    / "packages/luxar/src/luxar/demos/demo_gsplats_4d_cell_tracking_challenge.py"
)


def _load_demo():
    """Import the demo by path, as its own tests do."""
    spec = importlib.util.spec_from_file_location("_ct_demo_bundle", DEMO_PATH)
    if spec is None or spec.loader is None:  # pragma: no cover
        raise SystemExit(f"cannot load demo at {DEMO_PATH}")
    module = importlib.util.module_from_spec(spec)
    sys.modules["_ct_demo_bundle"] = module
    spec.loader.exec_module(module)
    return module


def _sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as fh:
        for chunk in iter(lambda: fh.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", type=Path, default=None)
    ap.add_argument(
        "--crops",
        default=None,
        help="Comma-separated crop ids; default = every crop with a complete fit cache.",
    )
    args = ap.parse_args()

    demo = _load_demo()
    out_dir = args.out or (REPO_ROOT / "delme" / "cell_tracking_bundle")

    if args.crops:
        crops = [c.strip() for c in args.crops.split(",") if c.strip()]
    else:
        crops = [
            c
            for c in demo.DATASETS
            if all(demo.is_fit_cached(c, t) for t in range(demo.N_TIMEPOINTS))
        ]
    if not crops:
        raise SystemExit(
            "No crop has a complete fit cache. Run the demo first (it fits and "
            f"caches under {demo.FITS_DIR})."
        )

    # Always the full cadence: the hosted product is deliberately ONE set at all
    # 100 timepoints (the timelapse IS the demo's subject), and a decimated
    # variant cannot be made by dropping timepoints anyway — lineage edges join
    # ADJACENT frames, so keeping every Nth node discards every edge with it.
    timepoints = list(range(demo.N_TIMEPOINTS))
    print(f"{len(timepoints)} timepoints, crops={len(crops)} -> {out_dir}")

    files: list[dict] = []
    for i, name in enumerate(crops, 1):
        print(f"[{i}/{len(crops)}] {name}")
        image_store = demo.DATA_DIR / "train" / f"{name}.zarr"
        geff_store = demo.DATA_DIR / "train" / f"{name}.geff"
        centre = demo.crop_centre_um(image_store)

        # Load the cached fits, then run the demo's own pipeline so the hosted
        # product is byte-for-byte what a local build makes.
        from luxar.gsplats import GSplatData

        per_tp = [
            GSplatData.load(demo._fit_cache_path(name, t), include_stats=False)
            for t in timepoints
        ]
        combined = demo.combine_to_4d(per_tp, centre)
        lod = demo.build_lod(combined)
        intensity, offset = demo.display_window(lod.amplitudes)
        tracks = demo.track_geometry(geff_store, centre, demo.N_TIMEPOINTS)

        crop = {
            "name": name,
            "lod": lod,
            "n_splats": int(lod.n_splats),
            "intensity": intensity,
            "offset": offset,
            "tracks": tracks,
            "extent_um": float(2.0 * centre.max()),
        }
        written = demo.save_precomputed_crop(crop, out_dir, len(timepoints))
        for path in written:
            files.append(
                {
                    "name": path.name,
                    "sha256": _sha256(path),
                    "bytes": path.stat().st_size,
                }
            )
            print(f"    {path.name}  {path.stat().st_size / 1e6:.1f} MB")

    total = sum(f["bytes"] for f in files)
    print(f"\n{len(files)} files, {total / 1e9:.2f} GB total")
    listing = out_dir / "manifest_files.json"
    listing.write_text(json.dumps(files, indent=2) + "\n")
    print(f"manifest `files` block written to {listing}")
    print(
        "\nNext: upload these to the cc-by Zenodo record, then replace the "
        "`gsplats_cell_tracking.files` list in "
        "packages/luxar/src/luxar/demos/data_manifest.json. Hosted-only file "
        "lists live in the committed manifest because the generator has no "
        "in-repo directory to scan.\n"
        "NOTE: the demo can only FETCH them once that record is PUBLISHED — an "
        "unpublished draft's files are not publicly downloadable, so the record's "
        "base_url stays null until then."
    )
    return 0


if __name__ == "__main__":
    sys.exit(main())
