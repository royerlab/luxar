#!/usr/bin/env python3
"""Build the uploadable artifacts for the ``gsplats_cell_tracking`` dataset.

The cell-tracking demo's raw source sits behind an authenticated Kaggle
competition endpoint, so the redistributable thing is the **derived product**:
per crop, the finished 4D Gaussian-splat volume (LOD ladder included) plus the
track geometry. With those hosted, the demo runs with no credentials and no GPU.

This reads the demo's own per-timepoint fit cache, runs the demo's own combine →
LOD → tracks pipeline, and writes one pair of files per crop, plus the
``files`` block to paste into ``scripts/gen_data_manifest.py`` (name + sha256 +
bytes, which is what makes a fetched copy verifiable).

    # every crop the fit cache holds, at the manifest's `light` cadence
    python scripts/build_cell_tracking_bundle.py --variant light

    # the full 100-timepoint variant
    python scripts/build_cell_tracking_bundle.py --variant full

Output lands in ``--out`` (default ``delme/cell_tracking_bundle/<variant>/``),
which is deliberately NOT under ``demos/data/``: at ~2 GB these files are for
Zenodo, and the in-repo tree is what R17 exists to empty.
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

# Timepoint stride per manifest variant. `light` keeps the 3x3 matrix and the
# animation while quartering the download; `full` is every timepoint.
VARIANT_STRIDE = {"light": 4, "full": 1}


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
    ap.add_argument("--variant", choices=sorted(VARIANT_STRIDE), default="light")
    ap.add_argument("--out", type=Path, default=None)
    ap.add_argument(
        "--crops",
        default=None,
        help="Comma-separated crop ids; default = every crop with a complete fit cache.",
    )
    args = ap.parse_args()

    demo = _load_demo()
    stride = VARIANT_STRIDE[args.variant]
    out_dir = args.out or (REPO_ROOT / "delme" / "cell_tracking_bundle" / args.variant)

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

    timepoints = list(range(0, demo.N_TIMEPOINTS, stride))
    print(
        f"variant={args.variant} stride={stride} "
        f"({len(timepoints)} of {demo.N_TIMEPOINTS} timepoints) "
        f"crops={len(crops)} -> {out_dir}"
    )

    files: list[dict] = []
    for i, name in enumerate(crops, 1):
        print(f"[{i}/{len(crops)}] {name}")
        image_store = demo.DATA_DIR / "train" / f"{name}.zarr"
        geff_store = demo.DATA_DIR / "train" / f"{name}.geff"
        centre = demo.crop_centre_um(image_store)

        # Load the cached fits at this variant's cadence, then run the demo's own
        # pipeline so the hosted product is byte-for-byte what a local build makes.
        from luxar.gsplats import GSplatData

        per_tp = [
            GSplatData.load(demo._fit_cache_path(name, t), include_stats=False)
            for t in timepoints
        ]
        combined = demo.combine_to_4d(per_tp, centre)
        lod = demo.build_lod(combined)
        intensity, offset = demo.display_window(lod.amplitudes)
        tracks = demo.track_geometry(geff_store, centre, demo.N_TIMEPOINTS)
        if tracks is not None and stride > 1:
            tracks = _restride_tracks(tracks, stride)

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
        "\nNext: upload these to the Zenodo record, then paste the block into the "
        f"`{args.variant}` variant of `gsplats_cell_tracking` in "
        "scripts/gen_data_manifest.py and re-run it."
    )
    return 0


def _restride_tracks(tracks: dict, stride: int) -> dict:
    """Keep only the timepoints this variant ships, and renumber them densely.

    The time column is a scene DIMENSION index, so dropping every Nth timepoint
    without renumbering would leave the markers pointing at slices that no longer
    exist. Line vertices are NOT dropped (that would invalidate the edge indices);
    only edges whose endpoints both survive are kept.
    """
    import numpy as np

    out = dict(tracks)

    verts = np.asarray(tracks["line_vertices"]).copy()
    t_col = verts.shape[1] - 1
    keep_vertex = (verts[:, t_col].astype(np.int64) % stride) == 0
    verts[:, t_col] = verts[:, t_col] // stride
    out["line_vertices"] = verts

    edges = np.asarray(tracks["line_indices"])
    if len(edges):
        both = keep_vertex[edges[:, 0]] & keep_vertex[edges[:, 1]]
        out["line_indices"] = edges[both]

    pts = np.asarray(tracks["point_positions"]).copy()
    keep_point = (pts[:, t_col].astype(np.int64) % stride) == 0
    pts = pts[keep_point]
    pts[:, t_col] = pts[:, t_col] // stride
    out["point_positions"] = pts
    out["point_colors"] = np.asarray(tracks["point_colors"])[keep_point]
    out["point_radii"] = np.asarray(tracks["point_radii"])[keep_point]
    out["n_cells"] = int(keep_point.sum())
    return out


if __name__ == "__main__":
    sys.exit(main())
