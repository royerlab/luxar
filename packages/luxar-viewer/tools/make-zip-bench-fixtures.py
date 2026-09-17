#!/usr/bin/env python3
"""Build the three artifacts the zipped-store benchmark compares.

One scene, packaged three ways:

* ``bench.luxar.zarr``               — a plain directory store (the baseline)
* ``bench-stored.luxar.zarr.zip``    — flat, ``ZIP_STORED``
* ``bench-deflate.luxar.zarr.zip``   — flat, ``ZIP_DEFLATED``

Both compressions are built because both are in circulation and they stress
different things: ``optimize.py``'s packaging step writes ``ZIP_STORED``, while
the shipped demo archives (e.g. ``desi_dr1_cosmic_web.luxar.zarr.zip``) are
100% DEFLATE. DEFLATE matters disproportionately here because ``unzipit`` ships
``useWorkers: false``, so every member it inflates runs on the MAIN THREAD —
a cost that shows up as jank rather than as wall-clock latency.

Members are keyed STORE-RELATIVE (flat), which is what `ZipFileStore` expects
and what `zarr.storage.ZipStore` writes. Mirrors `io/optimize.py::_package`.

Usage::

    hatch run python packages/luxar-viewer/tools/make-zip-bench-fixtures.py <out-dir> [--points N]
"""

from __future__ import annotations

import argparse
import shutil
import zipfile
from pathlib import Path

import numpy as np

from luxar import Dimensions, LuxarZarrCompiler


def build_scene(out_dir: Path, points: int, nodes: int) -> Path:
    """Compile a directory scene with enough MEMBERS for the archive's central
    directory to be a measurable cost rather than a rounding error.

    Deliberately many small nodes rather than a few large ones: member count is
    what the central-directory preamble scales with, while total point count is
    what the (software-rendered, headless) draw cost scales with. Keeping the
    first high and the second low is what makes the run finish in minutes
    instead of timing out."""
    store = out_dir / "bench.luxar.zarr"
    if store.exists():
        shutil.rmtree(store)

    rng = np.random.default_rng(0xC0FFEE)
    with LuxarZarrCompiler(store) as compiler:
        scene = compiler.create_scene(dimensions=Dimensions.default_3d())
        for index in range(nodes):
            positions = rng.standard_normal((points, 3)).astype(np.float32)
            colors = rng.random((points, 3)).astype(np.float32)
            scene.add_points(f"cloud_{index:02d}", positions, colors=colors)
    return store


def package(store: Path, artifact: Path, compression: int) -> None:
    """Zip ``store`` with members keyed store-relative (flat)."""
    artifact.unlink(missing_ok=True)
    with zipfile.ZipFile(artifact, "w", compression) as archive:
        for member in sorted(store.rglob("*")):
            if member.is_file():
                archive.write(member, member.relative_to(store).as_posix())


def describe(artifact: Path) -> str:
    with zipfile.ZipFile(artifact) as archive:
        infos = archive.infolist()
        # Central-directory size: the fixed preamble a reader downloads BEFORE
        # it can resolve a single chunk. 46 bytes of fixed header per record.
        central_directory = sum(
            46 + len(i.filename) + len(i.extra) + len(i.comment) for i in infos
        )
        duplicates = len(infos) - len({i.filename for i in infos})
    return (
        f"{artifact.name}: {artifact.stat().st_size / 1e6:.2f} MB, "
        f"{len(infos)} members, central directory ~{central_directory / 1024:.1f} kB, "
        f"duplicate members: {duplicates}"
    )


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("out_dir", type=Path)
    parser.add_argument("--points", type=int, default=4_000)
    parser.add_argument("--nodes", type=int, default=120)
    args = parser.parse_args()

    args.out_dir.mkdir(parents=True, exist_ok=True)
    store = build_scene(args.out_dir, args.points, args.nodes)

    stored = args.out_dir / "bench-stored.luxar.zarr.zip"
    deflate = args.out_dir / "bench-deflate.luxar.zarr.zip"
    package(store, stored, zipfile.ZIP_STORED)
    package(store, deflate, zipfile.ZIP_DEFLATED)

    files = sum(1 for p in store.rglob("*") if p.is_file())
    total = sum(p.stat().st_size for p in store.rglob("*") if p.is_file())
    print(f"{store.name}: {total / 1e6:.2f} MB across {files} files")
    print(describe(stored))
    print(describe(deflate))


if __name__ == "__main__":
    main()
