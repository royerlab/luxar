#!/usr/bin/env python3
"""Data-Driven Demo: Single-Cell 3D Genome (Dip-C) — Chromosomes as 3D Polylines

Visualize the folded three-dimensional structure of a single human cell's
genome. Dip-C (diploid chromatin conformation capture; Tan et al. 2018, Science)
reconstructs the 3D coordinates of every ~20 kb bead along all 46 chromosomes of
one cell, with the maternal and paternal copies resolved separately. Each
chromosome copy coils through the nucleus as Lines; a non-displayed
``haplotype`` dimension lets you scrub between the maternal and paternal copies
(or overlay them for the diploid view) — the two independently folded genomes.

================================================================================
THE 3D GENOME
================================================================================

Inside every nucleus, ~2 meters of DNA folds into a compact 3D structure that
organizes genes into territories and regulatory neighborhoods. Dip-C measures
this fold in a *single* cell and outputs a ``.3dg`` file: for each chromosome
(haplotype-resolved), an ordered list of bead coordinates
``chrom  genomic_position  x  y  z`` at ~20 kb / ~100 nm resolution.

WHAT THIS DEMO SHOWS
--------------------
- The 23 chromosomes of each haplotype (chr1..22, X), colored by chromosome and
  packed into a single Lines node.
- A non-displayed categorical ``haplotype`` dimension (Maternal / Paternal):
  scrub it (select the dimension, then step) to isolate one genome copy; the
  viewer culls the off-slice copy per-scrub. Opens showing the maternal copy.
- Hover a strand to read its chromosome and genomic coordinate.

DATA SOURCE & CITATION
----------------------
Tan, L., Xing, D., Chang, C.-H., Li, H., & Xie, X. S. (2018).
    "Three-dimensional genome structures of single diploid human cells."
    Science, 361(6405), 924–928. DOI: 10.1126/science.aat5641
Data: GEO accession GSE117876 (``GSE117876_RAW.tar``), GM12878 / PBMC single
    cells. Dip-C tools & ``.3dg`` format: https://github.com/tanlongzhi/dip-c

SELF-CONTAINED / CACHING
------------------------
On a fresh machine this demo bootstraps itself with no manual steps:
  1. Fast path: a small precomputed ``.npz`` shipped via Git LFS
     (``demos/data/dipc_genome/``), derived from one GM12878 cell.
  2. If that asset isn't pulled, it AUTOMATICALLY downloads the GEO archive to
     ``~/.cache/luxar/dipc_genome/``, extracts one cell's ``.3dg``, builds the
     polylines, and caches the ``.npz`` there — so subsequent runs are instant.
``--recompute`` forces the download + rebuild path.

USAGE
-----
    python demo_dipc_3d_genome.py [--recompute] [--no-serve]

Controls:
    - Mouse drag: rotate,  Scroll: zoom
    - Press '4' to select the haplotype dimension, then '[' / ']' to scrub
      between the Maternal and Paternal genomes (or use the dimension slider)
"""

from __future__ import annotations

DEMO_META = {
    "key": "dipc_3d_genome",
    "title": "Single-Cell 3D Genome (Dip-C)",
    "description": "A single cell's folded 3D genome from Dip-C: chromosomes as haplotype-resolved lines.",
    "category": "genomics",
    "geometry": "lines",
    "requirements": {
        "download_mb": 4,
        "compute": "light",
        "gpu": "none",
        "local_data": "git-lfs",
    },
    "caches": [],
    "outputs": ["dipc_3d_genome"],
}

import tempfile
from pathlib import Path

import numpy as np
from arbol import Arbol, aprint, asection

from luxar import Dimension, Dimensions, LuxarZarrCompiler
from luxar.demos import launch_viewer
from luxar.utils.demos import is_lfs_pointer, parse_demo_flags
from luxar.utils.paths import get_demos_output_dir

# -----------------------------------------------------------------------------
# Configuration
# -----------------------------------------------------------------------------

GEO_TAR_URL = (
    "https://ftp.ncbi.nlm.nih.gov/geo/series/GSE117nnn/GSE117876/suppl/"
    "GSE117876_RAW.tar"
)
EXPECTED_TAR_SIZE = 4_658_462_720  # bytes

DEMO_NAME = "dipc_genome"
NPZ_FILE = "dipc_gm12878.npz"

CACHE_DIR = Path.home() / ".cache" / "luxar" / DEMO_NAME
CACHE_NPZ = CACHE_DIR / NPZ_FILE
CACHE_TAR = CACHE_DIR / "GSE117876_RAW.tar"

# Shipped precomputed asset (Git LFS).
DATA_DIR = Path(__file__).resolve().parent / "data" / DEMO_NAME
LFS_NPZ = DATA_DIR / NPZ_FILE

# Standard chromosome ordering for coloring (autosomes + sex chromosomes).
CHROM_ORDER = [str(i) for i in range(1, 23)] + ["X", "Y"]

# A chromosome arm shorter than this many beads is dropped (fragmentary).
MIN_BEADS_PER_ARM = 4

Arbol.max_depth = 3


# -----------------------------------------------------------------------------
# .3dg parsing + geometry (pure, unit-testable)
# -----------------------------------------------------------------------------


def _split_chrom_haplotype(raw: str) -> tuple[str, int]:
    """Parse a Dip-C chromosome token into (base_name, haplotype).

    Handles ``1(mat)`` / ``chrX(pat)`` (haplotype in parentheses) and the
    ``chr1a`` / ``chr1b`` convention. Returns haplotype 0 for maternal/'a',
    1 for paternal/'b'.
    """
    name = raw.strip()
    if name.lower().startswith("chr"):
        name = name[3:]
    hap = 0
    if "(" in name:
        base, _, tag = name.partition("(")
        tag = tag.rstrip(")").lower()
        hap = 1 if tag.startswith("pat") else 0
        name = base
    elif (
        len(name) >= 2
        and name[-1] in ("a", "b")
        and (name[:-1][-1].isdigit() or name[:-1] in ("X", "Y"))
    ):
        hap = 0 if name[-1] == "a" else 1
        name = name[:-1]
    return name, hap


def parse_3dg(lines: list[str]) -> dict[tuple[str, int], np.ndarray]:
    """Parse ``.3dg`` text lines into per-(chrom, haplotype) bead tables.

    Each value is an ``(n, 4)`` float64 array of columns
    ``[genomic_position, x, y, z]`` sorted by genomic position.
    """
    rows: dict[tuple[str, int], list[tuple[float, float, float, float]]] = {}
    for line in lines:
        line = line.strip()
        if not line or line.startswith("#"):
            continue
        parts = line.split()
        if len(parts) < 5:
            continue
        chrom_raw, pos, x, y, z = parts[0], parts[1], parts[2], parts[3], parts[4]
        try:
            rec = (float(pos), float(x), float(y), float(z))
        except ValueError:
            continue
        base, hap = _split_chrom_haplotype(chrom_raw)
        rows.setdefault((base, hap), []).append(rec)

    out: dict[tuple[str, int], np.ndarray] = {}
    for key, recs in rows.items():
        arr = np.array(recs, dtype=np.float64)
        arr = arr[np.argsort(arr[:, 0], kind="stable")]  # order beads along the arm
        out[key] = arr
    return out


def chromosome_color(chrom: str) -> np.ndarray:
    """Deterministic RGB (float32, [0,1]) for a chromosome, by karyotype order."""
    try:
        idx = CHROM_ORDER.index(chrom)
    except ValueError:
        idx = len(CHROM_ORDER)
    n = len(CHROM_ORDER) + 1
    h = (idx / n) % 1.0
    # HSV → RGB with S=0.65, V=1.0 (readable, distinct hues).
    s, v = 0.65, 1.0
    h6 = h * 6.0
    c = v * s
    x = c * (1.0 - abs(h6 % 2.0 - 1.0))
    m = v - c
    r, g, b = [
        (c, x, 0.0),
        (x, c, 0.0),
        (0.0, c, x),
        (0.0, x, c),
        (x, 0.0, c),
        (c, 0.0, x),
    ][int(h6) % 6]
    return np.array([r + m, g + m, b + m], dtype=np.float32)


def _normalize_coords(polylines: list[dict]) -> None:
    """Center all beads at their common centroid and scale to a unit-ish box.

    Mutates each polyline's ``vertices`` in place so both haplotypes share one
    coordinate frame (Dip-C coordinates are arbitrary units).
    """
    if not polylines:
        return
    allv = np.concatenate([p["vertices"] for p in polylines], axis=0)
    center = allv.mean(axis=0)
    scale = float(np.percentile(np.linalg.norm(allv - center, axis=1), 99)) or 1.0
    for p in polylines:
        p["vertices"] = ((p["vertices"] - center) / scale).astype(np.float32)


def build_polylines(parsed: dict[tuple[str, int], np.ndarray]) -> list[dict]:
    """Convert parsed bead tables into renderable polyline records.

    Each record: ``{chrom, haplotype, vertices(n,3) float32, positions(n,) int64,
    color(3,) float32}``. Arms shorter than ``MIN_BEADS_PER_ARM`` are dropped.
    """
    polylines: list[dict] = []
    for (chrom, hap), arr in parsed.items():
        if arr.shape[0] < MIN_BEADS_PER_ARM:
            continue
        polylines.append(
            {
                "chrom": chrom,
                "haplotype": int(hap),
                "vertices": arr[:, 1:4].astype(np.float32),
                "positions": arr[:, 0].astype(np.int64),
                "color": chromosome_color(chrom),
            }
        )
    _normalize_coords(polylines)
    # Stable ordering: by karyotype index then haplotype.
    polylines.sort(
        key=lambda p: (
            CHROM_ORDER.index(p["chrom"]) if p["chrom"] in CHROM_ORDER else 99,
            p["haplotype"],
        )
    )
    return polylines


# -----------------------------------------------------------------------------
# Precomputed-asset (de)serialization
# -----------------------------------------------------------------------------


def save_polylines_npz(polylines: list[dict], path: Path) -> None:
    """Serialize polylines to a compact ``.npz`` (concatenated + offsets)."""
    vertices = np.concatenate([p["vertices"] for p in polylines], axis=0).astype(
        np.float32
    )
    positions = np.concatenate([p["positions"] for p in polylines], axis=0).astype(
        np.int64
    )
    offsets = np.array(
        [0] + [len(p["vertices"]) for p in polylines], dtype=np.int64
    ).cumsum()
    chroms = np.array([p["chrom"] for p in polylines])
    haplo = np.array([p["haplotype"] for p in polylines], dtype=np.int8)
    path.parent.mkdir(parents=True, exist_ok=True)
    # Write through a file handle: np.savez appends ".npz" to any path that
    # doesn't already end in it, which would break a ".part" temp name.
    tmp = path.parent / (path.name + ".part")
    with open(tmp, "wb") as fh:
        np.savez_compressed(
            fh,
            vertices=vertices,
            positions=positions,
            offsets=offsets,
            chroms=chroms,
            haplo=haplo,
        )
    tmp.rename(path)


def load_polylines_npz(path: Path) -> list[dict]:
    """Inverse of :func:`save_polylines_npz`."""
    with np.load(path, allow_pickle=False) as d:
        vertices, positions, offsets = d["vertices"], d["positions"], d["offsets"]
        chroms, haplo = d["chroms"], d["haplo"]
    polylines = []
    for k in range(len(offsets) - 1):
        a, b = int(offsets[k]), int(offsets[k + 1])
        chrom = str(chroms[k])
        polylines.append(
            {
                "chrom": chrom,
                "haplotype": int(haplo[k]),
                "vertices": vertices[a:b],
                "positions": positions[a:b],
                "color": chromosome_color(chrom),
            }
        )
    return polylines


# -----------------------------------------------------------------------------
# Download + extract (network; --recompute / fresh-system fallback)
# -----------------------------------------------------------------------------


def _extract_one_3dg(tar_path: Path) -> list[str]:
    """Extract the text lines of one ``.3dg`` cell from the GEO tar archive."""
    import gzip
    import tarfile

    with asection("Extracting a .3dg structure from the GEO archive"):
        with tarfile.open(tar_path, "r") as tf:
            members = tf.getmembers()
            candidates = [m for m in members if m.isfile() and ".3dg" in m.name.lower()]
            if not candidates:
                raise FileNotFoundError(
                    f"No .3dg member found in {tar_path.name}. "
                    f"Available (first 5): {[m.name for m in members[:5]]}"
                )
            # Prefer a GM12878 cell, then the dip-c "clean" final structure
            # (outlier beads removed); otherwise take the largest .3dg.
            gm = [m for m in candidates if "gm12878" in m.name.lower()]
            pool = gm or candidates
            clean = [m for m in pool if "clean" in m.name.lower()]
            pool = clean or pool
            chosen = max(pool, key=lambda m: m.size)
            aprint(f"Chosen cell: {chosen.name} ({chosen.size / 1e6:.1f} MB)")
            fobj = tf.extractfile(chosen)
            if fobj is None:
                raise FileNotFoundError(f"Could not read {chosen.name}")
            raw = fobj.read()
    if chosen.name.lower().endswith(".gz"):
        raw = gzip.decompress(raw)
    return raw.decode("utf-8", errors="replace").splitlines()


def build_from_geo() -> list[dict]:
    """Download the GEO archive, extract one cell, build + cache the polylines."""
    from luxar.utils.download import robust_download

    with asection("Downloading Dip-C data from GEO (GSE117876, 4.7 GB)"):
        CACHE_DIR.mkdir(parents=True, exist_ok=True)
        robust_download(
            GEO_TAR_URL,
            CACHE_TAR,
            max_retries=5,
            timeout=1800,
            expected_size=EXPECTED_TAR_SIZE,
        )
    lines = _extract_one_3dg(CACHE_TAR)
    with asection("Building chromosome polylines"):
        polylines = build_polylines(parse_3dg(lines))
        aprint(f"✓ Built {len(polylines)} chromosome polylines")
        save_polylines_npz(polylines, CACHE_NPZ)
        aprint(f"✓ Cached processed structure to {CACHE_NPZ.name}")
    return polylines


def load_or_build_polylines(recompute: bool) -> list[dict]:
    """Return chromosome polylines, self-contained on a fresh system.

    Resolution order: local processed cache → shipped LFS ``.npz`` → download +
    extract + build (auto, no manual ``git lfs pull``). ``recompute`` forces the
    last path.
    """
    if not recompute:
        if CACHE_NPZ.exists():
            aprint(f"  Using cached structure ({CACHE_NPZ.name})")
            return load_polylines_npz(CACHE_NPZ)
        if LFS_NPZ.exists() and not is_lfs_pointer(LFS_NPZ):
            aprint("  Copying shipped structure from package data to cache")
            CACHE_DIR.mkdir(parents=True, exist_ok=True)
            import shutil

            shutil.copy2(LFS_NPZ, CACHE_NPZ)
            return load_polylines_npz(CACHE_NPZ)
        aprint(
            "Precomputed structure not available (Git LFS asset not pulled). "
            "Falling back to download + build (one-time; result is cached)."
        )
    return build_from_geo()


# -----------------------------------------------------------------------------
# Scene construction
# -----------------------------------------------------------------------------

HAPLOTYPE_NAMES = ["Maternal", "Paternal"]


def _position_gradient(base: np.ndarray, n: int) -> np.ndarray:
    """Per-vertex color: the chromosome hue brightening along the arm."""
    t = np.linspace(0.45, 1.0, n, dtype=np.float32)[:, None]
    return (base[None, :] * t).astype(np.float32)


def _haplotype_geometry(
    polys: list[dict],
    hap_slot: int,
) -> tuple[np.ndarray, np.ndarray, list[str], np.ndarray]:
    """Pack one haplotype's chromosome polylines into indexed line geometry.

    Each bead is authored ONCE (unique per-vertex arrays) and connectivity is an
    explicit ``(E, 2)`` edge list — one edge per consecutive bead pair within a
    chromosome arm, no edge across arms. Sharing the joint vertex INDEX between
    the two edges that meet at a bead is what lets the viewer recognise the
    joint and suppress its end caps, so a thick chromosome reads as one
    continuous tube; duplicating the joint into two independent segment
    endpoints (the old ``segments`` authoring) hides the joint from that test
    and beads the curve.

    Every vertex is 4D: the first three columns are the bead's ``x, y, z`` and
    the fourth is ``hap_slot`` — the categorical ``haplotype`` coordinate. It is
    constant along an arm, so both endpoints of every edge share the same slot
    and an edge is wholly in- or out-of-slice when the viewer scrubs the
    non-displayed haplotype dimension. Returns ``(vertices(M,4), colors(M,3),
    labels[M], edges(E,2))`` with edge indices local to the returned vertices.
    """
    vparts: list[np.ndarray] = []
    cparts: list[np.ndarray] = []
    labels: list[str] = []
    eparts: list[np.ndarray] = []
    offset = 0
    for p in polys:
        v = p["vertices"]
        n = len(v)
        if n < 2:
            continue
        vert = np.empty((n, 4), dtype=np.float32)
        vert[:, :3] = v
        vert[:, 3] = hap_slot  # categorical haplotype coordinate
        # This arm's chain: (0,1), (1,2), ... offset into the growing block.
        start = np.arange(offset, offset + n - 1, dtype=np.uint32)
        eparts.append(np.column_stack([start, start + 1]))
        offset += n
        pos_mb = p["positions"] / 1_000_000.0
        vparts.append(vert)
        cparts.append(_position_gradient(p["color"], n))
        labels.extend(
            f"chr{p['chrom']}:{mb:.1f} Mb ({HAPLOTYPE_NAMES[p['haplotype']]})"
            for mb in pos_mb
        )
    return (
        np.concatenate(vparts),
        np.concatenate(cparts),
        labels,
        np.concatenate(eparts),
    )


def build_scene(output_path: Path, polylines: list[dict]) -> int:
    """Write the Dip-C genome scene. Returns the total bead (vertex) count.

    The maternal and paternal genomes share ONE Lines node, distinguished by a
    non-displayed categorical ``haplotype`` dimension (each vertex carries its
    haplotype as a 4th coordinate). Scrubbing that dimension in the viewer
    isolates one copy or, at the extremes, shows the diploid overlay — the
    viewer culls the off-slice haplotype per-scrub (fixed in #493; before that
    Lines couldn't be sliced by a non-displayed dimension, so the demo used two
    hard-toggled Layers instead).
    """
    with asection("Building 3D genome scene"):
        dims = Dimensions(
            [
                Dimension("x", unit="", display=True),
                Dimension("y", unit="", display=True),
                Dimension("z", unit="", display=True),
                Dimension(
                    "haplotype",
                    display=False,
                    categories=list(HAPLOTYPE_NAMES),
                ),
            ]
        )
        total = 0
        with LuxarZarrCompiler(output_path) as compiler:
            scene = compiler.create_scene(dimensions=dims)
            scene.attrs["title"] = "Dip-C: Single-Cell 3D Genome"

            vparts: list[np.ndarray] = []
            cparts: list[np.ndarray] = []
            labels: list[str] = []
            eparts: list[np.ndarray] = []
            vertex_offset = 0
            for hap in range(len(HAPLOTYPE_NAMES)):
                polys = [p for p in polylines if p["haplotype"] == hap]
                if not polys:
                    continue
                verts, colors, labs, edges = _haplotype_geometry(polys, hap)
                vparts.append(verts)
                cparts.append(colors)
                labels.extend(labs)
                # Shift this haplotype's edge indices into the concatenated
                # vertex block so both genome copies batch into ONE node.
                eparts.append((edges + vertex_offset).astype(np.uint32))
                vertex_offset += len(verts)

            if vparts:
                all_verts = np.concatenate(vparts)
                all_colors = np.concatenate(cparts)
                all_edges = np.concatenate(eparts)
                # One Lines node sliced by the non-displayed `haplotype` dim.
                # Indexed authoring: unique per-bead vertices + an explicit
                # edge list, so interior joints share their vertex index and
                # the viewer draws continuous chromosome tubes (see
                # `_haplotype_geometry`).
                # extend_to_all=[] is explicit: the genome copies live at their
                # own haplotype coordinate and must be culled off-slice, NOT
                # broadcast to every slice.
                scene.add_lines(
                    "genome",
                    vertices=all_verts,
                    widths=0.006,
                    colors=all_colors,
                    labels=labels,
                    indices=all_edges,
                    line_type="indexed",
                    sharpness=0.5,
                    opacity=0.95,
                    intensity=0.6,
                    blending_mode="luminous",
                    extend_to_all=[],
                )
                total = len(all_verts)

            scene.add_text(
                "Single-Cell 3D Genome (Dip-C)",
                position=(0.02, 0.02),
                font_size=0.045,
                anchor="top-left",
                color="rgba(255,255,255,0.65)",
                blend_mode="difference",
            )
            scene.add_text(
                "Tan et al. 2018 • chromosomes as 3D polylines",
                position=(0.98, 0.97),
                font_size=0.015,
                anchor="bottom-right",
                color="rgba(200,200,200,0.5)",
            )
            scene.add_text(
                "press '4' then '[' / ']' — scrub Maternal ⇄ Paternal",
                position=(0.02, 0.97),
                font_size=0.014,
                anchor="bottom-left",
                color="rgba(200,200,200,0.45)",
            )
        aprint(f"  ✓ {len(polylines)} polylines, {total:,} beads")
        return total


# -----------------------------------------------------------------------------
# Entry point
# -----------------------------------------------------------------------------


def main() -> None:
    flags = parse_demo_flags()
    recompute = flags["recompute"]
    no_serve = flags["no_serve"]

    aprint("=" * 70)
    aprint("SINGLE-CELL 3D GENOME (Dip-C)")
    aprint("=" * 70)
    aprint("Chromosomes as folded 3D polylines, maternal vs paternal.")
    aprint("")

    polylines = load_or_build_polylines(recompute)

    if no_serve:
        output_path = get_demos_output_dir() / "dipc_3d_genome.luxar.zarr"
        n = build_scene(output_path, polylines)
        aprint(f"Dataset generated at {output_path} ({n:,} beads)")
        return

    with tempfile.TemporaryDirectory(prefix="luxar_demo_dipc_") as tmpdir:
        output_path = Path(tmpdir) / "dipc_3d_genome.luxar.zarr"
        build_scene(output_path, polylines)
        aprint("")
        aprint("Data credit: Tan et al. 2018, Science 361:924 (GEO GSE117876)")
        launch_viewer(output_path)

    aprint("Cleanup complete")


if __name__ == "__main__":
    main()
