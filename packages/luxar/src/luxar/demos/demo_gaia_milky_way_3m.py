#!/usr/bin/env python3
"""Real Milky Way Stars from Gaia DR3 - 3 Million Star Dataset

DATASET SPECIFICATIONS:
=======================

Star Count & Selection:
- 3,000,000 stars (top 3M brightest from Gaia DR3)
- Selection query criteria:
  * parallax > 0.1 mas  → distances up to ~10 kpc from Sun
  * parallax_over_error > 5  → high-precision measurements only
  * Has BP and RP photometry → color information available
  * ORDER BY phot_g_mean_mag ASC → sorted by brightness

Physical Scale & Extent:
- Distance from Sun: 1.3 to ~10,000 pc (0.0013 to 10 kpc)
- Galactocentric extent: ~20 kpc across (X: -18 to +2 kpc)
- Cutoff radius: 30 kpc from Galactic Center
- Coordinate system: Galactocentric (origin at GC, Sun at -8.122 kpc X)

Photometry & Colors:
- Magnitude: G = 1.94 to 11.99
  * Brightest: mag 1.94 (Sirius-like, naked-eye visible)
  * Faintest: mag 11.99 (requires telescope)
- Colors: Real BP-RP color index from Gaia photometry
  * BP-RP → RGB conversion for stellar temperature visualization
  * Blue: Hot stars, Red: Cool stars

Visualization Parameters:
- Coordinate scaling: 10x multiplier (easier navigation)
- Point radii: 0.01 to 0.11 (scaled units), magnitude-dependent
- Reference markers:
  * Sun: Yellow, radius 0.35 (10x typical star)
  * Betelgeuse: Red, radius 0.35, at 168 pc
  * Rigel: Blue, radius 0.35, at 265 pc

Data Source & Attribution:
    ESA/Gaia/DPAC - Gaia Data Release 3 (2022)

    Mission: https://www.cosmos.esa.int/gaia
    Archive: https://gea.esac.esa.int/archive/

    Citation:
    Gaia Collaboration, Vallenari et al. (2023)
    "Gaia Data Release 3: Summary of the content and survey properties"
    Astronomy & Astrophysics, 674, A1
    DOI: 10.1051/0004-6361/202243940
    https://doi.org/10.1051/0004-6361/202243940

Data Generation Method:
    The raw data is built by luxar.demos._gaia_catalog. The compatibility CLI
    remains available as scripts/generate_galaxy_simple.py in a source checkout.

    Query executed via ESA Gaia Archive TAP service:
    ```sql
    SELECT TOP 3000000
      source_id, ra, dec, parallax, parallax_over_error,
      phot_g_mean_mag, phot_bp_mean_mag, phot_rp_mean_mag, bp_rp
    FROM gaiadr3.gaia_source
    WHERE parallax > 0.1
      AND parallax_over_error > 5
      AND phot_bp_mean_mag IS NOT NULL
      AND phot_rp_mean_mag IS NOT NULL
    ORDER BY phot_g_mean_mag ASC
    ```

    Post-processing:
    1. Distance computed from parallax: d[pc] = 1000 / parallax[mas]
    2. Transformed to Galactocentric coordinates using Astropy
       with galcen_distance = 8.122 kpc (GRAVITY Collaboration 2018)
    3. Filtered to stars within 30 kpc of Galactic Center
    4. Stored as raw zarr table (x_kpc, y_kpc, z_kpc, phot_g_mean_mag, bp_rp)

    References:
    - Galactocentric distance: GRAVITY Collaboration (2018), A&A 615, L15
      https://doi.org/10.1051/0004-6361/201833718

Scientific Context:
    The Gaia space telescope has measured positions, motions, and colors of
    ~1.8 billion stars in our galaxy. This demo shows the 3M brightest stars
    with high-quality distance measurements, transformed to a Galactocentric
    reference frame centered on the Galactic Center.

    Colors represent stellar temperature:
    - Blue: Hot, young stars (spectral type O, B, A)
    - White/Yellow: Sun-like stars (spectral type F, G)
    - Red/Orange: Cool, old stars (spectral type K, M)

    The coordinate system places you at approximately (-8.122, 0, 0) kpc from
    the Galactic Center - the location of our Sun!

Reference Markers:
    Three famous stars are marked for orientation:
    - Sun: Our home star at (-8.122, 0, 0) kpc
    - Betelgeuse: Red supergiant in Orion, ~168 pc from Sun
    - Rigel: Blue supergiant in Orion, ~265 pc from Sun

Usage:
    python demo_gaia_milky_way_3m.py
    python demo_gaia_milky_way_3m.py --build-catalog
    python demo_gaia_milky_way_3m.py --recompute

Controls:
    - Mouse drag: Rotate view
    - Mouse wheel: Zoom in/out
    - Ctrl+C: Stop and cleanup

Viewing Tips:
    - Start zoomed out to see the overall structure
    - Notice the thin disk of the Milky Way
    - Dense concentration toward the Galactic Center
    - Zoom in to see individual stars
    - Colors reflect real stellar temperatures!
"""

DEMO_META = {
    "key": "gaia_milky_way",
    "title": "Gaia Milky Way (3M stars)",
    "description": "Real Milky Way stars from Gaia DR3 (3M brightest) in galactocentric coordinates.",
    "category": "astronomy",
    "geometry": "points",
    "requirements": {
        # The opt-in first build queries the Gaia archive; cached runs are local.
        "download_mb": 500,
        "compute": "medium",
        "gpu": "none",
        # The catalog is CC BY-NC, so it is not shipped in-tree. The explicit
        # build/prompt keeps unattended `run-all` from starting a 90-minute TAP
        # query, so the registry still classifies it as opt-in local data.
        "local_data": "manual-file",
    },
    # Claiming the cache namespace is what ATTRIBUTES that directory to this
    # demo: `luxar demo` reports the demo as `cached`, and `demo cache list`
    # names this key against the bytes. It is deliberately NOT what protects the
    # hand-placed catalog from deletion — `registry.PROTECTED_INPUT_DIRS` is,
    # independently of DEMO_META (see the comment on it), so `demo cache clear`
    # spares the directory by key, under `--all` and under `--orphans` alike, and
    # no edit here can quietly disarm that. The two are separate on purpose:
    # declaring the name buys reporting, and only reporting.
    "caches": ["milky_way_gaia_3m"],
    "outputs": ["gaia_milky_way"],
    "citation": {
        "short": "Gaia Collaboration et al. 2023",
        "doi": "10.1051/0004-6361/202243940",
    },
}

import sys
import tempfile
from pathlib import Path
from typing import NoReturn

import numpy as np
from arbol import aprint, asection

from luxar import (
    CameraConfig,
    Dimension,
    Dimensions,
    LuxarZarrCompiler,
    ViewerConfig,
)
from luxar._zarr_compat import open_group
from luxar.demos import add_demo_caption, launch_viewer, substitutive_lod_or_flat
from luxar.demos._cinematic_camera import CINEMATIC_FOV_DEG
from luxar.demos._gaia_catalog import DEFAULT_CATALOG_FILE, RAW_ZARR_NAME
from luxar.utils.paths import get_demos_output_dir

# The Gaia catalog is CC BY-NC 3.0 IGO. NonCommercial survives derivation, so it
# applies to this point cloud too and is incompatible with a BSD-3 repository —
# the file is therefore NOT shipped, and is read from the local cache when a
# machine happens to have one. The opt-in builder queries the ESA archive and
# caches the result locally; nothing is fetched at import time.
SCRIPT_DIR = Path(__file__).parent
CACHE_FILE = DEFAULT_CATALOG_FILE
#: Legacy in-repo location, kept in the search order so a checkout that still
#: has the file (or a user who restores it by hand) keeps working.
REPO_FILE = SCRIPT_DIR / "data" / "milky_way_gaia_3m.zarr.zip"
#: The one top-level directory the catalog zip must contain. Derived from
#: CACHE_FILE rather than restated, because what names that member is exactly the
#: `--output` stem the rebuild command is given (= the cache file without .zip).
#: The raw table columns `load_and_convert_gaia_data` reads. Checked once at
#: extraction so a store that is not this table says which columns are missing,
#: instead of the converter dying on a bare `KeyError` half-way through the read.
RAW_TABLE_FIELDS = ("x_kpc", "y_kpc", "z_kpc", "phot_g_mean_mag", "bp_rp")
REBUILD_COMMAND = "  luxar demo run gaia_milky_way -- --build-catalog"
#: What to run when a catalog IS present but unusable. NOT `--build-catalog`:
#: the builder returns an already-cached zip untouched, so pointing a
#: wrong-stem/truncated/foreign-table copy at it is a no-op and the reader loops
#: on the same error. `--recompute` is the only spelling that replaces one.
REPLACE_COMMAND = "  luxar demo run gaia_milky_way -- --recompute"


class CatalogUnusable(FileNotFoundError):
    """The catalog is missing, or is not the one this demo can read.

    A distinct type, because the entry points below turn it into a terse
    ``❌ Error: …`` + exit 1 — right for a message that IS the advice, wrong for
    any other ``FileNotFoundError`` raised while converting or writing the scene,
    which is a bug and wants its traceback. Subclasses ``FileNotFoundError`` so
    callers that only care that the file is absent keep working.
    """


def resolve_data_file(
    *, build_catalog_requested: bool = False, recompute: bool = False
) -> Path:
    """The star catalog: local cache first, then the legacy in-repo copy.

    ``is_file()``, not ``exists()``: a *directory* at the catalog path is an easy
    way to end up here, because the load-bearing ``--output`` stem is the cache
    file minus ``.zip`` — hand the rebuild script the ``.zip`` path instead and it
    writes the raw zarr *directory* under that name (its zip lands beside it as
    ``.zarr.zarr.zip``). ``exists()`` accepts that, and ``zipfile`` then raises
    ``IsADirectoryError`` — an ``OSError``, not a ``FileNotFoundError``, so no
    handler on the way out catches it and the advice below never prints. Anything
    that is not a regular file (a directory, a broken symlink, a FIFO ``ZipFile``
    would block on) is simply not a candidate.
    """
    if recompute:
        from luxar.demos._gaia_catalog import build_catalog

        return build_catalog(cache_dir=CACHE_FILE.parent, recompute=True)

    for candidate in (CACHE_FILE, REPO_FILE):
        if candidate.is_file():
            return candidate

    should_build = build_catalog_requested
    stdin_is_tty = bool(
        sys.stdin is not None and getattr(sys.stdin, "isatty", lambda: False)()
    )
    if not should_build and stdin_is_tty:
        try:
            answer = input(
                "The Gaia catalog is not cached. Build it now from the ESA archive "
                "(~90 minutes for 3M stars)? [y/N] "
            )
        except EOFError:
            answer = ""
        should_build = answer.strip().lower() in {"y", "yes"}
    if should_build:
        from luxar.demos._gaia_catalog import build_catalog

        return build_catalog(cache_dir=CACHE_FILE.parent)

    raise CatalogUnusable(
        "Gaia star catalog not found.\n\n"
        "This dataset is CC BY-NC 3.0 IGO (NonCommercial), which the derived "
        "point cloud inherits, so it is deliberately not distributed with "
        "Luxar.\n"
        f"Put a copy of the catalog at exactly {CACHE_FILE} (that full path, "
        "filename included), or build it from the ESA Gaia archive with\n"
        f"{REBUILD_COMMAND}\n"
        "Install the demo dependencies first with `luxar demo deps --install`; "
        "the 3M-star query and CPU transform take about 90 minutes. Use "
        "`--recompute` instead of `--build-catalog` to replace a cached copy.\n"
        "Required acknowledgement when using Gaia data: this work has made use "
        "of data from the ESA mission Gaia, processed by the Gaia Data "
        "Processing and Analysis Consortium (DPAC)."
    )


def _exit_with_advice(exc: CatalogUnusable) -> NoReturn:
    """Report a catalog problem as a CLI error and exit non-zero.

    Both :func:`resolve_data_file` and :func:`_extract_raw_zarr` raise messages
    written to be READ — where the file belongs, the command that rebuilds it —
    so every entry point prints them the same way instead of letting one of them
    surface as a traceback with the advice buried in it.
    """
    aprint(f"\n❌ Error: {exc}")
    sys.exit(1)


def _extract_raw_zarr(data_zip_path: Path, dest: Path) -> Path:
    """Unpack the catalog zip into ``dest`` and return the raw zarr inside it.

    Everything that makes an archive readable *as this demo's catalog* is checked
    here, at the extraction, because none of it is checkable earlier and each
    failure mode otherwise escapes as a traceback that buries the advice: the zip
    has to open (a truncated copy raises ``BadZipFile``), it has to hold a
    top-level ``RAW_ZARR_NAME`` directory (a wrong ``--output`` stem extracts
    *successfully*, and only the store read later notices), and that directory has
    to be the raw star table (a store built by hand with other column names reads
    fine and dies on a bare ``KeyError`` mid-conversion). None of the exceptions
    involved is a ``CatalogUnusable``, which is the only type the entry points
    turn into advice — not ``BadZipFile``, not the bare ``KeyError``, and not any
    of what the store read raises (three classes, two from the open itself and one
    from reading a column's metadata; see the handler below) — so each one
    otherwise escapes as a traceback with nothing in it about the rebuild.

    The catalog is placed (or rebuilt) by hand, which is what makes all three
    ordinary rather than exotic — an interrupted ``scp``, the rebuild script's
    default ``--output``, a table assembled from one's own Gaia query.
    ``resolve_data_file`` cannot tell any of them apart, since reading the archive
    IS the check.
    """
    import zipfile

    try:
        with zipfile.ZipFile(data_zip_path, "r") as zip_ref:
            zip_ref.extractall(dest)
    except zipfile.BadZipFile as exc:
        raise CatalogUnusable(
            f"{data_zip_path} is not a readable zip archive ({exc}) — most "
            "likely a truncated or partial copy.\n"
            "Delete it and put a complete copy back, or rebuild it with\n"
            f"{REPLACE_COMMAND}"
        ) from exc
    raw_zarr_path = dest / RAW_ZARR_NAME
    if not raw_zarr_path.is_dir():
        raise CatalogUnusable(
            f"{data_zip_path} extracted, but holds no top-level "
            f"`{RAW_ZARR_NAME}/` directory — so the raw catalog is not where "
            "this demo reads it.\n"
            "That top-level name is load-bearing (it is the compatibility "
            "script's --output stem); replace this copy with\n"
            f"{REPLACE_COMMAND}"
        )
    try:
        # ``luxar._zarr_compat.open_group``, never a bare ``zarr.open``, and for
        # two independent reasons. (1) The facade passes
        # ``use_consolidated=False``, restoring the zarr-2 rule that a read sees
        # the arrays actually ON DISK. Nothing in the current build path
        # consolidates — the shipped zip carries no `.zmetadata` and the rebuild
        # script never writes one — so this reason does not bite on a catalog
        # obtained the documented way. It bites on the OTHER way the catalog
        # arrives: this file is hand-placed, and a store that its reader
        # re-exported or assembled themselves may well carry a consolidated
        # index, whereupon zarr 3's reversed default has the column check below
        # answer from that index and report a deleted column directory as present
        # — the guard passes and the converter reads that column as all-zeros
        # fill, which for `bp_rp` is a 3M-star scene with a dead colour index.
        # (2) The reason that bites on ANY catalog: ``zarr.open`` returns an
        # ``Array`` when the extracted directory is an array store rather than a
        # group (single-array ``zarr.save`` output — a table someone assembled
        # from their own query), and ``name not in <Array>`` falls back to the
        # SEQUENCE protocol: element-by-element, measured here at 33 s per 20 000
        # values (~1.6 ms each), i.e. over an hour on a 3M-row catalog before
        # printing advice that blames the columns. ``open_group`` raises on that
        # node instead, so it lands in the handler below within milliseconds.
        raw_table = open_group(str(raw_zarr_path), mode="r")
        missing = [name for name in RAW_TABLE_FIELDS if name not in raw_table]
    # ``FileNotFoundError``/``ValueError``, not zarr's own error names: zarr 3
    # deleted ``PathNotFoundError``, so naming it raises ``AttributeError`` while
    # merely BUILDING the handler tuple and the advice below is lost to that
    # traceback instead. The pair is really one class wide: zarr's own errors
    # descend from ``BaseZarrError``, itself a ``ValueError``, so the
    # ``FileNotFoundError`` arm is a subset kept because it is the spelling
    # ``luxar._zarr_compat.is_missing_error`` sanctions. ``ValueError`` is what
    # earns its place — the three failures measured on this path do NOT share a
    # narrower base, and only two of them come from the OPEN: a v2 array store
    # gives ``GroupNotFoundError`` (a ``FileNotFoundError``) and a v3 one
    # ``ContainsArrayError``. The third arrives one line later: a column whose
    # ``.zarray`` is corrupt JSON opens as a ``Group`` perfectly well, and it is
    # the membership check that reads that document and raises a bare
    # ``json.JSONDecodeError`` (not a zarr error at all) — which is why both
    # statements sit inside this one ``try``. Only the first is a
    # ``FileNotFoundError``, so the narrower handler let two thirds of the
    # docstring's "everything is checked here" escape as the traceback this
    # function exists to prevent.
    except (FileNotFoundError, ValueError) as exc:
        raise CatalogUnusable(
            f"{data_zip_path} holds a `{RAW_ZARR_NAME}/` directory, but it does "
            "not read as this demo's raw star table: it is not a zarr store, or "
            f"one of its columns has unreadable metadata ({exc}).\n"
            "Delete it and put a complete copy back, or rebuild it with\n"
            f"{REPLACE_COMMAND}"
        ) from exc
    if missing:
        raise CatalogUnusable(
            f"{data_zip_path} holds `{RAW_ZARR_NAME}/`, but it is missing the raw "
            f"Gaia column(s) {', '.join(missing)} — this demo reads a flat table "
            f"of {', '.join(RAW_TABLE_FIELDS)}, one value per star.\n"
            "Rebuild it (which writes exactly those columns) with\n"
            f"{REPLACE_COMMAND}"
        )
    return raw_zarr_path


def compute_colors(bp_rp: np.ndarray, phot_g_mean_mag: np.ndarray) -> np.ndarray:
    """Convert BP-RP color index and magnitude to RGB colors.

    Args:
        bp_rp: BP-RP color index from Gaia
        phot_g_mean_mag: G-band magnitude

    Returns:
        RGB colors (N, 3) as float32
    """
    bp_rp_norm = np.clip((bp_rp + 0.5) / 5.0, 0, 1)
    r = bp_rp_norm
    g = 1.0 - 2.0 * np.abs(bp_rp_norm - 0.5)
    b = 1.0 - bp_rp_norm

    brightness = np.clip((21 - phot_g_mean_mag) / 18.0, 0.1, 1.0)
    colors = np.stack([r * brightness, g * brightness, b * brightness], axis=1).astype(
        np.float32
    )
    return colors  # type: ignore[no-any-return]


def compute_radii(phot_g_mean_mag: np.ndarray) -> np.ndarray:
    """Compute point radii from magnitude.

    Uses visualization-friendly radii (not physically accurate).
    Points are 10-50x larger than actual stars for visibility.

    Args:
        phot_g_mean_mag: G-band magnitude

    Returns:
        Radii as float32 (in kpc)
    """
    mag_norm = np.clip((21 - phot_g_mean_mag) / 18.0, 0, 1)
    # Visualization scale: 0.001 to 0.011 kpc (1 to 11 pc)
    # Larger than physical stars but visible at galactic scales
    return (0.001 + 0.01 * mag_norm**2).astype(np.float32)  # type: ignore[no-any-return]


def load_and_convert_gaia_data(data_zarr_path: Path, output_path: Path) -> int:
    """Load raw Gaia zarr table and convert to Luxar format.

    This function demonstrates the pipeline:
    Raw Gaia data (zarr table) → Luxar scene (zarr)

    Args:
        data_zarr_path: Path to the extracted raw milky_way_gaia_3m.zarr
        output_path: Path for Luxar-formatted output

    Returns:
        Number of stars processed
    """
    # Scale multiplier for visualization
    # Coordinates are scaled up to make the scene easier to navigate
    SCALE = 10.0  # 10x larger for better visualization

    with asection("Loading Raw Gaia Data"):
        # Always an already-extracted directory: every caller unpacks the zip
        # first (via `_extract_raw_zarr`), because reading the zip in place
        # through a zip:// store is unreliable across zarr versions. Through the
        # facade for the same reason the guard there uses it: consolidated
        # metadata is not trusted, so a column that is not on disk raises here
        # rather than being served as zeros from a stale `.zmetadata`.
        store = open_group(str(data_zarr_path), mode="r")

        # Read arrays
        x_kpc = store["x_kpc"][:]
        y_kpc = store["y_kpc"][:]
        z_kpc = store["z_kpc"][:]
        phot_g_mean_mag = store["phot_g_mean_mag"][:]
        bp_rp = store["bp_rp"][:]

        n_stars = len(x_kpc)
        aprint(f"✓ Loaded {n_stars:,} stars")
        aprint(
            f"  Magnitude range: {phot_g_mean_mag.min():.2f} to {phot_g_mean_mag.max():.2f}"
        )

    with asection("Converting to Luxar Format"):
        # Combine positions and scale up for better visualization
        aprint(f"Creating position array (scaling by {SCALE}x for visibility)...")
        positions = np.column_stack(
            [x_kpc * SCALE, y_kpc * SCALE, z_kpc * SCALE]
        ).astype(np.float32)

        # Compute colors from Gaia photometry
        aprint("Computing stellar colors from BP-RP...")
        colors = compute_colors(bp_rp, phot_g_mean_mag)

        # Compute radii from brightness and scale
        aprint("Computing point radii from magnitude...")
        radii = compute_radii(phot_g_mean_mag) * SCALE

        aprint(f"✓ Prepared {n_stars:,} stars for Luxar")

    with asection("Creating Luxar Scene"):
        dims = Dimensions(
            [
                Dimension("x", unit="kpc", display=True),
                Dimension("y", unit="kpc", display=True),
                Dimension("z", unit="kpc", display=True),
            ]
        )

        # Start the camera pulled IN, framing the bright stellar bulk (the 3M
        # brightest stars cluster near the Sun, not the galactic centre). Robust
        # 2–98th percentile bounds ignore sparse-halo outliers that would
        # otherwise make the auto-fit zoom way out and leave the galaxy a tiny
        # dot.
        #
        # The second half of the original rationale — "closer start = the
        # coverage-fraction LOD immediately shows a finer level" — was a
        # workaround for #1361, where the finest level only engaged once the
        # object OVERFILLED the screen. The viewer's anchor now sits at half the
        # screen AREA (occupancy halving), so a plain fit is no longer stuck on a
        # coarse level and that part is redundant. The tighter framing is KEPT
        # purely as a composition choice (the galaxy fills the view); revisiting
        # it is a visual change, out of scope for the anchor fix.
        lo, hi = np.percentile(positions, [2, 98], axis=0)
        center = (lo + hi) / 2.0
        extent = float(np.max(hi - lo))
        fov_deg = CINEMATIC_FOV_DEG
        fit_dist = (extent * 0.5) / np.tan(np.radians(fov_deg) / 2.0)
        cam_dist = fit_dist * 0.65  # pull in ~35% tighter than a plain fit
        camera = CameraConfig(
            position=(
                float(center[0]),
                float(center[1] + extent * 0.15),
                float(center[2] + cam_dist),
            ),
            target=(float(center[0]), float(center[1]), float(center[2])),
            up=(0.0, 1.0, 0.0),
            near=float(max(0.5, cam_dist * 0.005)),
            far=float(cam_dist * 20.0 + extent * 10.0),
        )

        with LuxarZarrCompiler(output_path) as compiler:
            # Bake a dark-sky appearance: a moderate exposure keeps the
            # background black (a high exposure floods the faint-star haze into a
            # grey wash), a raised bloom threshold blooms only the brightest
            # stars, and ACES (set explicitly, the house default) supplies the
            # filmic rolloff. It does shift blue/red star hues a little; "None"
            # is the alternative if true stellar colour ever matters more
            # (#1459) — an exact passthrough, but not a free swap here: bloom is
            # on (strength 0.15, threshold 0.85) and is summed into the HDR
            # sample BEFORE tone mapping (`sampleHdrPlusBloom` in the viewer's
            # mega shader), so a "None" pin would flat-clip the bloomed star
            # cores that ACES's rolloff is holding together.
            scene = compiler.create_scene(
                dimensions=dims,
                viewer_config=ViewerConfig(
                    cinematic_mode=True,
                    camera=camera,
                    exposure=0.5,
                    tone_mapping="ACES",
                    bloom_enabled=True,
                    bloom_strength=0.15,
                    bloom_threshold=0.85,
                ),
                citation=DEMO_META["citation"],
            )

            # Add the stars with substitutive Points LOD: coarse levels replace
            # the 3M-star cloud with fewer, larger mass-preserving Gaussian splats
            # when the galaxy is small on screen, so the viewer only pays for the
            # detail it can resolve (the census demo uses the same wiring). The
            # `layer=True` flag rides onto the wrapper kind=lod group → one "Stars"
            # layer in the Layers panel.
            # Volumetric emission–absorption blending instead of plain
            # additive: dense sight-lines through the disc self-shadow instead
            # of saturating, which keeps the bulge from blowing out while the
            # spiral-arm structure stays readable.
            # The three appearance knobs are tuned as ONE set (and pinned by
            # tests/test_demo_gaia_milky_way_3m.py), because they all land on
            # the same two shader terms: ray mass = falloff * opacity, optical
            # depth tau = kappa * ray mass, emitted radiance = colour *
            # intensity * ray mass * S(tau).
            #   opacity 0.5 + kappa 0.12 keep tau low enough that the disc
            #     stays translucent front to back — a heavier tau hides the far
            #     side of the bulge behind the near side, which reads as a flat
            #     silhouette rather than depth.
            #   intensity 0.175 buys back the emission that the lower ray mass
            #     gives up: a 0–5.7 display range over the 0–32.3 data range in
            #     the Layers panel (the range is 1/intensity).
            # All three ride on the wrapper kind=lod group, so every level of
            # this MIXED ladder — lifted gsplats at the coarse levels, Points
            # at the finest — composites with the same tau and the same gain,
            # and the levels stay matched across an LOD switch.
            scene.add_points(
                "Stars",
                positions,
                colors=colors,
                radii=radii,
                opacity=0.5,
                blending_mode="volumetric",
                absorption=0.12,
                intensity=0.175,
                layer=True,
                substitutive_lod=substitutive_lod_or_flat(
                    dict(compression_factor=8, levels=3, device="auto")
                ),
            )

            # Add reference markers for famous stars
            aprint("Adding reference markers...")

            # Marker radius: 10x a typical scaled star, sized from the demo's OWN
            # radius law rather than a restated literal — `compute_radii` at the
            # magnitude whose normalized brightness is exactly 0.5, which its
            # (21 - mag) / 18 puts at G = 12.0 (~0.0035 kpc × SCALE = 0.035, so a
            # marker is 0.35). Retuning the law moves the markers with it.
            typical_star_radius = (
                float(compute_radii(np.array([12.0], dtype=np.float32))[0]) * SCALE
            )
            marker_radius = typical_star_radius * 10

            # The markers keep their ORIGINAL authored look, so — unlike the
            # retuned "Stars" node above — their kappa is not a free knob but a
            # rescale of the historical 1.3 through the 2026-08-02 ray-mass
            # unification: tau dropped its world-radius factor, so a bare 1.3
            # would now absorb ~3.5x harder (1 / (0.35 x 0.826)). Preserving
            # the authored look is exactly kappa * radius * chord.
            MARKER_ABSORPTION = (
                1.3 * marker_radius * float(np.sqrt(np.pi / np.log(100.0)))
            )

            # One string per marker, used TWICE: as the node's hover label and
            # as its legend row. Defined once so the tooltip and the legend
            # cannot drift apart.
            SUN_LABEL = "Sun — our star, 8.1 kpc from the Galactic Centre"
            BETELGEUSE_LABEL = (
                "Betelgeuse — red supergiant in Orion (~168 pc from the Sun)"
            )
            RIGEL_LABEL = "Rigel — blue supergiant in Orion (~265 pc from the Sun)"

            # Sun marker at the Sun's Galactocentric position
            r0_kpc = 8.122  # Sun-GC distance
            sun_position = np.array([[-r0_kpc * SCALE, 0.0, 0.0]], dtype=np.float32)
            sun_color = np.array([[1.0, 1.0, 0.0]], dtype=np.float32)  # Yellow

            scene.add_points(
                "Sun",
                sun_position,
                colors=sun_color,
                radii=marker_radius,
                opacity=1.0,
                blending_mode="volumetric",
                absorption=MARKER_ABSORPTION,
                labels=[SUN_LABEL],
                # One marker, one fixed destination: a single-element node needs
                # no `keys=` at all, since there is nothing per-element to
                # substitute. NASA's Sun page replaces SIMBAD, whose catalogue
                # does not resolve solar-system objects.
                link="https://science.nasa.gov/sun/",
                copy="Sun",
                layer=True,
            )
            aprint(f"  ✓ Sun at ({-r0_kpc * SCALE:.1f}, 0, 0)")

            # Betelgeuse (red supergiant in Orion, ~168 pc from Sun)
            betelgeuse_pos = np.array(
                [[-8.278192 * SCALE, -0.056179 * SCALE, -0.004962 * SCALE]],
                dtype=np.float32,
            )
            betelgeuse_color = np.array([[1.0, 0.3, 0.0]], dtype=np.float32)  # Red

            scene.add_points(
                "Betelgeuse",
                betelgeuse_pos,
                colors=betelgeuse_color,
                radii=marker_radius,
                opacity=1.0,
                blending_mode="volumetric",
                absorption=MARKER_ABSORPTION,
                labels=[BETELGEUSE_LABEL],
                # One marker, one fixed destination: a single-element node needs
                # no `keys=` at all, since there is nothing per-element to
                # substitute. SIMBAD is the canonical object page (#1917).
                link="https://simbad.cds.unistra.fr/simbad/sim-basic?Ident=Betelgeuse",
                copy="Betelgeuse",
                layer=True,
            )
            aprint("  ✓ Betelgeuse (red supergiant, 168 pc)")

            # Rigel (blue supergiant in Orion, ~265 pc from Sun)
            rigel_pos = np.array(
                [[-8.331409 * SCALE, -0.117085 * SCALE, -0.091686 * SCALE]],
                dtype=np.float32,
            )
            rigel_color = np.array([[0.5, 0.7, 1.0]], dtype=np.float32)  # Blue

            scene.add_points(
                "Rigel",
                rigel_pos,
                colors=rigel_color,
                radii=marker_radius,
                opacity=1.0,
                blending_mode="volumetric",
                absorption=MARKER_ABSORPTION,
                labels=[RIGEL_LABEL],
                # One marker, one fixed destination: a single-element node needs
                # no `keys=` at all, since there is nothing per-element to
                # substitute. SIMBAD is the canonical object page (#1917).
                link="https://simbad.cds.unistra.fr/simbad/sim-basic?Ident=Rigel",
                copy="Rigel",
                layer=True,
            )
            aprint("  ✓ Rigel (blue supergiant, 265 pc)")

            # Named-star legend: swatch colours read from the marker RGB above
            # (not re-invented), same add_html pattern as the other demos.
            def _swatch(rgb: np.ndarray) -> str:
                """Format a float 0-1 RGB triple as a CSS ``rgb()`` colour."""
                r, g, b = (int(round(float(c) * 255)) for c in rgb)
                return f"rgb({r},{g},{b})"

            _dot = (
                "display:inline-block;width:0.8em;height:0.8em;"
                "border-radius:50%;margin-right:0.5em;vertical-align:middle"
            )
            legend_html = (
                '<div style="font:13px sans-serif;color:#fff;line-height:1.7">'
                f'<span style="{_dot};background:{_swatch(sun_color[0])}"></span>'
                f"{SUN_LABEL}<br>"
                f'<span style="{_dot};background:{_swatch(betelgeuse_color[0])}"></span>'
                f"{BETELGEUSE_LABEL}<br>"
                f'<span style="{_dot};background:{_swatch(rigel_color[0])}"></span>'
                f"{RIGEL_LABEL}"
                "</div>"
            )
            scene.add_html(
                legend_html,
                position=(0.02, 0.98),
                anchor="bottom-left",
                opacity=0.92,
            )

            # Overlay annotations
            scene.add_text(
                "Milky Way (Gaia DR3)",
                position=(0.02, 0.02),
                font_size=0.055,
                anchor="top-left",
                color="rgba(255,255,255,0.6)",
                blend_mode="difference",
            )
            add_demo_caption(
                scene,
                "3M stars \u2022 Galactocentric coords",
                DEMO_META.get("citation"),
            )

        total_size = sum(
            f.stat().st_size for f in output_path.rglob("*") if f.is_file()
        )
        aprint(f"✓ Created Luxar scene: {output_path}")
        aprint(f"  Size: {total_size / 1e6:.1f} MB")

    return n_stars


def load_and_convert_from_zip(data_zip_path: Path, temp_dir: Path) -> Path:
    """Load raw Gaia data from zip and convert to Luxar format.

    Args:
        data_zip_path: Path to milky_way_gaia_3m.zarr.zip
        temp_dir: Temporary directory for extraction

    Returns:
        Path to Luxar-formatted zarr
    """
    with asection("Loading Gaia DR3 Dataset"):
        # No existence check: every caller passes `resolve_data_file()`, which
        # already raised (with the rebuild command) if nothing was found.
        aprint(f"Data file: {data_zip_path}")
        aprint(f"Size: {data_zip_path.stat().st_size / 1e6:.1f} MB")

        # Extract zarr from zip to temp directory
        aprint("\nExtracting Gaia data from zip...")
        raw_zarr_path = _extract_raw_zarr(data_zip_path, temp_dir)
        aprint(f"✓ Extracted to: {raw_zarr_path}")

    # Convert to Luxar format
    luxar_zarr_path = temp_dir / f"{DEMO_META['outputs'][0]}.luxar.zarr"
    load_and_convert_gaia_data(raw_zarr_path, luxar_zarr_path)

    return luxar_zarr_path


def main() -> None:
    """Main demo entry point."""
    aprint("=" * 70)
    aprint("GAIA DR3 MILKY WAY - 3 MILLION REAL STARS")
    aprint("=" * 70)
    aprint("")
    aprint("Visualizing 3,000,000 real stars from ESA's Gaia Data Release 3!")
    aprint("")
    aprint("Dataset Details:")
    aprint("  • Source: Gaia DR3 (European Space Agency)")
    aprint("  • Stars: 3 million brightest (mag 1.94 to 12.62)")
    aprint("  • Coordinates: Galactocentric (x, y, z in kpc)")
    aprint("  • Colors: Real stellar temperatures from BP-RP photometry")
    aprint("  • Quality: High-precision (parallax_over_error > 5)")
    aprint("")
    aprint("What You'll See:")
    aprint("  • The thin disk of the Milky Way")
    aprint("  • Dense central bulge (Galactic Center)")
    aprint("  • Real stellar distribution and colors")
    aprint("  • You are viewing from the Sun's position!")
    aprint("")
    aprint("Color Guide:")
    aprint("  🔵 Blue/White:  Hot young stars (O, B, A types)")
    aprint("  🟡 Yellow:      Sun-like stars (F, G types)")
    aprint("  🔴 Red/Orange:  Cool old stars (K, M types)")
    aprint("")
    aprint("Scientific Context:")
    aprint("  This is REAL astronomical data! Each point is an actual star")
    aprint("  measured by the Gaia space telescope. The colors represent real")
    aprint("  stellar temperatures, and the positions are transformed to a")
    aprint("  Galactocentric reference frame.")
    aprint("")
    aprint("  The Sun (and Earth) is at approximately (-8.122, 0, 0) kpc")
    aprint("  from the Galactic Center. You're viewing our galaxy from home!")
    aprint("")

    # Resolve the catalog ONCE, for both branches: `resolve_data_file`'s message
    # is the whole point of the missing-file path, so it must not surface as a
    # raw traceback on the default (serve) invocation either.
    try:
        data_file = resolve_data_file(
            build_catalog_requested="--build-catalog" in sys.argv,
            recompute="--recompute" in sys.argv,
        )
    except CatalogUnusable as e:
        _exit_with_advice(e)

    # If --no-serve, use persistent directory; otherwise temp for auto-cleanup
    if "--no-serve" in sys.argv:
        output_path = get_demos_output_dir() / f"{DEMO_META['outputs'][0]}.luxar.zarr"
        # Extract the raw .zarr from the zip to a temp dir, then convert to the
        # persistent output_path (same extraction the serve path uses — reading
        # the zip in place via a zip:// store is unreliable across zarr versions).
        # A catalog built with the wrong `--output` stem fails HERE rather than at
        # resolution, so this call needs the same clean-error exit.
        with tempfile.TemporaryDirectory(prefix="luxar_demo_gaia_") as tmpdir:
            try:
                raw_zarr_path = _extract_raw_zarr(data_file, Path(tmpdir))
            except CatalogUnusable as e:
                _exit_with_advice(e)
            load_and_convert_gaia_data(raw_zarr_path, output_path)
        aprint(f"Dataset generated at {output_path}")
        return

    # Use temporary directory for serving (auto-cleanup on exit)
    with tempfile.TemporaryDirectory(prefix="luxar_demo_gaia_") as tmpdir:
        tmp_path = Path(tmpdir)

        # Load from zip and convert to Luxar format (extracts to temp_dir).
        # Same clean-error exit as above: the extraction inside is where a
        # wrong-stem catalog is caught. This try wraps the CONVERSION too, so it
        # catches `CatalogUnusable` and not `FileNotFoundError` — a missing file
        # met while writing the scene is a bug, and swallowing its traceback into
        # "❌ Error" would report it as a catalog problem it is not.
        try:
            zarr_path = load_and_convert_from_zip(data_file, tmp_path)
        except CatalogUnusable as e:
            _exit_with_advice(e)

        aprint("")
        aprint("=" * 70)
        aprint("VIEWING TIPS")
        aprint("=" * 70)
        aprint("")
        aprint("Navigation:")
        aprint("  • Start zoomed OUT to see the full galactic structure")
        aprint("  • Look for the thin disk and central bulge")
        aprint("  • The Milky Way is a flat disk ~30 kpc across")
        aprint("  • Zoom IN to see individual stars with colors")
        aprint("")
        aprint("What to Explore:")
        aprint("  • Yellow marker: Our Sun (you are here!)")
        aprint("  • Red marker: Betelgeuse (red supergiant, 168 pc)")
        aprint("  • Blue marker: Rigel (blue supergiant, 265 pc)")
        aprint("  • Origin (0,0,0): The Galactic Center (8 kpc away)")
        aprint("  • Top-down view: See the disk structure")
        aprint("  • Edge-on view: See how thin the disk is")
        aprint("  • Navigate toward origin to approach Galactic Center")
        aprint("")
        aprint("This is Real Science:")
        aprint("  • Every point is a real star with measured position")
        aprint("  • Colors reflect actual stellar surface temperatures")
        aprint("  • Distances determined from parallax measurements")
        aprint("  • Published in: Gaia Collaboration (2022), A&A")
        aprint("")

        # Launch viewer
        launch_viewer(zarr_path)

    aprint("")
    aprint("Cleanup complete - temporary files removed")
    aprint("")
    aprint("=" * 70)
    aprint("DATA ATTRIBUTION & SOURCES")
    aprint("=" * 70)
    aprint("")
    aprint("Data Credit: ESA/Gaia/DPAC")
    aprint("  Mission: https://www.cosmos.esa.int/gaia")
    aprint("  Archive: https://gea.esac.esa.int/archive/")
    aprint("")
    aprint("Citation:")
    aprint("  Gaia Collaboration, Vallenari et al. (2023)")
    aprint('  "Gaia Data Release 3: Summary of the content and survey properties"')
    aprint("  Astronomy & Astrophysics, 674, A1")
    aprint("  https://doi.org/10.1051/0004-6361/202243940")
    aprint("")
    aprint("Data Generation:")
    aprint("  Built locally with `luxar demo run gaia_milky_way -- --build-catalog`")
    aprint("  Source: ESA Gaia DR3 (https://gea.esac.esa.int/archive/)")
    aprint("")
    aprint("Coordinate System:")
    aprint("  Galactocentric frame with R₀ = 8.122 kpc")
    aprint("  Reference: GRAVITY Collaboration (2018), A&A 615, L15")
    aprint("  https://doi.org/10.1051/0004-6361/201833718")
    aprint("")
    aprint("Thank you for exploring our galaxy!")
    aprint("")


if __name__ == "__main__":
    main()
