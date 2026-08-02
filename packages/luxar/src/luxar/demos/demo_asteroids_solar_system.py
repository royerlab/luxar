#!/usr/bin/env python3
"""Data-Driven Demo: The Solar System — ~1.5M Real Asteroids (JPL SBDB)

Every catalogued minor planet in the Solar System, placed in real 3D space by
propagating its measured Keplerian orbit to a common epoch. Asteroids are Points
coloured by semi-major axis; the eight planets, the Sun, and the planets' orbit
ellipses (Lines) give the scene its scale and structure. The main belt, the
Kirkwood gaps, the Hilda triangle, the Jupiter Trojan clouds, and the scattered
near-Earth swarm all emerge from the real orbital-element distribution.

================================================================================
WHAT THIS DEMO SHOWS
================================================================================

- **~1.5 million asteroids** from the NASA/JPL Small-Body Database (SBDB), each
  a Point at its true heliocentric-ecliptic position for one instant in time.
- **Color = semi-major axis** (AU) via a perceptual colormap, so the belt's
  radial structure — and the resonance gaps carved by Jupiter — is legible.
- **The eight planets + Sun** as reference markers, with each planet's orbit
  drawn as a Lines polyline.
- **Optional animation** (``--animate``): a time slider advances every body along
  its orbit, so the whole system revolves (inner bodies fast, outer bodies slow).

THE PHYSICS (all computed here, no astropy)
-------------------------------------------
Each body is stored as classical orbital elements (a, e, i, Ω, ω, M) at some
epoch. To place it in space we:
  1. Propagate the mean anomaly to a common epoch:  M = M0 + n·(t − epoch),
     with mean motion  n = k / a^1.5  (k = 0.01720209895 rad/day, the Gaussian
     gravitational constant; a in AU gives n in rad/day).
  2. Solve Kepler's equation  M = E − e·sin E  for the eccentric anomaly E
     (vectorized Newton iteration).
  3. True anomaly  ν = 2·atan2(√(1+e)·sin(E/2), √(1−e)·cos(E/2)),
     radius  r = a·(1 − e·cos E), position in the orbital plane (r·cosν, r·sinν).
  4. Rotate by argument of periapsis ω, inclination i, and longitude of
     ascending node Ω into heliocentric-ecliptic X, Y, Z (AU).

DATA SOURCE & CITATION
----------------------
NASA/JPL Small-Body Database (SBDB) Query API
    https://ssd-api.jpl.nasa.gov/doc/sbdb_query.html
    Data courtesy NASA/JPL-Caltech (SSD/CNEOS). Public domain.
Planet mean elements: E. M. Standish, "Keplerian Elements for Approximate
    Positions of the Major Planets" (JPL SSD), J2000 epoch.

USAGE
-----
    python demo_asteroids_solar_system.py [--animate] [--no-serve]
    python demo_asteroids_solar_system.py --max-asteroids 500000
    python demo_asteroids_solar_system.py --recompute        # re-fetch SBDB catalog

Options:
    --animate:        Time-slider version (subsampled) with bodies orbiting.
    --max-asteroids:  Cap the number of asteroids (default: all in the catalog).
    --recompute:      Re-download the SBDB catalog (otherwise the cached copy is reused).
    --no-serve:       Generate the scene but don't launch the viewer.

On first run this downloads the SBDB catalog (~a few hundred MB) to
``~/.cache/luxar/asteroids/``; subsequent runs are offline.

Controls:
    - Mouse drag: rotate,  Scroll: zoom,  Right-drag: pan
    - 'F': return to the Sun-centered outer-planet view
    - With --animate: press '1' then '[' / ']' to step through time
"""

from __future__ import annotations

DEMO_META = {
    "key": "asteroids_solar_system",
    "title": "Solar System — 1.5M Asteroids",
    "description": "~1.5M real minor planets from NASA/JPL SBDB, placed by propagating their Keplerian orbits.",
    "category": "astronomy",
    "geometry": "points+lines",
    "requirements": {
        "download_mb": 300,  # approx
        "compute": "heavy",
        "gpu": "none",
        "local_data": None,
    },
    "caches": ["asteroids"],
    "outputs": ["asteroids_solar_system"],
}

import json
import sys
import tempfile
from pathlib import Path

import numpy as np
from arbol import Arbol, aprint, asection

from luxar import Dimension, Dimensions, LuxarZarrCompiler
from luxar.core.viewer_config import CameraConfig, ViewerConfig
from luxar.demos import launch_viewer
from luxar.utils.demos import parse_demo_flags
from luxar.utils.paths import get_demos_output_dir

# -----------------------------------------------------------------------------
# Configuration
# -----------------------------------------------------------------------------

SBDB_FIELDS = "full_name,epoch,a,e,i,om,w,ma,H"
SBDB_URL = f"https://ssd-api.jpl.nasa.gov/sbdb_query.api?fields={SBDB_FIELDS}&sb-kind=a"

CACHE_DIR = Path.home() / ".cache" / "luxar" / "asteroids"
CACHE_JSON = CACHE_DIR / "sbdb_asteroids.json"
# Processed-catalog cache: parsing the multi-hundred-MB JSON is the slow step,
# so the cleaned element arrays are cached here and reused on later runs.
CACHE_PARSED = CACHE_DIR / "sbdb_parsed_catalog.npz"

# Common instant we place every body at: J2000.0 (Julian Date).
J2000_JD = 2451545.0
# Gaussian gravitational constant: mean motion n = GAUSS_K / a^1.5  [rad/day, a in AU].
GAUSS_K = 0.01720209895

# Hover labels are attached only to the brightest (largest-H-magnitude → biggest)
# asteroids to keep the scene light; the rest render without a tooltip.
LABEL_TOP_N = 40_000

# --animate parameters.
ANIMATE_MAX_ASTEROIDS = 200_000
ANIMATE_FRAMES = 48
ANIMATE_SPAN_DAYS = 4.0 * 365.25  # four years, so the inner belt visibly revolves

Arbol.max_depth = 3


# -----------------------------------------------------------------------------
# Orbital mechanics (pure, unit-testable)
# -----------------------------------------------------------------------------


def solve_kepler(
    mean_anomaly: np.ndarray, ecc: np.ndarray, iters: int = 8
) -> np.ndarray:
    """Solve Kepler's equation ``M = E - e·sin(E)`` for eccentric anomaly E.

    Vectorized Newton iteration. ``mean_anomaly`` and ``ecc`` are broadcast to a
    common shape; the result is returned in radians.
    """
    M = np.mod(np.asarray(mean_anomaly, dtype=np.float64) + np.pi, 2 * np.pi) - np.pi
    e = np.asarray(ecc, dtype=np.float64)
    # Danby's starter — robust and quick to converge for e < 1.
    E = M + e * np.sin(M)
    for _ in range(iters):
        f = E - e * np.sin(E) - M
        fp = 1.0 - e * np.cos(E)
        E = E - f / fp
    return E


def mean_motion(a: np.ndarray) -> np.ndarray:
    """Mean motion n [rad/day] for semi-major axis ``a`` [AU] (Sun-dominated)."""
    return GAUSS_K / np.power(np.asarray(a, dtype=np.float64), 1.5)


def propagate_mean_anomaly(
    m0_rad: np.ndarray, a: np.ndarray, epoch_jd: np.ndarray, target_jd: float
) -> np.ndarray:
    """Advance mean anomaly from each body's ``epoch_jd`` to ``target_jd`` [rad]."""
    dt = target_jd - np.asarray(epoch_jd, dtype=np.float64)
    return np.asarray(m0_rad, dtype=np.float64) + mean_motion(a) * dt


def elements_to_xyz(
    a: np.ndarray,
    e: np.ndarray,
    i: np.ndarray,
    Omega: np.ndarray,
    w: np.ndarray,
    M: np.ndarray,
) -> np.ndarray:
    """Classical elements → heliocentric-ecliptic Cartesian positions.

    All angles in radians, ``a`` in AU. Returns an ``(N, 3)`` float32 array of
    (X, Y, Z) in AU.
    """
    a = np.asarray(a, dtype=np.float64)
    e = np.asarray(e, dtype=np.float64)
    E = solve_kepler(M, e)
    # True anomaly and radius.
    nu = 2.0 * np.arctan2(
        np.sqrt(1.0 + e) * np.sin(E / 2.0), np.sqrt(1.0 - e) * np.cos(E / 2.0)
    )
    r = a * (1.0 - e * np.cos(E))
    # Position in the orbital plane.
    xo = r * np.cos(nu)
    yo = r * np.sin(nu)
    cO, sO = np.cos(Omega), np.sin(Omega)
    ci, si = np.cos(i), np.sin(i)
    cw, sw = np.cos(w), np.sin(w)
    x = xo * (cO * cw - sO * sw * ci) - yo * (cO * sw + sO * cw * ci)
    y = xo * (sO * cw + cO * sw * ci) - yo * (sO * sw - cO * cw * ci)
    z = xo * (sw * si) + yo * (cw * si)
    return np.column_stack([x, y, z]).astype(np.float32)


def orbit_polyline(
    a: float, e: float, i: float, Omega: float, w: float, n: int = 256
) -> np.ndarray:
    """Sample a full closed orbit ellipse as an ``(n, 3)`` float32 polyline (AU)."""
    nu = np.linspace(0.0, 2.0 * np.pi, n, dtype=np.float64)
    r = a * (1.0 - e * e) / (1.0 + e * np.cos(nu))
    xo = r * np.cos(nu)
    yo = r * np.sin(nu)
    cO, sO = np.cos(Omega), np.sin(Omega)
    ci, si = np.cos(i), np.sin(i)
    cw, sw = np.cos(w), np.sin(w)
    x = xo * (cO * cw - sO * sw * ci) - yo * (cO * sw + sO * cw * ci)
    y = xo * (sO * cw + cO * sw * ci) - yo * (sO * sw - cO * cw * ci)
    z = xo * (sw * si) + yo * (cw * si)
    return np.column_stack([x, y, z]).astype(np.float32)


# -----------------------------------------------------------------------------
# The eight planets — Standish J2000 mean elements
# (a[AU], e, i[deg], Ω[deg], ϖ=longitude-of-perihelion[deg], L=mean-longitude[deg])
# M = L − ϖ ;  ω = ϖ − Ω .  Colors are illustrative.
# -----------------------------------------------------------------------------

PLANETS = [
    # name,        a,           e,          i,          Omega,        varpi,        L,            color
    (
        "Mercury",
        0.38709927,
        0.20563593,
        7.00497902,
        48.33076593,
        77.45779628,
        252.25032350,
        (0.72, 0.66, 0.60),
    ),
    (
        "Venus",
        0.72333566,
        0.00677672,
        3.39467605,
        76.67984255,
        131.60246718,
        181.97909950,
        (0.95, 0.85, 0.55),
    ),
    (
        "Earth",
        1.00000261,
        0.01671123,
        -0.00001531,
        0.0,
        102.93768193,
        100.46457166,
        (0.35, 0.60, 1.00),
    ),
    (
        "Mars",
        1.52371034,
        0.09339410,
        1.84969142,
        49.55953891,
        -23.94362959,
        -4.55343205,
        (0.90, 0.40, 0.25),
    ),
    (
        "Jupiter",
        5.20288700,
        0.04838624,
        1.30439695,
        100.47390909,
        14.72847983,
        34.39644051,
        (0.90, 0.75, 0.55),
    ),
    (
        "Saturn",
        9.53667594,
        0.05386179,
        2.48599187,
        113.66242448,
        92.59887831,
        49.95424423,
        (0.95, 0.87, 0.65),
    ),
    (
        "Uranus",
        19.18916464,
        0.04725744,
        0.77263783,
        74.01692503,
        170.95427630,
        313.23810451,
        (0.60, 0.85, 0.90),
    ),
    (
        "Neptune",
        30.06992276,
        0.00859048,
        1.77004347,
        131.78422574,
        44.96476227,
        -55.12002969,
        (0.30, 0.45, 0.95),
    ),
]


def planet_state(target_jd: float = J2000_JD) -> list[dict]:
    """Compute each planet's position, orbit ellipse, elements at ``target_jd``."""
    out = []
    for name, a, e, i_deg, Om_deg, varpi_deg, L_deg, color in PLANETS:
        i, Om, varpi, L = np.radians([i_deg, Om_deg, varpi_deg, L_deg])
        w = varpi - Om
        m0 = L - varpi
        M = float(
            propagate_mean_anomaly(
                np.array([m0]), np.array([a]), np.array([J2000_JD]), target_jd
            )[0]
        )
        pos = elements_to_xyz(
            np.array([a]),
            np.array([e]),
            np.array([i]),
            np.array([Om]),
            np.array([w]),
            np.array([M]),
        )[0]
        out.append(
            {
                "name": name,
                "a": a,
                "e": e,
                "i": i,
                "Omega": Om,
                "w": w,
                "M": M,
                "position": pos,
                "orbit": orbit_polyline(a, e, i, Om, w),
                "color": color,
            }
        )
    return out


# -----------------------------------------------------------------------------
# Data loading (network)
# -----------------------------------------------------------------------------


def _looks_like_complete_json(path: Path) -> bool:
    """Cheap integrity check: a complete SBDB response ends with '}' (or ']')."""
    try:
        if path.stat().st_size < 128:
            return False
        with open(path, "rb") as f:
            f.seek(-64, 2)
            tail = f.read().rstrip()
        return tail.endswith(b"}") or tail.endswith(b"]")
    except OSError:
        return False


def download_sbdb(recompute: bool = False) -> Path:
    """Download the SBDB asteroid catalog to the cache (download-once).

    SBDB is a dynamic query endpoint (no stable size, Range-resume is
    inappropriate), so this uses a retrying ``requests`` session (transient
    5xx/429 backoff) writing atomically to a ``.part`` file, and verifies the
    result parses as JSON before accepting it — a truncated transfer is
    rejected rather than silently cached.
    """
    CACHE_DIR.mkdir(parents=True, exist_ok=True)
    if CACHE_JSON.exists() and not recompute and _looks_like_complete_json(CACHE_JSON):
        aprint(
            f"  Using cached {CACHE_JSON.name} ({CACHE_JSON.stat().st_size / 1e6:.0f} MB)"
        )
        return CACHE_JSON

    import requests
    from requests.adapters import HTTPAdapter
    from urllib3.util.retry import Retry

    aprint("Downloading SBDB asteroid catalog from JPL (this can take a minute)...")
    session = requests.Session()
    retry = Retry(
        total=5,
        backoff_factor=2,
        status_forcelist=[429, 500, 502, 503, 504],
        allowed_methods=["GET"],
    )
    session.mount("https://", HTTPAdapter(max_retries=retry))

    tmp = CACHE_JSON.parent / (CACHE_JSON.name + ".part")
    with session.get(SBDB_URL, stream=True, timeout=600) as r:
        r.raise_for_status()
        with open(tmp, "wb") as f:
            for chunk in r.iter_content(chunk_size=1 << 20):
                if chunk:
                    f.write(chunk)
    if not _looks_like_complete_json(tmp):
        tmp.unlink(missing_ok=True)
        raise RuntimeError(
            "SBDB download appears truncated (not valid JSON). Re-run to retry."
        )
    tmp.rename(CACHE_JSON)
    aprint(f"✓ Downloaded {CACHE_JSON.stat().st_size / 1e6:.0f} MB")
    return CACHE_JSON


def parse_sbdb(obj: dict) -> dict:
    """Parse an SBDB query response into numpy element arrays.

    Returns a dict with keys ``a,e,i,Omega,w,M,epoch,H,names`` where angles are
    radians and ``a`` is AU. Rows with non-finite or unbound (e ≥ 1) orbits are
    dropped so every survivor has a closed ellipse.
    """
    fields = obj["fields"]
    idx = {name: k for k, name in enumerate(fields)}
    data = obj["data"]
    cols = list(zip(*data)) if data else [[] for _ in fields]

    def col(name: str) -> np.ndarray:
        return np.array(cols[idx[name]], dtype=object)

    a = col("a").astype(np.float64)
    e = col("e").astype(np.float64)
    i = np.radians(col("i").astype(np.float64))
    Omega = np.radians(col("om").astype(np.float64))
    w = np.radians(col("w").astype(np.float64))
    M = np.radians(col("ma").astype(np.float64))
    epoch = col("epoch").astype(np.float64)
    # H may contain empty strings for a few bodies.
    H_raw = col("H")
    H = np.array(
        [float(v) if str(v).strip() not in ("", "None") else np.nan for v in H_raw]
    )
    names = col("full_name").astype(str)

    good = np.isfinite(a) & np.isfinite(e) & (a > 0) & (e >= 0) & (e < 1.0)
    good &= (
        np.isfinite(i)
        & np.isfinite(Omega)
        & np.isfinite(w)
        & np.isfinite(M)
        & np.isfinite(epoch)
    )
    return {
        "a": a[good],
        "e": e[good],
        "i": i[good],
        "Omega": Omega[good],
        "w": w[good],
        "M": M[good],
        "epoch": epoch[good],
        "H": H[good],
        "names": names[good],
    }


def _load_or_build_catalog(recompute: bool) -> dict:
    """Return the parsed element arrays, self-contained on a fresh system.

    Resolution order: cached processed ``.npz`` → (else) download raw JSON,
    parse, and cache the processed ``.npz`` for next time. ``recompute`` forces
    a fresh download + parse.
    """
    if CACHE_PARSED.exists() and not recompute:
        aprint(f"  Using cached parsed catalog ({CACHE_PARSED.name})")
        with np.load(CACHE_PARSED, allow_pickle=False) as d:
            return {k: d[k] for k in d.files}

    path = download_sbdb(recompute=recompute)
    aprint("Parsing catalog (one-time; result is cached)...")
    with open(path) as f:
        obj = json.load(f)
    cat = parse_sbdb(obj)
    CACHE_DIR.mkdir(parents=True, exist_ok=True)
    # Write through a file handle: np.savez appends ".npz" to any path that
    # doesn't already end in it, which would break a ".part" temp name.
    tmp = CACHE_PARSED.parent / (CACHE_PARSED.name + ".part")
    with open(tmp, "wb") as fh:
        np.savez(fh, **cat)
    tmp.rename(CACHE_PARSED)
    aprint(f"✓ Cached processed catalog to {CACHE_PARSED.name}")
    return cat


def load_asteroids(recompute: bool, max_asteroids: int | None) -> dict:
    """Load + parse the SBDB catalog, optionally capping to the brightest N."""
    with asection("Loading asteroid catalog (JPL SBDB)"):
        cat = _load_or_build_catalog(recompute)
        aprint(f"✓ {len(cat['a']):,} asteroids with bound orbits")
        if max_asteroids is not None and len(cat["a"]) > max_asteroids:
            cat = _take_brightest(cat, max_asteroids)
            aprint(f"  Capped to the brightest {max_asteroids:,} (smallest H)")
        return cat


def _take_brightest(cat: dict, n: int) -> dict:
    """Keep the ``n`` brightest bodies (smallest H); NaN H sorts last."""
    H = cat["H"].copy()
    H[np.isnan(H)] = np.inf
    keep = np.argsort(H, kind="stable")[:n]
    return {k: v[keep] for k, v in cat.items()}


# -----------------------------------------------------------------------------
# Color + labels
# -----------------------------------------------------------------------------


def build_labels(cat: dict, top_n: int = LABEL_TOP_N) -> list[str]:
    """Hover labels for the brightest ``top_n`` asteroids; '' for the rest."""
    n = len(cat["a"])
    labels = [""] * n
    H = cat["H"].copy()
    H[np.isnan(H)] = np.inf
    order = np.argsort(H, kind="stable")[: min(top_n, n)]
    names = cat["names"]
    a = cat["a"]
    for j in order:
        labels[j] = f"{names[j].strip()}  ·  a={a[j]:.2f} AU"
    return labels


# -----------------------------------------------------------------------------
# Scene construction
# -----------------------------------------------------------------------------


def _solar_system_viewer_config() -> ViewerConfig:
    """Return the shared Sun-centered opening view for both scene variants."""
    # The catalog contains a sparse tail of high-semi-major-axis objects. A
    # bounds fit includes those outliers, putting the Sun off-center and
    # shrinking the planets to a dot. Instead, orbit exactly around the
    # heliocentric origin and frame the ~30 AU Neptune orbit from a moderately
    # elevated ecliptic view. Dynamic clipping still keeps distant objects
    # available when the user zooms out.
    camera = CameraConfig(
        position=(50.0, -50.0, 30.0),
        target=(0.0, 0.0, 0.0),
        up=(0.0, 0.0, 1.0),
        fov=50.0,  # slight margin around Neptune's ~30 AU orbit
    )
    return ViewerConfig(
        camera=camera,
        tone_mapping="ACES",
        dynamic_clipping_enabled=True,
    )


# Appearance constants for the asteroid cloud and the orbit ellipses.
#
# ``ASTEROID_SCALAR_GAIN`` is NOT a brightness knob. On a colormapped node an
# authored ``intensity`` is consumed as the scalar DISPLAY WINDOW —
# ``[0, 1/gain]`` — so 0.09 maps a ∈ [0, ~11 AU] across the colormap. That is
# what makes the main belt, the Kirkwood gaps and the Trojan clouds legible;
# mapping the catalog's full range (up to ~14500 AU, a sparse scattered tail)
# would crush every real structure into the bottom of the LUT.
#
# Brightness for 1.55M additive points therefore rides on ``opacity``, the one
# knob that linearly scales an additive contribution. These were previously
# conflated: ``intensity=0.09`` doubled as an 11x dimming because the viewer
# applied it BOTH as the window and as a post-LUT gain. That double
# application was removed (#936/#1081), so the dimming is now explicit here.
# Without it the belt saturates to white and erases the Sun, the planets and
# the orbit ellipses.
ASTEROID_SCALAR_GAIN = 0.09  # display window [0, ~11 AU]  (NOT brightness)
# Opacity tuned by eye at ``ASTEROID_OPACITY_REF_N`` bodies on screen.
ASTEROID_OPACITY = 0.03
ASTEROID_OPACITY_REF_N = 1_552_890

# Orbit ellipses are reference geometry: they must stay quiet next to the
# planets but still read as continuous lines against the belt's glow.
ORBIT_COLOR_SCALE = 0.7  # per-vertex dimming vs the planet's own color
ORBIT_OPACITY = 1.0
ORBIT_INTENSITY = 1.0
ORBIT_WIDTH = 0.03


def asteroid_opacity(n_visible: int) -> float:
    """Additive-cloud opacity for ``n_visible`` asteroids on screen at once.

    Additive blending accumulates, so total point weight scales with the
    number of visible points. Holding ``opacity x N`` constant keeps aggregate
    additive weight stable whether the build shows the whole catalog, a
    ``--max-asteroids`` subset, or the animated build's per-frame subsample —
    and as the SBDB catalog grows.
    """
    if n_visible <= 0:
        return ASTEROID_OPACITY
    return float(min(1.0, ASTEROID_OPACITY * ASTEROID_OPACITY_REF_N / n_visible))


def _orbit_colors(p: dict) -> np.ndarray:
    """Dim per-vertex color for a planet's orbit ellipse."""
    return np.tile(
        np.array(p["color"], dtype=np.float32) * ORBIT_COLOR_SCALE,
        (len(p["orbit"]), 1),
    )


def _add_static_bodies(scene, planets: list[dict]) -> None:
    """Add Sun + planets (Points) + planet orbit ellipses (Lines) for the static
    single-epoch scene (3D, no time dimension)."""
    scene.add_points(
        "Sun",
        np.zeros((1, 3), dtype=np.float32),
        colors=np.array([[1.0, 0.95, 0.6]], dtype=np.float32),
        radii=0.6,
        opacity=1.0,
        blending_mode="additive",
        intensity=1.0,
        layer=True,
    )
    for p in planets:
        scene.add_points(
            p["name"],
            p["position"].reshape(1, 3),
            colors=np.array([p["color"]], dtype=np.float32),
            radii=0.25,
            opacity=1.0,
            blending_mode="normal",
            layer=True,
        )
        scene.add_lines(
            f"{p['name']} orbit",
            vertices=p["orbit"],
            widths=ORBIT_WIDTH,
            colors=_orbit_colors(p),
            line_type="loop",
            opacity=ORBIT_OPACITY,
            intensity=ORBIT_INTENSITY,
            blending_mode="additive",
            layer=True,
        )


def _add_animated_bodies(
    scene, planets_per_frame: list[list[dict]], time_dim: str
) -> None:
    """Add Sun + planets + orbits to the animated (time-slider) scene.

    Planets move, so they are stacked into a single 4-column ``(N, [x,y,z,t])``
    Points node (direct positional mapping — no ``dim_order``). The Sun and the
    orbit ellipses are static, so they are stored once and shown at every time
    slot via ``extend_to_all=[time_dim]`` (3-column data + ``dim_order``)."""
    n_frames = len(planets_per_frame)
    n_planets = len(planets_per_frame[0])

    # Moving planets: one node, all frames.
    ppos = np.empty((n_frames * n_planets, 4), dtype=np.float32)
    pcol = np.empty((n_frames * n_planets, 3), dtype=np.float32)
    for f, planets in enumerate(planets_per_frame):
        for j, p in enumerate(planets):
            k = f * n_planets + j
            ppos[k, :3] = p["position"]
            ppos[k, 3] = float(f)
            pcol[k] = p["color"]
    scene.add_points(
        "Planets",
        ppos,
        colors=pcol,
        radii=0.25,
        opacity=1.0,
        blending_mode="normal",
        layer=True,
    )

    # Static Sun — visible at every time slot.
    scene.add_points(
        "Sun",
        np.zeros((1, 3), dtype=np.float32),
        colors=np.array([[1.0, 0.95, 0.6]], dtype=np.float32),
        radii=0.6,
        opacity=1.0,
        blending_mode="additive",
        intensity=1.0,
        dim_order=["x", "y", "z"],
        extend_to_all=[time_dim],
        layer=True,
    )

    # Static orbit ellipses — drawn once, shown at every time slot.
    for p in planets_per_frame[0]:
        scene.add_lines(
            f"{p['name']} orbit",
            vertices=p["orbit"],
            widths=ORBIT_WIDTH,
            colors=_orbit_colors(p),
            line_type="loop",
            opacity=ORBIT_OPACITY,
            intensity=ORBIT_INTENSITY,
            blending_mode="additive",
            dim_order=["x", "y", "z"],
            extend_to_all=[time_dim],
            layer=True,
        )


def build_static_scene(output_path: Path, cat: dict) -> int:
    """Static single-epoch snapshot: ~1.5M asteroid points + planets + Sun."""
    with asection("Building static Solar System scene"):
        pos = elements_to_xyz(
            cat["a"], cat["e"], cat["i"], cat["Omega"], cat["w"], cat["M"]
        )
        aprint(f"  Placed {len(pos):,} asteroids at J2000")

        labels = build_labels(cat)
        semi_major = cat["a"].astype(np.float32)

        dims = Dimensions(
            [
                Dimension("x", unit="AU", display=True),
                Dimension("y", unit="AU", display=True),
                Dimension("z", unit="AU", display=True),
            ]
        )
        with LuxarZarrCompiler(output_path) as compiler:
            scene = compiler.create_scene(
                dimensions=dims, viewer_config=_solar_system_viewer_config()
            )

            scene.add_points(
                "Asteroids",
                pos,
                scalars=semi_major,
                colormap="turbo",
                radii=0.012,
                labels=labels,
                opacity=asteroid_opacity(len(pos)),
                blending_mode="additive",
                intensity=ASTEROID_SCALAR_GAIN,
                layer=True,
            )
            _add_static_bodies(scene, planet_state(J2000_JD))

            scene.add_text(
                "Solar System — JPL SBDB",
                position=(0.02, 0.02),
                font_size=0.05,
                anchor="top-left",
                color="rgba(255,255,255,0.65)",
                blend_mode="difference",
            )
            scene.add_text(
                f"{len(pos):,} asteroids • color = semi-major axis (AU)",
                position=(0.98, 0.97),
                font_size=0.015,
                anchor="bottom-right",
                color="rgba(200,200,200,0.5)",
            )
        return len(pos)


def build_animated_scene(output_path: Path, cat: dict) -> int:
    """Time-slider build: subsampled asteroids advanced along their orbits."""
    with asection("Building animated Solar System scene"):
        if len(cat["a"]) > ANIMATE_MAX_ASTEROIDS:
            cat = _take_brightest(cat, ANIMATE_MAX_ASTEROIDS)
        n_ast = len(cat["a"])
        aprint(
            f"  {n_ast:,} asteroids × {ANIMATE_FRAMES} frames = {n_ast * ANIMATE_FRAMES:,} points"
        )

        times_jd = J2000_JD + np.linspace(0.0, ANIMATE_SPAN_DAYS, ANIMATE_FRAMES)
        semi_major = cat["a"].astype(np.float32)

        dims = Dimensions(
            [
                Dimension("x", unit="AU", display=True),
                Dimension("y", unit="AU", display=True),
                Dimension("z", unit="AU", display=True),
                Dimension(
                    "time",
                    unit="months",
                    display=False,
                    discrete=True,
                    range=(0, ANIMATE_FRAMES - 1),
                    step=1.0,
                    description="~monthly steps over four years",
                ),
            ]
        )
        with LuxarZarrCompiler(output_path) as compiler:
            scene = compiler.create_scene(
                dimensions=dims, viewer_config=_solar_system_viewer_config()
            )

            all_pos = np.empty((n_ast * ANIMATE_FRAMES, 4), dtype=np.float32)
            all_scalars = np.empty(n_ast * ANIMATE_FRAMES, dtype=np.float32)
            for f, t in enumerate(times_jd):
                M = propagate_mean_anomaly(cat["M"], cat["a"], cat["epoch"], t)
                xyz = elements_to_xyz(
                    cat["a"], cat["e"], cat["i"], cat["Omega"], cat["w"], M
                )
                sl = slice(f * n_ast, (f + 1) * n_ast)
                all_pos[sl, :3] = xyz
                all_pos[sl, 3] = float(f)
                all_scalars[sl] = semi_major
            scene.add_points(
                "Asteroids",
                all_pos,
                scalars=all_scalars,
                colormap="turbo",
                radii=0.012,
                opacity=asteroid_opacity(n_ast),
                blending_mode="additive",
                intensity=ASTEROID_SCALAR_GAIN,
                layer=True,
            )

            # Planets move (one node across all frames); Sun + orbits are static.
            planets_per_frame = [planet_state(t) for t in times_jd]
            _add_animated_bodies(scene, planets_per_frame, time_dim="time")

            scene.add_text(
                "Solar System (animated)",
                position=(0.02, 0.02),
                font_size=0.05,
                anchor="top-left",
                color="rgba(255,255,255,0.65)",
                blend_mode="difference",
            )
        return n_ast * ANIMATE_FRAMES


# -----------------------------------------------------------------------------
# Entry point
# -----------------------------------------------------------------------------


def _int_arg(argv: list[str], flag: str, default: int | None) -> int | None:
    for i, arg in enumerate(argv):
        if arg == flag and i + 1 < len(argv):
            return int(argv[i + 1])
        if arg.startswith(flag + "="):
            return int(arg.split("=", 1)[1])
    return default


def main() -> None:
    argv = sys.argv[1:]
    flags = parse_demo_flags()
    animate = "--animate" in argv
    recompute = flags["recompute"]
    no_serve = flags["no_serve"]
    max_asteroids = _int_arg(argv, "--max-asteroids", None)

    aprint("=" * 70)
    aprint("THE SOLAR SYSTEM — ~1.5 MILLION REAL ASTEROIDS (JPL SBDB)")
    aprint("=" * 70)
    aprint("Real orbital elements propagated to heliocentric-ecliptic XYZ.")
    aprint("")

    cat = load_asteroids(recompute=recompute, max_asteroids=max_asteroids)

    def _build(path: Path) -> int:
        return (
            build_animated_scene(path, cat)
            if animate
            else build_static_scene(path, cat)
        )

    if no_serve:
        name = "asteroids_solar_system.luxar.zarr"
        output_path = get_demos_output_dir() / name
        n = _build(output_path)
        aprint(f"Dataset generated at {output_path} ({n:,} points)")
        return

    with tempfile.TemporaryDirectory(prefix="luxar_demo_asteroids_") as tmpdir:
        output_path = Path(tmpdir) / "asteroids_solar_system.luxar.zarr"
        _build(output_path)
        aprint("")
        aprint("Data credit: NASA/JPL-Caltech SBDB (https://ssd.jpl.nasa.gov)")
        launch_viewer(output_path)

    aprint("Cleanup complete")


if __name__ == "__main__":
    main()
