#!/usr/bin/env python3
"""Self-Contained Demo: Global Earthquake Visualization

Visualize real-time earthquake data from USGS on a 3D Earth sphere with vertical
spikes showing magnitude and color-coded by time.

================================================================================
SEISMOLOGY & PLATE TECTONICS
================================================================================

WHAT ARE EARTHQUAKES?
---------------------
Earthquakes occur when stress accumulated in Earth's crust is suddenly released,
causing seismic waves to radiate outward. Most earthquakes happen at plate
boundaries where tectonic plates interact.

THE RICHTER SCALE (LOCAL MAGNITUDE)
-----------------------------------
Developed by Charles Richter in 1935, measures earthquake magnitude logarithmically:
- Magnitude 4-5: Light (may cause minor damage)
- Magnitude 5-6: Moderate (can damage buildings)
- Magnitude 6-7: Strong (destructive in populated areas)
- Magnitude 7-8: Major (serious damage over large areas)
- Magnitude 8+: Great (catastrophic destruction)

Each whole number increase = 10× more ground motion, ~31× more energy released!

PLATE BOUNDARIES & EARTHQUAKE ZONES
------------------------------------
Three types of boundaries produce most earthquakes:

1. CONVERGENT (Subduction zones):
   - One plate slides beneath another
   - Deepest and most powerful earthquakes
   - Examples: Pacific Ring of Fire, Japan, Chile, Alaska
   - Can produce magnitude 9+ megaquakes

2. TRANSFORM (Strike-slip):
   - Plates slide past each other horizontally
   - Shallow but destructive earthquakes
   - Examples: San Andreas Fault (California), North Anatolian Fault (Turkey)

3. DIVERGENT (Mid-ocean ridges):
   - Plates pull apart, new crust forms
   - Frequent but usually weaker earthquakes
   - Examples: Mid-Atlantic Ridge, East African Rift

THE RING OF FIRE
----------------
The Pacific Ring of Fire is a 40,000 km horseshoe-shaped zone where ~90% of
the world's earthquakes occur. It includes:
- Japan, Philippines, Indonesia (western edge)
- New Zealand (southwest)
- Chile, Peru, Central America (eastern edge)
- Alaska, Cascadia (northeast)

This zone exists because the Pacific Plate is being subducted beneath
surrounding plates, creating intense seismic activity.

EARTHQUAKE DEPTH
----------------
- Shallow: 0-70 km (most destructive, close to surface)
- Intermediate: 70-300 km
- Deep: 300-700 km (only in subduction zones)

Shallow earthquakes cause more damage because seismic waves haven't
been attenuated by traveling through the mantle.

================================================================================

This demo visualizes:
- Earth as a 3D sphere with continents and oceans (opaque blending)
- Subtle cloud layer generated with Perlin noise (luminous blending)
- Earthquakes as glowing vertical spikes (luminous blending, height = magnitude)
- Color gradient showing earthquake age (red = recent, blue = older)
- Real USGS data from the past 30 days

The demo showcases the opaque vs luminous blending modes:
- Opaque Earth: solid rendering, occludes objects behind it
- Luminous clouds: subtle atmospheric glow layer
- Luminous rays: additive glow, rays behind Earth are hidden but overlapping rays combine

Usage:
    python demo_earthquakes_3d.py [--days=30] [--minmag=4.5]

Controls:
    - Rotate Earth to see different earthquake zones
    - Ring of Fire should be clearly visible!
    - Ctrl+C to stop and cleanup
"""

from __future__ import annotations

DEMO_META = {
    "key": "earthquakes",
    "title": "Global Earthquakes (3D)",
    "description": "Real USGS earthquakes on a 3D Blue Marble globe, spikes scaled by magnitude and colored by time.",
    "category": "geoscience",
    "geometry": "points+lines",
    "requirements": {
        "download_mb": 5,  # approx
        "compute": "medium",
        "gpu": "none",
        "local_data": None,
    },
    "caches": ["earthquakes"],
    "outputs": ["earthquakes"],
}

import sys
import tempfile
from datetime import datetime, timedelta, timezone
from pathlib import Path

import numpy as np
import requests
from arbol import aprint, asection
from PIL import Image

from luxar import Dimension, Dimensions, LuxarZarrCompiler
from luxar.demos import cached_download, launch_viewer
from luxar.utils.paths import get_demos_output_dir

# =============================================================================
# Configuration
# =============================================================================

# Earth radius (arbitrary units for visualization)
EARTH_RADIUS = 1.0

# Cloud layer radius (slightly above Earth surface)
CLOUD_RADIUS = 1.015  # About 1.5% above Earth surface (~100km at Earth scale)

# Sphere resolution (number of points on Earth surface)
EARTH_POINTS = 120000  # High resolution for texture mapping (4x increase)
CLOUD_POINTS = 60000  # Fewer points for clouds (they're diffuse)

# NASA Blue Marble image URL (2048x1024 equirectangular projection). The former
# neo.gsfc.nasa.gov/archive URL 404s; this eoimages.gsfc.nasa.gov Blue Marble
# "land shallow topo" image is the stable NASA replacement (already 2048x1024, so
# it needs no downscale). If this fetch fails the code falls back to a crude
# rectangular land heuristic (blocky continents) — keep this URL live.
BLUE_MARBLE_URL = "https://eoimages.gsfc.nasa.gov/images/imagerecords/57000/57752/land_shallow_topo_2048.jpg"

# Downscale target (balance between quality and memory)
TEXTURE_WIDTH = 2048
TEXTURE_HEIGHT = 1024

# Default query parameters
DEFAULT_DAYS = 30  # Last 30 days
DEFAULT_MIN_MAGNITUDE = 4.5  # Magnitude 4.5+ (significant earthquakes)

# Magnitude to height scaling. Earth radius is 1.0, so at 0.18 a mag 4.5-9 quake
# produced spikes 0.8-1.6 Earth-radii long (as tall as the planet). 0.03 keeps
# them short and readable (~0.13-0.27 radii).
MAGNITUDE_SCALE = 0.03  # Height = magnitude × scale

# Color scheme for time gradient
# Recent earthquakes are hot (red/orange), older ones are cool (blue/purple)
COLOR_RECENT = np.array([1.0, 0.2, 0.0])  # Bright red-orange
COLOR_OLD = np.array([0.2, 0.1, 0.6])  # Deep purple


# =============================================================================
# Fibonacci Sphere Generation
# =============================================================================


def generate_fibonacci_sphere(n_points: int, radius: float = 1.0) -> np.ndarray:
    """Generate uniformly distributed points on a sphere using Fibonacci spiral.

    This method produces a nearly uniform distribution of points on a sphere,
    much better than using regular latitude/longitude grids which cluster
    points near the poles.

    The algorithm uses the golden ratio (φ = 1.618...) to space points
    optimally as you spiral from pole to pole.

    Args:
        n_points: Number of points to generate
        radius: Sphere radius

    Returns:
        Array of shape (n_points, 3) with (x, y, z) coordinates
    """
    points = np.zeros((n_points, 3))

    # Golden ratio
    phi = (1 + np.sqrt(5)) / 2
    golden_angle = 2 * np.pi / phi

    for i in range(n_points):
        # Map i to [-1, 1] (cosine of latitude)
        # This ensures uniform area distribution
        y = 1 - (2 * i / (n_points - 1))

        # Radius at this latitude
        r_at_y = np.sqrt(1 - y * y)

        # Longitude using golden angle
        theta = golden_angle * i

        x = r_at_y * np.cos(theta)
        z = r_at_y * np.sin(theta)

        points[i] = [x * radius, y * radius, z * radius]

    return points.astype(np.float32)


def latlon_to_xyz(lat: float, lon: float, radius: float = 1.0) -> np.ndarray:
    """Convert latitude/longitude to 3D Cartesian coordinates.

    Geographic coordinate system:
    - Latitude: -90° (South Pole) to +90° (North Pole)
    - Longitude: -180° (West) to +180° (East)

    Conversion:
    - x = r × cos(lat) × cos(lon)
    - y = r × sin(lat)
    - z = r × cos(lat) × sin(lon)

    Args:
        lat: Latitude in degrees
        lon: Longitude in degrees
        radius: Sphere radius

    Returns:
        3D coordinate (x, y, z)
    """
    lat_rad = np.radians(lat)
    lon_rad = np.radians(lon)

    x = radius * np.cos(lat_rad) * np.cos(lon_rad)
    y = radius * np.sin(lat_rad)
    z = radius * np.cos(lat_rad) * np.sin(lon_rad)

    return np.array([x, y, z])


# =============================================================================
# NASA Blue Marble Texture Mapping
# =============================================================================


def download_and_prepare_earth_texture() -> np.ndarray:
    """Download NASA Blue Marble image and prepare for texture mapping.

    Downloads the NASA Blue Marble true-color Earth image in equirectangular
    projection, downscales it to a manageable size, and returns as numpy array.

    The Blue Marble images are created from MODIS satellite observations and
    show Earth's continents, oceans, ice, and clouds in beautiful detail.

    Image format: Equirectangular projection (simple lat/lon grid)
    - Longitude: -180° to +180° maps to left-to-right (0 to width)
    - Latitude: +90° to -90° maps to top-to-bottom (0 to height)

    The image is cached (skip-if-present) under
    ``~/.cache/luxar/earthquakes/`` via the shared download helper.

    Returns:
        RGB array of shape (height, width, 3) with values 0-255
    """
    with asection("Downloading NASA Blue Marble Earth texture"):
        # Route the download through the shared cache helper so the texture
        # lands under ~/.cache/luxar/earthquakes/ and repeat runs skip the
        # network entirely when the cached file is present.
        try:
            cache_path = cached_download(
                BLUE_MARBLE_URL, "earthquakes", "blue_marble_2048x1024.jpg"
            )
        except Exception as e:
            aprint(f"❌ Error downloading Blue Marble image: {e}")
            aprint("")
            aprint("💡 Falling back to heuristic coloring...")
            aprint("   (continents will be approximate)")
            raise

        img = Image.open(cache_path)

        # Downscale to target resolution
        if img.size[0] != TEXTURE_WIDTH or img.size[1] != TEXTURE_HEIGHT:
            aprint(
                f"Downscaling to {TEXTURE_WIDTH}x{TEXTURE_HEIGHT} "
                f"(from {img.size[0]}x{img.size[1]})..."
            )
            img = img.resize((TEXTURE_WIDTH, TEXTURE_HEIGHT), Image.Resampling.LANCZOS)
            aprint("✓ Downscaled")

        # Convert to numpy array
        texture = np.array(img, dtype=np.uint8)
        aprint(f"✓ Texture ready: {texture.shape} ({texture.dtype})")

        # Close image
        img.close()

    return texture


def sample_texture_at_latlon(
    lat: float,
    lon: float,
    texture: np.ndarray,
) -> np.ndarray:
    """Sample RGB color from Earth texture at given latitude/longitude.

    Uses bilinear interpolation for smooth color sampling.

    Equirectangular projection mapping:
    - lon ∈ [-180, +180] → x ∈ [0, width]
    - lat ∈ [+90, -90] → y ∈ [0, height]

    Note: latitude is inverted (top of image = +90°, bottom = -90°)

    Args:
        lat: Latitude in degrees (-90 to +90)
        lon: Longitude in degrees (-180 to +180)
        texture: RGB texture array (height, width, 3)

    Returns:
        RGB color as float32 array [0, 1]
    """
    height, width = texture.shape[:2]

    # Convert lat/lon to texture coordinates
    # lon: -180 to +180 → 0 to width
    x = (lon + 180.0) / 360.0 * width

    # lat: +90 to -90 → 0 to height (inverted!)
    y = (90.0 - lat) / 180.0 * height

    # Clamp to valid range
    x = np.clip(x, 0, width - 1)
    y = np.clip(y, 0, height - 1)

    # Bilinear interpolation for smooth sampling
    x0 = int(np.floor(x))
    x1 = min(x0 + 1, width - 1)
    y0 = int(np.floor(y))
    y1 = min(y0 + 1, height - 1)

    # Interpolation weights
    wx = x - x0
    wy = y - y0

    # Sample four nearest pixels
    c00 = texture[y0, x0].astype(np.float32)
    c01 = texture[y0, x1].astype(np.float32)
    c10 = texture[y1, x0].astype(np.float32)
    c11 = texture[y1, x1].astype(np.float32)

    # Bilinear interpolation
    c0 = c00 * (1 - wx) + c01 * wx
    c1 = c10 * (1 - wx) + c11 * wx
    color = c0 * (1 - wy) + c1 * wy

    # Normalize to [0, 1]
    return color / 255.0


def compute_earth_colors_from_texture(
    positions: np.ndarray,
    texture: np.ndarray,
) -> np.ndarray:
    """Compute colors for Earth surface points using NASA Blue Marble texture.

    Projects sphere points onto equirectangular texture map and samples colors.

    Args:
        positions: Array of (x, y, z) coordinates on sphere
        texture: RGB texture array from Blue Marble

    Returns:
        Array of RGB colors (n_points, 3) in range [0, 1]
    """
    n_points = len(positions)
    colors = np.zeros((n_points, 3), dtype=np.float32)

    aprint(f"Mapping {n_points:,} points to Earth texture...")

    # Vectorized lat/lon calculation for speed
    x = positions[:, 0]
    y = positions[:, 1]
    z = positions[:, 2]

    # Calculate lat/lon for all points
    lats = np.degrees(np.arcsin(y / EARTH_RADIUS))
    # Negate longitude to flip horizontally (match image orientation)
    lons = -np.degrees(np.arctan2(z, x))

    # Sample texture for each point
    for i in range(n_points):
        colors[i] = sample_texture_at_latlon(lats[i], lons[i], texture)

        # Progress indicator for large point counts
        if (i + 1) % 20000 == 0:
            aprint(
                f"  Progress: {i + 1:,}/{n_points:,} ({(i + 1) / n_points * 100:.1f}%)"
            )

    aprint("✓ Texture mapping complete")

    return colors


# =============================================================================
# Land/Sea Coloring (Fallback Heuristic - used if texture download fails)
# =============================================================================


def is_land(lat: float, lon: float) -> bool:
    """Determine if a lat/lon point is on land or in ocean.

    Uses a more detailed heuristic that approximates major continents
    and excludes major oceans and seas.

    Not perfect, but significantly better than rectangular regions!

    Args:
        lat: Latitude in degrees (-90 to 90)
        lon: Longitude in degrees (-180 to 180)

    Returns:
        True if land, False if ocean
    """
    # NORTH AMERICA
    if -170 < lon < -50 and 15 < lat < 72:
        # Exclude Alaska-Bering water gaps
        if lon < -130 and lat > 60 and lon > -170:
            return False
        # Exclude Hudson Bay
        if -95 < lon < -75 and 55 < lat < 65:
            return False
        # Exclude Gulf of Mexico (rough)
        if -98 < lon < -80 and 18 < lat < 30:
            return False
        return True

    # SOUTH AMERICA (narrower at bottom, wider at top)
    if -82 < lon < -34 and -56 < lat < 13:
        # Exclude the gap at Panama
        if lon > -80 and lat > 8:
            return False
        # Taper at southern tip (Chile)
        if lat < -40 and (lon < -75 or lon > -65):
            return False
        return True

    # EUROPE (including Scandinavia, Mediterranean)
    if -11 < lon < 40 and 35 < lat < 71:
        # Exclude Mediterranean Sea (rough)
        if 10 < lon < 37 and 35 < lat < 42:
            return False
        return True

    # AFRICA (wide at top, narrow at bottom)
    if -18 < lon < 52 and -35 < lat < 37:
        # Exclude Red Sea
        if 32 < lon < 45 and 12 < lat < 28:
            return False
        # Taper at southern tip
        if lat < -30 and (lon < 15 or lon > 35):
            return False
        return True

    # MIDDLE EAST / ARABIA
    if 35 < lon < 60 and 12 < lat < 42:
        return True

    # ASIA (large and complex)
    # Central/Northern Asia
    if 40 < lon < 180 and 35 < lat < 75:
        # Exclude Sea of Okhotsk roughly
        if 140 < lon < 160 and 50 < lat < 60:
            return False
        return True

    # Indian Subcontinent
    if 68 < lon < 90 and 8 < lat < 35:
        return True

    # Southeast Asia and Indonesia
    if 95 < lon < 140 and -10 < lat < 25:
        # This is complex (many islands), but approximate mainland
        if lat > 10:  # Mainland
            return True
        # Indonesia - spotty coverage is fine
        if 100 < lon < 125:
            return True

    # AUSTRALIA
    if 113 < lon < 154 and -44 < lat < -10:
        # Exclude Gulf of Carpentaria (rough)
        if 135 < lon < 142 and -15 < lat < -10:
            return False
        return True

    # NEW ZEALAND (two main islands)
    if 166 < lon < 179 and -47 < lat < -34:
        return True

    # ANTARCTICA (southern cap)
    if lat < -60:
        return True

    # GREENLAND
    if -75 < lon < -12 and 60 < lat < 84:
        return True

    # JAPAN (rough)
    if 128 < lon < 146 and 30 < lat < 46:
        return True

    # UK and Ireland
    if -11 < lon < 2 and 50 < lat < 60:
        return True

    # Madagascar
    if 43 < lon < 51 and -26 < lat < -12:
        return True

    # Default: ocean
    return False


def compute_earth_colors(positions: np.ndarray) -> np.ndarray:
    """Compute colors for Earth surface points (continents vs oceans).

    Uses a detailed heuristic that approximates real continental boundaries.
    Much more accurate than simple rectangular regions!

    Args:
        positions: Array of (x, y, z) coordinates on sphere

    Returns:
        Array of RGB colors (n_points, 3)
    """
    n_points = len(positions)
    colors = np.zeros((n_points, 3), dtype=np.float32)

    # Convert back to lat/lon for land/sea determination
    for i in range(n_points):
        x, y, z = positions[i]

        # Calculate lat/lon
        lat = np.degrees(np.arcsin(y / EARTH_RADIUS))
        lon = np.degrees(np.arctan2(z, x))

        if is_land(lat, lon):
            # Land: varied earth tones (browns, greens)
            base_color = np.array([0.35, 0.45, 0.25])  # Greenish-brown
            # Add variation based on latitude (greener near equator, browner at poles)
            lat_factor = 1.0 - abs(lat) / 90.0
            base_color[1] += 0.1 * lat_factor  # More green near equator
            # Add random variation for texture
            variation = np.random.uniform(-0.08, 0.08, 3)
            colors[i] = base_color + variation
        else:
            # Ocean: varied blue depths
            base_color = np.array([0.08, 0.18, 0.35])  # Deep ocean blue
            # Add depth variation
            variation = np.random.uniform(-0.04, 0.04, 3)
            colors[i] = base_color + variation

    return np.clip(colors, 0, 1)


# =============================================================================
# Perlin Noise for Cloud Generation
# =============================================================================


def _fade(t: np.ndarray) -> np.ndarray:
    """Perlin noise fade function: 6t^5 - 15t^4 + 10t^3."""
    return t * t * t * (t * (t * 6 - 15) + 10)


def _lerp(a: np.ndarray, b: np.ndarray, t: np.ndarray) -> np.ndarray:
    """Linear interpolation."""
    return a + t * (b - a)


def _grad3d(hash_val: int, x: float, y: float, z: float) -> float:
    """3D gradient function for Perlin noise."""
    h = hash_val & 15
    u = x if h < 8 else y
    v = y if h < 4 else (x if h in (12, 14) else z)
    return (u if (h & 1) == 0 else -u) + (v if (h & 2) == 0 else -v)


# Permutation table for Perlin noise (standard permutation)
_PERM = [
    151,
    160,
    137,
    91,
    90,
    15,
    131,
    13,
    201,
    95,
    96,
    53,
    194,
    233,
    7,
    225,
    140,
    36,
    103,
    30,
    69,
    142,
    8,
    99,
    37,
    240,
    21,
    10,
    23,
    190,
    6,
    148,
    247,
    120,
    234,
    75,
    0,
    26,
    197,
    62,
    94,
    252,
    219,
    203,
    117,
    35,
    11,
    32,
    57,
    177,
    33,
    88,
    237,
    149,
    56,
    87,
    174,
    20,
    125,
    136,
    171,
    168,
    68,
    175,
    74,
    165,
    71,
    134,
    139,
    48,
    27,
    166,
    77,
    146,
    158,
    231,
    83,
    111,
    229,
    122,
    60,
    211,
    133,
    230,
    220,
    105,
    92,
    41,
    55,
    46,
    245,
    40,
    244,
    102,
    143,
    54,
    65,
    25,
    63,
    161,
    1,
    216,
    80,
    73,
    209,
    76,
    132,
    187,
    208,
    89,
    18,
    169,
    200,
    196,
    135,
    130,
    116,
    188,
    159,
    86,
    164,
    100,
    109,
    198,
    173,
    186,
    3,
    64,
    52,
    217,
    226,
    250,
    124,
    123,
    5,
    202,
    38,
    147,
    118,
    126,
    255,
    82,
    85,
    212,
    207,
    206,
    59,
    227,
    47,
    16,
    58,
    17,
    182,
    189,
    28,
    42,
    223,
    183,
    170,
    213,
    119,
    248,
    152,
    2,
    44,
    154,
    163,
    70,
    221,
    153,
    101,
    155,
    167,
    43,
    172,
    9,
    129,
    22,
    39,
    253,
    19,
    98,
    108,
    110,
    79,
    113,
    224,
    232,
    178,
    185,
    112,
    104,
    218,
    246,
    97,
    228,
    251,
    34,
    242,
    193,
    238,
    210,
    144,
    12,
    191,
    179,
    162,
    241,
    81,
    51,
    145,
    235,
    249,
    14,
    239,
    107,
    49,
    192,
    214,
    31,
    181,
    199,
    106,
    157,
    184,
    84,
    204,
    176,
    115,
    121,
    50,
    45,
    127,
    4,
    150,
    254,
    138,
    236,
    205,
    93,
    222,
    114,
    67,
    29,
    24,
    72,
    243,
    141,
    128,
    195,
    78,
    66,
    215,
    61,
    156,
    180,
]
_PERM = _PERM + _PERM  # Double for overflow handling


def perlin_noise_3d(x: float, y: float, z: float) -> float:
    """Generate 3D Perlin noise value at given coordinates.

    Returns a value roughly in range [-1, 1].

    Args:
        x, y, z: 3D coordinates

    Returns:
        Noise value
    """
    # Find unit cube containing point
    xi = int(np.floor(x)) & 255
    yi = int(np.floor(y)) & 255
    zi = int(np.floor(z)) & 255

    # Find relative position in cube
    xf = x - np.floor(x)
    yf = y - np.floor(y)
    zf = z - np.floor(z)

    # Compute fade curves
    u = _fade(np.array([xf]))[0]
    v = _fade(np.array([yf]))[0]
    w = _fade(np.array([zf]))[0]

    # Hash coordinates of cube corners
    aaa = _PERM[_PERM[_PERM[xi] + yi] + zi]
    aba = _PERM[_PERM[_PERM[xi] + yi + 1] + zi]
    aab = _PERM[_PERM[_PERM[xi] + yi] + zi + 1]
    abb = _PERM[_PERM[_PERM[xi] + yi + 1] + zi + 1]
    baa = _PERM[_PERM[_PERM[xi + 1] + yi] + zi]
    bba = _PERM[_PERM[_PERM[xi + 1] + yi + 1] + zi]
    bab = _PERM[_PERM[_PERM[xi + 1] + yi] + zi + 1]
    bbb = _PERM[_PERM[_PERM[xi + 1] + yi + 1] + zi + 1]

    # Blend gradients
    x1 = _lerp(
        np.array([_grad3d(aaa, xf, yf, zf)]),
        np.array([_grad3d(baa, xf - 1, yf, zf)]),
        np.array([u]),
    )[0]
    x2 = _lerp(
        np.array([_grad3d(aba, xf, yf - 1, zf)]),
        np.array([_grad3d(bba, xf - 1, yf - 1, zf)]),
        np.array([u]),
    )[0]
    y1 = _lerp(np.array([x1]), np.array([x2]), np.array([v]))[0]

    x1 = _lerp(
        np.array([_grad3d(aab, xf, yf, zf - 1)]),
        np.array([_grad3d(bab, xf - 1, yf, zf - 1)]),
        np.array([u]),
    )[0]
    x2 = _lerp(
        np.array([_grad3d(abb, xf, yf - 1, zf - 1)]),
        np.array([_grad3d(bbb, xf - 1, yf - 1, zf - 1)]),
        np.array([u]),
    )[0]
    y2 = _lerp(np.array([x1]), np.array([x2]), np.array([v]))[0]

    return _lerp(np.array([y1]), np.array([y2]), np.array([w]))[0]


def fractal_noise_3d(
    x: float,
    y: float,
    z: float,
    octaves: int = 4,
    persistence: float = 0.5,
    lacunarity: float = 2.0,
) -> float:
    """Generate fractal (multi-octave) 3D Perlin noise.

    Combines multiple octaves of noise at different frequencies
    to create more natural-looking patterns.

    Args:
        x, y, z: 3D coordinates
        octaves: Number of noise layers to combine
        persistence: Amplitude decay per octave (0.5 = halve each time)
        lacunarity: Frequency increase per octave (2.0 = double each time)

    Returns:
        Combined noise value roughly in range [-1, 1]
    """
    total = 0.0
    amplitude = 1.0
    frequency = 1.0
    max_value = 0.0

    for _ in range(octaves):
        total += (
            perlin_noise_3d(x * frequency, y * frequency, z * frequency) * amplitude
        )
        max_value += amplitude
        amplitude *= persistence
        frequency *= lacunarity

    return total / max_value if max_value > 0 else 0.0


# =============================================================================
# Cloud Layer Generation
# =============================================================================


def generate_cloud_layer(
    n_points: int,
    radius: float,
    noise_scale: float = 3.0,
    threshold: float = 0.1,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Generate a cloud layer using Perlin noise.

    Creates cloud points distributed on a sphere with density and opacity
    controlled by Perlin noise to create realistic cloud patterns.

    Args:
        n_points: Base number of points to sample
        radius: Radius of the cloud sphere
        noise_scale: Scale of the noise (higher = finer detail)
        threshold: Noise threshold below which clouds don't appear

    Returns:
        Tuple of (positions, colors, radii, sharpness) for cloud points
    """
    aprint(f"Generating cloud layer ({n_points:,} candidate points)...")

    # Generate candidate positions using Fibonacci sphere
    candidates = generate_fibonacci_sphere(n_points, radius)

    # Evaluate noise at each point to determine cloud density
    cloud_positions = []
    cloud_colors = []
    cloud_radii = []
    cloud_sharpness = []

    # Cloud color: subtle white with slight blue tint
    base_color = np.array([0.9, 0.92, 1.0])

    for i in range(n_points):
        x, y, z = candidates[i]

        # Sample fractal noise for cloud density
        noise_val = fractal_noise_3d(
            x * noise_scale,
            y * noise_scale,
            z * noise_scale,
            octaves=4,
            persistence=0.5,
        )

        # Map noise to cloud density (only keep positive values above threshold)
        # This creates gaps in the clouds
        density = (noise_val + 1.0) / 2.0  # Map from [-1,1] to [0,1]

        # Apply threshold and non-linear mapping for more interesting patterns
        if density > threshold:
            # Keep this point as a cloud
            cloud_positions.append(candidates[i])

            # Vary opacity based on density (denser = more opaque)
            opacity_factor = (density - threshold) / (1.0 - threshold)
            opacity_factor = opacity_factor**0.7  # Non-linear for softer edges

            # Color with subtle variation
            variation = np.random.uniform(-0.02, 0.02, 3)
            color = base_color * (0.3 + 0.7 * opacity_factor) + variation
            cloud_colors.append(np.clip(color, 0, 1))

            # Radius varies with density (denser = larger points)
            base_radius = 0.006 + 0.004 * opacity_factor
            cloud_radii.append(base_radius)

            # Soft, diffuse sharpness (normalized [0, 1] knob; low = peakier/softer)
            cloud_sharpness.append(0.35)

        # Progress indicator
        if (i + 1) % 20000 == 0:
            aprint(f"  Progress: {i + 1:,}/{n_points:,}")

    n_clouds = len(cloud_positions)
    aprint(
        f"✓ Generated {n_clouds:,} cloud points ({n_clouds / n_points * 100:.1f}% density)"
    )

    if n_clouds == 0:
        return (
            np.array([]).reshape(0, 3).astype(np.float32),
            np.array([]).reshape(0, 3).astype(np.float32),
            np.array([]).astype(np.float32),
            np.array([]).astype(np.float32),
        )

    return (
        np.array(cloud_positions, dtype=np.float32),
        np.array(cloud_colors, dtype=np.float32),
        np.array(cloud_radii, dtype=np.float32),
        np.array(cloud_sharpness, dtype=np.float32),
    )


# =============================================================================
# USGS Earthquake Data
# =============================================================================


def download_earthquake_data(
    days: int = 30,
    min_magnitude: float = 4.5,
) -> list[dict]:
    """Download earthquake data from USGS API.

    USGS provides real-time earthquake data through their Earthquake Catalog API.
    Data includes location, magnitude, depth, time, and other properties.

    API Documentation: https://earthquake.usgs.gov/fdsnws/event/1/

    Args:
        days: Number of days to look back
        min_magnitude: Minimum magnitude to include

    Returns:
        List of earthquake dictionaries with keys:
        - latitude, longitude, depth, magnitude, time, place
    """
    with asection("Downloading USGS earthquake data"):
        # Calculate date range
        end_date = datetime.now(timezone.utc)
        start_date = end_date - timedelta(days=days)

        # Format dates for API (ISO 8601)
        start_str = start_date.strftime("%Y-%m-%d")
        end_str = end_date.strftime("%Y-%m-%d")

        # Construct API URL
        url = (
            f"https://earthquake.usgs.gov/fdsnws/event/1/query?"
            f"format=geojson&"
            f"starttime={start_str}&"
            f"endtime={end_str}&"
            f"minmagnitude={min_magnitude}"
        )

        aprint("Querying USGS API...")
        aprint(f"  Date range: {start_str} to {end_str}")
        aprint(f"  Minimum magnitude: {min_magnitude}")

        try:
            response = requests.get(url, timeout=30)
            response.raise_for_status()
            data = response.json()

            earthquakes = []
            for feature in data["features"]:
                props = feature["properties"]
                coords = feature["geometry"]["coordinates"]

                earthquakes.append(
                    {
                        "longitude": coords[0],
                        "latitude": coords[1],
                        "depth": coords[2],  # km
                        "magnitude": props["mag"],
                        "time": props["time"],  # Unix timestamp (ms)
                        "place": props["place"],
                    }
                )

            aprint(f"✓ Downloaded {len(earthquakes)} earthquakes")

            if len(earthquakes) > 0:
                mags = [eq["magnitude"] for eq in earthquakes]
                aprint(f"  Magnitude range: {min(mags):.1f} to {max(mags):.1f}")

                # Show largest earthquake
                largest = max(earthquakes, key=lambda x: x["magnitude"])
                aprint(f"  Largest: M{largest['magnitude']:.1f} - {largest['place']}")

            return earthquakes

        except requests.exceptions.RequestException as e:
            aprint(f"❌ Error downloading data: {e}")
            aprint("")
            aprint("💡 Possible issues:")
            aprint("   - No internet connection")
            aprint("   - USGS API is down")
            aprint("   - Firewall blocking requests")
            raise


# =============================================================================
# Earthquake Visualization Generation
# =============================================================================


def generate_earthquake_lines(
    earthquakes: list[dict],
    current_time: float,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Generate vertical spike lines for earthquakes.

    Each earthquake is represented as a vertical line extending radially
    outward from the Earth's surface. The line:
    - Starts at the surface
    - Extends outward with height proportional to magnitude
    - Is colored based on how recent the earthquake was

    Args:
        earthquakes: List of earthquake data dictionaries
        current_time: Current Unix timestamp (ms) for age calculation

    Returns:
        Tuple of (vertices, widths, colors, sharpness)
        - vertices: (N×2, 3) line segment endpoints
        - widths: (N×2,) line widths
        - colors: (N×2, 3) RGB colors
        - sharpness: (N×2,) line edge sharpness
    """
    if len(earthquakes) == 0:
        return (
            np.array([]).reshape(0, 3),
            np.array([]),
            np.array([]).reshape(0, 3),
            np.array([]),
        )

    vertices = []
    widths = []
    colors = []
    sharpness = []

    # Calculate time range for color mapping
    times = [eq["time"] for eq in earthquakes]
    oldest_time = min(times)
    time_range = current_time - oldest_time

    for eq in earthquakes:
        lat = eq["latitude"]
        lon = eq["longitude"]
        mag = eq["magnitude"]
        eq_time = eq["time"]

        # Base position on Earth surface
        # Negate longitude to match the flipped Earth texture coordinate system
        base_pos = latlon_to_xyz(lat, -lon, EARTH_RADIUS)

        # Calculate spike height based on magnitude
        # Higher magnitude = taller spike
        spike_height = mag * MAGNITUDE_SCALE

        # End position (radially outward)
        direction = base_pos / np.linalg.norm(base_pos)  # Unit vector
        end_pos = base_pos + direction * spike_height

        # Line segment: base to tip
        vertices.extend([base_pos, end_pos])

        # Line width scales with magnitude
        base_width = 0.004 + 0.006 * (mag - 4.5) / 5.0  # 4.5-9.5 mag range
        widths.extend([base_width * 1.2, base_width * 0.3])  # Taper to point

        # Color based on age (recent = hot, old = cool)
        age = current_time - eq_time
        age_fraction = age / time_range if time_range > 0 else 0
        age_fraction = np.clip(age_fraction, 0, 1)

        # Interpolate between recent (red) and old (purple)
        color = COLOR_RECENT * (1 - age_fraction) + COLOR_OLD * age_fraction

        # Boost brightness for larger magnitudes
        brightness_boost = 1.0 + 0.3 * (mag - 4.5) / 5.0
        color = np.clip(color * brightness_boost, 0, 1)

        colors.extend([color, color * 0.7])  # Fade toward tip

        # Sharp lines for clarity
        sharpness.extend([0.5, 0.5])

    return (
        np.array(vertices, dtype=np.float32),
        np.array(widths, dtype=np.float32),
        np.array(colors, dtype=np.float32),
        np.array(sharpness, dtype=np.float32),
    )


# =============================================================================
# Main Scene Generation
# =============================================================================


def generate_earthquake_scene(
    output_path: Path,
    days: int = DEFAULT_DAYS,
    min_magnitude: float = DEFAULT_MIN_MAGNITUDE,
) -> tuple[int, int, int]:
    """Generate complete earthquake visualization scene.

    Returns:
        Tuple of (earth_points, cloud_points, earthquake_lines)
    """
    # Download earthquake data
    earthquakes = download_earthquake_data(days, min_magnitude)

    if len(earthquakes) == 0:
        aprint("⚠️  No earthquakes found matching criteria")
        return 0, 0, 0

    # Download and prepare Earth texture (cached under ~/.cache/luxar/earthquakes/)
    try:
        earth_texture = download_and_prepare_earth_texture()
        use_texture = True
    except Exception:
        aprint("⚠️  Falling back to heuristic coloring")
        use_texture = False

    # Generate Earth sphere
    with asection(f"Generating Earth sphere ({EARTH_POINTS:,} points)"):
        earth_positions = generate_fibonacci_sphere(EARTH_POINTS, EARTH_RADIUS)

        if use_texture:
            earth_colors = compute_earth_colors_from_texture(
                earth_positions, earth_texture
            )
        else:
            earth_colors = compute_earth_colors(earth_positions)

        aprint(f"✓ Generated Earth surface with {EARTH_POINTS:,} points")

    # Generate cloud layer
    with asection("Generating cloud layer"):
        cloud_positions, cloud_colors, cloud_radii, cloud_sharpness = (
            generate_cloud_layer(
                CLOUD_POINTS,
                CLOUD_RADIUS,
                noise_scale=3.5,  # Moderate detail
                threshold=0.35,  # Sparse clouds (only ~40% coverage)
            )
        )
        n_clouds = len(cloud_positions)

    # Generate earthquake lines
    with asection("Generating earthquake visualization"):
        current_time = datetime.now(timezone.utc).timestamp() * 1000  # Convert to ms
        line_verts, line_widths, line_colors, line_sharp = generate_earthquake_lines(
            earthquakes, current_time
        )
        n_lines = len(line_verts) // 2
        aprint(f"✓ Generated {n_lines:,} earthquake spikes")

    # Write to Luxar format
    with asection("Writing to Zarr"):
        dims = Dimensions(
            [
                Dimension("x", unit="R⊕", display=True),  # Earth radii
                Dimension("y", unit="R⊕", display=True),
                Dimension("z", unit="R⊕", display=True),
            ]
        )

        with LuxarZarrCompiler(output_path) as compiler:
            scene = compiler.create_scene(dimensions=dims)

            # Add Earth surface
            earth_radii = np.full(len(earth_positions), 0.003, dtype=np.float32)
            earth_sharpness = np.full(len(earth_positions), 0.55, dtype=np.float32)

            # Earth surface uses opaque blending - solid rendering with depth write
            # This means the Earth will occlude earthquake rays behind it
            scene.add_points(
                "Earth",
                positions=earth_positions,
                colors=earth_colors,
                radii=earth_radii,
                sharpness=earth_sharpness,
                opacity=1.0,
                blending_mode="opaque",
                layer=True,
            )

            # Add cloud layer with luminous blending - subtle atmospheric glow
            # Clouds are semi-transparent and glow softly above the Earth surface
            if len(cloud_positions) > 0:
                scene.add_points(
                    "Clouds",
                    positions=cloud_positions,
                    colors=cloud_colors,
                    radii=cloud_radii,
                    sharpness=cloud_sharpness,
                    opacity=0.15,  # Very subtle - don't overwhelm the visualization
                    blending_mode="luminous",
                    layer=True,
                    intensity=0.5,
                )

            # Add earthquake lines with luminous blending - glowing additive effect
            # Luminous objects are occluded by opaque Earth but add together
            # where they overlap (multiple earthquake rays at same location glow brighter)
            if len(line_verts) > 0:
                # Hover labels: "M5.3 — 50 km W of Port Vila, Vanuatu"
                # Two vertices per earthquake (segment base + tip), same label for both
                eq_labels = []
                for eq in earthquakes:
                    mag = eq["magnitude"]
                    place = eq.get("place", "Unknown location") or "Unknown location"
                    label = f"M{mag:.1f} — {place}"
                    eq_labels.extend([label, label])  # base vertex + tip vertex

                scene.add_lines(
                    "Earthquakes",
                    vertices=line_verts,
                    widths=line_widths,
                    colors=line_colors,
                    sharpness=line_sharp,
                    line_type="segments",
                    blending_mode="luminous",
                    layer=True,
                    labels=eq_labels,
                )

            # Overlay annotations
            scene.add_text(
                "Global Earthquakes",
                position=(0.02, 0.02),
                font_size=0.055,
                anchor="top-left",
                color="rgba(255,255,255,0.6)",
                blend_mode="difference",
            )
            scene.add_text(
                f"{n_lines:,} earthquakes • Magnitude 4.5+ • USGS",
                position=(0.98, 0.97),
                font_size=0.012,
                anchor="bottom-right",
                color="rgba(200,200,200,0.45)",
            )

        aprint(f"✓ Written to {output_path}")

    return len(earth_positions), n_clouds, n_lines


# =============================================================================
# Main Entry Point
# =============================================================================


def main() -> None:
    """Main demo entry point."""
    days = DEFAULT_DAYS
    min_mag = DEFAULT_MIN_MAGNITUDE

    # Parse command line arguments
    for arg in sys.argv[1:]:
        if arg.startswith("--days="):
            days = int(arg.split("=")[1])
        elif arg.startswith("--minmag="):
            min_mag = float(arg.split("=")[1])

    aprint("=" * 70)
    aprint("GLOBAL EARTHQUAKE VISUALIZATION")
    aprint("=" * 70)
    aprint("")
    aprint("Real-time earthquake data from USGS visualized on 3D Earth")
    aprint("")
    aprint("  Data Source: U.S. Geological Survey")
    aprint("  https://earthquake.usgs.gov/")
    aprint("")
    aprint("  Query parameters:")
    aprint(f"    • Last {days} days")
    aprint(f"    • Minimum magnitude: {min_mag}")
    aprint("")
    aprint("  Visualization:")
    aprint("    • Earth sphere: continents/oceans (opaque - solid surface)")
    aprint("    • Cloud layer: Perlin noise patterns (luminous - subtle glow)")
    aprint("    • Vertical spikes: earthquake locations (luminous - glowing)")
    aprint("    • Spike height: magnitude (taller = stronger)")
    aprint("    • Spike color: time (red = recent, purple = older)")
    aprint("    • Blending: Earth occludes rays behind it, rays add together")
    aprint("")
    aprint("  What to look for:")
    aprint("    🔥 Ring of Fire - Pacific Ocean rim (most activity)")
    aprint("    🌊 Subduction zones - convergent plate boundaries")
    aprint("    ⚡ Transform faults - strike-slip boundaries")
    aprint("    🌋 Volcanic regions - often correlate with earthquakes")
    aprint("")

    # If --no-serve, use persistent directory; otherwise temp for auto-cleanup
    if "--no-serve" in sys.argv:
        output_path = get_demos_output_dir() / "earthquakes.luxar.zarr"
        try:
            total_points, total_clouds, total_lines = generate_earthquake_scene(
                output_path, days=days, min_magnitude=min_mag
            )
            if total_points == 0:
                aprint("\n❌ No data generated")
                return
        except Exception as e:
            aprint(f"\n❌ Error: {e}")
            sys.exit(1)
        aprint(f"Dataset generated at {output_path}")
        return

    # Use temporary directory for serving (auto-cleanup on exit)
    with tempfile.TemporaryDirectory(prefix="luxar_demo_earthquakes_") as tmpdir:
        output_path = Path(tmpdir) / "earthquakes.luxar.zarr"

        try:
            total_points, total_clouds, total_lines = generate_earthquake_scene(
                output_path, days=days, min_magnitude=min_mag
            )

            if total_points == 0:
                aprint("\n❌ No data generated")
                return

            aprint("")
            aprint("=" * 70)
            aprint(
                f"SCENE COMPLETE: {total_points:,} points, {total_lines:,} earthquakes"
            )
            aprint("=" * 70)
            aprint("")
            aprint("Earthquake Magnitude Scale:")
            aprint("  • 4-5: Light (minor damage possible)")
            aprint("  • 5-6: Moderate (building damage)")
            aprint("  • 6-7: Strong (destructive)")
            aprint("  • 7-8: Major (serious widespread damage)")
            aprint("  • 8-9: Great (catastrophic)")
            aprint("  • 9+: Megaquake (rare, extreme destruction)")
            aprint("")
            aprint("Major Earthquake Zones:")
            aprint("  Pacific Ring of Fire (90% of world's quakes)")
            aprint("  Himalayan belt (India-Asia collision)")
            aprint("  Mid-Atlantic Ridge (divergent boundary)")
            aprint("  San Andreas Fault (California transform)")
            aprint("")
            aprint("=" * 70)
            aprint("LAUNCHING VIEWER")
            aprint("=" * 70)
            aprint("Rotate the Earth to explore seismic activity patterns!")
            aprint("Press Ctrl+C when done.")
            aprint("")

            launch_viewer(output_path)

        except KeyboardInterrupt:
            aprint("\n🛑 Stopping demo...")
        except Exception as e:
            aprint(f"\n❌ Error: {e}")
            sys.exit(1)

    aprint("\nCleanup complete - temporary files removed")


if __name__ == "__main__":
    main()
