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
- Earth as a 3D sphere with continents and oceans
- Earthquakes as glowing vertical spikes (height = magnitude)
- Color gradient showing earthquake age (red = recent, blue = older)
- Real USGS data from the past 30 days

Usage:
    python demo_earthquakes_3d.py [--days=30] [--minmag=4.5]

Controls:
    - Rotate Earth to see different earthquake zones
    - Ring of Fire should be clearly visible!
    - Ctrl+C to stop and cleanup
"""

from __future__ import annotations

import subprocess
import sys
import tempfile
from datetime import datetime, timedelta, timezone
from pathlib import Path

import numpy as np
import requests
from arbol import aprint, asection
from PIL import Image

from luxar import Dimension, Dimensions, LuxarZarrCompiler

# =============================================================================
# Configuration
# =============================================================================

# Earth radius (arbitrary units for visualization)
EARTH_RADIUS = 1.0

# Sphere resolution (number of points on Earth surface)
EARTH_POINTS = 120000  # High resolution for texture mapping (4x increase)

# NASA Blue Marble image URL (5400x2700 equirectangular projection)
BLUE_MARBLE_URL = "https://neo.gsfc.nasa.gov/archive/bluemarble/bmng/world_8km/world.200401.3x5400x2700.jpg"

# Downscale target (balance between quality and memory)
TEXTURE_WIDTH = 2048
TEXTURE_HEIGHT = 1024

# Default query parameters
DEFAULT_DAYS = 30  # Last 30 days
DEFAULT_MIN_MAGNITUDE = 4.5  # Magnitude 4.5+ (significant earthquakes)

# Magnitude to height scaling
MAGNITUDE_SCALE = 0.18  # Height = magnitude × scale

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


def download_and_prepare_earth_texture(
    cache_path: Path | None = None,
) -> np.ndarray:
    """Download NASA Blue Marble image and prepare for texture mapping.

    Downloads the NASA Blue Marble true-color Earth image in equirectangular
    projection, downscales it to a manageable size, and returns as numpy array.

    The Blue Marble images are created from MODIS satellite observations and
    show Earth's continents, oceans, ice, and clouds in beautiful detail.

    Image format: Equirectangular projection (simple lat/lon grid)
    - Longitude: -180° to +180° maps to left-to-right (0 to width)
    - Latitude: +90° to -90° maps to top-to-bottom (0 to height)

    Args:
        cache_path: Optional path to cache the downloaded image

    Returns:
        RGB array of shape (height, width, 3) with values 0-255
    """
    with asection("Downloading NASA Blue Marble Earth texture"):
        # Check cache first
        if cache_path and cache_path.exists():
            aprint(f"Loading cached texture from {cache_path}")
            img = Image.open(cache_path)
        else:
            aprint("Downloading from NASA NEO...")
            aprint(f"  URL: {BLUE_MARBLE_URL}")
            aprint("  Original size: 5400x2700 pixels (~30 MB)")

            try:
                response = requests.get(BLUE_MARBLE_URL, timeout=120, stream=True)
                response.raise_for_status()

                # Load image from response
                from io import BytesIO

                img = Image.open(BytesIO(response.content))
                aprint(f"✓ Downloaded: {img.size[0]}x{img.size[1]} pixels")

                # Save to cache if specified
                if cache_path:
                    cache_path.parent.mkdir(parents=True, exist_ok=True)
                    img.save(cache_path, "JPEG", quality=85)
                    aprint(f"✓ Cached to {cache_path}")

            except requests.exceptions.RequestException as e:
                aprint(f"❌ Error downloading Blue Marble image: {e}")
                aprint("")
                aprint("💡 Falling back to heuristic coloring...")
                aprint("   (continents will be approximate)")
                raise

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
        sharpness.extend([2.0, 2.0])

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
) -> tuple[int, int]:
    """Generate complete earthquake visualization scene.

    Returns:
        Tuple of (total_points, total_line_segments)
    """
    # Download earthquake data
    earthquakes = download_earthquake_data(days, min_magnitude)

    if len(earthquakes) == 0:
        aprint("⚠️  No earthquakes found matching criteria")
        return 0, 0

    # Download and prepare Earth texture
    cache_dir = Path.home() / ".cache" / "luxar"
    cache_file = cache_dir / "blue_marble_2048x1024.jpg"

    try:
        earth_texture = download_and_prepare_earth_texture(cache_file)
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
            earth_radii = np.full(len(earth_positions), 0.01, dtype=np.float32)
            earth_sharpness = np.full(len(earth_positions), 3.0, dtype=np.float32)

            scene.add_points(
                "Earth",
                positions=earth_positions,
                colors=earth_colors,
                radii=earth_radii,
                sharpness=earth_sharpness,
                opacity=0.95,
            )

            # Add earthquake lines
            if len(line_verts) > 0:
                scene.add_lines(
                    "Earthquakes",
                    vertices=line_verts,
                    widths=line_widths,
                    colors=line_colors,
                    sharpness=line_sharp,
                    line_type="segments",
                )

        aprint(f"✓ Written to {output_path}")

    return len(earth_positions), n_lines


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
    aprint("    • Earth sphere: continents (green) and oceans (blue)")
    aprint("    • Vertical spikes: earthquake locations")
    aprint("    • Spike height: magnitude (taller = stronger)")
    aprint("    • Spike color: time (red = recent, purple = older)")
    aprint("")
    aprint("  What to look for:")
    aprint("    🔥 Ring of Fire - Pacific Ocean rim (most activity)")
    aprint("    🌊 Subduction zones - convergent plate boundaries")
    aprint("    ⚡ Transform faults - strike-slip boundaries")
    aprint("    🌋 Volcanic regions - often correlate with earthquakes")
    aprint("")

    with tempfile.TemporaryDirectory(prefix="luxar_demo_earthquakes_") as tmpdir:
        output_path = Path(tmpdir) / "earthquakes.zarr"

        try:
            total_points, total_lines = generate_earthquake_scene(
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
            aprint("  🔥 Pacific Ring of Fire (90% of world's quakes)")
            aprint("  🗻 Himalayan belt (India-Asia collision)")
            aprint("  🌊 Mid-Atlantic Ridge (divergent boundary)")
            aprint("  ⚡ San Andreas Fault (California transform)")
            aprint("")
            aprint("=" * 70)
            aprint("LAUNCHING VIEWER")
            aprint("=" * 70)
            aprint("Rotate the Earth to explore seismic activity patterns!")
            aprint("Press Ctrl+C when done.")
            aprint("")

            if "--no-serve" in sys.argv:
                aprint("✓ Dataset generated successfully (--no-serve mode)")
                return

            subprocess.run(
                ["luxar", "serve", str(output_path), "--viewer", "--open"],
                check=True,
            )

        except KeyboardInterrupt:
            aprint("\n🛑 Stopping demo...")
        except subprocess.CalledProcessError as e:
            aprint(f"\n❌ Error launching viewer: {e}")
            aprint("💡 Make sure viewer is built:")
            aprint("   cd packages/luxar-viewer && pnpm build")
            sys.exit(1)
        except FileNotFoundError as e:
            if "luxar" in str(e):
                aprint("\n❌ Error: 'luxar' command not found")
                aprint("💡 Install luxar: pip install -e .")
            else:
                aprint(f"\n❌ Error: {e}")
            sys.exit(1)
        except Exception as e:
            aprint(f"\n❌ Error: {e}")
            sys.exit(1)

    aprint("\n✓ Cleanup complete - temporary files removed")


if __name__ == "__main__":
    main()
