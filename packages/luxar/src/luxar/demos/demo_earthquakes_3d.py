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

from luxar import Dimension, Dimensions, LuxarZarrCompiler

# =============================================================================
# Configuration
# =============================================================================

# Earth radius (arbitrary units for visualization)
EARTH_RADIUS = 1.0

# Sphere resolution (number of points on Earth surface)
EARTH_POINTS = 30000  # Provides smooth sphere

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
# Land/Sea Coloring (Simple Heuristic)
# =============================================================================


def compute_earth_colors(positions: np.ndarray) -> np.ndarray:
    """Compute colors for Earth surface points (continents vs oceans).

    Uses a simple heuristic based on latitude/longitude patterns.
    Not geographically accurate but provides visual distinction.

    For a production version, you could use:
    - global-land-mask library
    - Coastline shapefiles
    - Satellite texture maps

    This heuristic roughly captures:
    - Pacific Ocean (large area)
    - Atlantic Ocean
    - Major continents

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

        # Simple heuristic for land vs sea
        # Default: ocean (blue)
        is_land = False

        # North America: rough approximation
        if -170 < lon < -50 and 15 < lat < 70:
            is_land = True

        # South America
        if -85 < lon < -35 and -55 < lat < 12:
            is_land = True

        # Europe and Africa
        if -15 < lon < 50 and -35 < lat < 70:
            is_land = True

        # Asia
        if 50 < lon < 150 and 0 < lat < 75:
            is_land = True

        # Australia
        if 110 < lon < 160 and -45 < lat < -10:
            is_land = True

        if is_land:
            # Land: greenish-brown
            colors[i] = np.array([0.3, 0.5, 0.2]) + np.random.uniform(-0.1, 0.1, 3)
        else:
            # Ocean: deep blue
            colors[i] = np.array([0.1, 0.2, 0.4]) + np.random.uniform(-0.05, 0.05, 3)

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
    with asection(f"Downloading USGS earthquake data"):
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

        aprint(f"Querying USGS API...")
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
        base_pos = latlon_to_xyz(lat, lon, EARTH_RADIUS)

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

    # Generate Earth sphere
    with asection(f"Generating Earth sphere ({EARTH_POINTS:,} points)"):
        earth_positions = generate_fibonacci_sphere(EARTH_POINTS, EARTH_RADIUS)
        earth_colors = compute_earth_colors(earth_positions)
        aprint(f"✓ Generated Earth surface")

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
    aprint(f"  Query parameters:")
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
            aprint(f"SCENE COMPLETE: {total_points:,} points, {total_lines:,} earthquakes")
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
