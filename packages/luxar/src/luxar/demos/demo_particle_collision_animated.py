#!/usr/bin/env python3
"""Animated Particle Collision Detector Visualization

A time-animated visualization of particle physics collisions inspired by CERN's
ATLAS and CMS detectors. Watch particle tracks grow outward from the collision
vertex as time advances, showing the event unfolding in real-time.

================================================================================
PHYSICS BACKGROUND
================================================================================

At particle colliders like the Large Hadron Collider (LHC), protons are
accelerated to nearly the speed of light and collided. The enormous energy
(~13 TeV at LHC) creates new particles via E=mc². These particles fly outward
and are detected by surrounding instruments.

MAGNETIC FIELD & TRACK CURVATURE
--------------------------------
A strong magnetic field (typically 2-4 Tesla) permeates the detector.
The Lorentz force causes charged particles to curve:

    F = q(v × B)

This results in helical trajectories with radius:

    r = p_T / (|q| × B)

where:
    - r = radius of curvature
    - p_T = transverse momentum (perpendicular to beam)
    - q = electric charge
    - B = magnetic field strength

Key insight: HIGHER MOMENTUM = LARGER RADIUS = STRAIGHTER TRACK
This is how physicists measure particle momentum!

The SIGN of the charge determines the DIRECTION of curvature:
- Positive particles (positrons, protons) curve one way
- Negative particles (electrons, muons⁻) curve the opposite way

PARTICLE IDENTIFICATION BY STOPPING LOCATION
--------------------------------------------
Different particles interact differently with matter:

1. ELECTRONS/POSITRONS (e⁻/e⁺):
   - Light (0.511 MeV/c²), easily deflected
   - Create electromagnetic showers via bremsstrahlung
   - Completely absorbed in EM calorimeter (~20 radiation lengths)
   - Tight helical tracks due to low mass

2. PHOTONS (γ):
   - No charge → NO TRACK (invisible in tracker)
   - Convert to e⁺e⁻ pairs in EM calorimeter
   - Deposit all energy in EM calorimeter

3. HADRONS (π±, K±, p):
   - Made of quarks, interact via strong force
   - Penetrate EM calorimeter, stop in hadronic calorimeter
   - Create hadronic showers (nuclear interactions)

4. MUONS (μ±):
   - Heavy leptons (105.7 MeV/c²), minimal ionizing
   - Penetrate ENTIRE detector (very weakly interacting)
   - Only particles reaching outermost muon chambers
   - Gentle curves due to high mass

5. NEUTRINOS (ν):
   - No charge, no strong interaction
   - Pass through detector completely undetected
   - Inferred from "missing energy" in event

JETS: QUARK/GLUON FRAGMENTATION
-------------------------------
Quarks and gluons cannot exist freely due to "color confinement."
When produced, they immediately "hadronize" into sprays of hadrons:

    q/g → π⁺ + π⁻ + π⁰ + K + p + ...

These collimated sprays are called "jets." Properties:
- Higher energy jets are more collimated (smaller opening angle)
- Jet composition: ~60% pions, ~25% kaons, ~15% protons/neutrons
- Energy distributed exponentially among jet constituents

DETECTOR STRUCTURE (barrel region)
----------------------------------
Modern detectors are layered cylinders:

    [Beam Pipe] → [Tracker] → [EM Cal] → [Had Cal] → [Muon Chambers]
       0.5m         1-4m       4.5-6m     6.5-10m       11-15m

Each layer serves a purpose:
- Tracker: Precise position measurements for momentum reconstruction
- EM Calorimeter: Dense material (lead/scintillator) stops e±/γ
- Hadronic Calorimeter: Iron/scintillator stops hadrons
- Muon Chambers: Gas detectors for muon tracking

================================================================================
ANIMATION APPROACH (Memory Efficient)
================================================================================

Instead of duplicating geometry for each frame, we use color animation:
- Positions: Same XYZ geometry replicated for each time frame
- Colors: Different per frame - real color for "past/present", black for "future"

Each vertex has a "birth time" based on its arc length from the collision point.
At frame t, vertices with birth_time <= t show their real color, others are black.

This gives the effect of particles flying outward from the collision vertex!

================================================================================

Usage:
    python demo_particle_collision_animated.py [--events=N] [--frames=N] [--jets=N]

Controls:
    - Use TIME slider to scrub through the collision event
    - Ctrl+C to stop and cleanup
    - Browser opens automatically
"""

from __future__ import annotations

DEMO_META = {
    "key": "particle_collision_animated",
    "title": "Animated Particle Collision Detector Visualization",
    "description": "Time-animated particle-collision tracks curving in a detector's magnetic field.",
    "category": "synthetic",
    "geometry": "points+lines",
    "requirements": {
        "download_mb": 0,
        "compute": "light",
        "gpu": "none",
        "local_data": None,
    },
    "caches": [],
    "outputs": ["collision_animated"],
}

import sys
import tempfile
from pathlib import Path

import numpy as np
from arbol import aprint, asection

from luxar import Dimension, Dimensions, LuxarZarrCompiler
from luxar.demos import launch_viewer

# Reuse the shared physics constants + event generator from the static particle
# collision demo (now a normal sibling import; see the retired sys.modules alias).
from luxar.demos.demo_particle_collision import (
    B_FIELD,
    BEAM_PIPE_RADIUS,
    DETECTOR_LENGTH,
    ECAL_INNER,
    ECAL_OUTER,
    HCAL_INNER,
    HCAL_OUTER,
    MUON_INNER,
    MUON_OUTER,
    TRACKER_INNER,
    TRACKER_OUTER,
    Particle,
    generate_collision_event,
)
from luxar.utils.paths import get_demos_output_dir


def generate_helix_track_with_times(
    particle: Particle,
    rng: np.random.Generator,
    n_points: int = 1000,  # High resolution for animation resampling
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Generate a helical track for a charged particle with birth times.

    PHYSICS:
    --------
    The Lorentz force F = q(v × B) causes charged particles to follow
    helical paths in a uniform magnetic field:

    - Radius of curvature: r = p_T / (|q| × B)
      → Higher momentum = larger radius = STRAIGHTER track
      → This is how we MEASURE momentum in real detectors!

    - Sign of charge determines direction of curvature
      → Positive particles curve clockwise (from above)
      → Negative particles curve counter-clockwise

    - Helix pitch determined by p_z/p_T ratio
      → Forward-going particles have elongated helices

    The track width in visualization represents the particle's energy -
    higher energy particles leave "brighter" tracks (more ionization).

    Args:
        particle: Particle object with kinematics
        rng: Random number generator for small variations
        n_points: Maximum number of points along track

    Returns:
        vertices: (N*2, 3) array of line segment vertices (start, end pairs)
        widths: (N*2,) array of line widths (energy visualization)
        colors: (N*2, 3) array of RGB colors (particle identification)
        birth_times: (N*2,) normalized birth time [0, 1] for each vertex
    """
    if particle.charge == 0:
        # Neutral particles (photons, neutrons) have no charge
        # → No Lorentz force → straight line trajectory
        # They are only "seen" when they deposit energy in calorimeters
        return generate_straight_track_with_times(particle, rng, n_points)

    # =========================================================================
    # HELIX PARAMETER CALCULATION
    # =========================================================================

    pt = particle.pt  # Transverse momentum (perpendicular to B-field)
    pz = particle.pz  # Longitudinal momentum (along beam/B-field direction)

    # RADIUS OF CURVATURE: r = p_T / (|q| × B)
    # -----------------------------------------
    # This is THE fundamental equation of charged particle tracking!
    # - Real detectors use r to MEASURE p_T (momentum spectroscopy)
    # - Factor of 2.0 is a visualization scaling factor
    # - In real units: r[m] = p_T[GeV/c] / (0.3 × |q| × B[T])
    #   where 0.3 comes from unit conversion (c in appropriate units)
    radius = pt / (abs(particle.charge) * B_FIELD) * 2.0

    # Minimum radius to prevent visual artifacts for very soft particles
    radius = max(radius, 0.3)

    # ANGULAR VELOCITY: ω = q × B / p_T
    # ---------------------------------
    # This determines how quickly the particle spirals.
    # The sign of charge determines the direction of rotation:
    # - Positive charge → clockwise rotation (from +z looking down)
    # - Negative charge → counter-clockwise rotation
    omega = particle.charge * B_FIELD / pt if pt > 0.1 else 0.01

    # Initial azimuthal angle (direction particle is heading in x-y plane)
    phi0 = np.arctan2(particle.py, particle.px)

    # CENTER OF HELIX CIRCLE
    # ----------------------
    # The particle doesn't spiral around the collision point!
    # It spirals around a point offset perpendicular to its initial direction.
    # The offset direction depends on charge sign.
    cx = particle.origin[0] - radius * np.sin(phi0) * np.sign(particle.charge)
    cy = particle.origin[1] + radius * np.cos(phi0) * np.sign(particle.charge)

    # DETECTOR BOUNDARIES
    # -------------------
    # Track terminates when particle:
    # 1. Exits radially (absorbed in calorimeter or escapes)
    # 2. Exits longitudinally (beyond detector endcap)
    max_radius = particle.stops_at if particle.stops_at else MUON_OUTER
    max_z = DETECTOR_LENGTH

    # =========================================================================
    # HELIX POINT GENERATION WITH ARC LENGTH TRACKING
    # =========================================================================

    points = []
    arc_lengths = []  # Track cumulative arc length for birth time calculation
    t = 0  # Parametric time along helix
    dt = 0.005  # Step size (smaller = smoother curves, needed for animation)
    total_arc = 0.0
    prev_point = None

    while len(points) < n_points:
        # HELIX PARAMETRIC EQUATIONS
        # --------------------------
        # x(t) = cx + r × sin(φ₀ + ω×t)
        # y(t) = cy - r × cos(φ₀ + ω×t)
        # z(t) = z₀ + v_z × t
        #
        # The x-y motion is circular, z advances linearly → HELIX
        phi = phi0 + omega * t * np.sign(particle.charge)
        x = cx + radius * np.sin(phi) * np.sign(particle.charge)
        y = cy - radius * np.cos(phi) * np.sign(particle.charge)
        z = particle.origin[2] + pz * t * 0.3  # Scale z velocity for visualization

        # Check if particle has exited detector volume
        r = np.sqrt(x**2 + y**2)
        if r > max_radius or abs(z) > max_z:
            break

        point = np.array([x, y, z])

        # Track arc length for birth time calculation
        if prev_point is not None:
            total_arc += np.linalg.norm(point - prev_point)

        points.append(point)
        arc_lengths.append(total_arc)
        prev_point = point
        t += dt

        # ELECTROMAGNETIC SHOWER SIMULATION
        # ----------------------------------
        # Electrons/positrons undergo bremsstrahlung (emit photons when
        # deflected by nuclei). This triggers an electromagnetic cascade:
        # e → γ + e → e⁺e⁻ + e → many particles
        # The shower develops rapidly once inside the EM calorimeter.
        if r > ECAL_INNER and particle.stops_at == ECAL_OUTER:
            # Track ends as particle showers
            break

    if len(points) < 2:
        points = [
            particle.origin.tolist(),
            (particle.origin + [0.1, 0.1, 0.1]).tolist(),
        ]
        arc_lengths = [0.0, 0.1]

    points = np.array(points, dtype=np.float32)
    arc_lengths = np.array(arc_lengths, dtype=np.float32)

    # Normalize arc lengths to [0, 1] for birth times
    # birth_time = 0 at collision vertex, = 1 at track end
    max_arc = arc_lengths[-1] if arc_lengths[-1] > 0 else 1.0
    birth_times_per_point = arc_lengths / max_arc

    # Create line segments from points
    n_segments = len(points) - 1
    vertices = np.zeros((n_segments * 2, 3), dtype=np.float32)
    vertices[0::2] = points[:-1]  # Segment start points
    vertices[1::2] = points[1:]  # Segment end points

    # Birth times for segment vertices (continuous at joints)
    birth_times = np.zeros(n_segments * 2, dtype=np.float32)
    birth_times[0::2] = birth_times_per_point[:-1]
    birth_times[1::2] = birth_times_per_point[1:]

    # Width tapers along track (energy loss visualization)
    # Use per-vertex widths for continuity at joints
    base_width = 0.015 + 0.01 * (particle.pt / 50.0)
    n_pts = len(points)
    vertex_widths = base_width * (1.0 - 0.5 * np.linspace(0, 1, n_pts))

    # Expand to segment format: widths[2i] = vertex i, widths[2i+1] = vertex i+1
    widths = np.zeros(n_segments * 2, dtype=np.float32)
    widths[0::2] = vertex_widths[:-1]
    widths[1::2] = vertex_widths[1:]

    # Colors with smooth fade along track
    base_color = np.array(particle.color, dtype=np.float32)
    vertex_fades = 1.0 - 0.3 * np.linspace(0, 1, n_pts)
    vertex_colors = base_color * vertex_fades[:, np.newaxis]

    # Add slight random variation per-vertex
    vertex_colors += rng.uniform(-0.05, 0.05, vertex_colors.shape).astype(np.float32)
    vertex_colors = np.clip(vertex_colors, 0, 1)

    # Expand to segment format
    colors = np.zeros((n_segments * 2, 3), dtype=np.float32)
    colors[0::2] = vertex_colors[:-1]
    colors[1::2] = vertex_colors[1:]

    return vertices, widths, colors, birth_times


def generate_straight_track_with_times(
    particle: Particle,
    rng: np.random.Generator,
    n_points: int = 500,  # High resolution for animation resampling
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Generate a straight track for neutral particles with birth times.

    Neutral particles (photons, neutrons) have no charge and thus no
    Lorentz force. They travel in straight lines until they interact
    in the calorimeters.
    """
    direction = np.array([particle.px, particle.py, particle.pz])
    direction = direction / (np.linalg.norm(direction) + 1e-10)

    max_radius = particle.stops_at if particle.stops_at else MUON_OUTER

    points = []
    distances = []
    for i in range(n_points):
        t = i * 0.5
        pos = particle.origin + direction * t
        r = np.sqrt(pos[0] ** 2 + pos[1] ** 2)
        if r > max_radius or abs(pos[2]) > DETECTOR_LENGTH:
            break
        points.append(pos)
        distances.append(np.linalg.norm(pos - particle.origin))

    if len(points) < 2:
        return (
            np.array([]).reshape(0, 3),
            np.array([]),
            np.array([]).reshape(0, 3),
            np.array([]),
        )

    points = np.array(points, dtype=np.float32)
    distances = np.array(distances, dtype=np.float32)

    # Normalize distances to [0, 1] for birth times
    max_dist = distances[-1] if distances[-1] > 0 else 1.0
    birth_times_per_point = distances / max_dist

    n_segments = len(points) - 1
    vertices = np.zeros((n_segments * 2, 3), dtype=np.float32)
    vertices[0::2] = points[:-1]
    vertices[1::2] = points[1:]

    birth_times = np.zeros(n_segments * 2, dtype=np.float32)
    birth_times[0::2] = birth_times_per_point[:-1]
    birth_times[1::2] = birth_times_per_point[1:]

    # Dashed appearance for neutral particles
    widths = np.full(n_segments * 2, 0.008, dtype=np.float32)

    base_color = np.array(particle.color, dtype=np.float32)
    colors = np.tile(base_color, (n_segments * 2, 1))

    return vertices, widths, colors, birth_times


# =============================================================================
# Detector Geometry
# =============================================================================


def generate_detector_geometry(
    rng: np.random.Generator,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Generate enhanced detector outline with full detail.

    Since we use extend_to_all=["time"], this geometry is only stored once
    but visible at all time frames. This allows rich detail without memory cost.

    Rings are authored as unique vertices + CLOSED edge loops (returned in
    ``edges``, consumed via ``line_type="indexed"``) so their joints share
    vertex indices and render seamlessly; ribs/spokes/beam lines are
    genuinely independent single edges.

    The detector is inspired by ATLAS/CMS with:
    - Beam pipe (central tube where particles collide)
    - Silicon tracker layers (precise vertex tracking)
    - Electromagnetic calorimeter (stops electrons/photons)
    - Hadronic calorimeter (stops hadrons)
    - Muon spectrometer (only muons reach here)
    """
    vertices = []
    widths = []
    colors = []
    sharpness_vals = []
    edges: list[tuple[int, int]] = []

    def add_ring(radius, z, n_seg, color, width, sharp):
        """One closed ring: n_seg unique vertices + n_seg loop edges."""
        base = len(vertices)
        for a in np.linspace(0, 2 * np.pi, n_seg, endpoint=False):
            vertices.append([radius * np.cos(a), radius * np.sin(a), z])
            widths.append(width)
            colors.append(color)
            sharpness_vals.append(sharp)
        edges.extend((base + i, base + (i + 1) % n_seg) for i in range(n_seg))

    def add_segment(p0, p1, color, width, sharp):
        """One independent straight segment (two vertices, one edge)."""
        base = len(vertices)
        vertices.extend([p0, p1])
        widths.extend([width, width])
        colors.extend([color, color])
        sharpness_vals.extend([sharp, sharp])
        edges.append((base, base + 1))

    # Full resolution detector layers with realistic color coding
    # Colors inspired by actual detector component conventions
    layers = [
        # Inner tracking: gold/bronze tones (silicon sensors)
        (BEAM_PIPE_RADIUS, [0.3, 0.25, 0.15], 0.012, 48),  # Beam pipe
        (TRACKER_INNER, [0.35, 0.30, 0.15], 0.010, 64),  # Pixel detector
        (
            (TRACKER_INNER + TRACKER_OUTER) / 2,
            [0.30, 0.28, 0.12],
            0.008,
            64,
        ),  # Strip tracker
        (TRACKER_OUTER, [0.25, 0.25, 0.10], 0.008, 64),  # TRT outer
        # EM Calorimeter: green/teal (lead-liquid argon)
        (ECAL_INNER, [0.10, 0.30, 0.25], 0.010, 72),
        (ECAL_OUTER, [0.08, 0.28, 0.22], 0.010, 72),
        # Hadronic Calorimeter: orange/copper (iron-scintillator)
        (HCAL_INNER, [0.35, 0.20, 0.10], 0.012, 72),
        ((HCAL_INNER + HCAL_OUTER) / 2, [0.30, 0.18, 0.08], 0.010, 72),
        (HCAL_OUTER, [0.25, 0.15, 0.06], 0.012, 72),
        # Muon chambers: purple/blue (drift tubes)
        (MUON_INNER, [0.20, 0.10, 0.30], 0.010, 80),
        ((MUON_INNER + MUON_OUTER) / 2, [0.18, 0.08, 0.28], 0.008, 80),
        (MUON_OUTER, [0.15, 0.06, 0.25], 0.010, 80),
    ]

    # Multiple z-planes for depth perception
    z_positions = [-DETECTOR_LENGTH * 0.5, 0, DETECTOR_LENGTH * 0.5]

    # Draw circular cross-sections at each z-plane (closed indexed rings)
    for radius, color, width, n_seg in layers:
        for z in z_positions:
            add_ring(radius, z, n_seg, color, width, 0.5)

    # Longitudinal ribs connecting the z-planes (structural support visualization)
    n_long = 24  # More longitudinal lines for better structure
    for i in range(n_long):
        angle = 2 * np.pi * i / n_long
        for radius, color, width, _ in layers:
            x = radius * np.cos(angle)
            y = radius * np.sin(angle)

            # Full length longitudinal lines (independent segments)
            add_segment(
                [x, y, -DETECTOR_LENGTH * 0.5],
                [x, y, DETECTOR_LENGTH * 0.5],
                color,
                width * 0.5,
                0.45,
            )

    # Endcap disks (circular rings at z-ends showing layer boundaries)
    endcap_z = [DETECTOR_LENGTH * 0.5, -DETECTOR_LENGTH * 0.5]
    endcap_layers = [
        (TRACKER_OUTER, ECAL_INNER, [0.20, 0.25, 0.15], 0.006),  # Tracker endcap
        (ECAL_OUTER, HCAL_INNER, [0.10, 0.25, 0.20], 0.006),  # ECAL endcap
        (HCAL_OUTER, MUON_INNER, [0.25, 0.15, 0.08], 0.006),  # HCAL endcap
    ]

    n_radial = 16  # Radial spokes in endcaps
    for z in endcap_z:
        for r_inner, r_outer, color, width in endcap_layers:
            # Radial lines connecting inner to outer radius
            for i in range(n_radial):
                angle = 2 * np.pi * i / n_radial
                x_inner = r_inner * np.cos(angle)
                y_inner = r_inner * np.sin(angle)
                x_outer = r_outer * np.cos(angle)
                y_outer = r_outer * np.sin(angle)

                add_segment(
                    [x_inner, y_inner, z], [x_outer, y_outer, z], color, width, 0.45
                )

    # Beam pipe extension (thin central tube)
    beam_color = [0.4, 0.35, 0.2]  # Golden beam pipe
    n_beam_seg = 8
    for i in range(n_beam_seg):
        angle = 2 * np.pi * i / n_beam_seg
        x = BEAM_PIPE_RADIUS * np.cos(angle)
        y = BEAM_PIPE_RADIUS * np.sin(angle)

        add_segment(
            [x, y, -DETECTOR_LENGTH * 0.8],
            [x, y, DETECTOR_LENGTH * 0.8],
            beam_color,
            0.015,
            0.6,
        )

    # Interaction point marker rings (where collisions happen)
    ip_radii = [0.3, 0.6, 1.0]
    ip_color = [0.5, 0.45, 0.2]  # Golden IP markers
    for r in ip_radii:
        add_ring(r, 0, 32, ip_color, 0.008, 0.65)

    return (
        np.array(vertices, dtype=np.float32),
        np.array(widths, dtype=np.float32),
        np.array(colors, dtype=np.float32),
        np.array(sharpness_vals, dtype=np.float32),
        np.array(edges, dtype=np.uint32),
    )


# =============================================================================
# Calorimeter Energy Deposits
# =============================================================================
#
# CALORIMETRY: MEASURING PARTICLE ENERGY
# --------------------------------------
# Calorimeters are dense absorbers that stop particles and measure their
# total energy. The particle creates a "shower" of secondary particles.
#
# ELECTROMAGNETIC CALORIMETER (ECAL)
# ----------------------------------
# Stops electrons, positrons, and photons via:
# - Bremsstrahlung: e → e + γ
# - Pair production: γ → e⁺ + e⁻
#
# HADRONIC CALORIMETER (HCAL)
# ---------------------------
# Stops hadrons via strong interactions:
# - Nuclear spallation: hadron + nucleus → many hadrons


def generate_calorimeter_deposits_with_times(
    particles: list[Particle],
    rng: np.random.Generator,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Generate calorimeter energy deposits with birth times."""
    positions = []
    colors = []
    radii = []
    sharpness = []
    birth_times = []

    for particle in particles:
        if particle.charge == 0 and particle.particle_type == "photon":
            deposit_radius = ECAL_INNER + rng.uniform(0.2, 1.0)
        elif particle.stops_at == ECAL_OUTER:
            deposit_radius = ECAL_INNER + rng.uniform(0.3, 1.2)
        elif particle.stops_at == HCAL_OUTER:
            deposit_radius = HCAL_INNER + rng.uniform(0.5, 2.5)
        else:
            continue  # Muons don't deposit much

        direction = np.array([particle.px, particle.py, particle.pz])
        direction = direction / (np.linalg.norm(direction) + 1e-10)

        phi = np.arctan2(particle.py, particle.px)
        deposit_x = deposit_radius * np.cos(phi)
        deposit_y = deposit_radius * np.sin(phi)
        deposit_z = particle.origin[2] + direction[2] * deposit_radius * 0.8

        # Birth time: deposits appear towards end of animation (0.6-1.0)
        deposit_distance = np.sqrt(deposit_x**2 + deposit_y**2 + deposit_z**2)
        base_birth_time = 0.6 + 0.4 * (deposit_distance / MUON_OUTER)

        n_shower = int(3 + particle.energy / 10)
        shower_spread = 0.3 + particle.energy / 200

        for _ in range(n_shower):
            offset = rng.normal(0, shower_spread, 3)
            pos = np.array([deposit_x, deposit_y, deposit_z]) + offset

            positions.append(pos)
            colors.append(particle.color)

            size = 0.1 + 0.05 * np.log1p(particle.energy)
            radii.append(size * rng.uniform(0.7, 1.3))
            # Soft, glowing appearance (normalized [0, 1] knob; low = peakier/softer)
            sharpness.append(rng.uniform(0.25, 0.4))
            birth_times.append(base_birth_time + rng.uniform(-0.05, 0.1))

    if not positions:
        return (
            np.array([]).reshape(0, 3),
            np.array([]).reshape(0, 3),
            np.array([]),
            np.array([]),
            np.array([]),
        )

    return (
        np.array(positions, dtype=np.float32),
        np.array(colors, dtype=np.float32),
        np.array(radii, dtype=np.float32),
        np.array(sharpness, dtype=np.float32),
        np.clip(np.array(birth_times, dtype=np.float32), 0, 1),
    )


# =============================================================================
# Main Animation Generation
# =============================================================================


def generate_animated_detector_scene(
    output_path: Path,
    n_events: int = 5,
    n_jets_per_event: int = 4,
    n_frames: int = 250,
) -> tuple[int, int]:
    """Generate animated particle detector visualization.

    Animation approach:
    - Each vertex/point has a "birth time" based on arc length from collision
    - For frame F, vertices with birth_time <= F/N_frames show real color
    - Vertices with birth_time > F/N_frames show black (future = invisible)
    - This creates the effect of tracks growing outward from collision
    """
    total_segments = 0
    total_points = 0

    with LuxarZarrCompiler(output_path) as compiler:
        # 4D dimensions: x, y, z, time (discrete)
        # Physics: particles travel at ~c (3×10^8 m/s)
        # To traverse 15m detector: ~50 nanoseconds
        # Each frame represents ~0.2 ns at 250 frames
        total_time_ns = 50.0  # nanoseconds for full event evolution
        time_per_frame = total_time_ns / (n_frames - 1)

        dims = Dimensions(
            [
                Dimension("x", unit="m", display=True),
                Dimension("y", unit="m", display=True),
                Dimension("z", unit="m", display=True),
                Dimension(
                    "time",
                    unit="ns",
                    display=False,
                    # Frame 0 is the collision instant: the detector is present
                    # (extend_to_all) but no tracks have grown yet, so the range
                    # starts at 0 to cover it.
                    range=(0.0, total_time_ns),
                    step=time_per_frame,
                    discrete=True,
                ),
            ]
        )
        scene = compiler.create_scene(dimensions=dims)

        rng = np.random.default_rng(42)

        # =====================================================================
        # DETECTOR GEOMETRY (static - extends to all time values)
        # Using extend_to_all to show detector at every time without duplication
        # =====================================================================
        with asection("Creating detector geometry"):
            det_verts, det_widths, det_colors, det_sharp, det_edges = (
                generate_detector_geometry(rng)
            )

            n_det_verts = len(det_verts)

            # Place detector at time=0 with extend_to_all=["time"]
            # This makes it visible at ALL time values without replication
            det_positions_4d = np.zeros((n_det_verts, 4), dtype=np.float32)
            det_positions_4d[:, :3] = det_verts
            det_positions_4d[:, 3] = 0  # Time = 0

            # Indexed authoring: ring joints share vertex indices, so the
            # viewer renders each ring as a continuous loop instead of a
            # bead chain of independent chords.
            scene.add_lines(
                "detector_geometry",
                vertices=det_positions_4d,
                widths=det_widths,
                colors=det_colors,
                sharpness=det_sharp,
                indices=det_edges,
                line_type="indexed",
                extend_to_all=["time"],  # Extend to all time values!
                layer=True,
            )
            n_det = len(det_edges)
            aprint(
                f"Detector geometry: {n_det:,} segments (extends to all time values)"
            )
            total_segments += n_det

        # =====================================================================
        # GENERATE ALL TRACKS AND DEPOSITS
        # =====================================================================
        all_track_verts = []
        all_track_widths = []
        all_track_colors = []
        all_track_birth_times = []

        all_deposit_pos = []
        all_deposit_colors = []
        all_deposit_radii = []
        all_deposit_sharp = []
        all_deposit_birth_times = []

        with asection(f"Generating {n_events} collision events"):
            for event_idx in range(n_events):
                event_offset = np.array([0, 0, event_idx * 0.1])

                particles = generate_collision_event(
                    rng, n_jets=n_jets_per_event, n_leptons=rng.integers(1, 4)
                )

                aprint(f"  Event {event_idx + 1}: {len(particles)} particles")

                for particle in particles:
                    particle.origin = particle.origin + event_offset
                    vertices, widths, colors, birth_times = (
                        generate_helix_track_with_times(particle, rng)
                    )

                    if len(vertices) > 0:
                        all_track_verts.append(vertices)
                        all_track_widths.append(widths)
                        all_track_colors.append(colors)
                        all_track_birth_times.append(birth_times)

                pos, colors, radii, sharp, birth_times = (
                    generate_calorimeter_deposits_with_times(particles, rng)
                )
                if len(pos) > 0:
                    all_deposit_pos.append(pos)
                    all_deposit_colors.append(colors)
                    all_deposit_radii.append(radii)
                    all_deposit_sharp.append(sharp)
                    all_deposit_birth_times.append(birth_times)

        # =====================================================================
        # WRITE ANIMATED TRACKS
        # Each frame shows tracks up to that time, resampled to 64 segments
        # This gives smooth 250-frame animation with constant 64 segments/track
        # =====================================================================
        SEGMENTS_PER_TRACK = 64

        with asection("Writing animated particle tracks"):
            if all_track_verts:
                n_tracks = len(all_track_verts)
                aprint(
                    f"  Tracks: {n_tracks}, resampling to {SEGMENTS_PER_TRACK} segments each"
                )

                frame_positions = []
                frame_colors = []
                frame_widths = []

                for f in range(n_frames):
                    if f == 0:
                        continue  # No tracks visible at t=0 (particles just starting)

                    time_ns = f * time_per_frame
                    frame_progress = f / (n_frames - 1)

                    for track_idx in range(n_tracks):
                        track_verts = all_track_verts[
                            track_idx
                        ]  # (N*2, 3) segment vertices
                        track_colors = all_track_colors[track_idx]
                        track_widths = all_track_widths[track_idx]
                        track_birth = all_track_birth_times[track_idx]

                        if len(track_verts) < 4:
                            continue

                        # Find how much of track is visible (birth_time <= frame_progress)
                        # track_birth is per-vertex, use segment start vertices
                        segment_births = track_birth[0::2]
                        visible_mask = segment_births <= frame_progress
                        n_visible = np.sum(visible_mask)

                        if n_visible < 1:
                            continue

                        # Get the visible portion of the track
                        last_visible_idx = np.where(visible_mask)[0][-1]
                        visible_end = (
                            last_visible_idx + 1
                        ) * 2  # Include both vertices of last segment

                        # Extract visible track points (convert segments to polyline)
                        seg_starts = track_verts[0:visible_end:2]
                        seg_ends = track_verts[1:visible_end:2]
                        # Build continuous polyline: start0, end0=start1, end1=start2, ...
                        polyline = np.vstack([seg_starts, seg_ends[-1:]])

                        color_starts = track_colors[0:visible_end:2]
                        color_ends = track_colors[1:visible_end:2]
                        color_polyline = np.vstack([color_starts, color_ends[-1:]])

                        width_starts = track_widths[0:visible_end:2]
                        width_ends = track_widths[1:visible_end:2]
                        width_polyline = np.concatenate([width_starts, width_ends[-1:]])

                        # ALWAYS resample to SEGMENTS_PER_TRACK segments (SEGMENTS_PER_TRACK+1 points)
                        # This gives fine detail at early times (short track = many segments)
                        # and coarser detail at late times (full track = same segment count)
                        n_pts = len(polyline)
                        if n_pts < 2:
                            continue

                        # Compute cumulative arc length for interpolation
                        diffs = np.diff(polyline, axis=0)
                        seg_lengths = np.linalg.norm(diffs, axis=1)
                        arc_length = np.concatenate([[0], np.cumsum(seg_lengths)])
                        total_length = arc_length[-1]

                        if total_length < 1e-6:
                            continue

                        # ALWAYS use SEGMENTS_PER_TRACK segments - this is the key!
                        # Early frames: short track, 64 segments = very fine detail
                        # Late frames: full track, 64 segments = coarser detail
                        n_resample = SEGMENTS_PER_TRACK + 1
                        target_arcs = np.linspace(0, total_length, n_resample)

                        resampled_pts = np.zeros((n_resample, 3), dtype=np.float32)
                        resampled_colors = np.zeros((n_resample, 3), dtype=np.float32)
                        resampled_widths = np.zeros(n_resample, dtype=np.float32)

                        for i, target in enumerate(target_arcs):
                            # Find segment containing this arc length
                            idx = np.searchsorted(arc_length, target, side="right") - 1
                            idx = np.clip(idx, 0, n_pts - 2)

                            # Interpolate within segment
                            seg_start_arc = arc_length[idx]
                            seg_end_arc = arc_length[idx + 1]
                            seg_len = seg_end_arc - seg_start_arc

                            if seg_len > 1e-9:
                                t = (target - seg_start_arc) / seg_len
                            else:
                                t = 0.0

                            resampled_pts[i] = (
                                polyline[idx] * (1 - t) + polyline[idx + 1] * t
                            )
                            resampled_colors[i] = (
                                color_polyline[idx] * (1 - t)
                                + color_polyline[idx + 1] * t
                            )
                            resampled_widths[i] = (
                                width_polyline[idx] * (1 - t)
                                + width_polyline[idx + 1] * t
                            )

                        # Per-VERTEX arrays with the frame time in the 4th
                        # column; the edge list below connects consecutive
                        # resampled points so interior joints share their
                        # vertex index (continuous track, no bead chain).
                        track_verts_4d = np.zeros((n_resample, 4), dtype=np.float32)
                        track_verts_4d[:, :3] = resampled_pts
                        track_verts_4d[:, 3] = time_ns

                        frame_positions.append(track_verts_4d)
                        frame_colors.append(resampled_colors)
                        frame_widths.append(resampled_widths)

                if frame_positions:
                    all_positions = np.concatenate(frame_positions, axis=0)
                    all_colors = np.concatenate(frame_colors, axis=0)
                    all_widths = np.concatenate(frame_widths, axis=0)

                    # One indexed node: per-(frame, track) edge lists over the
                    # concatenated unique vertices. Interior joints share
                    # indices (continuous tracks); separate tracks/frames
                    # stay disconnected.
                    track_edges = []
                    offset = 0
                    for verts in frame_positions:
                        n_pts = len(verts)
                        idx = np.arange(offset, offset + n_pts - 1, dtype=np.uint32)
                        track_edges.append(np.column_stack([idx, idx + 1]))
                        offset += n_pts
                    track_edges_arr = np.concatenate(track_edges, axis=0)

                    scene.add_lines(
                        "particle_tracks",
                        vertices=all_positions,
                        widths=all_widths,
                        colors=all_colors,
                        sharpness=0.5,
                        indices=track_edges_arr,
                        line_type="indexed",
                        # NOTE: NOT using extend_to_all - tracks should only be visible at their birth time
                        layer=True,
                    )

                    total_verts = len(all_positions)
                    aprint(f"  Total track vertices: {total_verts:,}")
                    aprint(
                        f"  ~{SEGMENTS_PER_TRACK} segments/track x {n_tracks} tracks x {n_frames} frames"
                    )
                    total_segments += len(track_edges_arr)

        # =====================================================================
        # WRITE ANIMATED CALORIMETER DEPOSITS (efficient: per-frame geometry)
        # =====================================================================
        with asection("Writing animated calorimeter deposits"):
            if all_deposit_pos:
                deposit_positions = np.concatenate(all_deposit_pos, axis=0)
                deposit_colors_real = np.concatenate(all_deposit_colors, axis=0)
                deposit_radii = np.concatenate(all_deposit_radii, axis=0)
                deposit_sharpness = np.concatenate(all_deposit_sharp, axis=0)
                deposit_birth_times = np.concatenate(all_deposit_birth_times, axis=0)

                n_deposits = len(deposit_positions)
                aprint(f"  Deposit points: {n_deposits:,}")

                # Convert birth_times to birth_frames
                deposit_birth_frames = (deposit_birth_times * (n_frames - 1)).astype(
                    np.int32
                )

                # Build per-frame geometry
                frame_positions = []
                frame_colors = []
                frame_radii = []
                frame_sharp = []

                for f in range(n_frames):
                    time_ns = f * time_per_frame

                    # Find deposits born by this frame
                    visible_mask = deposit_birth_frames <= f
                    visible_indices = np.where(visible_mask)[0]

                    if len(visible_indices) > 0:
                        visible_pos = deposit_positions[visible_indices]
                        visible_colors = deposit_colors_real[visible_indices]
                        visible_radii = deposit_radii[visible_indices]
                        visible_sharp = deposit_sharpness[visible_indices]

                        # Add 4D positions
                        positions_4d = np.zeros((len(visible_pos), 4), dtype=np.float32)
                        positions_4d[:, :3] = visible_pos
                        positions_4d[:, 3] = time_ns

                        frame_positions.append(positions_4d)
                        frame_colors.append(visible_colors)
                        frame_radii.append(visible_radii)
                        frame_sharp.append(visible_sharp)

                if frame_positions:
                    all_positions = np.concatenate(frame_positions, axis=0)
                    all_colors = np.concatenate(frame_colors, axis=0)
                    all_radii = np.concatenate(frame_radii, axis=0)
                    all_sharp = np.concatenate(frame_sharp, axis=0)

                    scene.add_points(
                        "calorimeter_deposits",
                        positions=all_positions,
                        colors=all_colors,
                        radii=all_radii,
                        sharpness=all_sharp,
                        layer=True,
                        intensity=0.125,
                    )

                    total_pts = len(all_positions)
                    naive_pts = n_deposits * n_frames
                    efficiency = 100 * (1 - total_pts / naive_pts)
                    aprint(
                        f"  Total deposit points: {total_pts:,} ({efficiency:.0f}% smaller than naive)"
                    )
                    total_points += total_pts

        # =====================================================================
        # COLLISION VERTEX (pulsing flash at t=0)
        # =====================================================================
        with asection("Adding animated vertex markers"):
            n_vertex_per_event = 20
            vertex_pos_base = []
            vertex_colors_base = []
            vertex_radii_base = []

            for event_idx in range(n_events):
                center = np.array([0, 0, event_idx * 0.1])
                for _ in range(n_vertex_per_event):
                    offset = rng.normal(0, 0.03, 3)
                    vertex_pos_base.append(center + offset)
                    vertex_colors_base.append([1.0, 0.9, 0.5])  # Golden flash
                    vertex_radii_base.append(rng.uniform(0.02, 0.05))

            vertex_pos_base = np.array(vertex_pos_base, dtype=np.float32)
            vertex_colors_base = np.array(vertex_colors_base, dtype=np.float32)
            vertex_radii_base = np.array(vertex_radii_base, dtype=np.float32)

            n_vertex_points = len(vertex_pos_base)

            vertex_positions_4d = np.zeros(
                (n_frames * n_vertex_points, 4), dtype=np.float32
            )
            vertex_colors_4d = np.zeros(
                (n_frames * n_vertex_points, 3), dtype=np.float32
            )
            vertex_radii_4d = np.zeros(n_frames * n_vertex_points, dtype=np.float32)

            for f in range(n_frames):
                frame_progress = f / (n_frames - 1)
                time_ns = f * time_per_frame  # Physical time in nanoseconds
                start = f * n_vertex_points
                end = start + n_vertex_points

                vertex_positions_4d[start:end, :3] = vertex_pos_base
                vertex_positions_4d[start:end, 3] = time_ns

                # Brightness fades over time (bright flash at collision)
                brightness = max(0.2, 1.0 - 0.8 * frame_progress)
                vertex_colors_4d[start:end] = vertex_colors_base * brightness

                # Size also decreases
                vertex_radii_4d[start:end] = vertex_radii_base * max(
                    0.3, 1.0 - 0.5 * frame_progress
                )

            scene.add_points(
                "collision_vertices",
                positions=vertex_positions_4d,
                colors=vertex_colors_4d,
                radii=vertex_radii_4d,
                sharpness=0.8,
                layer=True,
                intensity=0.125,
            )
            aprint(f"  Vertex markers: {n_vertex_points} points x {n_frames} frames")
            total_points += n_vertex_points * n_frames

        # --- Overlays ---
        # Title
        scene.add_text(
            "Particle Collision (Animated)",
            position=(0.02, 0.02),
            font_size=0.055,
            anchor="top-left",
            color="rgba(255,255,255,0.6)",
            blend_mode="difference",
        )

        # Particle type legend (bottom-left) — same as static demo
        scene.add_html(
            '<div style="font-size:1.3vh;line-height:1.6;background:rgba(0,0,0,0.5);padding:0.6vh;border-radius:3px">'
            '<div style="font-weight:bold;color:#ccc;margin-bottom:0.4vh">Particle Tracks</div>'
            '<div><span style="color:#6699ff">\u2588</span> e\u207b/e\u207a (electrons)</div>'
            '<div><span style="color:#ff4466">\u2588</span> \u03bc\u207b/\u03bc\u207a (muons)</div>'
            '<div><span style="color:#44cc44">\u2588</span> \u03c0\u00b1/K\u00b1 (hadrons)</div>'
            '<div><span style="color:#ffaa22">\u2588</span> p/p\u0304 (protons)</div>'
            '<div><span style="color:#ffff44">\u2588</span> \u03b3 (photons)</div>'
            "</div>",
            position=(0.02, 0.97),
            anchor="bottom-left",
        )

        # Info (bottom-right)
        scene.add_text(
            "Animated tracks \u2022 Detector simulation",
            position=(0.98, 0.97),
            font_size=0.015,
            anchor="bottom-right",
            color="rgba(200,200,200,0.45)",
        )

    return total_segments, total_points


# =============================================================================
# Main Entry Point
# =============================================================================


def main() -> None:
    """Main demo entry point."""
    n_events = 5
    n_jets = 4
    n_frames = 250

    if len(sys.argv) > 1:
        for arg in sys.argv[1:]:
            if arg.startswith("--events="):
                n_events = int(arg.split("=")[1])
            elif arg.startswith("--jets="):
                n_jets = int(arg.split("=")[1])
            elif arg.startswith("--frames="):
                n_frames = int(arg.split("=")[1])

    aprint("=" * 70)
    aprint("ANIMATED PARTICLE COLLISION VISUALIZATION")
    aprint("=" * 70)
    aprint("")
    aprint("Watch particle tracks grow outward from the collision point!")
    aprint("")
    aprint("  Physics Features:")
    aprint("    - Charged particles curve in magnetic field (Lorentz force)")
    aprint("    - Electrons/positrons: tight spirals, stop in EM calorimeter")
    aprint("    - Muons: gentle curves, traverse entire detector")
    aprint("    - Hadrons: medium curves, stop in hadronic calorimeter")
    aprint("    - Jets: collimated sprays from quark/gluon fragmentation")
    aprint("")
    aprint("  Animation Timeline (50 nanoseconds total):")
    aprint("    - t=0 ns: Collision flash at vertex (golden glow)")
    aprint("    - t=0-20 ns: Tracks emerge and extend through tracker")
    aprint("    - t=20-35 ns: Particles reach calorimeters")
    aprint("    - t=35-50 ns: Energy deposits appear, full event visible")
    aprint("")
    aprint("  Particle Colors:")
    aprint("    - Blue: electrons")
    aprint("    - Red: positrons")
    aprint("    - Green: muons (negative)")
    aprint("    - Yellow: muons (positive)")
    aprint("    - Orange/Brown: pions")
    aprint("    - Pink: kaons")
    aprint("    - Purple: protons")
    aprint("    - White/cream: photons")
    aprint("")
    aprint(f"Events: {n_events} | Jets: {n_jets} | Frames: {n_frames}")
    aprint("")

    # If --no-serve, use persistent directory; otherwise temp for auto-cleanup
    if "--no-serve" in sys.argv:
        output_path = get_demos_output_dir() / "collision_animated.luxar.zarr"
        with asection("Generating animated detector scene"):
            total_segments, total_points = generate_animated_detector_scene(
                output_path,
                n_events=n_events,
                n_jets_per_event=n_jets,
                n_frames=n_frames,
            )
        aprint(f"Dataset generated at {output_path}")
        aprint(
            f"Total: {total_segments:,} segments, {total_points:,} points, {n_frames} frames"
        )
        return

    # Use temporary directory for serving (auto-cleanup on exit)
    with tempfile.TemporaryDirectory(prefix="luxar_demo_collision_anim_") as tmpdir:
        output_path = Path(tmpdir) / "collision_animated.luxar.zarr"

        with asection("Generating animated detector scene"):
            total_segments, total_points = generate_animated_detector_scene(
                output_path,
                n_events=n_events,
                n_jets_per_event=n_jets,
                n_frames=n_frames,
            )

        aprint("")
        aprint("=" * 70)
        aprint("ANIMATION COMPLETE")
        aprint(f"  {total_segments:,} line segments")
        aprint(f"  {total_points:,} points")
        aprint(f"  {n_frames} frames")
        aprint("=" * 70)
        aprint("")
        aprint("Use the TIME slider to watch the collision unfold!")
        aprint("")

        launch_viewer(output_path)

    aprint("Cleanup complete")


if __name__ == "__main__":
    main()
