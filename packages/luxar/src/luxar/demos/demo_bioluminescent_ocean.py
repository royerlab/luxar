#!/usr/bin/env python3
"""Self-Contained Demo: Bioluminescent Ocean

An ethereal underwater visualization featuring gracefully animated jellyfish
with flowing tentacles, glowing plankton, and bioluminescent deep-sea atmosphere.

================================================================================
BIOLUMINESCENCE: NATURE'S LIVING LIGHT
================================================================================

Bioluminescence is the production of light by living organisms through chemical
reactions. It occurs in many marine species, from bacteria to fish.

JELLYFISH BIOLUMINESCENCE
-------------------------
Many jellyfish species produce light through proteins like:
- Green Fluorescent Protein (GFP) - discovered in Aequorea victoria
- Aequorin - a calcium-activated photoprotein
- Luciferin/luciferase systems

The light serves various purposes:
- Defense (startling predators)
- Attracting prey
- Communication
- Camouflage (counter-illumination)

JELLYFISH ANATOMY
-----------------
- Bell (medusa): The dome-shaped body that pulses for locomotion
- Oral arms: Frilly appendages around the mouth
- Tentacles: Long trailing appendages with stinging cells (cnidocytes)
- Gastrovascular cavity: Central digestive system
- Radial canals: Distribute nutrients through the bell

JELLYFISH LOCOMOTION
--------------------
Jellyfish move by jet propulsion:
1. Bell contracts, expelling water downward
2. This pushes the jellyfish upward
3. Bell relaxes and refills with water
4. Cycle repeats at ~0.5-2 Hz depending on species

The motion is remarkably efficient - jellyfish are among the most
energy-efficient swimmers in the animal kingdom.

DEEP SEA ENVIRONMENT
--------------------
Below 200m (mesopelagic zone), sunlight fades rapidly.
Below 1000m (bathypelagic zone), it's completely dark.
In this darkness, bioluminescence becomes the primary light source.

~76% of deep-sea animals produce their own light!

================================================================================

Usage:
    python demo_bioluminescent_ocean.py [--jellyfish=N] [--frames=N]

Controls:
    - Use dimension sliders to animate through time
    - Ctrl+C to stop and cleanup
"""

from __future__ import annotations

import sys
import tempfile
from dataclasses import dataclass
from pathlib import Path

import numpy as np
from arbol import aprint, asection

from luxar import Dimension, Dimensions, LuxarZarrCompiler
from luxar.demos import launch_viewer
from luxar.utils.paths import get_demos_output_dir

# =============================================================================
# Configuration
# =============================================================================

# Animation settings
DEFAULT_N_FRAMES = 250  # Smooth animation
DEFAULT_N_JELLYFISH = 8

# Ocean volume (arbitrary units)
OCEAN_WIDTH = 30.0
OCEAN_HEIGHT = 40.0
OCEAN_DEPTH = 30.0

# Jellyfish parameters
BELL_SEGMENTS = 24  # Radial segments for bell
BELL_RINGS = 12  # Concentric rings on bell
N_TENTACLES = 12  # Number of main tentacles
TENTACLE_SEGMENTS = 40  # Points per tentacle
N_ORAL_ARMS = 4  # Frilly oral arms

# Animation parameters
PULSE_FREQUENCY = 0.02  # Pulses per frame (slower = more graceful)
DRIFT_SPEED = 0.015  # Gentle upward drift


# =============================================================================
# Color Palettes - Ethereal Bioluminescent Colors
# =============================================================================

# Jellyfish color schemes (bell_color, tentacle_color, glow_color)
JELLYFISH_PALETTES = [
    # Crystal blue jellyfish
    {
        "bell": np.array([0.3, 0.6, 1.0]),
        "tentacle": np.array([0.2, 0.4, 0.9]),
        "glow": np.array([0.5, 0.8, 1.0]),
        "name": "crystal_blue",
    },
    # Pink/magenta moon jelly
    {
        "bell": np.array([0.9, 0.4, 0.7]),
        "tentacle": np.array([0.7, 0.3, 0.6]),
        "glow": np.array([1.0, 0.5, 0.8]),
        "name": "pink_moon",
    },
    # Golden sea nettle
    {
        "bell": np.array([1.0, 0.7, 0.3]),
        "tentacle": np.array([0.9, 0.5, 0.2]),
        "glow": np.array([1.0, 0.8, 0.4]),
        "name": "golden_nettle",
    },
    # Deep purple
    {
        "bell": np.array([0.6, 0.3, 0.9]),
        "tentacle": np.array([0.5, 0.2, 0.8]),
        "glow": np.array([0.7, 0.4, 1.0]),
        "name": "deep_purple",
    },
    # Cyan/turquoise
    {
        "bell": np.array([0.2, 0.9, 0.8]),
        "tentacle": np.array([0.1, 0.7, 0.7]),
        "glow": np.array([0.3, 1.0, 0.9]),
        "name": "turquoise",
    },
    # Ghost white
    {
        "bell": np.array([0.9, 0.9, 1.0]),
        "tentacle": np.array([0.7, 0.7, 0.9]),
        "glow": np.array([1.0, 1.0, 1.0]),
        "name": "ghost",
    },
]


# =============================================================================
# Jellyfish Geometry Generation
# =============================================================================


@dataclass
class JellyfishParams:
    """Parameters defining a single jellyfish."""

    position: np.ndarray  # Center position (x, y, z)
    size: float  # Bell radius
    palette: dict  # Color palette
    pulse_phase: float  # Starting phase in pulse cycle
    pulse_speed: float  # Individual pulse frequency multiplier
    drift_direction: np.ndarray  # Slight drift direction
    wobble_phase: float  # Phase for gentle side-to-side motion
    tentacle_lengths: np.ndarray  # Length of each tentacle


def generate_bell_points(
    jelly: JellyfishParams,
    frame: int,
    n_frames: int,
    rng: np.random.Generator,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Generate the bell (dome) of a jellyfish as points.

    The bell pulses rhythmically - contracting to push water out,
    then relaxing to refill.

    Returns:
        positions, colors, radii, sharpness
    """
    # Calculate pulse state (0 = relaxed, 1 = contracted)
    t = frame / n_frames
    pulse = 0.5 + 0.5 * np.sin(
        2 * np.pi * (frame * PULSE_FREQUENCY * jelly.pulse_speed + jelly.pulse_phase)
    )

    # Bell shape parameters vary with pulse
    # When contracted: flatter, wider
    # When relaxed: taller, narrower
    height_factor = 0.6 + 0.3 * (1 - pulse)  # Taller when relaxed
    width_factor = 1.0 + 0.15 * pulse  # Wider when contracted

    positions = []
    colors = []
    radii = []
    sharpness = []

    # Generate bell surface points
    for ring in range(BELL_RINGS):
        ring_frac = ring / (BELL_RINGS - 1)  # 0 at top, 1 at edge

        # Bell profile: dome shape using cosine
        # Modified by pulse state
        ring_radius = jelly.size * width_factor * np.sin(ring_frac * np.pi / 2)
        ring_height = jelly.size * height_factor * np.cos(ring_frac * np.pi / 2)

        # Add slight wobble
        wobble = (
            0.02
            * jelly.size
            * np.sin(2 * np.pi * (t * 0.5 + jelly.wobble_phase + ring_frac * 0.3))
        )

        for seg in range(BELL_SEGMENTS):
            angle = 2 * np.pi * seg / BELL_SEGMENTS

            # Position on bell surface
            x = ring_radius * np.cos(angle) + wobble * np.cos(angle * 2)
            y = ring_height
            z = ring_radius * np.sin(angle) + wobble * np.sin(angle * 2)

            # Transform to world position with drift
            drift = jelly.drift_direction * frame * DRIFT_SPEED
            pos = jelly.position + np.array([x, y, z]) + drift

            positions.append(pos)

            # Color: brighter at edges, bioluminescent glow
            edge_glow = ring_frac**0.5  # Brighter toward edges
            color = jelly.palette["bell"] * (0.6 + 0.4 * edge_glow)

            # Add subtle pulsing glow
            glow_intensity = 0.2 * pulse
            color = color + jelly.palette["glow"] * glow_intensity

            colors.append(np.clip(color, 0, 1))

            # Size varies: smaller at top, larger at edges
            point_radius = 0.08 * jelly.size * (0.5 + 0.5 * ring_frac)
            radii.append(point_radius)

            # Softer/glowier appearance (normalized [0, 1] knob)
            sharpness.append(0.3 + 0.2 * rng.random())

    # Add central glow points (bioluminescent organs)
    n_glow = 8
    for i in range(n_glow):
        angle = 2 * np.pi * i / n_glow
        glow_r = jelly.size * 0.3
        glow_h = jelly.size * height_factor * 0.5

        drift = jelly.drift_direction * frame * DRIFT_SPEED
        pos = jelly.position + np.array(
            [glow_r * np.cos(angle), glow_h, glow_r * np.sin(angle)]
        )
        pos = pos + drift

        positions.append(pos)
        colors.append(jelly.palette["glow"] * (0.8 + 0.2 * pulse))
        radii.append(0.15 * jelly.size)
        sharpness.append(0.3)  # Very soft glow

    return (
        np.array(positions, dtype=np.float32),
        np.array(colors, dtype=np.float32),
        np.array(radii, dtype=np.float32),
        np.array(sharpness, dtype=np.float32),
    )


def generate_tentacles(
    jelly: JellyfishParams,
    frame: int,
    n_frames: int,
    rng: np.random.Generator,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Generate flowing tentacles as line segments.

    Tentacles wave gracefully with a combination of:
    - Overall drift from jellyfish movement
    - Sinusoidal waves propagating down the tentacle
    - Random perturbations for organic feel

    Returns:
        vertices, widths, colors, sharpness
    """
    t = frame / n_frames

    # Pulse state affects tentacle attachment point
    pulse = 0.5 + 0.5 * np.sin(
        2 * np.pi * (frame * PULSE_FREQUENCY * jelly.pulse_speed + jelly.pulse_phase)
    )
    width_factor = 1.0 + 0.15 * pulse

    all_vertices = []
    all_widths = []
    all_colors = []
    all_sharpness = []

    for tent_idx in range(N_TENTACLES):
        # Attachment point on bell edge
        attach_angle = 2 * np.pi * tent_idx / N_TENTACLES
        attach_r = jelly.size * width_factor * 0.95
        attach_x = attach_r * np.cos(attach_angle)
        attach_z = attach_r * np.sin(attach_angle)
        attach_y = 0  # Bottom of bell

        # Tentacle length (varies per tentacle)
        tent_length = jelly.tentacle_lengths[tent_idx]

        # Generate points along tentacle
        points = []
        for seg in range(TENTACLE_SEGMENTS):
            seg_frac = seg / (TENTACLE_SEGMENTS - 1)

            # Base position: hanging down with slight outward curve
            base_y = -seg_frac * tent_length
            outward = 0.2 * jelly.size * seg_frac * (1 + 0.5 * seg_frac)
            base_x = attach_x + outward * np.cos(attach_angle)
            base_z = attach_z + outward * np.sin(attach_angle)

            # Wave motion: sinusoidal waves propagating down
            # Multiple frequencies for organic motion
            wave_phase = t * 3 + seg_frac * 4 + tent_idx * 0.5
            wave1 = 0.15 * jelly.size * np.sin(wave_phase * 2 * np.pi) * seg_frac
            wave2 = 0.08 * jelly.size * np.sin(wave_phase * 1.3 * 2 * np.pi) * seg_frac
            wave3 = 0.05 * jelly.size * np.sin(wave_phase * 2.1 * 2 * np.pi) * seg_frac

            # Apply waves perpendicular to tentacle direction
            perp_angle = attach_angle + np.pi / 2
            wave_x = (wave1 + wave3) * np.cos(perp_angle)
            wave_z = (wave1 + wave3) * np.sin(perp_angle)
            wave_y = wave2

            # Combine position
            x = base_x + wave_x
            y = attach_y + base_y + wave_y
            z = base_z + wave_z

            # Add global drift
            drift = jelly.drift_direction * frame * DRIFT_SPEED
            # Tentacle tips lag behind (drag effect)
            drag_factor = 1.0 - 0.3 * seg_frac
            effective_drift = drift * drag_factor

            pos = jelly.position + np.array([x, y, z]) + effective_drift
            points.append(pos)

        # Convert to line segments
        points = np.array(points, dtype=np.float32)
        for i in range(len(points) - 1):
            all_vertices.extend([points[i], points[i + 1]])

            # Width tapers along tentacle
            seg_frac = i / (len(points) - 1)
            width = 0.02 * jelly.size * (1 - 0.8 * seg_frac)
            all_widths.extend([width, width * 0.95])

            # Color fades along tentacle
            fade = 1.0 - 0.5 * seg_frac
            color = jelly.palette["tentacle"] * fade
            all_colors.extend([color, color * 0.98])

            all_sharpness.extend([0.5, 0.5])

    return (
        np.array(all_vertices, dtype=np.float32),
        np.array(all_widths, dtype=np.float32),
        np.array(all_colors, dtype=np.float32),
        np.array(all_sharpness, dtype=np.float32),
    )


def generate_oral_arms(
    jelly: JellyfishParams,
    frame: int,
    n_frames: int,
    rng: np.random.Generator,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Generate oral arms (frilly appendages around mouth).

    These are shorter, thicker, and more frilly than tentacles.

    Returns:
        vertices, widths, colors, sharpness
    """
    t = frame / n_frames
    _ = 0.5 + 0.5 * np.sin(
        2 * np.pi * (frame * PULSE_FREQUENCY * jelly.pulse_speed + jelly.pulse_phase)
    )

    all_vertices = []
    all_widths = []
    all_colors = []
    all_sharpness = []

    oral_arm_length = jelly.size * 1.2
    oral_arm_segments = 25

    for arm_idx in range(N_ORAL_ARMS):
        attach_angle = 2 * np.pi * arm_idx / N_ORAL_ARMS + np.pi / N_ORAL_ARMS
        attach_r = jelly.size * 0.2

        points = []
        for seg in range(oral_arm_segments):
            seg_frac = seg / (oral_arm_segments - 1)

            # Oral arms hang more directly down with gentle curve
            base_y = -seg_frac * oral_arm_length
            outward = 0.1 * jelly.size * np.sin(seg_frac * np.pi)
            base_x = attach_r * np.cos(attach_angle) + outward * np.cos(attach_angle)
            base_z = attach_r * np.sin(attach_angle) + outward * np.sin(attach_angle)

            # Frilly motion
            frilly_phase = t * 2 + seg_frac * 3 + arm_idx * 0.7
            frilly = 0.1 * jelly.size * np.sin(frilly_phase * 2 * np.pi) * seg_frac

            x = base_x + frilly * np.cos(attach_angle + np.pi / 2)
            y = base_y
            z = base_z + frilly * np.sin(attach_angle + np.pi / 2)

            drift = jelly.drift_direction * frame * DRIFT_SPEED
            drag_factor = 1.0 - 0.2 * seg_frac
            pos = jelly.position + np.array([x, y, z]) + drift * drag_factor

            points.append(pos)

        points = np.array(points, dtype=np.float32)
        for i in range(len(points) - 1):
            all_vertices.extend([points[i], points[i + 1]])

            seg_frac = i / (len(points) - 1)
            width = 0.04 * jelly.size * (1 - 0.6 * seg_frac)
            all_widths.extend([width, width * 0.95])

            # Oral arms are slightly more colorful
            color = jelly.palette["bell"] * 0.8 + jelly.palette["glow"] * 0.2
            fade = 1.0 - 0.3 * seg_frac
            all_colors.extend([color * fade, color * fade * 0.98])

            all_sharpness.extend([1.0, 1.0])

    return (
        np.array(all_vertices, dtype=np.float32),
        np.array(all_widths, dtype=np.float32),
        np.array(all_colors, dtype=np.float32),
        np.array(all_sharpness, dtype=np.float32),
    )


# =============================================================================
# Plankton and Ambient Particles
# =============================================================================


def generate_plankton(
    n_particles: int,
    frame: int,
    n_frames: int,
    rng: np.random.Generator,
    base_positions: np.ndarray,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Generate floating bioluminescent plankton.

    These drift gently and twinkle with bioluminescent light.

    Returns:
        positions, colors, radii, sharpness
    """
    t = frame / n_frames

    # Gentle drift and twinkle
    positions = base_positions.copy()

    # Add subtle motion
    for i in range(len(positions)):
        # Each particle has unique phase
        phase = i * 0.1
        positions[i, 0] += 0.1 * np.sin(2 * np.pi * (t * 0.3 + phase))
        positions[i, 1] += 0.05 * np.sin(2 * np.pi * (t * 0.2 + phase * 1.3))
        positions[i, 2] += 0.1 * np.sin(2 * np.pi * (t * 0.25 + phase * 0.7))

    # Twinkling brightness
    twinkle = np.zeros(n_particles)
    for i in range(n_particles):
        phase = i * 0.17
        twinkle[i] = 0.3 + 0.7 * (0.5 + 0.5 * np.sin(2 * np.pi * (t * 2 + phase)))

    # Colors: mix of cyan, blue, green bioluminescence
    colors = np.zeros((n_particles, 3), dtype=np.float32)
    color_types = [
        np.array([0.2, 0.8, 1.0]),  # Cyan
        np.array([0.3, 0.5, 1.0]),  # Blue
        np.array([0.2, 1.0, 0.6]),  # Green
        np.array([0.5, 0.3, 1.0]),  # Purple
    ]

    for i in range(n_particles):
        base_color = color_types[i % len(color_types)]
        colors[i] = base_color * twinkle[i]

    # Small, soft points
    radii = 0.03 + 0.02 * rng.random(n_particles)
    radii = radii.astype(np.float32)

    sharpness = 0.3 + 0.4 * rng.random(n_particles)
    sharpness = sharpness.astype(np.float32)

    return positions.astype(np.float32), colors, radii, sharpness


def generate_deep_particles(
    n_particles: int,
    frame: int,
    n_frames: int,
    rng: np.random.Generator,
    base_positions: np.ndarray,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Generate marine snow and suspended particles.

    These create depth and atmosphere in the scene.
    """
    t = frame / n_frames

    positions = base_positions.copy()

    # Very slow sinking motion
    for i in range(len(positions)):
        phase = i * 0.07
        positions[i, 1] -= t * 0.5  # Slow sink
        positions[i, 0] += 0.05 * np.sin(2 * np.pi * (t * 0.1 + phase))
        positions[i, 2] += 0.05 * np.cos(2 * np.pi * (t * 0.1 + phase))

    # Wrap around vertically
    positions[:, 1] = (
        np.mod(positions[:, 1] + OCEAN_HEIGHT / 2, OCEAN_HEIGHT) - OCEAN_HEIGHT / 2
    )

    # Dim, blue-tinted colors
    colors = np.zeros((n_particles, 3), dtype=np.float32)
    for i in range(n_particles):
        brightness = 0.1 + 0.1 * rng.random()
        colors[i] = np.array([0.3, 0.4, 0.6]) * brightness

    radii = 0.02 + 0.01 * rng.random(n_particles)
    radii = radii.astype(np.float32)

    sharpness = np.full(n_particles, 0.5, dtype=np.float32)

    return positions.astype(np.float32), colors, radii, sharpness


# =============================================================================
# Main Scene Generation
# =============================================================================


def create_jellyfish(
    n_jellyfish: int,
    rng: np.random.Generator,
) -> list[JellyfishParams]:
    """Create a school of jellyfish with varied parameters."""
    jellyfish = []

    for i in range(n_jellyfish):
        # Random position in ocean volume
        position = np.array(
            [
                rng.uniform(-OCEAN_WIDTH / 2, OCEAN_WIDTH / 2),
                rng.uniform(-OCEAN_HEIGHT / 3, OCEAN_HEIGHT / 2),
                rng.uniform(-OCEAN_DEPTH / 2, OCEAN_DEPTH / 2),
            ]
        )

        # Size variation
        size = rng.uniform(1.5, 4.0)

        # Random palette
        palette = JELLYFISH_PALETTES[i % len(JELLYFISH_PALETTES)]

        # Animation parameters
        pulse_phase = rng.uniform(0, 1)
        pulse_speed = rng.uniform(0.8, 1.2)

        # Gentle drift (mostly upward)
        drift_direction = np.array(
            [
                rng.uniform(-0.3, 0.3),
                rng.uniform(0.5, 1.0),  # Upward bias
                rng.uniform(-0.3, 0.3),
            ]
        )
        drift_direction = drift_direction / np.linalg.norm(drift_direction)

        wobble_phase = rng.uniform(0, 1)

        # Variable tentacle lengths
        tentacle_lengths = size * (2.0 + 1.5 * rng.random(N_TENTACLES))

        jellyfish.append(
            JellyfishParams(
                position=position,
                size=size,
                palette=palette,
                pulse_phase=pulse_phase,
                pulse_speed=pulse_speed,
                drift_direction=drift_direction,
                wobble_phase=wobble_phase,
                tentacle_lengths=tentacle_lengths,
            )
        )

    return jellyfish


def generate_ocean_scene(
    output_path: Path,
    n_jellyfish: int = DEFAULT_N_JELLYFISH,
    n_frames: int = DEFAULT_N_FRAMES,
) -> tuple[int, int]:
    """Generate the complete bioluminescent ocean scene.

    Returns:
        (total_line_segments, total_points)
    """
    rng = np.random.default_rng(42)

    # Create jellyfish
    jellyfish_list = create_jellyfish(n_jellyfish, rng)

    # Pre-generate static positions for plankton and particles
    n_plankton = 2000
    plankton_base = np.column_stack(
        [
            rng.uniform(-OCEAN_WIDTH / 2, OCEAN_WIDTH / 2, n_plankton),
            rng.uniform(-OCEAN_HEIGHT / 2, OCEAN_HEIGHT / 2, n_plankton),
            rng.uniform(-OCEAN_DEPTH / 2, OCEAN_DEPTH / 2, n_plankton),
        ]
    )

    n_deep_particles = 1500
    deep_base = np.column_stack(
        [
            rng.uniform(-OCEAN_WIDTH / 2, OCEAN_WIDTH / 2, n_deep_particles),
            rng.uniform(-OCEAN_HEIGHT / 2, OCEAN_HEIGHT / 2, n_deep_particles),
            rng.uniform(-OCEAN_DEPTH / 2, OCEAN_DEPTH / 2, n_deep_particles),
        ]
    )

    total_segments = 0
    total_points = 0

    with LuxarZarrCompiler(output_path) as compiler:
        # 4D: x, y, z, time
        dims = Dimensions(
            [
                Dimension("x", unit="m", display=True),
                Dimension("y", unit="m", display=True),
                Dimension("z", unit="m", display=True),
                Dimension(
                    "time",
                    unit="frame",
                    display=False,
                    range=(0, n_frames - 1),
                    step=1,
                    discrete=True,
                ),
            ]
        )
        scene = compiler.create_scene(dimensions=dims)

        with asection(f"Generating {n_frames} frames of animation"):
            # Collect all data across frames
            all_bell_pos = []
            all_bell_colors = []
            all_bell_radii = []
            all_bell_sharp = []

            all_tent_verts = []
            all_tent_widths = []
            all_tent_colors = []
            all_tent_sharp = []

            for frame in range(n_frames):
                if frame % 50 == 0:
                    aprint(f"  Frame {frame}/{n_frames}")

                frame_bell_pos = []
                frame_bell_colors = []
                frame_bell_radii = []
                frame_bell_sharp = []

                frame_tent_verts = []
                frame_tent_widths = []
                frame_tent_colors = []
                frame_tent_sharp = []

                # Generate each jellyfish
                for jelly in jellyfish_list:
                    # Bell points
                    pos, col, rad, shrp = generate_bell_points(
                        jelly, frame, n_frames, rng
                    )
                    # Add time dimension
                    pos_4d = np.column_stack([pos, np.full(len(pos), frame)])
                    frame_bell_pos.append(pos_4d)
                    frame_bell_colors.append(col)
                    frame_bell_radii.append(rad)
                    frame_bell_sharp.append(shrp)

                    # Tentacle lines
                    verts, widths, colors, shrp = generate_tentacles(
                        jelly, frame, n_frames, rng
                    )
                    # Add time dimension to vertices
                    verts_4d = np.column_stack([verts, np.full(len(verts), frame)])
                    frame_tent_verts.append(verts_4d)
                    frame_tent_widths.append(widths)
                    frame_tent_colors.append(colors)
                    frame_tent_sharp.append(shrp)

                    # Oral arms
                    verts, widths, colors, shrp = generate_oral_arms(
                        jelly, frame, n_frames, rng
                    )
                    verts_4d = np.column_stack([verts, np.full(len(verts), frame)])
                    frame_tent_verts.append(verts_4d)
                    frame_tent_widths.append(widths)
                    frame_tent_colors.append(colors)
                    frame_tent_sharp.append(shrp)

                # Plankton
                pos, col, rad, shrp = generate_plankton(
                    n_plankton, frame, n_frames, rng, plankton_base
                )
                pos_4d = np.column_stack([pos, np.full(len(pos), frame)])
                frame_bell_pos.append(pos_4d)
                frame_bell_colors.append(col)
                frame_bell_radii.append(rad)
                frame_bell_sharp.append(shrp)

                # Deep particles (marine snow)
                pos, col, rad, shrp = generate_deep_particles(
                    n_deep_particles, frame, n_frames, rng, deep_base
                )
                pos_4d = np.column_stack([pos, np.full(len(pos), frame)])
                frame_bell_pos.append(pos_4d)
                frame_bell_colors.append(col)
                frame_bell_radii.append(rad)
                frame_bell_sharp.append(shrp)

                # Concatenate frame data
                all_bell_pos.append(np.concatenate(frame_bell_pos, axis=0))
                all_bell_colors.append(np.concatenate(frame_bell_colors, axis=0))
                all_bell_radii.append(np.concatenate(frame_bell_radii, axis=0))
                all_bell_sharp.append(np.concatenate(frame_bell_sharp, axis=0))

                all_tent_verts.append(np.concatenate(frame_tent_verts, axis=0))
                all_tent_widths.append(np.concatenate(frame_tent_widths, axis=0))
                all_tent_colors.append(np.concatenate(frame_tent_colors, axis=0))
                all_tent_sharp.append(np.concatenate(frame_tent_sharp, axis=0))

        # Write all points
        with asection("Writing jellyfish and plankton points"):
            all_pos = np.concatenate(all_bell_pos, axis=0)
            all_col = np.concatenate(all_bell_colors, axis=0)
            all_rad = np.concatenate(all_bell_radii, axis=0)
            all_shrp = np.concatenate(all_bell_sharp, axis=0)

            scene.add_points(
                "bioluminescent_life",
                positions=all_pos.astype(np.float32),
                colors=all_col.astype(np.float32),
                radii=all_rad.astype(np.float32),
                sharpness=all_shrp.astype(np.float32),
                intensity=0.5,
                layer=True,
            )
            total_points = len(all_pos)
            aprint(f"Total points: {total_points:,}")

        # Write all lines (tentacles + oral arms)
        with asection("Writing tentacles and oral arms"):
            all_verts = np.concatenate(all_tent_verts, axis=0)
            all_widths = np.concatenate(all_tent_widths, axis=0)
            all_colors = np.concatenate(all_tent_colors, axis=0)
            all_shrp = np.concatenate(all_tent_sharp, axis=0)

            scene.add_lines(
                "tentacles",
                vertices=all_verts.astype(np.float32),
                widths=all_widths.astype(np.float32),
                colors=all_colors.astype(np.float32),
                sharpness=all_shrp.astype(np.float32),
                line_type="segments",
                intensity=0.5,
                layer=True,
            )
            total_segments = len(all_verts) // 2
            aprint(f"Total line segments: {total_segments:,}")

        # --- Overlays ---
        # Title
        scene.add_text(
            "Bioluminescent Ocean",
            position=(0.02, 0.02),
            font_size=0.055,
            anchor="top-left",
            color="rgba(255,255,255,0.6)",
            blend_mode="difference",
        )

        # Info
        scene.add_text(
            "8 jellyfish \u2022 250 frames",
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
    n_jellyfish = DEFAULT_N_JELLYFISH
    n_frames = DEFAULT_N_FRAMES

    # Parse arguments
    for arg in sys.argv[1:]:
        if arg.startswith("--jellyfish="):
            n_jellyfish = int(arg.split("=")[1])
        elif arg.startswith("--frames="):
            n_frames = int(arg.split("=")[1])

    aprint("=" * 70)
    aprint("BIOLUMINESCENT OCEAN")
    aprint("=" * 70)
    aprint("")
    aprint("An ethereal underwater world of glowing jellyfish and plankton.")
    aprint("")
    aprint("  Features:")
    aprint("    - Gracefully pulsing jellyfish bells")
    aprint("    - Flowing, wave-like tentacle motion")
    aprint("    - Twinkling bioluminescent plankton")
    aprint("    - Marine snow drifting through the depths")
    aprint("")
    aprint("  Animation:")
    aprint("    - Use the time slider to animate")
    aprint("    - ~250 frames of smooth motion")
    aprint("")
    aprint(f"Jellyfish: {n_jellyfish} | Frames: {n_frames}")
    aprint("")

    # If --no-serve, use persistent directory; otherwise temp for auto-cleanup
    if "--no-serve" in sys.argv:
        output_path = get_demos_output_dir() / "ocean.luxar.zarr"
        with asection("Generating ocean scene"):
            generate_ocean_scene(
                output_path, n_jellyfish=n_jellyfish, n_frames=n_frames
            )
        aprint(f"Dataset generated at {output_path}")
        return

    # Use temporary directory for serving (auto-cleanup on exit)
    with tempfile.TemporaryDirectory(prefix="luxar_demo_ocean_") as tmpdir:
        output_path = Path(tmpdir) / "ocean.luxar.zarr"

        with asection("Generating ocean scene"):
            total_segments, total_points = generate_ocean_scene(
                output_path, n_jellyfish=n_jellyfish, n_frames=n_frames
            )

        aprint("")
        aprint("=" * 70)
        aprint(f"SCENE COMPLETE: {total_segments:,} segments, {total_points:,} points")
        aprint("=" * 70)
        aprint("")
        aprint("Jellyfish color guide:")
        for palette in JELLYFISH_PALETTES:
            aprint(f"  - {palette['name']}")
        aprint("")
        aprint("=" * 70)
        aprint("LAUNCHING VIEWER")
        aprint("=" * 70)
        aprint("Use the TIME dimension slider to animate the jellyfish!")
        aprint("Press Ctrl+C when done.")
        aprint("")

        launch_viewer(output_path)

    aprint("Cleanup complete - temporary files removed")


if __name__ == "__main__":
    main()
