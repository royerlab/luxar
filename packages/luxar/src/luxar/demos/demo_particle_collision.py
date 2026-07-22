#!/usr/bin/env python3
"""Self-Contained Demo: Particle Collision Detector Visualization

A realistic visualization of particle physics collisions inspired by CERN's
ATLAS and CMS detectors. Shows particle tracks, jets, and energy deposits
from high-energy proton-proton collisions.

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

Usage:
    python demo_particle_collision.py [--events=N] [--jets=N]

Controls:
    - Ctrl+C to stop and cleanup
    - Browser opens automatically
"""

from __future__ import annotations

DEMO_META = {
    "key": "collision",
    "title": "Collision",
    "description": "A synthetic particle-physics collision with curved tracks, jets, and calorimeter deposits.",
    "category": "synthetic",
    "geometry": "points",
    "requirements": {
        "download_mb": 0,
        "compute": "light",
        "gpu": "none",
        "local_data": None,
    },
    "caches": [],
    "outputs": ["collision"],
}

import sys
import tempfile
from dataclasses import dataclass
from pathlib import Path
from typing import Optional

import numpy as np
from arbol import aprint, asection

from luxar import Dimension, Dimensions, LuxarZarrCompiler
from luxar.demos import launch_viewer
from luxar.utils.paths import get_demos_output_dir

# =============================================================================
# Physics Constants and Detector Geometry
# =============================================================================
#
# These values are scaled for visualization but maintain realistic RATIOS.
# Real detector dimensions (ATLAS): beam pipe ~5cm, tracker ~1m, ECAL ~2m, etc.
# We scale by ~10x for better visualization of track curvature.

# Detector layer radii (visualization units, roughly corresponding to meters)
# Real ATLAS: beam pipe=5cm, pixel=5-12cm, SCT=30-52cm, TRT=56-107cm
BEAM_PIPE_RADIUS = 0.5  # Central vacuum tube where beams collide
TRACKER_INNER = 1.0  # Silicon pixel detector - highest precision
TRACKER_OUTER = 4.0  # Silicon strip + transition radiation tracker
ECAL_INNER = 4.5  # Electromagnetic calorimeter - lead/liquid-argon
ECAL_OUTER = 6.0  # ~25 radiation lengths to fully contain EM showers
HCAL_INNER = 6.5  # Hadronic calorimeter - iron/scintillator tiles
HCAL_OUTER = 10.0  # ~10 interaction lengths for hadron containment
MUON_INNER = 11.0  # Muon spectrometer - drift tubes and chambers
MUON_OUTER = 15.0  # Outermost detector layer
DETECTOR_LENGTH = 25.0  # Half-length in z (beam direction)

# Magnetic field strength
# Real ATLAS: 2 Tesla solenoid in inner detector
# Real CMS: 3.8 Tesla (strongest at any collider)
B_FIELD = 2.0  # Tesla (affects curvature via r = p_T / (q*B))

# =============================================================================
# Particle Properties (from Particle Data Group - PDG 2024)
# =============================================================================
# Masses in MeV/c² (rest mass energy equivalent)
# These are REAL measured values from experiments!
#
# Stopping behavior based on particle-matter interactions:
# - Electrons: Bremsstrahlung radiation → EM shower → stopped in ECAL
# - Photons: Pair production (γ → e⁺e⁻) → EM shower → stopped in ECAL
# - Hadrons: Strong nuclear interactions → hadronic shower → stopped in HCAL
# - Muons: Minimum ionizing particles (MIP) → penetrate everything

PARTICLE_TYPES = {
    # LEPTONS (fundamental particles, no strong interaction)
    "electron": {
        "charge": -1,
        "mass": 0.511,  # MeV/c² - PDG value: 0.51099895 MeV/c²
        "color": [0.2, 0.6, 1.0],  # Blue - convention in particle physics
        "stops_at": ECAL_OUTER,  # Creates EM shower via bremsstrahlung
    },
    "positron": {
        "charge": +1,
        "mass": 0.511,  # Antiparticle of electron, same mass (CPT theorem)
        "color": [1.0, 0.4, 0.4],  # Red - opposite charge = opposite color
        "stops_at": ECAL_OUTER,  # Also creates EM shower, then annihilates
    },
    "muon_minus": {
        "charge": -1,
        "mass": 105.7,  # MeV/c² - PDG value: 105.6583755 MeV/c²
        "color": [0.2, 1.0, 0.4],  # Green - distinct from electrons
        "stops_at": None,  # Penetrates entire detector! (MIP behavior)
    },
    "muon_plus": {
        "charge": +1,
        "mass": 105.7,  # Antimuon, same mass
        "color": [1.0, 1.0, 0.2],  # Yellow-green
        "stops_at": None,  # Also penetrates everything
    },
    # HADRONS (made of quarks, feel strong force)
    # Pions: lightest mesons, most common in jets
    "pion_plus": {
        "charge": +1,
        "mass": 139.6,  # MeV/c² - PDG value: 139.57039 MeV/c²
        "color": [1.0, 0.6, 0.2],  # Orange
        "stops_at": HCAL_OUTER,  # Hadronic shower via strong interaction
    },
    "pion_minus": {
        "charge": -1,
        "mass": 139.6,  # Same mass (isospin symmetry)
        "color": [0.8, 0.4, 0.1],  # Darker orange
        "stops_at": HCAL_OUTER,
    },
    # Kaons: strange mesons (contain strange quark)
    "kaon": {
        "charge": +1,
        "mass": 493.7,  # MeV/c² - PDG value: 493.677 MeV/c² (K⁺)
        "color": [0.9, 0.3, 0.6],  # Pink/magenta
        "stops_at": HCAL_OUTER,
    },
    # Proton: stable baryon (uud quarks)
    "proton": {
        "charge": +1,
        "mass": 938.3,  # MeV/c² - PDG value: 938.27208816 MeV/c²
        "color": [0.6, 0.2, 0.8],  # Purple
        "stops_at": HCAL_OUTER,
    },
    # BOSONS
    "photon": {
        "charge": 0,  # No charge → NO TRACK in magnetic field!
        "mass": 0,  # Massless (travels at speed of light)
        "color": [1.0, 1.0, 0.8],  # White/cream (light!)
        "stops_at": ECAL_OUTER,  # Converts to e⁺e⁻ pair → EM shower
    },
    # Neutron: neutral baryon (udd quarks)
    "neutron": {
        "charge": 0,  # No charge → no track
        "mass": 939.6,  # MeV/c² - PDG value: 939.56542052 MeV/c²
        "color": [0.5, 0.5, 0.5],  # Gray (neutral)
        "stops_at": HCAL_OUTER,  # Strong interaction → hadronic shower
    },
}


# =============================================================================
# Particle Track Generation
# =============================================================================
#
# HELIX PHYSICS EXPLANATION
# -------------------------
# In a uniform magnetic field B along the z-axis (beam direction), charged
# particles experience the Lorentz force:
#
#     F = q(v × B)
#
# This force is always perpendicular to velocity, so it changes direction
# but not speed. The result is circular motion in the x-y plane (transverse)
# combined with constant velocity along z → HELIX.
#
# The radius of the circular motion is:
#
#     r = m*v_T / (|q|*B) = p_T / (|q|*B)
#
# where p_T = transverse momentum = sqrt(px² + py²)
#
# KEY INSIGHTS:
# 1. Higher p_T → larger radius → straighter track
# 2. Positive charge → curves one way, negative → opposite
# 3. Mass doesn't directly affect radius (only through momentum)
# 4. Electrons curve tightly because they typically have low p_T
# 5. Muons curve gently because they typically have high p_T
#
# The helix pitch (z-advance per revolution) depends on p_z/p_T ratio.


@dataclass
class Particle:
    """Represents a particle with kinematic properties.

    In particle physics, we typically work with:
    - Energy E (GeV) - total relativistic energy
    - Momentum p (GeV/c) - 3-vector (px, py, pz)
    - Mass m (GeV/c²) - rest mass

    Related by: E² = (pc)² + (mc²)²

    For visualization, we use natural units where c = 1.
    """

    particle_type: str
    energy: float  # Total energy in GeV
    px: float  # x-component of momentum (GeV/c)
    py: float  # y-component of momentum (GeV/c)
    pz: float  # z-component of momentum (beam direction)
    origin: np.ndarray  # Starting point (collision vertex)

    @property
    def momentum(self) -> float:
        """Total momentum magnitude |p| = sqrt(px² + py² + pz²)."""
        return np.sqrt(self.px**2 + self.py**2 + self.pz**2)

    @property
    def pt(self) -> float:
        """Transverse momentum p_T = sqrt(px² + py²).

        This is the key quantity for track curvature since the magnetic
        field is along z. Higher p_T means straighter tracks.
        """
        return np.sqrt(self.px**2 + self.py**2)

    @property
    def charge(self) -> int:
        """Electric charge in units of elementary charge e."""
        return PARTICLE_TYPES[self.particle_type]["charge"]

    @property
    def color(self) -> list:
        """Visualization color for this particle type."""
        return PARTICLE_TYPES[self.particle_type]["color"]

    @property
    def stops_at(self) -> Optional[float]:
        """Detector radius where this particle is absorbed (None = escapes)."""
        return PARTICLE_TYPES[self.particle_type]["stops_at"]


def generate_helix_track(
    particle: Particle,
    rng: np.random.Generator,
    n_points: int = 100,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Generate a helical track for a charged particle in magnetic field.

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
    """
    if particle.charge == 0:
        # Neutral particles (photons, neutrons) have no charge
        # → No Lorentz force → straight line trajectory
        # They are only "seen" when they deposit energy in calorimeters
        return generate_straight_track(particle, rng, n_points)

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
    # HELIX POINT GENERATION
    # =========================================================================

    points = []
    t = 0  # Parametric time along helix
    dt = 0.05  # Step size (smaller = smoother curves)

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

        points.append([x, y, z])
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

    points = np.array(points, dtype=np.float32)

    # Create line segments
    n_segments = len(points) - 1
    vertices = np.zeros((n_segments * 2, 3), dtype=np.float32)
    vertices[0::2] = points[:-1]
    vertices[1::2] = points[1:]

    # Width tapers along track (energy loss visualization)
    # Use per-vertex widths for continuity at joints
    base_width = 0.015 + 0.01 * (particle.pt / 50.0)
    n_points = len(points)
    vertex_widths = base_width * (1.0 - 0.5 * np.linspace(0, 1, n_points))

    # Expand to segment format: widths[2i] = vertex i, widths[2i+1] = vertex i+1
    # This ensures widths match at shared vertices (joints)
    widths = np.zeros(n_segments * 2, dtype=np.float32)
    widths[0::2] = vertex_widths[:-1]  # start of each segment
    widths[1::2] = vertex_widths[1:]  # end of each segment

    # Colors with smooth fade along track
    # Use per-vertex colors for continuity at joints
    base_color = np.array(particle.color, dtype=np.float32)
    vertex_fades = 1.0 - 0.3 * np.linspace(0, 1, n_points)
    vertex_colors = base_color * vertex_fades[:, np.newaxis]

    # Add slight random variation per-vertex (before expansion for continuity)
    vertex_colors += rng.uniform(-0.05, 0.05, vertex_colors.shape).astype(np.float32)
    vertex_colors = np.clip(vertex_colors, 0, 1)

    # Expand to segment format
    colors = np.zeros((n_segments * 2, 3), dtype=np.float32)
    colors[0::2] = vertex_colors[:-1]  # start of each segment
    colors[1::2] = vertex_colors[1:]  # end of each segment

    return vertices, widths, colors


def generate_straight_track(
    particle: Particle,
    rng: np.random.Generator,
    n_points: int = 50,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Generate a straight track for neutral particles."""
    direction = np.array([particle.px, particle.py, particle.pz])
    direction = direction / (np.linalg.norm(direction) + 1e-10)

    max_radius = particle.stops_at if particle.stops_at else MUON_OUTER

    points = []
    for i in range(n_points):
        t = i * 0.5
        pos = particle.origin + direction * t
        r = np.sqrt(pos[0] ** 2 + pos[1] ** 2)
        if r > max_radius or abs(pos[2]) > DETECTOR_LENGTH:
            break
        points.append(pos)

    if len(points) < 2:
        return np.array([]).reshape(0, 3), np.array([]), np.array([]).reshape(0, 3)

    points = np.array(points, dtype=np.float32)
    n_segments = len(points) - 1

    vertices = np.zeros((n_segments * 2, 3), dtype=np.float32)
    vertices[0::2] = points[:-1]
    vertices[1::2] = points[1:]

    # Dashed appearance for neutral particles
    widths = np.full(n_segments * 2, 0.008, dtype=np.float32)

    base_color = np.array(particle.color, dtype=np.float32)
    colors = np.tile(base_color, (n_segments * 2, 1))

    return vertices, widths, colors


# =============================================================================
# Jet Generation
# =============================================================================
#
# QUANTUM CHROMODYNAMICS (QCD) AND JET PHYSICS
# ---------------------------------------------
# Jets are the experimental signature of quarks and gluons. Due to a property
# called "color confinement," quarks/gluons cannot exist freely. When produced
# in a collision, they immediately undergo "hadronization" - a QCD process
# where the color field energy creates quark-antiquark pairs that combine
# into colorless hadrons (mesons and baryons).
#
# FRAGMENTATION PROCESS:
#   q → q + (q̄q) → q + (q̄q) + (q̄q) → ...
#   This cascade produces a collimated spray of hadrons: a "jet"
#
# KEY JET PROPERTIES:
# 1. COLLIMATION: Higher energy → narrower jet (Lorentz boost)
#    Typical cone radius: R = sqrt(Δη² + Δφ²) ≈ 0.4
#
# 2. FRAGMENTATION FUNCTION: Energy distribution follows D(z)
#    where z = E_hadron / E_parton
#    Approximately exponential: more soft particles than hard ones
#
# 3. PARTICLE COMPOSITION (typical jet):
#    - ~60% pions (π±, π⁰) - lightest mesons
#    - ~25% kaons (K±, K⁰) - contain strange quarks
#    - ~15% protons/neutrons - baryons
#    - Plus photons from π⁰ → γγ decay
#
# 4. MULTIPLICITY: <n> ≈ 2.5 × ln(E_jet/1GeV)
#    More particles at higher energy


def generate_jet(
    origin: np.ndarray,
    direction: np.ndarray,
    energy: float,
    rng: np.random.Generator,
    n_particles: int = 15,
) -> list[Particle]:
    """Generate a jet of hadrons from quark/gluon fragmentation.

    PHYSICS:
    --------
    Jets arise from QCD color confinement. When a quark or gluon is produced,
    it cannot escape - instead, the strong force field energy creates new
    quark-antiquark pairs that combine into observable hadrons.

    This simulation models:
    1. Jet cone angle inversely proportional to energy (Lorentz boost)
    2. Exponential fragmentation function (more soft particles)
    3. Realistic hadron composition (mostly pions)

    Args:
        origin: Collision vertex position
        direction: Initial parton (quark/gluon) direction
        energy: Total jet energy in GeV
        rng: Random number generator
        n_particles: Number of hadrons in jet

    Returns:
        List of Particle objects representing jet constituents
    """
    particles = []

    # Normalize jet axis direction
    jet_dir = direction / (np.linalg.norm(direction) + 1e-10)

    # JET CONE ANGLE
    # --------------
    # Higher energy jets are more collimated due to Lorentz boost.
    # Typical LHC jets: R ≈ 0.4 for 100 GeV, narrower for TeV jets.
    # Formula approximates this behavior.
    cone_angle = 0.3 / (1 + energy / 100)

    # Create orthonormal basis for distributing particles within cone
    # perp1 and perp2 are perpendicular to jet axis
    if abs(jet_dir[2]) < 0.9:
        perp1 = np.cross(jet_dir, [0, 0, 1])
    else:
        perp1 = np.cross(jet_dir, [1, 0, 0])
    perp1 = perp1 / np.linalg.norm(perp1)
    perp2 = np.cross(jet_dir, perp1)

    # FRAGMENTATION FUNCTION D(z)
    # ---------------------------
    # The fraction z = E_hadron/E_jet follows an exponential-like distribution.
    # This means many soft particles and few hard ones.
    # Real fragmentation functions are measured experimentally (e.g., at LEP).
    z_fractions = rng.exponential(0.3, n_particles)
    z_fractions = z_fractions / z_fractions.sum()

    # HADRON COMPOSITION
    # ------------------
    # Pions dominate because they're the lightest mesons (easiest to produce).
    # Kaons require strange quark production (suppressed).
    # Baryons (protons) require diquark production (even more suppressed).
    hadron_types = ["pion_plus", "pion_minus", "kaon", "proton"]
    hadron_weights = [0.35, 0.35, 0.15, 0.15]  # Realistic composition

    for i in range(n_particles):
        # Random angle within jet cone
        theta = rng.exponential(cone_angle)
        phi = rng.uniform(0, 2 * np.pi)

        # Particle direction
        p_dir = (
            jet_dir * np.cos(theta)
            + perp1 * np.sin(theta) * np.cos(phi)
            + perp2 * np.sin(theta) * np.sin(phi)
        )

        # Particle energy and momentum
        p_energy = energy * z_fractions[i]
        p_momentum = p_energy  # Relativistic approximation

        px, py, pz = p_dir * p_momentum

        # Random hadron type
        p_type = rng.choice(hadron_types, p=hadron_weights)

        particles.append(
            Particle(
                particle_type=p_type,
                energy=p_energy,
                px=px,
                py=py,
                pz=pz,
                origin=origin.copy(),
            )
        )

    return particles


# =============================================================================
# Calorimeter Energy Deposits
# =============================================================================
#
# CALORIMETRY: MEASURING PARTICLE ENERGY
# --------------------------------------
# Calorimeters are dense absorbers that stop particles and measure their
# total energy. The particle creates a "shower" of secondary particles,
# and the calorimeter samples this shower.
#
# ELECTROMAGNETIC CALORIMETER (ECAL)
# ----------------------------------
# Stops electrons, positrons, and photons via:
# - Bremsstrahlung: e → e + γ (electron radiates photon)
# - Pair production: γ → e⁺ + e⁻ (photon converts to e⁺e⁻)
#
# This alternates: e → γ → e⁺e⁻ → γγ → ... creating an electromagnetic cascade.
# Shower depth: ~20-25 radiation lengths (X₀)
# ATLAS ECAL: Lead absorber + liquid argon sampling
# CMS ECAL: Lead tungstate (PbWO₄) crystals
#
# HADRONIC CALORIMETER (HCAL)
# ---------------------------
# Stops hadrons (pions, kaons, protons, neutrons) via strong interactions:
# - Nuclear spallation: hadron + nucleus → many hadrons
# - Includes both EM component (π⁰ → γγ) and hadronic component
#
# Hadronic showers are:
# - Larger than EM showers (more material needed)
# - More irregular (nuclear interactions are stochastic)
# - Shower depth: ~10 interaction lengths (λ)
#
# MUONS: THE PENETRATING PARTICLES
# --------------------------------
# Muons are minimum ionizing particles (MIPs). They lose energy slowly
# via ionization but don't shower. They pass through ALL calorimeters
# and are detected in the outermost muon chambers.


def generate_calorimeter_deposits(
    particles: list[Particle],
    rng: np.random.Generator,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Generate energy deposit points representing calorimeter showers.

    PHYSICS:
    --------
    When particles enter calorimeters, they create showers of secondary
    particles. We visualize these as clusters of points:

    - ECAL deposits: Tight clusters from e±/γ (EM showers)
    - HCAL deposits: Broader clusters from hadrons (nuclear showers)
    - Muons: No deposits (they pass through!)

    The shower size scales with log(energy) due to the multiplicative
    nature of the cascade process.

    Returns:
        positions: (N, 3) array of deposit positions
        colors: (N, 3) array of RGB colors (matches particle type)
        radii: (N,) array of point sizes (energy visualization)
        sharpness: (N,) array of point sharpness (soft, glowing appearance)
    """
    positions = []
    colors = []
    radii = []
    sharpness = []

    for particle in particles:
        if particle.charge == 0 and particle.particle_type == "photon":
            # Photons deposit in ECAL
            deposit_radius = ECAL_INNER + rng.uniform(0.2, 1.0)
        elif particle.stops_at == ECAL_OUTER:
            # Electrons shower in ECAL
            deposit_radius = ECAL_INNER + rng.uniform(0.3, 1.2)
        elif particle.stops_at == HCAL_OUTER:
            # Hadrons deposit in HCAL
            deposit_radius = HCAL_INNER + rng.uniform(0.5, 2.5)
        else:
            continue  # Muons don't deposit much

        # Calculate deposit position along particle trajectory
        direction = np.array([particle.px, particle.py, particle.pz])
        direction = direction / (np.linalg.norm(direction) + 1e-10)

        # Find intersection with calorimeter
        phi = np.arctan2(particle.py, particle.px)
        deposit_x = deposit_radius * np.cos(phi)
        deposit_y = deposit_radius * np.sin(phi)
        deposit_z = particle.origin[2] + direction[2] * deposit_radius * 0.8

        # Create cluster of deposit points (shower)
        n_shower = int(3 + particle.energy / 10)
        shower_spread = 0.3 + particle.energy / 200

        for _ in range(n_shower):
            offset = rng.normal(0, shower_spread, 3)
            pos = np.array([deposit_x, deposit_y, deposit_z]) + offset

            positions.append(pos)
            colors.append(particle.color)

            # Size proportional to energy
            size = 0.1 + 0.05 * np.log1p(particle.energy)
            radii.append(size * rng.uniform(0.7, 1.3))

            # Soft, glowing appearance (normalized [0, 1] knob; low = peakier/softer)
            sharpness.append(rng.uniform(0.25, 0.4))

    if not positions:
        return (
            np.array([]).reshape(0, 3),
            np.array([]).reshape(0, 3),
            np.array([]),
            np.array([]),
        )

    return (
        np.array(positions, dtype=np.float32),
        np.array(colors, dtype=np.float32),
        np.array(radii, dtype=np.float32),
        np.array(sharpness, dtype=np.float32),
    )


# =============================================================================
# Detector Geometry
# =============================================================================


def generate_detector_geometry(
    rng: np.random.Generator,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Generate detector outline as subtle reference lines.

    Creates circular cross-sections of detector layers.
    """
    vertices = []
    widths = []
    colors = []
    sharpness_vals = []

    # Layer definitions: (radius, color, width, n_segments)
    layers = [
        (TRACKER_INNER, [0.15, 0.15, 0.2], 0.008, 64),
        (TRACKER_OUTER, [0.15, 0.15, 0.2], 0.008, 64),
        (ECAL_INNER, [0.1, 0.2, 0.15], 0.006, 48),
        (ECAL_OUTER, [0.1, 0.2, 0.15], 0.006, 48),
        (HCAL_INNER, [0.2, 0.15, 0.1], 0.006, 48),
        (HCAL_OUTER, [0.2, 0.15, 0.1], 0.006, 48),
        (MUON_INNER, [0.15, 0.1, 0.2], 0.005, 32),
        (MUON_OUTER, [0.15, 0.1, 0.2], 0.005, 32),
    ]

    # Create circles at z=0 and end caps
    z_positions = [0, -DETECTOR_LENGTH * 0.8, DETECTOR_LENGTH * 0.8]

    for radius, color, width, n_seg in layers:
        for z in z_positions:
            angles = np.linspace(0, 2 * np.pi, n_seg + 1)
            for i in range(n_seg):
                x1 = radius * np.cos(angles[i])
                y1 = radius * np.sin(angles[i])
                x2 = radius * np.cos(angles[i + 1])
                y2 = radius * np.sin(angles[i + 1])

                vertices.extend([[x1, y1, z], [x2, y2, z]])
                widths.extend([width, width])
                colors.extend([color, color])
                sharpness_vals.extend([0.5, 0.5])

    # Add longitudinal lines connecting layers
    n_long = 16
    for i in range(n_long):
        angle = 2 * np.pi * i / n_long
        for radius, color, width, _ in layers[::2]:  # Every other layer
            x = radius * np.cos(angle)
            y = radius * np.sin(angle)

            vertices.extend(
                [[x, y, -DETECTOR_LENGTH * 0.8], [x, y, DETECTOR_LENGTH * 0.8]]
            )
            widths.extend([width * 0.5, width * 0.5])
            colors.extend([color, color])
            sharpness_vals.extend([0.5, 0.5])

    return (
        np.array(vertices, dtype=np.float32),
        np.array(widths, dtype=np.float32),
        np.array(colors, dtype=np.float32),
        np.array(sharpness_vals, dtype=np.float32),
    )


# =============================================================================
# Collision Event Generation
# =============================================================================
#
# HIGH-ENERGY PROTON-PROTON COLLISIONS
# ------------------------------------
# At the LHC, protons collide at center-of-mass energy √s = 13.6 TeV.
# But protons are composite particles (made of quarks and gluons).
# The actual hard collision involves partons (quarks/gluons) carrying
# only a fraction of the proton momentum.
#
# TYPICAL EVENT TOPOLOGY
# ----------------------
# Most interesting events involve heavy particle production and decay:
#
# 1. tt̄ (top-antitop) production:
#    pp → tt̄ → (bW⁺)(b̄W⁻) → jets + leptons + missing energy
#    - 4-6 jets (2 b-jets + 2-4 from W decay)
#    - 0-2 leptons (from W → ℓν)
#    - Missing energy (neutrinos)
#
# 2. W/Z + jets:
#    pp → W/Z + jets
#    - 2-4 jets
#    - 1-2 leptons (from W/Z decay)
#
# 3. Higgs production:
#    pp → H → various decay modes
#    - H → γγ: two photons
#    - H → ZZ → 4ℓ: four leptons
#    - H → bb̄: two b-jets
#
# CONSERVATION LAWS
# -----------------
# - Momentum conservation: Σp = 0 (in center-of-mass frame)
#   Jets tend to be back-to-back in azimuth
# - Energy conservation: ΣE = √s
# - Charge conservation: ΣQ = 0 (protons are neutral overall)
# - Lepton number, baryon number conserved


def generate_collision_event(
    rng: np.random.Generator,
    n_jets: int = 4,
    n_leptons: int = 2,
) -> list[Particle]:
    """Generate a complete collision event mimicking LHC physics.

    PHYSICS:
    --------
    Simulates proton-proton collision producing:
    - Multiple jets (from hard-scattered quarks/gluons)
    - Isolated leptons (from W/Z boson decays)
    - Possibly photons (from π⁰ decay or direct production)

    Event topology inspired by tt̄ and W/Z + jets processes.

    The simulation includes:
    - Rough momentum conservation (back-to-back jets)
    - Realistic angular distributions (central + forward)
    - Beam spot smearing (proton bunches have finite size)

    Args:
        rng: Random number generator
        n_jets: Number of jets to generate
        n_leptons: Number of isolated leptons

    Returns:
        List of all Particle objects in the event
    """
    particles = []
    origin = np.array([0.0, 0.0, 0.0])

    # BEAM SPOT / INTERACTION POINT
    # -----------------------------
    # Real collisions don't happen at a perfect point. The proton bunches
    # have finite size: σ_x ≈ σ_y ≈ 15 μm, σ_z ≈ 5 cm at LHC.
    # We add small Gaussian smearing to simulate this.
    vertex = origin + rng.normal(0, 0.02, 3)

    # ==========================================================================
    # JET PRODUCTION
    # ==========================================================================
    # Jets come from hard-scattered partons. In 2→2 processes like qq̄ → qq̄,
    # momentum conservation means jets tend to be back-to-back in azimuth (φ).
    # The total "visible" energy is a fraction of √s (some goes to beam remnants).

    total_energy = 500 + rng.uniform(0, 500)  # GeV, typical hard scatter scale

    for i in range(n_jets):
        # ANGULAR DISTRIBUTION
        # --------------------
        # Pseudorapidity η = -ln(tan(θ/2)) is uniform for minimum-bias events.
        # cos(θ) from -0.95 to 0.95 covers |η| < 2.5 (typical tracker acceptance).
        # Forward jets (high |η|) are common but harder to measure precisely.
        theta = np.arccos(rng.uniform(-0.95, 0.95))
        phi = rng.uniform(0, 2 * np.pi)

        # MOMENTUM CONSERVATION (approximate)
        # -----------------------------------
        # In 2→2 scattering, jets are back-to-back (Δφ ≈ π).
        # We approximate this by alternating jets ~180° apart with some smearing.
        if i > 0 and i % 2 == 1:
            phi = (phi + np.pi + rng.normal(0, 0.3)) % (2 * np.pi)

        direction = np.array(
            [np.sin(theta) * np.cos(phi), np.sin(theta) * np.sin(phi), np.cos(theta)]
        )

        # JET ENERGY
        # ----------
        # Energy fluctuates around equal sharing, modified by parton distribution
        # functions (PDFs) and matrix element effects.
        jet_energy = total_energy / n_jets * rng.uniform(0.5, 1.5)

        # Generate hadrons within the jet
        # Multiplicity scales roughly with log(E): more energy → more particles
        jet_particles = generate_jet(
            vertex, direction, jet_energy, rng, n_particles=int(8 + jet_energy / 30)
        )
        particles.extend(jet_particles)

    # ==========================================================================
    # ISOLATED LEPTON PRODUCTION
    # ==========================================================================
    # Isolated leptons (not inside jets) typically come from W/Z boson decay:
    # - W⁺ → ℓ⁺ν (electron or muon + neutrino)
    # - Z → ℓ⁺ℓ⁻ (lepton-antilepton pair)
    #
    # "Isolated" means separated from jets by ΔR > 0.4 typically.
    # Lepton p_T spectrum peaks around M_W/2 ≈ 40 GeV for W decay.

    lepton_types = [
        ("electron", "positron"),  # First generation
        ("muon_minus", "muon_plus"),  # Second generation (tau rare, hard to ID)
    ]

    for i in range(n_leptons):
        # Leptons from W/Z tend to be central (|η| < 2.5)
        theta = np.arccos(rng.uniform(-0.8, 0.8))
        phi = rng.uniform(0, 2 * np.pi)

        direction = np.array(
            [np.sin(theta) * np.cos(phi), np.sin(theta) * np.sin(phi), np.cos(theta)]
        )

        # Typical W decay: lepton p_T ~ 20-80 GeV
        lepton_energy = rng.uniform(20, 100)
        px, py, pz = direction * lepton_energy

        # LEPTON FLAVOR
        # -------------
        # At LHC, electron and muon channels have similar acceptance.
        # Tau leptons decay quickly and are harder to reconstruct.
        pair = lepton_types[i % len(lepton_types)]
        l_type = pair[rng.integers(0, 2)]  # Random charge (particle or antiparticle)

        particles.append(
            Particle(
                particle_type=l_type,
                energy=lepton_energy,
                px=px,
                py=py,
                pz=pz,
                origin=vertex.copy(),
            )
        )

    # ==========================================================================
    # PHOTON PRODUCTION
    # ==========================================================================
    # Isolated photons can come from:
    # 1. Direct production: qg → qγ (prompt photons)
    # 2. Fragmentation: jet → γ + X
    # 3. Rare decays: H → γγ (Higgs discovery channel!)
    #
    # π⁰ → γγ decay produces many photons, but these are inside jets (not isolated).

    if rng.random() < 0.3:  # ~30% of events have an isolated photon
        theta = np.arccos(rng.uniform(-0.7, 0.7))
        phi = rng.uniform(0, 2 * np.pi)
        direction = np.array(
            [np.sin(theta) * np.cos(phi), np.sin(theta) * np.sin(phi), np.cos(theta)]
        )
        photon_energy = rng.uniform(10, 50)

        particles.append(
            Particle(
                particle_type="photon",
                energy=photon_energy,
                px=direction[0] * photon_energy,
                py=direction[1] * photon_energy,
                pz=direction[2] * photon_energy,
                origin=vertex.copy(),
            )
        )

    return particles


# =============================================================================
# Main Scene Generation
# =============================================================================


def generate_detector_scene(
    output_path: Path,
    n_events: int = 5,
    n_jets_per_event: int = 4,
) -> tuple[int, int]:
    """Generate complete particle detector visualization.

    Returns:
        (total_segments, total_points)
    """
    total_segments = 0
    total_points = 0

    with LuxarZarrCompiler(output_path) as compiler:
        dims = Dimensions(
            [
                Dimension("x", unit="m", display=True),
                Dimension("y", unit="m", display=True),
                Dimension("z", unit="m", display=True),
            ]
        )
        scene = compiler.create_scene(dimensions=dims)

        rng = np.random.default_rng(42)

        # Generate detector geometry
        with asection("Creating detector geometry"):
            det_verts, det_widths, det_colors, det_sharp = generate_detector_geometry(
                rng
            )
            scene.add_lines(
                "detector_geometry",
                vertices=det_verts,
                widths=det_widths,
                colors=det_colors,
                sharpness=det_sharp,
                line_type="segments",
                layer=True,
            )
            n_det = len(det_verts) // 2
            aprint(f"Detector geometry: {n_det:,} segments")
            total_segments += n_det

        # Collect all tracks and deposits across events
        all_track_verts = []
        all_track_widths = []
        all_track_colors = []
        all_deposit_pos = []
        all_deposit_colors = []
        all_deposit_radii = []
        all_deposit_sharp = []

        # Generate collision events
        with asection(f"Generating {n_events} collision events"):
            for event_idx in range(n_events):
                # Slightly offset each event for visual separation
                event_offset = np.array([0, 0, event_idx * 0.1])

                particles = generate_collision_event(
                    rng, n_jets=n_jets_per_event, n_leptons=rng.integers(1, 4)
                )

                aprint(f"  Event {event_idx + 1}: {len(particles)} particles")

                # Generate tracks for each particle
                for particle in particles:
                    particle.origin = particle.origin + event_offset

                    vertices, widths, colors = generate_helix_track(particle, rng)

                    if len(vertices) > 0:
                        all_track_verts.append(vertices)
                        all_track_widths.append(widths)
                        all_track_colors.append(colors)

                # Generate calorimeter deposits
                pos, colors, radii, sharp = generate_calorimeter_deposits(
                    particles, rng
                )
                if len(pos) > 0:
                    all_deposit_pos.append(pos)
                    all_deposit_colors.append(colors)
                    all_deposit_radii.append(radii)
                    all_deposit_sharp.append(sharp)

        # Write all tracks
        with asection("Writing particle tracks"):
            if all_track_verts:
                track_vertices = np.concatenate(all_track_verts, axis=0)
                track_widths = np.concatenate(all_track_widths, axis=0)
                track_colors = np.concatenate(all_track_colors, axis=0)

                scene.add_lines(
                    "particle_tracks",
                    vertices=track_vertices,
                    widths=track_widths,
                    colors=track_colors,
                    sharpness=0.5,
                    line_type="segments",
                    layer=True,
                )
                n_tracks = len(track_vertices) // 2
                aprint(f"Particle tracks: {n_tracks:,} segments")
                total_segments += n_tracks

        # Write calorimeter deposits
        with asection("Writing calorimeter deposits"):
            if all_deposit_pos:
                deposit_positions = np.concatenate(all_deposit_pos, axis=0)
                deposit_colors = np.concatenate(all_deposit_colors, axis=0)
                deposit_radii = np.concatenate(all_deposit_radii, axis=0)
                deposit_sharpness = np.concatenate(all_deposit_sharp, axis=0)

                scene.add_points(
                    "calorimeter_deposits",
                    positions=deposit_positions,
                    colors=deposit_colors,
                    radii=deposit_radii,
                    sharpness=deposit_sharpness,
                    layer=True,
                    intensity=0.125,
                )
                aprint(f"Calorimeter deposits: {len(deposit_positions):,} points")
                total_points += len(deposit_positions)

        # Add collision vertex markers
        with asection("Adding vertex markers"):
            n_vertex_points = n_events * 20
            vertex_pos = []
            vertex_colors = []
            vertex_radii = []

            for event_idx in range(n_events):
                center = np.array([0, 0, event_idx * 0.1])
                for _ in range(20):
                    offset = rng.normal(0, 0.03, 3)
                    vertex_pos.append(center + offset)
                    vertex_colors.append([1.0, 0.9, 0.5])  # Golden glow
                    vertex_radii.append(rng.uniform(0.02, 0.05))

            scene.add_points(
                "collision_vertices",
                positions=np.array(vertex_pos, dtype=np.float32),
                colors=np.array(vertex_colors, dtype=np.float32),
                radii=np.array(vertex_radii, dtype=np.float32),
                sharpness=0.8,
                layer=True,
                intensity=0.125,
            )
            aprint(f"Vertex markers: {n_vertex_points} points")
            total_points += n_vertex_points

        # --- Overlays ---
        # Title
        scene.add_text(
            "Particle Collision Detector",
            position=(0.02, 0.02),
            font_size=0.055,
            anchor="top-left",
            color="rgba(255,255,255,0.6)",
            blend_mode="difference",
        )

        # Particle type legend (bottom-left)
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

        # Physics note (bottom-right)
        scene.add_text(
            "Track curvature \u221d 1/momentum",
            position=(0.98, 0.97),
            font_size=0.015,
            anchor="bottom-right",
            color="rgba(200,200,200,0.5)",
        )

    return total_segments, total_points


# =============================================================================
# Main Entry Point
# =============================================================================


def main() -> None:
    """Main demo entry point."""
    n_events = 8
    n_jets = 4

    if len(sys.argv) > 1:
        for arg in sys.argv[1:]:
            if arg.startswith("--events="):
                n_events = int(arg.split("=")[1])
            elif arg.startswith("--jets="):
                n_jets = int(arg.split("=")[1])

    aprint("=" * 70)
    aprint("PARTICLE COLLISION DETECTOR VISUALIZATION")
    aprint("=" * 70)
    aprint("")
    aprint("Simulating high-energy particle collisions in a detector.")
    aprint("")
    aprint("  Physics Features:")
    aprint("    - Charged particles curve in magnetic field (Lorentz force)")
    aprint("    - Electrons/positrons: tight spirals, stop in EM calorimeter")
    aprint("    - Muons: gentle curves, traverse entire detector")
    aprint("    - Hadrons: medium curves, stop in hadronic calorimeter")
    aprint("    - Jets: collimated sprays from quark/gluon fragmentation")
    aprint("")
    aprint("  Detector Layers:")
    aprint("    - Inner tracker: precision position measurements")
    aprint("    - EM calorimeter: stops electrons and photons")
    aprint("    - Hadronic calorimeter: stops hadrons")
    aprint("    - Muon chambers: only muons reach here")
    aprint("")
    aprint("  Visualization:")
    aprint("    - Lines: particle trajectories with energy-dependent width")
    aprint("    - Points: calorimeter energy deposits (shower clusters)")
    aprint("    - Colors: particle type identification")
    aprint("")
    aprint(f"Events: {n_events} | Jets per event: {n_jets}")
    aprint("")

    # If --no-serve, use persistent directory; otherwise temp for auto-cleanup
    if "--no-serve" in sys.argv:
        output_path = get_demos_output_dir() / "collision.luxar.zarr"
        with asection("Generating detector scene"):
            total_segments, total_points = generate_detector_scene(
                output_path, n_events=n_events, n_jets_per_event=n_jets
            )
        aprint(f"Dataset generated at {output_path}")
        aprint(f"Total: {total_segments:,} line segments, {total_points:,} points")
        return

    # Use temporary directory for serving (auto-cleanup on exit)
    with tempfile.TemporaryDirectory(prefix="luxar_demo_collision_") as tmpdir:
        output_path = Path(tmpdir) / "collision.luxar.zarr"

        with asection("Generating detector scene"):
            total_segments, total_points = generate_detector_scene(
                output_path, n_events=n_events, n_jets_per_event=n_jets
            )

        aprint("")
        aprint("=" * 70)
        aprint(
            f"SCENE COMPLETE: {total_segments:,} line segments, {total_points:,} points"
        )
        aprint("=" * 70)
        aprint("")
        aprint("Particle colors:")
        aprint("  - Blue: electrons")
        aprint("  - Red: positrons")
        aprint("  - Green: muons (negative)")
        aprint("  - Yellow: muons (positive)")
        aprint("  - Orange/Brown: pions")
        aprint("  - Pink: kaons")
        aprint("  - Purple: protons")
        aprint("  - White/cream: photons")
        aprint("")
        aprint("=" * 70)
        aprint("LAUNCHING VIEWER")
        aprint("=" * 70)
        aprint("Browser will open automatically. Press Ctrl+C when done.")
        aprint("")

        launch_viewer(output_path)

    aprint("Cleanup complete - temporary files removed")


if __name__ == "__main__":
    main()
