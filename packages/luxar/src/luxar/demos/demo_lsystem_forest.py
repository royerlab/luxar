#!/usr/bin/env python3
"""Self-Contained Demo: L-System Tree Forest

This demo demonstrates the Lines node type with beautiful procedural trees
generated using L-system grammars - formal languages that create organic
branching structures through simple rules.

Showcases:
- The new Lines node type with thousands of segments
- Width tapering from trunk to twigs (fractal realism)
- Color gradients (bark to foliage)
- 3D branching in all directions
- Multiple tree varieties and seasons
- Per-vertex attributes (width, color, sharpness)

L-System Grammar:
    F: Move forward, drawing a segment
    +/-: Rotate around vertical axis (yaw)
    ^/&: Rotate around lateral axis (pitch)
    [/]: Push/pop state (branching point)

Mathematical Background:
    L-systems were invented by botanist Aristid Lindenmayer in 1968 to model
    plant development. A string is iteratively expanded using production rules,
    then interpreted as 3D turtle graphics. The recursive structure naturally
    creates realistic branching patterns.

Usage:
    python demo_lsystem_forest.py [--iterations=N] [--trees=N]

Controls:
    - Ctrl+C to stop and cleanup
    - Browser opens automatically
"""

from __future__ import annotations

import sys
import tempfile
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional

import numpy as np
from arbol import aprint, asection

from luxar import Dimension, Dimensions, LuxarZarrCompiler
from luxar.demos import launch_viewer
from luxar.utils.paths import get_demos_output_dir

# =============================================================================
# Turtle Graphics for L-System Interpretation
# =============================================================================


@dataclass
class TurtleState:
    """3D turtle state for L-system interpretation."""

    position: np.ndarray = field(default_factory=lambda: np.array([0.0, 0.0, 0.0]))
    heading: np.ndarray = field(default_factory=lambda: np.array([0.0, 0.0, 1.0]))
    left: np.ndarray = field(default_factory=lambda: np.array([0.0, 1.0, 0.0]))
    up: np.ndarray = field(default_factory=lambda: np.array([1.0, 0.0, 0.0]))
    width: float = 1.0
    depth: int = 0

    def copy(self) -> TurtleState:
        """Create a deep copy of the state."""
        return TurtleState(
            position=self.position.copy(),
            heading=self.heading.copy(),
            left=self.left.copy(),
            up=self.up.copy(),
            width=self.width,
            depth=self.depth,
        )


def rotation_matrix(axis: np.ndarray, angle: float) -> np.ndarray:
    """Create rotation matrix around arbitrary axis using Rodrigues' formula."""
    axis = axis / np.linalg.norm(axis)
    K = np.array(
        [[0, -axis[2], axis[1]], [axis[2], 0, -axis[0]], [-axis[1], axis[0], 0]]
    )
    return np.eye(3) + np.sin(angle) * K + (1 - np.cos(angle)) * K @ K


# =============================================================================
# L-System Grammar and Expansion
# =============================================================================


@dataclass
class LSystem:
    """L-System grammar with expansion rules."""

    axiom: str
    rules: dict[str, str]
    angle: float = np.radians(25)
    length: float = 1.0
    width: float = 0.15
    width_decay: float = 0.7
    length_decay: float = 0.9
    randomness: float = 0.15

    def expand(self, iterations: int) -> str:
        """Expand the L-system string for given iterations."""
        current = self.axiom
        for _ in range(iterations):
            next_str = ""
            for char in current:
                next_str += self.rules.get(char, char)
            current = next_str
        return current

    def interpret(
        self, string: str, rng: Optional[np.random.Generator] = None
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        """Interpret L-system string and generate segments.

        Returns:
            Tuple of (start_positions, end_positions, depths) for each segment
        """
        if rng is None:
            rng = np.random.default_rng(42)

        state = TurtleState(width=self.width)
        stack: list[TurtleState] = []
        segments_start: list[np.ndarray] = []
        segments_end: list[np.ndarray] = []
        depths: list[int] = []
        current_length = self.length

        for char in string:
            angle_var = 1.0 + (rng.random() - 0.5) * 2 * self.randomness

            if char == "F":
                start = state.position.copy()
                state.position = state.position + state.heading * current_length
                segments_start.append(start)
                segments_end.append(state.position.copy())
                depths.append(state.depth)

            elif char == "+":
                angle = self.angle * angle_var
                R = rotation_matrix(state.up, angle)
                state.heading = R @ state.heading
                state.left = R @ state.left

            elif char == "-":
                angle = self.angle * angle_var
                R = rotation_matrix(state.up, -angle)
                state.heading = R @ state.heading
                state.left = R @ state.left

            elif char == "^":
                angle = self.angle * angle_var
                R = rotation_matrix(state.left, angle)
                state.heading = R @ state.heading
                state.up = R @ state.up

            elif char == "&":
                angle = self.angle * angle_var
                R = rotation_matrix(state.left, -angle)
                state.heading = R @ state.heading
                state.up = R @ state.up

            elif char == "\\":
                angle = self.angle * angle_var
                R = rotation_matrix(state.heading, angle)
                state.left = R @ state.left
                state.up = R @ state.up

            elif char == "/":
                angle = self.angle * angle_var
                R = rotation_matrix(state.heading, -angle)
                state.left = R @ state.left
                state.up = R @ state.up

            elif char == "[":
                stack.append(state.copy())
                state.depth += 1
                state.width *= self.width_decay
                current_length *= self.length_decay

            elif char == "]":
                if stack:
                    state = stack.pop()
                    current_length = self.length * (self.length_decay**state.depth)

        if not segments_start:
            return np.array([]).reshape(0, 3), np.array([]).reshape(0, 3), np.array([])

        return (
            np.array(segments_start, dtype=np.float32),
            np.array(segments_end, dtype=np.float32),
            np.array(depths, dtype=np.int32),
        )


# =============================================================================
# Tree Generation with Colors and Attributes
# =============================================================================


def create_tree_colors(
    depths: np.ndarray,
    max_depth: int,
    color_scheme: str = "autumn",
) -> np.ndarray:
    """Create color gradients based on branch depth."""
    n = len(depths)
    t = depths / max(max_depth, 1)

    color_palettes = {
        "autumn": [
            (0.0, np.array([0.35, 0.22, 0.12])),  # Bark
            (0.3, np.array([0.45, 0.30, 0.15])),  # Branch
            (0.5, np.array([0.9, 0.5, 0.1])),  # Orange
            (0.7, np.array([0.95, 0.3, 0.1])),  # Red
            (1.0, np.array([0.95, 0.85, 0.2])),  # Yellow
        ],
        "spring": [
            (0.0, np.array([0.35, 0.22, 0.12])),
            (0.3, np.array([0.4, 0.35, 0.2])),
            (0.5, np.array([0.5, 0.7, 0.3])),
            (0.7, np.array([0.7, 0.85, 0.5])),
            (1.0, np.array([0.95, 0.7, 0.85])),  # Pink blossom
        ],
        "winter": [
            (0.0, np.array([0.25, 0.18, 0.12])),
            (0.3, np.array([0.4, 0.35, 0.3])),
            (0.5, np.array([0.6, 0.6, 0.65])),
            (0.7, np.array([0.8, 0.82, 0.85])),
            (1.0, np.array([0.95, 0.97, 1.0])),  # Frost
        ],
        "summer": [
            (0.0, np.array([0.3, 0.2, 0.1])),
            (0.3, np.array([0.35, 0.3, 0.15])),
            (0.5, np.array([0.2, 0.5, 0.15])),
            (0.7, np.array([0.3, 0.65, 0.2])),
            (1.0, np.array([0.4, 0.75, 0.3])),
        ],
        "cherry": [
            (0.0, np.array([0.25, 0.15, 0.1])),
            (0.3, np.array([0.35, 0.2, 0.15])),
            (0.5, np.array([0.6, 0.3, 0.35])),
            (0.7, np.array([0.9, 0.5, 0.6])),
            (1.0, np.array([1.0, 0.75, 0.85])),  # Cherry blossom
        ],
    }

    palette = color_palettes.get(color_scheme, color_palettes["autumn"])
    colors = np.zeros((n, 3), dtype=np.float32)

    for i in range(n):
        depth_t = t[i]
        # Find surrounding palette entries
        for j in range(len(palette) - 1):
            t0, c0 = palette[j]
            t1, c1 = palette[j + 1]
            if t0 <= depth_t <= t1:
                blend = (depth_t - t0) / (t1 - t0) if t1 > t0 else 0
                colors[i] = c0 * (1 - blend) + c1 * blend
                break
        else:
            colors[i] = palette[-1][1]

    return colors


def create_tree_widths(
    depths: np.ndarray,
    max_depth: int,
    base_width: float = 0.15,
    min_width: float = 0.008,
) -> np.ndarray:
    """Create width arrays that taper with depth."""
    t = depths / max(max_depth, 1)
    decay = np.exp(-t * 3)
    widths = min_width + (base_width - min_width) * decay
    return widths.astype(np.float32)


def create_tree(
    lsystem: LSystem,
    iterations: int,
    position: tuple[float, float, float] = (0, 0, 0),
    scale: float = 1.0,
    rotation: float = 0.0,
    color_scheme: str = "autumn",
    seed: int = 42,
    add_leaves: bool = True,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, Optional[dict]]:
    """Create a single tree with all attributes.

    Returns:
        Tuple of (vertices, widths, colors, sharpness, leaves_dict) for segments line type
        leaves_dict contains: positions, colors, radii, sharpness for leaf points (or None)
    """
    rng = np.random.default_rng(seed)
    string = lsystem.expand(iterations)
    starts, ends, depths = lsystem.interpret(string, rng)

    if len(starts) == 0:
        empty = np.array([], dtype=np.float32).reshape(0, 3)
        return empty, np.array([]), empty, np.array([]), None

    max_depth = depths.max() if len(depths) > 0 else 1

    # Colors and widths - use same values at both ends for continuity at joints
    # The depth-based taper (create_tree_widths) handles trunk-to-twig transition
    start_colors = create_tree_colors(depths, max_depth, color_scheme)
    end_colors = start_colors  # Same color at joints for continuity
    start_widths = create_tree_widths(
        depths, max_depth, lsystem.width * scale, 0.005 * scale
    )
    end_widths = start_widths  # Same width at joints for continuity

    # Apply transform
    if rotation != 0:
        R = rotation_matrix(np.array([0, 0, 1]), rotation)
        starts = (R @ starts.T).T
        ends = (R @ ends.T).T

    starts = starts * scale + np.array(position)
    ends = ends * scale + np.array(position)

    # Find branch tips (deepest segments) for leaves
    leaves_dict = None
    if add_leaves and max_depth >= 2:
        # Get positions at the tips (high depth segments)
        tip_threshold = max_depth * 0.6  # Top 40% of depth
        tip_mask = depths >= tip_threshold
        tip_ends = ends[tip_mask]

        if len(tip_ends) > 0:
            # Sample some tip positions for leaves (not too many)
            n_leaves = min(len(tip_ends), int(len(tip_ends) * 0.7))
            leaf_indices = rng.choice(len(tip_ends), size=n_leaves, replace=False)
            leaf_positions = tip_ends[leaf_indices]

            # Add slight random offset to each leaf
            leaf_positions = (
                leaf_positions
                + rng.uniform(-0.1, 0.1, size=leaf_positions.shape) * scale
            )

            # Create leaf colors based on the tree's color scheme
            leaf_colors = create_tree_colors(
                np.full(n_leaves, max_depth), max_depth, color_scheme
            )
            # Add brightness variation to leaves
            brightness = rng.uniform(0.8, 1.3, size=(n_leaves, 1))
            leaf_colors = np.clip(leaf_colors * brightness, 0.0, 1.0).astype(np.float32)

            # Leaf radii: small fluffy balls
            leaf_radii = (
                rng.uniform(0.08, 0.18, size=n_leaves).astype(np.float32) * scale
            )

            # Leaf sharpness: very soft/fluffy (low values = peakier/softer edges)
            leaf_sharpness = rng.uniform(0.2, 0.35, size=n_leaves).astype(np.float32)

            leaves_dict = {
                "positions": leaf_positions.astype(np.float32),
                "colors": leaf_colors,
                "radii": leaf_radii,
                "sharpness": leaf_sharpness,
            }

    # Interleave for segments line type
    n = len(starts)
    vertices = np.zeros((n * 2, 3), dtype=np.float32)
    vertices[0::2] = starts
    vertices[1::2] = ends

    widths = np.zeros(n * 2, dtype=np.float32)
    widths[0::2] = start_widths
    widths[1::2] = end_widths

    colors = np.zeros((n * 2, 3), dtype=np.float32)
    colors[0::2] = start_colors
    colors[1::2] = end_colors

    # Sharpness: crisp trunk, softer tips (normalized knob, valid range: 0.0 to 1.0)
    t = depths / max(max_depth, 1)
    # Trunk: 0.65, tips: 0.3 - nice gradient from sharp to soft
    sharpness_base = 0.65 - t * 0.35
    sharpness = np.zeros(n * 2, dtype=np.float32)
    sharpness[0::2] = sharpness_base
    sharpness[1::2] = np.maximum(sharpness_base * 0.95, 0.25)

    return vertices, widths, colors, sharpness, leaves_dict


# =============================================================================
# Predefined Tree Types (base parameters - will be varied per instance)
# =============================================================================

TREE_RULES = {
    "elegant": LSystem(
        axiom="X",
        rules={"X": "F[+X][-X][^X][&X]FX", "F": "FF"},
        angle=np.radians(25),
        length=0.4,
        width=0.12,
        width_decay=0.68,
        length_decay=0.75,
        randomness=0.45,  # High randomness for organic look
    ),
    "fractal": LSystem(
        axiom="FA",
        rules={"A": "[+FA][-FA][^FA][&FA]"},
        angle=np.radians(30),
        length=0.8,
        width=0.15,
        width_decay=0.65,
        length_decay=0.7,
        randomness=0.4,  # Was 0.1 - way too uniform
    ),
    "willow": LSystem(
        axiom="FFFFA",
        rules={"A": "[&&&'B][&&&''B][&&&'''B]FA", "B": "&&F[-F]BF[+F]"},
        angle=np.radians(18),
        length=0.5,
        width=0.08,
        width_decay=0.78,
        length_decay=0.88,
        randomness=0.5,  # Extra random for droopy organic willow
    ),
    "bush": LSystem(
        axiom="FA",
        rules={"A": "[++++FA][----FA][^^^^FA][&&&&FA]FA"},
        angle=np.radians(12),
        length=0.35,
        width=0.05,
        width_decay=0.8,
        length_decay=0.85,
        randomness=0.55,  # Bushes are messy
    ),
}


def vary_lsystem(
    base: LSystem, rng: np.random.Generator, variation: float = 0.25
) -> LSystem:
    """Create a varied copy of an L-system with different parameters.

    This creates 'family resemblance' - trees of the same type look similar but not identical.
    Each tree gets its own unique angle, length, and randomness variations.
    """
    return LSystem(
        axiom=base.axiom,
        rules=base.rules,
        # Angle varies ±25% - gives different spread patterns
        angle=base.angle * (1.0 + rng.uniform(-variation, variation)),
        # Length varies ±25% - affects tree height/density
        length=base.length * (1.0 + rng.uniform(-variation, variation)),
        # Width varies less to maintain proportions
        width=base.width * (1.0 + rng.uniform(-variation * 0.4, variation * 0.4)),
        width_decay=np.clip(base.width_decay + rng.uniform(-0.1, 0.1), 0.5, 0.9),
        length_decay=np.clip(base.length_decay + rng.uniform(-0.1, 0.1), 0.55, 0.95),
        # Randomness varies significantly - some trees more chaotic than others
        randomness=np.clip(base.randomness + rng.uniform(-0.15, 0.2), 0.3, 0.7),
    )


def poisson_disk_sampling(
    rng: np.random.Generator,
    width: float,
    height: float,
    min_dist: float,
    max_points: int,
    k: int = 30,
) -> list[tuple[float, float]]:
    """Bridson's algorithm for Poisson disk sampling.

    Generates points with guaranteed minimum distance between them,
    creating natural-looking distributions like forests.

    Args:
        rng: Random number generator
        width: Area width
        height: Area height
        min_dist: Minimum distance between points
        max_points: Maximum number of points to generate
        k: Number of candidates to try before rejecting (default 30)

    Returns:
        List of (x, y) positions
    """
    cell_size = min_dist / np.sqrt(2)
    grid_width = int(np.ceil(width / cell_size))
    grid_height = int(np.ceil(height / cell_size))

    # Grid stores index into points list, -1 means empty
    grid = np.full((grid_width, grid_height), -1, dtype=np.int32)
    points: list[tuple[float, float]] = []
    active: list[int] = []

    def grid_coords(x: float, y: float) -> tuple[int, int]:
        return int(x / cell_size), int(y / cell_size)

    def is_valid(x: float, y: float) -> bool:
        if x < 0 or x >= width or y < 0 or y >= height:
            return False
        gx, gy = grid_coords(x, y)
        # Check neighboring cells
        for dx in range(-2, 3):
            for dy in range(-2, 3):
                nx, ny = gx + dx, gy + dy
                if 0 <= nx < grid_width and 0 <= ny < grid_height:
                    idx = grid[nx, ny]
                    if idx >= 0:
                        px, py = points[idx]
                        if (x - px) ** 2 + (y - py) ** 2 < min_dist**2:
                            return False
        return True

    # Start with random point
    x0 = rng.uniform(0, width)
    y0 = rng.uniform(0, height)
    points.append((x0, y0))
    gx, gy = grid_coords(x0, y0)
    grid[gx, gy] = 0
    active.append(0)

    while active and len(points) < max_points:
        # Pick random active point
        idx = rng.integers(0, len(active))
        px, py = points[active[idx]]

        found = False
        for _ in range(k):
            # Generate random point in annulus [min_dist, 2*min_dist]
            angle = rng.uniform(0, 2 * np.pi)
            r = rng.uniform(min_dist, 2 * min_dist)
            x = px + r * np.cos(angle)
            y = py + r * np.sin(angle)

            if is_valid(x, y):
                new_idx = len(points)
                points.append((x, y))
                gx, gy = grid_coords(x, y)
                grid[gx, gy] = new_idx
                active.append(new_idx)
                found = True
                break

        if not found:
            active.pop(idx)

    return points


# =============================================================================
# Forest Generation
# =============================================================================


def generate_forest(
    output_path: Path,
    iterations: int = 5,
    n_trees: int = 800,
) -> int:
    """Generate a complete forest scene.

    Returns:
        Total number of line segments
    """
    total_segments = 0

    with LuxarZarrCompiler(output_path) as compiler:
        dims = Dimensions(
            [
                Dimension("x", unit="m", display=True),
                Dimension("y", unit="m", display=True),
                Dimension("z", unit="m", display=True),
            ]
        )
        scene = compiler.create_scene(dimensions=dims)

        # Ground grid - same 70x70m area
        with asection("Creating ground plane"):
            grid_size = 70.0
            n_lines = 35
            grid_vertices = []
            for i in range(n_lines + 1):
                pos = -grid_size / 2 + i * grid_size / n_lines
                grid_vertices.append([-grid_size / 2, pos, 0])
                grid_vertices.append([grid_size / 2, pos, 0])
                grid_vertices.append([pos, -grid_size / 2, 0])
                grid_vertices.append([pos, grid_size / 2, 0])

            grid_vertices = np.array(grid_vertices, dtype=np.float32)
            grid_colors = np.full(
                (len(grid_vertices), 3), [0.12, 0.2, 0.08], dtype=np.float32
            )

            scene.add_lines(
                "ground",
                vertices=grid_vertices,
                widths=0.02,
                colors=grid_colors,
                sharpness=0.8,
                line_type="segments",
            )
            n_grid = len(grid_vertices) // 2
            aprint(f"Ground: {n_grid} segments")
            total_segments += n_grid

        # Generate forest with Poisson disk sampling
        with asection(f"Creating {n_trees} trees with Poisson disk sampling"):
            rng = np.random.default_rng(123)

            # Color schemes
            base_schemes = ["autumn", "spring", "summer", "winter", "cherry"]
            tree_types = list(TREE_RULES.keys())

            # Poisson disk sampling for natural tree distribution
            # For 800 trees in 70x70m area, use ~2m minimum spacing
            forest_size = 70.0
            min_spacing = 2.2  # Allows ~800-1000 trees in the area

            aprint(f"  Running Poisson disk sampling (min spacing: {min_spacing}m)...")
            raw_positions = poisson_disk_sampling(
                rng=rng,
                width=forest_size,
                height=forest_size,
                min_dist=min_spacing,
                max_points=n_trees,
                k=30,
            )

            # Center positions around origin
            positions = [
                (x - forest_size / 2, y - forest_size / 2) for x, y in raw_positions
            ]
            aprint(f"  Placed {len(positions)} trees using Poisson disk sampling")

            # Collect all leaves for batch addition
            all_leaf_positions = []
            all_leaf_colors = []
            all_leaf_radii = []
            all_leaf_sharpness = []

            for i, (x, y) in enumerate(positions):
                # Consistent scale range with slight variation
                base_scale = 0.9
                scale = base_scale * (0.85 + 0.3 * rng.random())

                # Color scheme - assign based on position for some clustering
                # Trees near each other tend to be similar type/color
                noise = rng.random() * 0.3
                scheme_idx = int(
                    (x + forest_size / 2 + noise * 10)
                    / (forest_size / len(base_schemes))
                ) % len(base_schemes)
                scheme = base_schemes[scheme_idx]

                # Tree type with some spatial clustering too
                type_noise = rng.random() * 0.2
                type_idx = int(
                    (y + forest_size / 2 + type_noise * 10)
                    / (forest_size / len(tree_types))
                ) % len(tree_types)
                tree_type = tree_types[type_idx]

                # Create a varied version of the base L-system (family resemblance)
                varied_lsystem = vary_lsystem(
                    TREE_RULES[tree_type], rng, variation=0.18
                )

                rot = rng.uniform(0, 2 * np.pi)

                # Same iterations for all trees (consistent detail level)
                tree_iterations = iterations - 1

                vertices, widths, colors, sharpness, leaves_dict = create_tree(
                    varied_lsystem,
                    iterations=tree_iterations,
                    position=(x, y, 0),
                    scale=scale,
                    rotation=rot,
                    color_scheme=scheme,
                    seed=42 + i * 17,
                    add_leaves=True,
                )

                # Add subtle color variation per tree
                color_shift = rng.uniform(-0.06, 0.06, size=3).astype(np.float32)
                brightness = rng.uniform(0.88, 1.12)
                colors = np.clip(colors * brightness + color_shift, 0.0, 1.0)

                scene.add_lines(
                    f"tree_{i:04d}",
                    vertices=vertices,
                    widths=widths,
                    colors=colors,
                    sharpness=sharpness,
                    line_type="segments",
                )

                # Collect leaves
                if leaves_dict is not None:
                    # Apply same color variation to leaves
                    leaf_colors = np.clip(
                        leaves_dict["colors"] * brightness + color_shift, 0.0, 1.0
                    )
                    all_leaf_positions.append(leaves_dict["positions"])
                    all_leaf_colors.append(leaf_colors)
                    all_leaf_radii.append(leaves_dict["radii"])
                    all_leaf_sharpness.append(leaves_dict["sharpness"])

                n = len(vertices) // 2
                if i % 100 == 0:
                    aprint(
                        f"  Tree {i:04d}/{len(positions)} ({scheme} {tree_type}): {n:,} segments"
                    )
                total_segments += n

        # Add all leaves as a single points layer
        if all_leaf_positions:
            with asection("Creating foliage (fluffy leaf points)"):
                leaf_positions = np.concatenate(all_leaf_positions, axis=0)
                leaf_colors = np.concatenate(all_leaf_colors, axis=0)
                leaf_radii = np.concatenate(all_leaf_radii, axis=0)
                leaf_sharpness = np.concatenate(all_leaf_sharpness, axis=0)

                scene.add_points(
                    "foliage",
                    positions=leaf_positions,
                    colors=leaf_colors,
                    radii=leaf_radii,
                    sharpness=leaf_sharpness,
                    intensity=0.031,
                )
                aprint(f"  Foliage: {len(leaf_positions):,} fluffy leaf points")

        # --- Overlays ---
        # Title
        scene.add_text(
            "L-System Forest",
            position=(0.02, 0.02),
            font_size=0.055,
            anchor="top-left",
            color="rgba(255,255,255,0.6)",
            blend_mode="difference",
        )

        # Info
        scene.add_text(
            "Procedural L-systems",
            position=(0.98, 0.97),
            font_size=0.015,
            anchor="bottom-right",
            color="rgba(200,200,200,0.45)",
        )

    return total_segments


# =============================================================================
# Main Entry Point
# =============================================================================


def main() -> None:
    """Main demo entry point."""
    iterations = 5
    n_trees = 800

    if len(sys.argv) > 1:
        for arg in sys.argv[1:]:
            if arg.startswith("--iterations="):
                iterations = int(arg.split("=")[1])
            elif arg.startswith("--trees="):
                n_trees = int(arg.split("=")[1])

    aprint("=" * 70)
    aprint("L-SYSTEM TREE FOREST DEMO")
    aprint("=" * 70)
    aprint("")
    aprint("Generating a dense forest using L-system grammars.")
    aprint("This demo showcases both Lines and Points node types:")
    aprint("")
    aprint("  Lines (branches):")
    aprint("    - Hundreds of thousands of line segments")
    aprint("    - Width tapering: thick trunks -> thin twigs")
    aprint("    - Color gradients: bark -> leaf colors")
    aprint("")
    aprint("  Points (foliage):")
    aprint("    - Soft fluffy points at branch tips")
    aprint("    - Low sharpness for organic appearance")
    aprint("")
    aprint("  Tree Generation:")
    aprint("    - Poisson disk sampling for natural spacing")
    aprint("    - 4 tree families with per-instance variation")
    aprint("    - 5 color schemes with spatial clustering")
    aprint("")
    aprint(f"Iterations: {iterations} | Trees: {n_trees}")
    aprint("")

    # If --no-serve, use persistent directory; otherwise temp for auto-cleanup
    if "--no-serve" in sys.argv:
        output_path = get_demos_output_dir() / "forest.zarr"
        with asection("Generating forest"):
            generate_forest(output_path, iterations=iterations, n_trees=n_trees)
        aprint(f"Dataset generated at {output_path}")
        return

    # Use temporary directory for serving (auto-cleanup on exit)
    with tempfile.TemporaryDirectory(prefix="luxar_demo_forest_") as tmpdir:
        output_path = Path(tmpdir) / "forest.zarr"

        with asection("Generating forest"):
            total_segments = generate_forest(
                output_path, iterations=iterations, n_trees=n_trees
            )

        aprint("")
        aprint("=" * 70)
        aprint(f"FOREST COMPLETE: {total_segments:,} total line segments")
        aprint("=" * 70)
        aprint("")
        aprint("Tree varieties (randomly distributed):")
        aprint("  - elegant: Classic branching pattern")
        aprint("  - fractal: Mathematical recursive structure")
        aprint("  - willow: Drooping weeping branches")
        aprint("  - bush: Dense low growth")
        aprint("")
        aprint("Color schemes (with per-tree variation):")
        aprint("  - autumn: Brown -> Orange -> Red -> Yellow")
        aprint("  - spring: Brown -> Green -> Pink blossoms")
        aprint("  - summer: Brown -> Rich forest greens")
        aprint("  - winter: Dark bark -> Gray -> White frost")
        aprint("  - cherry: Brown -> Pink cherry blossoms")
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
