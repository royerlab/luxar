#!/usr/bin/env python3
"""Self-Contained Demo: Nuclear Pore Complex with Perfect 8-Fold Symmetry

This demo demonstrates:
- Downloading real Nup107-160 subcomplex structure from PDB
- Creating C-alpha backbone trace for clean visualization
- Applying PERFECT 8-fold rotational symmetry
- Color-coding each spoke for beautiful symmetry display
- Van der Waals radii for realistic atomic sizes

Visualization approach:
- C-alpha trace (backbone only) - reduces visual clutter
- Each of 8 spokes gets a distinct color
- Shows the beautiful octagonal architecture clearly
- Central pore is visible!

================================================================================
NUCLEAR PORE COMPLEX: THE GATEWAY TO THE NUCLEUS
================================================================================

The Nuclear Pore Complex (NPC) is one of the largest and most important protein
complexes in eukaryotic cells. It serves as the sole gateway between the
nucleus and cytoplasm, regulating all molecular traffic in and out.

SCALE AND COMPLEXITY:
- ~1,000 protein molecules per NPC
- ~125 MDa (megadaltons) molecular weight
- ~120 nm diameter, ~75 nm height
- ~30 different proteins (nucleoporins or "Nups")
- **PERFECT 8-fold rotational symmetry** (octagonal structure)

KEY STRUCTURAL COMPONENTS:
1. **Nup107-160 Complex (Y-Complex)**:
   - Forms the structural scaffold
   - Y-shaped architecture visible in side view
   - 10 proteins per complex
   - 8 copies form the octagonal ring

2. **Central Channel/Pore**:
   - ~40 nm diameter passage
   - Visible as the hole in the center
   - Allows molecular transport

VISUALIZATION STRATEGY:
- Download ONE Nup107-160 Y-complex (PDB: 3I4R)
- Extract C-alpha atoms only (protein backbone trace)
- Apply 8-fold rotational symmetry
- Color each spoke differently to show symmetry
- Position at correct radius to create ring with central pore

This creates a clean, beautiful visualization similar to textbook illustrations!

REFERENCE:
Bui, K.H. et al. (2013)
"Integrated structural analysis of the human nuclear pore complex scaffold"
Cell 155(6): 1233-1243
DOI: 10.1016/j.cell.2013.10.055

Inspiration: https://pdb101.rcsb.org/motm/205

Usage:
    python demo_nuclear_pore_complex.py [--representation=TYPE]

    Representation types:
    - calpha: C-alpha trace (clean backbone view)
    - all: All atoms (default, dense, detailed)
    - backbone: Backbone atoms only (C, N, O, CA)

Controls:
    - Rotate to see PERFECT 8-fold symmetry
    - Top-down view: Beautiful octagonal ring with central pore!
    - Each spoke is a different color
    - Ctrl+C to stop and cleanup
"""

import sys
import tempfile
import urllib.request
from pathlib import Path

import numpy as np
from arbol import aprint, asection

from luxar import Dimension, Dimensions, LuxarZarrCompiler
from luxar.demos import launch_viewer
from luxar.utils.paths import get_demos_output_dir

# =============================================================================
# PDB File Parsing
# =============================================================================


def download_pdb(pdb_id: str, output_path: Path) -> Path:
    """Download PDB file from RCSB.

    Args:
        pdb_id: 4-character PDB ID
        output_path: Where to save the file

    Returns:
        Path to downloaded file
    """
    url = f"https://files.rcsb.org/download/{pdb_id}.pdb"
    aprint(f"Downloading {pdb_id} from RCSB PDB...")
    aprint(f"  URL: {url}")

    try:
        urllib.request.urlretrieve(url, output_path)
        size_mb = output_path.stat().st_size / (1024 * 1024)
        aprint(f"✓ Downloaded {size_mb:.1f} MB")
        return output_path
    except Exception as e:
        aprint(f"❌ Error downloading: {e}")
        raise


def parse_pdb_atoms(
    pdb_path: Path, atom_filter: str = "all", max_atoms: int = 100000
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Parse PDB file and extract atom coordinates.

    Args:
        pdb_path: Path to PDB file
        atom_filter: "calpha", "backbone", or "all"
        max_atoms: Maximum atoms to read

    Returns:
        Tuple of (positions, element_types, atom_names, is_backbone)
    """
    positions = []
    elements = []
    atom_names = []
    is_backbone_list = []

    # Element to index mapping
    element_map = {"C": 0, "N": 1, "O": 2, "S": 3, "P": 4}
    default_element = 0

    # Backbone atom names
    backbone_atoms = {"CA", "C", "N", "O"}

    with open(pdb_path, "r") as f:
        for line in f:
            if len(positions) >= max_atoms:
                break

            if line.startswith("ATOM  "):
                try:
                    atom_name = line[12:16].strip()

                    # Filter atoms based on type
                    if atom_filter == "calpha":
                        if atom_name != "CA":
                            continue
                    elif atom_filter == "backbone":
                        if atom_name not in backbone_atoms:
                            continue
                    # "all" - no filter

                    x = float(line[30:38].strip())
                    y = float(line[38:46].strip())
                    z = float(line[46:54].strip())

                    # Element symbol
                    element = line[76:78].strip()
                    if not element:
                        element = atom_name[0] if atom_name else "C"

                    element_idx = element_map.get(element, default_element)

                    # Is this a backbone atom?
                    is_backbone = atom_name in backbone_atoms

                    positions.append([x, y, z])
                    elements.append(element_idx)
                    atom_names.append(atom_name)
                    is_backbone_list.append(is_backbone)

                except (ValueError, IndexError):
                    continue

    if not positions:
        raise ValueError(f"No atoms found in {pdb_path}")

    return (
        np.array(positions, dtype=np.float32),
        np.array(elements, dtype=np.int32),
        np.array(atom_names),
        np.array(is_backbone_list, dtype=bool),
    )


# =============================================================================
# Geometric Transformations
# =============================================================================


def rotation_matrix_z(angle: float) -> np.ndarray:
    """Create 3D rotation matrix around Z axis.

    Args:
        angle: Rotation angle in radians

    Returns:
        3x3 rotation matrix
    """
    c, s = np.cos(angle), np.sin(angle)
    return np.array([[c, -s, 0], [s, c, 0], [0, 0, 1]], dtype=np.float32)


def apply_rotational_symmetry(
    positions: np.ndarray,
    elements: np.ndarray,
    is_backbone: np.ndarray,
    n_fold: int = 8,
    spoke_radius: float = 8.0,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Apply n-fold rotational symmetry to create a ring with central pore.

    Args:
        positions: (N, 3) array of coordinates (one spoke, centered)
        elements: (N,) array of element types
        is_backbone: (N,) array of backbone flags
        n_fold: Symmetry order (8 for NPC)
        spoke_radius: Distance from center to position each spoke

    Returns:
        Tuple of (all_positions, all_elements, all_is_backbone, spoke_ids)
    """
    all_positions = []
    all_elements = []
    all_is_backbone = []
    all_spoke_ids = []

    for i in range(n_fold):
        angle = 2 * np.pi * i / n_fold
        R = rotation_matrix_z(angle)

        # Rotate the spoke
        rotated = (R @ positions.T).T

        # Translate outward to create ring
        translation = np.array(
            [spoke_radius * np.cos(angle), spoke_radius * np.sin(angle), 0.0]
        )
        positioned = rotated + translation

        all_positions.append(positioned)
        all_elements.append(elements)
        all_is_backbone.append(is_backbone)
        all_spoke_ids.append(np.full(len(positions), i, dtype=np.int32))

    return (
        np.vstack(all_positions),
        np.concatenate(all_elements),
        np.concatenate(all_is_backbone),
        np.concatenate(all_spoke_ids),
    )


def center_structure(positions: np.ndarray) -> np.ndarray:
    """Center structure at origin.

    Args:
        positions: (N, 3) array of coordinates

    Returns:
        Centered positions
    """
    center = np.mean(positions, axis=0)
    return positions - center


# =============================================================================
# Color Schemes and Atomic Properties
# =============================================================================


def element_to_vdw_radius(elements: np.ndarray) -> np.ndarray:
    """Convert element types to van der Waals radii.

    Uses standard van der Waals radii in Angstroms.

    Args:
        elements: (N,) array of element indices

    Returns:
        (N,) radii in Angstroms
    """
    radii = np.zeros(len(elements), dtype=np.float32)

    # Van der Waals radii (Angstroms)
    vdw_radii = {
        0: 1.70,  # Carbon
        1: 1.55,  # Nitrogen
        2: 1.52,  # Oxygen
        3: 1.80,  # Sulfur
        4: 1.80,  # Phosphorus
    }

    for element_idx, radius in vdw_radii.items():
        mask = elements == element_idx
        radii[mask] = radius

    return radii


def spoke_to_color(spoke_ids: np.ndarray, n_spokes: int = 8) -> np.ndarray:
    """Convert spoke IDs to distinct rainbow colors.

    Each of the 8 spokes gets a different color to show symmetry clearly.

    Args:
        spoke_ids: (N,) array of spoke IDs (0-7)
        n_spokes: Total number of spokes

    Returns:
        (N, 3) RGB colors
    """
    colors = np.zeros((len(spoke_ids), 3), dtype=np.float32)

    # Rainbow gradient for spokes
    for i in range(n_spokes):
        hue = i / n_spokes
        h = hue * 6.0
        c = 1.0
        x = c * (1 - abs(h % 2 - 1))

        if h < 1:
            r, g, b = c, x, 0
        elif h < 2:
            r, g, b = x, c, 0
        elif h < 3:
            r, g, b = 0, c, x
        elif h < 4:
            r, g, b = 0, x, c
        elif h < 5:
            r, g, b = x, 0, c
        else:
            r, g, b = c, 0, x

        mask = spoke_ids == i
        colors[mask] = [r, g, b]

    return colors


def element_to_color(
    elements: np.ndarray, is_backbone: np.ndarray = None
) -> np.ndarray:
    """Convert element types to CPK colors.

    Args:
        elements: (N,) array of element indices
        is_backbone: (N,) optional boolean array for backbone atoms

    Returns:
        (N, 3) RGB colors
    """
    colors = np.zeros((len(elements), 3), dtype=np.float32)

    # Vibrant CPK coloring
    color_map = {
        0: [0.9, 0.9, 0.9],  # Carbon - bright white/gray
        1: [0.3, 0.5, 1.0],  # Nitrogen - blue
        2: [1.0, 0.3, 0.3],  # Oxygen - red
        3: [1.0, 0.9, 0.2],  # Sulfur - yellow
        4: [1.0, 0.5, 0.0],  # Phosphorus - orange
    }

    for element_idx, color in color_map.items():
        mask = elements == element_idx
        colors[mask] = color

    # Dim sidechain atoms if backbone info provided
    if is_backbone is not None:
        sidechain_mask = ~is_backbone
        colors[sidechain_mask] *= 0.6  # Dim sidechains relative to backbone

    return colors


def backbone_sidechain_color(
    elements: np.ndarray, is_backbone: np.ndarray
) -> np.ndarray:
    """Color atoms by backbone vs sidechain with element distinction.

    Args:
        elements: (N,) array of element indices
        is_backbone: (N,) boolean array

    Returns:
        (N, 3) RGB colors
    """
    colors = np.zeros((len(elements), 3), dtype=np.float32)

    # Backbone: Warm colors (red/orange/yellow)
    # Sidechain: Cool colors (blue/cyan/green)

    backbone_color_map = {
        0: [1.0, 0.8, 0.6],  # Carbon - warm tan
        1: [1.0, 0.6, 0.4],  # Nitrogen - orange
        2: [1.0, 0.4, 0.3],  # Oxygen - red-orange
        3: [1.0, 0.9, 0.3],  # Sulfur - yellow
        4: [1.0, 0.7, 0.2],  # Phosphorus - gold
    }

    sidechain_color_map = {
        0: [0.6, 0.8, 0.9],  # Carbon - light blue
        1: [0.3, 0.5, 1.0],  # Nitrogen - blue
        2: [0.3, 0.8, 0.7],  # Oxygen - cyan
        3: [0.4, 1.0, 0.4],  # Sulfur - green
        4: [0.5, 0.9, 0.6],  # Phosphorus - pale green
    }

    # Apply backbone colors
    for element_idx, color in backbone_color_map.items():
        mask = (elements == element_idx) & is_backbone
        colors[mask] = color

    # Apply sidechain colors
    for element_idx, color in sidechain_color_map.items():
        mask = (elements == element_idx) & (~is_backbone)
        colors[mask] = color

    return colors


# =============================================================================
# Main Generation
# =============================================================================


def generate_nuclear_pore_complex(
    output_path: Path,
    pdb_id: str = "3I4R",
    max_atoms: int = 100000,
    n_fold: int = 8,
    representation: str = "calpha",
    color_by: str = "spoke",
) -> int:
    """Generate Nuclear Pore Complex with perfect 8-fold symmetry.

    Args:
        output_path: Where to write zarr
        pdb_id: PDB ID (default: 3I4R - Nup107-160 subcomplex)
        max_atoms: Maximum atoms per spoke
        n_fold: Symmetry order (8 for NPC)
        representation: "calpha", "backbone", or "all"
        color_by: "spoke" or "element"

    Returns:
        Total number of atoms visualized
    """
    with tempfile.TemporaryDirectory(prefix="luxar_pdb_") as tmpdir:
        pdb_path = Path(tmpdir) / f"{pdb_id}.pdb"

        # Download structure
        with asection(f"Downloading PDB structure {pdb_id}"):
            aprint("Structure: Nup107-160 Y-complex (one spoke)")
            aprint(f"Will apply {n_fold}-fold rotational symmetry")
            aprint(f"Representation: {representation}")
            aprint("")
            download_pdb(pdb_id, pdb_path)

        # Parse structure
        with asection(f"Parsing PDB file ({representation} atoms)"):
            positions, elements, atom_names, is_backbone = parse_pdb_atoms(
                pdb_path, atom_filter=representation, max_atoms=max_atoms
            )
            n_backbone = np.sum(is_backbone)
            n_sidechain = len(positions) - n_backbone

            aprint(f"✓ Parsed {len(positions):,} atoms (one spoke)")
            aprint(f"  Backbone: {n_backbone:,} atoms")
            aprint(f"  Sidechain: {n_sidechain:,} atoms")

            if representation == "calpha":
                aprint("  C-alpha atoms only (backbone trace)")
            elif representation == "backbone":
                aprint("  Backbone atoms (CA, C, N, O)")
            else:
                aprint("  All atoms")

            aprint(
                f"  Coordinate range: {positions.min():.1f} to {positions.max():.1f} Å"
            )

        # Center and scale
        with asection("Preparing structure"):
            positions = center_structure(positions)

            # Scale to nm
            positions = positions * 0.1  # Å to nm

            aprint("✓ Centered and scaled to nm")
            aprint(f"  Range: {positions.min():.1f} to {positions.max():.1f} nm")

        # Apply 8-fold symmetry with proper ring geometry
        with asection(f"Applying {n_fold}-fold rotational symmetry"):
            # Position spokes at radius to create ring with central pore
            # Real NPC: ~60 nm radius, ~40 nm central pore
            # Scaled down for visualization
            spoke_radius = 8.0  # nm - creates visible central pore
            sym_positions, sym_elements, sym_is_backbone, spoke_ids = (
                apply_rotational_symmetry(
                    positions, elements, is_backbone, n_fold, spoke_radius=spoke_radius
                )
            )
            aprint(f"✓ Created {len(sym_positions):,} atoms ({n_fold} copies)")
            aprint(f"  Perfect C{n_fold} symmetry applied")
            aprint(f"  Spoke radius: {spoke_radius:.1f} nm")
            aprint(f"  Ring diameter: ~{spoke_radius * 2:.1f} nm")
            aprint("  Central pore: Visible in center!")

        # Generate colors and radii
        with asection(f"Generating colors and radii (color_by={color_by})"):
            if color_by == "spoke":
                colors = spoke_to_color(spoke_ids, n_spokes=n_fold)
                aprint("✓ Rainbow coloring by spoke:")
                aprint(f"  Each of {n_fold} spokes has distinct color")
                aprint("  Shows 8-fold symmetry clearly!")
            elif color_by == "element":
                colors = element_to_color(sym_elements)
                aprint("✓ CPK coloring by element:")
                aprint("  White: Carbon")
                aprint("  Blue: Nitrogen")
                aprint("  Red: Oxygen")
                aprint("  Yellow: Sulfur")
                aprint("  Orange: Phosphorus")
            elif color_by == "backbone":
                colors = backbone_sidechain_color(sym_elements, sym_is_backbone)
                aprint("✓ Backbone vs sidechain coloring:")
                aprint("  Warm (red/orange/yellow): Backbone atoms")
                aprint("  Cool (blue/cyan/green): Sidechain atoms")
            else:  # element-dimmed
                colors = element_to_color(sym_elements, sym_is_backbone)
                aprint("✓ CPK with dimmed sidechains:")
                aprint("  Bright: Backbone atoms")
                aprint("  Dim: Sidechain atoms")

            # Van der Waals radii - DIFFERENT for each element!
            vdw_radii_angstrom = element_to_vdw_radius(sym_elements)
            radii = vdw_radii_angstrom * 0.1  # Convert to nm

            # Scale radii for visibility
            if representation == "calpha":
                radii *= 0.4  # Larger for C-alpha trace
            else:
                radii *= 0.4  # Moderate scaling for all atoms

            aprint("✓ Van der Waals radii applied (different per element):")
            aprint(f"  C: {1.70 * 0.1 * 0.4:.3f} nm")
            aprint(f"  N: {1.55 * 0.1 * 0.4:.3f} nm")
            aprint(f"  O: {1.52 * 0.1 * 0.4:.3f} nm")
            aprint(f"  S: {1.80 * 0.1 * 0.4:.3f} nm")

        # Write to Zarr
        with asection("Creating Luxar scene"):
            with LuxarZarrCompiler(output_path) as compiler:
                dims = Dimensions(
                    [
                        Dimension("x", unit="nm", display=True),
                        Dimension("y", unit="nm", display=True),
                        Dimension("z", unit="nm", display=True),
                    ]
                )
                scene = compiler.create_scene(dimensions=dims)

                # Sharpness for crisp, sharp protein atoms (normalized [0, 1] knob)
                sharpness = np.full(len(sym_positions), 0.85, dtype=np.float32)

                scene.add_points(
                    "nuclear_pore_complex",
                    positions=sym_positions,
                    colors=colors,
                    radii=radii,
                    sharpness=sharpness,
                    opacity=0.95,
                    intensity=0.125,
                )

                # Overlay annotations
                scene.add_text(
                    "Nuclear Pore Complex",
                    position=(0.02, 0.02),
                    font_size=0.055,
                    anchor="top-left",
                    color="rgba(255,255,255,0.6)",
                    blend_mode="difference",
                )
                scene.add_text(
                    "8-fold symmetry \u2022 PDB structure",
                    position=(0.98, 0.97),
                    font_size=0.015,
                    anchor="bottom-right",
                    color="rgba(200,200,200,0.45)",
                )

            aprint(f"✓ Scene created with {len(sym_positions):,} atoms")
            size_mb = sum(
                f.stat().st_size for f in output_path.rglob("*") if f.is_file()
            ) / (1024 * 1024)
            aprint(f"  Dataset size: {size_mb:.1f} MB")

    return len(sym_positions)


# =============================================================================
# Main Entry Point
# =============================================================================


def main() -> None:
    """Main demo entry point."""
    # Parse arguments
    pdb_id = "3I4R"
    max_atoms = 100000
    n_fold = 8
    representation = "all"  # Show all atoms by default
    color_by = "element"  # CPK coloring by element

    if len(sys.argv) > 1:
        for arg in sys.argv[1:]:
            if arg.startswith("--pdb="):
                pdb_id = arg.split("=")[1]
            elif arg.startswith("--max-atoms="):
                max_atoms = int(arg.split("=")[1])
            elif arg.startswith("--symmetry="):
                n_fold = int(arg.split("=")[1])
            elif arg.startswith("--representation="):
                representation = arg.split("=")[1]
            elif arg.startswith("--color="):
                color_by = arg.split("=")[1]

    aprint("=" * 70)
    aprint("NUCLEAR PORE COMPLEX - PERFECT 8-FOLD SYMMETRY")
    aprint("=" * 70)
    aprint("")
    aprint("The Gateway to the Nucleus - Beautiful Octagonal Architecture!")
    aprint("")
    aprint("Visualization approach:")
    aprint(f"  • Representation: {representation}")
    if representation == "calpha":
        aprint("    (C-alpha backbone trace - clean, shows protein fold)")
    aprint(f"  • Color scheme: {color_by}")
    if color_by == "spoke":
        aprint("    (Each spoke a different color - shows symmetry!)")
    aprint("")
    aprint("Structure:")
    aprint("  • ONE Nup107-160 Y-complex from PDB")
    aprint("  • Applied PERFECT 8-fold rotational symmetry")
    aprint("  • Positioned to create ring with CENTRAL PORE")
    aprint("")
    aprint("What makes this beautiful:")
    aprint("  • Octagonal ring structure (top-down view)")
    aprint("  • Each spoke is a different rainbow color")
    aprint("  • Central pore clearly visible")
    aprint("  • Perfect 45° rotational symmetry")
    aprint("")
    aprint("Parameters:")
    aprint(f"  PDB ID: {pdb_id}")
    aprint(f"  Symmetry: C{n_fold} (8-fold)")
    aprint("")

    # If --no-serve, use persistent directory; otherwise temp for auto-cleanup
    if "--no-serve" in sys.argv:
        output_path = get_demos_output_dir() / "nuclear_pore_complex.zarr"
        try:
            n_atoms = generate_nuclear_pore_complex(
                output_path,
                pdb_id=pdb_id,
                max_atoms=max_atoms,
                n_fold=n_fold,
                representation=representation,
                color_by=color_by,
            )
        except Exception as e:
            aprint(f"\n Error: {e}")
            aprint("\nPossible issues:")
            aprint("  - Network connection failed")
            aprint("  - PDB ID not found")
            aprint("  - File format error")
            sys.exit(1)
        aprint(f"Dataset generated at {output_path}")
        aprint(f"Total atoms: {n_atoms:,}")
        return

    # Use temporary directory for serving (auto-cleanup on exit)
    with tempfile.TemporaryDirectory(prefix="luxar_demo_npc_") as tmpdir:
        output_path = Path(tmpdir) / "nuclear_pore_complex.zarr"

        try:
            n_atoms = generate_nuclear_pore_complex(
                output_path,
                pdb_id=pdb_id,
                max_atoms=max_atoms,
                n_fold=n_fold,
                representation=representation,
                color_by=color_by,
            )
        except Exception as e:
            aprint(f"\n Error: {e}")
            aprint("\nPossible issues:")
            aprint("  - Network connection failed")
            aprint("  - PDB ID not found")
            aprint("  - File format error")
            sys.exit(1)

        aprint("")
        aprint("=" * 70)
        aprint("VIEWING TIPS")
        aprint("=" * 70)
        aprint("")
        aprint("Navigation:")
        aprint("  - Top-down (Z-axis): OCTAGONAL RING with CENTRAL PORE!")
        aprint("  - Rotate slowly: See 8 distinct colored spokes")
        aprint("  - Rotate by 45 degrees: Symmetry test - should look identical!")
        aprint("  - Side view: See Y-shaped Nup107-160 complexes")
        aprint("")
        aprint("What to look for:")
        aprint("  - 8 rainbow-colored spokes arranged in perfect octagon")
        aprint("  - Central pore/channel in the middle (molecular highway!)")
        aprint("  - Each spoke is identical (perfect symmetry)")
        aprint("  - Protein backbone showing 3D architecture")
        aprint("")
        aprint("Color guide (spoke mode):")
        aprint(
            "  Red -> Orange -> Yellow -> Green -> Cyan -> Blue -> Purple -> Magenta"
        )
        aprint("  (Each color = one of the 8 identical spokes)")
        aprint("")
        aprint(f"Total atoms: {n_atoms:,}")
        aprint("")
        aprint("=" * 70)
        aprint("LAUNCHING VIEWER")
        aprint("=" * 70)
        aprint("Tip: Look from TOP-DOWN to see the beautiful octagon!")
        aprint("Browser will open automatically. Press Ctrl+C when done.")
        aprint("")

        launch_viewer(output_path)

    aprint("Cleanup complete")
    aprint("")
    aprint("Try different representations:")
    aprint("  --representation=calpha    (clean backbone trace)")
    aprint("  --representation=backbone  (CA, C, N, O atoms)")
    aprint("  --representation=all       (default, all atoms, very dense)")
    aprint("")
    aprint("Try different coloring:")
    aprint("  --color=element       (CPK coloring - default)")
    aprint("  --color=spoke         (rainbow spokes)")
    aprint("  --color=backbone      (warm=backbone, cool=sidechain)")
    aprint("  --color=element-dimmed (CPK with dimmed sidechains)")
    aprint("")


if __name__ == "__main__":
    main()
