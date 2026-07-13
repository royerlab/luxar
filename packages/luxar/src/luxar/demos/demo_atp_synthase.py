#!/usr/bin/env python3
"""Self-Contained Demo: ATP Synthase - Nature's Molecular Turbine

Visualize the complete ATP Synthase structure showing the F1 catalytic head
and F0 membrane rotor with all subunits color-coded.

================================================================================
ATP SYNTHASE: THE MOLECULAR POWER PLANT
================================================================================

ATP synthase is one of nature's most elegant molecular machines - a rotary
motor that generates ATP (the universal energy currency of life) by harnessing
the flow of protons down a concentration gradient.

SCALE AND IMPORTANCE:
- ~600 kDa molecular weight
- ~10-20 nm diameter (F1 head), ~8 nm height
- Produces ~100-150 ATP molecules per second
- ALL living things use this enzyme (bacteria to humans!)
- Powers ~95% of cellular energy production

ARCHITECTURE:
===============

The enzyme has two main parts:

1. **F1 (Catalytic Head)** - Sticks into mitochondrial matrix/bacterial cytoplasm:
   - α3β3 hexamer forms the catalytic core
   - 3 catalytic sites where ATP is made
   - γ subunit (central stalk) - ROTATES inside the hexamer
   - δ and ε subunits - regulate activity

2. **F0 (Membrane Rotor)** - Embedded in membrane:
   - c-ring (10-15 c subunits) - forms the rotating ring
   - a subunit - provides proton channel
   - b2 subunit - peripheral stalk (holds F1 stationary)

HOW IT WORKS - THE ROTARY MECHANISM:
====================================

ATP synthase is a TURBINE - literally!

1. **Protons flow** through the a-c interface (down concentration gradient)
2. **c-ring rotates** (like a waterwheel turned by proton flow)
3. **γ stalk rotates** with the c-ring (120° steps)
4. **αβ catalytic sites** undergo conformational changes as γ rotates
5. **ATP is synthesized** - each 120° rotation makes one ATP!

The rotation is DIRECTIONAL and REVERSIBLE:
- Forward: H+ flow → rotation → ATP synthesis (normal mode)
- Reverse: ATP hydrolysis → rotation → H+ pumping (can run backwards!)

ENERGY COUPLING:
- ~3-4 protons needed per ATP synthesized
- Efficiency: ~80-90% (incredible for a molecular machine!)
- Rotation speed: ~100-200 Hz (6000-12000 RPM!)

This is the SAME principle as a hydroelectric dam - mechanical rotation
driven by flow, coupled to energy generation!

NOBEL PRIZE:
Paul D. Boyer and John E. Walker won the 1997 Nobel Prize in Chemistry
for elucidating the mechanism of ATP synthesis.

STRUCTURE DETAILS:
==================

F1 subunit composition:
- α subunit (3 copies) - catalytic hexamer scaffold
- β subunit (3 copies) - contains active sites for ATP synthesis
- γ subunit (1 copy) - central rotating shaft
- δ subunit (1 copy) - connects γ to c-ring
- ε subunit (1 copy) - regulatory, inhibits when needed

F0 subunit composition:
- c subunit (10-15 copies) - rotating ring
- a subunit (1 copy) - proton half-channels
- b subunit (2 copies) - peripheral stalk, stator

VISUALIZATION STRATEGY:
- Download complete ATP synthase structure (PDB: 5DN6)
- Show all atoms or C-alpha backbone
- Color by chain/subunit to show architecture
- Rotating γ stalk visible in center
- c-ring visible as membrane portion

REFERENCES:
-----------
Structure: Mitochondrial ATP synthase
PDB ID: 5DN6
Zhou, A. et al. (2015)
"Structure and conformational states of the bovine mitochondrial ATP synthase"
eLife 4:e10180
DOI: 10.7554/eLife.10180

Review: https://pdb101.rcsb.org/motm/72

Usage:
    python demo_atp_synthase.py [--representation=TYPE]

    Representation types:
    - all: All atoms (default, shows full detail)
    - calpha: C-alpha trace (clean backbone view)
    - backbone: Backbone atoms only (C, N, O, CA)

Controls:
    - Rotate to see the complete motor architecture
    - Top view: See hexagonal F1 head with rotating γ stalk
    - Side view: See membrane portion (F0) and catalytic head (F1)
    - Each chain/subunit has a distinct color
    - Ctrl+C to stop and cleanup
"""

import shutil
import sys
import tempfile
from pathlib import Path

import numpy as np
from arbol import aprint, asection

from luxar import Dimension, Dimensions, LuxarZarrCompiler
from luxar.demos import cached_download, launch_viewer
from luxar.utils.paths import get_demos_output_dir

# =============================================================================
# PDB File Parsing
# =============================================================================


def download_pdb(pdb_id: str, output_path: Path) -> Path:
    """Fetch a PDB structure from RCSB, cached across runs.

    The download is cached under ``~/.cache/luxar/atp_synthase/`` (via
    :func:`cached_download`), so repeat runs never re-hit RCSB; a copy is placed
    at ``output_path`` for the caller's temp-dir workflow.

    Args:
        pdb_id: 4-character PDB ID
        output_path: Where to place the (copied) file

    Returns:
        Path to the file at ``output_path``
    """
    url = f"https://files.rcsb.org/download/{pdb_id}.pdb"
    cached = cached_download(url, "atp_synthase", f"{pdb_id}.pdb")
    shutil.copy2(cached, output_path)
    return output_path


def parse_pdb_atoms(
    pdb_path: Path, atom_filter: str = "all", max_atoms: int = 100000
) -> tuple[np.ndarray, np.ndarray, np.ndarray, list[str]]:
    """Parse PDB file and extract atom coordinates.

    Args:
        pdb_path: Path to PDB file
        atom_filter: "calpha", "backbone", or "all"
        max_atoms: Maximum atoms to read

    Returns:
        Tuple of (positions, element_types, chain_ids, chain_names)
    """
    positions = []
    elements = []
    chain_ids = []

    # Element to index mapping
    element_map = {"C": 0, "N": 1, "O": 2, "S": 3, "P": 4}
    default_element = 0

    # Backbone atom names
    backbone_atoms = {"CA", "C", "N", "O"}

    # Track unique chains
    seen_chains = set()

    with open(pdb_path) as f:
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

                    # Chain identifier
                    chain_id = line[21:22].strip()
                    if not chain_id:
                        chain_id = "A"

                    # Element symbol
                    element = line[76:78].strip()
                    if not element:
                        element = atom_name[0] if atom_name else "C"

                    element_idx = element_map.get(element, default_element)

                    positions.append([x, y, z])
                    elements.append(element_idx)
                    chain_ids.append(chain_id)
                    seen_chains.add(chain_id)

                except (ValueError, IndexError):
                    continue

    if not positions:
        raise ValueError(f"No atoms found in {pdb_path}")

    # Convert chain IDs to unique list
    unique_chains = sorted(seen_chains)
    aprint(f"  Found {len(unique_chains)} chains: {', '.join(unique_chains)}")

    return (
        np.array(positions, dtype=np.float32),
        np.array(elements, dtype=np.int32),
        chain_ids,  # Keep as list for now
        unique_chains,
    )


# =============================================================================
# Geometric Utilities
# =============================================================================


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


def chain_to_color(chain_ids: list[str], unique_chains: list[str]) -> np.ndarray:
    """Convert chain IDs to distinct colors.

    Each chain/subunit gets a different color to show the architecture.

    Args:
        chain_ids: List of chain IDs for each atom
        unique_chains: List of unique chain identifiers

    Returns:
        (N, 3) RGB colors
    """
    n_chains = len(unique_chains)

    colors = np.zeros((len(chain_ids), 3), dtype=np.float32)

    # Generate distinct colors using rainbow gradient
    for i, chain in enumerate(unique_chains):
        hue = i / n_chains
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

        # Apply to all atoms in this chain
        for j, atom_chain in enumerate(chain_ids):
            if atom_chain == chain:
                colors[j] = [r, g, b]

    return colors


def element_to_color(elements: np.ndarray) -> np.ndarray:
    """Convert element types to CPK colors.

    Args:
        elements: (N,) array of element indices

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

    return colors


# =============================================================================
# Main Generation
# =============================================================================


def generate_atp_synthase(
    output_path: Path,
    pdb_id: str = "5DN6",
    max_atoms: int = 200000,
    representation: str = "all",
    color_by: str = "chain",
) -> int:
    """Generate ATP Synthase molecular motor visualization.

    Args:
        output_path: Where to write zarr
        pdb_id: PDB ID (default: 5DN6 - bovine mitochondrial ATP synthase)
        max_atoms: Maximum atoms to load
        representation: "calpha", "backbone", or "all"
        color_by: "chain" or "element"

    Returns:
        Total number of atoms visualized
    """
    with tempfile.TemporaryDirectory(prefix="luxar_pdb_") as tmpdir:
        pdb_path = Path(tmpdir) / f"{pdb_id}.pdb"

        # Download structure
        with asection(f"Downloading PDB structure {pdb_id}"):
            aprint("Structure: ATP Synthase (F1F0 complex)")
            aprint("Source: Bovine mitochondrial ATP synthase")
            aprint(f"Representation: {representation}")
            aprint("")
            download_pdb(pdb_id, pdb_path)

        # Parse structure
        with asection(f"Parsing PDB file ({representation} atoms)"):
            positions, elements, chain_ids, unique_chains = parse_pdb_atoms(
                pdb_path, atom_filter=representation, max_atoms=max_atoms
            )

            aprint(f"✓ Parsed {len(positions):,} atoms")
            aprint(f"  {len(unique_chains)} chains/subunits")

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

        # Generate colors and radii
        with asection(f"Generating colors and radii (color_by={color_by})"):
            if color_by == "chain":
                colors = chain_to_color(chain_ids, unique_chains)
                aprint("✓ Rainbow coloring by chain/subunit:")
                aprint(f"  {len(unique_chains)} distinct colors (one per chain)")
                aprint("  Shows F1 head, F0 rotor, and peripheral stalk!")
            elif color_by == "element":
                colors = element_to_color(elements)
                aprint("✓ CPK coloring by element:")
                aprint("  White: Carbon")
                aprint("  Blue: Nitrogen")
                aprint("  Red: Oxygen")
                aprint("  Yellow: Sulfur")
                aprint("  Orange: Phosphorus")
            else:
                colors = element_to_color(elements)

            # Van der Waals radii
            vdw_radii_angstrom = element_to_vdw_radius(elements)
            radii = vdw_radii_angstrom * 0.1  # Convert to nm

            # Scale radii for visibility
            if representation == "calpha":
                radii *= 0.4  # Moderate for C-alpha trace
            else:
                radii *= 0.4  # Moderate scaling for all atoms

            aprint("✓ Van der Waals radii applied:")
            aprint(f"  C: {1.70 * 0.1 * 0.4:.3f} nm")
            aprint(f"  N: {1.55 * 0.1 * 0.4:.3f} nm")
            aprint(f"  O: {1.52 * 0.1 * 0.4:.3f} nm")

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

                # Sharpness for protein atoms (normalized [0, 1] knob; 0.5 = Gaussian)
                sharpness = np.full(len(positions), 0.5, dtype=np.float32)

                scene.add_points(
                    "atp_synthase",
                    positions=positions,
                    colors=colors,
                    radii=radii,
                    sharpness=sharpness,
                    opacity=0.95,
                    intensity=0.0625,
                )

                # Overlay annotations
                scene.add_text(
                    "ATP Synthase",
                    position=(0.02, 0.02),
                    font_size=0.055,
                    anchor="top-left",
                    color="rgba(255,255,255,0.6)",
                    blend_mode="difference",
                )
                scene.add_text(
                    "PDB 5DN6 \u2022 Molecular turbine",
                    position=(0.98, 0.97),
                    font_size=0.015,
                    anchor="bottom-right",
                    color="rgba(200,200,200,0.45)",
                )

            aprint(f"✓ Scene created with {len(positions):,} atoms")
            size_mb = sum(
                f.stat().st_size for f in output_path.rglob("*") if f.is_file()
            ) / (1024 * 1024)
            aprint(f"  Dataset size: {size_mb:.1f} MB")

    return len(positions)


# =============================================================================
# Main Entry Point
# =============================================================================


def main() -> None:
    """Main demo entry point."""
    # Parse arguments
    pdb_id = "5DN6"
    max_atoms = 200000
    representation = "all"  # Show all atoms by default
    color_by = "chain"  # Color by chain to show subunits

    if len(sys.argv) > 1:
        for arg in sys.argv[1:]:
            if arg.startswith("--pdb="):
                pdb_id = arg.split("=")[1]
            elif arg.startswith("--max-atoms="):
                max_atoms = int(arg.split("=")[1])
            elif arg.startswith("--representation="):
                representation = arg.split("=")[1]
            elif arg.startswith("--color="):
                color_by = arg.split("=")[1]

    aprint("=" * 70)
    aprint("ATP SYNTHASE - NATURE'S MOLECULAR TURBINE")
    aprint("=" * 70)
    aprint("")
    aprint("The enzyme that powers ALL life!")
    aprint("")
    aprint("Structure:")
    aprint("  • F1 catalytic head (α3β3γδε) - where ATP is made")
    aprint("  • F0 membrane rotor (c-ring + a + b2) - driven by protons")
    aprint("  • Central rotating shaft (γ subunit) - the axle!")
    aprint("  • Peripheral stalk (b2) - holds F1 stationary")
    aprint("")
    aprint("How it works:")
    aprint("  1. Protons flow through F0 (down gradient)")
    aprint("  2. c-ring rotates like a turbine wheel")
    aprint("  3. γ shaft rotates with c-ring (120° steps)")
    aprint("  4. Rotation triggers ATP synthesis in β subunits")
    aprint("  5. Makes ~100-150 ATP per second!")
    aprint("")
    aprint("This is literally a ROTARY MOTOR at molecular scale!")
    aprint("Same principle as hydroelectric dam - flow → rotation → energy")
    aprint("")
    aprint("Visualization:")
    aprint(f"  • Representation: {representation}")
    if representation == "all":
        aprint("    (All atoms - full structural detail)")
    aprint(f"  • Coloring: {color_by}")
    if color_by == "chain":
        aprint("    (Each chain/subunit a different color)")
    aprint("")
    aprint(f"PDB ID: {pdb_id}")
    aprint("")

    # If --no-serve, use persistent directory; otherwise temp for auto-cleanup
    if "--no-serve" in sys.argv:
        output_path = get_demos_output_dir() / "atp_synthase.luxar.zarr"
        try:
            generate_atp_synthase(
                output_path,
                pdb_id=pdb_id,
                max_atoms=max_atoms,
                representation=representation,
                color_by=color_by,
            )
        except Exception as e:
            aprint(f"\n❌ Error: {e}")
            aprint("\nPossible issues:")
            aprint("  • Network connection failed")
            aprint("  • PDB ID not found")
            aprint("  • File format error")
            sys.exit(1)
        aprint(f"Dataset generated at {output_path}")
        return

    # Generate structure
    with tempfile.TemporaryDirectory(prefix="luxar_demo_atp_") as tmpdir:
        output_path = Path(tmpdir) / "atp_synthase.luxar.zarr"

        try:
            n_atoms = generate_atp_synthase(
                output_path,
                pdb_id=pdb_id,
                max_atoms=max_atoms,
                representation=representation,
                color_by=color_by,
            )
        except Exception as e:
            aprint(f"\n❌ Error: {e}")
            aprint("\nPossible issues:")
            aprint("  • Network connection failed")
            aprint("  • PDB ID not found")
            aprint("  • File format error")
            sys.exit(1)

        aprint("")
        aprint("=" * 70)
        aprint("VIEWING TIPS")
        aprint("=" * 70)
        aprint("")
        aprint("What to look for:")
        aprint("  • F1 head: Hexagonal arrangement (α3β3)")
        aprint("  • γ shaft: Central rotating stalk (different color)")
        aprint("  • F0 rotor: Membrane-embedded c-ring")
        aprint("  • Peripheral stalk: Connects to side of F1")
        aprint("  • Each subunit type has distinct color")
        aprint("")
        aprint("Architecture:")
        aprint("  • Top view: See hexagonal F1 head")
        aprint("  • Side view: See F1 (top) and F0 (bottom)")
        aprint("  • The γ shaft goes through the CENTER!")
        aprint("")
        aprint(f"Total atoms: {n_atoms:,}")
        aprint("")
        aprint("Nobel Prize 1997:")
        aprint("  Paul Boyer & John Walker discovered the rotary mechanism")
        aprint("  One of biology's most elegant molecular machines!")
        aprint("")
        aprint("=" * 70)
        aprint("LAUNCHING VIEWER")
        aprint("=" * 70)
        aprint("Tip: Rotate to see the molecular motor from all angles!")
        aprint("Browser will open automatically. Press Ctrl+C when done.")
        aprint("")

        launch_viewer(output_path)

    aprint("Cleanup complete")
    aprint("")
    aprint("Try different representations:")
    aprint("  --representation=calpha    (clean backbone trace)")
    aprint("  --representation=backbone  (CA, C, N, O atoms)")
    aprint("  --representation=all       (default, all atoms)")
    aprint("")
    aprint("Try different coloring:")
    aprint("  --color=chain     (default, rainbow by subunit)")
    aprint("  --color=element   (CPK coloring)")
    aprint("")
    aprint("Other ATP synthase structures to try:")
    aprint("  --pdb=6J5K   (Yeast mitochondrial)")
    aprint("  --pdb=6B8H   (Bovine F1 portion only)")
    aprint("")


if __name__ == "__main__":
    main()
