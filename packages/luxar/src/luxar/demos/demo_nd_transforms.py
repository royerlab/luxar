#!/usr/bin/env python3
"""Self-Contained Demo: Multi-Instrument Observatory with nD Transforms

This demo demonstrates Luxar's nD transform feature — per-dimension affine
transforms on non-displayed dimensions. It shows how data from different
"instruments" (groups) can be aligned in time and channel space using
nd_transform, without modifying the raw data.

================================================================================
CONCEPT: MULTI-INSTRUMENT OBSERVATORY
================================================================================

An observatory captures the same galaxy cluster with three instruments:

1. **Optical Telescope** — captures at times [0..9] in optical channels
   (Hα, OIII, SII), stored in raw detector coordinates.

2. **Radio Telescope** — captures at times [0..4] in radio channels,
   but the data was taken 5 time units later than the optical data.
   We use nd_transform to shift it: Time offset=5.

3. **X-ray Satellite** — captures at times [0..4] at 2× time resolution,
   so each X-ray frame spans half the time. We use nd_transform to
   scale and align: Time scale=2, offset=0. Channels are also remapped
   to match the optical channel ordering.

All three instruments share the same 3D spatial coordinates (X, Y, Z).
The nd_transform on each group aligns the Time and Channel dimensions
so the viewer's time slider shows a coherent multi-instrument view.

================================================================================
FEATURES DEMONSTRATED
================================================================================

- nd_transform with affine: scale + offset on Time dimension
- nd_transform with permutation on Channel (categorical) dimension
- Hierarchical inheritance: parent group nd_transform applies to children
- Multiple groups in the same scene with different nd_transforms
- Spatial (4x4) transform combined with nd_transform on the same group

Usage:
    python demo_nd_transforms.py

Controls:
    - Press '4' to select Time dimension, then use [ ] to navigate
    - Press '5' to select Channel dimension, then use [ ] to navigate
    - Ctrl+C to stop and cleanup
    - Browser opens automatically
"""

DEMO_META = {
    "key": "nd_transforms",
    "title": "Multi-Instrument Observatory with nD Transforms",
    "description": "Multi-instrument observatory aligned in time and channel via per-dimension nD transforms.",
    "category": "synthetic",
    "geometry": "points",
    "requirements": {
        "download_mb": 0,
        "compute": "light",
        "gpu": "none",
        "local_data": None,
    },
    "caches": [],
    "outputs": ["nd_transforms_observatory"],
}

from pathlib import Path

import numpy as np
from arbol import aprint, asection

from luxar import (
    Dimension,
    Dimensions,
    LuxarZarrCompiler,
    ViewerConfig,
    transforms,
)
from luxar.demos import launch_viewer
from luxar.utils.paths import get_demos_output_dir


def generate_galaxy_cluster(
    rng: np.random.Generator,
    n_points: int,
    center: tuple[float, float, float],
    spread: float,
) -> np.ndarray:
    """Generate a simple galaxy cluster (3D positions)."""
    positions = rng.normal(0, spread, (n_points, 3)).astype(np.float32)
    # Add some structure: elongated in X, thinner in Z
    positions[:, 0] *= 1.5
    positions[:, 2] *= 0.4
    positions += np.array(center, dtype=np.float32)
    return positions


def make_5d_data(
    positions_3d: np.ndarray,
    time_steps: list[int],
    channel: int,
    rng: np.random.Generator,
) -> tuple[np.ndarray, np.ndarray]:
    """Replicate 3D positions into 5D (X, Y, Z, Time, Channel) with variation."""
    all_pos = []
    all_rad = []

    for t in time_steps:
        n = positions_3d.shape[0]
        # Small random perturbation per time step (evolution)
        jitter = rng.normal(0, 0.3, (n, 3)).astype(np.float32)
        pos5d = np.zeros((n, 5), dtype=np.float32)
        pos5d[:, :3] = positions_3d + jitter
        pos5d[:, 3] = t
        pos5d[:, 4] = channel
        all_pos.append(pos5d)

        # Brightness varies with time
        brightness = 0.5 + 0.5 * np.sin(t * 0.6 + channel)
        radii = np.full(n, 0.4 + 0.1 * brightness, dtype=np.float32)
        all_rad.append(radii)

    positions = np.vstack(all_pos)
    radii = np.concatenate(all_rad)

    return positions, radii


def generate_demo(output_path: Path) -> int:
    """Generate the multi-instrument observatory scene.

    Returns:
        Total number of points generated.
    """
    rng = np.random.default_rng(42)

    # Shared galaxy cluster positions (3D)
    cluster = generate_galaxy_cluster(rng, n_points=3000, center=(0, 0, 0), spread=10.0)
    satellite_cluster = generate_galaxy_cluster(
        rng, n_points=2000, center=(15, 5, 0), spread=6.0
    )

    total_points = 0

    # Scene dimensions: 5D (X, Y, Z, Time, Channel)
    dims = Dimensions(
        [
            Dimension(name="X", unit="Mpc", range=(-80, 80), display=True),
            Dimension(name="Y", unit="Mpc", range=(-50, 50), display=True),
            Dimension(name="Z", unit="Mpc", range=(-25, 25), display=True),
            Dimension(
                name="Time",
                unit="epoch",
                range=(0, 14),
                display=False,
                discrete=True,
                step=1.0,
            ),
            Dimension(
                name="Channel",
                unit="",
                categories=["Hα", "OIII", "SII"],
                display=False,
            ),
        ]
    )

    with asection("Writing Multi-Instrument Observatory Scene"):
        with LuxarZarrCompiler(output_path) as compiler:
            scene = compiler.create_scene(
                dimensions=dims,
                viewer_config=ViewerConfig(
                    background_color="#050510",
                    bloom_enabled=True,
                    bloom_strength=0.4,
                    bloom_threshold=0.3,
                    auto_rotate=True,
                    auto_rotate_speed=0.3,
                ),
            )

            # ================================================================
            # INSTRUMENT 1: Optical Telescope (baseline, no nd_transform)
            # Captures at times [0..9], all 3 channels
            # ================================================================
            with asection("Instrument 1: Optical Telescope"):
                optical_group = scene.add_group(
                    "Optical",
                    # No nd_transform — this is the reference frame
                    opacity=0.9,
                    blending_mode="additive",
                )

                # Optical channels: Hα (red), OIII (teal), SII (yellow)
                channel_colors = {
                    0: np.array([1.0, 0.2, 0.1]),  # Hα → red
                    1: np.array([0.1, 0.8, 0.7]),  # OIII → teal
                    2: np.array([0.9, 0.8, 0.2]),  # SII → yellow
                }

                for ch_idx, ch_color in channel_colors.items():
                    positions, radii = make_5d_data(
                        cluster,
                        list(range(10)),
                        ch_idx,
                        rng,
                    )
                    n = positions.shape[0]
                    colors = np.tile(ch_color, (n, 1)).astype(np.float32)
                    colors *= rng.uniform(0.5, 1.5, (n, 1)).astype(np.float32)

                    optical_group.add_points(
                        f"Optical_Ch{ch_idx}",
                        positions,
                        colors=colors,
                        radii=radii,
                        sharpness=0.6,
                        extend_to_all=[],
                        intensity=0.125,
                    )
                    total_points += n
                    aprint(f"  Channel {ch_idx}: {n:,} points")

            # ================================================================
            # INSTRUMENT 2: Radio Telescope (time-shifted by +5)
            # Captures at local times [0..4], but they correspond to
            # world times [5..9] (started observing 5 epochs later).
            # ================================================================
            with asection("Instrument 2: Radio Telescope (time offset +5)"):
                radio_group = scene.add_group(
                    "Radio",
                    transform=transforms.translate(0.5, 0, 0),  # Slight spatial offset
                    nd_transform={
                        "Time": {"offset": 5.0},  # Local t=0 → world t=5
                    },
                    opacity=0.8,
                    blending_mode="additive",
                )

                # Radio sees only one "channel" mapped to OIII
                positions, radii = make_5d_data(
                    cluster,
                    list(range(5)),
                    1,
                    rng,  # ch=1 (OIII)
                )
                n = positions.shape[0]
                colors = np.tile(
                    np.array([0.3, 0.6, 1.0]),
                    (n, 1),
                ).astype(np.float32)
                colors *= rng.uniform(0.5, 2.0, (n, 1)).astype(np.float32)

                radio_group.add_points(
                    "Radio_Emission",
                    positions,
                    colors=colors,
                    radii=radii * 1.5,
                    sharpness=0.5,
                    extend_to_all=[],
                    intensity=0.125,
                )
                total_points += n
                aprint(f"  Radio emission: {n:,} points (time offset=5)")

            # ================================================================
            # INSTRUMENT 3: X-ray Satellite (time scaled ×2)
            # Captures at local times [0..4], but at 2× resolution,
            # so world time = 2 * local_time. Also channels are permuted:
            # satellite records [SII, Hα, OIII] → need permutation [1,2,0]
            # to map to scene order [Hα, OIII, SII].
            # ================================================================
            with asection(
                "Instrument 3: X-ray Satellite (time scale ×2, channel permuted)"
            ):
                # Satellite channel order: [SII, Hα, OIII]
                # Scene channel order:     [Hα, OIII, SII]
                # Permutation: sat_ch0(SII)→scene_ch2, sat_ch1(Hα)→scene_ch0, sat_ch2(OIII)→scene_ch1
                # So: permutation[0]=2, permutation[1]=0, permutation[2]=1 → [2, 0, 1]
                xray_group = scene.add_group(
                    "Xray",
                    transform=transforms.translate(-0.5, 0, 0),
                    nd_transform={
                        "Time": {"scale": 2.0},
                        "Channel": {"permutation": [2, 0, 1]},
                    },
                )

                # X-ray data: bright point sources
                xray_colors_by_ch = {
                    0: np.array([1.0, 0.5, 1.0]),  # Sat ch0 (→ SII after perm)
                    1: np.array([1.0, 0.3, 0.3]),  # Sat ch1 (→ Hα after perm)
                    2: np.array([0.3, 1.0, 1.0]),  # Sat ch2 (→ OIII after perm)
                }

                for ch_idx, ch_color in xray_colors_by_ch.items():
                    # Fewer points, more concentrated (satellite cluster)
                    positions, radii = make_5d_data(
                        satellite_cluster,
                        list(range(5)),
                        ch_idx,
                        rng,
                    )
                    n = positions.shape[0]
                    colors = np.tile(ch_color, (n, 1)).astype(np.float32)
                    colors *= rng.uniform(0.8, 3.0, (n, 1)).astype(np.float32)

                    xray_group.add_points(
                        f"Xray_Ch{ch_idx}",
                        positions,
                        colors=colors,
                        radii=radii * 0.8,
                        sharpness=0.8,
                        blending_mode="additive",
                        extend_to_all=[],
                        intensity=0.125,
                    )
                    total_points += n
                    aprint(f"  X-ray ch{ch_idx}: {n:,} points (scale=2, permuted)")

            # ================================================================
            # OVERLAYS
            # ================================================================
            # Title
            scene.add_text(
                "Multi-Instrument Observatory",
                position=(0.02, 0.02),
                font_size=0.055,
                anchor="top-left",
                color="rgba(255,255,255,0.6)",
                blend_mode="difference",
            )

            # Info
            scene.add_text(
                "3 instruments \u2022 nD transforms",
                position=(0.98, 0.97),
                font_size=0.015,
                anchor="bottom-right",
                color="rgba(200,200,200,0.45)",
            )

            # ================================================================
            # VERIFICATION: Print world nd_transforms
            # ================================================================
            with asection("Verifying nd_transforms"):
                aprint(
                    f"Optical world_nd_transform: {optical_group.world_nd_transform}"
                )
                aprint(f"Radio world_nd_transform:   {radio_group.world_nd_transform}")
                aprint(f"Xray world_nd_transform:    {xray_group.world_nd_transform}")

        aprint(f"\nTotal points: {total_points:,}")
        aprint(f"Written to {output_path}")

    return total_points


def main() -> None:
    """Main demo entry point."""
    aprint("=" * 70)
    aprint("ND TRANSFORMS DEMO: MULTI-INSTRUMENT OBSERVATORY")
    aprint("=" * 70)
    aprint("")
    aprint("Demonstrates per-dimension transforms on non-displayed dimensions:")
    aprint("  - Time offset (Radio telescope delayed by 5 epochs)")
    aprint("  - Time scale (X-ray satellite at 2× temporal resolution)")
    aprint("  - Channel permutation (X-ray channels in different order)")
    aprint("  - Combined spatial transform + nd_transform on same group")
    aprint("")

    output_path = get_demos_output_dir() / "nd_transforms_observatory.luxar.zarr"

    total_points = generate_demo(output_path)

    aprint("")
    aprint(f"Generated {total_points:,} points total")
    aprint("")
    aprint("Navigation tips:")
    aprint("  - Press '4' to select Time, use [ ] to step through epochs")
    aprint("  - Press '5' to select Channel, use [ ] to switch Hα/OIII/SII")
    aprint("  - Radio data appears at Time ≥ 5 (time-shifted)")
    aprint("  - X-ray data appears at even Time values (time-scaled)")
    aprint("")

    launch_viewer(output_path)


if __name__ == "__main__":
    main()
