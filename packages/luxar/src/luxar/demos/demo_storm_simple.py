#!/usr/bin/env python3
"""Simple 3D STORM Demo - Just Super-Resolution, No Complications

This is a simplified version that JUST shows super-resolution STORM data
as 3D Gaussian splats. Works reliably.
"""

import subprocess
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from arbol import aprint, asection

from luxar import Dimension, Dimensions, LuxarZarrCompiler
from luxar.gsplats.fit_result import GSplatData

# Paths
CACHE_DIR = Path.home() / ".cache" / "luxar" / "storm_data"
CSV_FILE = CACHE_DIR / "Cos7_MT_A647_FOV_4_Localizations.csv"
OUTPUT_DIR = Path(__file__).parent.parent.parent / "examples"
OUTPUT_PATH = OUTPUT_DIR / "storm_simple.zarr"

PIXEL_SIZE = 106.0  # nm


def main():
    """Simple STORM visualization."""
    max_loc = 100000

    if not CSV_FILE.exists():
        aprint(f"❌ Data file not found: {CSV_FILE}")
        aprint("Please download manually from https://zenodo.org/record/3547521")
        sys.exit(1)

    # Parse CSV
    with asection("Loading STORM data"):
        df = pd.read_csv(CSV_FILE, nrows=max_loc)
        aprint(f"✓ Loaded {len(df):,} localizations")

        # Extract coordinates (in nm)
        x_nm = df['x_nm'].values
        y_nm = df['y_nm'].values
        z_nm = df['z_nm'].values

        # Convert to μm and center
        centers = np.column_stack([x_nm, y_nm, z_nm]).astype(np.float32) / 1000
        centers = centers - centers.mean(axis=0)

        aprint(f"✓ Centered at origin")
        aprint(f"  Range: {centers.min(axis=0)} to {centers.max(axis=0)} μm")

        # Precision (CRLB in pixels, convert to μm)
        prec_x = df['crlb_x'].values * PIXEL_SIZE / 1000
        prec_y = df['crlb_y'].values * PIXEL_SIZE / 1000
        prec_z = df['crlb_z'].values * PIXEL_SIZE / 1000

        # Create diagonal covariances
        n = len(centers)
        cholesky = np.zeros((n, 6), dtype=np.float32)
        for i in range(n):
            # Diagonal covariance
            sx, sy, sz = prec_x[i] * 1.5, prec_y[i] * 1.5, prec_z[i] * 1.5
            cov = np.diag([sx**2, sy**2, sz**2])
            L = np.linalg.cholesky(cov)
            cholesky[i] = [L[0,0], L[1,0], L[1,1], L[2,0], L[2,1], L[2,2]]

        # Amplitudes from photons
        amps = df['photons'].values.astype(np.float32)
        amps = amps / np.percentile(amps, 99) * 0.3

        # Colors and sharpness
        colors = np.full((n, 3), [0.3, 0.9, 0.9], dtype=np.float32)
        sharpness = np.full(n, 3.0, dtype=np.float32)

        aprint(f"✓ Created {n:,} gsplats")

    # Create scene
    with asection("Creating scene"):
        OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

        dims = Dimensions([
            Dimension("x", unit="μm", display=True),
            Dimension("y", unit="μm", display=True),
            Dimension("z", unit="μm", display=True),
        ])

        with LuxarZarrCompiler(OUTPUT_PATH) as compiler:
            scene = compiler.create_scene(dimensions=dims)

            gsplat_data = GSplatData(
                centers=centers,
                cholesky_factors=cholesky,
                amplitudes=amps,
                sharpnesses=sharpness,
                colors=colors,
            )

            scene.add_gsplats_from_data(
                name="storm",
                result=gsplat_data,
                opacity=0.9,
                blending_mode="additive",
            )

        aprint(f"✓ Scene saved: {OUTPUT_PATH}")

    # Launch viewer
    if "--no-serve" not in sys.argv:
        aprint("\n🚀 Launching viewer...")
        subprocess.run(["luxar", "serve", str(OUTPUT_PATH), "--viewer", "--open"], check=True)


if __name__ == "__main__":
    main()
