#!/usr/bin/env python3
"""Progressive Writing Example - Demonstrates memory-efficient scene building.

This example shows how to use the new LuxarZarrCompiler for progressive writing,
which enables handling datasets larger than available RAM.
"""

from pathlib import Path

import numpy as np
from arbol import aprint

from luxar import Dimension, Dimensions
from luxar.compiler import LuxarZarrCompiler


def main():
    """Demonstrate progressive writing with LuxarZarrCompiler."""
    output_path = Path(__file__).parent / "progressive_writing_example.zarr"

    # Define dimensions for the scene
    dimensions = Dimensions(
        [
            Dimension("x", unit="um", display=True),
            Dimension("y", unit="um", display=True),
            Dimension("z", unit="um", display=True),
        ]
    )

    # Use context manager for automatic finalization
    with LuxarZarrCompiler(output_path) as compiler:
        # Create scene with dimensions
        scene = compiler.create_scene(dimensions=dimensions)

        aprint("📊 Creating scene with progressive writing...")

        # Create a group for organization
        scene.add_group("DataSet1", opacity=0.9)

        # Generate and write first point cloud
        # Data is written immediately and not kept in memory
        n_points_1 = 10000
        positions_1 = np.random.randn(n_points_1, 3).astype(np.float32) * 10
        colors_1 = np.random.rand(n_points_1, 3).astype(np.float32)

        # This writes directly to disk
        compiler.write_points(
            "DataSet1/cloud1", positions_1, colors=colors_1, opacity=0.8
        )
        aprint(f"  ✓ Wrote {n_points_1:,} points to cloud1")

        # Generate and write second point cloud
        # Previous data is already on disk, not in memory
        n_points_2 = 15000
        positions_2 = np.random.randn(n_points_2, 3).astype(np.float32) * 15
        positions_2[:, 2] += 20  # Offset in Z

        # HDR colors (values > 1.0 for emission)
        colors_2 = np.random.rand(n_points_2, 3).astype(np.float32) * 2.0

        compiler.write_points(
            "DataSet1/cloud2",
            positions_2,
            colors=colors_2,
            opacity=0.7,
            blending_mode="additive",
        )
        aprint(f"  ✓ Wrote {n_points_2:,} points to cloud2")

        # Create another group
        scene.add_group("DataSet2")

        # Demonstrate streaming large dataset in batches
        aprint("📊 Streaming large dataset in batches...")

        # Create resizable dataset for streaming
        total_points = 0
        batch_size = 5000
        n_batches = 10

        # Initialize resizable dataset
        dataset_path = "DataSet2/streaming_points/positions"
        positions_dataset = compiler.create_resizable_dataset(
            dataset_path,
            dtype=np.float32,
            shape=(0, 3),  # Start with 0 points
            maxshape=(None, 3),  # Unlimited points
        )

        # Stream data in batches
        for i in range(n_batches):
            # Generate batch (simulating reading from file/network)
            batch_positions = np.random.randn(batch_size, 3).astype(np.float32)
            batch_positions *= (i + 1) * 5  # Vary scale per batch

            # Resize and append
            new_total = total_points + batch_size
            positions_dataset.resize((new_total, 3))
            positions_dataset[total_points:new_total] = batch_positions

            total_points = new_total
            aprint(f"  ✓ Streamed batch {i + 1}/{n_batches} ({batch_size:,} points)")

        # Store metadata for streaming points
        compiler.write_group(
            "DataSet2/streaming_points",
            type="points",
            num_points=total_points,
            streaming=True,
        )

        aprint(
            f"✅ Scene created with {n_points_1 + n_points_2 + total_points:,} total points"
        )
        aprint(f"📁 Data written progressively to: {output_path}")

        # Context manager automatically finalizes the store
        aprint("✅ Scene finalized and ready for viewing")
        aprint("\nTo view the scene, run:")
        aprint("  cd packages/luxar-player && pnpm dev")
        aprint(f"  # Then open http://localhost:5173/?source=../../{output_path}")


if __name__ == "__main__":
    main()
