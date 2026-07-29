#!/usr/bin/env python3
"""Progressive Writing Example - Memory-efficient scene building for large datasets.

This example demonstrates:
- Using write_points() for direct-to-disk writing (bypasses memory)
- Creating resizable datasets for dynamic/streaming data
- Writing data in batches without loading all into RAM
- Difference between add_points() (in-memory) vs write_points() (streaming)
- Handling datasets larger than available memory

Educational value:
- Learn WHEN to use progressive writing (datasets >1GB or dynamic size)
- Understand memory vs disk trade-offs
- Master batch processing patterns for large data
- Avoid out-of-memory errors with huge datasets

Key principle:
- write_points() writes data directly to zarr (doesn't stay in memory)
- add_points() keeps references until finalization (uses more memory)
- Use progressive writing when dataset size is unknown or very large
"""

import numpy as np
from _overlay_style import add_explainer
from arbol import aprint, asection

from luxar import Dimension, Dimensions
from luxar.compiler import LuxarZarrCompiler
from luxar.utils.paths import get_examples_output_dir


def main():
    """Demonstrate progressive writing with LuxarZarrCompiler."""
    output_path = get_examples_output_dir() / "progressive_writing_example.luxar.zarr"

    with asection("Scene Setup and Dimensions"):
        # Define dimensions for the scene
        dimensions = Dimensions(
            [
                Dimension("x", unit="um", display=True),
                Dimension("y", unit="um", display=True),
                Dimension("z", unit="um", display=True),
            ]
        )
        aprint(f"Defined {len(dimensions)} dimensions for the scene")

    # Use context manager for automatic finalization
    with LuxarZarrCompiler(output_path) as compiler:
        # Create scene with dimensions
        scene = compiler.create_scene(dimensions=dimensions)

        with asection("Progressive Point Writing"):
            aprint("📊 Creating scene with progressive writing...")

            # Create a group for organization
            scene.add_group("DataSet1", opacity=0.9)

            # Generate and write first points
            # Data is written immediately and not kept in memory
            n_points_1 = 10000
            positions_1 = np.random.randn(n_points_1, 3).astype(np.float32) * 10
            colors_1 = np.random.rand(n_points_1, 3).astype(np.float32)

            # This writes directly to disk
            compiler.write_points(
                "DataSet1/cloud1", positions_1, colors=colors_1, opacity=0.8
            )
            aprint(f"  ✓ Wrote {n_points_1:,} points to cloud1")

            # Generate and write second points
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

        with asection("Batch Streaming Large Dataset"):
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
                aprint(
                    f"  ✓ Streamed batch {i + 1}/{n_batches} ({batch_size:,} points)"
                )

            # Tag the manually-streamed group as a points node.
            compiler.write_group(
                "DataSet2/streaming_points",
                type="points",
            )

        add_explainer(
            scene,
            title="Progressive Writing",
            body=(
                "This scene is built without holding all points in RAM: "
                "<code>write_points()</code> streams clouds straight to disk, "
                "and <code>create_resizable_dataset()</code> appends batches "
                "to a growable array. Use this for datasets larger than memory."
            ),
            observe=[
                "Two point clouds (cloud1, cloud2) plus a streamed cloud render together.",
                "cloud2 sits offset in Z and uses additive HDR blending.",
                "The streamed cloud's batches spread out as per-batch scale grows.",
            ],
            observe_label="Look for",
        )

        with asection("Scene Completion and Viewing Instructions"):
            aprint(
                f"✅ Scene created with {n_points_1 + n_points_2 + total_points:,} total points"
            )
            aprint(f"📁 Data written progressively to: {output_path}")
            aprint("")
            aprint("MEMORY BENEFITS:")
            aprint(f"- Only current batch in memory (not all {total_points:,} points)")
            aprint("- Previous batches written to disk and freed")
            aprint("- Essential technique for datasets larger than RAM")

            # Context manager automatically finalizes the store
            aprint("")
            aprint("✅ Scene finalized and ready for viewing")
            aprint("")
            aprint("=" * 60)
            aprint("To view the scene:")
            aprint(f"  luxar serve {output_path}")
            aprint("")
            aprint("WHEN TO USE THIS TECHNIQUE:")
            aprint("- Dataset size unknown (streaming from network/file)")
            aprint("- Dataset larger than available RAM")
            aprint("- Need to process data in chunks")
            aprint("- Building scenes incrementally over time")
            aprint("=" * 60)


if __name__ == "__main__":
    main()
