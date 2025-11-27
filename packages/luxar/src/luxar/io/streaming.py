"""luxar.streaming – Streaming support for handling huge points.

This module provides the StreamingPoints class for progressively writing
massive points that exceed available RAM.
"""

from __future__ import annotations

from typing import Any, Dict, Optional

import numpy as np
from arbol import aprint
from numpy.typing import NDArray

from ..io.writer import ZarrWriterProtocol
from ..validation.base import (
    validate_colors_for_writing,
    validate_positions_for_writing,
    validate_radii_for_writing,
    validate_sharpness_for_writing,
)


class StreamingPoints:
    """Special node type for streaming huge points.

    This class enables appending point data in batches without loading
    existing data into memory, perfect for datasets larger than RAM.

    Args:
        name: Name for the points
        writer: Writer interface for progressive writing
        expected_dims: Expected dimensionality of points
        initial_capacity: Initial capacity (will auto-grow)

    Examples:
        Basic streaming:
        >>> with LuxarZarrCompiler('output.zarr') as compiler:
        ...     streaming = StreamingPoints("huge_cloud", compiler, expected_dims=3)
        ...     for batch in data_generator():
        ...         streaming.append_batch(batch)  # Written immediately
        ...     streaming.finalize()

        Streaming with attributes:
        >>> streaming = StreamingPoints("colored_cloud", compiler)
        >>> for i in range(1000):
        ...     positions = load_positions_batch(i)
        ...     colors = load_colors_batch(i)
        ...     radii = np.full(len(positions), 0.1, dtype=np.float32)
        ...     streaming.append_batch(positions, colors=colors, radii=radii)
        >>> metadata = streaming.finalize(opacity=0.8)

        Using generator convenience method:
        >>> def batch_generator():
        ...     for i in range(100):
        ...         yield np.random.randn(1000, 3).astype(np.float32)
        >>>
        >>> streaming = StreamingPoints("generated", compiler)
        >>> total = streaming.append_from_generator(batch_generator())
        >>> print(f"Streamed {total:,} points")
    """

    def __init__(
        self,
        name: str,
        writer: ZarrWriterProtocol,
        expected_dims: int = 3,
        initial_capacity: int = 0,
    ) -> None:
        """Initialize streaming points.

        Args:
            name: Name for this points
            writer: Writer for progressive output
            expected_dims: Expected dimensionality
            initial_capacity: Initial array capacity
        """
        self.name = name
        self.writer = writer
        self.expected_dims = expected_dims
        self.position = 0
        self.total_points = 0

        # Dataset handles (created on first append)
        self.positions_dataset: Optional[Any] = None
        self.colors_dataset: Optional[Any] = None
        self.radii_dataset: Optional[Any] = None
        self.sharpness_dataset: Optional[Any] = None

        # Metadata tracking
        self.has_colors = False
        self.has_radii = False
        self.has_sharpness = False

        aprint(f"📊 Created streaming points '{name}' for {expected_dims}D data")

    def append_batch(
        self,
        positions: NDArray[np.float32],
        colors: Optional[NDArray[np.float32]] = None,
        radii: Optional[NDArray[np.float32]] = None,
        sharpness: Optional[NDArray[np.float32]] = None,
    ) -> None:
        """Append a batch of points without loading existing data.

        Args:
            positions: Point positions of shape (N, D)
            colors: Optional HDR colors of shape (N, 3)
            radii: Optional radii of shape (N,)
            sharpness: Optional sharpness of shape (N,)

        Raises:
            ValueError: If batch dimensions don't match expected
        """
        # Validate batch positions
        n_points, n_dims = validate_positions_for_writing(
            positions, context="batch positions"
        )

        if n_dims != self.expected_dims:
            raise ValueError(f"Expected {self.expected_dims}D points, got {n_dims}D")

        # Create datasets on first batch
        if self.positions_dataset is None:
            self._create_datasets(n_dims)

        # Resize datasets
        new_size = self.position + n_points
        self._resize_datasets(new_size)

        # Write batch data
        start = self.position
        end = new_size

        self.positions_dataset[start:end] = positions

        if colors is not None:
            validate_colors_for_writing(colors, n_points, context="batch colors")
            if self.colors_dataset is None:
                self._create_colors_dataset()
                # Resize the newly created dataset to match current size
                self.colors_dataset.resize((new_size, 3))
            self.colors_dataset[start:end] = colors
            self.has_colors = True

        if radii is not None:
            validate_radii_for_writing(radii, n_points, context="batch radii")
            if self.radii_dataset is None:
                self._create_radii_dataset()
                # Resize the newly created dataset to match current size
                self.radii_dataset.resize((new_size,))
            self.radii_dataset[start:end] = radii
            self.has_radii = True

        if sharpness is not None:
            validate_sharpness_for_writing(
                sharpness, n_points, context="batch sharpness"
            )
            if self.sharpness_dataset is None:
                self._create_sharpness_dataset()
                # Resize the newly created dataset to match current size
                self.sharpness_dataset.resize((new_size,))
            self.sharpness_dataset[start:end] = sharpness
            self.has_sharpness = True

        # Update position
        self.position = new_size
        self.total_points += n_points

        aprint(
            f"  ✓ Appended batch: {n_points:,} points (total: {self.total_points:,})"
        )

    def _create_datasets(self, n_dims: int) -> None:
        """Create the initial resizable datasets."""
        # Positions dataset
        self.positions_dataset = self.writer.create_resizable_dataset(
            f"{self.name}/positions",
            dtype=np.float32,
            shape=(0, n_dims),
            maxshape=(None, n_dims),
        )

    def _create_colors_dataset(self) -> None:
        """Create colors dataset when first needed."""
        self.colors_dataset = self.writer.create_resizable_dataset(
            f"{self.name}/colors",
            dtype=np.float32,
            shape=(0, 3),
            maxshape=(None, 3),
        )

    def _create_radii_dataset(self) -> None:
        """Create radii dataset when first needed."""
        self.radii_dataset = self.writer.create_resizable_dataset(
            f"{self.name}/radii",
            dtype=np.float32,
            shape=(0,),
            maxshape=(None,),
        )

    def _create_sharpness_dataset(self) -> None:
        """Create sharpness dataset when first needed."""
        self.sharpness_dataset = self.writer.create_resizable_dataset(
            f"{self.name}/sharpness",
            dtype=np.float32,
            shape=(0,),
            maxshape=(None,),
        )

    def _resize_datasets(self, new_size: int) -> None:
        """Resize all active datasets."""
        if self.positions_dataset is not None:
            self.positions_dataset.resize((new_size, self.expected_dims))

        if self.colors_dataset is not None:
            self.colors_dataset.resize((new_size, 3))

        if self.radii_dataset is not None:
            self.radii_dataset.resize((new_size,))

        if self.sharpness_dataset is not None:
            self.sharpness_dataset.resize((new_size,))

    def finalize(self, **attrs: Any) -> Dict[str, Any]:
        """Finalize the streaming points and write metadata.

        Args:
            **attrs: Additional attributes to store

        Returns:
            Metadata dictionary about the streamed points
        """
        # Write group metadata
        self.writer.write_group(
            self.name,
            type="points",
            n_points=self.total_points,
            streaming=True,
            has_colors=self.has_colors,
            has_radii=self.has_radii,
            has_sharpness=self.has_sharpness,
            **attrs,
        )

        metadata = {
            "n_points": self.total_points,
            "dims": self.expected_dims,
            "path": self.name,
            "has_colors": self.has_colors,
            "has_radii": self.has_radii,
            "has_sharpness": self.has_sharpness,
            "streaming": True,
        }

        aprint(
            f"✅ Finalized streaming points '{self.name}' with {self.total_points:,} points"
        )

        return metadata

    def append_from_generator(
        self,
        generator,
        batch_size: Optional[int] = None,
        max_batches: Optional[int] = None,
    ) -> int:
        """Convenience method to append from a generator.

        Args:
            generator: Generator yielding (positions, colors, radii, sharpness) tuples
            batch_size: If set, accumulate this many points before writing
            max_batches: Maximum number of batches to process

        Returns:
            Total number of points appended
        """
        batch_count = 0
        points_added = 0

        for data in generator:
            # Unpack data (generator can yield 1-4 items)
            if isinstance(data, tuple):
                positions = data[0]
                colors = data[1] if len(data) > 1 else None
                radii = data[2] if len(data) > 2 else None
                sharpness = data[3] if len(data) > 3 else None
            else:
                positions = data
                colors = radii = sharpness = None

            # Append batch
            self.append_batch(positions, colors, radii, sharpness)
            points_added += len(positions)
            batch_count += 1

            # Check max batches
            if max_batches is not None and batch_count >= max_batches:
                break

        return points_added
