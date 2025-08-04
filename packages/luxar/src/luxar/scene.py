from __future__ import annotations

import tempfile
from pathlib import Path
from typing import Any, Optional, Union

import numpy as np
import zarr
from arbol import aprint

from ._io import DEFAULT_COMP
from .config import (
    DEFAULT_UNITS,
    DEFAULT_VERSION,
    check_dataset_size_warning,
)
from .node import Node
from .points import Points
from .types import (
    ColorArray,
    CompressorProtocol,
    LuxarVersion,
    PathLike,
    PhysicalUnit,
    PositionArray,
    validate_physical_unit,
)


class Scene(Node):
    """
    Scene root – opens a Zarr store and exposes builder helpers.

    The Scene class represents the root of a Luxar scene hierarchy. It manages
    a Zarr store containing all scene data and provides high-level methods for
    building complex 3D scenes with point clouds and hierarchical grouping.

    Args:
        store_path: Path to the Zarr store to create or open. If None, uses temporary directory
        units: Physical units for the scene coordinates
        version: Luxar scene format version
        compressor: Compressor for Zarr datasets
    """

    def __init__(
        self,
        store_path: Optional[PathLike] = None,
        *,
        units: Union[PhysicalUnit, str] = DEFAULT_UNITS,
        version: Union[LuxarVersion, str] = DEFAULT_VERSION,
        compressor: Optional[CompressorProtocol] = DEFAULT_COMP,
    ) -> None:
        """
        Initialize a new Luxar scene.

        Args:
            store_path: Path to Zarr store, or None for temporary store
            units: Physical units for scene coordinates
            version: Luxar scene format version
            compressor: Compressor for Zarr datasets

        Raises:
            ValueError: If scene initialization fails
        """
        try:
            # Validate inputs
            validated_units = (
                validate_physical_unit(units) if isinstance(units, str) else units
            )

            # Handle store path
            resolved_store_path: PathLike
            if store_path is None:
                tmpdir = tempfile.TemporaryDirectory()
                resolved_store_path = Path(tmpdir.name) / "scene.zarr"
                self._tmpdir: Optional[tempfile.TemporaryDirectory[str]] = tmpdir
                aprint(
                    f"No store_path provided, using temporary directory: {resolved_store_path}"
                )
            else:
                self._tmpdir = None
                resolved_store_path = Path(store_path)
                aprint(f"Creating scene at store_path: {resolved_store_path}")

            # Create root Zarr group (format=2 for JavaScript compatibility)
            root = zarr.open_group(resolved_store_path, mode="w")
            root.attrs.update(
                {"luxar_version": version, "units": validated_units, "type": "scene"}
            )

            # Initialize parent Node
            super().__init__("Scene", root)

            # Store configuration
            self._compressor: Optional[CompressorProtocol] = compressor
            self._store_path: PathLike = resolved_store_path

            aprint(f"✓ Scene initialized successfully at {resolved_store_path}")

        except Exception as e:
            aprint(f"Failed to initialize Scene: {e}")
            raise ValueError(f"Could not initialize Scene: {e}") from e

    # ---------------------------------------------------------- builder helpers
    def add_group(self, name: str, **attrs: Any) -> Node:
        """
        Create and add a child group node to the scene.

        Args:
            name: Name of the group
            **attrs: Additional attributes for the group

        Returns:
            The created group node

        Raises:
            ValueError: If group creation fails
        """
        try:
            aprint(f"Adding group node '{name}'.")
            return super().add_group(name, **attrs)
        except Exception as e:
            aprint(f"Failed to add group node '{name}': {e}")
            raise ValueError(f"Could not add group '{name}': {e}") from e

    def add_points(
        self,
        name: str,
        positions: Union[PositionArray, np.ndarray[Any, Any]],
        colors: Optional[Union[ColorArray, np.ndarray[Any, Any]]] = None,
        radii: Optional[Union[np.ndarray[Any, np.dtype[np.float32]], np.ndarray[Any, Any]]] = None,
        sharpness: Optional[Union[np.ndarray[Any, np.dtype[np.float32]], np.ndarray[Any, Any]]] = None,
        parent: Optional[Node] = None,
        **attrs: Any,
    ) -> Points:
        """
        Add a point cloud node to the scene.

        Args:
            name: Name of the point cloud node
            positions: Array of shape (N, 3) for point positions
            colors: Optional array of shape (N, 3) for point colors
            radii: Optional array of shape (N,) for point radii
            sharpness: Optional array of shape (N,) for point edge sharpness
            parent: Parent node, defaults to scene root
            **attrs: Additional attributes for the node

        Returns:
            The created Points node

        Raises:
            ValueError: If point cloud creation fails
        """
        try:
            n_points = (
                positions.shape[0] if hasattr(positions, "shape") else len(positions)
            )
            aprint(f"Adding points node '{name}' with {n_points:,} points.")

            parent_node = parent or self
            return Points(
                name,
                positions,
                colors,
                radii,
                sharpness,
                parent=parent_node,
                compressor=self._compressor,
                **attrs,
            )
        except Exception as e:
            aprint(f"Failed to add points node '{name}': {e}")
            raise ValueError(f"Could not add points '{name}': {e}") from e

    def finalize(self) -> None:
        """
        Finalize the scene by consolidating Zarr metadata.

        This operation optimizes the Zarr store for reading by consolidating
        all metadata into a single file, improving load performance.

        Raises:
            ValueError: If metadata consolidation fails
        """
        try:
            aprint("Finalizing scene and consolidating Zarr metadata.")
            zarr.consolidate_metadata(self._group.store)
            aprint("✓ Scene finalized successfully.")
        except Exception as e:
            aprint(f"Failed to finalize scene: {e}")
            raise ValueError(f"Could not finalize scene: {e}") from e

    # ---------------------------------------------------------- convenience API
    @classmethod
    def random_demo(
        cls, store: PathLike, n: int = 10_000, seed: Optional[int] = None
    ) -> Scene:
        """
        Create a demo scene with a Lorenz attractor visualization.

        This creates a beautiful butterfly-shaped 3D structure with colors
        that transition smoothly over time, demonstrating the Luxar scene format
        with an aesthetically pleasing mathematical visualization.

        Args:
            store: Path to the Zarr store to create
            n: Number of points to generate along the attractor
            seed: Random seed for reproducible results

        Returns:
            The created demo scene

        Raises:
            ValueError: If demo scene creation fails
        """
        try:
            aprint(f"Creating Lorenz attractor demo scene with {n:,} points.")

            # Check for performance warnings
            warning = check_dataset_size_warning(n)
            if warning:
                aprint(f"Performance warning: {warning}")

            # Lorenz attractor parameters
            sigma = 10.0
            rho = 28.0
            beta = 8.0 / 3.0
            dt = 0.01

            # Initialize arrays
            positions = np.zeros((n, 3), dtype=np.float32)

            # Starting point (with small random perturbation if seed is provided)
            rng = np.random.default_rng(seed)
            x, y, z = 0.1, 0.0, 0.0
            if seed is not None:
                x += rng.uniform(-0.01, 0.01)

            # Generate Lorenz attractor points
            for i in range(n):
                # Lorenz equations
                dx = sigma * (y - x) * dt
                dy = (x * (rho - z) - y) * dt
                dz = (x * y - beta * z) * dt

                x += dx
                y += dy
                z += dz

                positions[i] = [x, y, z]

            # Scale positions to fit nicely in view
            positions *= 0.1

            # Center the attractor at its center of mass
            center_of_mass = np.mean(positions, axis=0)
            positions -= center_of_mass

            # Create time-based colors with smooth transitions
            # Using HSV color space for smooth color transitions
            t = np.linspace(0, 1, n)
            hue = (t * 2) % 1.0  # Cycle through hues twice

            # Convert HSV to RGB
            colors = np.zeros((n, 3), dtype=np.uint8)
            for i in range(n):
                h = hue[i]
                # Full saturation and value for vibrant colors
                s, v = 1.0, 1.0

                # HSV to RGB conversion
                c = v * s
                x = c * (1 - abs((h * 6) % 2 - 1))
                m = v - c

                h_i = int(h * 6)
                if h_i == 0:
                    r, g, b = c, x, 0.0
                elif h_i == 1:
                    r, g, b = x, c, 0.0
                elif h_i == 2:
                    r, g, b = 0.0, c, x
                elif h_i == 3:
                    r, g, b = 0.0, x, c
                elif h_i == 4:
                    r, g, b = x, 0.0, c
                else:
                    r, g, b = c, 0.0, x

                colors[i] = ((r + m) * 255, (g + m) * 255, (b + m) * 255)

            # Generate radii based on position in the trajectory (growing over time)
            # This creates a visual effect of the attractor "growing" as it evolves
            radii = np.linspace(0.05, 0.2, n).astype(np.float32)
            
            # Create scene and add data
            scene = cls(store)
            scene.add_points("LorenzAttractor", positions, colors, radii=radii)
            scene.finalize()

            aprint(
                f"✓ Lorenz attractor demo scene created at {scene.get_store_path()}."
            )
            return scene

        except Exception as e:
            aprint(f"Failed to create demo scene: {e}")
            raise ValueError(f"Could not create demo scene: {e}") from e

    def get_store_path(self) -> PathLike:
        """
        Get the path to the backing Zarr store.

        Returns:
            Path to the Zarr store backing this scene
        """
        return self._store_path

    def to_zarr(self, path: PathLike) -> None:
        """
        Export scene to a new Zarr store location.

        Args:
            path: Destination path for the Zarr store

        Raises:
            NotImplementedError: Scene export is not yet implemented
        """
        aprint(f"Exporting scene to {path}")
        # This would require copying the entire Zarr store
        # Implementation depends on zarr library capabilities
        raise NotImplementedError("Scene export not yet implemented")
