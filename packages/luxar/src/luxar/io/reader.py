"""luxar.io.reader – Reading and loading Luxar zarr scenes.

This module provides the LuxarScene class for reading Luxar zarr format files.
It supports automatic decoding of encoded arrays via ArrayDecoder.
"""

from pathlib import Path
from typing import Any, Dict, List, Optional, Union

import numpy as np
import zarr
from numcodecs import Blosc

from ..core.dimensions import Dimensions
from ..core.transforms import read_transform_from_zarr
from ..encoding.decoder import ArrayDecoder

# Default compressor: fast, bit‑shuffle‑friendly
DEFAULT_COMP = Blosc(cname="zstd", clevel=3, shuffle=Blosc.BITSHUFFLE)


class LuxarScene:
    """Read-only access to a Luxar zarr scene.

    Provides Python API to read and validate Luxar scene files. Enables:
    - Round-trip testing (write → read → verify)
    - Python-based scene analysis and inspection
    - Format validation and debugging
    - Data extraction for processing

    Example:
        >>> scene = LuxarScene.load('scene.zarr')
        >>> print(f"Version: {scene.version}")
        >>> print(f"Nodes: {[n['name'] for n in scene.nodes]}")
        >>> points = scene.get_points('my_cloud')
        >>> positions = points['positions']  # Decoded array
    """

    def __init__(self, root: zarr.Group, path: Path):
        """Initialize from an open zarr group.

        Args:
            root: Open zarr group (root of scene)
            path: Path to the zarr store
        """
        self._root = root
        self._path = path
        self._decoder = ArrayDecoder()

    @classmethod
    def load(cls, path: Union[str, Path]) -> "LuxarScene":
        """Load a Luxar scene from a zarr store.

        Args:
            path: Path to the zarr store

        Returns:
            LuxarScene instance

        Raises:
            FileNotFoundError: If path doesn't exist
            ValueError: If not a valid Luxar scene
        """
        path = Path(path)
        if not path.exists():
            raise FileNotFoundError(f"Scene not found: {path}")

        root = zarr.open_group(path, mode="r")

        # Validate it's a Luxar scene
        node_type = root.attrs.get("type")
        if node_type != "scene":
            raise ValueError(
                f"Not a valid Luxar scene: expected type='scene', got '{node_type}'"
            )

        return cls(root, path)

    @property
    def path(self) -> Path:
        """Path to the zarr store."""
        return self._path

    @property
    def version(self) -> str:
        """Luxar format version."""
        return self._root.attrs.get("luxar_version", "unknown")

    @property
    def root_attrs(self) -> Dict[str, Any]:
        """All root attributes."""
        return dict(self._root.attrs)

    @property
    def dimensions(self) -> Optional[Dimensions]:
        """Scene dimensions if defined."""
        dims_data = self._root.attrs.get("scene_dimensions")
        if dims_data is None:
            return None
        return Dimensions.from_dict(dims_data)

    @property
    def nodes(self) -> List[Dict[str, Any]]:
        """List all nodes with metadata.

        Returns list of dicts with node info:
        - name: Node name (path from root)
        - type: 'points', 'gsplats', 'lines', or 'group'
        - Additional metadata depending on type
        """
        nodes = []
        self._collect_nodes(self._root, "", nodes)
        return nodes

    def _collect_nodes(
        self, group: zarr.Group, prefix: str, nodes: List[Dict[str, Any]]
    ) -> None:
        """Recursively collect node information."""
        for name in group.group_keys():
            child = group[name]
            full_name = f"{prefix}/{name}" if prefix else name
            node_type = child.attrs.get("type", "group")

            info: Dict[str, Any] = {
                "name": full_name,
                "type": node_type,
            }

            if node_type == "points":
                info["n_points"] = child.attrs.get("n_points", 0)
                info["ordering"] = child.attrs.get("ordering", "none")
                # Check if arrays exist
                info["has_colors"] = "colors" in child
                info["has_radii"] = "radii" in child
                info["has_sharpness"] = "sharpness" in child
            elif node_type == "gsplats":
                info["n_splats"] = child.attrs.get("n_splats", 0)
                info["ndim"] = child.attrs.get("ndim", 3)
                info["has_colors"] = child.attrs.get("has_colors", False)
                info["has_sharpness"] = child.attrs.get("has_sharpness", False)
            elif node_type == "lines":
                info["n_vertices"] = child.attrs.get("n_vertices", 0)
                info["n_segments"] = child.attrs.get("n_segments", 0)
                info["line_type"] = child.attrs.get("line_type", "segments")
            elif node_type == "group":
                # Recursively collect children
                self._collect_nodes(child, full_name, nodes)

            nodes.append(info)

    def list_points(self) -> List[str]:
        """Names of all points nodes."""
        return [n["name"] for n in self.nodes if n["type"] == "points"]

    def list_gsplats(self) -> List[str]:
        """Names of all gsplats nodes."""
        return [n["name"] for n in self.nodes if n["type"] == "gsplats"]

    def list_lines(self) -> List[str]:
        """Names of all lines nodes."""
        return [n["name"] for n in self.nodes if n["type"] == "lines"]

    def list_groups(self) -> List[str]:
        """Names of all group nodes."""
        return [n["name"] for n in self.nodes if n["type"] == "group"]

    def has_node(self, name: str) -> bool:
        """Check if a node exists."""
        try:
            _ = self._root[name]
            return True
        except KeyError:
            return False

    def get_node_type(self, name: str) -> str:
        """Get the type of a node."""
        if not self.has_node(name):
            raise KeyError(f"Node not found: {name}")
        return self._root[name].attrs.get("type", "group")

    def get_node_metadata(self, name: str) -> Dict[str, Any]:
        """Get metadata for a node (no data arrays)."""
        if not self.has_node(name):
            raise KeyError(f"Node not found: {name}")
        return dict(self._root[name].attrs)

    def get_points(self, name: str) -> Dict[str, Any]:
        """Load points node data with automatic decoding.

        Args:
            name: Name of the points node

        Returns:
            Dict with:
            - positions: (N, D) float32 array
            - colors: (N, 3) float32 array or None
            - radii: (N,) float32 array or None
            - sharpness: (N,) float32 array or None
            - chunk_bounds: (num_chunks, D, 2) array or None
            - metadata: dict with node attributes

        Raises:
            KeyError: If node doesn't exist
            ValueError: If node is not a points node
        """
        if not self.has_node(name):
            raise KeyError(f"Points node not found: {name}")

        group = self._root[name]
        if group.attrs.get("type") != "points":
            raise ValueError(f"Node '{name}' is not a points node")

        # Decode arrays
        positions = self._decode_array(group, "positions")
        colors = self._decode_array(group, "colors")
        radii = self._decode_array(group, "radii")
        sharpness = self._decode_array(group, "sharpness")

        # Load chunk bounds if present
        chunk_bounds = None
        if "chunk_bounds" in group:
            chunk_bounds = group["chunk_bounds"][:]

        # Build metadata
        metadata = dict(group.attrs)

        # Parse transform if present
        if "transform" in metadata:
            metadata["transform"] = read_transform_from_zarr(metadata["transform"])

        return {
            "positions": positions,
            "colors": colors,
            "radii": radii,
            "sharpness": sharpness,
            "chunk_bounds": chunk_bounds,
            "metadata": metadata,
        }

    def get_gsplats(self, name: str) -> Dict[str, Any]:
        """Load gsplats node data with automatic decoding.

        Args:
            name: Name of the gsplats node

        Returns:
            Dict with:
            - centers: (N, D) float32 array
            - amplitudes: (N,) float32 array
            - cholesky_factors: (N, k) float32 array
            - colors: (N, 3) float32 array or None
            - sharpness: (N,) float32 array or None
            - chunk_bounds: (num_chunks, D, 2) array or None
            - metadata: dict with node attributes

        Raises:
            KeyError: If node doesn't exist
            ValueError: If node is not a gsplats node
        """
        if not self.has_node(name):
            raise KeyError(f"GSplats node not found: {name}")

        group = self._root[name]
        if group.attrs.get("type") != "gsplats":
            raise ValueError(f"Node '{name}' is not a gsplats node")

        # Decode arrays
        centers = self._decode_array(group, "centers")
        amplitudes = self._decode_array(group, "amplitudes")
        cholesky_factors = self._decode_array(group, "cholesky_factors")
        colors = self._decode_array(group, "colors")
        sharpness = self._decode_array(group, "sharpness")

        # Load chunk bounds if present
        chunk_bounds = None
        if "chunk_bounds" in group:
            chunk_bounds = group["chunk_bounds"][:]

        # Build metadata
        metadata = dict(group.attrs)

        # Parse transform if present
        if "transform" in metadata:
            metadata["transform"] = read_transform_from_zarr(metadata["transform"])

        return {
            "centers": centers,
            "amplitudes": amplitudes,
            "cholesky_factors": cholesky_factors,
            "colors": colors,
            "sharpness": sharpness,
            "chunk_bounds": chunk_bounds,
            "metadata": metadata,
        }

    def get_lines(self, name: str) -> Dict[str, Any]:
        """Load lines node data with automatic decoding.

        Args:
            name: Name of the lines node

        Returns:
            Dict with:
            - vertices: (N, D) float32 array
            - widths: (N,) float32 array
            - colors: (N, 3) float32 array or None
            - sharpness: (N,) float32 array or None
            - indices: (M,) uint32 array or None
            - metadata: dict with node attributes

        Raises:
            KeyError: If node doesn't exist
            ValueError: If node is not a lines node
        """
        if not self.has_node(name):
            raise KeyError(f"Lines node not found: {name}")

        group = self._root[name]
        if group.attrs.get("type") != "lines":
            raise ValueError(f"Node '{name}' is not a lines node")

        # Decode arrays
        vertices = self._decode_array(group, "vertices")
        widths = self._decode_array(group, "widths")
        colors = self._decode_array(group, "colors")
        sharpness = self._decode_array(group, "sharpness")
        indices = self._decode_array(group, "indices")

        # Build metadata
        metadata = dict(group.attrs)

        # Parse transform if present
        if "transform" in metadata:
            metadata["transform"] = read_transform_from_zarr(metadata["transform"])

        return {
            "vertices": vertices,
            "widths": widths,
            "colors": colors,
            "sharpness": sharpness,
            "indices": indices,
            "metadata": metadata,
        }

    def _decode_array(
        self, group: zarr.Group, array_name: str
    ) -> Optional[np.ndarray]:
        """Decode an array from a group if it exists.

        Args:
            group: Zarr group containing the array
            array_name: Name of the array

        Returns:
            Decoded numpy array, or None if array doesn't exist
        """
        if array_name not in group:
            return None

        zarr_array = group[array_name]
        return self._decoder.decode(zarr_array, self._root)
