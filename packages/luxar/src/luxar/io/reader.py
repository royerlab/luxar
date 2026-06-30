"""luxar.io.reader – Reading and loading Luxar zarr scenes.

This module provides the LuxarScene class for reading Luxar zarr format files.
It supports automatic decoding of encoded arrays via ArrayDecoder.
"""

import warnings
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Iterator, List, Optional, Union

import numpy as np
import zarr
from numcodecs import Blosc

from ..core.dimensions import Dimensions
from ..core.transforms import read_transform_from_zarr
from ..encoding.decoder import ArrayDecoder
from ..typing_utils.constants import LUXAR_VERSION_CURRENT

# Default compressor: fast, bit‑shuffle‑friendly
DEFAULT_COMP = Blosc(cname="zstd", clevel=3, shuffle=Blosc.BITSHUFFLE)


_SENTINEL = object()


class _DictCompatMixin:
    """Mixin that provides dict-style access for backward compatibility."""

    def __getitem__(self, key: str) -> Any:
        try:
            return getattr(self, key)
        except AttributeError:
            raise KeyError(key) from None

    def get(self, key: str, default: Any = None) -> Any:
        """Dict-compatible .get() with default."""
        val = getattr(self, key, _SENTINEL)
        return default if val is _SENTINEL else val

    def keys(self) -> Iterator[str]:
        return iter(f.name for f in self.__dataclass_fields__.values())  # type: ignore[attr-defined]

    def values(self) -> Iterator[Any]:
        return iter(getattr(self, f.name) for f in self.__dataclass_fields__.values())  # type: ignore[attr-defined]

    def items(self) -> Iterator[tuple[str, Any]]:
        return iter(
            (f.name, getattr(self, f.name))
            for f in self.__dataclass_fields__.values()  # type: ignore[attr-defined]
        )

    def __iter__(self) -> Iterator[str]:
        return self.keys()

    def __len__(self) -> int:
        return len(self.__dataclass_fields__)  # type: ignore[attr-defined]

    def __contains__(self, key: str) -> bool:
        return key in self.__dataclass_fields__  # type: ignore[attr-defined]


@dataclass(frozen=True)
class PointsData(_DictCompatMixin):
    """Structured result from get_points()."""

    positions: np.ndarray
    colors: Optional[np.ndarray]
    radii: Optional[np.ndarray]
    sharpness: Optional[np.ndarray]
    chunk_bounds: Optional[np.ndarray]
    metadata: Dict[str, Any]


@dataclass(frozen=True)
class LinesData(_DictCompatMixin):
    """Structured result from get_lines()."""

    vertices: np.ndarray
    widths: np.ndarray
    colors: Optional[np.ndarray]
    sharpness: Optional[np.ndarray]
    segments: Optional[np.ndarray]
    metadata: Dict[str, Any]
    indices: Optional[np.ndarray] = None


@dataclass(frozen=True)
class GSplatsData(_DictCompatMixin):
    """Structured result from get_gsplats()."""

    centers: np.ndarray
    amplitudes: np.ndarray
    cholesky_factors: np.ndarray
    colors: Optional[np.ndarray]
    chunk_bounds: Optional[np.ndarray]
    metadata: Dict[str, Any]


class LuxarScene:
    """Read-only access to a Luxar zarr scene.

    Provides Python API to read and validate Luxar scene files. Enables:
    - Round-trip testing (write → read → verify)
    - Python-based scene analysis and inspection
    - Format validation and debugging
    - Data extraction for processing

    Example:
        >>> scene = LuxarScene.load('scene.zarr')
        >>> aprint(f"Version: {scene.version}")
        >>> aprint(f"Nodes: {[n['name'] for n in scene.nodes]}")
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

        # Warn on version mismatch (non-fatal: older files should still load)
        file_version = root.attrs.get("luxar_version")
        if file_version is not None and str(file_version) != LUXAR_VERSION_CURRENT:
            warnings.warn(
                f"Version mismatch: file='{file_version}', "
                f"library='{LUXAR_VERSION_CURRENT}'.",
                UserWarning,
                stacklevel=2,
            )

        return cls(root, path)

    @property
    def path(self) -> Path:
        """Path to the zarr store."""
        return self._path

    @property
    def version(self) -> str:
        """Luxar format version."""
        return str(self._root.attrs.get("luxar_version", "unknown"))

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
        nodes: List[Dict[str, Any]] = []
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

            # Include transform if present (all node types can have transforms)
            if "transform" in child.attrs:
                info["transform"] = read_transform_from_zarr(child.attrs["transform"])

            # Include nd_transform if present (dict, no conversion needed)
            if "nd_transform" in child.attrs:
                info["nd_transform"] = child.attrs["nd_transform"]

            if node_type == "points":
                info["n_points"] = child.attrs.get("n_points", 0)
                info["ordering"] = child.attrs.get("ordering", "none")
                # Check if arrays exist
                info["has_colors"] = "colors" in child
                info["has_radii"] = "radii" in child
                info["has_sharpness"] = "sharpnesses" in child
            elif node_type == "gsplats":
                info["n_splats"] = child.attrs.get("n_splats", 0)
                info["ndim"] = child.attrs.get("ndim", 3)
                info["has_colors"] = child.attrs.get("has_colors", False)
            elif node_type == "lines":
                info["n_vertices"] = child.attrs.get("n_vertices", 0)
                info["n_segments"] = child.attrs.get("n_segments", 0)
                info["line_type"] = child.attrs.get(
                    "original_line_type",
                    child.attrs.get("line_type", "segments"),
                )
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
        return str(self._root[name].attrs.get("type", "group"))

    def get_node_metadata(self, name: str) -> Dict[str, Any]:
        """Get metadata for a node (no data arrays).

        Note: transforms are returned as raw column-major lists.
        Use get_group() for automatic transform conversion.
        """
        if not self.has_node(name):
            raise KeyError(f"Node not found: {name}")
        return dict(self._root[name].attrs)

    def get_group(self, name: str) -> Dict[str, Any]:
        """Get group node metadata with parsed transform.

        Args:
            name: Name of the group node

        Returns:
            Dictionary with group metadata. Transform (if present) is
            converted from column-major list to a 4x4 NumPy matrix.

        Raises:
            KeyError: If node doesn't exist
            ValueError: If node is not a group node
        """
        if not self.has_node(name):
            raise KeyError(f"Group node not found: {name}")

        group = self._root[name]
        if group.attrs.get("type") not in ("group", None):
            raise ValueError(f"Node '{name}' is not a group node")

        metadata = dict(group.attrs)

        # Parse transform if present
        if "transform" in metadata:
            metadata["transform"] = read_transform_from_zarr(metadata["transform"])

        return metadata

    def get_points(self, name: str) -> PointsData:
        """Load points node data with automatic decoding.

        Args:
            name: Name of the points node

        Returns:
            PointsData with fields: positions, colors, radii, sharpness,
            chunk_bounds, metadata.  Supports dict-style access for
            backward compatibility (e.g. ``data["positions"]``).

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
        if positions is None:
            raise ValueError(f"Points node '{name}' missing required 'positions' array")
        colors = self._decode_array(group, "colors")
        radii = self._decode_array(group, "radii")
        sharpness = self._decode_array(group, "sharpnesses")

        # Load chunk bounds if present
        chunk_bounds = None
        if "chunk_bounds" in group:
            chunk_bounds = group["chunk_bounds"][:]

        # Build metadata
        metadata = dict(group.attrs)

        # Parse transform if present
        if "transform" in metadata:
            metadata["transform"] = read_transform_from_zarr(metadata["transform"])

        return PointsData(
            positions=positions,
            colors=colors,
            radii=radii,
            sharpness=sharpness,
            chunk_bounds=chunk_bounds,
            metadata=metadata,
        )

    def get_gsplats(self, name: str) -> GSplatsData:
        """Load gsplats node data with automatic decoding.

        Args:
            name: Name of the gsplats node

        Returns:
            GSplatsData with fields: centers, amplitudes, cholesky_factors,
            colors, chunk_bounds, metadata.  Supports dict-style access for
            backward compatibility.

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
        if centers is None:
            raise ValueError(f"GSplats node '{name}' missing required 'centers' array")
        amplitudes = self._decode_array(group, "amplitudes")
        if amplitudes is None:
            raise ValueError(
                f"GSplats node '{name}' missing required 'amplitudes' array"
            )
        cholesky_factors = self._decode_cholesky(group)
        if cholesky_factors is None:
            raise ValueError(
                f"GSplats node '{name}' missing required cholesky factors "
                f"('cholesky_factors_diag' for v3.1, or 'cholesky_factors' for v3.0)"
            )
        colors = self._decode_array(group, "colors")

        # Load chunk bounds if present
        chunk_bounds = None
        if "chunk_bounds" in group:
            chunk_bounds = group["chunk_bounds"][:]

        # Build metadata
        metadata = dict(group.attrs)

        # Parse transform if present
        if "transform" in metadata:
            metadata["transform"] = read_transform_from_zarr(metadata["transform"])

        return GSplatsData(
            centers=centers,
            amplitudes=amplitudes,
            cholesky_factors=cholesky_factors,
            colors=colors,
            chunk_bounds=chunk_bounds,
            metadata=metadata,
        )

    def get_lines(self, name: str) -> LinesData:
        """Load lines node data with automatic decoding.

        Args:
            name: Name of the lines node

        Returns:
            LinesData with fields: vertices, widths, colors, sharpness,
            segments, metadata.  Supports dict-style access for backward
            compatibility.

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
        if vertices is None:
            raise ValueError(f"Lines node '{name}' missing required 'vertices' array")
        widths = self._decode_array(group, "widths")
        if widths is None:
            raise ValueError(f"Lines node '{name}' missing required 'widths' array")
        colors = self._decode_array(group, "colors")
        sharpness = self._decode_array(group, "sharpnesses")
        segments = self._decode_array(group, "segments")
        if segments is None:
            segments = self._decode_array(group, "indices")
            if segments is not None and segments.ndim == 1 and len(segments) % 2 == 0:
                segments = segments.reshape(-1, 2)

        # Build metadata
        metadata = dict(group.attrs)

        # Parse transform if present
        if "transform" in metadata:
            metadata["transform"] = read_transform_from_zarr(metadata["transform"])

        return LinesData(
            vertices=vertices,
            widths=widths,
            colors=colors,
            sharpness=sharpness,
            segments=segments,
            indices=segments,
            metadata=metadata,
        )

    def _decode_array(self, group: zarr.Group, array_name: str) -> Optional[np.ndarray]:
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

    def _decode_cholesky(self, group: zarr.Group) -> Optional[np.ndarray]:
        """Decode Cholesky factors, recombining the v3.1 split layout.

        Delegates the layout/version handling to the shared
        :func:`luxar.gsplats.utils.trils.recombine_cholesky` so the v3.0
        fallback, the corruption invariant, and the error message stay in one
        place (the gsplat-tree decoder uses the same helper). ``_decode_array``
        already returns ``None`` for an absent array, matching the callback
        contract.
        """
        from ..gsplats.utils.trils import recombine_cholesky

        return recombine_cholesky(lambda name: self._decode_array(group, name))
