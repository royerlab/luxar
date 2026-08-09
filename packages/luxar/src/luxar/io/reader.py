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

from ..core.dimensions import Dimensions
from ..core.transforms import read_transform_from_zarr
from ..core.viewer_config import ViewerConfig
from ..encoding.compression import WIDTH_AWARE_DEFAULT
from ..encoding.decoder import ArrayDecoder
from ..typing_utils.constants import LUXAR_VERSION_CURRENT

# Default compressor: the width-aware policy sentinel — each array gets a
# zstd-l9 configuration keyed on its stored dtype (byte shuffle for multi-byte
# integer codes, no shuffle for uint8/floats), resolved at write time by
# luxar.encoding.compression.resolve_compressor. See the manuscript
# supplementary ``codec_selection`` for the measurements behind the policy.
DEFAULT_COMP = WIDTH_AWARE_DEFAULT


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
class MeshData(_DictCompatMixin):
    """Structured result from get_mesh().

    ``normal_dims`` is surfaced as its own field rather than left in ``metadata``
    because it is not optional context — a normals array cannot be oriented
    without it, so any consumer reading ``normals`` must read this too.
    """

    vertices: np.ndarray
    faces: np.ndarray
    normals: Optional[np.ndarray]
    normal_dims: Optional[List[int]]
    colors: Optional[np.ndarray]
    scalars: Optional[np.ndarray]
    metadata: Dict[str, Any]


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
            ValueError: If not a valid Luxar scene, or if the store is marked
                ``incomplete`` (the writer exited with an error before
                finalizing, so nodes/metadata may be missing).
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

        # Reject a store the writer marked incomplete. The root type='scene'
        # attr is written before any node, so a with-block that raised leaves
        # a partial store that otherwise looks valid; the compiler stamps
        # ``incomplete`` on error instead of finalizing.
        if root.attrs.get("incomplete"):
            raise ValueError(
                f"Scene at {path} is incomplete: the writer exited with an "
                "error before finalizing, so the store may be missing nodes "
                "or metadata. Rebuild the scene from scratch."
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
    def viewer_config(self) -> Optional[ViewerConfig]:
        """Scene viewer configuration if defined.

        The read-only mirror of :attr:`luxar.core.scene.Scene.viewer_config`, and
        parsed the same way — the attr is a plain dict on disk, so a consumer
        that means to CARRY the config into a new scene (``luxar mesh lod``
        rewriting a ladder, say) needs the object, not the dict. Losing it is not
        cosmetic: a scene that dropped ``tone_mapping`` silently falls back to
        the viewer's ACES default, which shifts the hues of a custom colormap
        LUT.

        Returns:
            The parsed :class:`~luxar.core.viewer_config.ViewerConfig`, or
            ``None`` when the scene declares none.
        """
        vc_dict = self._root.attrs.get("viewer_config")
        if vc_dict is None:
            return None
        return ViewerConfig.from_dict(vc_dict)

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
            elif node_type == "mesh":
                info["n_vertices"] = child.attrs.get("n_vertices", 0)
                info["n_faces"] = child.attrs.get("n_faces", 0)
                info["ndim"] = child.attrs.get("ndim", 3)
                info["has_normals"] = child.attrs.get("has_normals", False)
                info["shading"] = child.attrs.get("shading", "flat")
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

    def list_meshes(self) -> List[str]:
        """Names of all mesh nodes."""
        return [n["name"] for n in self.nodes if n["type"] == "mesh"]

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

    def get_colormap_lut(self, name: str) -> Optional[np.ndarray]:
        """Get a node's custom colormap LUT, or ``None`` if it has none.

        A node whose ``colormap`` attr is the sentinel ``"custom"`` carries the
        actual colors in a ``colormap_lut`` dataset — the writer resolves any
        colormap that is not one of the builtin names (an ndarray LUT, but also
        a plain matplotlib/colorcet name) to that pair, so the viewer needs
        neither library at display time.

        Returned raw, not through ``_decode_array``: the LUT is written as a
        plain unencoded ``(256, 3)`` uint8 dataset, so there is no encoding
        header for the decoder to read.

        Args:
            name: Name of the node

        Returns:
            The ``(256, 3)`` uint8 LUT, or ``None`` when the node has no
            ``colormap_lut`` dataset (a builtin or absent colormap).

        Raises:
            KeyError: If node doesn't exist
        """
        if not self.has_node(name):
            raise KeyError(f"Node not found: {name}")
        group = self._root[name]
        if "colormap_lut" not in group:
            return None
        return np.asarray(group["colormap_lut"][:])

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

    def get_mesh(self, name: str) -> MeshData:
        """Load mesh node data with automatic decoding.

        Args:
            name: Name of the mesh node

        Returns:
            MeshData with fields: vertices, faces, normals, normal_dims, colors,
            scalars, metadata. Supports dict-style access for backward
            compatibility.

        Raises:
            KeyError: If node doesn't exist
            ValueError: If node is not a mesh node, or is missing a required array
        """
        if not self.has_node(name):
            raise KeyError(f"Mesh node not found: {name}")

        group = self._root[name]
        if group.attrs.get("type") != "mesh":
            raise ValueError(f"Node '{name}' is not a mesh node")

        vertices = self._decode_array(group, "vertices")
        if vertices is None:
            raise ValueError(f"Mesh node '{name}' missing required 'vertices' array")
        faces = self._decode_array(group, "faces")
        if faces is None:
            raise ValueError(f"Mesh node '{name}' missing required 'faces' array")
        # Normalize to (F, 3) so a consumer never has to guess the layout. The
        # writer always emits (F, 3), but an externally produced store may store
        # the flat form the write-side validator also accepts.
        if faces.ndim == 1:
            faces = faces.reshape(-1, 3)

        normals = self._decode_array(group, "normals")
        colors = self._decode_array(group, "colors")
        scalars = self._decode_array(group, "scalars")

        metadata = dict(group.attrs)

        if "transform" in metadata:
            metadata["transform"] = read_transform_from_zarr(metadata["transform"])

        # Read normal_dims only when normals are actually present. A stray attr
        # without the array would otherwise be handed on as if it oriented
        # something.
        raw_dims = metadata.get("normal_dims")
        normal_dims = (
            [int(d) for d in raw_dims]
            if normals is not None and raw_dims is not None
            else None
        )

        # Two consistency checks on the normals pair, because ``MeshData`` declares
        # shapes its consumers rely on: ``normal_dims`` is a dimension TRIPLE, and
        # ``normals`` is one 3-vector per vertex. Handing back a 2-entry triple or a
        # short array satisfies the type annotation while breaking the first consumer
        # that indexes ``[2]`` or zips against the vertices — a corrupt-store failure
        # showing up far from its cause.
        #
        # Deliberately NOT checked here: whether every face index is within range.
        # That needs a full scan of F indices on every read, and the viewer's loader
        # already owns it as an admission gate (MESH_NODE_SPEC.md §3.5 Stage 2)
        # precisely because an out-of-range index can trap its kernels. Python
        # consumers index numpy, which raises on its own.
        if normal_dims is not None and len(normal_dims) != 3:
            raise ValueError(
                f"Mesh node '{name}' has a malformed 'normal_dims' attr "
                f"{normal_dims!r}: expected exactly 3 dimension indices naming the "
                f"dimensions the (V, 3) normals describe."
            )
        if normals is not None and normals.shape[0] != vertices.shape[0]:
            raise ValueError(
                f"Mesh node '{name}' has {normals.shape[0]} normals for "
                f"{vertices.shape[0]} vertices — normals are per-vertex, so the "
                f"counts must match."
            )

        return MeshData(
            vertices=vertices,
            faces=faces,
            normals=normals,
            normal_dims=normal_dims,
            colors=colors,
            scalars=scalars,
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
