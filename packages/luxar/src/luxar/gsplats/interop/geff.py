"""Read **GEFF** tracking graphs (the cell-lineage exchange format).

`GEFF <https://github.com/live-image-tracking-tools/geff>`_ (Graph Exchange File
Format) is the on-disk interchange format for cell-tracking graphs used across
the live-image-tracking ecosystem — ``ultrack``, ``trackedit``, ``tracksdata``,
``traccuracy`` — and the format the Biohub *Cell Tracking During Development*
challenge ships its ground truth in. A GEFF store is a **zarr v3** group::

    <name>.geff/
      zarr.json                     # {"attributes": {"geff": {axes, ...}}}
      nodes/ids                     # (N,)   node ids
      nodes/props/{t,z,y,x}/values  # (N,)   coordinates, in VOXEL units
      edges/ids                     # (E, 2) (source_id, target_id) pairs
      edges/props/...

This module reads that into a plain-NumPy :class:`TrackingGraph` — no zarr 3, no
``geff`` package, no ``networkx``. Luxar pins ``zarr>=2.16,<3.0``, so the read
goes through :mod:`luxar.io.zarr_v3`.

Beyond reading, :class:`TrackingGraph` does the graph work a visualisation
actually needs: physical-unit positions, connected-component **lineage ids** for
colouring, division detection, and edges remapped to positional indices so the
lineage forest can go straight into ``scene.add_lines(line_type="indexed")``.

See also :mod:`luxar.gsplats.interop.tracksdata`, which goes the other way —
fitted Gaussian splats *into* a ``tracksdata`` graph for linking and proofreading.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Optional, Sequence, Tuple, Union

import numpy as np

__all__ = [
    "TrackingGraph",
    "read_geff",
]

# GEFF spatial axes, coarsest-to-finest, matching Luxar's ZYX ordering for
# volumetric data. "t" is handled separately (it is not a spatial axis).
_SPATIAL = ("z", "y", "x")


@dataclass
class TrackingGraph:
    """A cell-tracking lineage graph in plain NumPy arrays."""

    node_ids: np.ndarray
    """``(N,)`` node identifiers as stored.

    Often *not* contiguous — the Biohub challenge encodes
    ``global_t * 1e9 + cell_id``. Positional indices into the coordinate arrays
    are what the rest of this class works in; :meth:`index_of` maps ids to them.
    """

    t: np.ndarray
    """``(N,)`` integer timepoint per node."""

    positions: np.ndarray
    """``(N, 3)`` node coordinates in **voxel** units, ordered ``(z, y, x)``."""

    edges: np.ndarray
    """``(E, 2)`` directed ``(source_id, target_id)`` pairs, in *node id* space.

    Cell-tracking edges point forward in time.
    """

    scale: Tuple[float, float, float] = (1.0, 1.0, 1.0)
    """Voxel size ``(z, y, x)`` from the GEFF axis metadata (see :meth:`positions_um`)."""

    units: Tuple[Optional[str], ...] = (None, None, None)
    """Physical unit per spatial axis, when the store declares one."""

    # -- basics -----------------------------------------------------------

    @property
    def n_nodes(self) -> int:
        return int(self.node_ids.shape[0])

    @property
    def n_edges(self) -> int:
        return int(self.edges.shape[0])

    @property
    def n_timepoints(self) -> int:
        """One past the largest timepoint index (0 for an empty graph)."""
        return int(self.t.max()) + 1 if self.n_nodes else 0

    def __repr__(self) -> str:
        span = (
            f"t={int(self.t.min())}..{int(self.t.max())}" if self.n_nodes else "empty"
        )
        n_lineages = len(set(self.lineage_ids().tolist())) if self.n_nodes else 0
        return (
            f"TrackingGraph(nodes={self.n_nodes}, edges={self.n_edges}, {span}, "
            f"lineages={n_lineages}, divisions={len(self.divisions())})"
        )

    def positions_um(self) -> np.ndarray:
        """``(N, 3)`` positions in physical units — voxel coordinates × voxel size."""
        return self.positions.astype(np.float64) * np.asarray(
            self.scale, dtype=np.float64
        )

    def index_of(self) -> Dict[int, int]:
        """Map node id -> positional index."""
        return {int(nid): i for i, nid in enumerate(self.node_ids)}

    def edge_indices(self) -> np.ndarray:
        """``(E, 2)`` edges remapped from node **ids** to positional **indices**.

        Edges naming a node absent from the store are dropped — a crop of a
        larger movie can legitimately reference cells outside its own bounds.
        """
        lookup = self.index_of()
        rows = [
            (lookup[int(s)], lookup[int(d)])
            for s, d in self.edges
            if int(s) in lookup and int(d) in lookup
        ]
        return np.asarray(rows, dtype=np.int64).reshape(-1, 2)

    # -- graph structure --------------------------------------------------

    def _degrees(self) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """Out-degree, in-degree, and the index-space edge list."""
        eidx = self.edge_indices()
        out = (
            np.bincount(eidx[:, 0], minlength=self.n_nodes)
            if len(eidx)
            else np.zeros(self.n_nodes, dtype=np.int64)
        )
        inn = (
            np.bincount(eidx[:, 1], minlength=self.n_nodes)
            if len(eidx)
            else np.zeros(self.n_nodes, dtype=np.int64)
        )
        return out, inn, eidx

    def divisions(self) -> np.ndarray:
        """Positional indices of dividing cells (out-degree >= 2)."""
        out, _, _ = self._degrees()
        return np.flatnonzero(out >= 2)

    def lineage_ids(self) -> np.ndarray:
        """``(N,)`` connected-component id per node — one id per lineage tree.

        Components are computed on the *undirected* graph, so a whole lineage
        (a founder cell and every descendant) shares one id and can be given one
        colour. Ids are assigned in order of each component's earliest node.
        """
        _, _, eidx = self._degrees()
        parent = np.arange(self.n_nodes, dtype=np.int64)

        def find(i: int) -> int:
            while parent[i] != i:
                parent[i] = parent[parent[i]]
                i = int(parent[i])
            return i

        for s, d in eidx:
            rs, rd = find(int(s)), find(int(d))
            if rs != rd:
                parent[max(rs, rd)] = min(rs, rd)

        roots = np.array([find(i) for i in range(self.n_nodes)], dtype=np.int64)
        # Renumber densely, ordered by first appearance, so ids index a palette.
        _, first = np.unique(roots, return_index=True)
        order = roots[np.sort(first)]
        remap = {int(r): k for k, r in enumerate(order)}
        return np.array([remap[int(r)] for r in roots], dtype=np.int64)


def read_geff(path: Union[str, Path]) -> TrackingGraph:
    """Read a ``.geff`` tracking graph into a :class:`TrackingGraph`.

    Parameters
    ----------
    path
        Path to the ``.geff`` store root (the directory holding ``zarr.json``).

    Returns
    -------
    TrackingGraph
        Node ids, timepoints, voxel-space ZYX positions, edges, and the voxel
        size read from the GEFF axis metadata.

    Raises
    ------
    ValueError
        If the store is not a GEFF group, or lacks the ``t``/``z``/``y``/``x``
        node properties this reader needs.
    """
    from luxar.io.zarr_v3 import Zarr3Group, open_zarr_v3

    root = Path(path)
    group = open_zarr_v3(root)
    if not isinstance(group, Zarr3Group):
        raise ValueError(f"{root} is a zarr array, not a GEFF group.")

    meta = group.attrs.get("geff")
    if meta is None:
        raise ValueError(
            f"{root} has no 'geff' attribute — not a GEFF store. "
            f"Attributes present: {sorted(group.attrs)}"
        )

    missing = [
        name for name in ("t",) + _SPATIAL if f"nodes/props/{name}/values" not in group
    ]
    if missing:
        raise ValueError(
            f"{root}: GEFF store is missing node properties {missing}; this reader "
            "needs t, z, y and x."
        )

    node_ids = np.asarray(group["nodes/ids"])
    t = np.asarray(group["nodes/props/t/values"]).astype(np.int64)
    coords = [np.asarray(group[f"nodes/props/{name}/values"]) for name in _SPATIAL]
    positions = np.stack(coords, axis=-1).astype(np.float64)

    edges = (
        np.asarray(group["edges/ids"]).reshape(-1, 2)
        if "edges/ids" in group
        else np.zeros((0, 2), dtype=node_ids.dtype)
    )

    axes: Sequence[dict] = meta.get("axes") or []
    by_name = {str(ax.get("name")).lower(): ax for ax in axes}
    scale = tuple(float(by_name.get(name, {}).get("scale") or 1.0) for name in _SPATIAL)
    units = tuple(by_name.get(name, {}).get("unit") for name in _SPATIAL)

    return TrackingGraph(
        node_ids=node_ids,
        t=t,
        positions=positions,
        edges=edges,
        scale=scale,  # type: ignore[arg-type]
        units=units,
    )
