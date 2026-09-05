"""Read a ``.gsplats.zarr`` tree's SHAPE without decoding a single chunk.

:func:`luxar.gsplats.io.load_gsplats.load_gsplat_node` reconstructs a full
``GSplatNode``, which means decoding every array of every leaf into RAM. That is
right when you are going to render, refit or rewrite the splats. It is the wrong
tool for answering *how many parts does this have, and how many splats in each* —
and that is what ``luxar gsplat info`` prints for a partition or nested store.

Measured on a 10-part partition, 960K splats, 6.4 MB on disk: ``info`` issued
**120 decode calls — exactly three full reads of the tree** — materialising
115 MB, 17x the store's on-disk size, to print 25 lines of metadata. The three
reads were a flat load that raises for a tree and is discarded, a second full
read used only as a validity gate with its return value thrown away, and a third
for the summary itself (audit A14-02). The operational record has this shape
reaching 116 GB RSS on a 13 GB store.

Everything that summary prints — kind, dimensionality, per-leaf splat counts,
child counts, position bounds, the ``fitting/`` record — lives in ``attrs`` or in
an array's *metadata*. Reading ``array.shape`` does not fetch chunks.

This module also carries the size ESTIMATE that :func:`load_gsplat_node` warns
with, since the estimate is derived from the same metadata walk.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING, Literal

import numpy as np

if TYPE_CHECKING:  # pragma: no cover - typing only
    import zarr

#: Arrays whose decoded form defaults to float32 for legacy encodings that do
#: not stamp the original dtype. Current encodings carry ``original_dtype``.
_DECODES_TO_FLOAT32 = ("centers", "amplitudes", "colors")

NodeKind = Literal["leaf", "lod", "partition"]


@dataclass(frozen=True)
class GSplatTreeSummary:
    """Structure of one node, read from attrs and array metadata only.

    Deliberately not a :class:`~luxar.gsplats.tree.GSplatNode`: it holds no
    arrays and cannot be rendered or written. Anything that needs the values
    must call :func:`~luxar.gsplats.io.load_gsplats.load_gsplat_node` and pay
    for them.
    """

    kind: NodeKind
    #: Number of centre columns, or 0 when the node holds no leaf.
    ndim: int
    #: Splats under this node, summed over every leaf.
    n_splats: int
    children: tuple["GSplatTreeSummary", ...] = ()
    #: Splat counts of each leaf under this node, in tree order.
    leaf_counts: tuple[int, ...] = ()
    #: What a full decode of this subtree would materialise, in bytes.
    #:
    #: An ESTIMATE, and named one: it assumes the documented float32 decode for
    #: the quantized channels and the stored dtype for the rest, and it counts
    #: the arrays a leaf read touches rather than transient copies inside it. It
    #: is a floor for the peak, which is the direction that matters for a
    #: warning.
    estimated_decoded_bytes: int = 0


def _array_bytes(group: "zarr.Group", name: str) -> int:
    """Decoded size of ``group[name]``, without reading a chunk."""
    if name not in group:
        return 0
    array = group[name]
    encoding = dict(array.attrs.get("encoding", {}))
    if "original_shape" in encoding:
        shape = tuple(encoding["original_shape"])
    elif encoding.get("name") == "broadcasted":
        shape = (int(encoding.get("n_elements", array.shape[0])), *array.shape[1:])
    else:
        shape = array.shape
    count = 1
    for dim in shape:
        count *= int(dim)
    original_dtype = encoding.get("original_dtype")
    if original_dtype is not None:
        itemsize = int(np.dtype(original_dtype).itemsize)
    else:
        itemsize = 4 if name in _DECODES_TO_FLOAT32 else int(array.dtype.itemsize)
    return count * itemsize


def _leaf_arrays_summary(group: "zarr.Group") -> tuple[int, int, int]:
    """``(n_splats, ndim, estimated_decoded_bytes)`` for one splat set."""
    if "centers" not in group:
        return 0, 0, 0
    shape = group["centers"].shape
    n_splats = int(group.attrs.get("n_splats", shape[0]))
    ndim = int(group.attrs.get("ndim", shape[1] if len(shape) > 1 else 0))

    total = _array_bytes(group, "centers") + _array_bytes(group, "amplitudes")
    total += _array_bytes(group, "colors") + _array_bytes(group, "label_ids")

    # Cholesky is the one channel whose STORED shape does not predict its
    # decoded shape. It is written as `cholesky_factors_diag` +
    # `cholesky_factors_offdiag`, and a uniform-covariance leaf stores a SINGLE
    # row for the whole set — observed `(1, 3)` for 100 splats. `_decode_cholesky`
    # broadcasts that back to `(N, tri)` float32, so summing the stored arrays
    # would under-report by a factor of N. Compute the decoded size instead.
    if "cholesky_factors" in group or any(
        f"cholesky_factors_{part}" in group for part in ("diag", "offdiag")
    ):
        tri = ndim * (ndim + 1) // 2
        total += n_splats * tri * 4

    return n_splats, ndim, total


def _child_names(group: "zarr.Group", prefix: str) -> list[str]:
    names = [str(name) for name in group if str(name).startswith(prefix)]
    # `child_10` must not sort before `child_2`.
    return sorted(names, key=lambda name: int(name.rsplit("_", 1)[1]))


def read_gsplat_tree_summary(group: "zarr.Group") -> GSplatTreeSummary:
    """Summarise the subtree rooted at ``group`` from metadata alone.

    Mirrors the dispatch in
    :func:`luxar.io._compiler.gsplat_tree.read_gsplat_node` so the two cannot
    disagree about a store's shape — ``TestAgreesWithTheFullReader`` asserts
    that on every fixture.

    Raises:
        ValueError: The group is not a recognisable gsplat node. Same contract
            as the full reader, so callers can keep using this as a validity
            gate.
    """
    kind = group.attrs.get("kind")

    if kind in ("lod", "partition"):
        prefix = "child_" if kind == "lod" else "part_"
        children = tuple(
            read_gsplat_tree_summary(group[name])
            for name in _child_names(group, prefix)
        )
        leaf_counts = tuple(count for child in children for count in child.leaf_counts)
        return GSplatTreeSummary(
            kind=kind,
            ndim=next((child.ndim for child in children if child.ndim), 0),
            n_splats=sum(child.n_splats for child in children),
            children=children,
            leaf_counts=leaf_counts,
            estimated_decoded_bytes=sum(
                child.estimated_decoded_bytes for child in children
            ),
        )

    # A leaf: one splat set, or an additive ladder of them.
    n_additive = int(group.attrs.get("n_additive_sublods", 1))
    if n_additive > 1:
        sublods = [
            _leaf_arrays_summary(group[f"additive_{i}"]) for i in range(n_additive)
        ]
    else:
        sublods = [_leaf_arrays_summary(group)]

    if not sublods or all(n == 0 and d == 0 for n, d, _ in sublods):
        raise ValueError(
            f"not a gsplat node: group has no 'kind' attribute and no 'centers' "
            f"array (attrs: {sorted(group.attrs)})"
        )

    # SUM over the ladder, and ndim from the FIRST rung -- matching
    # `GSplatLeaf.n_splats` and `.ndim` exactly. A ladder's rungs are stored as
    # increments, not as nested prefixes, so the finest rung alone undercounts:
    # measured 1,250 against the full reader's 5,000 on a 4-rung `stream` store
    # before this was corrected.
    n_splats = sum(count for count, _, _ in sublods)
    ndim = sublods[0][1]
    return GSplatTreeSummary(
        kind="leaf",
        ndim=ndim,
        n_splats=n_splats,
        leaf_counts=(n_splats,),
        estimated_decoded_bytes=sum(size for _, _, size in sublods),
    )
