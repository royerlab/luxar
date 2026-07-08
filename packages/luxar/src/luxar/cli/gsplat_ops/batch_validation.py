"""Structural validation helpers for batch-fitted tile stores."""

from __future__ import annotations

import json
from pathlib import Path


def validate_leaf_arrays(node_dir: Path, label: str) -> str:
    """Check a v3.x gsplats leaf's required array sub-dirs (no decode)."""
    # Cholesky factors are stored as the v3.1 split (``cholesky_factors_diag``,
    # optionally + ``cholesky_factors_offdiag``) or a single v3.0
    # ``cholesky_factors`` array. The diagonal is the marker for the split.
    diag_dir = node_dir / "cholesky_factors_diag"
    is_split = diag_dir.is_dir()
    chol_name = "cholesky_factors_diag" if is_split else "cholesky_factors"
    for arr_name in ("centers", "amplitudes", chol_name):
        arr_dir = node_dir / arr_name
        if not arr_dir.is_dir():
            return f"missing_{arr_name}@{label}"
        if not (arr_dir / ".zarray").exists():
            return f"no_zarray_{arr_name}@{label}"

    # v3.1 split: for d > 1 the off-diagonal array is mandatory — only d == 1
    # omits it. A leaf with the diagonal but no off-diagonal is a partial /
    # corrupt write; surface it (recoverable via re-fit) rather than passing.
    if is_split:
        try:
            d = int(json.loads((diag_dir / ".zarray").read_text())["shape"][1])
        except (
            OSError,
            json.JSONDecodeError,
            KeyError,
            IndexError,
            TypeError,  # shape is null / scalar / non-subscriptable
            ValueError,
        ):
            return f"no_zarray_cholesky_factors_diag@{label}"
        if d > 1 and not (node_dir / "cholesky_factors_offdiag" / ".zarray").exists():
            return f"missing_cholesky_factors_offdiag@{label}"
    return "ok"


def validate_node_dir(node_dir: Path, label: str) -> str:
    """Structurally validate a v3.0+ node subtree on disk (no array decode)."""
    zattrs_path = node_dir / ".zattrs"
    if not zattrs_path.exists():
        # Every node (root, child_<i>, part_<i>) must carry its .zattrs; a
        # metadata-stripped node is corrupt, not a bare single-set leaf.
        return f"no_zattrs@{label}"
    try:
        attrs = json.loads(zattrs_path.read_text())
    except (json.JSONDecodeError, OSError):
        return f"corrupt_zattrs@{label}"

    kind = attrs.get("kind")
    if kind in ("lod", "partition"):
        prefix = "child_" if kind == "lod" else "part_"
        children = sorted(
            d for d in node_dir.iterdir() if d.is_dir() and d.name.startswith(prefix)
        )
        if not children:
            return f"{kind}_no_children@{label}"
        for child in children:
            reason = validate_node_dir(child, f"{label}/{child.name}")
            if reason != "ok":
                return reason
        return "ok"

    # Leaf: a single splat set, or an additive ladder (additive_<i>/ subgroups).
    n_additive = int(attrs.get("n_additive_sublods", 1))
    if n_additive > 1:
        for i in range(n_additive):
            reason = validate_leaf_arrays(
                node_dir / f"additive_{i}", f"{label}/additive_{i}"
            )
            if reason != "ok":
                return reason
        return "ok"
    return validate_leaf_arrays(node_dir, label)


def validate_tile(tile_path: Path) -> str:
    """Validate a single v3 tile's integrity. Returns ``'ok'`` or reason."""
    # .zmetadata is written last by consolidate_metadata — best completeness signal.
    if not (tile_path / ".zmetadata").exists():
        return "no_zmetadata (save incomplete)"

    zattrs_path = tile_path / ".zattrs"
    if not zattrs_path.exists():
        return "no_zattrs"
    try:
        attrs = json.loads(zattrs_path.read_text())
    except (json.JSONDecodeError, OSError):
        return "corrupt_zattrs"

    if attrs.get("format_type") != "gsplats_zarr":
        return f"bad_format_type: {attrs.get('format_type')}"

    from luxar.gsplats.io.save_gsplats import SUPPORTED_FORMAT_VERSIONS

    version = attrs.get("format_version")
    if version not in SUPPORTED_FORMAT_VERSIONS:
        # Not corrupt — just unmigrated. Surface it instead of classifying it as
        # corrupt (which would let --fix delete a recoverable tile).
        return f"unsupported_format_version: {version} (run gsplat migrate-format)"

    return validate_node_dir(tile_path, ".")
