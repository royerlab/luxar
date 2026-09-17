"""Structural validation helpers for batch-fitted tile stores.

Metadata is read straight off disk rather than by opening each node as a zarr
store — a batch run holds thousands of tiles and this walk is a fast structural
verdict, not a decode. The reads go through :mod:`luxar._zarr_compat` so they
answer for a v2 tile and a v3 one alike: a long-lived batch directory can
easily hold both, since tiles written before the format flip are still valid
input to a resumed run.
"""

from __future__ import annotations

from pathlib import Path

from luxar._zarr_compat import is_consolidated, read_array_meta, read_node_attrs


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
        if read_array_meta(arr_dir) is None:
            return f"no_zarray_{arr_name}@{label}"

    # v3.1 split: for d > 1 the off-diagonal array is mandatory — only d == 1
    # omits it. A leaf with the diagonal but no off-diagonal is a partial /
    # corrupt write; surface it (recoverable via re-fit) rather than passing.
    if is_split:
        meta = read_array_meta(diag_dir)
        try:
            d = int(meta["shape"][1])  # type: ignore[index]
        except (
            KeyError,
            IndexError,
            TypeError,  # meta is None, or shape is null / scalar
            ValueError,
        ):
            return f"no_zarray_cholesky_factors_diag@{label}"
        offdiag = node_dir / "cholesky_factors_offdiag"
        if d > 1 and read_array_meta(offdiag) is None:
            return f"missing_cholesky_factors_offdiag@{label}"
    return "ok"


def validate_node_dir(node_dir: Path, label: str) -> str:
    """Structurally validate a v3.0+ node subtree on disk (no array decode)."""
    # Every node (root, child_<i>, part_<i>) must carry its attributes; a
    # metadata-stripped node is corrupt, not a bare single-set leaf. Absent,
    # unreadable and EMPTY are one verdict here — all three mean "no usable node
    # metadata" — where the v2-only version could tell a missing file from a
    # corrupt one. The distinction never reached the caller: all return a reason
    # string and all are repaired the same way, by re-fitting the tile.
    #
    # `not attrs` rather than `attrs is None` is what keeps the two formats
    # equivalent. v2 stores attributes in a SEPARATE `.zattrs`, so stripping
    # them removes the document and reads back as None; v3 stores them INSIDE
    # `zarr.json` next to `node_type`, so stripping them leaves a structurally
    # valid document whose attributes are `{}`. Measured: with `attrs is None`,
    # a v3 leaf with intact arrays and no attributes validated as "ok" while the
    # identical v2 corruption returned `no_zattrs`. Every node of a real tree
    # carries 11-19 attributes (root: format_type/format_version/...; child_i /
    # part_i: 18-19 each), so an empty mapping is never legitimate.
    attrs = read_node_attrs(node_dir)
    if not attrs:
        return f"no_zattrs@{label}"

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
    # Consolidated metadata is written LAST — the best completeness signal there
    # is. Where it lives depends on the format (v2's separate `.zmetadata`, v3's
    # `consolidated_metadata` inside the root `zarr.json`), so the question is
    # asked rather than the filename named. The verdict string keeps saying
    # `no_zmetadata` because it is a stable identifier that operators grep for
    # and `--fix` keys on; renaming it to suit v3 would break that for no gain.
    if not is_consolidated(tile_path):
        return "no_zmetadata (save incomplete)"

    # `not attrs` for the same reason as in `validate_node_dir`: at v3 an
    # attribute-stripped root still has a `zarr.json`, and reporting it as
    # `bad_format_type: None` would name the wrong defect.
    attrs = read_node_attrs(tile_path)
    if not attrs:
        return "no_zattrs"

    if attrs.get("format_type") != "gsplats_zarr":
        return f"bad_format_type: {attrs.get('format_type')}"

    from luxar.gsplats.io.save_gsplats import FORMAT_VERSION, SUPPORTED_FORMAT_VERSIONS
    from luxar.typing_utils.format_version import (
        FormatVersionOutcome,
        check_format_version,
    )

    version = attrs.get("format_version")
    outcome, _ = check_format_version(
        "gsplats", version, FORMAT_VERSION, SUPPORTED_FORMAT_VERSIONS
    )
    if outcome is FormatVersionOutcome.REFUSE:
        # Not corrupt — just unmigrated. Surface it instead of classifying it as
        # corrupt (which would let --fix delete a recoverable tile). A newer
        # MINOR is readable under the shared policy and falls through.
        return f"unsupported_format_version: {version} (run gsplat migrate-format)"

    return validate_node_dir(tile_path, ".")
