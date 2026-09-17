"""The checks ``gsplat doctor`` runs, and the registry it runs them from.

Adding a check means writing a function of the same shape and appending it to
:data:`ALL_CHECKS` — nothing else in the doctor knows about any particular
condition. A check reads the store, decides what is wrong, and attaches a
closure that would repair it; it must NOT write, because the user may only have
asked for a diagnosis. Split this module into a package when the list outgrows
one file.

Conventions worth keeping:

* Report a condition even when it cannot be repaired — a finding with no ``fix``
  and a concrete ``remedy`` is more use than silence.
* Never repair on a guess. Where the correct value cannot be recovered from the
  store, say so and name what can produce it.
* Repairs here are attr-level: the doctor's finalize re-stamps the root
  ``content_hash`` and re-consolidates metadata for the whole run, so a fix must
  not do either itself.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any, Dict, List, Optional, Tuple

import numpy as np

from .model import Check, Finding

if TYPE_CHECKING:  # pragma: no cover - typing only
    import zarr

__all__ = ["ALL_CHECKS", "check_gsplat_readable", "check_partition_split_planes"]


class _GsplatStructureError(ValueError):
    """Structural validation failure with its store-relative location."""

    def __init__(self, path: str, message: str) -> None:
        super().__init__(message)
        self.path = path


def _iter_groups(group: "zarr.Group", path: str = "") -> "List[Tuple[str, Any]]":
    """Every group in the store, depth-first, with its store-relative path."""
    out = [(path, group)]
    for name in sorted(group.group_keys()):
        out.extend(_iter_groups(group[name], f"{path}/{name}" if path else name))
    return out


def _logical_array_shape(
    root: "zarr.Group", array: "zarr.Array", name: str, where: str
) -> Tuple[int, ...]:
    """Decoded shape inferred from zarr and encoding metadata, without reading data."""
    shape = tuple(int(size) for size in array.shape)
    encoding = array.attrs.get("encoding")
    if not isinstance(encoding, dict):
        return shape
    if encoding.get("name") == "broadcasted":
        if not shape:
            raise ValueError("broadcasted array has no leading dimension")
        return (int(encoding["n_elements"]), *shape[1:])
    if encoding.get("name") == "array_ref":
        target = encoding.get("target")
        try:
            root[encoding["target"]]
        except KeyError as exc:
            raise ValueError(
                f"array {name!r} at {where} references a missing target {target!r}"
            ) from exc
        original_shape = encoding.get("original_shape")
        if not isinstance(original_shape, (list, tuple)) or not original_shape:
            raise ValueError("array_ref is missing a non-empty original_shape")
        return tuple(int(size) for size in original_shape)
    return shape


def _leaf_array_lengths(
    root: "zarr.Group", group: "zarr.Group", path: str
) -> Dict[str, int]:
    """Required leaf-array lengths from metadata only."""
    where = path or "root"
    arrays = dict(group.arrays())
    names = ["centers", "amplitudes"]
    if "cholesky_factors_diag" in arrays:
        names.append("cholesky_factors_diag")
        diag_shape = _logical_array_shape(
            root, arrays["cholesky_factors_diag"], "cholesky_factors_diag", where
        )
        if len(diag_shape) < 2:
            raise ValueError(
                f"array 'cholesky_factors_diag' at {where} has invalid shape "
                f"{diag_shape}"
            )
        if diag_shape[1] > 1:
            names.append("cholesky_factors_offdiag")
    else:
        names.append("cholesky_factors")
    if "colors" in arrays:
        names.append("colors")

    lengths: Dict[str, int] = {}
    for name in names:
        if name not in arrays:
            raise ValueError(f"missing required array {name!r} at {where}")
        shape = _logical_array_shape(root, arrays[name], name, where)
        if not shape:
            raise ValueError(f"array {name!r} at {where} has no leading dimension")
        lengths[name] = shape[0]
    return lengths


def _validate_leaf_arrays(root: "zarr.Group", group: "zarr.Group", path: str) -> None:
    """Require every leaf array to describe the same number of splats."""
    try:
        lengths = _leaf_array_lengths(root, group, path)
    except Exception as exc:
        raise _GsplatStructureError(path, str(exc)) from exc
    if len(set(lengths.values())) == 1:
        return
    joined = ", ".join(f"{name}={length}" for name, length in lengths.items())
    raise _GsplatStructureError(
        path, f"array lengths disagree at {path or 'root'}: {joined}"
    )


def _validate_gsplat_structure(
    root: "zarr.Group", group: "zarr.Group", path: str = ""
) -> None:
    """Mirror ``read_gsplat_node`` using group and array metadata only."""
    kind = group.attrs.get("kind")
    if kind == "lod":
        prefix = "child_"
    elif kind == "partition":
        prefix = "part_"
    else:
        n_additive = int(group.attrs.get("n_additive_sublods", 1))
        if n_additive > 1:
            for index in range(n_additive):
                name = f"additive_{index}"
                if name not in group:
                    child_path = f"{path}/{name}" if path else name
                    raise _GsplatStructureError(
                        child_path,
                        f"missing required group {name!r} at {path or 'root'}",
                    )
                _validate_leaf_arrays(
                    root, group[name], f"{path}/{name}" if path else name
                )
            return
        _validate_leaf_arrays(root, group, path)
        return

    names = [name for name in group.group_keys() if str(name).startswith(prefix)]
    if not names:
        raise _GsplatStructureError(
            path, f"{kind} group at {path or 'root'} has no children"
        )
    for index in range(len(names)):
        name = f"{prefix}{index}"
        if name not in group:
            child_path = f"{path}/{name}" if path else name
            raise _GsplatStructureError(
                child_path, f"missing required group {name!r} at {path or 'root'}"
            )
        _validate_gsplat_structure(
            root, group[name], f"{path}/{name}" if path else name
        )


def check_gsplat_readable(root: "zarr.Group") -> List[Finding]:
    """A standalone gsplat store must use a format the current reader accepts."""
    if root.attrs.get("format_type") != "gsplats_zarr":
        return []

    from luxar.gsplats.io.save_gsplats import FORMAT_VERSION, SUPPORTED_FORMAT_VERSIONS
    from luxar.typing_utils.format_version import (
        FormatVersionOutcome,
        check_format_version,
    )

    version = root.attrs.get("format_version")
    outcome, message = check_format_version(
        "gsplats", version, FORMAT_VERSION, SUPPORTED_FORMAT_VERSIONS
    )
    if outcome is FormatVersionOutcome.REFUSE:
        return [
            Finding(
                check="format-version",
                severity="error",
                path="",
                summary=f"unsupported gsplat format version {version!r}",
                detail=(
                    "The current Luxar reader cannot open this store; supported "
                    f"versions are {SUPPORTED_FORMAT_VERSIONS}. " + message
                ),
                remedy=(
                    "Convert it with `luxar gsplat migrate-format <input> "
                    "<output.gsplats.zarr>`."
                ),
            )
        ]
    if outcome is FormatVersionOutcome.NEWER_MINOR:
        # Same policy as every reader: a newer minor loads with a warning, so
        # the doctor reports it as such rather than as an unreadable store.
        return [
            Finding(
                check="format-version",
                severity="warning",
                path="",
                summary=f"newer gsplat format version {version!r}",
                detail=message,
                remedy="Upgrade Luxar to read the newer content.",
            )
        ]

    try:
        _validate_gsplat_structure(root, root)
    except _GsplatStructureError as exc:
        path = exc.path
        error = str(exc)
    except Exception as exc:
        path = ""
        error = str(exc)
    else:
        return []
    return [
        Finding(
            check="readability",
            severity="error",
            path=path,
            summary="gsplat store cannot be read",
            detail=(
                f"The store's tree or leaf-array metadata is incomplete: {error}. "
                "This check does not read chunk payloads, so corruption inside "
                "stored chunks may only surface when the data is loaded."
            ),
            remedy="Restore the incomplete store or regenerate it from its source.",
        )
    ]


def _part_boxes(
    group: "zarr.Group",
) -> "Optional[List[Tuple[np.ndarray, np.ndarray]]]":
    """Each part's content box, indexed by ``child_index``.

    ``None`` when the parts do not describe a usable box set: a missing
    ``position_bounds``, a ``child_index`` that is not a permutation of
    ``0..n-1``, inconsistent bounds widths, or fewer than two spatial dims
    (splitting needs two, so 1D data is never partitioned).
    """
    by_index: Dict[int, Tuple[np.ndarray, np.ndarray]] = {}
    names = [n for n in group.group_keys() if str(n).startswith("part_")]
    if not names:
        return None
    for name in names:
        child = group[name]
        bounds = child.attrs.get("position_bounds")
        index = child.attrs.get("child_index")
        if bounds is None or index is None:
            return None
        try:
            lo = np.asarray(bounds["min"], dtype=float)
            hi = np.asarray(bounds["max"], dtype=float)
        except (KeyError, TypeError, ValueError):
            return None
        if lo.shape != hi.shape or lo.size < 2:
            return None
        by_index[int(index)] = (lo, hi)
    if sorted(by_index) != list(range(len(names))):
        return None
    boxes = [by_index[i] for i in range(len(names))]
    if any(lo.shape != boxes[0][0].shape for lo, _ in boxes[1:]):
        return None
    return boxes


def _reconstruction_axes(
    root: "zarr.Group", boxes: "List[Tuple[np.ndarray, np.ndarray]]"
) -> Tuple[int, ...]:
    width = len(boxes[0][0])
    fallback = tuple(range(min(3, width)))
    scene_dimensions = root.attrs.get("scene_dimensions")
    if scene_dimensions is not None:
        from luxar.core.dimensions import Dimensions

        displayed = Dimensions.from_dict(scene_dimensions).displayed
        axes = tuple(axis for axis in displayed if 0 <= axis < width)
        return axes or fallback
    return fallback


def check_partition_split_planes(root: "zarr.Group") -> List[Finding]:
    """A ``kind=partition`` should record how its parts stack up (``bsp_tree``).

    The viewer orders partition parts back-to-front by traversing those planes —
    exact for any camera pose, including inside the volume. Without them it sorts
    parts by content centroid, which is not a valid painter's order: it flips
    discretely as the camera moves, so an order-dependent blending mode
    (``normal`` or ``volumetric``) pops at the seams on every orbit.

    The conditions below are silent in the viewer:

    * **Missing.** Written by a producer that did not record its planes (any
      tiled fit before #1555), or dropped by a tool that rebuilt the tree. When
      the parts are disjoint the planes are recoverable from the part boxes, so
      this repairs itself; when they are not — a uniform-tiled fit keeps each
      tile's apodization halo, so its parts genuinely intersect — no exact tree
      exists and the finding stands unfixed rather than inventing one.
    * **Stale or mismatched.** Present but disagreeing with where the parts
      actually sit: planes left in a pre-transform coordinate space, or labels
      naming a different part set. Worse than missing, because the traversal
      still returns a plausible permutation — the ordering is confidently wrong
      instead of falling back. Repaired by recovering planes when possible, and
      by REMOVING the tree when not: the centroid fallback is at least honest.
    * **Approximate.** Some producer planes are center-based rather than exact
      separators of the written part bounds: uniform tiles overlap, while lines
      and mesh split polyline/face centroids whose vertices can cross a cut. A
      stored tree is reported as a note when every cut remains plausible under
      the viewer's center-straddle rule and per-axis tolerance floor. A cut
      outside that band is repaired when its position can be recovered safely,
      or removed when it cannot.
    """
    from luxar.core.group.partition import (
        reconstruct_serialized_bsp_tree,
        serialized_bsp_tree_separates,
        serialized_bsp_tree_straddles_centers,
    )

    findings: List[Finding] = []
    for path, group in _iter_groups(root):
        if group.attrs.get("kind") != "partition":
            continue
        where = path or "<root>"
        boxes = _part_boxes(group)
        stored = group.attrs.get("bsp_tree")

        if stored is not None and boxes is not None:
            if serialized_bsp_tree_separates(dict(stored), boxes):
                continue  # healthy
            stored_dict = dict(stored)
            rebuilt = reconstruct_serialized_bsp_tree(
                boxes, axes=_reconstruction_axes(root, boxes)
            )
            labels_name_parts = _labels_name_the_parts(stored_dict, len(boxes))
            approximate_ok = rebuilt is None or group.attrs.get("display_type") in (
                "lines",
                "mesh",
            )
            if (
                labels_name_parts
                and approximate_ok
                and serialized_bsp_tree_straddles_centers(stored_dict, boxes)
            ):
                findings.append(_approximate_finding(group, where, len(boxes), rebuilt))
                continue
            if rebuilt is None and labels_name_parts:
                recovered = _recover_frame_scale(stored_dict, boxes)
                if recovered is not None:
                    repaired, factors, frame_scale_supported = recovered
                    findings.append(
                        _misframed_finding(
                            group,
                            where,
                            len(boxes),
                            repaired,
                            factors,
                            frame_scale_supported,
                        )
                    )
                    continue
                # The overlap exception is only for a geometrically plausible
                # approximate tree. A grosser violation is stale even though no
                # exact replacement can be reconstructed from intersecting boxes.
                findings.append(
                    _stale_finding(
                        group, where, len(boxes), None, overlap_violation=True
                    )
                )
                continue
            findings.append(_stale_finding(group, where, len(boxes), rebuilt))
            continue

        if stored is not None:
            # Cannot judge it: no usable part boxes to check against. Say so
            # rather than either trusting or condemning the tree.
            findings.append(
                Finding(
                    check="split-planes",
                    severity="note",
                    path=where,
                    summary="split planes present but not verifiable",
                    detail=(
                        "The parts carry no usable position_bounds, so there is "
                        "nothing to check the stored planes against."
                    ),
                    remedy="Re-run doctor after re-writing the store with a current Luxar.",
                )
            )
            continue

        if boxes is None:
            findings.append(
                Finding(
                    check="split-planes",
                    severity="warning",
                    path=where,
                    summary="no split planes, and the part boxes cannot be read",
                    detail=(
                        "Parts are ordered by centroid, which is not a valid "
                        "painter's order and pops at the seams under `normal` or "
                        "`volumetric` blending."
                    ),
                    remedy="Re-fit, or re-write the store, with a current Luxar.",
                )
            )
            continue

        rebuilt = reconstruct_serialized_bsp_tree(
            boxes, axes=_reconstruction_axes(root, boxes)
        )
        findings.append(_missing_finding(group, where, len(boxes), rebuilt))
    return findings


def _recover_frame_scale(
    stored: Dict[str, Any], boxes: "List[Tuple[np.ndarray, np.ndarray]]"
) -> "Optional[Tuple[Dict[str, Any], Tuple[float, ...], bool]]":
    """Recover a tree, per-axis factors, and whether they prove a frame scale.

    The known producer failures are pure coordinate-frame scales
    (``--downscale`` and ``voxel_size``). Each node estimates its intended cut
    from the midpoint of the two sides' measured overlap band. The median ratio
    is only accepted when applying one factor per axis makes EVERY node pass the
    overlap-tolerant center-straddling check; otherwise guessing would be worse
    than the honest centroid fallback.
    """
    from luxar.core.group.partition import (
        map_serialized_bsp_tree,
        serialized_bsp_tree_axis_overlap_floors,
        serialized_bsp_tree_straddles_centers,
    )

    overlap_floors = serialized_bsp_tree_axis_overlap_floors(stored, boxes)
    if overlap_floors is None:
        return None
    axes = range(len(boxes[0][0]))
    ratios: Dict[int, List[float]] = {axis: [] for axis in axes}
    ranges: Dict[int, List[Tuple[float, float]]] = {axis: [] for axis in axes}
    if not _collect_frame_scale_ranges(stored, boxes, overlap_floors, ratios, ranges):
        return None
    factors = _resolve_frame_factors(ratios, ranges, len(boxes[0][0]))
    if factors is None:
        return None
    repaired = map_serialized_bsp_tree(stored, linear=np.diag(factors))
    if repaired is None or not serialized_bsp_tree_straddles_centers(repaired, boxes):
        return None
    return repaired, factors, _frame_scale_is_supported(ratios, factors)


def _collect_frame_scale_ranges(
    node: Dict[str, Any],
    boxes: "List[Tuple[np.ndarray, np.ndarray]]",
    overlap_floors: "Tuple[float, ...]",
    ratios: Dict[int, List[float]],
    ranges: Dict[int, List[Tuple[float, float]]],
) -> bool:
    from luxar.core.group.partition import serialized_bsp_leaf_labels

    if "part" in node:
        return True
    try:
        axis = int(node["axis"])
        split = float(node["split"])
        left = serialized_bsp_leaf_labels(node["left"])
        right = serialized_bsp_leaf_labels(node["right"])
        if axis not in ratios or not np.isfinite(split):
            return False
        left_high = max(float(boxes[i][1][axis]) for i in left)
        right_low = min(float(boxes[i][0][axis]) for i in right)
    except (KeyError, TypeError, ValueError, IndexError, OverflowError):
        return False
    overlap = max(overlap_floors[axis], left_high - right_low)
    band_low, band_high = sorted((right_low, left_high))
    band_low -= overlap
    band_high += overlap
    if split == 0.0:
        valid = band_low <= 0.0 <= band_high
    else:
        ratio = 0.5 * (left_high + right_low) / split
        low, high = sorted((band_low / split, band_high / split))
        valid = np.isfinite(ratio) and ratio > 0.0 and high > 0.0
        if valid:
            ratios[axis].append(ratio)
            ranges[axis].append((max(low, np.nextafter(0.0, 1.0)), high))
    return (
        valid
        and _collect_frame_scale_ranges(
            node["left"], boxes, overlap_floors, ratios, ranges
        )
        and _collect_frame_scale_ranges(
            node["right"], boxes, overlap_floors, ratios, ranges
        )
    )


def _resolve_frame_factors(
    ratios: Dict[int, List[float]],
    ranges: Dict[int, List[Tuple[float, float]]],
    ndim: int,
) -> "Optional[Tuple[float, ...]]":
    factors = []
    for axis in range(ndim):
        axis_ranges = ranges.get(axis)
        if not axis_ranges:
            factors.append(1.0)
            continue
        low = max(bounds[0] for bounds in axis_ranges)
        high = min(bounds[1] for bounds in axis_ranges)
        if low > high:
            return None
        estimate = float(np.median(ratios[axis]))
        if not low <= estimate <= high:
            return None
        factors.append(estimate)
    return tuple(factors)


def _frame_scale_is_supported(
    ratios: Dict[int, List[float]], factors: Tuple[float, ...]
) -> bool:
    """Whether changed axes have repeated evidence or store-wide corroboration."""
    proven_factors = []
    singleton_ratios = []
    for axis, factor in enumerate(factors):
        if np.isclose(factor, 1.0):
            continue
        samples = ratios[axis]
        if len(samples) < 2:
            singleton_ratios.extend(samples)
            continue
        estimate = float(np.median(samples))
        spread = (max(samples) - min(samples)) / abs(estimate)
        if spread > 0.05:
            return False
        proven_factors.append(factor)
    return not singleton_ratios or (
        bool(proven_factors)
        and all(
            any(abs(ratio - factor) / abs(factor) <= 0.05 for factor in proven_factors)
            for ratio in singleton_ratios
        )
    )


def _labels_name_the_parts(stored: Dict[str, Any], n_parts: int) -> bool:
    """True when a stored tree's leaves are exactly ``0..n_parts-1``, each once.

    The half of the soundness check that does NOT depend on the geometry: a tree
    naming the right part set may still have stale planes, but one naming the
    wrong set is broken however the parts sit.
    """
    from luxar.core.group.partition import serialized_bsp_leaf_labels

    try:
        return sorted(serialized_bsp_leaf_labels(stored)) == list(range(n_parts))
    except (KeyError, TypeError, ValueError):  # malformed node shape
        return False


def _approximate_finding(
    group: "zarr.Group",
    where: str,
    n_parts: int,
    rebuilt: Optional[Dict[str, Any]],
) -> Finding:
    def replace() -> None:
        assert rebuilt is not None
        group.attrs["bsp_tree"] = rebuilt

    if rebuilt is None:
        remedy = (
            "Nothing to do. Exact ordering requires non-overlapping part bounds; "
            "for tiled gsplats, use a content plan or non-overlapping tiles."
        )
    else:
        remedy = (
            "Replace the stored planes with exact cuts recovered from the part boxes."
        )

    return Finding(
        check="split-planes",
        severity="note",
        path=where,
        summary=f"split planes over {n_parts} parts are centroid-valid but approximate",
        detail=(
            "The stored planes do not separate the part bounds, but every cut "
            "still lies between its child-box centers. This is the documented shape of "
            "uniform-tiled fits, whose apodized tiles keep their overlap band, and "
            "centroid-split lines or mesh parts, whose vertices can cross a cut. "
            "The producer's cuts still confine ordering ambiguity to geometry that "
            "crosses a cut instead of letting whole parts swap."
        ),
        remedy=remedy,
        fix=replace if rebuilt is not None else None,
    )


def _misframed_finding(
    group: "zarr.Group",
    where: str,
    n_parts: int,
    repaired: Dict[str, Any],
    factors: Tuple[float, ...],
    frame_scale_supported: bool,
) -> Finding:
    used = ", ".join(
        f"axis {axis} ×{factor:g}"
        for axis, factor in enumerate(factors)
        if not np.isclose(factor, 1.0)
    )

    def replace() -> None:
        group.attrs["bsp_tree"] = repaired

    if frame_scale_supported:
        summary = f"split planes for {n_parts} parts use a different coordinate frame"
        detail = (
            "The parts overlap, but their centers do not straddle the stored "
            "planes even after allowing the measured overlap band and its "
            "per-axis tolerance floor. A single "
            f"per-axis scale explains every plane ({used}), matching a tree "
            "written before a downscale or voxel-size conversion was applied "
            "to the splat centers."
        )
        remedy = "Rescale the stored planes into the parts' coordinate frame."
    else:
        summary = f"split planes for {n_parts} parts fall outside their overlap bands"
        detail = (
            "The parts' centers do not straddle the stored planes even after "
            "allowing the measured overlap band and its per-axis tolerance "
            "floor. The plane positions can be "
            "recovered from those bands, but the raw ratios do not provide "
            "enough consistent evidence to identify a coordinate-frame scale."
        )
        remedy = "Replace the stored planes with the recovered overlap-band cuts."

    return Finding(
        check="split-planes",
        severity="error",
        path=where,
        summary=summary,
        detail=detail,
        remedy=remedy,
        fix=replace,
    )


def _missing_finding(
    group: "zarr.Group", where: str, n_parts: int, rebuilt: Optional[Dict[str, Any]]
) -> Finding:
    detail = (
        f"{n_parts} parts are ordered by content centroid, which is not a valid "
        "painter's order: it flips discretely as the camera moves, so an "
        "order-dependent blending mode pops at the seams on every orbit."
    )
    if rebuilt is None:
        return Finding(
            check="split-planes",
            severity="warning",
            path=where,
            summary="no split planes, and the parts are not disjoint",
            detail=detail
            + " The parts overlap, so no exact ordering exists to recover — "
            "typical of a uniform-tiled fit, whose apodized tiles keep their halo.",
            remedy=(
                "Re-fit with a current Luxar, which records the tiling's own "
                "planes (approximate for uniform tiling, and documented as such)."
            ),
        )

    def apply() -> None:
        group.attrs["bsp_tree"] = rebuilt

    return Finding(
        check="split-planes",
        severity="error",
        path=where,
        summary=f"no split planes recorded for {n_parts} parts",
        detail=detail,
        remedy=(
            "Recover them from the part boxes — the parts are disjoint, so the "
            "planes are exact."
        ),
        fix=apply,
    )


def _stale_finding(
    group: "zarr.Group",
    where: str,
    n_parts: int,
    rebuilt: Optional[Dict[str, Any]],
    *,
    overlap_violation: bool = False,
) -> Finding:
    if overlap_violation:
        reason = (
            "The parts' centers do not straddle the stored planes even after "
            "allowing the measured overlap band and its per-axis tolerance "
            "floor, so the viewer "
        )
    else:
        reason = "The stored planes do not separate the parts they name, so the viewer "
    detail = (
        reason
        + "orders confidently WRONG rather than falling back to centroids. Usually "
        "a tree left behind in a pre-transform coordinate space, or one written "
        "against a different part set."
    )
    if rebuilt is None:

        def drop() -> None:
            del group.attrs["bsp_tree"]

        return Finding(
            check="split-planes",
            severity="error",
            path=where,
            summary="split planes disagree with the parts, and cannot be rebuilt",
            detail=detail,
            remedy=(
                "Remove them, so ordering falls back to the centroid heuristic — "
                "approximate, but honest. Re-fit to get exact ordering back."
            ),
            fix=drop,
        )

    def replace() -> None:
        group.attrs["bsp_tree"] = rebuilt

    return Finding(
        check="split-planes",
        severity="error",
        path=where,
        summary=f"split planes disagree with the {n_parts} parts they name",
        detail=detail,
        remedy="Rebuild them from the part boxes, which are disjoint.",
        fix=replace,
    )


#: Every check, in the order the doctor runs them.
ALL_CHECKS: "List[Check]" = [check_gsplat_readable, check_partition_split_planes]
