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

__all__ = ["ALL_CHECKS", "check_partition_split_planes"]


def _iter_groups(group: "zarr.Group", path: str = "") -> "List[Tuple[str, Any]]":
    """Every group in the store, depth-first, with its store-relative path."""
    out = [(path, group)]
    for name in sorted(group.group_keys()):
        out.extend(_iter_groups(group[name], f"{path}/{name}" if path else name))
    return out


def _part_boxes(
    group: "zarr.Group",
) -> "Optional[List[Tuple[np.ndarray, np.ndarray]]]":
    """Each part's content box, indexed by ``child_index``.

    ``None`` when the parts do not describe a usable box set: a missing
    ``position_bounds``, a ``child_index`` that is not a permutation of
    ``0..n-1``, or fewer than two spatial dims (splitting needs two, so 1D data
    is never partitioned). Only the first three dims are read — a stacked
    time/channel axis is never a split axis.
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
        spatial = min(3, int(lo.size))
        by_index[int(index)] = (lo[:spatial], hi[:spatial])
    if sorted(by_index) != list(range(len(names))):
        return None
    return [by_index[i] for i in range(len(names))]


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
    * **Approximate.** Overlapping parts cannot be separated exactly. A stored
      tree is reported as a note when every cut remains plausible within the
      measured overlap band. A cut outside that band is repaired when its
      position can be recovered safely, or removed when it cannot.
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
            rebuilt = reconstruct_serialized_bsp_tree(boxes)
            if rebuilt is None and _labels_name_the_parts(dict(stored), len(boxes)):
                stored_dict = dict(stored)
                if serialized_bsp_tree_straddles_centers(stored_dict, boxes):
                    findings.append(_approximate_finding(where, len(boxes)))
                    continue
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

        rebuilt = reconstruct_serialized_bsp_tree(boxes)
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
        serialized_bsp_tree_straddles_centers,
    )

    ratios: Dict[int, List[float]] = {0: [], 1: [], 2: []}
    ranges: Dict[int, List[Tuple[float, float]]] = {0: [], 1: [], 2: []}
    if not _collect_frame_scale_ranges(stored, boxes, ratios, ranges):
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
    if split == 0.0:
        low, high = sorted((right_low, left_high))
        valid = low <= 0.0 <= high
    else:
        ratio = 0.5 * (left_high + right_low) / split
        low, high = sorted((right_low / split, left_high / split))
        valid = np.isfinite(ratio) and ratio > 0.0 and high > 0.0
        if valid:
            ratios[axis].append(ratio)
            ranges[axis].append((max(low, np.nextafter(0.0, 1.0)), high))
    return (
        valid
        and _collect_frame_scale_ranges(node["left"], boxes, ratios, ranges)
        and _collect_frame_scale_ranges(node["right"], boxes, ratios, ranges)
    )


def _resolve_frame_factors(
    ratios: Dict[int, List[float]],
    ranges: Dict[int, List[Tuple[float, float]]],
    ndim: int,
) -> "Optional[Tuple[float, ...]]":
    factors = []
    for axis in range(ndim):
        if not ranges[axis]:
            factors.append(1.0)
            continue
        low = max(bounds[0] for bounds in ranges[axis])
        high = min(bounds[1] for bounds in ranges[axis])
        if low > high:
            return None
        estimate = float(np.median(ratios[axis]))
        factors.append(min(max(estimate, low), high))
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


def _approximate_finding(where: str, n_parts: int) -> Finding:
    return Finding(
        check="split-planes",
        severity="note",
        path=where,
        summary=f"split planes over {n_parts} parts that overlap — approximate",
        detail=(
            "The parts share space, so no tree separates them and the stored "
            "planes cannot be checked exactly. This is the documented shape of a "
            "uniform-tiled fit, whose apodized tiles keep their overlap band: the "
            "cuts sit at each band's midplane, which confines any misordering to "
            "the band instead of letting whole parts swap."
        ),
        remedy=(
            "Nothing to do. For an exactly-ordered partition, fit with a content "
            "plan (or non-overlapping tiles) instead."
        ),
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
            "planes even after allowing the measured overlap band. A single "
            f"per-axis scale explains every plane ({used}), matching a tree "
            "written before a downscale or voxel-size conversion was applied "
            "to the splat centers."
        )
        remedy = "Rescale the stored planes into the parts' coordinate frame."
    else:
        summary = f"split planes for {n_parts} parts fall outside their overlap bands"
        detail = (
            "The parts' centers do not straddle the stored planes even after "
            "allowing the measured overlap band. The plane positions can be "
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
            "allowing the measured overlap band, so the viewer "
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
ALL_CHECKS: "List[Check]" = [check_partition_split_planes]
