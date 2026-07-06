"""In-place Q·e quality annotation of an existing ``.gsplats.zarr`` store.

Legacy datasets predate the build-time quality stamps
(:mod:`luxar.gsplats.lod.quality`), so the viewer falls back to committed-*count*
crossovers for LOD upgrade decisions — the currency the sibling-aware ladder
work showed is structurally late on shared-base stream ladders. This module
retrofits the stamps **without refitting or re-laddering**:

* ``lod_stats.energy_fraction_cum`` per additive sub-LOD — the cumulative
  self-energy fraction ``e(k)`` of the committed prefix. Cheap: an O(N) pass
  over ``amplitudes`` + the Cholesky *diagonal* only (the on-disk order **is**
  the ladder order).
* ``level_stats.reference_energy`` per leaf — the absolute self-energy weight
  ``w`` used for partition-level quality aggregation.
* ``level_stats.quality`` per lod-group child (opt-in, ``with_quality=True``) —
  the measured mixture-L² quality ``Q`` of each level vs its group's finest
  content, via :func:`~luxar.gsplats.lod.quality.mixture_quality`. This loads
  full splat arrays (the level + the finest reference both resident), so it is
  the expensive half; the free ``e(k)``/``w`` stamps alone already enable the
  viewer's energy-threshold upgrade rule.

Attrs are merged into the existing ``lod_stats`` / ``level_stats`` dicts (the
keys the reader already recovers — format-additive, no version bump). After a
non-dry run the root ``content_hash`` is re-stamped **before**
``zarr.consolidate_metadata`` (the writer's order), so the viewer's persistent
cache invalidates on the changed attrs.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Tuple

import numpy as np
import zarr

__all__ = [
    "AnnotateReport",
    "LeafStamp",
    "LevelStamp",
    "annotate_quality_store",
]


@dataclass(frozen=True)
class LeafStamp:
    """The e(k)/w stamps computed for one leaf (splat set or additive ladder)."""

    path: str
    n_splats: int
    #: Cumulative energy fraction per additive sub-LOD; last entry is 1.0.
    energy_fraction_cum: List[float]
    #: Absolute self-energy weight ``w`` (``Σ aᵢ²·π^{D/2}·|Σᵢ|^{1/2}``).
    reference_energy: float


@dataclass(frozen=True)
class LevelStamp:
    """The measured Q stamp for one lod-group child (``with_quality`` only)."""

    path: str
    n_splats: int
    quality: float
    #: The group-consistent w: the FINEST child's total self-energy.
    reference_energy: float


@dataclass
class AnnotateReport:
    """Everything :func:`annotate_quality_store` computed (and, unless
    ``dry_run``, wrote)."""

    path: str
    dry_run: bool
    leaves: List[LeafStamp] = field(default_factory=list)
    levels: List[LevelStamp] = field(default_factory=list)


# ── per-chunk self-energy (amplitudes + Cholesky diagonal only) ─────────────


def _decode_diag(group: zarr.Group, root: zarr.Group, decoder: Any) -> np.ndarray:
    """Decode the Cholesky diagonal ``(N, d)`` — the only covariance part the
    self-energy needs (``|Σ|^{1/2} = |Π diag|``).

    v3.1 stores the diagonal as its own ``cholesky_factors_diag`` array (read
    directly — the off-diagonal never leaves disk); older stores fall back to
    recombining the full tril and slicing the diagonal columns.
    """
    if "cholesky_factors_diag" in group:
        return np.asarray(decoder.decode(group["cholesky_factors_diag"], root))
    from luxar.io._compiler.gsplat_tree import _decode_cholesky

    chol = np.asarray(_decode_cholesky(group, root, decoder))
    # tril width k = d(d+1)/2 → d; diagonal at column indices cumsum(1..d)-1.
    d = int((math.isqrt(8 * chol.shape[1] + 1) - 1) // 2)
    idx = np.cumsum(np.arange(1, d + 1)) - 1
    return chol[:, idx]


def _chunk_self_energy(
    group: zarr.Group, root: zarr.Group, decoder: Any
) -> Tuple[float, int, int]:
    """One splat set's raw self-energy sum ``Σ aᵢ²·|Π diagᵢ|`` (no ``π^{D/2}``
    constant — it cancels in fractions and is re-applied for the absolute w).

    Returns ``(raw_energy, n_splats, ndim)``.
    """
    amps = np.asarray(decoder.decode(group["amplitudes"], root), dtype=np.float64)
    if amps.size == 0:
        return 0.0, 0, 0
    diag = np.asarray(_decode_diag(group, root, decoder), dtype=np.float64)
    raw = float(np.sum(amps**2 * np.abs(np.prod(diag, axis=1))))
    return raw, int(amps.size), int(diag.shape[1])


def _merge_attr_dict(
    group: zarr.Group, key: str, updates: Dict[str, Any], *, dry_run: bool
) -> None:
    """Merge ``updates`` into the existing ``group.attrs[key]`` dict, through
    the writer's JSON-safety filter (NaN/Inf guarded)."""
    if dry_run:
        return
    from luxar.io._compiler.gsplat_tree import json_safe_value

    existing = group.attrs.get(key, {})
    merged = dict(existing) if isinstance(existing, dict) else {}
    for k, v in updates.items():
        ok, safe = json_safe_value(v)
        if ok:
            merged[k] = safe
    group.attrs[key] = merged


# ── the walk ────────────────────────────────────────────────────────────────


def _child_count(group: zarr.Group, prefix: str) -> int:
    return sum(1 for name in group if str(name).startswith(prefix))


def _annotate_leaf(
    group: zarr.Group,
    root: zarr.Group,
    decoder: Any,
    report: AnnotateReport,
    *,
    dry_run: bool,
) -> None:
    """Stamp ``energy_fraction_cum`` per sub-LOD + ``reference_energy`` on one
    leaf. Only ``amplitudes`` + the Cholesky diagonal are decoded, one sub-LOD
    resident at a time."""
    n_additive = int(group.attrs.get("n_additive_sublods", 1))
    sub_groups = (
        [group[f"additive_{i}"] for i in range(n_additive)]
        if n_additive > 1
        else [group]
    )

    raw_energies: List[float] = []
    counts: List[int] = []
    ndim = 0
    for sub in sub_groups:
        raw, n, d = _chunk_self_energy(sub, root, decoder)
        raw_energies.append(raw)
        counts.append(n)
        ndim = max(ndim, d)

    total_raw = float(sum(raw_energies))
    if total_raw > 0.0:
        cum = np.cumsum(raw_energies)
        e_cum = [min(1.0, max(0.0, float(c / total_raw))) for c in cum]
    else:
        # Empty / zero-energy leaf: every prefix trivially carries all of the
        # (zero) energy — mirrors the build-side empty-leaf stamp.
        e_cum = [1.0] * len(sub_groups)
    reference_energy = total_raw * math.pi ** (ndim / 2.0) if ndim else 0.0

    for sub, e in zip(sub_groups, e_cum):
        _merge_attr_dict(sub, "lod_stats", {"energy_fraction_cum": e}, dry_run=dry_run)
    # The ladder's own total is the FALLBACK w (setdefault semantics): a lod
    # group's quality pass overwrites its children with the group-consistent
    # finest energy — matching make_additive_lod / make_substitutive_lod.
    existing = group.attrs.get("level_stats", {})
    if not (isinstance(existing, dict) and "reference_energy" in existing):
        _merge_attr_dict(
            group,
            "level_stats",
            {"reference_energy": reference_energy},
            dry_run=dry_run,
        )

    report.leaves.append(
        LeafStamp(
            path=str(group.path or "/"),
            n_splats=int(sum(counts)),
            energy_fraction_cum=e_cum,
            reference_energy=reference_energy,
        )
    )


def _node_content(node: Any) -> Any:
    """The splat content a node REPRESENTS, as one flat :class:`GSplatData`.

    * leaf → its own splats (ladder flattened);
    * partition → the concatenation of its parts (disjoint regions);
    * lod group → the content of its FINEST child (the coarser levels are
      *representations* of that content, not additional content).

    Colors are dropped — the L² quality measurement is geometry+amplitude only.
    """
    from luxar.gsplats.gsplat_data import GSplatData
    from luxar.gsplats.tree import GSplatLodGroup, GSplatPartition

    if isinstance(node, GSplatLodGroup):
        return _node_content(node.children[-1])
    if isinstance(node, GSplatPartition):
        parts = [_node_content(child) for child in node.children]
        parts = [p for p in parts if p.n_splats > 0]
        if not parts:
            raise ValueError("partition has no non-empty parts")
        return GSplatData(
            centers=np.concatenate([np.asarray(p.centers) for p in parts]),
            amplitudes=np.concatenate([np.asarray(p.amplitudes) for p in parts]),
            cholesky_factors=np.concatenate(
                [np.asarray(p.cholesky_factors) for p in parts]
            ),
        )
    return GSplatData.from_tree(node).flattened()


def _load_flat(group: zarr.Group, root: zarr.Group, decoder: Any) -> Any:
    """Materialize the content one node subtree represents (see
    :func:`_node_content`)."""
    from luxar.io._compiler.gsplat_tree import read_gsplat_node

    return _node_content(read_gsplat_node(group, root, decoder))


def _stamp_finest_leaves(group: zarr.Group, root: zarr.Group, *, dry_run: bool) -> None:
    """Stamp ``quality: 1.0`` on every leaf of a finest-child subtree (they ARE
    the reference content) — mirrors the build-side overview stamping."""
    kind = group.attrs.get("kind")
    if kind == "lod":
        # A nested lod group's own finest child is the finest content here.
        n = _child_count(group, "child_")
        if n:
            _stamp_finest_leaves(group[f"child_{n - 1}"], root, dry_run=dry_run)
        return
    if kind == "partition":
        for i in range(_child_count(group, "part_")):
            _stamp_finest_leaves(group[f"part_{i}"], root, dry_run=dry_run)
        return
    _merge_attr_dict(group, "level_stats", {"quality": 1.0}, dry_run=dry_run)


def _annotate_node(
    group: zarr.Group,
    root: zarr.Group,
    decoder: Any,
    report: AnnotateReport,
    *,
    with_quality: bool,
    max_pair_splats: int,
    device: str,
    dry_run: bool,
) -> None:
    kind = group.attrs.get("kind")

    if kind == "lod":
        n = _child_count(group, "child_")
        children = [group[f"child_{i}"] for i in range(n)]
        for child in children:
            _annotate_node(
                child,
                root,
                decoder,
                report,
                with_quality=with_quality,
                max_pair_splats=max_pair_splats,
                device=device,
                dry_run=dry_run,
            )
        if not with_quality or n == 0:
            return

        # Q pass: measure every coarser child against the group's FINEST child
        # (child_{n-1}; on-disk children are coarsest-first). This is the one
        # place full arrays load — the reference plus one level at a time.
        from luxar.gsplats.lod.quality import mixture_quality, total_self_energy

        ref = _load_flat(children[-1], root, decoder)
        ref_w = total_self_energy(ref)
        for i, child in enumerate(children):
            if i == n - 1:
                quality = 1.0
                n_splats = int(ref.n_splats)
                # Every leaf of the finest subtree is reference content.
                _stamp_finest_leaves(child, root, dry_run=dry_run)
            else:
                level = _load_flat(child, root, decoder)
                n_splats = int(level.n_splats)
                quality = mixture_quality(
                    level, ref, max_pair_splats=max_pair_splats, device=device
                ).quality
            _merge_attr_dict(
                child,
                "level_stats",
                {"quality": quality, "reference_energy": ref_w},
                dry_run=dry_run,
            )
            report.levels.append(
                LevelStamp(
                    path=str(child.path or "/"),
                    n_splats=n_splats,
                    quality=float(quality),
                    reference_energy=float(ref_w),
                )
            )
        return

    if kind == "partition":
        for i in range(_child_count(group, "part_")):
            _annotate_node(
                group[f"part_{i}"],
                root,
                decoder,
                report,
                with_quality=with_quality,
                max_pair_splats=max_pair_splats,
                device=device,
                dry_run=dry_run,
            )
        return

    _annotate_leaf(group, root, decoder, report, dry_run=dry_run)


def annotate_quality_store(
    path: str | Path,
    *,
    with_quality: bool = False,
    max_pair_splats: int = 2_000_000,
    device: str = "auto",
    dry_run: bool = False,
) -> AnnotateReport:
    """Annotate an existing ``.gsplats.zarr`` **directory** store in place.

    Parameters
    ----------
    path
        A ``.gsplats.zarr`` directory (compressed ``.zip``/``.tar.gz`` stores
        are rejected — extraction is temp-dir based, so in-place is impossible).
    with_quality
        Also measure per-level Q vs each lod group's finest content
        (loads full splat arrays; the e(k)/w stamps alone are O(N) cheap).
    max_pair_splats, device
        Forwarded to :func:`~luxar.gsplats.lod.quality.mixture_quality`.
    dry_run
        Compute and report everything, write nothing.

    Returns
    -------
    AnnotateReport
        The computed stamps per leaf (e(k), w) and per lod-group child (Q).
    """
    from luxar.encoding import ArrayDecoder
    from luxar.gsplats.io.save_gsplats import _stamp_content_hash

    path = Path(path)
    if not path.is_dir():
        raise ValueError(
            f"annotate requires an uncompressed .gsplats.zarr directory; "
            f"got {path} (unpack .zip/.tar.gz stores first — in-place "
            f"annotation of a compressed archive is impossible)"
        )

    root = zarr.open_group(str(path), mode="r" if dry_run else "r+")
    fmt = root.attrs.get("format_type")
    if fmt != "gsplats_zarr":
        raise ValueError(
            f"{path} is not a standalone .gsplats.zarr store "
            f"(format_type={fmt!r}); scene-embedded gsplats are annotated by "
            f"re-exporting the scene"
        )

    report = AnnotateReport(path=str(path), dry_run=dry_run)
    _annotate_node(
        root,
        root,
        ArrayDecoder(),
        report,
        with_quality=with_quality,
        max_pair_splats=max_pair_splats,
        device=device,
        dry_run=dry_run,
    )

    if not dry_run:
        # The writer's finalize order: hash BEFORE consolidating, so the new
        # hash lands inside .zmetadata too and the viewer's OPFS cache
        # invalidates on the changed attrs.
        _stamp_content_hash(root)
        zarr.consolidate_metadata(root.store)

    return report
