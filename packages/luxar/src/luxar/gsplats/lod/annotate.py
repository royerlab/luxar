"""In-place Q·e quality annotation of an existing ``.gsplats.zarr`` store.

Legacy datasets predate the build-time quality stamps
(:mod:`luxar.gsplats.lod.quality`), so the viewer falls back to committed-*count*
crossovers for LOD upgrade decisions — the currency the sibling-aware ladder
work showed is structurally late on shared-base stream ladders. This module
retrofits the stamps **without refitting or re-laddering**:

* ``lod_stats.energy_fraction_cum`` per additive sub-LOD — the cumulative
  self-energy fraction ``e(k)`` of the committed prefix. Cheap: an O(N) pass
  over the ALPHA-EFFECTIVE amplitudes (``A·α`` — RGBA color-alpha folded in,
  matching the build path's ``effective_amplitudes``) + Cholesky factors (the
  on-disk order **is** the ladder order). The same covariance decode supplies
  the per-level footprint stamp without materializing full splat objects.
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
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import zarr

from luxar._zarr_compat import consolidate, open_group
from luxar.gsplats.utils.spatial_axes import spatial_axes_from_max_sigma
from luxar.utils.lod_methods import is_reveal_method

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
    #: Empty when a nonempty leaf has zero effective energy (no stamp written,
    #: matching the build path).
    energy_fraction_cum: List[float]
    #: Absolute self-energy weight ``w`` (``Σ aᵢ²·π^{D/2}·|Σᵢ|^{1/2}``), or
    #: ``None`` when no weight was written — a REVEAL ladder carries neither half
    #: of the e/w pair, and the report must not name a number the store does not
    #: hold.
    reference_energy: Optional[float]


@dataclass(frozen=True)
class LevelStamp:
    """The measured Q stamp for one lod-group child (``with_quality`` only)."""

    path: str
    n_splats: int
    quality: float
    #: The group-consistent w: the FINEST child's total self-energy. ``None``
    #: for a reveal child, which carries no weight (see :class:`LeafStamp`).
    reference_energy: Optional[float]


@dataclass
class AnnotateReport:
    """Everything :func:`annotate_quality_store` computed (and, unless
    ``dry_run``, wrote)."""

    path: str
    dry_run: bool
    leaves: List[LeafStamp] = field(default_factory=list)
    levels: List[LevelStamp] = field(default_factory=list)


# ── per-chunk self-energy (alpha-effective amplitudes + Cholesky diagonal) ──


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


def _apply_alpha_effective(
    amps: np.ndarray, group: zarr.Group, root: zarr.Group, decoder: Any
) -> np.ndarray:
    """Fold the per-splat color-alpha opacity into ``amps`` (``A·α``), mirroring
    :func:`luxar.gsplats.utils.alpha.effective_amplitudes` on the zarr store.

    Only RGBA colors (2-D, ``shape[1] == 4``) with a row count matching ``amps``
    contribute an alpha factor; anything else (no ``colors`` array, RGB, a shape
    mismatch) leaves ``amps`` raw — a no-op for fitted (non-RGBA) data.
    """
    from luxar.gsplats.utils.alpha import apply_alpha_to_amplitudes

    if "colors" not in group:
        return amps
    colors = np.asarray(decoder.decode(group["colors"], root))
    # Row-count guard the object path never needs: a store could carry a stale
    # colors array; a mismatch means the alpha is not per-splat, so stay raw.
    if colors.shape[0] != amps.shape[0]:
        return amps
    return apply_alpha_to_amplitudes(amps, colors)


def _chunk_self_energy(
    group: zarr.Group, root: zarr.Group, decoder: Any
) -> Tuple[float, int, int, np.ndarray]:
    """One splat set's raw self-energy sum ``Σ aᵢ²·|Π diagᵢ|`` (no ``π^{D/2}``
    constant — it cancels in fractions and is re-applied for the absolute w).

    ``aᵢ`` is the ALPHA-EFFECTIVE amplitude ``A·α`` (RGBA color-alpha folded in
    via :func:`_apply_alpha_effective`, matching the build path's
    ``effective_amplitudes``); equal to the raw amplitude for non-RGBA data.

    Returns ``(raw_energy, n_splats, ndim, marginal_sigmas)``.
    """
    amps = np.asarray(decoder.decode(group["amplitudes"], root), dtype=np.float64)
    if amps.size == 0:
        return 0.0, 0, 0, np.empty((0, 0), dtype=np.float64)
    amps = _apply_alpha_effective(amps, group, root, decoder)
    from luxar.gsplats.utils.trils import unpack_tril
    from luxar.io._compiler.gsplat_tree import _decode_cholesky

    chol = np.asarray(_decode_cholesky(group, root, decoder), dtype=np.float64)
    ndim = int((math.isqrt(8 * chol.shape[1] + 1) - 1) // 2)
    unpacked = unpack_tril(chol, ndim)
    diag = np.diagonal(unpacked, axis1=1, axis2=2)
    marginal_sigmas = np.sqrt(np.sum(unpacked**2, axis=2))
    raw = float(np.sum(amps**2 * np.abs(np.prod(diag, axis=1))))
    return raw, int(amps.size), ndim, marginal_sigmas


def _stamp_level_footprint(
    group: zarr.Group, sigma_chunks: List[np.ndarray], *, dry_run: bool
) -> None:
    """Stamp one LOD child's median scale from its decoded covariance chunks."""
    nonempty = [sigmas for sigmas in sigma_chunks if sigmas.size]
    if not nonempty:
        return
    max_sigma = np.maximum.reduce([sigmas.max(axis=0) for sigmas in nonempty])
    axes = spatial_axes_from_max_sigma(max_sigma)
    footprints = [
        np.exp(np.mean(np.log(np.clip(sigmas[:, axes], 1e-12, None)), axis=1))
        for sigmas in nonempty
    ]
    finite = np.concatenate(footprints)
    finite = finite[np.isfinite(finite) & (finite > 0)]
    if finite.size:
        _merge_attr_dict(
            group,
            "level_stats",
            {
                "median_footprint": float(np.median(finite)),
                "footprint_dims": axes.tolist(),
            },
            dry_run=dry_run,
        )


def _merge_attr_dict(
    group: zarr.Group, key: str, updates: Dict[str, Any], *, dry_run: bool
) -> None:
    """Merge ``updates`` into the existing ``group.attrs[key]`` dict, through
    the writer's JSON-safety filter (NaN/Inf guarded)."""
    if dry_run:
        return
    from luxar.typing_utils.json_safe import json_safe_value

    existing = group.attrs.get(key, {})
    merged = dict(existing) if isinstance(existing, dict) else {}
    for k, v in updates.items():
        ok, safe = json_safe_value(v)
        if ok:
            merged[k] = safe
    group.attrs[key] = merged


def _ladder_sub_groups(group: zarr.Group) -> List[zarr.Group]:
    """The additive sub-LOD groups of a leaf, or the leaf itself when unladdered."""
    n_additive = int(group.attrs.get("n_additive_sublods", 1))
    if n_additive > 1:
        return [group[f"additive_{i}"] for i in range(n_additive)]
    return [group]


def _ladder_is_reveal(sub_groups: List[zarr.Group]) -> bool:
    """Whether this ladder was ordered by a REVEAL method, read from its own stamps.

    The ordering method is recorded per sub-LOD as ``lod_stats.lod_method``, which
    is provenance a reveal DOES carry (only the energy keys are omitted) — so the
    store itself says whether it may be energy-stamped. Any sub-LOD reporting a
    reveal method is enough: a ladder has one ordering.
    """
    for sub in sub_groups:
        stats = sub.attrs.get("lod_stats", {})
        if isinstance(stats, dict) and is_reveal_method(str(stats.get("lod_method"))):
            return True
    return False


def _remove_attr_key(group: zarr.Group, key: str, name: str, *, dry_run: bool) -> None:
    """Remove ``name`` from the ``group.attrs[key]`` dict, keeping other keys.

    A re-annotation must ERASE a stale stamp the current data no longer earns
    (e.g. one written by an older annotate under the raw-amplitude convention
    on a leaf the build path leaves unstamped) — merging alone would silently
    preserve the wrong value."""
    if dry_run:
        return
    existing = group.attrs.get(key)
    if isinstance(existing, dict) and name in existing:
        group.attrs[key] = {k: v for k, v in existing.items() if k != name}


# ── the walk ────────────────────────────────────────────────────────────────


def _child_count(group: zarr.Group, prefix: str) -> int:
    return sum(1 for name in group if str(name).startswith(prefix))


def _resolve_e_cum(
    sub_groups: List[zarr.Group],
    raw_energies: List[float],
    total_raw: float,
    n_total: int,
) -> Optional[List[Optional[float]]]:
    """Per-sub-LOD cumulative energy fractions, or ``None`` to stamp nothing.

    Extracted from :func:`_annotate_leaf` to keep it under the C901 ratchet —
    the decision is now four-way (reveal / positive energy / empty leaf / zero
    effective energy) and each arm mirrors a specific build-path behaviour.

    ``None`` makes the caller ERASE any existing stamp rather than merely skip
    writing, so a store wrongly stamped by an earlier build is repaired.
    """
    if _ladder_is_reveal(sub_groups):
        # A REVEAL ladder (`-m radial`) is authored with NO energy stamps: the
        # viewer brightens an incomplete ladder by 1/e(k), which is backwards for
        # a partial object rendered at full brightness. This module's contract is
        # to mirror the build path exactly, and the build path omits them — so
        # stamping here would silently re-arm the very compensation the reveal
        # exists to avoid. Verified: before this guard, `annotate-quality` on a
        # radial store stamped 6/6 sub-LODs and added `reference_energy`.
        #
        # `e_cum = None` also makes the loop below ERASE any stale stamp, so a
        # store wrongly annotated by an earlier build is repaired rather than
        # merely left alone — the same repair posture the zero-energy path uses.
        e_cum = None
    elif total_raw > 0.0:
        cum = np.cumsum(raw_energies)
        # Per sub-LOD cumulative fraction. Mirror the build (additive.py's
        # `if np.isfinite(e_frac)`): a legacy inf amplitude/Cholesky gives
        # total_raw=inf and c/total_raw=nan, where the build SKIPS the stamp —
        # so mark a non-finite fraction with None (no stamp), NOT the 0.0 that
        # `min(1, max(0, nan))` would fabricate.
        fracs = [float(c / total_raw) for c in cum]
        e_cum = [(min(1.0, max(0.0, f)) if np.isfinite(f) else None) for f in fracs]
    elif n_total == 0:
        # Genuinely empty leaf: the build stamps a trivial e(k)=1.0 (additive.py
        # n==0 branch); mirror it so annotate reproduces the build byte-for-byte.
        e_cum = [1.0] * len(sub_groups)
    else:
        # Nonempty but zero effective energy (e.g. a fully transparent RGBA
        # leaf, α≡0): the build path only writes energy_fraction_cum when
        # energy_total > 0, so stamp NOTHING here to match it exactly.
        e_cum = None
    return e_cum


def _annotate_leaf(
    group: zarr.Group,
    root: zarr.Group,
    decoder: Any,
    report: AnnotateReport,
    *,
    is_lod_child: bool = False,
    dry_run: bool,
) -> None:
    """Stamp ``energy_fraction_cum`` per sub-LOD + ``reference_energy`` on one
    leaf. Only ``amplitudes`` (folded to alpha-effective ``A·α`` via any RGBA
    ``colors``) + Cholesky factors are decoded, one sub-LOD resident at a time.
    LOD children reuse those covariance chunks for their footprint stamp."""
    sub_groups = _ladder_sub_groups(group)

    raw_energies: List[float] = []
    counts: List[int] = []
    sigma_chunks: List[np.ndarray] = []
    ndim = 0
    for sub in sub_groups:
        raw, n, d, sigmas = _chunk_self_energy(sub, root, decoder)
        raw_energies.append(raw)
        counts.append(n)
        sigma_chunks.append(sigmas)
        ndim = max(ndim, d)

    if is_lod_child:
        _stamp_level_footprint(group, sigma_chunks, dry_run=dry_run)

    total_raw = float(sum(raw_energies))
    n_total = int(sum(counts))
    e_cum = _resolve_e_cum(sub_groups, raw_energies, total_raw, n_total)
    reference_energy = total_raw * math.pi ** (ndim / 2.0) if ndim else 0.0
    if not math.isfinite(reference_energy):
        # A legacy store with an inf amplitude/diagonal: the build resets w to
        # 0.0 (additive.py) so the leaf carries zero weight in aggregation
        # rather than being left unstamped — mirror it exactly.
        reference_energy = 0.0

    for i, sub in enumerate(sub_groups):
        e = e_cum[i] if e_cum is not None else None
        if e is None:
            # The build path leaves this sub-LOD unstamped (zero effective
            # energy / non-finite fraction) — erase any stale stamp a previous
            # annotate run wrote (e.g. under the old raw-amplitude convention)
            # so a re-run repairs the store instead of preserving it.
            _remove_attr_key(sub, "lod_stats", "energy_fraction_cum", dry_run=dry_run)
        else:
            _merge_attr_dict(
                sub, "lod_stats", {"energy_fraction_cum": e}, dry_run=dry_run
            )
    # The ladder's own total IS the build-time w for a standalone leaf or a
    # partition part, so OVERWRITE it — a re-run must repair a stale value
    # (e.g. the old raw-amplitude convention, ~3× off on classical RGBA
    # imports). A DIRECT lod-group child is the one exception: its build-time
    # w is the group-consistent FINEST energy (make_additive_lod /
    # make_substitutive_lod), which the e-only pass cannot compute — keep
    # setdefault semantics there (the with_quality pass overwrites it with
    # the correct group value).
    reveal = e_cum is None and _ladder_is_reveal(sub_groups)
    if reveal:
        # Both-or-neither: the sub-LODs above carry no `energy_fraction_cum`, so
        # the leaf must carry no `reference_energy` either — a weight with nothing
        # to weight is a half-written stamp, and the viewer's display gate poisons
        # the whole subtree aggregate on a half-stamped leaf. Erase rather than
        # skip, so a store wrongly annotated earlier is repaired.
        _remove_attr_key(group, "level_stats", "reference_energy", dry_run=dry_run)
        write_w = False
    elif is_lod_child:
        existing = group.attrs.get("level_stats", {})
        write_w = not (isinstance(existing, dict) and "reference_energy" in existing)
    else:
        write_w = True
    if write_w:
        _merge_attr_dict(
            group,
            "level_stats",
            {"reference_energy": reference_energy},
            dry_run=dry_run,
        )

    report.leaves.append(
        LeafStamp(
            path=str(group.path or "/"),
            n_splats=n_total,
            energy_fraction_cum=(
                [e for e in e_cum if e is not None] if e_cum is not None else []
            ),
            # `None` only for a reveal, whose weight was ERASED. The
            # `is_lod_child` skip below also leaves `write_w` False, but there
            # the leaf keeps the group-consistent weight already on disk.
            reference_energy=None if reveal else reference_energy,
        )
    )


def _node_content(node: Any) -> Any:
    """The splat content a node REPRESENTS, as one flat :class:`GSplatData`.

    * leaf → its own splats (ladder flattened);
    * partition → the concatenation of its parts (disjoint regions);
    * lod group → the content of its FINEST child (the coarser levels are
      *representations* of that content, not additional content).

    Colors are carried through both branches so the alpha-effective amplitude
    convention (``A·α``) holds uniformly for the quality/energy measurement:
    the leaf branch keeps its colors and the partition branch merges them
    through the SAME canonical helper the leaf branch uses
    (:func:`~luxar.gsplats.gsplat_data._merge_lod_colors`, via
    ``GSplatData.flattened``). That helper normalizes integer alpha to ``[0, 1]``
    and widens RGB→opaque RGBA before concatenating, so the partition reference
    matches a fresh build byte-for-byte — dropping them (or a bespoke
    ``np.concatenate``) would either score the reference on raw amplitudes (a
    ~2.8× mixed-convention error on RGBA data) or leak an unnormalized integer
    alpha (a ~255×/65535× misweight) / silently drop RGBA parts' alpha on a
    RGB+RGBA width mix.
    """
    from luxar.gsplats.gsplat_data import GSplatData, _merge_lod_colors
    from luxar.gsplats.tree import GSplatLodGroup, GSplatPartition

    if isinstance(node, GSplatLodGroup):
        return _node_content(node.children[-1])
    if isinstance(node, GSplatPartition):
        parts = [_node_content(child) for child in node.children]
        parts = [p for p in parts if p.n_splats > 0]
        if not parts:
            raise ValueError("partition has no non-empty parts")
        # Merge colors with the canonical helper (same as flattened()): it
        # normalizes integer alpha, widens RGB→opaque RGBA, and white-fills a
        # colorless part on a mixed presence — so the concatenated reference is
        # build-identical. Guard a malformed (non-2-D) part-colors array (which
        # would IndexError on the helper's ``shape[1]``) by degrading to the
        # colorless path rather than crashing.
        if any(p.colors is not None and np.asarray(p.colors).ndim != 2 for p in parts):
            colors = None
        else:
            colors = _merge_lod_colors(parts)
        return GSplatData(
            centers=np.concatenate([np.asarray(p.centers) for p in parts]),
            amplitudes=np.concatenate([np.asarray(p.amplitudes) for p in parts]),
            cholesky_factors=np.concatenate(
                [np.asarray(p.cholesky_factors) for p in parts]
            ),
            colors=colors,
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
    is_lod_child: bool = False,
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
                is_lod_child=True,
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
            level_stamp: Dict[str, Any] = {"quality": quality}
            child_is_reveal = _ladder_is_reveal(_ladder_sub_groups(child))
            if not child_is_reveal:
                # A reveal child carries no `energy_fraction_cum` (the e-only
                # pass above erased any), so it must carry no `reference_energy`
                # either — otherwise this Q pass would put back exactly the
                # half-written stamp that pass just removed. `quality` still
                # goes on: it is a standalone readout, not half of the e pair.
                level_stamp["reference_energy"] = ref_w
            _merge_attr_dict(child, "level_stats", level_stamp, dry_run=dry_run)
            report.levels.append(
                LevelStamp(
                    path=str(child.path or "/"),
                    n_splats=n_splats,
                    quality=float(quality),
                    reference_energy=None if child_is_reveal else float(ref_w),
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

    _annotate_leaf(
        group, root, decoder, report, is_lod_child=is_lod_child, dry_run=dry_run
    )


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
        (loads full splat arrays; the e(k)/w/footprint stamps alone are O(N)).
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

    root = open_group(path, mode="r" if dry_run else "r+")
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
        consolidate(root)

    return report
