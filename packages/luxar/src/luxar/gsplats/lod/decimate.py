"""Reduce a fitted gsplat dataset to a target splat count.

The LOD machinery next door builds *structures* — a substitutive ladder, an
additive ordering, a partition. This module answers the blunter question people
actually ask of a fitted dataset: **"this is bigger than I need; make it
smaller."** The result is a single flat :class:`GSplatData`, not a tree.

Two families, and the difference between them is worth more than the count you
pick:

``prefix``
    Keep the first ``target`` splats of an additive ORDERING (:func:`~luxar.
    gsplats.lod.additive.compute_additive_order`). Splats are *discarded*, so
    the object loses mass and dims. Cheap — the ordering is O(N log N) and is
    often already stored in the file.

``merge``
    Substitutive reduction: neighbouring splats are MERGED into representatives
    carrying their combined mass, so nothing is thrown away, it is summarised.
    Costs a clustering pass, and preserves brightness.

WHICH TO USE — measured, not asserted. On a 1.65M-splat light-sheet fit of a
zebrafish embryo, scored as foreground PSNR against the source volume (PSNR
restricted to signal voxels — a global PSNR over a 97.8%-empty stack mostly
measures how well a scheme reproduces black, which both do perfectly):

    kept      merge      prefix(self_energy)
    50%       44.52 dB   45.48 dB      <- prefix wins
    25%       41.66 dB   39.11 dB
    10%       38.28 dB   34.46 dB
     5%       36.21 dB   32.39 dB
     1%       33.10 dB   29.58 dB      <- merge wins by 3.5 dB

Below roughly half, merging leads by 3-4 dB at equal count: merge at 10% matches
what a prefix needs ~40% to reach. Above half the ranking inverts — there is
little redundancy left to summarise and merging only blurs, while the prefix is
free. :data:`PREFIX_ABOVE_FRACTION` encodes that crossover and is what
``method="auto"`` follows.

Quality falls smoothly (~3-4 dB per halving) with no knee, so there is no single
"correct" reduction — pick from the curve for the use you have in mind.
"""

from __future__ import annotations

from typing import Literal, Optional, Sequence, Union

import numpy as np
from arbol import aprint, asection

from luxar.gsplats._data.filtering import (
    scrub_measured_stats,
    stats_after_structure_change,
)
from luxar.gsplats.gsplat_data import GSplatData
from luxar.gsplats.lod.additive import compute_additive_order
from luxar.gsplats.lod.substitutive import merge_to_count
from luxar.utils.lod_methods import AutoOrMethod as AdditiveOrdering

#: Which reduction FAMILY to use — distinct from `AdditiveOrdering`, which
#: selects the ordering the `prefix` family then takes from.
MethodName = Literal["merge", "prefix"]
AutoOrMethod = Union[MethodName, Literal["auto"]]

#: Above this kept-fraction ``method="auto"`` picks ``prefix``, below it picks
#: ``merge`` — the measured crossover (see the module docstring). It is a
#: fraction of the INPUT count, so it tracks how much redundancy is left to
#: summarise rather than any absolute size.
PREFIX_ABOVE_FRACTION = 0.5


def resolve_target_count(target: Union[int, float], n_in: int) -> int:
    """Resolve a ``target`` to an absolute splat count.

    ``target`` is a fraction of the input when given as a float in ``(0, 1]``,
    and an absolute count when given as an int. ``1`` is therefore one splat and
    ``1.0`` is "keep everything" — the int/float distinction carries the meaning,
    which is why this is a named function with its own tests rather than an
    inline cast.
    """
    if isinstance(target, bool):  # bool is an int subclass; never a valid target
        raise TypeError("target must be an int count or a float fraction, not bool")
    if isinstance(target, float):
        if not (0.0 < target <= 1.0):
            raise ValueError(
                f"A float target is a FRACTION and must be in (0, 1]; got {target}. "
                "Pass an int for an absolute splat count."
            )
        n = int(round(n_in * target))
    else:
        n = int(target)
        if n < 1:
            raise ValueError(
                f"An int target is a splat COUNT and must be >= 1; got {n}"
            )
    return max(1, min(n, n_in))


def resolved_merge_coarsen_dims(
    coarsen_dims: Optional[Sequence[int]], ndim: int
) -> Optional[list[int]]:
    """The dims a ``merge`` ACTUALLY coarsens over, in the producer's spelling.

    Mirrors :func:`~luxar.gsplats.lod.substitutive._normalise_coarsen_dims`,
    which ``merge_to_count`` has already run (and validated) by the time this is
    called: ``None`` means "coarsen every dim, no barrier", and a request naming
    every dim collapses to the same thing.

    Spelled as :func:`~luxar.gsplats.lod.substitutive.make_substitutive_lod`
    spells it — the sibling producer of the same operator — so a store that went
    through ``lod`` and one that went through ``decimate`` publish the same value
    for the same choice. That matters concretely: ``make_substitutive_lod``
    stamps a literal ``None`` for coarsen-everything, which the writer reads as
    "no barrier provenance → auto-detect", and stamping ``[0, …, d-1]`` here
    instead would assert an explicit EMPTY barrier and move every chunk of an
    otherwise unchanged pipeline.
    """
    if coarsen_dims is None:
        return None
    dims = sorted({int(d) for d in coarsen_dims})
    return None if len(dims) == ndim else dims


def resolve_method(method: AutoOrMethod, n_target: int, n_in: int) -> MethodName:
    """Resolve ``"auto"`` against the measured crossover (see module docstring)."""
    if method != "auto":
        if method not in ("merge", "prefix"):
            raise ValueError(
                f"Unknown decimation method {method!r}; use merge/prefix/auto"
            )
        return method
    return "prefix" if n_target >= PREFIX_ABOVE_FRACTION * n_in else "merge"


def decimate(
    data: GSplatData,
    *,
    target: Union[int, float],
    method: AutoOrMethod = "auto",
    prefix_method: AdditiveOrdering = "auto",
    device: Union[str, None] = "auto",
    seed: Optional[int] = None,
    coarsen_dims: Optional[Sequence[int]] = None,
    lloyd_iterations: int = 5,
    verbose: bool = False,
) -> GSplatData:
    """Reduce *data* to ``target`` splats and return a flat dataset.

    Args:
        data: Source dataset. A multi-level input is reduced from its finest
            content (the same convention :func:`make_substitutive_lod` uses).
        target: Absolute count (``int``) or fraction of the input (``float`` in
            ``(0, 1]``). See :func:`resolve_target_count`.
        method: ``"merge"``, ``"prefix"``, or ``"auto"`` (the measured rule —
            see the module docstring).
        prefix_method: Ordering for ``method="prefix"``, passed to
            :func:`compute_additive_order` (``auto`` / ``self_energy`` /
            ``mass`` / ``greedy`` / ``radial`` / ...).
        device: Device for the clustering pass (merge only).
        seed: Seed for the ``random`` ordering (prefix only; every other
            ordering, and the clustering, is deterministic).
        coarsen_dims: Center-column indices merging may combine over; the rest
            are hard barriers (merge only). Default: all dims. A ``merge``
            stamps the RESOLVED set on the result (the writer turns it into the
            chunk-ordering barrier); a ``prefix`` ignores the argument and keeps
            the input's stamp, having coarsened nothing.
        lloyd_iterations: Lloyd refinement passes (merge only).
        verbose: Narrate the reduction.

    Returns:
        A flat :class:`GSplatData` with ``<= target`` splats, and close to it.
        Returns the input unchanged when ``target`` resolves to the full count.
        ``merge`` can land slightly under the request — the clustering drops
        degenerate (empty / non-positive-mass) clusters, so a 165,340 ask on the
        1.65M-splat reference dataset yields 165,276. The one case that lands
        OVER is a ``coarsen_dims`` target below the number of barrier groups:
        every group keeps at least one representative rather than whole
        timepoints/channels being deleted to hit a count (the reduction says so
        on the console).

    Raises:
        ValueError: on an out-of-range target or an unknown method.
    """
    n_in = int(data.n_splats)
    n_target = resolve_target_count(target, n_in)
    chosen = resolve_method(method, n_target, n_in)

    if n_target >= n_in:
        if verbose:
            aprint(f"Target {n_target:,} >= input {n_in:,} — returning input unchanged")
        return data

    with (
        asection(
            f"Decimating {n_in:,} -> {n_target:,} splats "
            f"({100.0 * n_target / n_in:.1f}%, method={chosen})"
        )
        if verbose
        else _null_section()
    ):
        if chosen == "prefix":
            order = compute_additive_order(data, method=prefix_method, seed=seed)
            keep = np.asarray(order)[:n_target]
            out = _subset(data, keep)
        else:
            # Ask the merge for the requested count DIRECTLY. The obvious route
            # — `make_substitutive_lod` — reduces by an INTEGER per-level factor,
            # so the counts it can land on are quantised (N/2, N/3, N/4, ...) and
            # a request generally falls between two of them: ceil(N/target)
            # undershoots (a 10% ask on 1.65M returned 9.1%), and floor(N/target)
            # overshoots and would need the surplus trimmed away — which throws
            # out representatives that carry a whole cluster's mass, dimming the
            # object by up to a third and losing exactly the property merging
            # exists for. `merge_to_count` asks the same operator for M = target
            # bins instead, so nothing is discarded and any target is reachable
            # (a factor >= 2 could never honour a target above half the input).
            out = merge_to_count(
                data,
                n_target=n_target,
                lloyd_iterations=lloyd_iterations,
                device=device,
                coarsen_dims=coarsen_dims,
            )
        # `_subset` / `merge_to_count` build from bare arrays, so a reduction used
        # to arrive with NO provenance at all — dropping even the descriptive keys
        # (`fitter_name`, `iterations`, `time_seconds`, the source grid) that no
        # reduction makes false, and leaving the scrub below nothing to scrub.
        # Carry the input's stats over first, so the rule below is what decides
        # what survives instead of an accident of how the arrays were rebuilt.
        #
        # Then: BOTH families reach here having changed which splats the artifact
        # holds (the one path that returns the input verbatim already returned
        # above, so this is unconditional) — a prefix discards the tail, a merge
        # replaces neighbours with representatives — so the fit's measured
        # reconstruction scores, and the input's own cull/reduction record, no
        # longer describe it. The merge family is the case a count-based predicate
        # would miss: it lands on exactly the count the user asked for while every
        # surviving splat is a new one. The region stamps are NOT touched: neither
        # family is a spatial restriction, so the survivors still represent the
        # whole fitted volume.
        #
        # The input's TOPOLOGY record goes with them, and it goes HERE rather than
        # in the CLI because the RETURN TYPE is the reason: this function's
        # contract is one flat leaf whatever it was handed, so an inherited
        # `lod_kind: substitutive` / `n_substitutive_levels: 4` /
        # `lod_cutpoints: [...]` is false of every result it can produce — no
        # caller can want it kept. Scrubbing in the command instead left the
        # public `luxar.gsplats.lod.decimate` API publishing the defect (#1600).
        # `coarsen_dims` is exempt from that scrub (the writer reads it back to
        # derive the chunk-ordering barrier) — see _STRUCTURE_SCOPE_EXEMPT_KEYS —
        # and is RE-STAMPED just below instead.
        out = GSplatData.from_tree(
            out.tree, stats=stats_after_structure_change(data.stats)
        )
        if chosen == "merge":
            # The one inherited pipeline key this reduction OWNS. The writer
            # derives the ordering barrier from its complement
            # (`_barrier_from_coarsen_dims`), so an inherited `[0, 1, 2]` over a
            # merge told `--coarsen-dims 1,2,3` — or over the default, which
            # coarsens EVERYTHING — puts the barrier on an axis this merge just
            # blended, and the absent-stamp direction silently fell back to
            # auto-detect instead of recording the barrier the user asked for
            # (#1600). Stamped unconditionally: the merge decides these dims
            # whether or not the input had an opinion.
            out.stats["coarsen_dims"] = resolved_merge_coarsen_dims(
                coarsen_dims, data.ndim
            )
        # A `prefix` KEEPS the input's stamp. It merges nothing — every surviving
        # splat is one of the input's, at its own coordinates — so whichever axes
        # were hard barriers for this splat set still are, and the inherited value
        # (and the layout the writer derives from it) stays true of the result.
        # Stamping the request would be the lie here: `coarsen_dims` is a
        # merge-only knob that this family ignored.
        scrub_measured_stats(out)
        from luxar.gsplats.lod.restamp import refresh_reduction_lod_stats

        out = refresh_reduction_lod_stats(out, data)
        if verbose:
            aprint(f"Result: {out.n_splats:,} splats")
        return out


def _subset(data: GSplatData, idx: np.ndarray) -> GSplatData:
    """A flat dataset holding only the splats at *idx*, colours carried along."""
    colors = data.colors
    return GSplatData(
        centers=np.asarray(data.centers)[idx],
        amplitudes=np.asarray(data.amplitudes)[idx],
        cholesky_factors=np.asarray(data.cholesky_factors)[idx],
        colors=None if colors is None else np.asarray(colors)[idx],
        truncation_radius=data.truncation_radius,
    )


class _null_section:
    """A no-op stand-in for ``asection`` when not narrating."""

    def __enter__(self) -> None:
        return None

    def __exit__(self, *exc: object) -> Literal[False]:
        return False
