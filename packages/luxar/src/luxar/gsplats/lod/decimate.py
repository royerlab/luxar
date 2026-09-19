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

import warnings
from typing import Literal, Optional, Sequence, Union

import numpy as np
from arbol import aprint, asection

from luxar.gsplats._data.filtering import (
    scrub_measured_stats,
    stats_after_structure_change,
)
from luxar.gsplats.gsplat_data import GSplatData
from luxar.gsplats.lod.additive import compute_additive_order

# `resolved_merge_coarsen_dims` is imported rather than defined here: it moved
# next to `_normalise_coarsen_dims` (the collapse it compensates for) when
# `make_substitutive_lod` and the `batch-fit merge` per-part record started
# sharing it, so the three producers of the `coarsen_dims` stamp cannot drift
# apart again. Still reachable under its original
# `luxar.gsplats.lod.decimate` name.
from luxar.gsplats.lod.substitutive import merge_to_count, resolved_merge_coarsen_dims
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


def _validate_coarsen_dims(
    coarsen_dims: Optional[Sequence[int]], data: GSplatData
) -> None:
    """Range-check a ``coarsen_dims`` request for EITHER family.

    Reuses the merge's own validator rather than restating its rules, so the two
    families reject exactly the same requests. Only ``merge`` reaches that
    validator on its own (inside ``merge_to_count``), which left an out-of-range
    index a hard error on one family and silently accepted on the other — and
    with ``method="auto"`` which family you get depends on the kept fraction.

    Validation ONLY: the merge path re-runs the same call for real, and the
    continuous-barrier ``RuntimeWarning`` belongs to the reduction that actually
    groups by those dims, so it is suppressed here rather than emitted twice.
    """
    if coarsen_dims is None:
        return
    from luxar.gsplats.lod.substitutive import _normalise_coarsen_dims

    with warnings.catch_warnings():
        warnings.simplefilter("ignore", RuntimeWarning)
        _normalise_coarsen_dims(coarsen_dims, data)


def resolve_method(method: AutoOrMethod, n_target: int, n_in: int) -> MethodName:
    """Resolve ``"auto"`` against the measured crossover (see module docstring)."""
    if method != "auto":
        if method not in ("merge", "prefix"):
            raise ValueError(
                f"Unknown decimation method {method!r}; use merge/prefix/auto"
            )
        return method
    return "prefix" if n_target >= PREFIX_ABOVE_FRACTION * n_in else "merge"


def _resolve_labeled_method(
    data: GSplatData, method: AutoOrMethod, chosen: MethodName
) -> tuple[MethodName, bool]:
    """Keep labeled ``auto`` conservative while permitting explicit merge."""
    label_override = method == "auto" and data.label_ids is not None
    if label_override:
        warnings.warn(
            "method='auto' selected 'prefix' because the input carries "
            "categorical channel 'label_ids'; prefix preserves exact input rows "
            "by default. Pass method='merge' to coarsen within exact label groups",
            UserWarning,
            stacklevel=3,
        )
        return "prefix", True
    return chosen, False


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
            see the module docstring). Labeled inputs constrain ``"auto"`` to
            ``"prefix"`` as the conservative default; explicit ``"merge"``
            coarsens independently within exact label groups.
        prefix_method: Ordering for ``method="prefix"``, passed to
            :func:`compute_additive_order` (``auto`` / ``self_energy`` /
            ``mass`` / ``greedy`` / ``radial`` / ...).
        device: Device for the clustering pass (merge only).
        seed: Seed for the ``random`` ordering (prefix only; every other
            ordering, and the clustering, is deterministic).
        coarsen_dims: Center-column indices merging may combine over; the rest
            are hard barriers (merge only). Default: all dims. A ``merge``
            stamps the RESOLVED set on the result (the writer turns it into the
            chunk-ordering barrier); a ``prefix`` ignores the argument (a
            ``UserWarning``, so the notice survives ``verbose=False``) and keeps
            the input's stamp, having coarsened nothing. Under the ``luxar`` CLI
            that warning renders as an arbol line like any other output.
            The request is range-validated for BOTH families, before
            the family is chosen — under ``method="auto"`` which one runs
            depends on the kept fraction, and an argument may not be a hard
            error on one path and silently accepted on the other.
        lloyd_iterations: Lloyd refinement passes (merge only).
        verbose: Narrate the reduction.

    Returns:
        A flat :class:`GSplatData` with ``<= target`` splats, and close to it.
        Returns the input unchanged when ``target`` resolves to the full count.
        ``merge`` can land slightly under the request — the clustering drops
        degenerate (empty / non-positive-mass) clusters, so a 165,340 ask on the
        1.65M-splat reference dataset yields 165,276. A request below the number
        of coordinate and/or label barrier groups lands OVER: every group keeps
        at least one representative rather than whole timepoints, channels, or
        classes being deleted to hit a count (the reduction says so on the
        console).

    Raises:
        ValueError: on an out-of-range target, an unknown method, or a
            ``coarsen_dims`` index outside ``[0, data.ndim)``.
    """
    n_in = int(data.n_splats)
    n_target = resolve_target_count(target, n_in)
    chosen = resolve_method(method, n_target, n_in)
    _validate_coarsen_dims(coarsen_dims, data)

    if n_target >= n_in:
        if verbose:
            aprint(f"Target {n_target:,} >= input {n_in:,} — returning input unchanged")
        return data

    chosen, label_override = _resolve_labeled_method(data, method, chosen)

    if coarsen_dims is not None and chosen == "prefix":
        # The family decides whether this knob means anything, and with
        # `method="auto"` the family flips at the measured crossover — so a
        # request honoured at -f 0.4 is silently dropped at -f 0.5, taking the
        # output's chunk layout with it. Say so rather than letting the caller
        # infer it from the stamp (#1600 review).
        #
        # A WARNING, not an `aprint`: this is a library function, and "your
        # argument had a surprising effect" is what `GSplatData.filter` already
        # warns for (`_data/filtering.py`). Every other line this function
        # writes is gated on `verbose`, so an unconditional print would put a
        # programmatic `decimate(..., verbose=False)` on stdout unbidden. The
        # CLI still shows it: `luxar`'s root callback installs
        # `install_arbol_warnings`, which renders warnings as arbol lines.
        if label_override:
            why = (
                " (method='auto' resolved to prefix because the input carries "
                "categorical channel 'label_ids')"
            )
        elif method == "auto":
            why = (
                f" (method='auto' resolved to prefix: the request keeps "
                f"{100.0 * n_target / n_in:.1f}% of the input, at or above the "
                f"{100.0 * PREFIX_ABOVE_FRACTION:.0f}% crossover — pass "
                f"method='merge' to force a merge)"
            )
        else:
            why = ""
        warnings.warn(
            f"coarsen_dims={sorted({int(d) for d in coarsen_dims})} is IGNORED "
            f"by the 'prefix' family{why}: a prefix keeps whole input splats at "
            "their own coordinates and merges no axis, so it coarsens nothing "
            "and the input's own coarsen_dims stamp (and the chunk-ordering "
            "barrier derived from it) stays true of the survivors",
            UserWarning,
            stacklevel=2,
        )

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
            # bins instead, so nothing is discarded and any target at or above
            # the coordinate/label barrier-group count is reachable (a factor
            # >= 2 could never honour a target above half the input).
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
            #
            # Always the EXPLICIT dim list, coarsen-everything included — a
            # written `null` is indistinguishable from an absent key to the
            # writer and lands back on auto-detect, which re-imposes a barrier
            # on a blended axis whenever the reduction left that axis' grid
            # intact (and is redundant when it did not — measured both ways in
            # `resolved_merge_coarsen_dims`). Shared with
            # `make_substitutive_lod` and the `batch-fit merge` per-part record,
            # which resolve their own stamp through the same function, so the
            # producers of this key cannot spell the same choice two ways.
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
        label_ids=(None if data.label_ids is None else np.asarray(data.label_ids)[idx]),
        label_vocabulary=data.label_vocabulary,
        truncation_radius=data.truncation_radius,
    )


class _null_section:
    """A no-op stand-in for ``asection`` when not narrating."""

    def __enter__(self) -> None:
        return None

    def __exit__(self, *exc: object) -> Literal[False]:
        return False
