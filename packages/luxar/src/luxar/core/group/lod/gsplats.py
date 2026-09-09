"""luxar.core.group.lod.gsplats – GSplats-specific LOD axis resolvers.

Interprets ``substitutive_lod=`` (historically ``lod_group=``) and
``additive_lod=``
convenience kwargs against a :class:`~luxar.gsplats.gsplat_data.GSplatData`
input, used by ``Scene.add_gsplats_from_data(...)``. These two resolvers are
the GSplats counterparts to ``lod.points.resolve_additive_axis_points`` /
``lod.lines.resolve_additive_axis_lines`` — peers, one per leaf geometry.

The geometry-agnostic machinery they build on — ``coverage_fractions``,
``validate_lod_group``, ``resolve_display_type`` and the monotonicity guard —
lives in the type-neutral :mod:`luxar.core.group.lod.group`.

:func:`resolve_additive_rungs` is a third, ordering-free member of the same
family: it applies the ``additive_lod=`` vocabulary to ONE leaf and answers only
"how many rungs would this leave?", for gates that must know whether a ladder
will exist before any data is resolved (#1632).
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any, List, Optional, Tuple, Union

from .group import MAX_COVERAGE_FRACTION, _assert_strict_ascending

if TYPE_CHECKING:
    from ....gsplats.gsplat_data import GSplatData


#: Sentinel-typed alias for the value vocabulary of ``substitutive_lod=`` and
#: ``additive_lod=``. Order: ``None`` (pass-through), ``True`` (require
#: stored), ``False`` (collapse to finest), ``dict[str, Any]`` (compute,
#: optionally with ``recompute=True``).
LODAxisSpec = Union[None, bool, dict]


# ────────────────────────────────────────────────────────────────────────
# Resolvers for the ``substitutive_lod=`` / ``additive_lod=`` convenience kwargs
# ────────────────────────────────────────────────────────────────────────


def resolve_substitutive_axis_gsplats(
    data: "GSplatData",
    spec: LODAxisSpec,
) -> Tuple["GSplatData", Optional[List[float]]]:
    """Apply ``substitutive_lod=`` semantics to ``data``.

    Returns ``(resolved_data, explicit_coverage_fractions_or_None)``.

    - ``explicit_coverage_fractions`` is non-None only when the user passed
      ``dict(coverage_fractions=[...])`` (strict-ascending, in
      ``[0, MAX_COVERAGE_FRACTION]`` — an explicit list keeps the legacy
      ``selector="coverage"`` diagonal units its values were authored in) —
      otherwise downstream code auto-derives per-level SCREEN-AREA thresholds
      via :func:`luxar.core.group.lod.group.derive_coverage_fractions`
      (occupancy halving, ``selector="screen-area"``: full detail while the
      node occupies at least half the screen, re-anchored at fills-screen
      area 1.0 when the insertion point is inside a ``kind=partition``). There
      is no per-dataset anchor knob.

    Semantics:

    +---------------------------+-----------------------------------------+
    | Spec                      | Behaviour                               |
    +===========================+=========================================+
    | ``None`` (default)        | Auto-lower: if ``n_substitutive > 1``,  |
    |                           | keep the full pyramid so it becomes a   |
    |                           | ``kind=lod`` Group (no work discarded). |
    |                           | Pass ``substitutive_lod=False`` to      |
    |                           | collapse to the finest level.          |
    |                           | the finest level instead.               |
    +---------------------------+-----------------------------------------+
    | ``True``                  | Require ``n_substitutive > 1`` already; |
    |                           | raise otherwise. Use stored.            |
    +---------------------------+-----------------------------------------+
    | ``False``                 | Collapse to finest substitutive level   |
    |                           | (index 0), discard the rest.            |
    +---------------------------+-----------------------------------------+
    | ``dict(...)``             | Use stored if present; else compute via |
    |                           | :func:`make_substitutive_lod` with the  |
    |                           | dict as kwargs.                         |
    +---------------------------+-----------------------------------------+
    | ``dict(..., recompute=    | Force recompute, ignoring stored.       |
    | True)``                   |                                         |
    +---------------------------+-----------------------------------------+
    """
    explicit_coverage_fractions: Optional[List[float]] = None

    if spec is None:
        # Auto-lower: a multi-substitutive pyramid is expensive to build, so
        # the default no longer silently drops it. Returning the full data
        # routes it to the kind=lod Group builder (same as
        # ``substitutive_lod=True``) downstream. Use
        # ``substitutive_lod=False`` to collapse to the finest level.
        return data, None

    if spec is True:
        if data.n_substitutive <= 1:
            raise ValueError(
                "substitutive_lod=True (or its lod_group=True alias) requires "
                "the input GSplatData to already "
                "carry substitutive levels (n_substitutive > 1); got "
                f"n_substitutive={data.n_substitutive}. Use "
                "substitutive_lod=dict(...) to compute them on the fly."
            )
        return data, None

    if spec is False:
        if data.n_substitutive > 1:
            # Convention: finest substitutive level is index 0.
            return data.at_substitutive(0), None
        return data, None

    if isinstance(spec, dict):
        kwargs = dict(spec)  # copy — don't mutate caller's dict
        recompute = bool(kwargs.pop("recompute", False))
        explicit_coverage_fractions = kwargs.pop("coverage_fractions", None)
        if explicit_coverage_fractions is not None:
            explicit_coverage_fractions = [
                float(v) for v in explicit_coverage_fractions
            ]
            # Fail fast on bad explicit lists at the resolver instead of
            # deferring to a later validate_lod_group() the user may never
            # call. Same invariant the auto-derivation enforces.
            if not explicit_coverage_fractions:
                raise ValueError(
                    "substitutive_lod=dict(coverage_fractions=...) must be non-empty "
                    f"(one strictly-ascending value in "
                    f"[0, {MAX_COVERAGE_FRACTION:g}] per substitutive level)"
                )
            _assert_strict_ascending(
                explicit_coverage_fractions,
                "substitutive_lod=dict(coverage_fractions=...)",
            )
            if (
                explicit_coverage_fractions[0] < 0.0
                or explicit_coverage_fractions[-1] > MAX_COVERAGE_FRACTION
            ):
                raise ValueError(
                    "substitutive_lod=dict(coverage_fractions=...): values must lie in "
                    f"[0, {MAX_COVERAGE_FRACTION:g}] (coarsest→finest); got "
                    f"{explicit_coverage_fractions}. An explicit list keeps the "
                    "legacy selector='coverage' diagonal units, whose upper "
                    "bound is SCREEN_FILL_DIAGONAL_RATIO/FILL_FACTOR — roughly "
                    "the metric a screen-filling object produces. (Omit the "
                    "list for the derived screen-area ladder.)"
                )

        if data.n_substitutive > 1 and not recompute:
            # Stored pyramid takes precedence — but silently accepting
            # compute-affecting kwargs would mask user intent. Reject any
            # leftover keys (only ``coverage_fractions`` / ``recompute`` are
            # honored on the stored path).
            if kwargs:
                raise ValueError(
                    "substitutive_lod=dict(...) carries compute kwargs "
                    f"({sorted(kwargs)}) but data already has "
                    f"n_substitutive={data.n_substitutive}. Pass "
                    "recompute=True to override the stored pyramid, or "
                    "drop the compute kwargs to reuse it."
                )
            return data, explicit_coverage_fractions

        # Compute. Align with ``make_substitutive_lod``'s canonical default
        # (3 levels) so ``substitutive_lod=dict()`` yields the same pyramid as a
        # bare ``make_substitutive_lod(data)`` call.
        kwargs.setdefault("levels", 3)
        from ....gsplats.lod.substitutive import make_substitutive_lod

        new_data = make_substitutive_lod(data, **kwargs)
        return new_data, explicit_coverage_fractions

    raise TypeError(
        "substitutive_lod (or lod_group alias) must be None, bool, or dict; "
        f"got {type(spec).__name__}"
    )


def resolve_additive_axis_gsplats(
    data: "GSplatData",
    spec: LODAxisSpec,
) -> "GSplatData":
    """Apply ``additive_lod=`` semantics uniformly across substitutive levels.

    Semantics mirror :func:`resolve_substitutive_axis_gsplats` but operate per
    substitutive level (each level's additive ladder is treated
    independently).

    :func:`resolve_additive_rungs` states this same vocabulary a second time, as
    an ordering-free rung COUNT for one leaf (#1632). The two are one contract:
    every rule in the table below is read off there, key for key, so a change
    here is a change there.

    +---------------------------+-----------------------------------------+
    | Spec                      | Behaviour                               |
    +===========================+=========================================+
    | ``None`` (default)        | Pass-through; use whatever ladder each  |
    |                           | substitutive level carries.             |
    +---------------------------+-----------------------------------------+
    | ``True``                  | Require every substitutive level to     |
    |                           | already carry ``> 1`` additive sub-LODs.|
    +---------------------------+-----------------------------------------+
    | ``False``                 | Flatten each substitutive level's       |
    |                           | additive ladder into a single sub-LOD.  |
    +---------------------------+-----------------------------------------+
    | ``dict(...)``             | Compute ladder on each level missing    |
    |                           | one; pass stored through otherwise.     |
    +---------------------------+-----------------------------------------+
    | ``dict(..., recompute=    | Force recompute on every level.         |
    | True)``                   |                                         |
    +---------------------------+-----------------------------------------+

    A stack built by ``GSplatData.combine_as_new_dimension`` can already carry
    merged source ladders, so use ``recompute=True`` when the authored spec must
    replace them. The shadowed case is stamped with ``n_sources`` and no
    ``lod_method`` in the rung's ``lod_stats``.
    """
    from ....gsplats.gsplat_data import GSplatData, SubstitutiveLevel

    if spec is None:
        return data

    if spec is True:
        for i, lvl in enumerate(data.substitutive_levels):
            if lvl.n_additive_lods <= 1:
                raise ValueError(
                    "additive_lod=True requires every substitutive level "
                    "to already carry an additive ladder "
                    "(n_additive_lods > 1); substitutive level "
                    f"{i} has n_additive_lods={lvl.n_additive_lods}. Use "
                    "additive_lod=dict(...) to compute on the fly."
                )
        return data

    if spec is False:
        # Flatten each substitutive level into a single AdditiveSubLOD.
        new_levels: list[SubstitutiveLevel] = []
        for lvl in data.substitutive_levels:
            single = (
                GSplatData(additive_sublods=list(lvl.additive_sublods))
                .flattened()
                .additive_sublods[0]
            )
            new_levels.append(
                SubstitutiveLevel(
                    additive_sublods=[single],
                    compression_factor=lvl.compression_factor,
                    parent_method=lvl.parent_method,
                    level_index=lvl.level_index,
                    stats=dict(lvl.stats),
                )
            )
        return GSplatData.from_substitutive_levels(
            new_levels,
            stats=dict(data.stats),
        )

    if isinstance(spec, dict):
        kwargs = dict(spec)
        recompute = bool(kwargs.pop("recompute", False))
        # Default to 4-level ladder for a bare ``dict()`` so the convenience
        # API yields a useful result without parameters.
        kwargs.setdefault("n_lods", 4)
        from ....gsplats.lod.additive import (
            clamp_counts_breakpoints,
            make_additive_lod,
            validate_counts_breakpoints,
        )

        # Explicit ``counts:`` breakpoints must still be sane for the WHOLE
        # group: strictly validate ONCE against the finest substitutive level
        # (== the full dataset N), so a dataset-scale typo aborts loudly —
        # only the coarser (smaller) levels clamp, in the loop below.
        if "breakpoints" in kwargs:
            validate_counts_breakpoints(
                kwargs["breakpoints"],
                data.substitutive_levels[0].n_splats_total,
            )

        result = data
        for s in range(result.n_substitutive):
            needs_compute = (
                recompute or result.substitutive_levels[s].n_additive_lods <= 1
            )
            if needs_compute:
                # Clamp explicit ``counts:`` breakpoints to THIS level's size —
                # coarser substitutive levels are smaller by K^s, so a fixed
                # counts list sized for the finest level would otherwise abort
                # the build ("largest breakpoint exceeds N"). String/energy
                # specs pass through (already size-adaptive). Mirrors the CLI
                # per-part sites (recipes._ladder_for_part, pyramid loop).
                level_kwargs = dict(kwargs)
                if "breakpoints" in level_kwargs:
                    level_kwargs["breakpoints"] = clamp_counts_breakpoints(
                        level_kwargs["breakpoints"],
                        result.at_substitutive(s).n_splats,
                    )
                result = make_additive_lod(result, substitutive_level=s, **level_kwargs)
        return result

    raise TypeError(
        f"additive_lod must be None, bool, or dict; got {type(spec).__name__}"
    )


def resolve_additive_rungs(
    # NOT ``LODAxisSpec``, unlike the resolver: this is handed the RAW
    # ``attrs.get("additive_lod")`` straight out of a caller's ``**attrs``, so
    # "anything else" is a real runtime case here (the resolver only ever sees a
    # value that has already reached its own signature). Typing it as the closed
    # union would make the final ``return None`` unreachable to mypy — and that
    # branch is precisely the one that tells a caller this vocabulary has nothing
    # to say about the spec, so the caller must fall back to what it already knows
    # rather than guess at a spec the resolver will ``TypeError`` on downstream.
    spec: Any,
    *,
    stored_rungs: int,
    n_splats: int,
) -> Optional[int]:
    """How many additive rungs would ``spec`` leave on ONE leaf / level? (#1632)

    The ordering-free counterpart of :func:`resolve_additive_axis_gsplats`,
    applying the same ``additive_lod=`` VOCABULARY to a single leaf without
    building anything. **The two are one contract stated twice** — every rule
    below is that resolver's own rule, read off the same branch — so a change to
    either is a change to both, and neither should be edited alone.

    It exists for callers that must know *whether a ladder will exist* before the
    data does: the file/graft partition-vs-ladder gate
    (:func:`~luxar.core.group.gsplats_pipeline.from_io._reject_a_partition_beside_a_stored_ladder`)
    runs above ``graft_gsplat_node``'s wrapper build, and ``additive_lod=`` is not
    even a parameter of that function — it rides in ``**attrs`` down to each
    part's own ``add_gsplats_from_data_impl``, which BUILDS the ladder there,
    after the ``kind=partition`` wrapper already exists. Judging the kwarg's mere
    PRESENCE would be wrong in both directions: ``{"n_lods": 1}`` resolves to a
    single rung and takes the flat route (``partition=`` is legitimate), while
    ``{"method": "radial"}`` has no ``n_lods`` to read and still resolves to the
    ``n_lods=4`` default — four rungs on any leaf of >= 4 splats, and ``n`` on a
    smaller one, since equal-count cuts clamp to the leaf's own size.

    +---------------------------+-----------------------------------------+
    | Spec                      | Rungs                                   |
    +===========================+=========================================+
    | ``None``                  | ``stored_rungs`` (pass-through).        |
    +---------------------------+-----------------------------------------+
    | ``True``                  | ``stored_rungs`` — ``True`` only        |
    |                           | ASSERTS a stored ladder, it never       |
    |                           | builds one.                             |
    +---------------------------+-----------------------------------------+
    | ``False``                 | ``1`` (it flattens every level).        |
    +---------------------------+-----------------------------------------+
    | ``dict(...)``             | STRICT ``counts:`` validation first,    |
    |                           | always (see below); then                |
    |                           | ``stored_rungs`` unless the resolver    |
    |                           | would compute (``recompute`` or         |
    |                           | ``stored_rungs <= 1``), in which case   |
    |                           | :func:`~luxar.gsplats.lod.additive.additive_rung_count`|
    |                           | over the same ``n_lods=4`` default.     |
    +---------------------------+-----------------------------------------+
    | anything else             | ``None`` — the resolver ``TypeError``s  |
    |                           | on it downstream, at its own site.      |
    +---------------------------+-----------------------------------------+

    STRICT FIRST, CLAMP SECOND — the resolver's own order, and the order is the
    whole answer for an explicit ``counts:`` list.
    :func:`resolve_additive_axis_gsplats` runs
    :func:`~luxar.gsplats.lod.additive.validate_counts_breakpoints` against
    ``substitutive_levels[0].n_splats_total`` BEFORE its per-level
    :func:`~luxar.gsplats.lod.additive.clamp_counts_breakpoints` loop, so a list
    larger than the data ABORTS ("largest breakpoint N exceeds N=…") rather than
    shrinking. That validation is UNCONDITIONAL there — it runs on any spec
    carrying ``breakpoints``, above the per-level "does this level need
    computing?" test — so it is unconditional here too, above the pass-through
    return for a stored ladder. A spec the resolver would abort on therefore
    answers UNKNOWN even when nothing would have been computed, rather than
    reporting the stored count of a call that cannot run. The clamp is there for
    the COARSER levels of a multi-level pyramid, whose N the caller cannot know;
    a single grafted leaf IS that ``substitutive_levels[0]``, i.e. exactly the N
    the strict validator uses. So
    clamping first would answer a confident, WRONG count precisely where the
    resolver raises — measured, ``{"recompute": True, "breakpoints": [100]}`` on
    a laddered 12-splat leaf clamped to ``[12]``, counted one rung, let the gate
    skip, and stranded the childless ``kind=partition`` this query exists to
    prevent (``additive_rung_count(12, 4, [100])`` answers ``None`` on its own;
    the clamp destroyed the signal before it got there). A rejection is reported
    as UNKNOWN, which is honest twice over: the count is unreadable AND the call
    is doomed, and the gate's fallback to ``stored_rungs`` then refuses it
    cleanly with an empty store instead of stranding a wrapper.

    The clamp is kept behind the validator as the MIRROR of the resolver's loop,
    not as a live filter: once the list is known to fit ``n``, the clamp can only
    drop a cut equal to ``n``, which ``_resolve_breakpoints`` re-appends. Same
    standing as ``additive_rung_count``'s de-duplication replay.

    Where that OVER-REFUSES: the matrix-shaped ``kind=lod`` call site asks per
    leaf, COARSER (smaller) levels included, and there the resolver would CLAMP
    rather than abort. Those leaves answer UNKNOWN, the gate falls back to their
    stored count, and a call that would in fact have worked is refused.
    Measured, on a matrix-shaped ``kind=lod`` store whose coarse leaf holds 16
    splats (2 rungs) and whose finest holds 64 (2 rungs), with
    ``add_gsplats_from_file("g", …, partition={"max_elements": 8},
    additive_lod={"recompute": True, "breakpoints": [64]})``: the gate validates
    ``[64]`` PER LEAF, so the coarse leaf's 16 answers UNKNOWN, falls back to its
    stored 2 rungs, and the call is refused with the stored pair — while the SAME
    call through ``add_gsplats_from_data`` SUCCEEDS, because the resolver
    validates once against ``substitutive_levels[0]`` (64, passes) and CLAMPS the
    coarse level to ``[16]``, so both levels really do collapse to one rung.

    It is a false refusal, then — the same class as the ``{"recompute": True,
    "breakpoints": [1.0]}`` over-refusal the gate already admits — but not a
    regression: the pre-#1632 store-only gate refused that call too, on the
    stored ladder alone. And it is safe in DIRECTION: it refuses with an EMPTY
    store, never strands, and the matrix-shaped branch cannot strand in any case,
    its data door naming the caller's own node rather than a child.

    Why it is not fixed here: the two call sites genuinely differ. A grafted leaf
    IS the resolver's ``substitutive_levels[0]``, so per-leaf strict validation is
    exactly right there; the matrix-shaped branch instead hands the WHOLE tree to
    ONE resolver call, which validates globally and clamps per level. Telling the
    two apart would mean threading the branch down into the gate, for a case that
    is already refused — correctly enough that nothing is written.

    ``None`` means UNKNOWN — a spec type this vocabulary does not cover, or a
    ladder :func:`~luxar.gsplats.lod.additive.additive_rung_count` cannot count
    without the ordering (energy-fraction breakpoints) or would refuse outright
    (a malformed spec). It is a statement about the SPEC, never about the input:
    a caller that cannot act on an unknown should fall back to what it already
    knows (the gate falls back to ``stored_rungs``, since an unreadable kwarg
    cannot be trusted to have removed a ladder that is demonstrably there),
    rather than guess.

    What it deliberately does NOT report is where the resulting ladder came
    from. An earlier draft returned a ``computed`` flag alongside the count for
    the gate's remedy wording, but that is the wrong discriminator and its one
    caller now asks its own question — ``len(leaf.additive_sublods) > 1``,
    i.e. does the STORE carry a ladder — because a stored ladder survives
    dropping the kwarg whether or not the kwarg also rebuilds one.

    Out of scope, deliberately: ``spec is True`` on an UNLADDERED input answers
    ``1`` here, which is honest (there is no ladder), even though
    :func:`resolve_additive_axis_gsplats` will then raise "additive_lod=True
    requires every substitutive level to already carry an additive ladder". That
    is an invalid CALL, not a rung count, and it is the resolver's to report.
    """
    if spec is None or spec is True:
        return int(stored_rungs)
    if spec is False:
        return 1
    if isinstance(spec, dict):
        kwargs = dict(spec)
        recompute = bool(kwargs.pop("recompute", False))
        from ....gsplats.lod.additive import (
            additive_rung_count,
            clamp_counts_breakpoints,
            validate_counts_breakpoints,
        )

        breakpoints = kwargs.get("breakpoints", "equal-count")
        try:
            validate_counts_breakpoints(breakpoints, int(n_splats))
        except ValueError:
            # The resolver ABORTS on this list rather than clamping it — see
            # STRICT FIRST, CLAMP SECOND above. Clamping here instead answers a
            # confident wrong count and strands the wrapper. ABOVE the
            # pass-through return below, because the resolver's own call is
            # unconditional too: it validates whenever ``breakpoints`` is
            # present, before its per-level "does this level need computing?"
            # test, so a list the resolver would abort on is UNKNOWN here even
            # when the stored ladder would have been passed straight through.
            return None
        if not (recompute or stored_rungs <= 1):
            return int(stored_rungs)
        return additive_rung_count(
            int(n_splats),
            kwargs.get("n_lods", 4),
            clamp_counts_breakpoints(breakpoints, int(n_splats)),
        )
    return None
