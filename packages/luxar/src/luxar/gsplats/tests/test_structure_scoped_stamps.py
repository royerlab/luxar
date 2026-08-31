"""The third ``stats`` hygiene axis: the artifact's own TOPOLOGY record.

The domain half of #1600's structure-scoped item; the command-level end-to-end
claims (does the scrub survive both writers, does the chunk-ordering barrier
survive it) live in ``cli/tests/test_gsplat_structure_scoped_stamps.py``.

What is pinned here is the rule itself: :func:`stats_after_structure_change`
takes the topology record and NOTHING else, the three axes are disjoint, and
``coarsen_dims`` is exempt because the writer reads it back. Plus the one place
the rule is applied at THIS layer rather than by a command:
:func:`~luxar.gsplats.lod.decimate.decimate`, whose contract is one flat leaf
whatever it was handed — advertised as public API in ``CLAUDE.md``, so a
CLI-only scrub left it publishing the defect.

Surviving the scrub is not the same as being TRUE, and the two stamps a rewrite
still OWNS after it are pinned here as well: the ``coarsen_dims`` a reduction
actually coarsened over, and the root ladder summary
(:func:`~luxar.gsplats.lod.restamp.refresh_root_ladder_summary`) a re-ladder
replaces.
"""

from __future__ import annotations

from typing import Any, Dict

import numpy as np
import pytest

from luxar.gsplats._data.filtering import (
    _CONTENT_SCOPED_OP_RECORD_KEYS,
    _CONTENT_SCOPED_STATS_KEYS,
    _REGION_SCOPED_STATS_KEYS,
    _STRUCTURE_SCOPE_EXEMPT_KEYS,
    _STRUCTURE_SCOPED_STATS_KEYS,
    stats_after_structure_change,
)
from luxar.gsplats.gsplat_data import AdditiveSubLOD, GSplatData, SubstitutiveLevel
from luxar.gsplats.io.save_gsplats import NORMALIZATION_STATS_KEYS
from luxar.gsplats.lod.decimate import decimate
from luxar.gsplats.lod.restamp import refresh_root_ladder_summary
from luxar.gsplats.tree import GSplatNode, GSplatPartition

#: One recognisable value per registry key, so a survivor is unmistakable.
_TOPOLOGY: Dict[str, Any] = {
    key: f"topology-{key}" for key in _STRUCTURE_SCOPED_STATS_KEYS
}

#: Everything a structure change leaves alone, one representative per axis.
_KEEP: Dict[str, Any] = {
    # Exempt by name (the barrier trap).
    "coarsen_dims": [0, 1, 2],
    # The normalization block: the INPUT VOLUME's intensity scale.
    "floor": 113.0,
    "image_min": 0.0,
    "image_max": 4095.0,
    "intensity_range": 4095.0,
    # The run happened, and regrouping splats does not un-happen it.
    "fitter_name": "probe",
    "iterations": 123,
    # The other two axes have their own predicates; this one must not pre-empt
    # them (a `flatten` is neither a crop nor a content change).
    "source_shape": [64, 64, 64],
    "occupancy": 0.0123,
    "psnr_db": 44.4587,
    "culling_method": "cumulative",
    # A HYPOTHETICAL control rather than observed behaviour: `additive.py` stamps
    # this inside a rung's own stats and it was never measured in a root
    # `pipeline/` group. Kept because it is the shape of key a lazy
    # `key.startswith("lod_")` scrub would take — such a rewrite goes red here.
    "lod_stream_chunk_splats": 14000,
}


def test_the_three_axes_are_disjoint() -> None:
    """No key may be scrubbed by two rules with two different predicates.

    An overlap would mean a ``flatten`` silently applying the content or region
    rule as well — the exact confusion #1600's first half was about.
    """
    structure = set(_STRUCTURE_SCOPED_STATS_KEYS)
    assert not structure & set(_REGION_SCOPED_STATS_KEYS)
    assert not structure & set(_CONTENT_SCOPED_STATS_KEYS)
    assert not structure & set(_CONTENT_SCOPED_OP_RECORD_KEYS)
    assert not structure & set(_STRUCTURE_SCOPE_EXEMPT_KEYS)
    assert not structure & set(NORMALIZATION_STATS_KEYS)
    # No duplicate rows in the registry (a tuple happily holds two).
    assert len(structure) == len(_STRUCTURE_SCOPED_STATS_KEYS)


def test_coarsen_dims_is_exempt_because_the_writer_reads_it_back() -> None:
    """The trap, pinned at the unit level.

    ``_barrier_from_coarsen_dims`` turns this key into ``write_gsplats_tree``'s
    ordering barrier, so dropping it changes the output's CHUNK LAYOUT rather
    than merely deleting a stamp. The layout consequence is measured in the
    command-level twin of this file; here it is enough that the key can never
    fall into the deny-list by accident.
    """
    assert "coarsen_dims" in _STRUCTURE_SCOPE_EXEMPT_KEYS
    assert "coarsen_dims" not in _STRUCTURE_SCOPED_STATS_KEYS
    kept = stats_after_structure_change({"coarsen_dims": [0, 1, 2], "lod_kind": "x"})
    assert kept == {"coarsen_dims": [0, 1, 2]}


def test_the_scrub_takes_the_topology_record_and_nothing_else() -> None:
    scrubbed = stats_after_structure_change({**_TOPOLOGY, **_KEEP})
    assert not [k for k in _STRUCTURE_SCOPED_STATS_KEYS if k in scrubbed], (
        f"topology stamps survived: {sorted(set(scrubbed) & set(_TOPOLOGY))}"
    )
    assert scrubbed == _KEEP


def test_the_scrub_does_not_mutate_the_dict_the_caller_owns() -> None:
    """``GSplatData`` is conceptually immutable and every call site passes a dict
    it still holds (``flatten``'s freshly loaded root stats; ``partition``'s and
    ``decimate()``'s INPUT dataset's own ``stats``). An in-place scrub there would
    strip the topology record off that input mid-operation."""
    source = {**_TOPOLOGY, **_KEEP}
    before = dict(source)
    result = stats_after_structure_change(source)
    assert source == before, "the caller's dict was edited"
    assert result is not source
    # No assertion on the copy's DEPTH: pinning `result["coarsen_dims"] is
    # source["coarsen_dims"]` would gate a safe change (hardening `dict(stats)`
    # to a deep copy) while staying green for the real defect it looks like it
    # guards (an in-place edit of a nested value, which mutates BOTH sides). The
    # pair above is the whole contract.


def test_an_empty_or_topology_free_dict_is_returned_unchanged() -> None:
    """A plain fit publishes no topology record; the scrub must be a no-op there
    rather than inventing keys or an empty ``pipeline/`` group."""
    assert stats_after_structure_change({}) == {}
    assert stats_after_structure_change(dict(_KEEP)) == _KEEP


def _laddered_stats() -> Dict[str, Any]:
    """The full topology record plus everything that must survive it."""
    return {**_TOPOLOGY, **_KEEP}


def _flat_dataset(n: int = 200) -> GSplatData:
    """A flat 3D dataset publishing the full topology record."""
    rng = np.random.default_rng(0)
    chol = np.zeros((n, 6), dtype=np.float32)
    chol[:, [0, 2, 5]] = 2.0  # isotropic sigma=2 (packed lower-triangular 3D)
    return GSplatData(
        centers=(rng.random((n, 3)) * 100.0).astype(np.float32),
        amplitudes=np.linspace(1.0, 0.1, n).astype(np.float32),
        cholesky_factors=chol,
        stats=_laddered_stats(),
    )


@pytest.mark.parametrize("method", ["prefix", "merge"])
def test_the_python_decimate_api_drops_the_topology_record(method: str) -> None:
    """``decimate()`` itself must scrub, not just ``gsplat decimate``.

    The hole this closes: the scrub used to live in the CLI, so
    ``decimate(GSplatData.load(pyramid), target=50)`` returned one flat leaf still
    carrying ``lod_kind: substitutive`` / ``n_substitutive_levels`` / the ladder
    summary — and ``CLAUDE.md`` advertises ``luxar.gsplats.lod.decimate`` as the
    Python entry point, so that is a published API returning a lie. Both families
    are covered because they reach the scrub by different routes (a prefix subset
    vs a merge that lands on the requested count with every splat replaced).
    """
    data = _flat_dataset()
    out = decimate(data, target=50, method=method)

    assert out is not data and out.n_splats <= 50
    survivors = [k for k in _STRUCTURE_SCOPED_STATS_KEYS if k in out.stats]
    assert not survivors, f"the flat result still advertises a topology: {survivors}"
    # The exempt key and the descriptive fit provenance come through (dropping
    # those would trade one silent loss for another) — except that a `merge`
    # RE-STAMPS `coarsen_dims` with what it coarsened over, which on this 3D
    # fixture is every dim. Spelled EXPLICITLY, never `None`: the writer cannot
    # tell a written null from an absent key. Both families land on the same
    # value here only because the inherited stamp happens to say the same
    # thing — the two are told apart in
    # `test_decimate_stamps_the_coarsen_dims_the_reduction_used`, whose fixture
    # is stamped differently on purpose.
    coarsen = out.stats["coarsen_dims"]
    assert (list(coarsen) if coarsen is not None else None) == [0, 1, 2]
    assert out.stats["fitter_name"] == _KEEP["fitter_name"]
    assert out.stats["iterations"] == _KEEP["iterations"]
    # ...and the caller's own dataset was not scrubbed underneath it.
    assert data.stats == _laddered_stats()


def test_a_no_op_decimate_returns_the_input_verbatim() -> None:
    """The early return keeps its topology: nothing changed, so nothing is false.

    ``target >= n_splats`` returns the INPUT object, still a pyramid if it was
    one. Scrubbing there would delete a true record (and mutate a dataset the
    caller still holds) — and it must not acquire a fresh ``coarsen_dims``
    either, since no reduction ran.
    """
    data = _flat_dataset()
    assert decimate(data, target=data.n_splats, coarsen_dims=[0, 1]) is data
    assert data.stats == _laddered_stats()


def test_decimate_carries_label_prefix_and_refuses_label_merge() -> None:
    data = _flat_dataset()
    labeled = GSplatData(
        centers=data.centers,
        amplitudes=data.amplitudes,
        cholesky_factors=data.cholesky_factors,
        label_ids=np.arange(data.n_splats, dtype=np.uint16),
        label_vocabulary={i: str(i) for i in range(data.n_splats)},
    )

    prefix = decimate(labeled, target=data.n_splats // 2, method="prefix")
    assert prefix.label_ids is not None
    for center, label_id in zip(prefix.centers, prefix.label_ids):
        matches = np.flatnonzero(np.all(labeled.centers == center, axis=1))
        assert matches.size == 1
        assert int(label_id) == int(matches[0])

    with pytest.raises(ValueError, match="cannot coarsen.*label_ids"):
        decimate(labeled, target=data.n_splats // 2, method="merge", device="cpu")

    with pytest.warns(UserWarning) as caught:
        automatic = decimate(
            labeled,
            target=data.n_splats // 4,
            method="auto",
            coarsen_dims=[0, 1, 2],
        )
    messages = [str(warning.message) for warning in caught]
    assert any("selected 'prefix'" in message for message in messages)
    ignored = next(message for message in messages if "is IGNORED" in message)
    assert "categorical channel 'label_ids'" in ignored
    assert "crossover" not in ignored
    assert "method='merge'" not in ignored
    assert automatic.label_ids is not None
    assert automatic.n_splats == data.n_splats // 4
    for center, label_id in zip(automatic.centers, automatic.label_ids):
        matches = np.flatnonzero(np.all(labeled.centers == center, axis=1))
        assert matches.size == 1
        assert int(label_id) == int(matches[0])


def _stacked_dataset(n: int = 200) -> GSplatData:
    """4D splats on three integer timepoints, stamped ``coarsen_dims=[1, 2, 3]``.

    A DIFFERENT list from any request below, so an inherited value is never
    mistaken for a re-stamped one; and the barrier it implies (``[0]``) is a
    continuous spatial axis, i.e. exactly the stale-provenance case.
    """
    rng = np.random.default_rng(0)
    chol = np.zeros((n, 10), dtype=np.float32)
    chol[:, [d * (d + 1) // 2 + d for d in range(4)]] = 2.0
    centers = np.empty((n, 4), dtype=np.float32)
    centers[:, :3] = rng.random((n, 3)) * 100.0
    centers[:, 3] = rng.integers(0, 3, size=n).astype(np.float32)
    return GSplatData(
        centers=centers,
        amplitudes=np.linspace(1.0, 0.1, n).astype(np.float32),
        cholesky_factors=chol,
        stats={**_laddered_stats(), "coarsen_dims": [1, 2, 3]},
    )


@pytest.mark.parametrize(
    ("method", "request_dims", "expected"),
    [
        # A proper subset: stamped as asked, so the writer's barrier is [3].
        ("merge", [0, 1, 2], [0, 1, 2]),
        # Every dim, spelled out: the reduction blends over all four, so the
        # stamp names all four. NOT `None` — a written null is indistinguishable
        # from an absent key to `_barrier_from_coarsen_dims`, which then
        # auto-detects and can re-impose a barrier on the axis just blended.
        ("merge", [0, 1, 2, 3], [0, 1, 2, 3]),
        # ...and the default means exactly the same reduction, so it must
        # publish exactly the same stamp.
        ("merge", None, [0, 1, 2, 3]),
        # A prefix merges nothing: the knob is merge-only and the input's stamp
        # is still true of the survivors, so neither is overwritten by the other.
        ("prefix", [0, 1, 2], [1, 2, 3]),
    ],
    ids=["merge-subset", "merge-all-dims", "merge-default", "prefix-inherits"],
)
def test_decimate_stamps_the_coarsen_dims_the_reduction_used(
    method: str, request_dims: "list[int] | None", expected: "list[int] | None"
) -> None:
    """``coarsen_dims`` survives the topology scrub; it must still be true.

    The writer reads it back to place the chunk-ordering barrier
    (``_barrier_from_coarsen_dims``), so an inherited list over a merge that
    coarsened different axes puts the barrier on an axis this reduction blended
    (#1600). The command-level twin measures the resulting LAYOUT; here it is
    the value, at the domain layer the public API also goes through.
    """
    data = _stacked_dataset()
    out = decimate(data, target=50, method=method, coarsen_dims=request_dims)

    got = out.stats["coarsen_dims"]
    assert (list(got) if got is not None else None) == expected
    assert data.stats["coarsen_dims"] == [1, 2, 3], "the caller's dict was edited"


# ── the root ladder summary a re-ladder replaces (`gsplat additive`) ──


def _sublod(n: int) -> AdditiveSubLOD:
    """One rung of ``n`` 3D splats; geometry irrelevant, only the widths are."""
    chol = np.zeros((n, 6), dtype=np.float32)
    chol[:, [0, 2, 5]] = 2.0
    return AdditiveSubLOD(
        centers=np.zeros((n, 3), dtype=np.float32),
        amplitudes=np.ones(n, dtype=np.float32),
        cholesky_factors=chol,
    )


def _laddered_tree(rungs_per_level: "list[list[int]]", **stats: Any) -> GSplatNode:
    """A tree with the given rung widths per level, FINEST LEVEL FIRST.

    One level gives a bare leaf; several give a ``kind=lod`` group, which stores
    its children the other way round (coarsest-first) — the mirror
    :func:`refresh_root_ladder_summary` has to get right.
    """
    return GSplatData.from_substitutive_levels(
        [
            SubstitutiveLevel(
                additive_sublods=[_sublod(n) for n in rungs],
                compression_factor=4**index,
                level_index=index,
                stats=dict(stats),
            )
            for index, rungs in enumerate(rungs_per_level)
        ]
    ).tree


def test_the_root_ladder_summary_follows_lod_substitutive_level() -> None:
    """The rule is the producers': the root summarises ONE named level.

    ``make_additive_lod`` stamps the root block for the level it laddered and
    names it in ``lod_substitutive_level``; ``_map_substitutive`` and ``cull``
    already refresh it that way. ``gsplat additive`` re-ladders every leaf and
    left the block describing the ladder it had just replaced (#1600).
    """
    tree = _laddered_tree(
        [[10, 10, 10], [8, 8], [5]],
        lod_method="mass",
        lod_breakpoints_kind="explicit-counts",
    )
    stale = {
        "lod_method": "greedy",
        "lod_n_lods": 3,
        "lod_breakpoints_kind": "equal-count",
        "lod_cutpoints": [10, 20, 30],
        "lod_substitutive_level": 2,
        "recipe": "levels",
    }
    fresh = refresh_root_ladder_summary(stale, tree)

    # Level 2 is the coarsest (matrix order) = the group's FIRST child.
    assert fresh["lod_n_lods"] == 1
    assert fresh["lod_cutpoints"] == [5]
    # Read back off the rebuilt leaf, so an `auto` request publishes what it
    # resolved to rather than the word "auto".
    assert fresh["lod_method"] == "mass"
    assert fresh["lod_breakpoints_kind"] == "explicit-counts"
    # A re-ladder moves no level, and the rest of the record is not its business.
    assert fresh["lod_substitutive_level"] == 2
    assert fresh["recipe"] == "levels"
    assert stale["lod_n_lods"] == 3, "the caller's dict was edited"


def test_the_root_ladder_summary_refreshes_only_keys_that_were_there() -> None:
    """A store that never published a summary must not acquire one.

    Same contract as ``_refresh_ladder_summary``, whose count/cutpoint half this
    reuses: the refresh corrects a claim, it does not start making one.

    The METHOD half needs a leaf that actually publishes one, or the readback
    branch has nothing to invent and the direction goes unmeasured: dropping the
    ``key in refreshed`` guard left every test green while a partial summary
    ACQUIRED ``lod_method`` / ``lod_breakpoints_kind`` it had never had.
    """
    bare = _laddered_tree([[10, 10, 10]])
    assert refresh_root_ladder_summary({"recipe": "stream"}, bare) == {
        "recipe": "stream"
    }
    assert refresh_root_ladder_summary({"lod_n_lods": 9}, bare) == {"lod_n_lods": 3}

    stamped = _laddered_tree(
        [[10, 10, 10]], lod_method="mass", lod_breakpoints_kind="explicit-counts"
    )
    assert refresh_root_ladder_summary(
        {"lod_n_lods": 2, "recipe": "stream"}, stamped
    ) == {
        "lod_n_lods": 3,
        "recipe": "stream",
    }


def test_a_summary_key_the_rebuilt_leaf_cannot_supply_is_dropped() -> None:
    """...and the other direction: a value that cannot be refreshed goes.

    The two halves of the same refresh disagreed. ``_refresh_recipe_ladder``
    DELETES its ``method`` when the rebuilt leaf publishes none, while the
    ``lod_*`` half silently kept the inherited one — the ladder-that-is-gone
    claim this function exists to scrub (#1600 review). Unreachable from
    ``gsplat additive`` today, because ``make_additive_lod`` always stamps both;
    reachable by anyone calling this public function on a hand-built tree.

    ``lod_n_lods`` in the same dict is the control: the refresh still ran, so the
    deletion is the rule rather than an early return.
    """
    bare = _laddered_tree([[10, 10, 10]])  # a leaf publishing no ladder stats
    assert refresh_root_ladder_summary(
        {
            "lod_method": "greedy",
            "lod_breakpoints_kind": "equal-count",
            "lod_n_lods": 9,
            "recipe": "stream",
        },
        bare,
    ) == {"lod_n_lods": 3, "recipe": "stream"}


@pytest.mark.parametrize(
    ("level", "expected_rungs"),
    [
        # Below the range: the summary falls back to the FINEST level (matrix
        # index 0 = the group's last child). Unclamped this indexes one past the
        # end of the children list and raises IndexError.
        (-1, 3),
        # Past the end: the COARSEST level. Unclamped, `len - 1 - 3` is -1, which
        # silently wraps round to the finest child — a plausible wrong answer.
        (3, 1),
    ],
    ids=["below-range", "past-the-end"],
)
def test_an_out_of_range_summary_level_is_clamped(
    level: int, expected_rungs: int
) -> None:
    """``lod_substitutive_level`` is read off a store that may no longer fit it.

    The index is an inherited stamp, not a computed one — a store whose level
    count shrank (or a hand-edited one) can name a level that is not there. Both
    directions are failures without the clamp, and neither is loud: negative
    raises ``IndexError`` from inside a metadata refresh, and over-range wraps
    into a DIFFERENT child and summarises the wrong ladder.
    """
    tree = _laddered_tree([[10, 10, 10], [8, 8], [5]])
    fresh = refresh_root_ladder_summary(
        {"lod_n_lods": 99, "lod_substitutive_level": level}, tree
    )
    assert fresh["lod_n_lods"] == expected_rungs
    # The index itself is not the refresh's business — a re-ladder moves no level.
    assert fresh["lod_substitutive_level"] == level


# ── the SAME ladder, one spelling over: `batch-fit merge`'s pipeline knobs ──


def _batch_merge_stream_pipeline() -> Dict[str, Any]:
    """The root ``pipeline/`` block a ``batch-fit merge --recipe stream`` writes.

    Taken from the producer itself rather than hand-copied, so a key added to
    ``_recipe_pipeline_info`` shows up here instead of quietly going unhandled.
    """
    from luxar.gsplats.batch.merge_orchestrator import _recipe_pipeline_info
    from luxar.gsplats.lod.recipes import RecipeParams

    info = _recipe_pipeline_info(
        "stream",
        RecipeParams(n_lods=6, additive_method="mass", breakpoints="equal-count"),
    )
    assert info == {
        "recipe": "stream",
        "lod_kind": "additive",
        "per_part": True,
        "n_lods": 6,
        "method": "mass",
        "breakpoints": "equal-count",
    }, f"the producer's key set moved: {info}"
    return dict(info)


def test_the_batch_merge_ladder_knobs_are_refreshed_too() -> None:
    """``n_lods`` / ``method`` / ``breakpoints`` describe the same ladder.

    ``gsplat additive`` over a ``batch-fit merge`` output is an advertised use
    case, and that store's summary is spelled WITHOUT the ``lod_`` prefix — so
    the five prefixed keys were refreshed while ``n_lods: 6`` rode through
    unchanged next to them (#1600 review).
    """
    tree = _laddered_tree([[10, 10, 10]], lod_method="self_energy")
    fresh = refresh_root_ladder_summary(_batch_merge_stream_pipeline(), tree)

    assert fresh["n_lods"] == 3, "the rung count still describes the old ladder"
    assert fresh["method"] == "self_energy", "read back off the leaf that was written"
    # The build SPEC ("stream:14000" / "counts:5,15,40" / "equal-count") is in a
    # different vocabulary from the leaf's resolved `lod_breakpoints_kind` and
    # cannot be read back off a ladder, so it is dropped rather than invented.
    assert "breakpoints" not in fresh
    # Untouched: a per-leaf re-ladder leaves a per-part ladder per-part, and the
    # mechanism is still additive.
    assert fresh["per_part"] is True
    assert fresh["lod_kind"] == "additive"
    assert fresh["recipe"] == "stream"


def test_the_levels_branch_keeps_its_substitutive_method() -> None:
    """``method`` is shared between the two branches and means different things.

    ``_recipe_pipeline_info("levels", …)`` stamps the SUBSTITUTIVE merge method
    under the same key, and a re-ladder does not touch it. Refreshing on the key
    name alone would overwrite it with an additive ordering.
    """
    from luxar.gsplats.batch.merge_orchestrator import _recipe_pipeline_info
    from luxar.gsplats.lod.recipes import RecipeParams

    info = dict(
        _recipe_pipeline_info("levels", RecipeParams(substitutive_method="greedy"))
        or {}
    )
    assert info["method"] == "greedy" and info["lod_kind"] == "substitutive"
    fresh = refresh_root_ladder_summary(
        info, _laddered_tree([[10, 10, 10]], lod_method="self_energy")
    )
    assert fresh["method"] == "greedy"
    assert fresh["additive_ladders"] is True  # still true: every leaf is laddered


def test_a_partition_drops_the_batch_merge_ladder_knobs_as_well() -> None:
    """The shape that cannot be summarised drops BOTH spellings, or neither.

    Dropping ``lod_n_lods`` because "parts hold different rung counts" while
    leaving ``n_lods: 6`` two keys away — asserting exactly that number, under
    ``per_part: True`` — is the same lie the drop exists to prevent.
    """
    parts = GSplatPartition(
        children=[_laddered_tree([[10, 10]]), _laddered_tree([[7, 7, 7]])]
    )
    fresh = refresh_root_ladder_summary(_batch_merge_stream_pipeline(), parts)
    assert fresh == {"recipe": "stream", "lod_kind": "additive", "per_part": True}


def test_a_partition_root_publishes_no_single_ladder() -> None:
    """Parts differ in count, so any root rung number is true of at most one.

    The block is dropped rather than refreshed from an arbitrary part — which is
    also what the ``tiles`` / ``overview`` / ``adaptive`` builders publish at the
    root, for the same reason.
    """
    parts = GSplatPartition(
        children=[_laddered_tree([[10, 10]]), _laddered_tree([[7, 7, 7]])]
    )
    fresh = refresh_root_ladder_summary(
        {
            "lod_method": "greedy",
            "lod_n_lods": 3,
            "lod_breakpoints_kind": "equal-count",
            "lod_cutpoints": [10, 20, 30],
            "lod_substitutive_level": 0,
            "recipe": "tiles",
        },
        parts,
    )
    assert fresh == {"recipe": "tiles"}
