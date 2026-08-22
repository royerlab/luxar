"""The third ``stats`` hygiene axis: the artifact's own TOPOLOGY record.

The domain half of #1600's structure-scoped item; the command-level end-to-end
claims (does the scrub survive both writers, does the chunk-ordering barrier
survive it) live in ``cli/tests/test_gsplat_structure_scoped_stamps.py``.

What is pinned here is the rule itself: :func:`stats_after_structure_change`
takes the topology record and NOTHING else, the three axes are disjoint, and
``coarsen_dims`` is exempt because the writer reads it back.
"""

from __future__ import annotations

from typing import Any, Dict

from luxar.gsplats._data.filtering import (
    _CONTENT_SCOPED_OP_RECORD_KEYS,
    _CONTENT_SCOPED_STATS_KEYS,
    _REGION_SCOPED_STATS_KEYS,
    _STRUCTURE_SCOPE_EXEMPT_KEYS,
    _STRUCTURE_SCOPED_STATS_KEYS,
    stats_after_structure_change,
)
from luxar.gsplats.io.save_gsplats import NORMALIZATION_STATS_KEYS

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
    # A per-rung ladder stamp that reached the top-level dict: the rung it
    # describes still exists in a flattened result.
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
    """``GSplatData`` is conceptually immutable and two call sites pass a dict
    they still hold (the freshly loaded root stats, the reduced dataset's own
    ``stats``). An in-place scrub there would strip the topology record off the
    INPUT object mid-command."""
    source = {**_TOPOLOGY, **_KEEP}
    before = dict(source)
    result = stats_after_structure_change(source)
    assert source == before, "the caller's dict was edited"
    assert result is not source
    # Nested values are shared (a shallow copy is enough — nothing edits them),
    # but the top-level mapping is independent in both directions.
    result["lod_kind"] = "mutated"
    assert "lod_kind" not in source or source["lod_kind"] != "mutated"


def test_an_empty_or_topology_free_dict_is_returned_unchanged() -> None:
    """A plain fit publishes no topology record; the scrub must be a no-op there
    rather than inventing keys or an empty ``pipeline/`` group."""
    assert stats_after_structure_change({}) == {}
    assert stats_after_structure_change(dict(_KEEP)) == _KEEP
