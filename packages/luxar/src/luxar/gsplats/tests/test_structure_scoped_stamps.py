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
from luxar.gsplats.gsplat_data import GSplatData
from luxar.gsplats.io.save_gsplats import NORMALIZATION_STATS_KEYS
from luxar.gsplats.lod.decimate import decimate

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
    # those would trade one silent loss for another).
    assert list(out.stats["coarsen_dims"]) == _KEEP["coarsen_dims"]
    assert out.stats["fitter_name"] == _KEEP["fitter_name"]
    assert out.stats["iterations"] == _KEEP["iterations"]
    # ...and the caller's own dataset was not scrubbed underneath it.
    assert data.stats == _laddered_stats()


def test_a_no_op_decimate_returns_the_input_verbatim() -> None:
    """The early return keeps its topology: nothing changed, so nothing is false.

    ``target >= n_splats`` returns the INPUT object, still a pyramid if it was
    one. Scrubbing there would delete a true record (and mutate a dataset the
    caller still holds).
    """
    data = _flat_dataset()
    assert decimate(data, target=data.n_splats) is data
    assert data.stats == _laddered_stats()
