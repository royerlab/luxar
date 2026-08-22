"""No ``gsplat`` command may advertise a topology it does not have.

The third hygiene axis of #1600 (the first two live in
``test_gsplat_content_scoped_metrics.py`` and its domain twin). Three commands
change the artifact's STRUCTURE KIND while threading the input's ``stats``
through to the output's ``pipeline/`` group, so a flattened four-level pyramid
published ``lod_kind: substitutive`` / ``n_substitutive_levels: 4`` /
``lod_cutpoints: [2, 4, 5, 7]`` for a store that is a single flat leaf:

* ``flatten``   → one flat leaf (writes through ``GSplatData.save``)
* ``decimate``  → one flat leaf (writes through ``write_gsplats_tree``)
* ``partition`` → ``kind=partition`` of bare leaves (``write_gsplats_tree``)

Both writers are represented on purpose: a fix wired into one would leave the
other silently exempt.

Two things must SURVIVE the scrub, which is why it is a deny-list of topology
keys rather than "drop the ``pipeline/`` group": the normalization block (the
input volume's intensity scale) and ``coarsen_dims``, which the writer READS BACK
to derive the chunk-ordering barrier — so a naive scrub does not merely delete a
stamp, it changes the output's layout.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any, Dict, List, Sequence

import numpy as np
import pytest
import zarr
from typer.testing import CliRunner

from luxar._zarr_compat import read_node_attrs
from luxar.cli import app
from luxar.gsplats._data.filtering import (
    _STRUCTURE_SCOPE_EXEMPT_KEYS,
    _STRUCTURE_SCOPED_STATS_KEYS,
)
from luxar.gsplats.gsplat_data import GSplatData
from luxar.gsplats.io.load_gsplats import load_gsplat_node
from luxar.gsplats.io.save_gsplats import NORMALIZATION_STATS_KEYS

#: The whole topology record at NON-DEFAULT values. A default (``method: "auto"``,
#: ``conserve_mass: True``, ``refine: "none"``) is indistinguishable from a fresh
#: writer's own idea of the field, so a carry bug hides behind it — the issue says
#: so explicitly. Spelled out literally, then tied to the implementation constant
#: by ``test_the_fixture_covers_the_registry`` in the one direction that is safe.
_TOPOLOGY: Dict[str, Any] = {
    "lod_kind": "substitutive",
    "compression_factor": 7,
    "method": "greedy_lloyd",
    "n_substitutive_levels": 5,
    "coverage_inflation": 2.5,
    "conserve_mass": False,
    "refine": "l2",
    "refine_iters": 77,
    "lod_method": "self_energy",
    "lod_n_lods": 6,
    "lod_breakpoints_kind": "stream",
    "lod_cutpoints": [3, 7, 15, 31, 63, 127],
    "lod_substitutive_level": 4,
    "recipe": "adaptive",
    "per_part": True,
    "n_lods": 6,
    "breakpoints": "stream:1234",
    "levels": 5,
    "additive_ladders": False,
}

#: The two blocks that share ``pipeline/`` and must come out the other side.
#: ``coarsen_dims`` is a 4D coarsen-the-spatial-axes list, so its complement is a
#: real barrier ([3]) — a value the writer acts on rather than a decoration.
_EXEMPT: Dict[str, Any] = {
    "coarsen_dims": [0, 1, 2],
    "floor": 113.0,
    "image_min": 0.0,
    "image_max": 4095.0,
    "intensity_range": 4095.0,
}

#: Fit provenance the scrub must not touch (#1737 / #1601 threaded this through;
#: regressing it would trade one silent loss for another).
_DESCRIPTIVE: Dict[str, Any] = {
    "fitter_name": "probe",
    "iterations": 123,
    "converged": True,
    "source_shape": [64, 64, 64, 3],
    "occupancy": 0.0123,
}


def test_the_fixture_covers_the_registry() -> None:
    """Anchor: a key added to the registry must get a fixture row.

    Without this the rows below would iterate ``_TOPOLOGY`` and stay green for a
    key nobody ever wrote into the fixture — the vacuous-test failure mode.
    """
    assert set(_TOPOLOGY) == set(_STRUCTURE_SCOPED_STATS_KEYS), (
        "fixture/registry drift — missing rows: "
        f"{sorted(set(_STRUCTURE_SCOPED_STATS_KEYS) - set(_TOPOLOGY))}; "
        f"stale rows: {sorted(set(_TOPOLOGY) - set(_STRUCTURE_SCOPED_STATS_KEYS))}"
    )
    # The three sets that share `pipeline/` must not overlap, or the scrub would
    # take one of the survivors with it.
    assert not set(_STRUCTURE_SCOPED_STATS_KEYS) & set(_STRUCTURE_SCOPE_EXEMPT_KEYS)
    assert not set(_STRUCTURE_SCOPED_STATS_KEYS) & set(NORMALIZATION_STATS_KEYS)
    assert set(_EXEMPT) == set(_STRUCTURE_SCOPE_EXEMPT_KEYS) | set(
        NORMALIZATION_STATS_KEYS
    )


def _chol4(n: int) -> np.ndarray:
    """Isotropic sigma=2 Cholesky factors for 4D splats (10 packed entries)."""
    chol = np.zeros((n, 10), dtype=np.float32)
    chol[:, [d * (d + 1) // 2 + d for d in range(4)]] = 2.0
    return chol


def _data(n: int, *, stats: Dict[str, Any], time_step: float = 1.0) -> GSplatData:
    """4D splats on ``n_tps=3`` timepoints spaced ``time_step`` apart."""
    rng = np.random.default_rng(0)
    centers = np.empty((n, 4), dtype=np.float32)
    centers[:, :3] = rng.random((n, 3)) * 100.0
    centers[:, 3] = rng.integers(0, 3, size=n).astype(np.float32) * time_step
    return GSplatData(
        centers=centers,
        amplitudes=np.linspace(1.0, 0.1, n).astype(np.float32),
        cholesky_factors=_chol4(n),
        stats={**stats, "n_splats": n},
    )


def _fixture(path: Path, n: int = 200) -> Path:
    """A store publishing the full non-default topology record."""
    _data(n, stats={**_TOPOLOGY, **_EXEMPT, **_DESCRIPTIVE}).save(
        path, include_fitting_info=True
    )
    return path


def _run(argv: Sequence[str], src: Path, out: Path) -> None:
    resolved = [a.format(**{"in": str(src), "out": str(out)}) for a in argv]
    result = CliRunner().invoke(app, ["gsplat", *resolved])
    assert result.exit_code == 0, result.output


def _pipeline_attrs(path: Path) -> Dict[str, Any]:
    """The store's ``pipeline/`` group attrs, ``{}`` when the group is absent."""
    return dict(read_node_attrs(path / "pipeline") or {})


def _fitting_attrs(path: Path) -> Dict[str, Any]:
    return dict(read_node_attrs(path / "fitting") or {})


def _root_stats(path: Path) -> Dict[str, Any]:
    _, stats = load_gsplat_node(path, include_stats=True)
    return dict(stats or {})


def _ordering_barriers(path: Path) -> List[List[int]]:
    """Every ``slice_dims`` in the store, one per splat-holding group.

    That attr IS the write-side ordering barrier (its complement is
    ``ordering_dims``), so it is what a dropped ``coarsen_dims`` would move — a
    partition puts one under each ``part_*``, a flat leaf one at the root.
    """
    found: List[List[int]] = []

    def walk(group: "zarr.Group") -> None:
        if "slice_dims" in group.attrs:
            found.append([int(d) for d in group.attrs["slice_dims"]])
        for name in group.group_keys():
            walk(group[name])

    walk(zarr.open_group(str(path), mode="r"))
    return found


#: ``(id, argv)`` for the three commands that publish a different structure KIND
#: than they were given. Both writers are covered — see the module docstring.
_KIND_CHANGING: List[tuple[str, Sequence[str]]] = [
    ("flatten", ("flatten", "{in}", "{out}")),
    ("decimate", ("decimate", "{in}", "{out}", "--target", "50")),
    ("partition", ("partition", "{in}", "{out}", "--parts", "4")),
]


@pytest.mark.parametrize(
    "argv", [argv for _, argv in _KIND_CHANGING], ids=[i for i, _ in _KIND_CHANGING]
)
def test_a_kind_change_drops_the_topology_record(
    tmp_path: Path, argv: Sequence[str]
) -> None:
    src = _fixture(tmp_path / "pyr.gsplats.zarr")
    assert set(_pipeline_attrs(src)) >= set(_TOPOLOGY), (
        "the fixture published no topology record — the test would prove nothing"
    )
    out = tmp_path / "out.gsplats.zarr"
    _run(argv, src, out)

    pipeline = _pipeline_attrs(out)
    survivors = [k for k in _STRUCTURE_SCOPED_STATS_KEYS if k in pipeline]
    assert not survivors, f"stale topology stamps on disk: {survivors}"
    # ...and not smuggled back in through the loader's merged view either.
    stats = _root_stats(out)
    assert not [k for k in _STRUCTURE_SCOPED_STATS_KEYS if k in stats], (
        f"the loaded stats still carry the topology record: {stats}"
    )

    # The survivors, by key and by value.
    for key, value in _EXEMPT.items():
        got = pipeline.get(key)
        assert (list(got) if isinstance(got, list) else got) == value, (
            f"{key!r} must survive a structure change (got {got!r})"
        )
    # The fit provenance still round-trips through `fitting/` (#1737 / #1601).
    fitting = _fitting_attrs(out)
    for key, value in _DESCRIPTIVE.items():
        got = fitting.get(key)
        assert (list(got) if isinstance(got, list) else got) == value, (
            f"{key!r} was dropped from fitting/ (got {got!r})"
        )
        assert key in stats


@pytest.mark.parametrize(
    "argv", [argv for _, argv in _KIND_CHANGING], ids=[i for i, _ in _KIND_CHANGING]
)
def test_the_ordering_barrier_survives_the_scrub(
    tmp_path: Path, argv: Sequence[str]
) -> None:
    """The ``coarsen_dims`` trap, asserted on the LAYOUT rather than the key.

    ``_barrier_from_coarsen_dims`` derives the chunk-ordering barrier from the
    complement of ``coarsen_dims``, so a scrub that took it would silently smear
    every chunk across the stacked axis. The timepoints sit on a HALF-INTEGER
    grid on purpose: ``detect_barrier_dims`` does not recognise that as
    categorical, so the provenance is the ONLY thing that can produce the
    barrier — the negative control below measures exactly that.
    """
    src = tmp_path / "pyr.gsplats.zarr"
    _data(200, stats={**_TOPOLOGY, **_EXEMPT, **_DESCRIPTIVE}, time_step=0.5).save(
        src, include_fitting_info=True
    )
    out = tmp_path / "out.gsplats.zarr"
    _run(argv, src, out)

    barriers = _ordering_barriers(out)
    assert barriers, "no ordering attrs written at all"
    assert all(b == [3] for b in barriers), (
        f"the stacked axis lost its ordering barrier: {barriers}"
    )

    # Negative control: with no `coarsen_dims` provenance the same store gets NO
    # barrier, so the assertion above is really reading the exempted key.
    bare = tmp_path / "bare.gsplats.zarr"
    _data(200, stats=dict(_TOPOLOGY), time_step=0.5).save(
        bare, include_fitting_info=True
    )
    bare_out = tmp_path / "bare_out.gsplats.zarr"
    _run(argv, bare, bare_out)
    assert all(b == [] for b in _ordering_barriers(bare_out)), (
        "auto-detect found the barrier on its own; the control proves nothing"
    )


#: Recipes run to MEASURE what the builders stamp. ``levels`` exercises the
#: substitutive producer (and, via its default ladder, the additive one);
#: ``stream`` the additive producer alone; ``flat`` the degenerate case; ``tiles``
#: the composed node-tree write path, whose only root stamp is ``recipe``.
_MEASURED_RECIPES: List[Sequence[str]] = [
    ("--recipe", "flat"),
    ("--recipe", "stream", "--n-lods", "3"),
    ("--recipe", "levels", "-K", "4", "-L", "3", "--coarsen-dims", "0,1,2"),
    ("--recipe", "tiles", "--max-elements", "80"),
]


def _stamped_topology_keys(tmp_path: Path) -> "set[str]":
    """What the PRODUCERS actually put in a root ``pipeline/`` group.

    Measured, not copied from the registry: a hand-written mirror of the constant
    would make the completeness check below tautological. Two sources, both real
    — ``gsplat lod``'s own output for every recipe family, and
    ``_recipe_pipeline_info``, the literal stamp site of the ``batch-fit merge``
    per-part recipe (the only producer of ``per_part`` / ``n_lods`` /
    ``breakpoints`` / ``levels`` / ``additive_ladders``).
    """
    src = tmp_path / "bare.gsplats.zarr"
    _data(200, stats=dict(_DESCRIPTIVE)).save(src, include_fitting_info=True)
    assert not _pipeline_attrs(src), (
        "the bare fixture already has a pipeline group; the diff would be muddied"
    )

    keys: "set[str]" = set()
    for i, extra in enumerate(_MEASURED_RECIPES):
        out = tmp_path / f"recipe_{i}.gsplats.zarr"
        _run(("lod", "{in}", "{out}", *extra), src, out)
        keys |= set(_pipeline_attrs(out))

    from luxar.gsplats.batch.merge_orchestrator import _recipe_pipeline_info
    from luxar.gsplats.lod.recipes import RecipeParams

    for recipe in ("stream", "levels"):
        keys |= set(_recipe_pipeline_info(recipe, RecipeParams()) or {})
    return keys


def test_every_topology_stamp_the_builders_produce_is_classified(
    tmp_path: Path,
) -> None:
    """A builder cannot start stamping a topology key without a classification.

    Closes the registry against the producers in BOTH directions: a new stamp
    that is neither structure-scoped nor explicitly exempt fails the first
    assertion (it would ride through a ``flatten`` as a lie), and a registry key
    no producer writes any more fails the second (dead weight, and a sign the
    stamp was renamed rather than removed).
    """
    stamped = _stamped_topology_keys(tmp_path)
    assert "lod_kind" in stamped and "recipe" in stamped, (
        f"the measurement missed something obvious: {sorted(stamped)}"
    )

    classified = (
        set(_STRUCTURE_SCOPED_STATS_KEYS)
        | set(_STRUCTURE_SCOPE_EXEMPT_KEYS)
        | set(NORMALIZATION_STATS_KEYS)
    )
    assert not stamped - classified, (
        "unclassified topology stamp(s) — add them to "
        "_STRUCTURE_SCOPED_STATS_KEYS (they will be dropped on a structure "
        "change) or to _STRUCTURE_SCOPE_EXEMPT_KEYS with the reason: "
        f"{sorted(stamped - classified)}"
    )
    assert not set(_STRUCTURE_SCOPED_STATS_KEYS) - stamped, (
        "the registry names keys no producer stamps any more: "
        f"{sorted(set(_STRUCTURE_SCOPED_STATS_KEYS) - stamped)}"
    )
