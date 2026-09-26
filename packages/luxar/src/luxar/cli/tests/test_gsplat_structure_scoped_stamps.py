"""No ``gsplat`` command may advertise a topology it does not have.

The third hygiene axis of #1600 (the first two live in
``test_gsplat_content_scoped_metrics.py`` and its domain twin). Four commands
change the artifact's STRUCTURE KIND while threading the input's ``stats``
through to the output's ``pipeline/`` group, so a flattened four-level pyramid
published ``lod_kind: substitutive`` / ``n_substitutive_levels: 4`` /
``lod_cutpoints: [2, 4, 5, 7]`` for a store that is a single flat leaf:

* ``flatten``   → one flat leaf (writes through ``GSplatData.save``)
* ``decimate``  → one flat leaf (writes through ``write_gsplats_tree``)
* ``partition`` → ``kind=partition`` of bare leaves (``write_gsplats_tree``)
* ``lod``       → whatever ``--recipe`` says, from any matrix-shaped input: a
  laddered or substitutive store is legal (the gate refuses a partition and a
  lod group with non-leaf children) and every recipe starts from
  ``data.flattened()``, so no ``lod`` output preserves its input's shape

Both writers are represented on purpose: a fix wired into one would leave the
other silently exempt. ``lod`` is covered per RECIPE for the same reason — the
recipe is where this hid, because ``stream`` re-stamps the ladder half of the
record (making it look handled) while inheriting the substitutive half, and
``flat`` published a full substitutive record one line above its own
``recipe: flat``. Unlike the other three, ``lod`` DOES publish a topology
record: its own. So each row states the exact key set its builder stamps, and
absence is the claim — no positive "this is not a pyramid" stamp is invented.

Two things must SURVIVE the scrub, which is why it is a deny-list of topology
keys rather than "drop the ``pipeline/`` group": the normalization block (the
input volume's intensity scale) and ``coarsen_dims``, which the writer READS BACK
to derive the chunk-ordering barrier — so a naive scrub does not merely delete a
stamp, it changes the output's layout.

Surviving the scrub is not the same as being TRUE, and the two rewrites that
preserve the structure KIND while invalidating a stamp inside it are pinned here
as well: ``decimate``'s ``coarsen_dims`` (which reduction axes it merged over)
and ``additive``'s root ladder summary (which ladder the store now has).

:func:`test_every_gsplat_command_is_classified` closes the table against the
commands that actually exist. ``lod`` slipped the first pass of this axis
precisely because nothing did that — the sibling appearance-carry guard in
``test_gsplat_cli_extended.py`` had already learned the lesson from ``merge``.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any, Dict, List, Sequence

import numpy as np
import pytest
import zarr
from typer.testing import CliRunner

from luxar._zarr_compat import open_group, read_node_attrs
from luxar.cli import app
from luxar.cli.tests.test_gsplat_content_scoped_metrics import (
    _NO_PROVENANCE as _CONTENT_NO_PROVENANCE,
)
from luxar.cli.tests.test_gsplat_content_scoped_metrics import (
    _REWRITE_CASES as _CONTENT_REWRITE_CASES,
)
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
    "color_weight": 6.0,
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
    "floor_strategy": "specimen",
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
    # The per-row "what this command stamps itself" sets are spelled literally
    # (they were measured), so anchor them to the registry too: a key that fell
    # out of it would otherwise sit in a row unchecked, since the row test only
    # ever iterates keys that are IN the registry.
    published = {k for _, _, keys in _KIND_CHANGING for k in keys}
    assert not published - set(_STRUCTURE_SCOPED_STATS_KEYS), (
        "a _KIND_CHANGING row names key(s) the registry does not: "
        f"{sorted(published - set(_STRUCTURE_SCOPED_STATS_KEYS))}"
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

    # Through the facade, so this reads a v2 and a v3 store alike (and ignores
    # the consolidated index, which the writer may have left one level up).
    walk(open_group(path, mode="r"))
    return found


#: The substitutive producer's own block and the additive producer's ladder
#: summary, as MEASURED at the root of a ``gsplat lod`` output. Split out because
#: the two halves are stamped independently — which is what let ``stream`` look
#: handled (it re-stamps the ladder) while carrying a false substitutive block.
_SUBSTITUTIVE_BLOCK = frozenset(
    {
        "lod_kind",
        "compression_factor",
        "method",
        "n_substitutive_levels",
        "coverage_inflation",
        "color_weight",
        "conserve_mass",
        "refine",
        "refine_iters",
    }
)
_LADDER_SUMMARY = frozenset(
    {
        "lod_method",
        "lod_n_lods",
        "lod_breakpoints_kind",
        "lod_cutpoints",
        "lod_substitutive_level",
    }
)

#: ``(id, argv, published)`` for every command that publishes a different
#: structure KIND than it was given. Both writers are covered — see the module
#: docstring. ``published`` is the EXACT set of topology keys that command's own
#: builder stamps at the root, so the row pins both halves of the claim: nothing
#: inherited, and nothing invented either. The three non-``lod`` commands build
#: no topology at all, hence the empty sets.
#:
#: The five ``_recipe_pipeline_info`` keys (``per_part`` / ``n_lods`` /
#: ``breakpoints`` / ``levels`` / ``additive_ladders``) appear in NO row: only a
#: ``batch-fit merge`` stamps them, and it is a producer rather than a rewriter.
#: The fixture plants them anyway, so every row below proves they are dropped.
_KIND_CHANGING: List[tuple[str, Sequence[str], frozenset[str]]] = [
    ("flatten", ("flatten", "{in}", "{out}"), frozenset()),
    ("decimate", ("decimate", "{in}", "{out}", "--target", "50"), frozenset()),
    ("partition", ("partition", "{in}", "{out}", "--parts", "4"), frozenset()),
    # `lod`, one row per recipe: the recipe axis is where the defect hid.
    ("lod:flat", ("lod", "{in}", "{out}", "--recipe", "flat"), frozenset({"recipe"})),
    (
        "lod:stream",
        ("lod", "{in}", "{out}", "--recipe", "stream", "--n-lods", "3"),
        _LADDER_SUMMARY | {"recipe"},
    ),
    (
        "lod:levels",
        (
            "lod",
            "{in}",
            "{out}",
            "--recipe",
            "levels",
            "-K",
            "4",
            "-L",
            "3",
            "--coarsen-dims",
            "0,1,2",
        ),
        _SUBSTITUTIVE_BLOCK | _LADDER_SUMMARY | {"recipe"},
    ),
    (
        "lod:tiles",
        ("lod", "{in}", "{out}", "--recipe", "tiles", "--max-elements", "80"),
        frozenset({"recipe"}),
    ),
    (
        "lod:overview",
        (
            "lod",
            "{in}",
            "{out}",
            "--recipe",
            "overview",
            "--max-elements",
            "80",
            "-K",
            "4",
            "--coarsen-dims",
            "0,1,2",
        ),
        frozenset({"recipe"}),
    ),
    (
        "lod:adaptive",
        (
            "lod",
            "{in}",
            "{out}",
            "--recipe",
            "adaptive",
            "--max-elements",
            "80",
            "-K",
            "4",
            "-L",
            "2",
            "--coarsen-dims",
            "0,1,2",
        ),
        frozenset({"recipe"}),
    ),
]

#: ``gsplat`` commands that do NOT change the structure kind, each with the
#: reason. Union'd with :data:`_KIND_CHANGING` by
#: :func:`test_every_gsplat_command_is_classified`, so a NEW command has to say
#: which it is instead of inheriting a silent pass — which is exactly how ``lod``
#: slipped this axis's first pass.
_NOT_KIND_CHANGING: Dict[str, str] = {
    # ── structure-PRESERVING rewrites: the kind stays true of the output ──
    "additive": "re-ladders every leaf in place; leaf / lod levels / partition "
    "parts all keep their shape (the root ladder SUMMARY does move with the "
    "re-ladder and is re-stamped from the tree written — "
    "test_additive_refreshes_the_root_ladder_summary)",
    "cull": "a culled pyramid is still that pyramid (and the counts that moved "
    "are re-stamped, not dropped)",
    "filter": "same splat set narrowed, same tree shape",
    "slice": "a coordinate-range filter; structure untouched",
    "transform": "geometry only",
    "reencode": "preserves the tree verbatim",
    "migrate-format": "translates a legacy LAYOUT to the current one; the "
    "structure kind it describes is the same one",
    "annotate-quality": "rewrites the input in place, stamping quality attrs; "
    "no structural rewrite at all",
    # ── producers: the record they publish is their own, freshly stamped ──
    "fit": "produces the fit; there is no inherited topology to invalidate",
    "merge": "publishes no inherited provenance at all",
    "batch-fit run": "produces the fit (local multi-GPU)",
    "batch-fit submit": "produces the fit (Slurm array)",
    "batch-fit merge": "assembles a kind=partition from its OWN tiles and "
    "stamps the per-part record fresh (`_recipe_pipeline_info`)",
    # ── not an in->out .gsplats.zarr rewrite ──
    "cal": "writes a calibration JSON",
    "benchmark": "GPU profile, no dataset output",
    "compare": "measures, writes no store",
    "render": "writes a volume",
    "denoise": "operates on a volume, not on splats",
    "convert": "writes a .luxar.zarr scene, not a .gsplats.zarr",
    "export": "writes a classical PLY, which carries no Luxar stats",
    "import": "reads a foreign file that carries no Luxar stats",
    "info": "read-only",
    "view": "read-only (serves a viewer)",
    "napari": "read-only",
    "doctor": "read-only diagnostics",
    "batch-fit status": "read-only",
    "batch-fit validate": "read-only (or deletes corrupt tiles with --fix)",
    "batch-fit cancel": "cancels Slurm jobs",
    "batch-fit denoise-calibrate": "operates on a volume, writes JSON",
    "batch-fit denoise-preprocess": "operates on a volume",
    "batch-fit resolve-floor": "resolves a floor level, writes JSON/manifest",
}


def _command_of(argv: Sequence[str], registered: "set[str]") -> str:
    """The command an argv row invokes, resolved against the REGISTERED names.

    Read off the argv rather than the row id: ``lod`` appears under six
    recipe-suffixed ids, and a command can be two tokens (``batch-fit merge``).
    """
    two = " ".join(argv[:2])
    return two if two in registered else argv[0]


def test_every_gsplat_command_is_classified() -> None:
    """A new command cannot skip this axis by not being in the table.

    ``_KIND_CHANGING`` was a hand-written three-row table with nothing closing it
    against the commands that exist, and ``lod`` — a fourth publisher of the same
    defect, across six recipes — slipped the whole first pass. The sibling
    appearance-carry guard (``test_gsplat_cli_extended.py``) had already learned
    this from ``merge``; its command enumerator is imported rather than copied so
    the three guards cannot drift on what "a gsplat command" is.

    A command that genuinely changes the structure kind belongs in
    ``_KIND_CHANGING`` — i.e. it has to be FIXED, not merely listed. There is no
    "known-broken" bucket on purpose: after #1600 no gsplat rewriter is in that
    state, and offering the bucket is how one gets there again.
    """
    from luxar.cli.tests.test_gsplat_content_scoped_metrics import (
        _registered_gsplat_commands,
    )

    registered = _registered_gsplat_commands()
    assert "lod" in registered and "batch-fit merge" in registered, (
        f"the enumeration missed something obvious: {sorted(registered)}"
    )

    classified = {_command_of(argv, registered) for _, argv, _ in _KIND_CHANGING} | set(
        _NOT_KIND_CHANGING
    )
    assert not registered - classified, (
        "unclassified gsplat command(s) — either it changes the structure kind "
        "(fix it to scrub, and add a _KIND_CHANGING row stating the topology "
        "keys it stamps itself) or it does not (add a _NOT_KIND_CHANGING reason): "
        f"{sorted(registered - classified)}"
    )
    assert not classified - registered, (
        f"the tables name commands that no longer exist: "
        f"{sorted(classified - registered)}"
    )
    assert all(_NOT_KIND_CHANGING.values()), "every exemption needs a reason"


#: The rows whose ``coarsen_dims`` provenance is untouched by the command, so the
#: written ordering barrier is a clean read of the exempted key. ``lod`` is left
#: out: its substitutive builder RE-STAMPS ``coarsen_dims`` from
#: ``--coarsen-dims`` (to the full ``[0, …, d-1]`` list when the flag is absent,
#: since that is the reduction it then performs), which is a claim about what
#: that builder stamps rather than about this scrub —
#: :func:`test_a_levels_build_stamps_the_coarsen_dims_it_used` measures it.
#: ``decimate`` is out for the same reason since #1600 — its ``merge`` family
#: re-stamps the dims it resolved, and
#: :func:`test_decimate_stamps_the_coarsen_dims_it_used` measures that barrier
#: (including the case where the INHERITED one must not be re-imposed).
_BARRIER_ROWS = [
    row
    for row in _KIND_CHANGING
    if not row[0].startswith("lod:") and row[0] != "decimate"
]


@pytest.mark.parametrize(
    ("argv", "published"),
    [(argv, published) for _, argv, published in _KIND_CHANGING],
    ids=[i for i, _, _ in _KIND_CHANGING],
)
def test_a_kind_change_drops_the_topology_record(
    tmp_path: Path, argv: Sequence[str], published: "frozenset[str]"
) -> None:
    src = _fixture(tmp_path / "pyr.gsplats.zarr")
    assert set(_pipeline_attrs(src)) >= set(_TOPOLOGY), (
        "the fixture published no topology record — the test would prove nothing"
    )
    out = tmp_path / "out.gsplats.zarr"
    _run(argv, src, out)

    pipeline = _pipeline_attrs(out)
    survivors = {k for k in _STRUCTURE_SCOPED_STATS_KEYS if k in pipeline}
    assert survivors == set(published), (
        f"published topology keys {sorted(survivors)} != this command's own "
        f"record {sorted(published)}; inherited: "
        f"{sorted(survivors - set(published))}, missing: "
        f"{sorted(set(published) - survivors)}"
    )
    # ...and not smuggled back in through the loader's merged view either.
    stats = _root_stats(out)
    assert {k for k in _STRUCTURE_SCOPED_STATS_KEYS if k in stats} == set(published), (
        f"the loaded stats disagree with the on-disk record: {stats}"
    )
    # What IS published must be the command's own value, not the fixture's. Only
    # `recipe` is checked positively (it names this run); `lod_kind` is skipped
    # because its only value is the mechanism name, so a re-stamp and a carry are
    # indistinguishable BY VALUE there — the key-set assertion above is what
    # covers it (a `flat` leaf publishing `lod_kind` at all is the defect).
    for key in sorted(published - {"recipe", "lod_kind"}):
        assert pipeline[key] != _TOPOLOGY[key], (
            f"{key!r} still carries the input's value {_TOPOLOGY[key]!r} — "
            "re-stamped from the inherited record rather than freshly built"
        )
    if "recipe" in published:
        assert pipeline["recipe"] == argv[argv.index("--recipe") + 1]

    # The survivors, by key and by value. ``decimate`` is the one exception, and
    # only on ``coarsen_dims``: it does not pass that key through, it RE-STAMPS
    # the dims its reduction actually coarsened over (#1600), which for the
    # default merge here is "all of them". Pinned by
    # :func:`test_decimate_stamps_the_coarsen_dims_it_used`.
    restamped = {"coarsen_dims"} if argv[0] == "decimate" else set()
    for key, value in _EXEMPT.items():
        if key in restamped:
            continue
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
    "argv",
    [argv for _, argv, _ in _BARRIER_ROWS],
    ids=[i for i, _, _ in _BARRIER_ROWS],
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


#: ``(id, lod tail, additive tail, method, breakpoints kind)`` — one row per
#: SHAPE ``gsplat additive`` re-ladders and can summarise: a single flat leaf
#: (the summary IS that leaf's ladder) and a ``kind=lod`` group (the summary is
#: the level ``lod_substitutive_level`` names, the rule
#: ``_map_substitutive`` / ``cull`` already follow). Both rows move the METHOD
#: and the rung count together, because the summary is refreshed key by key —
#: a row that only moved the counts would leave ``lod_method`` unmeasured.
_RE_LADDERED: List[tuple[str, Sequence[str], Sequence[str], str, str]] = [
    (
        "leaf",
        ("--recipe", "stream", "--n-lods", "3"),
        ("--n-lods", "6", "--add-method", "mass"),
        "mass",
        "equal-count",
    ),
    (
        "levels",
        (
            "--recipe",
            "levels",
            "-K",
            "4",
            "-L",
            "1",
            "--n-lods",
            "2",
            "--coarsen-dims",
            "0,1,2",
        ),
        ("-b", "counts:5,15,40", "--add-method", "self_energy"),
        "self_energy",
        "explicit-counts",
    ),
]


@pytest.mark.parametrize(
    ("lod_tail", "additive_tail", "method", "kind"),
    [(lod, add, method, kind) for _, lod, add, method, kind in _RE_LADDERED],
    ids=[row[0] for row in _RE_LADDERED],
)
def test_additive_refreshes_the_root_ladder_summary(
    tmp_path: Path,
    lod_tail: Sequence[str],
    additive_tail: Sequence[str],
    method: str,
    kind: str,
) -> None:
    """``additive`` REPLACES the ladder its root block summarises (#1600).

    Every leaf's own stats are rebuilt (``_merged_leaf_meta`` exists for that),
    but the root ``pipeline/`` block was threaded through from the input, so a
    store re-laddered from three rungs to six went on advertising three, under
    the method and breakpoints kind of the ladder that no longer existed. The
    structure KIND is untouched here — this is the stale-VALUE half of the same
    hygiene axis, which is why ``additive`` sits in ``_NOT_KIND_CHANGING``.
    """
    src = _fixture(tmp_path / "src.gsplats.zarr")
    pyr = tmp_path / "pyr.gsplats.zarr"
    _run(("lod", "{in}", "{out}", *lod_tail), src, pyr)
    before = _pipeline_attrs(pyr)
    assert set(_LADDER_SUMMARY) <= set(before), (
        f"the input published no ladder summary; nothing to go stale: {before}"
    )

    out = tmp_path / "out.gsplats.zarr"
    _run(("additive", "{in}", "{out}", *additive_tail), pyr, out)
    after = _pipeline_attrs(out)

    # The summary must describe the ladder that is ON DISK, rung for rung.
    loaded = GSplatData.load(out)
    level = loaded.substitutive_levels[int(after["lod_substitutive_level"])]
    counts = [lod.n_splats for lod in level.additive_sublods]
    assert after["lod_n_lods"] == len(counts)
    assert after["lod_cutpoints"] == list(np.cumsum(counts))
    assert after["lod_method"] == method
    assert after["lod_breakpoints_kind"] == kind
    # ...and it MOVED. Without these the row would pass on a ladder that the
    # rebuild happened to reproduce, proving nothing about the refresh.
    assert before["lod_n_lods"] != after["lod_n_lods"]
    assert before["lod_method"] != after["lod_method"]
    # A re-ladder moves no LEVEL, so the summary still names the same one.
    assert after["lod_substitutive_level"] == before["lod_substitutive_level"]


def test_additive_refreshes_the_batch_merge_spelling_of_the_same_summary(
    tmp_path: Path,
) -> None:
    """The OTHER root ladder summary in the same ``pipeline/`` group (#1600).

    ``batch-fit merge --recipe stream`` stamps its per-part ladder knobs WITHOUT
    the ``lod_`` prefix (``_recipe_pipeline_info``), and ``gsplat additive`` over
    a batch-fit partition is an advertised use case — so a store re-laddered
    from six rungs to two dropped ``lod_n_lods`` (a partition has no single
    summary) while ``n_lods: 6`` rode through three keys away, asserting exactly
    the number the drop exists to avoid asserting.

    The fixture's ``pipeline/`` block comes from the producer itself, so a knob
    added there cannot quietly go unhandled.
    """
    from luxar.gsplats.batch.merge_orchestrator import _recipe_pipeline_info
    from luxar.gsplats.io.save_gsplats import write_gsplats_tree
    from luxar.gsplats.lod.additive import make_additive_lod
    from luxar.gsplats.lod.recipes import RecipeParams
    from luxar.gsplats.tree import GSplatLeaf, iter_leaves, map_leaves

    pipeline = _recipe_pipeline_info(
        "stream", RecipeParams(n_lods=6, additive_method="mass")
    )
    assert pipeline and pipeline["n_lods"] == 6

    parts = _data(200, stats={}).to_spatial_partition(max_elements=80)
    laddered = map_leaves(
        parts,
        lambda leaf: make_additive_lod(GSplatData.from_tree(leaf), n_lods=6).tree,
    )
    src = tmp_path / "batch.gsplats.zarr"
    write_gsplats_tree(src, laddered, pipeline_info=dict(pipeline))
    assert _pipeline_attrs(src)["n_lods"] == 6
    assert all(len(lf.additive_sublods) == 6 for lf in iter_leaves(laddered))

    out = tmp_path / "out.gsplats.zarr"
    _run(("additive", "{in}", "{out}", "--n-lods", "2"), src, out)

    node, _ = load_gsplat_node(out, include_stats=True)
    leaves = [lf for lf in iter_leaves(node) if isinstance(lf, GSplatLeaf)]
    assert leaves and all(len(lf.additive_sublods) == 2 for lf in leaves), (
        "the re-ladder did not happen; the assertion below would prove nothing"
    )
    after = _pipeline_attrs(out)
    assert not {"n_lods", "method", "breakpoints"} & set(after), (
        "the parts hold different ladders, so no root rung count/ordering is "
        f"true of more than one of them: {after}"
    )
    # Still true of the output, and therefore untouched: every leaf was
    # laddered independently, by the additive mechanism, under the same recipe.
    assert after["per_part"] is True
    assert after["lod_kind"] == "additive"
    assert after["recipe"] == "stream"


# ── the generic post-condition: a published ladder must be the one on disk ──


#: The root keys that state a RUNG COUNT for one ladder, in both spellings —
#: ``make_additive_lod``'s and ``_recipe_pipeline_info``'s.
_LADDER_COUNT_KEYS = ("lod_n_lods", "lod_cutpoints", "n_lods")


def _assert_published_ladder_matches_disk(path: Path) -> None:
    """If a store publishes a ladder rung count, it must be the ladder it has.

    The generic form of the two hand-written ``additive`` claims above, applied
    to EVERY rewriting command instead of the ones somebody remembered to list.
    That closure is the point: the kind-change rule is held against the
    registered command set twice over, while "a re-ladder must refresh the root
    summary" lived in two hand-written rows — the same shape as the ``merge`` /
    ``lod`` misses this module exists to prevent.

    Deliberately independent of :func:`~luxar.gsplats.lod.restamp
    ._root_summary_leaf`: the level is resolved here from the tree's own
    coarsest-first invariant, so the production helper cannot certify itself.
    """
    from luxar.gsplats.tree import GSplatLeaf, GSplatLodGroup

    pipeline = _pipeline_attrs(path)
    published = {k: pipeline[k] for k in _LADDER_COUNT_KEYS if k in pipeline}
    if not published:
        return  # absent is always honest — nothing is claimed

    node, _ = load_gsplat_node(path, include_stats=True)
    if pipeline.get("per_part") is True:
        # The one shape where a root count over MANY ladders is legitimate, so
        # it gets its own rule rather than the raise below.
        _assert_per_part_ladder_matches_disk(path, node, published)
        return
    if isinstance(node, GSplatLeaf):
        levels: List[Any] = [node]
    elif isinstance(node, GSplatLodGroup) and all(
        isinstance(child, GSplatLeaf) for child in node.children
    ):
        # A tree lod group stores children coarsest-first; the summary index is
        # a MATRIX index (finest-first), hence the mirror.
        levels = list(reversed(node.children))
    else:
        raise AssertionError(
            f"{path.name} publishes a root ladder summary {sorted(published)} on "
            "a shape that HAS no single ladder (a partition, or a lod group "
            "whose summary child is a subtree) without the `per_part` stamp that "
            "would make it a claim about every part — the number is true of at "
            "most one part, so the block must be dropped instead"
        )

    index = int(pipeline.get("lod_substitutive_level") or 0)
    assert 0 <= index < len(levels), (
        f"{path.name} summarises level {index} of {len(levels)} — the store "
        "does not have the level its own summary names"
    )
    counts = [int(sub.n_splats) for sub in levels[index].additive_sublods]
    cutpoints = [int(value) for value in np.cumsum(counts)]
    if "lod_n_lods" in published:
        assert published["lod_n_lods"] == len(counts), (
            f"{path.name} advertises {published['lod_n_lods']} rungs; the "
            f"ladder on disk has {len(counts)}"
        )
    if "n_lods" in published:
        assert published["n_lods"] == len(counts), (
            f"{path.name} advertises {published['n_lods']} rungs (the "
            f"batch-merge spelling); the ladder on disk has {len(counts)}"
        )
    if "lod_cutpoints" in published:
        assert list(published["lod_cutpoints"]) == cutpoints, (
            f"{path.name} advertises cutpoints {list(published['lod_cutpoints'])}; "
            f"the ladder on disk cuts at {cutpoints}"
        )


def _assert_per_part_ladder_matches_disk(
    path: Path, node: Any, published: Dict[str, Any]
) -> None:
    """A ``per_part`` count is honest iff EVERY leaf's ladder agrees with it.

    ``batch-fit merge --recipe stream --n-lods 6`` stamps ``n_lods`` on a
    ``kind=partition`` ROOT under ``per_part: True`` on purpose
    (``_recipe_pipeline_info``), and one recipe built every part, so the number
    can be true of all of them at once. Sending that store to the
    single-summary-leaf rule above would make the guard raise on CORRECT output
    — the worst failure mode a guard has, because the reflex is then to "fix"
    the producer (#1600 review). Today no row reaches here (every fixture is a
    flat leaf and ``batch-fit merge`` is not a rewriter), which is exactly why
    the branch carries its own honest/dishonest pair of tests.

    A rung COUNT can hold across parts; a cutpoint LIST generally cannot, since
    the parts hold different splat counts — so it is checked the same way and
    simply fails for a producer that publishes one.
    """
    from luxar.gsplats.tree import GSplatLeaf, iter_leaves

    leaves = [lf for lf in iter_leaves(node) if isinstance(lf, GSplatLeaf)]
    assert leaves, (
        f"{path.name} publishes a per-part ladder summary {sorted(published)} "
        "over a store with no leaves at all"
    )
    counts = sorted({len(lf.additive_sublods) for lf in leaves})
    for key in ("n_lods", "lod_n_lods"):
        if key in published:
            assert counts == [int(published[key])], (
                f"{path.name} advertises {published[key]} rungs per part "
                f"({key}, per_part: True); the {len(leaves)} leaves on disk "
                f"carry {counts}"
            )
    if "lod_cutpoints" in published:
        cuts = sorted(
            {
                tuple(
                    int(v)
                    for v in np.cumsum([int(s.n_splats) for s in lf.additive_sublods])
                )
                for lf in leaves
            }
        )
        assert cuts == [tuple(int(v) for v in published["lod_cutpoints"])], (
            f"{path.name} advertises cutpoints "
            f"{list(published['lod_cutpoints'])} per part; the leaves on disk "
            f"cut at {cuts}"
        )


def _per_part_partition(tmp_path: Path, name: str, rungs: Sequence[int]) -> Path:
    """A ``kind=partition`` whose parts carry ``rungs[i]`` rungs, root ``n_lods=6``.

    The shape ``batch-fit merge --recipe stream --n-lods 6`` writes: the root
    block comes from ``_recipe_pipeline_info`` itself rather than a hand-written
    dict, so the honest case below is the producer's own output.
    """
    from dataclasses import replace

    from luxar.gsplats.batch.merge_orchestrator import _recipe_pipeline_info
    from luxar.gsplats.io.save_gsplats import write_gsplats_tree
    from luxar.gsplats.lod.additive import make_additive_lod
    from luxar.gsplats.lod.recipes import RecipeParams
    from luxar.gsplats.tree import GSplatPartition

    pipeline = _recipe_pipeline_info("stream", RecipeParams(n_lods=6))
    assert pipeline and pipeline["per_part"] is True and pipeline["n_lods"] == 6

    parts = _data(200, stats={}).to_spatial_partition(max_elements=80)
    assert isinstance(parts, GSplatPartition) and len(parts.children) == len(rungs)
    laddered = replace(
        parts,
        children=[
            make_additive_lod(GSplatData.from_tree(child), n_lods=n).tree
            for child, n in zip(parts.children, rungs)
        ],
    )
    out = tmp_path / name
    write_gsplats_tree(out, laddered, pipeline_info=dict(pipeline))
    return out


def test_the_ladder_guard_accepts_an_honest_per_part_partition(tmp_path: Path) -> None:
    """A root ``n_lods`` over parts that ALL have that many rungs is honest.

    The guard used to raise for any rung count on a partition, which is exactly
    what ``batch-fit merge --recipe stream --n-lods 6`` writes on purpose. It is
    unreachable from the rows today only because the fixture is a flat leaf, so
    widening either side would have turned correct output red (#1600 review).
    """
    honest = _per_part_partition(tmp_path, "honest.gsplats.zarr", [6, 6, 6, 6])
    assert _pipeline_attrs(honest)["n_lods"] == 6
    _assert_published_ladder_matches_disk(honest)  # must not raise


@pytest.mark.parametrize(
    "rungs", [[6, 6, 6, 2], [2, 2, 2, 2]], ids=["one-part-differs", "all-parts-differ"]
)
def test_the_ladder_guard_still_catches_a_dishonest_per_part_partition(
    tmp_path: Path, rungs: List[int]
) -> None:
    """...and ``per_part`` is a claim about EVERY part, not an exemption.

    Widening the guard must not turn it off: a root count no part has, and a
    root count only some parts have, both still fail.
    """
    bad = _per_part_partition(tmp_path, "dishonest.gsplats.zarr", rungs)
    with pytest.raises(AssertionError, match="rungs per part"):
        _assert_published_ladder_matches_disk(bad)


#: Every rewriting command, with the argv that drives it — imported rather than
#: restated so this post-condition inherits the sibling guard's closure against
#: the registry (``test_the_ladder_post_condition_covers_every_rewriter``).
_LADDER_ROWS: List[tuple[str, Sequence[str]]] = [
    *((row_id, argv) for row_id, argv, _ in _CONTENT_REWRITE_CASES),
    *_CONTENT_NO_PROVENANCE.items(),
]


def _data3(n: int, *, stats: Dict[str, Any]) -> GSplatData:
    """3D splats — what the imported argv rows are written against."""
    rng = np.random.default_rng(0)
    chol = np.zeros((n, 6), dtype=np.float32)
    chol[:, [0, 2, 5]] = 2.0
    return GSplatData(
        centers=(rng.random((n, 3)) * 100.0).astype(np.float32),
        amplitudes=np.linspace(1.0, 0.1, n).astype(np.float32),
        cholesky_factors=chol,
        stats={**stats, "n_splats": n},
    )


def _laddered_fixture(tmp_path: Path, n_lods: int = 6) -> Path:
    """A flat leaf whose ladder is summarised in BOTH spellings, honestly.

    ``gsplat lod --recipe stream`` stamps the ``lod_*`` five and nothing else, so
    a fixture built from it alone leaves the un-prefixed trio
    (:func:`~luxar.gsplats.batch.merge_orchestrator._recipe_pipeline_info`'s
    ``n_lods`` / ``method`` / ``breakpoints``, what ``batch-fit merge --recipe
    stream`` publishes for the same ladder) exercised by nothing but the one
    hand-written ``additive`` test above — precisely the "somebody remembered to
    list it" hole this post-condition exists to close (#1600 review). The trio's
    VALUES come from the producer rather than a literal, so a knob added there
    cannot quietly go unhandled here either.

    The producer's whole ``stream`` block is copied, not just the trio: it is
    ``lod_kind: "additive"`` that tells a consumer the shared ``method`` key is
    the ADDITIVE ordering (:func:`~luxar.gsplats.lod.restamp._recipe_ladder_keys`
    reads exactly that), so a fixture carrying the trio without it would be a
    shape no producer writes. ``per_part`` is the one key left off: this store is
    one flat leaf, not a partition, and the per-part branch of the guard already
    has its own honest/dishonest pair of tests above.

    Both spellings start out TRUE of the ladder on disk, so a row that goes red
    is describing the command's staleness and not the fixture's.
    """
    from luxar.gsplats.batch.merge_orchestrator import _recipe_pipeline_info
    from luxar.gsplats.lod.recipes import RecipeParams

    flat = tmp_path / "flat.gsplats.zarr"
    _data3(200, stats=dict(_DESCRIPTIVE)).save(flat, include_fitting_info=True)
    laddered = tmp_path / "laddered.gsplats.zarr"
    _run(
        ("lod", "{in}", "{out}", "--recipe", "stream", "--n-lods", str(n_lods)),
        flat,
        laddered,
    )

    data = GSplatData.load(laddered, include_stats=True)
    rungs = len(data.additive_sublods)
    assert rungs == n_lods, f"the ladder did not build: {rungs} rungs"
    recipe = _recipe_pipeline_info(
        "stream", RecipeParams(n_lods=rungs, additive_method="mass")
    )
    assert recipe and recipe["n_lods"] == rungs
    data.stats.update({k: v for k, v in recipe.items() if k != "per_part"})
    src = tmp_path / "both-spellings.gsplats.zarr"
    data.save(src, include_fitting_info=True)
    return src


@pytest.mark.parametrize(
    "argv", [argv for _, argv in _LADDER_ROWS], ids=[i for i, _ in _LADDER_ROWS]
)
def test_a_published_ladder_summary_describes_the_ladder_on_disk(
    tmp_path: Path, argv: Sequence[str]
) -> None:
    """No rewriter may publish a rung count its output does not have.

    One post-condition over every rewriting command, so the re-ladder rule does
    not depend on a hand-written row naming ``additive``. Refreshing, dropping
    and never having claimed anything all satisfy it — what it forbids is the
    one thing #1600 is about: a number that describes the ladder the command
    replaced.

    Both spellings of the count are in play (see :func:`_laddered_fixture`): a
    ``cull`` / ``filter`` that prunes rungs refreshed the ``lod_*`` half through
    ``_refresh_ladder_summary`` while the batch-merge ``n_lods`` rode through
    stale, which is what the trio in the fixture catches.
    """
    src = _laddered_fixture(tmp_path)
    assert {"lod_n_lods", "lod_cutpoints", "n_lods"} <= set(_pipeline_attrs(src)), (
        "the fixture publishes no ladder summary — nothing could go stale"
    )
    _assert_published_ladder_matches_disk(src)  # the fixture itself is honest

    out = tmp_path / "out.gsplats.zarr"
    _run(argv, src, out)
    # A row with no `{out}` rewrites the input in place.
    _assert_published_ladder_matches_disk(out if out.exists() else src)


def test_the_ladder_post_condition_covers_every_rewriter() -> None:
    """...and it is closed against the commands that exist, not a list.

    The sibling module already classifies every registered ``gsplat`` command as
    a rewriter (with the argv that drives it) or not, and its own guard test
    keeps that classification complete. Reusing it here means a NEW rewriting
    command is carried into this post-condition by the classification it already
    has to make, rather than by remembering this file.
    """
    from luxar.cli.tests.test_gsplat_content_scoped_metrics import (
        _NON_REWRITE,
        _registered_gsplat_commands,
    )

    registered = _registered_gsplat_commands()
    covered = {_command_of(argv, registered) for _, argv in _LADDER_ROWS}
    assert "additive" in covered, f"the enumeration missed the obvious: {covered}"
    assert not registered - (covered | set(_NON_REWRITE)), (
        "rewriting gsplat command(s) that no ladder post-condition runs over: "
        f"{sorted(registered - (covered | set(_NON_REWRITE)))}"
    )


#: ``(id, argv tail, time_step, stamp, barrier)`` for ``decimate``'s
#: ``coarsen_dims``. The fixture is stamped ``[1, 2, 3]`` (barrier ``[0]``) so a
#: re-stamp is visible BOTH in the key and in the layout.
#:
#: ``time_step`` picks the stacked axis's grid, and it is load-bearing per row:
#: the expected barrier must be one the writer's auto-detect fallback CANNOT
#: produce on THIS ROW'S OUTPUT, or the row passes on the fallback rather than on
#: the stamp. A half-integer grid is invisible to auto-detect (it finds nothing),
#: which serves the rows whose expected barrier is non-empty.
#:
#: The coarsen-EVERYTHING rows need the opposite — a grid where the fallback
#: would loudly disagree with the empty barrier — and a FINE integer grid does
#: not give it, which is what the first version of these rows got wrong. Merging
#: over the stacked axis averages neighbouring timepoints together, the
#: coordinates stop being integral, and ``detect_barrier_dims`` finds nothing on
#: the RESULT: measured at ``time_step=1.0``, the explicit stamp and the ``None``
#: spelling both write ``[[]]``, so the layout assertion held under the very
#: defect it exists to catch. Widely-separated timepoints (1000 apart against a
#: spatial extent of 100) are never clustered together, so the grid survives the
#: merge and the fallback re-imposes ``[3]``.
#: :func:`test_decimate_stamps_the_coarsen_dims_it_used`
#: measures that control per row rather than trusting this comment.
_DECIMATE_COARSEN: List[tuple[str, Sequence[str], float, Any, List[int]]] = [
    # The request the merge was given, and the barrier IT implies.
    (
        "merge-honours-the-request",
        ("-m", "merge", "--coarsen-dims", "0,1,2"),
        0.5,
        [0, 1, 2],
        [3],
    ),
    # No flag: the merge blends over every axis, so no axis is a barrier — and
    # the inherited one must not be re-imposed on splats it just blended. The
    # stamp is the EXPLICIT all-dims list: a written `null` is indistinguishable
    # from an absent key to `_barrier_from_coarsen_dims`, which falls through to
    # auto-detect and (on this widely-spaced grid, which the merge leaves
    # intact) hands back the barrier [3] the merge just blended over.
    ("merge-coarsens-everything", ("-m", "merge"), 1000.0, [0, 1, 2, 3], []),
    # The same reduction spelled out explicitly must publish the same stamp.
    (
        "merge-all-dims-spelled-out",
        ("-m", "merge", "--coarsen-dims", "0,1,2,3"),
        1000.0,
        [0, 1, 2, 3],
        [],
    ),
    # A prefix merges nothing, so the input's stamp is still true of the
    # survivors and stays — the ignored request must NOT be published.
    (
        "prefix-inherits",
        ("-m", "prefix", "--coarsen-dims", "0,1,2"),
        0.5,
        [1, 2, 3],
        [0],
    ),
]


@pytest.mark.parametrize(
    ("tail", "time_step", "stamp", "barrier"),
    [(tail, step, stamp, bar) for _, tail, step, stamp, bar in _DECIMATE_COARSEN],
    ids=[row[0] for row in _DECIMATE_COARSEN],
)
def test_decimate_stamps_the_coarsen_dims_it_used(
    tmp_path: Path,
    tail: Sequence[str],
    time_step: float,
    stamp: Any,
    barrier: List[int],
) -> None:
    """``coarsen_dims`` is exempt from the scrub, not exempt from being TRUE.

    The writer turns this key into the chunk-ordering barrier
    (``_barrier_from_coarsen_dims``), so publishing the input's list over a
    reduction that coarsened different axes does not merely misdescribe the
    output — it puts the barrier on an axis the merge blended, and the
    no-inherited-stamp direction dropped the user's own ``--coarsen-dims`` on
    the floor and fell back to auto-detect (#1600). Asserted on the LAYOUT as
    well as the key, for the same reason the scrub twin is.
    """
    src = tmp_path / "pyr.gsplats.zarr"
    _data(
        200,
        stats={**_TOPOLOGY, **_EXEMPT, **_DESCRIPTIVE, "coarsen_dims": [1, 2, 3]},
        time_step=time_step,
    ).save(src, include_fitting_info=True)
    out = tmp_path / "out.gsplats.zarr"
    _run(("decimate", "{in}", "{out}", "--target", "50", *tail), src, out)

    got = _pipeline_attrs(out).get("coarsen_dims")
    assert (list(got) if isinstance(got, list) else got) == stamp
    barriers = _ordering_barriers(out)
    assert barriers, "no ordering attrs written at all"
    assert all(b == barrier for b in barriers), (
        f"the written ordering barrier {barriers} does not follow the stamp"
    )

    # The control that makes the row mean something: with NO `coarsen_dims`
    # provenance THIS OUTPUT gets a DIFFERENT barrier, so the assertion above
    # cannot be satisfied by the auto-detect fallback. This is what the
    # coarsen-everything rows need most — spelling their stamp `None` would
    # write a null the writer reads as "no provenance", landing right here.
    auto = _auto_detected_barriers(out, tmp_path)
    assert auto and all(b != barrier for b in auto), (
        f"auto-detection produces {auto} on this result, which the expected "
        f"barrier {barrier} cannot be distinguished from — the row would pass "
        "on the fallback rather than on the stamp"
    )


def _auto_detected_barriers(out: Path, tmp_path: Path) -> List[List[int]]:
    """What the writer's fallback would place on THIS REDUCTION'S OUTPUT.

    Measured by re-saving the output with its ``coarsen_dims`` stamp removed, so
    ``_barrier_from_coarsen_dims`` returns ``None`` and the writer falls back to
    per-leaf auto-detection — the exact path a null-or-absent stamp lands on.

    The OUTPUT, not the input fixture, because that is the splat set the
    fallback would auto-detect from and a ``merge`` rewrites the very
    coordinates it coarsened over. Measuring the input answers a different
    question, and the two disagree exactly where it matters: on a fine integer
    grid auto-detection says ``[3]`` of the input fixture and ``[]`` of the
    coarsen-everything result, so a control taken on the input passed while the
    pre-fix ``None`` spelling wrote the same layout the row expects (#1600
    review).
    """
    node, stats = load_gsplat_node(out, include_stats=True)
    bare = dict(stats or {})
    bare.pop("coarsen_dims", None)
    probe = tmp_path / "autodetect.gsplats.zarr"
    GSplatData.from_tree(node, stats=bare).save(probe, include_fitting_info=True)
    return _ordering_barriers(probe)


#: ``(id, argv tail, time_step, stamp, barrier)`` for ``lod --recipe levels``'s
#: ``coarsen_dims`` — the substitutive builder's half of the same claim
#: :data:`_DECIMATE_COARSEN` makes for ``decimate``'s ``merge`` family, and the
#: reason the two tables look alike: both producers now resolve the stamp through
#: one shared :func:`~luxar.gsplats.lod.substitutive.resolved_merge_coarsen_dims`,
#: so a divergence between them is a test failure rather than a format quirk.
#:
#: ``time_step`` is load-bearing exactly as it is over there: the expected barrier
#: must be one the writer's auto-detect fallback CANNOT produce on THIS ROW'S
#: OUTPUT. Widely-separated timepoints (1000 apart against a spatial extent of
#: 100) are never clustered together, so the stacked axis' integer grid survives
#: the merge and the fallback re-imposes ``[3]`` — which is precisely the barrier
#: the coarsen-everything rows blended away. Measured on this fixture, the two
#: coarsen-everything rows came out of the pre-fix ``None`` stamp with a MIXTURE
#: of ``[]`` and ``[3]`` over the store's 12 splat-holding groups, and come out
#: of the explicit stamp uniformly ``[]``. A half-integer grid is invisible to
#: auto-detect and serves the row whose expected barrier is non-empty.
_LEVELS_COARSEN: List[tuple[str, Sequence[str], float, Any, List[int]]] = [
    # No flag: the reduction blends over every axis, so no axis is a barrier.
    ("levels-coarsens-everything", (), 1000.0, [0, 1, 2, 3], []),
    # The same reduction spelled out. `_normalise_coarsen_dims` collapses a
    # request naming every dim to the same internal `None`, so this is the
    # second spelling of the same bug and must publish the same stamp.
    (
        "levels-all-dims-spelled-out",
        ("--coarsen-dims", "0,1,2,3"),
        1000.0,
        [0, 1, 2, 3],
        [],
    ),
    # A proper subset was always stamped honestly; the row is the control that
    # the change did not disturb it.
    ("levels-honours-the-request", ("--coarsen-dims", "0,1,2"), 0.5, [0, 1, 2], [3]),
]


@pytest.mark.parametrize(
    ("tail", "time_step", "stamp", "barrier"),
    [(tail, step, stamp, bar) for _, tail, step, stamp, bar in _LEVELS_COARSEN],
    ids=[row[0] for row in _LEVELS_COARSEN],
)
def test_a_levels_build_stamps_the_coarsen_dims_it_used(
    tmp_path: Path,
    tail: Sequence[str],
    time_step: float,
    stamp: Any,
    barrier: List[int],
) -> None:
    """``lod --recipe levels`` owes its output the barrier it actually earned.

    The substitutive builder used to publish a literal ``null`` whenever it
    coarsened every dimension, and the writer cannot tell that from an absent
    key: both mean "no provenance" and fall through to ``detect_barrier_dims``.
    On data whose stacked axis the merge leaves gridded, that guess re-imposes
    the very barrier every level was just blended over (#1600). Asserted on the
    LAYOUT as well as the key — the key alone would not show that a stamp is
    load-bearing.
    """
    src = tmp_path / "flat.gsplats.zarr"
    _data(200, stats=dict(_DESCRIPTIVE), time_step=time_step).save(
        src, include_fitting_info=True
    )
    out = tmp_path / "out.gsplats.zarr"
    _run(
        ("lod", "{in}", "{out}", "--recipe", "levels", "-K", "4", "-L", "2", *tail),
        src,
        out,
    )

    got = _pipeline_attrs(out).get("coarsen_dims")
    assert (list(got) if isinstance(got, list) else got) == stamp
    barriers = _ordering_barriers(out)
    assert barriers, "no ordering attrs written at all"
    assert all(b == barrier for b in barriers), (
        f"the written ordering barrier {barriers} does not follow the stamp"
    )

    # The control that makes the row mean something: strip the stamp and the
    # SAME store gets a different LAYOUT, so the assertion above cannot be
    # satisfied by the fallback it exists to displace. Compared as the set of
    # layouts rather than per group: a `levels` build has one group per level
    # plus its ladder rungs, and auto-detection answers them INDEPENDENTLY — it
    # keeps the barrier on whichever ones the merge left gridded — so the honest
    # claim is that the fallback does not produce the stamped layout throughout.
    auto = _auto_detected_barriers(out, tmp_path)
    assert auto and set(map(tuple, auto)) != {tuple(barrier)}, (
        f"auto-detection produces {auto} on this result, which the expected "
        f"barrier {barrier} cannot be distinguished from — the row would pass "
        "on the fallback rather than on the stamp"
    )


@pytest.mark.parametrize(
    ("recipe", "tail", "expected", "forbidden"),
    [
        ("levels", ("-K", "4", "-L", "2"), "also published", "does not publish"),
        (
            "overview",
            ("--max-elements", "80", "-K", "4"),
            "does not publish",
            "also published",
        ),
        (
            "adaptive",
            ("--max-elements", "80", "-K", "4", "-L", "2"),
            "does not publish",
            "also published",
        ),
    ],
)
def test_the_no_coarsen_dims_warning_tells_each_recipe_its_own_truth(
    tmp_path: Path,
    recipe: str,
    tail: Sequence[str],
    expected: str,
    forbidden: str,
) -> None:
    """>3D and no ``--coarsen-dims``: what the flag would buy differs by recipe.

    All three coarsen, so all three warn — but only ``levels`` publishes the
    resolved choice, so only there does the flag also fix the chunk layout.
    ``overview`` / ``adaptive`` stamp nothing (#1600), and telling their users
    otherwise is a promise the store does not keep.
    """
    src = tmp_path / "flat.gsplats.zarr"
    _data(200, stats=dict(_DESCRIPTIVE)).save(src, include_fitting_info=True)
    out = tmp_path / "out.gsplats.zarr"
    result = CliRunner().invoke(
        app,
        ["gsplat", "lod", str(src), str(out), "--recipe", recipe, *tail],
    )
    assert result.exit_code == 0, result.output
    assert "no --coarsen-dims" in result.output, "the 4D input drew no warning at all"
    assert expected in result.output, (
        f"--recipe {recipe} was not told what its own stamp does:\n{result.output}"
    )
    assert forbidden not in result.output, (
        f"--recipe {recipe} was told another recipe's story:\n{result.output}"
    )


def test_a_levels_build_on_a_fine_grid_gets_one_layout_for_the_whole_ladder(
    tmp_path: Path,
) -> None:
    """The other grid, where the fallback is not wrong so much as INCONSISTENT.

    A step-1 stacked axis is the case the coarsen-everything rows above cannot
    use: merging averages neighbouring timepoints together, the coordinates stop
    being integral, and ``detect_barrier_dims`` finds nothing — on the levels it
    merged. The FINEST level of a substitutive ladder is the input unreduced, so
    its grid survives and the fallback still barriers it. Measured pre-fix
    directly on the three substitutive levels, coarsest→finest: ``[[], [], [3]]``
    — one ladder, two layouts, chosen per group by a heuristic (the ladder rungs
    this store also carries split the same way). The stamp settles all of them.

    Also the guard against "fix the loud grid only": a change that special-cased
    the widely-spaced case would leave this ladder split.
    """
    src = tmp_path / "flat.gsplats.zarr"
    _data(200, stats=dict(_DESCRIPTIVE), time_step=1.0).save(
        src, include_fitting_info=True
    )
    out = tmp_path / "out.gsplats.zarr"
    _run(("lod", "{in}", "{out}", "--recipe", "levels", "-K", "4", "-L", "2"), src, out)

    assert _pipeline_attrs(out).get("coarsen_dims") == [0, 1, 2, 3]
    barriers = _ordering_barriers(out)
    assert len(barriers) >= 3, f"expected one barrier per level, got {barriers}"
    assert all(b == [] for b in barriers), (
        f"the ladder did not get one layout: {barriers}"
    )
    # The control: without the stamp the levels disagree with each other, which
    # is the defect this row is about (a uniform fallback would make the
    # assertion above pass for the wrong reason).
    auto = _auto_detected_barriers(out, tmp_path)
    assert len(set(map(tuple, auto))) > 1, (
        f"auto-detection is uniform ({auto}) on this fixture, so the row proves "
        "nothing about the ladder being split"
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
    — ``gsplat lod``'s own output, and ``_recipe_pipeline_info``, the literal
    stamp site of the ``batch-fit merge`` per-part recipe (the only producer of
    ``per_part`` / ``n_lods`` / ``breakpoints`` / ``levels`` /
    ``additive_ladders``).

    :data:`_MEASURED_RECIPES` is one recipe per distinct ROOT-stamp shape, not one
    per recipe name. ``overview`` and ``adaptive`` are omitted because both were
    measured to stamp exactly what ``tiles`` does — ``recipe`` alone at the root,
    everything else living in a part's or a level's own group — so running them
    would add nothing to the union this returns.
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
    """A LOD builder cannot start stamping a topology key without a classification.

    Closes the registry against the producers in BOTH directions: a new stamp
    that is neither structure-scoped nor explicitly exempt fails the first
    assertion (it would ride through a ``flatten`` as a lie), and a registry key
    no producer writes any more fails the second (dead weight, and a sign the
    stamp was renamed rather than removed).

    "The producers" here means the LOD builders only — what ``gsplat lod``'s
    recipes and ``_recipe_pipeline_info`` write. One root-``pipeline/`` stamper is
    NOT covered: ``merge_tile_results`` (``gsplats/fit_tiled_gsplats.py``) puts
    ``tiled_fitting`` / ``num_tiles`` / ``tile_size`` / ``overlap`` /
    ``volume_shape`` / ``splats_per_tile`` / ``progressive`` in the same group, and
    its per-tile counts ride through a ``flatten`` onto a partless store. Which
    axis those belong on is an open question (``splats_per_tile`` is arguably
    content-scoped — invalidated by a ``decimate`` rather than by a kind change),
    so they are deliberately unregistered and recorded on #1600 instead.
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
