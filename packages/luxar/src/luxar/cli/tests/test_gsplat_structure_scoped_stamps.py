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

    # Read the command off the argv, not the row id: `lod` appears under six
    # recipe-suffixed ids. Resolved against the REGISTERED names because a
    # command can be two tokens (`batch-fit merge`).
    def command_of(argv: Sequence[str]) -> str:
        two = " ".join(argv[:2])
        return two if two in registered else argv[0]

    classified = {command_of(argv) for _, argv, _ in _KIND_CHANGING} | set(
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
#: out: its substitutive builder RE-STAMPS ``coarsen_dims`` from ``--coarsen-dims``
#: (to ``None`` when the flag is absent), which is a claim about what that builder
#: stamps rather than about this scrub. ``decimate`` is out for the same reason
#: since #1600 — its ``merge`` family re-stamps the dims it resolved, and
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


#: ``(id, argv tail, stamp, barrier)`` for ``decimate``'s ``coarsen_dims``. The
#: fixture is stamped ``[1, 2, 3]`` (barrier ``[0]``) so a re-stamp is visible
#: BOTH in the key and in the layout: every expected barrier below differs from
#: the inherited one, and the timepoints sit on a half-integer grid so
#: ``detect_barrier_dims`` cannot supply ``[3]`` on its own.
_DECIMATE_COARSEN: List[tuple[str, Sequence[str], Any, List[int]]] = [
    # The request the merge was given, and the barrier IT implies.
    (
        "merge-honours-the-request",
        ("-m", "merge", "--coarsen-dims", "0,1,2"),
        [0, 1, 2],
        [3],
    ),
    # No flag: the merge blends over every axis, so no axis is a barrier — and
    # the inherited one must not be re-imposed on splats it just blended.
    ("merge-coarsens-everything", ("-m", "merge"), None, []),
    # A prefix merges nothing, so the input's stamp is still true of the
    # survivors and stays — the ignored request must NOT be published.
    ("prefix-inherits", ("-m", "prefix", "--coarsen-dims", "0,1,2"), [1, 2, 3], [0]),
]


@pytest.mark.parametrize(
    ("tail", "stamp", "barrier"),
    [(tail, stamp, barrier) for _, tail, stamp, barrier in _DECIMATE_COARSEN],
    ids=[row[0] for row in _DECIMATE_COARSEN],
)
def test_decimate_stamps_the_coarsen_dims_it_used(
    tmp_path: Path, tail: Sequence[str], stamp: Any, barrier: List[int]
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
        time_step=0.5,
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
