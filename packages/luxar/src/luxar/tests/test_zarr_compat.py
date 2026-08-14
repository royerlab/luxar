"""Tests for :mod:`luxar._zarr_compat` — the zarr-format seam.

Luxar runs on zarr-python 3 but WRITES zarr format 2. These tests pin the parts
of that arrangement that would otherwise fail silently: a store that quietly
became format 3, a compressor that quietly became Blosc when RAW was meant, and
a production writer that quietly stopped naming its compressor at all.
"""

from __future__ import annotations

import ast
import json
import zipfile
from pathlib import Path

import numpy as np
import pytest
import zarr
from numcodecs import Blosc

from luxar import _zarr_compat as zc

PROD_ROOT = Path(__file__).resolve().parents[1]


# ---------------------------------------------------------------------------
# The format we write
# ---------------------------------------------------------------------------


def test_zarr_format_is_2() -> None:
    """The declared write-format. Flipping this is a deliberate, reviewed act."""
    assert zc.ZARR_FORMAT == 2


def test_open_group_creates_a_format_2_store(tmp_path: Path) -> None:
    """A created store must be v2 on disk: `.zgroup` present, no `zarr.json`."""
    p = tmp_path / "s.zarr"
    zc.open_group(p, mode="w")
    assert json.loads((p / ".zgroup").read_text())["zarr_format"] == 2
    assert not (p / "zarr.json").exists()


def test_open_group_writes_v2_even_when_the_global_default_is_3(
    tmp_path: Path,
) -> None:
    """The facade must not depend on ``zarr.config``'s ambient default.

    Setting ``default_zarr_format=2`` globally would be an easier fix than
    threading ``zarr_format`` through, but it makes correctness depend on import
    order and on nobody else touching the config. This asserts the facade is
    explicit: with the global default set to 3 — the value zarr 3 ships — the
    store must STILL come out as format 2.
    """
    with zarr.config.set({"default_zarr_format": 3}):
        p = tmp_path / "cfg.zarr"
        zc.open_group(p, mode="w")
        assert (p / ".zgroup").exists(), "wrote a v3 store under a v3 default"
        assert json.loads((p / ".zgroup").read_text())["zarr_format"] == 2


def test_memory_group_and_create_root_group_are_format_2(tmp_path: Path) -> None:
    """The other two group constructors must pin the format too, not just `open_group`."""
    assert zc.memory_group().metadata.zarr_format == 2
    store = zc.open_store(tmp_path / "r.zarr", mode="w")
    assert zc.create_root_group(store).metadata.zarr_format == 2


def test_read_is_version_agnostic(tmp_path: Path) -> None:
    """Being on zarr-python 3 is worth it precisely because v3 becomes readable.

    2.18 could not open a format-3 store at all, which is what blocked consuming
    zarr v3 output from other tools. Both formats must open through the facade.
    """
    for fmt in (2, 3):
        p = tmp_path / f"v{fmt}.zarr"
        g = zarr.create_group(store=str(p), zarr_format=fmt)
        g.attrs["marker"] = fmt
        assert zc.open_group(p, mode="r").attrs["marker"] == fmt


# ---------------------------------------------------------------------------
# The compressor hazard
# ---------------------------------------------------------------------------


def _zarray(group: zarr.Group, path: Path, name: str) -> dict:
    """Read an array's raw v2 `.zarray` document straight off disk.

    Deliberately not via zarr's own API: these tests are about what LANDS on
    disk, and zarr would happily normalise away the difference being asserted.
    """
    return json.loads((path / name / ".zarray").read_text())


def test_compressor_none_stores_raw_not_blosc(tmp_path: Path) -> None:
    """``None`` must mean RAW.

    This is the sharp edge of the zarr-3 API: ``compressors`` defaults to
    ``"auto"``, so a writer that means "no compression" and simply omits the
    argument gets Blosc/lz4 instead. Luxar stores its packed label byte-blobs
    raw, and that must survive.
    """
    p = tmp_path / "raw.zarr"
    g = zc.open_group(p, mode="w")
    zc.create_array(
        g, "blob", data=np.arange(16, dtype=np.uint8), chunks=(16,), compressor=None
    )
    assert _zarray(g, p, "blob")["compressor"] is None


def test_compressor_object_is_used_verbatim(tmp_path: Path) -> None:
    """A numcodecs codec must reach `.zarray` unaltered — this is Luxar's zstd-9 policy."""
    p = tmp_path / "z.zarr"
    g = zc.open_group(p, mode="w")
    zc.create_array(
        g,
        "codes",
        data=np.arange(32, dtype=np.uint16),
        chunks=(16,),
        compressor=Blosc(cname="zstd", clevel=9, shuffle=Blosc.SHUFFLE),
    )
    c = _zarray(g, p, "codes")["compressor"]
    assert (c["id"], c["cname"], c["clevel"], c["shuffle"]) == ("blosc", "zstd", 9, 1)


def test_compressor_auto_matches_zarr2s_implicit_default(tmp_path: Path) -> None:
    """``"auto"`` is the compatibility escape hatch, and it must stay lz4/5.

    Test fixtures that never named a compressor relied on zarr 2's default. If
    zarr's ``"auto"`` ever drifts, those fixtures change bytes — this is where
    that shows up.
    """
    p = tmp_path / "a.zarr"
    g = zc.open_group(p, mode="w")
    zc.create_array(
        g, "x", data=np.arange(8, dtype=np.float32), chunks=(8,), compressor="auto"
    )
    c = _zarray(g, p, "x")["compressor"]
    assert c is not None and c["id"] == "blosc"
    assert (c["cname"], c["clevel"]) == ("lz4", 5)


def test_the_compressor_DEFAULT_is_auto_not_raw(tmp_path: Path) -> None:
    """Omitting ``compressor`` must behave like zarr 2's implicit default.

    Distinct from the ``compressor="auto"`` test above, which passes the value
    explicitly and so cannot notice the DEFAULT changing. Mutation testing found
    that gap: flipping the default to ``None`` left the whole suite green while
    silently turning ~79 test fixtures from Blosc to RAW — a byte change with no
    failing test anywhere.
    """
    p = tmp_path / "default.zarr"
    g = zc.open_group(p, mode="w")
    zc.create_array(g, "d", data=np.arange(8, dtype=np.float32), chunks=(8,))
    c = _zarray(g, p, "d")["compressor"]
    assert c is not None, "the default must not be RAW"
    assert (c["id"], c["cname"], c["clevel"]) == ("blosc", "lz4", 5)


def test_chunks_true_is_translated_and_never_reaches_zarr(tmp_path: Path) -> None:
    """``chunks=True`` must be translated, not forwarded.

    zarr 3 rejects a bool outright, and `ChunkSpec` defaults to ``True``, so a
    broken translation breaks every resizable array. Mutation testing found this
    unpinned in the facade's own suite: the only coverage lived in a different
    test file, via a helper that only tests exercise.
    """
    g = zc.memory_group()
    a = zc.create_array(
        g, "auto", shape=(64, 3), dtype=np.float32, chunks=True, compressor=None
    )
    assert isinstance(a.chunks, tuple) and all(isinstance(c, int) for c in a.chunks)
    # `False` means one chunk spanning the array, and must not become "auto".
    b = zc.create_array(
        g, "whole", shape=(64, 3), dtype=np.float32, chunks=False, compressor=None
    )
    assert b.chunks == (64, 3)
    # An int is a real chunk size and must survive as one — bool subclasses int,
    # so an equality-based check here would conflate `True` with `1`.
    c = zc.create_array(
        g, "sized", shape=(64,), dtype=np.float32, chunks=8, compressor=None
    )
    assert c.chunks == (8,)


def test_a_numcodecs_filter_lands_in_the_v2_filters_field(tmp_path: Path) -> None:
    """``luxar_delta_v1`` rides in the v2 ``.zarray`` ``filters`` list.

    The viewer resolves it by that exact shape (zarrita maps a v2 filter ``id``
    to the codec name ``numcodecs.<id>``), so the field must not migrate into a
    v3-style codec pipeline while we still write format 2.
    """
    from numcodecs import Delta

    p = tmp_path / "f.zarr"
    g = zc.open_group(p, mode="w")
    zc.create_array(
        g,
        "d",
        data=np.arange(16, dtype=np.uint16),
        chunks=(16,),
        compressor=None,
        filters=[Delta(dtype="<u2")],
    )
    filters = _zarray(g, p, "d")["filters"]
    assert filters and filters[0]["id"] == "delta"


# ---------------------------------------------------------------------------
# create_array's zarr-2 argument tolerance
# ---------------------------------------------------------------------------


def test_data_and_shape_together_are_accepted(tmp_path: Path) -> None:
    """zarr 3 raises on ``data=`` + ``shape=``; zarr 2 did not, and callers pass both.

    ``luxar.io._compiler.spatial_ordering`` passes data, shape AND dtype for its
    chunk-bounds arrays.
    """
    g = zc.memory_group()
    data = np.arange(12, dtype=np.float32).reshape(4, 3)
    a = zc.create_array(
        g,
        "cb",
        data=data,
        shape=data.shape,
        dtype=np.float32,
        chunks=(4, 3),
        compressor=None,
    )
    assert a.shape == (4, 3)
    assert np.array_equal(a[:], data)


def test_a_contradictory_shape_is_rejected() -> None:
    """Tolerating `data=` + `shape=` must not extend to tolerating a MISMATCH."""
    g = zc.memory_group()
    with pytest.raises(ValueError, match="contradicts"):
        zc.create_array(
            g, "bad", data=np.zeros((4, 3), np.float32), shape=(9, 9), compressor=None
        )


def test_dtype_wins_over_the_supplied_datas_dtype() -> None:
    """An explicit `dtype=` must still narrow the stored type, as it did in zarr 2.

    zarr 3 derives dtype from `data`, so without the cast a caller passing
    float64 data with `dtype=np.float32` would silently store float64 — doubling
    the array on disk.
    """
    g = zc.memory_group()
    a = zc.create_array(
        g,
        "cast",
        data=np.arange(4, dtype=np.float64),
        dtype=np.float32,
        compressor=None,
    )
    assert a.dtype == np.float32


def test_needs_data_or_shape() -> None:
    """Neither `data` nor `shape` is a caller bug, not a zero-length array."""
    g = zc.memory_group()
    with pytest.raises(ValueError, match="needs either"):
        zc.create_array(g, "nope", compressor=None)


# ---------------------------------------------------------------------------
# Errors, closing, consolidation
# ---------------------------------------------------------------------------


def test_is_missing_error_covers_group_not_found(tmp_path: Path) -> None:
    """zarr 3 kept ``GroupNotFoundError`` but dropped ``PathNotFoundError``.

    ``GroupNotFoundError`` subclasses ``FileNotFoundError``, which is what makes
    the single-exception replacement of the old two-name tuple correct.
    """
    p = tmp_path / "arr.zarr"
    zarr.create_array(store=str(p), shape=(2,), dtype=np.float32, zarr_format=2)
    with pytest.raises(FileNotFoundError) as gi:
        zarr.open_group(str(p), mode="r")  # a node, but an array not a group
    assert zc.is_missing_error(gi.value)

    with pytest.raises(FileNotFoundError) as missing:
        zarr.open_group(str(tmp_path / "absent.zarr"), mode="r")
    assert zc.is_missing_error(missing.value)

    assert not zc.is_missing_error(ValueError("unrelated"))


def test_close_is_a_no_op_for_a_local_store(tmp_path: Path) -> None:
    """zarr 3's Group has no ``close()``; the facade must not explode."""
    zc.close(zc.open_group(tmp_path / "c.zarr", mode="w"))


def test_consolidate_writes_zmetadata_indexing_arrays(tmp_path: Path) -> None:
    """``.zmetadata`` is load-bearing, not an optimisation.

    The viewer's scene loader builds its whole graph from the store's
    ``contents()`` listing, which comes from this document, and has no
    directory-walking fallback.
    """
    p = tmp_path / "cm.zarr"
    g = zc.open_group(p, mode="w")
    zc.create_array(g, "a", data=np.arange(4, dtype=np.float32), compressor=None)
    zc.consolidate(g)
    meta = json.loads((p / ".zmetadata").read_text())
    assert meta["zarr_consolidated_format"] == 1
    assert "a/.zarray" in meta["metadata"]
    # v2 document names only — a v3 consolidation would key on `zarr.json`.
    suffixes = {k.rsplit("/", 1)[-1] for k in meta["metadata"]}
    assert suffixes <= {".zgroup", ".zattrs", ".zarray"}, suffixes


def test_append_to_an_existing_v3_store_does_not_shadow_it(tmp_path: Path) -> None:
    """``mode="a"`` must NOT pin the format against a store that already exists.

    ``"a"`` is create-or-open. Pinning `zarr_format=2` unconditionally does not
    fail on a v3 store — it writes a SECOND, v2 root beside the v3 one, leaving
    `zarr.json` and `.zgroup` side by side. Everything written afterwards lands in
    the v2 view while an auto-detecting reader resolves the v3 one, so the write
    succeeds and readers cannot see it. Worse than an error, and reachable now
    that Luxar can be pointed at foreign v3 stores at all —
    `denoise_workers.py` opens its output with `mode="a"`.
    """
    p = tmp_path / "foreign_v3.zarr"
    zarr.create_group(store=str(p), zarr_format=3).attrs["origin"] = "other-tool"

    g = zc.open_group(p, mode="a")
    zc.create_array(g, "added", data=np.arange(3, dtype=np.float32), compressor=None)

    assert not (p / ".zgroup").exists(), "wrote a v2 shadow root beside the v3 one"
    reader = zarr.open_group(str(p), mode="r")  # auto-detect, as a consumer would
    assert reader.metadata.zarr_format == 3
    assert "added" in reader, "the append is invisible to an auto-detecting reader"
    assert reader.attrs["origin"] == "other-tool", "clobbered the foreign attrs"


def test_append_still_creates_format_2_when_there_is_nothing_there(
    tmp_path: Path,
) -> None:
    """The other half of the `mode="a"` rule: creating still pins the format.

    Guards against "fixing" the shadowing bug by dropping `zarr_format` from `"a"`
    altogether, which would silently start emitting format-3 stores whenever a
    writer used create-or-open. Includes the pre-made-empty-directory case, since
    a caller making its output dir first must not change the format.
    """
    fresh = tmp_path / "fresh.zarr"
    zc.open_group(fresh, mode="a")
    assert json.loads((fresh / ".zgroup").read_text())["zarr_format"] == 2

    premade = tmp_path / "premade.zarr"
    premade.mkdir()
    zc.open_group(premade, mode="a")
    assert json.loads((premade / ".zgroup").read_text())["zarr_format"] == 2


@pytest.mark.parametrize("mode", ["w", "a"])
def test_creating_a_fresh_zipped_store_still_pins_the_format(
    tmp_path: Path, mode: str
) -> None:
    """A brand-new ``.zarr.zip`` must come out format 2 in both creating modes.

    This guards an ORDERING dependency inside :func:`open_group`: it constructs the
    store before consulting :func:`_metadata_docs_exist`, and for a zip that helper
    treats "the file exists" as "the store exists". It works today only because
    zarr's ``ZipStore`` is LAZY — constructing one does not create the archive — so
    the helper still sees nothing and the format is pinned. If a future zarr made
    ZipStore eager, the pin would be skipped and a fresh archive would silently
    come out format 3. That is not a hypothesis worth leaving unguarded.
    """
    p = tmp_path / f"fresh_{mode}.zarr.zip"
    g = zc.open_group(p, mode=mode)
    zc.create_array(g, "a", data=np.arange(4, dtype=np.float32), compressor=None)
    zc.close(g)  # ZipStore must be closed to flush the archive

    with zipfile.ZipFile(p) as zf:
        names = zf.namelist()
    assert any(n.endswith(".zgroup") for n in names), (
        f"fresh zipped store is not format 2; archive holds {names[:5]}"
    )
    assert not any(n.endswith("zarr.json") for n in names)


def test_open_store_honours_mode(tmp_path: Path) -> None:
    """A declared read must yield a READ-ONLY store, not merely a readable one.

    Handing back a writable store for `mode="r"` makes the argument a decoration:
    a read path that acquired a write by accident would not be caught anywhere.
    """
    assert zc.open_store(tmp_path / "ro.zarr", mode="r").read_only is True
    assert zc.open_store(tmp_path / "rw.zarr", mode="w").read_only is False
    with pytest.raises(Exception):  # noqa: B017 - zarr's own refusal, type is its business
        zarr.create_group(store=zc.open_store(tmp_path / "refuse.zarr", mode="r"))


def test_reads_do_not_trust_stale_consolidated_metadata(tmp_path: Path) -> None:
    """A deleted array must read as ABSENT, not as present-per-the-stale-index.

    zarr 2 only consulted ``.zmetadata`` through the separate
    ``open_consolidated``, so ``open_group`` always saw what was on disk. zarr 3
    reversed that default. The difference silently disables Luxar's detection of a
    partially written store: the writers have crash-safety machinery precisely
    because half-written stores happen, and the readers' "required array is
    missing" guards are the backstop — they have to see the filesystem.
    """
    import shutil

    p = tmp_path / "partial.zarr"
    g = zc.open_group(p, mode="w")
    zc.create_array(g, "keep", data=np.arange(4, dtype=np.float32), compressor=None)
    zc.create_array(g, "doomed", data=np.arange(4, dtype=np.float32), compressor=None)
    zc.consolidate(g)

    shutil.rmtree(p / "doomed")  # simulate a partial/interrupted write
    assert (p / ".zmetadata").is_file(), "the stale index must still be present"

    reopened = zc.open_group(p, mode="r")
    assert "doomed" not in reopened, "read answered from the stale consolidated index"
    assert sorted(reopened.array_keys()) == ["keep"]

    # And prove the hazard is real rather than hypothetical: zarr's own default
    # still reports the deleted array. If a future zarr changes that, this line
    # fails and the workaround above can go.
    trusting = zarr.open_group(str(p), mode="r")
    assert "doomed" in trusting


def test_open_store_dispatches_zip_by_suffix(tmp_path: Path) -> None:
    """zarr 2 sniffed ``.zip`` inside ``zarr.open``; zarr 3 needs it explicit."""
    assert isinstance(zc.open_store(tmp_path / "d.zarr"), zarr.storage.LocalStore)
    zs = zc.open_store(tmp_path / "z.zarr.zip", mode="w")
    assert isinstance(zs, zarr.storage.ZipStore)
    zs.close()


# ---------------------------------------------------------------------------
# The production-explicitness lint
# ---------------------------------------------------------------------------


def _production_files() -> list[Path]:
    return [
        p
        for p in sorted(PROD_ROOT.rglob("*.py"))
        if "tests" not in p.parts and p.name != "_zarr_compat.py"
    ]


def test_production_create_array_calls_pass_a_compressor() -> None:
    """Every PRODUCTION ``create_array`` must name its compressor.

    ``compressor`` defaults to ``"auto"`` so that test fixtures which never named
    one keep zarr 2's bytes. That default must never be what a real writer gets:
    Luxar's stores carry a measured policy (zstd clevel 9, shuffle by dtype
    width) and some arrays are deliberately RAW, and "auto" is neither. Enforced
    by AST because it is exactly the kind of omission that reads fine in review
    and only shows up as a silent change in the bytes on disk.
    """
    offenders: list[str] = []
    for path in _production_files():
        try:
            tree = ast.parse(path.read_text(encoding="utf-8"))
        except SyntaxError:  # pragma: no cover - defensive
            continue
        for node in ast.walk(tree):
            if not isinstance(node, ast.Call):
                continue
            fn = node.func
            name = fn.id if isinstance(fn, ast.Name) else getattr(fn, "attr", None)
            if name != "create_array":
                continue
            if "compressor" not in {k.arg for k in node.keywords}:
                offenders.append(f"{path.relative_to(PROD_ROOT)}:{node.lineno}")

    assert not offenders, (
        "production create_array() calls with no explicit compressor "
        f"(they would silently get Blosc/lz4/5): {offenders}"
    )


def test_no_zarr_2_create_dataset_calls_remain() -> None:
    """``Group.create_dataset`` was REMOVED in zarr 3 — no call site may survive.

    Scanned as text over the whole repo rather than just this package, because the
    call sites that actually bit were outside it (the viewer's fixture generators)
    and because a branch merged from before this migration reintroduces them
    invisibly: that is exactly how `io/tests/test_volume_lazy.py` arrived mid-PR.
    ``AttributeError`` at runtime is a poor substitute for failing here.
    """
    repo = PROD_ROOT.parents[3]
    offenders: list[str] = []
    for path in sorted(repo.rglob("*.py")):
        parts = set(path.parts)
        if (
            "node_modules" in parts
            or ".venv" in parts
            or "_zarr_compat.py" == path.name
        ):
            continue
        for lineno, line in enumerate(
            path.read_text(encoding="utf-8", errors="replace").splitlines(), 1
        ):
            # `create_resizable_dataset` is Luxar's OWN API and stays.
            if ".create_dataset(" in line and "create_resizable_dataset" not in line:
                offenders.append(f"{path.relative_to(repo)}:{lineno}")

    assert not offenders, (
        "zarr 3 removed Group.create_dataset; use "
        f"luxar._zarr_compat.create_array instead: {offenders}"
    )


#: zarr entry points that CREATE a node, and therefore decide a format. Reading
#: is version-agnostic and needs no pin; creating does. ``consolidate_metadata``
#: is absent on purpose — it follows the format already on the store.
_ZARR_CREATING_CALLS = frozenset(
    {
        "array",
        "create",
        "create_array",
        "create_group",
        "empty",
        "full",
        "group",
        "ones",
        "open",
        "open_array",
        "open_group",
        "save",
        "save_array",
        "zeros",
    }
)


def _is_read_only_call(call: ast.Call) -> bool:
    """Does the call pass a literal ``mode="r"``? Then it creates nothing."""
    return any(
        k.arg == "mode" and isinstance(k.value, ast.Constant) and k.value.value == "r"
        for k in call.keywords
    )


def _zarr_import_names(tree: ast.AST) -> tuple[set[str], set[str]]:
    """``(module names bound to zarr, creating names imported FROM zarr)``.

    Matching only the literal identifier ``zarr`` would let two entirely ordinary
    spellings through — ``import zarr as z`` and ``from zarr import open_group`` —
    so the aliases are resolved instead of assumed. Names imported from
    ``luxar._zarr_compat`` deliberately do NOT count: that module IS the fix.
    """
    modules, direct = {"zarr"}, set()
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            for alias in node.names:
                if alias.name == "zarr" or alias.name.startswith("zarr."):
                    modules.add(alias.asname or alias.name.split(".")[0])
        elif isinstance(node, ast.ImportFrom):
            mod = node.module or ""
            if mod == "zarr" or mod.startswith("zarr."):
                for alias in node.names:
                    if alias.name in _ZARR_CREATING_CALLS:
                        direct.add(alias.asname or alias.name)
    return modules, direct


def _root_name(node: ast.expr) -> str | None:
    """Leftmost identifier of an attribute chain — ``zarr`` in ``zarr.api.open``."""
    while isinstance(node, ast.Attribute):
        node = node.value
    return node.id if isinstance(node, ast.Name) else None


def _format_unpinned_zarr_writes(source: str) -> list[int]:
    """Line numbers of zarr node-creating calls that pin no format."""
    try:
        tree = ast.parse(source)
    except SyntaxError:  # pragma: no cover - defensive
        return []
    modules, direct = _zarr_import_names(tree)
    hits = []
    for node in ast.walk(tree):
        if not isinstance(node, ast.Call):
            continue
        func = node.func
        if isinstance(func, ast.Attribute):
            # Resolve the chain root so `zarr.api.synchronous.open` is caught too.
            if _root_name(func) not in modules or func.attr not in _ZARR_CREATING_CALLS:
                continue
        elif isinstance(func, ast.Name):
            if func.id not in direct:
                continue
        else:
            continue
        if _is_read_only_call(node):
            continue
        if "zarr_format" not in {k.arg for k in node.keywords}:
            hits.append(node.lineno)
    return hits


def test_no_writer_creates_a_store_without_pinning_the_format() -> None:
    """A writer that skips the facade silently emits format 3.

    zarr 3's default is format 3, and the default applies at the node that gets
    CREATED — so a bare ``zarr.open(mode="a")`` root makes every array under it
    v3 no matter what :func:`create_array` is told, and a bare ``zarr.save``
    writes a ``zarr.json`` + ``c/`` array where the rest of Luxar writes
    ``.zarray`` + dot-separated chunks. Neither raises; the store is simply the
    wrong format, which is why this is a lint and not a runtime check.

    Scope is every writer that ships or produces a consumed artifact: the
    package, the repo scripts, the viewer's fixture generators, AND the example
    scripts — those write the `datasets/examples/` scenes that `make run-examples`
    produces and the E2E suite loads, so a v3 store there would surface as a
    baffling viewer failure rather than as a format complaint. Go through
    :mod:`luxar._zarr_compat`, or pass ``zarr_format=ZARR_FORMAT`` explicitly.
    """
    repo_root = PROD_ROOT.parents[3]
    extra_dirs = [
        repo_root / "scripts",
        repo_root / "stats",
        repo_root / "packages" / "luxar" / "examples",
        repo_root / "packages" / "luxar-viewer" / "tests" / "fixtures",
    ]
    files = list(_production_files())
    for d in extra_dirs:
        if d.is_dir():  # absent when the package is tested outside the repo
            files.extend(sorted(d.rglob("*.py")))

    offenders = [
        f"{path.relative_to(repo_root)}:{lineno}"
        for path in files
        for lineno in _format_unpinned_zarr_writes(path.read_text(encoding="utf-8"))
    ]
    assert not offenders, (
        "zarr calls that create a node without pinning the format (they would "
        f"silently write format 3): {offenders}"
    )


def test_the_format_lint_can_actually_fail() -> None:
    """Guard the guard: the walk must flag the two real shapes and spare the rest.

    Lines 1-2 are the exact regressions this lint exists for (a bare ``save``,
    and the creating ``mode="a"`` open whose children inherit the format). The
    rest must stay quiet, or the lint would be unusable noise on read paths.
    """
    assert _format_unpinned_zarr_writes(
        "zarr.save(p, a)\n"
        "zarr.open(p, mode='a')\n"
        "zarr.open(p, mode='r')\n"
        "zarr.save(p, a, zarr_format=2)\n"
        "zarr.consolidate_metadata(s)\n"
        "zarr.storage.ZipStore(p, mode='w')\n"
    ) == [1, 2]


def test_the_format_lint_resists_the_obvious_evasions() -> None:
    """Aliasing or from-importing zarr must not slip a creating call past the lint.

    ``import zarr as z`` and ``from zarr import open_group`` are not adversarial
    tricks, they are two ordinary spellings — and matching the literal identifier
    ``zarr`` misses both. Nothing in the repo writes them today, which is exactly
    why this is pinned now rather than after one appears.
    """
    # Aliased module, and a nested module path.
    assert _format_unpinned_zarr_writes("import zarr as z\nz.open(p, mode='w')\n") == [
        2
    ]
    assert _format_unpinned_zarr_writes(
        "import zarr\nzarr.api.synchronous.open(p, mode='w')\n"
    ) == [2]
    # From-imported creating name, bare and aliased.
    assert _format_unpinned_zarr_writes(
        "from zarr import open_group\nopen_group(p, mode='w')\n"
    ) == [2]
    assert _format_unpinned_zarr_writes(
        "from zarr import create_group as cg\ncg(store=s)\n"
    ) == [2]
    # ...and the same spellings stay quiet when they DO pin, or only read.
    assert (
        _format_unpinned_zarr_writes(
            "import zarr as z\nz.open(p, mode='w', zarr_format=2)\nz.open(p, mode='r')\n"
        )
        == []
    )
    # The facade's own names must never be flagged — that module is the fix, and
    # `luxar._zarr_compat` must not be mistaken for the `zarr` package.
    assert (
        _format_unpinned_zarr_writes(
            "from luxar._zarr_compat import open_group, create_root_group\n"
            "open_group(p, mode='w')\n"
            "create_root_group(store)\n"
        )
        == []
    )


def test_the_lint_can_actually_fail(tmp_path: Path) -> None:
    """Guard the guard: prove the AST walk detects a missing compressor.

    Without this, a refactor that broke the visitor (e.g. stopped matching
    ``ast.Name`` callees) would leave the lint above passing vacuously forever.
    """
    src = "create_array(g, 'x', data=d)\ncreate_array(g, 'y', compressor=None)\n"
    tree = ast.parse(src)
    missing = [
        n.lineno
        for n in ast.walk(tree)
        if isinstance(n, ast.Call)
        and getattr(n.func, "id", None) == "create_array"
        and "compressor" not in {k.arg for k in n.keywords}
    ]
    assert missing == [1]
