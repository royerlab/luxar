"""Tests for :mod:`luxar._zarr_compat` — the zarr-format seam.

Luxar runs on zarr-python 3 but WRITES zarr format 2. These tests pin the parts
of that arrangement that would otherwise fail silently: a store that quietly
became format 3, a compressor that quietly became Blosc when RAW was meant, and
a production writer that quietly stopped naming its compressor at all.
"""

from __future__ import annotations

import ast
import json
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
