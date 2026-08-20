"""Scene-attribute regression tests for the Gaia Milky Way demo."""

from __future__ import annotations

import shutil
import sys
import zipfile
from pathlib import Path

import numpy as np
import pytest
import zarr

from luxar._zarr_compat import (
    consolidate,
    create_array,
    is_consolidated,
    open_group,
)
from luxar.demos._dependencies import SUBSTITUTIVE_LOD_MODULES, is_installed
from luxar.demos.demo_gaia_milky_way_3m import (
    CACHE_FILE,
    DEMO_META,
    RAW_TABLE_FIELDS,
    RAW_ZARR_NAME,
    _extract_raw_zarr,
    compute_radii,
    load_and_convert_gaia_data,
)
from luxar.demos.registry import DEMO_CACHE_ROOT


def _write_tiny_gaia_table(path: Path, n_stars: int = 16) -> zarr.Group:
    """Write the five raw columns consumed by the demo converter."""
    root = open_group(str(path), mode="w")
    values = {
        "x_kpc": np.linspace(-2.0, 2.0, n_stars, dtype=np.float32),
        "y_kpc": np.linspace(-1.0, 1.0, n_stars, dtype=np.float32),
        "z_kpc": np.linspace(-0.2, 0.2, n_stars, dtype=np.float32),
        "phot_g_mean_mag": np.linspace(2.0, 20.0, n_stars, dtype=np.float32),
        "bp_rp": np.linspace(-0.5, 3.0, n_stars, dtype=np.float32),
    }
    for name, data in values.items():
        create_array(root, name, data=data, shape=data.shape, dtype=data.dtype)
    return root


def _catalog_zip(tmp_path: Path, member_name: str) -> Path:
    """Build a catalog zip whose one top-level directory is ``member_name``.

    Mirrors ``scripts/generate_galaxy_simple.py``: the raw zarr is written next
    to the zip and its files are stored relative to the PARENT, so the zip's
    top-level directory name is exactly the ``--output`` stem.
    """
    raw = tmp_path / member_name
    _write_tiny_gaia_table(raw)
    zip_path = tmp_path / "catalog.zarr.zip"
    with zipfile.ZipFile(zip_path, "w") as zf:
        for f in raw.rglob("*"):
            if f.is_file():
                zf.write(f, f.relative_to(tmp_path))
    return zip_path


def _foreign_table_catalog_zip(tmp_path: Path) -> Path:
    """A correctly-named zip holding a store that is NOT the raw star table.

    What a user who assembles their own Gaia query ends up with: the directory is
    where the demo looks and opens as a zarr group, but the columns are spelled
    differently, so the converter would fail on a bare ``KeyError``.
    """
    raw = tmp_path / RAW_ZARR_NAME
    root = open_group(str(raw), mode="w")
    for name in ("x", "y", "z", "mag", "colour"):
        values = np.linspace(0.0, 1.0, 8, dtype=np.float32)
        create_array(root, name, data=values, shape=values.shape, dtype=values.dtype)
    zip_path = tmp_path / "catalog.zarr.zip"
    with zipfile.ZipFile(zip_path, "w") as zf:
        for f in raw.rglob("*"):
            if f.is_file():
                zf.write(f, f.relative_to(tmp_path))
    return zip_path


def _truncated_catalog_zip(tmp_path: Path) -> Path:
    """A catalog zip cut in half — what an interrupted copy or rebuild leaves.

    Losing the tail loses the end-of-central-directory record, so ``ZipFile``
    rejects the file outright rather than extracting part of it.
    """
    zip_path = _catalog_zip(tmp_path, RAW_ZARR_NAME)
    intact = zip_path.read_bytes()
    zip_path.write_bytes(intact[: len(intact) // 2])
    return zip_path


def _marker_label(node: zarr.Group) -> str:
    """Decode a single-point node's one hover label from its UTF-8 CSR pair."""
    offsets = node["label_offsets"][:]
    return bytes(node["label_bytes"][offsets[0] : offsets[1]]).decode("utf-8")


def test_authored_nodes_keep_gaia_volumetric_appearance(tmp_path: Path) -> None:
    """The built scene pins the demo's volumetric compositing knobs, and where.

    Two claims: the four "Stars" knobs hold their tuned values, and they are set
    on the kind=lod WRAPPER with every level inert under it — which is what makes
    every level of the mixed ladder composite identically. The markers' kappa is
    pinned too, recomputed from the demo's own radius law.
    """
    raw = tmp_path / "gaia.zarr"
    scene_path = tmp_path / "gaia.luxar.zarr"
    _write_tiny_gaia_table(raw)

    assert load_and_convert_gaia_data(raw, scene_path) == 16

    scene = zarr.open(str(scene_path), mode="r")

    # "Stars" carries three knobs that are tuned as ONE set, because they land
    # on the same two shader terms: ray mass = falloff * opacity, optical depth
    # tau = kappa * ray mass, radiance = colour * intensity * ray mass * S(tau).
    # opacity 0.5 and kappa 0.12 keep tau low enough that the disc stays
    # translucent front to back, and intensity 0.175 buys back the emission the
    # lower ray mass gives up (a 0-5.7 display range, i.e. 1/intensity, over the
    # 0-32.3 data range). Changing one alone re-lights the scene, so all three
    # are pinned together — and they are pinned on the kind=lod WRAPPER, which
    # is what makes every level of this mixed ladder (lifted gsplats coarse,
    # Points finest) composite with the same tau and the same gain.
    stars_node = scene["Stars"]
    stars = dict(stars_node.attrs)
    assert stars["blending_mode"] == "volumetric"
    assert stars["opacity"] == pytest.approx(0.5)
    assert stars["absorption"] == pytest.approx(0.12)
    assert stars["intensity"] == pytest.approx(0.175)

    # The markers keep their ORIGINAL authored look, so their kappa is not a
    # free knob: it is the historical 1.3 rescaled through the 2026-08-02
    # ray-mass unification, which dropped tau's world-radius factor (a bare 1.3
    # would now absorb ~3.5x harder). Preserving the authored look is exactly
    # kappa * radius * chord — the value the demo computes as MARKER_ABSORPTION,
    # recomputed here from first principles rather than copied, so a change to
    # either side has to be deliberate.
    #
    # The demo sizes a marker off its OWN radius law at "mid-brightness": the
    # magnitude whose normalized brightness is exactly 0.5, which compute_radii's
    # (21 - mag) / 18 puts at G = 12.0. Derived here instead of restating the
    # expression, so retuning the radius law has to be deliberate too.
    mid_star_radius = float(compute_radii(np.array([12.0], dtype=np.float32))[0])
    marker_radius = mid_star_radius * 10.0 * 10  # x SCALE=10, marker = 10x a star
    expected_marker_kappa = 1.3 * marker_radius * float(np.sqrt(np.pi / np.log(100.0)))
    for name in ("Sun", "Betelgeuse", "Rigel"):
        marker = dict(scene[name].attrs)
        assert marker["blending_mode"] == "volumetric"
        assert marker["opacity"] == pytest.approx(1.0)
        assert marker["absorption"] == pytest.approx(expected_marker_kappa)

    # The "on the WRAPPER" half of the "Stars" claim needs the wrapper to exist:
    # without torch+scipy the demo's substitutive_lod_or_flat() degrades "Stars"
    # to a flat leaf, and the value assertions above pass while saying nothing
    # about levels. Skip rather than fail on such a machine — but only after the
    # marker checks, which do not depend on the ladder.
    missing = [m for m in SUBSTITUTIVE_LOD_MODULES if not is_installed(m)]
    if missing:
        pytest.skip(
            f"substitutive Points LOD needs {list(SUBSTITUTIVE_LOD_MODULES)}; "
            f"{missing} missing, so 'Stars' is a flat leaf here and the "
            "wrapper-vs-level claim is not testable"
        )
    assert stars["kind"] == "lod", (
        "'Stars' must be the kind=lod wrapper — a flat leaf would satisfy the "
        "value assertions above without pinning anything about the ladder"
    )
    # ...and every level must be INERT under it — which is NOT the same as
    # carrying no attrs. The viewer composes opacity/absorption/intensity/gamma
    # MULTIPLICATIVELY root->leaf and offset ADDITIVELY (viewer
    # src/data/attrs-composer.ts), so a level is inert exactly when its value is
    # that operator's identity. The writers stamp exactly those identities onto
    # every node unconditionally (`apply_default_render_attrs`,
    # io/_compiler/node_common.py:450-472, mirrored for gsplats in
    # io/_compiler/gsplat_assembly.py), so absence is not available to assert and
    # would not mean anything if it were.
    #
    # All five are checked, not just the three the demo authors: `gamma` and
    # `offset` are stamped on every level and set on NEITHER the wrapper nor the
    # markers, so a level that acquired a non-identity one would re-light that
    # level alone — the mixed ladder mismatching across an LOD switch, which is
    # the exact failure this test exists to catch.
    #
    # `blending_mode` is the exception, and the reason it gets its own check: it
    # has no identity value, so it is nearest-setter-wins, and the writers
    # deliberately refuse to stamp a default for it. It is therefore the one knob
    # a level could use to override the wrapper for itself alone.
    #
    # Every child is checked, not just the first: the coarse gsplat levels and
    # the finest Points level are written by different writers.
    levels = sorted(stars_node.group_keys())
    assert levels, "a kind=lod wrapper must have level children"
    for level in levels:
        level_attrs = dict(stars_node[level].attrs)
        assert "blending_mode" not in level_attrs, (
            f"level {level!r} sets its own blending_mode; under nearest-setter-wins "
            "that overrides the wrapper's 'volumetric' for this level alone"
        )
        # key -> the identity of the operator the viewer composes it with.
        for key, identity in (
            ("opacity", 1.0),
            ("absorption", 1.0),
            ("intensity", 1.0),
            ("gamma", 1.0),
            ("offset", 0.0),
        ):
            # The identity IS the invariant that matters: at it, the value the
            # renderer composes for this level is the wrapper's, unchanged.
            child_value = level_attrs.get(key, identity)
            assert child_value == pytest.approx(identity), (
                f"level {level!r} has {key}={child_value}, not the composition "
                f"identity {identity} — it would re-light this level alone "
                f"(wrapper {key}={stars.get(key, 'unset')})"
            )


def test_named_star_legend_is_derived_from_the_marker_nodes(tmp_path: Path) -> None:
    """Legend rows and swatches restate the markers, they do not re-invent them.

    The legend duplicates information that also lives on the marker nodes (the
    label text) or is computed from them (the swatch colour), so both are
    asserted against the nodes rather than against literals — a marker recoloured
    or relabelled without touching the legend fails here.
    """
    raw = tmp_path / "gaia.zarr"
    scene_path = tmp_path / "gaia.luxar.zarr"
    _write_tiny_gaia_table(raw)
    load_and_convert_gaia_data(raw, scene_path)

    scene = zarr.open(str(scene_path), mode="r")
    overlays = scene["overlays"]
    html_overlays = [
        dict(overlays[g].attrs)
        for g in overlays.group_keys()
        if dict(overlays[g].attrs).get("type") == "overlay_html"
    ]
    assert len(html_overlays) == 1, "expected exactly one HTML overlay (the legend)"
    legend = html_overlays[0]
    assert legend["anchor"] == "bottom-left"

    # The marker set comes from the STORE, not a literal: a marker node is a
    # scene-level node carrying hover labels ("Stars" carries none). Ladder
    # wrappers are excluded by their `kind`: a kind=lod group can carry a UNION
    # label CSR, so a labelled "Stars" would otherwise be counted as a marker and
    # fail below as a confusing swatch-count mismatch.
    marker_names = sorted(
        g
        for g in scene.group_keys()
        if "label_bytes" in set(scene[g].array_keys())
        and "kind" not in dict(scene[g].attrs)
    )
    for name in marker_names:
        node = scene[name]
        # Same string in the tooltip and in the legend row.
        assert _marker_label(node) in legend["html"]
        # `float(c) * 255` below only holds while colours stay float 0-1 on disk;
        # a uint8 (0-255) encoding would multiply by 255 twice and mismatch by a
        # confusing factor rather than naming the cause.
        assert np.issubdtype(node["colors"].dtype, np.floating), (
            f"{name!r} colours are {node['colors'].dtype} on disk, not float — the "
            "0-1 to 0-255 conversion here no longer applies (encoding changed)"
        )
        r, g, b = (int(round(float(c) * 255)) for c in node["colors"][0])
        assert f"background:rgb({r},{g},{b})" in legend["html"]

    # One swatch per marker, no more and no fewer: a fourth marker left out of
    # the legend, or a legend row for a marker that is gone, both fail here where
    # the per-marker loop above would not notice.
    swatches = legend["html"].count("background:rgb(")
    assert swatches == len(marker_names) > 0, (
        f"legend has {swatches} swatches for {len(marker_names)} marker nodes "
        f"({marker_names})"
    )

    # The labels are only *visible* because the compiler auto-injects a hover
    # overlay when any node carries labels; a suppressed or pre-empted injection
    # would leave them as dead bytes on disk.
    assert "__hover_text" in set(overlays.group_keys())


def test_declared_cache_namespace_is_where_the_catalog_is_read() -> None:
    """The declared cache name and the directory read must stay one directory.

    ``DEMO_META["caches"]`` is what `luxar demo cache …` walks and what attributes
    the directory to this demo; CACHE_FILE is where the demo actually reads it.
    The two spell the same directory in two places, and a drift is silent in both
    directions: `luxar demo` reports this demo as uncached while a real catalog
    sits there, and `cache clear gaia_milky_way` names a directory nothing reads.
    (Deletion is not among the stakes — `registry.PROTECTED_INPUT_DIRS` spares
    the catalog by directory name, independently of this declaration.)
    """
    assert DEMO_META["caches"] == [CACHE_FILE.parent.name]
    # ...and that name resolves under the cache root, which is how
    # `registry.demo_cache_dirs` turns a claim into the directory it clears.
    assert DEMO_CACHE_ROOT / CACHE_FILE.parent.name == CACHE_FILE.parent


class TestRawZarrExtraction:
    """A catalog zip is usable only if it reads, holds, AND contains the table.

    The rebuild script's default ``--output`` is ``galaxy.zarr``, so the wrong
    stem is the easy mistake to make — and it extracts *successfully*, failing
    only later inside ``zarr.open``. A truncated copy is the second way a
    hand-placed file passes ``resolve_data_file``'s file check and is still not a
    catalog, and a store with other column names (a query someone ran themselves)
    is the third. These pin the guards that turn each into an error naming the
    problem and the rebuild command, and the CLI paths that print them.
    """

    def test_the_expected_stem_extracts_to_a_readable_raw_table(
        self, tmp_path: Path
    ) -> None:
        zip_path = _catalog_zip(tmp_path, RAW_ZARR_NAME)
        dest = tmp_path / "unpacked"

        raw_zarr_path = _extract_raw_zarr(zip_path, dest)

        assert raw_zarr_path == dest / RAW_ZARR_NAME
        # Readable by the converter, not merely present: the guard exists to
        # protect that read, so the happy path asserts the read itself.
        assert (
            load_and_convert_gaia_data(raw_zarr_path, tmp_path / "s.luxar.zarr") == 16
        )

    def test_a_wrong_stem_names_the_directory_and_the_rebuild(
        self, tmp_path: Path
    ) -> None:
        zip_path = _catalog_zip(tmp_path, "galaxy.zarr")  # the script's default

        with pytest.raises(FileNotFoundError) as excinfo:
            _extract_raw_zarr(zip_path, tmp_path / "unpacked")

        message = str(excinfo.value)
        assert RAW_ZARR_NAME in message
        # The phrase that only the wrong-stem guard can produce. Naming
        # RAW_ZARR_NAME and the rebuild command is not enough to pin it: an
        # ABSENT extracted path is also a FileNotFoundError, so with the
        # `is_dir()` check gone the store read below fails and its "not a zarr
        # store" advice carries all three of those strings too.
        assert "holds no top-level" in message
        assert "--build-catalog" in message
        assert "--output" in message

    def test_a_foreign_table_names_the_columns_the_demo_reads(
        self, tmp_path: Path
    ) -> None:
        """The right directory name is not the same as the right table.

        A hand-assembled Gaia query lands here: the zip opens, the directory is
        where the demo looks, and the columns are spelled differently — which the
        converter meets as a bare ``KeyError`` on the second read, naming one
        column and nothing about what the demo wanted.
        """
        zip_path = _foreign_table_catalog_zip(tmp_path)

        with pytest.raises(FileNotFoundError) as excinfo:
            _extract_raw_zarr(zip_path, tmp_path / "unpacked")

        message = str(excinfo.value)
        # Every column it needs, not just the first one missing: the reader is
        # holding a table with other names and has to map all five.
        for field in RAW_TABLE_FIELDS:
            assert field in message
        assert "--build-catalog" in message

    def test_a_consolidated_store_missing_a_column_is_still_caught(
        self, tmp_path: Path
    ) -> None:
        """The column check must see the disk, not a stale consolidated index.

        Consolidation has to be arranged here because nothing in the build path
        does it: the shipped zip carries no index and the rebuild script
        never writes one. A hand-placed catalog is exactly where that stops being
        a guarantee, though — its reader may have re-exported or assembled the
        store themselves, index included — and zarr 3 REVERSED zarr 2's default:
        a bare ``zarr.open`` consults `.zmetadata` automatically, so a column
        whose array directory never arrived (a partial copy, a hand-edited store)
        is still reported as present. The guard then passes and the converter
        reads that column as all-zeros fill — for ``bp_rp`` a 3M-star scene with
        every star the same colour, silently.
        ``luxar._zarr_compat.open_group`` exists for exactly this: it passes
        ``use_consolidated=False``.
        """
        removed = "bp_rp"
        raw = tmp_path / RAW_ZARR_NAME
        consolidate(_write_tiny_gaia_table(raw))
        # `is_consolidated`, not a named document: format 2 writes a separate
        # `.zmetadata` while format 3 embeds the index in the root `zarr.json`,
        # and either one is enough to shadow the disk.
        assert is_consolidated(raw), (
            "nothing to test: without consolidated metadata a raw `zarr.open` "
            "would read the disk too and this test could not fail"
        )
        shutil.rmtree(raw / removed)
        zip_path = tmp_path / "catalog.zarr.zip"
        with zipfile.ZipFile(zip_path, "w") as zf:
            for f in raw.rglob("*"):
                if f.is_file():
                    zf.write(f, f.relative_to(tmp_path))

        with pytest.raises(FileNotFoundError) as excinfo:
            _extract_raw_zarr(zip_path, tmp_path / "unpacked")

        message = str(excinfo.value)
        # `removed in message` would not do: the advice lists all five
        # RAW_TABLE_FIELDS as "what this demo reads" whatever is missing, so the
        # assertion has to reach the MISSING list specifically.
        assert f"column(s) {removed}" in message
        assert "--build-catalog" in message

    def test_a_directory_that_is_not_a_zarr_store_names_the_rebuild(
        self, tmp_path: Path
    ) -> None:
        """...and the right directory name is not even necessarily a zarr store.

        The store read answers this one with ``GroupNotFoundError``, which is not
        a ``CatalogUnusable`` — so as invisible to the entry points' handlers as the
        ``KeyError`` above, however it is spelled. The assertions below are on the
        MESSAGE for that reason: ``GroupNotFoundError`` is itself a
        ``FileNotFoundError`` under zarr 3, so the raised type alone cannot tell a
        caught-and-readvised failure from the raw zarr error escaping.
        """
        raw = tmp_path / RAW_ZARR_NAME
        raw.mkdir()
        (raw / "notes.txt").write_text("not a zarr store")
        zip_path = tmp_path / "catalog.zarr.zip"
        with zipfile.ZipFile(zip_path, "w") as zf:
            zf.write(raw / "notes.txt", Path(RAW_ZARR_NAME) / "notes.txt")

        with pytest.raises(FileNotFoundError) as excinfo:
            _extract_raw_zarr(zip_path, tmp_path / "unpacked")

        message = str(excinfo.value)
        assert "not a zarr store" in message
        assert "--build-catalog" in message

    @pytest.mark.parametrize("kind", ["v3-array-store", "corrupt-column-metadata"])
    def test_a_store_zarr_cannot_open_as_this_table_reaches_the_advice(
        self, tmp_path: Path, kind: str
    ) -> None:
        """Two failures that are ``ValueError``, not ``FileNotFoundError``.

        zarr's own errors descend from ``BaseZarrError`` → ``ValueError`` and only
        *some* of them also subclass ``FileNotFoundError``; a corrupt metadata
        document does not even reach zarr's exceptions. So a handler narrowed to
        ``FileNotFoundError`` loses both. Only the first is a failure to OPEN the
        store: the corrupt column opens as a ``Group`` and dies in the membership
        check, which is why the guard wraps both statements. Both cases are
        ordinary for a hand-placed catalog: the single-array
        ``zarr.save(path, arr)`` leaves an ARRAY store at that path on a stock
        zarr 3 — a v3 one, which answers ``ContainsArrayError`` where a v2 array
        answers the ``FileNotFoundError``-flavoured ``GroupNotFoundError`` —
        whereas the multi-column ``zarr.save(path, x_kpc=…, …)`` someone would
        reach for to save a five-column table writes a readable GROUP and gets
        past here; and a truncated per-file copy can leave a column's ``.zarray``
        as invalid JSON, which is a bare ``json.JSONDecodeError``.

        The assertion is on the "not a zarr store" branch specifically: reading
        through ``zarr.open`` instead would raise too, but from the column check —
        an ``Array`` contains none of the five names, so the message would blame
        the columns of a store that is not a table at all.
        """
        raw = tmp_path / RAW_ZARR_NAME
        if kind == "v3-array-store":
            # zarr_format=3 explicitly: the ambient default follows whatever
            # Luxar writes (luxar/conftest.py), and this case is specifically
            # the v3 array store, whose ContainsArrayError is NOT the
            # FileNotFoundError a v2 array store answers with.
            zarr.create_array(
                store=str(raw), shape=(4,), dtype=np.float32, zarr_format=3
            )
        else:
            _write_tiny_gaia_table(raw)
            # Corrupt the document the table's own format actually wrote: a
            # stray `.zarray` beside a format-3 `zarr.json` is ignored, and the
            # store would then open and read perfectly.
            docs = [
                p
                for p in (raw / "bp_rp" / ".zarray", raw / "bp_rp" / "zarr.json")
                if p.exists()
            ]
            assert len(docs) == 1, f"expected one column metadata document, got {docs}"
            docs[0].write_text("{ not json")
        zip_path = tmp_path / "catalog.zarr.zip"
        with zipfile.ZipFile(zip_path, "w") as zf:
            for f in raw.rglob("*"):
                if f.is_file():
                    zf.write(f, f.relative_to(tmp_path))

        with pytest.raises(FileNotFoundError) as excinfo:
            _extract_raw_zarr(zip_path, tmp_path / "unpacked")

        message = str(excinfo.value)
        assert "not a zarr store" in message
        assert "--build-catalog" in message

    def test_the_checked_columns_are_the_ones_the_converter_reads(
        self, tmp_path: Path
    ) -> None:
        """The guard's column list must not drift from the converter's reads.

        Pinned from both sides: the fixture writes exactly ``RAW_TABLE_FIELDS``,
        and the happy-path test above converts that fixture — so a converter that
        starts reading a sixth column fails there rather than passing this guard
        and dying later on real data.
        """
        raw = tmp_path / "gaia.zarr"
        _write_tiny_gaia_table(raw)

        assert set(zarr.open(str(raw), mode="r").array_keys()) == set(RAW_TABLE_FIELDS)

    def test_a_truncated_archive_names_the_rebuild(self, tmp_path: Path) -> None:
        """``exists()`` cannot tell a catalog from half of one — reading it can.

        The catalog is copied or rebuilt by hand, so a partial file is an ordinary
        outcome; without this guard ``zipfile.BadZipFile`` escapes both entry
        points (it is not a ``FileNotFoundError``) as the raw traceback the rest of
        this module exists to avoid.
        """
        zip_path = _truncated_catalog_zip(tmp_path)

        with pytest.raises(FileNotFoundError) as excinfo:
            _extract_raw_zarr(zip_path, tmp_path / "unpacked")

        message = str(excinfo.value)
        assert str(zip_path) in message
        assert "truncated" in message
        assert "--build-catalog" in message

    @pytest.mark.parametrize("extra_argv", [[], ["--no-serve"]], ids=["serve", "build"])
    @pytest.mark.parametrize(
        "broken,expected",
        [
            # Each expectation is a phrase only ONE guard can print, not merely a
            # string the message happens to contain: `RAW_ZARR_NAME` alone would
            # also match the "not a zarr store" advice a missing extracted
            # directory falls through to (see the wrong-stem test above).
            ("wrong-stem", "holds no top-level"),
            ("truncated", "truncated"),
            ("foreign-table", "x_kpc"),
        ],
    )
    def test_both_entry_points_report_it_as_a_cli_error(
        self,
        tmp_path: Path,
        monkeypatch,
        capsys,
        extra_argv: list[str],
        broken: str,
        expected: str,
    ) -> None:
        """Not a traceback: the message is advice, and it has to be readable.

        Both invocations extract, in two different places (``main`` directly for
        ``--no-serve``, ``load_and_convert_from_zip`` when serving), so both are
        checked — a guard raised past one of them would bury the advice. Every
        unusable-catalog kind is checked at both, because each is raised from a
        different point inside ``_extract_raw_zarr``.
        """
        import luxar.demos.demo_gaia_milky_way_3m as demo

        builders = {
            "wrong-stem": lambda p: _catalog_zip(p, "galaxy.zarr"),
            "truncated": _truncated_catalog_zip,
            "foreign-table": _foreign_table_catalog_zip,
        }
        catalog = builders[broken](tmp_path)
        monkeypatch.setattr(demo, "CACHE_FILE", catalog)
        monkeypatch.setattr(demo, "REPO_FILE", tmp_path / "absent.zip")
        monkeypatch.setattr(demo, "get_demos_output_dir", lambda: tmp_path / "out")
        monkeypatch.setattr(sys, "argv", ["demo_gaia_milky_way_3m.py", *extra_argv])

        with pytest.raises(SystemExit) as excinfo:
            demo.main()

        assert excinfo.value.code == 1
        out = capsys.readouterr().out
        # What went wrong, in the reader's terms — the missing directory for a
        # wrong stem, the unreadable file for a partial copy, the columns the demo
        # reads for a table that is not this one.
        assert expected in out
        assert "--build-catalog" in out
        assert "Traceback" not in out

    def test_an_unrelated_missing_file_keeps_its_traceback(
        self, tmp_path: Path, monkeypatch
    ) -> None:
        """The clean exit is for advice only — a bug must not borrow it.

        The serve branch's handler wraps the whole conversion, not just the
        extraction, so a plain ``FileNotFoundError`` raised while building or
        writing the scene would be reported as a catalog problem and exit 1 with
        its traceback swallowed. Catching ``CatalogUnusable`` is what keeps the
        two apart; this fails (as ``SystemExit``) if the handler widens again.
        """
        import luxar.demos.demo_gaia_milky_way_3m as demo

        monkeypatch.setattr(demo, "CACHE_FILE", _catalog_zip(tmp_path, RAW_ZARR_NAME))
        monkeypatch.setattr(demo, "REPO_FILE", tmp_path / "absent.zip")
        monkeypatch.setattr(sys, "argv", ["demo_gaia_milky_way_3m.py"])

        def _boom(*_args: object, **_kwargs: object) -> int:
            raise FileNotFoundError("a bug in the writer, not the catalog")

        monkeypatch.setattr(demo, "load_and_convert_gaia_data", _boom)

        with pytest.raises(FileNotFoundError, match="a bug in the writer"):
            demo.main()


class TestDataFileResolution:
    """The Gaia catalog is CC BY-NC, so it is not shipped with the repository.

    What the demo owes a user without it is a message that names the file, the
    place to put it, why it is absent, the command that rebuilds it, the issue
    that will do that automatically, and the mandatory ESA/DPAC acknowledgement —
    not a ``git lfs pull`` for a file that is no longer in the tree.
    """

    def test_cache_wins_over_the_legacy_in_repo_copy(
        self, tmp_path: Path, monkeypatch
    ) -> None:
        import luxar.demos.demo_gaia_milky_way_3m as demo

        cache = tmp_path / "cache" / "milky_way_gaia_3m.zarr.zip"
        repo = tmp_path / "repo" / "milky_way_gaia_3m.zarr.zip"
        for p in (cache, repo):
            p.parent.mkdir(parents=True, exist_ok=True)
            p.write_bytes(b"stand-in")
        monkeypatch.setattr(demo, "CACHE_FILE", cache)
        monkeypatch.setattr(demo, "REPO_FILE", repo)

        assert demo.resolve_data_file() == cache

    def test_legacy_in_repo_copy_still_works(self, tmp_path: Path, monkeypatch) -> None:
        import luxar.demos.demo_gaia_milky_way_3m as demo

        repo = tmp_path / "repo" / "milky_way_gaia_3m.zarr.zip"
        repo.parent.mkdir(parents=True)
        repo.write_bytes(b"stand-in")
        monkeypatch.setattr(demo, "CACHE_FILE", tmp_path / "absent.zip")
        monkeypatch.setattr(demo, "REPO_FILE", repo)

        assert demo.resolve_data_file() == repo

    def test_a_directory_at_the_catalog_path_is_not_a_catalog(
        self, tmp_path: Path, monkeypatch
    ) -> None:
        """Only a regular file counts — a directory there must reach the advice.

        Giving the rebuild script the ``.zip`` path instead of the load-bearing
        stem writes the raw zarr *directory* under that exact name, and
        ``exists()`` cannot tell the two apart. Resolving it would hand a
        directory to ``zipfile``, whose ``IsADirectoryError`` is an ``OSError``
        rather than a ``FileNotFoundError`` — so neither entry point's handler
        catches it and the whole message below is replaced by a traceback.
        """
        import luxar.demos.demo_gaia_milky_way_3m as demo

        cache = tmp_path / "cache" / "milky_way_gaia_3m.zarr.zip"
        cache.mkdir(parents=True)
        monkeypatch.setattr(demo, "CACHE_FILE", cache)
        monkeypatch.setattr(demo, "REPO_FILE", tmp_path / "repo" / "absent.zip")

        with pytest.raises(FileNotFoundError) as excinfo:
            demo.resolve_data_file()
        assert "--build-catalog" in str(excinfo.value)

    def test_absent_everywhere_explains_why(self, tmp_path: Path, monkeypatch) -> None:
        import luxar.demos.demo_gaia_milky_way_3m as demo

        cache = tmp_path / "cache" / "milky_way_gaia_3m.zarr.zip"
        monkeypatch.setattr(demo, "CACHE_FILE", cache)
        monkeypatch.setattr(demo, "REPO_FILE", tmp_path / "repo" / "absent.zip")
        monkeypatch.setattr(demo.sys, "stdin", None)

        with pytest.raises(FileNotFoundError) as excinfo:
            demo.resolve_data_file()
        message = str(excinfo.value)
        assert str(cache) in message
        # Why it is absent, and the opt-in build that fills the gap.
        assert "NonCommercial" in message
        assert "--build-catalog" in message
        assert "luxar demo deps --install" in message
        # Using Gaia data at all obliges the acknowledgement, so it travels with
        # the instructions rather than only living in the module docstring.
        assert "Gaia Data Processing and Analysis Consortium (DPAC)" in message
        assert "git lfs" not in message.lower()

    def test_interactive_decline_keeps_the_actionable_error(
        self, tmp_path: Path, monkeypatch
    ) -> None:
        import luxar.demos.demo_gaia_milky_way_3m as demo

        class InteractiveInput:
            @staticmethod
            def isatty() -> bool:
                return True

        monkeypatch.setattr(
            demo, "CACHE_FILE", tmp_path / "cache" / "milky_way_gaia_3m.zarr.zip"
        )
        monkeypatch.setattr(demo, "REPO_FILE", tmp_path / "repo" / "absent.zip")
        monkeypatch.setattr(demo.sys, "stdin", InteractiveInput())
        monkeypatch.setattr("builtins.input", lambda _prompt: "no")

        with pytest.raises(demo.CatalogUnusable, match="--build-catalog"):
            demo.resolve_data_file()

    def test_explicit_build_populates_the_cache(
        self, tmp_path: Path, monkeypatch
    ) -> None:
        import luxar.demos._gaia_catalog as builder
        import luxar.demos.demo_gaia_milky_way_3m as demo

        cache = tmp_path / "cache" / "milky_way_gaia_3m.zarr.zip"
        monkeypatch.setattr(demo, "CACHE_FILE", cache)
        monkeypatch.setattr(demo, "REPO_FILE", tmp_path / "repo" / "absent.zip")
        calls: list[tuple[Path, bool]] = []

        def fake_build(*, cache_dir: Path, recompute: bool = False) -> Path:
            calls.append((cache_dir, recompute))
            return cache

        monkeypatch.setattr(builder, "build_catalog", fake_build)

        assert demo.resolve_data_file(build_catalog_requested=True) == cache
        assert calls == [(cache.parent, False)]

    def test_recompute_replaces_an_existing_cache(
        self, tmp_path: Path, monkeypatch
    ) -> None:
        import luxar.demos._gaia_catalog as builder
        import luxar.demos.demo_gaia_milky_way_3m as demo

        cache = tmp_path / "cache" / "milky_way_gaia_3m.zarr.zip"
        cache.parent.mkdir(parents=True)
        cache.write_bytes(b"old")
        monkeypatch.setattr(demo, "CACHE_FILE", cache)
        monkeypatch.setattr(demo, "REPO_FILE", tmp_path / "repo" / "absent.zip")
        calls: list[tuple[Path, bool]] = []

        def fake_build(*, cache_dir: Path, recompute: bool = False) -> Path:
            calls.append((cache_dir, recompute))
            return cache

        monkeypatch.setattr(builder, "build_catalog", fake_build)

        assert demo.resolve_data_file(recompute=True) == cache
        assert calls == [(cache.parent, True)]

    def test_interactive_first_run_offers_the_build(
        self, tmp_path: Path, monkeypatch
    ) -> None:
        import luxar.demos._gaia_catalog as builder
        import luxar.demos.demo_gaia_milky_way_3m as demo

        class InteractiveInput:
            @staticmethod
            def isatty() -> bool:
                return True

        cache = tmp_path / "cache" / "milky_way_gaia_3m.zarr.zip"
        monkeypatch.setattr(demo, "CACHE_FILE", cache)
        monkeypatch.setattr(demo, "REPO_FILE", tmp_path / "repo" / "absent.zip")
        monkeypatch.setattr(demo.sys, "stdin", InteractiveInput())
        monkeypatch.setattr("builtins.input", lambda _prompt: "yes")
        monkeypatch.setattr(builder, "build_catalog", lambda **_kwargs: cache)

        assert demo.resolve_data_file() == cache
