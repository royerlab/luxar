"""Tests for the demo scene-staleness fingerprint (issue #1957).

Demos cache their built ``.luxar.zarr`` and used to reuse it whenever the path
merely EXISTED. That let a scene written by an older version of a demo be served
forever: #1957 was reported against an ocean-currents scene whose missing
streamlines had been fixed three weeks earlier, because nothing ever rebuilt the
stale store on disk.
"""

from __future__ import annotations

from ast import AST
from collections.abc import Iterator
from pathlib import Path

import pytest

import luxar.utils.source_fingerprints as source_fingerprints
from luxar._zarr_compat import consolidate, open_group
from luxar.demos._support.runtime.flags import parse_demo_flags
from luxar.demos._support.runtime.provenance import (
    BUILDER_FINGERPRINT_ATTR,
    _clear_demo_source_fingerprint_caches,
    demo_source_fingerprint,
    scene_is_current,
)
from luxar.utils.source_fingerprints import (
    imported_source_files,
    production_source_fingerprint,
)


def _write_scene(path: Path, fingerprint: str | None, *, finished: bool = True) -> Path:
    """Create a minimal zarr group, optionally stamped with ``fingerprint``."""
    group = open_group(path, mode="w")
    group.attrs["type"] = "scene"
    if fingerprint is not None:
        group.attrs[BUILDER_FINGERPRINT_ATTR] = fingerprint
    if finished:
        consolidate(group)
    return path


@pytest.fixture(autouse=True)
def _reset_demo_source_fingerprint_caches() -> Iterator[None]:
    _clear_demo_source_fingerprint_caches()
    yield
    _clear_demo_source_fingerprint_caches()


# ------------------------------------------------------------------ fingerprint


def test_fingerprint_is_stable_for_unchanged_source(tmp_path: Path) -> None:
    source = tmp_path / "demo_thing.py"
    source.write_text("X = 1\n")
    assert demo_source_fingerprint(source) == demo_source_fingerprint(source)


def test_fingerprint_changes_when_the_source_changes(tmp_path: Path) -> None:
    """Any edit that could change the output must change the fingerprint.

    Including a one-character constant tweak — LINE_OPACITY 0.95 -> 0.77 is
    exactly the kind of change that must reach a user with a cached scene.
    """
    source = tmp_path / "demo_thing.py"
    source.write_text("LINE_OPACITY = 0.95\n")
    before = demo_source_fingerprint(source)
    source.write_text("LINE_OPACITY = 0.77\n")
    assert demo_source_fingerprint(source) != before


def test_fingerprint_changes_when_production_writer_changes(tmp_path: Path) -> None:
    package_root = tmp_path / "luxar"
    source = package_root / "demos/demo_thing.py"
    writer = package_root / "encoding/writer.py"
    source.parent.mkdir(parents=True)
    writer.parent.mkdir(parents=True)
    source.write_text("from luxar.encoding import writer\nLINE_OPACITY = 0.95\n")
    (package_root / "__init__.py").write_text("")
    (writer.parent / "__init__.py").write_text("")
    writer.write_text("ENCODING_VERSION = 1\n")

    before = demo_source_fingerprint(source, package_root=package_root)
    writer.write_text("ENCODING_VERSION = 2\n")

    assert demo_source_fingerprint(source, package_root=package_root) != before


def test_fingerprint_ignores_unrelated_production_source(tmp_path: Path) -> None:
    package_root = tmp_path / "luxar"
    source = package_root / "demos/demo_thing.py"
    writer = package_root / "encoding/writer.py"
    unrelated = package_root / "unrelated.py"
    source.parent.mkdir(parents=True)
    writer.parent.mkdir(parents=True)
    source.write_text("from luxar.encoding import writer\n")
    (package_root / "__init__.py").write_text("")
    (writer.parent / "__init__.py").write_text("")
    writer.write_text("ENCODING_VERSION = 1\n")
    unrelated.write_text("VALUE = 1\n")

    before = demo_source_fingerprint(source, package_root=package_root)
    unrelated.write_text("VALUE = 2\n")

    assert demo_source_fingerprint(source, package_root=package_root) == before


def test_fingerprint_confines_out_of_tree_module_imports(tmp_path: Path) -> None:
    import_root = tmp_path / "site-packages"
    package_root = import_root / "luxar"
    sibling_root = import_root / "thirdparty"
    source = import_root / "user_demo.py"
    sibling_helper = sibling_root / "heavy.py"
    package_root.mkdir(parents=True)
    sibling_root.mkdir()
    (package_root / "__init__.py").write_text("")
    (sibling_root / "__init__.py").write_text("")
    source.write_text("import thirdparty.heavy\n")
    sibling_helper.write_text("VALUE = 1\n")

    before = demo_source_fingerprint(source, package_root=package_root)
    sibling_helper.write_text("VALUE = 2\n")

    assert demo_source_fingerprint(source, package_root=package_root) == before


def test_fingerprint_follows_transitive_relative_demo_imports(tmp_path: Path) -> None:
    package_root = tmp_path / "luxar"
    source = package_root / "demos/demo_thing.py"
    helper = package_root / "demos/_shared.py"
    nested = package_root / "demos/_nested.py"
    source.parent.mkdir(parents=True)
    (package_root / "__init__.py").write_text("")
    (source.parent / "__init__.py").write_text("")
    source.write_text("from . import _shared\n")
    helper.write_text("from ._nested import VALUE\n")
    nested.write_text("VALUE = 1\n")

    before = demo_source_fingerprint(source, package_root=package_root)
    nested.write_text("VALUE = 2\n")

    assert demo_source_fingerprint(source, package_root=package_root) != before


def test_demo_fingerprints_reuse_parsed_shared_imports(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    package_root = tmp_path / "luxar"
    demos_root = package_root / "demos"
    shared = package_root / "shared.py"
    demos_root.mkdir(parents=True)
    (package_root / "__init__.py").write_text("")
    (demos_root / "__init__.py").write_text("")
    first = demos_root / "first.py"
    second = demos_root / "second.py"
    first.write_text("from luxar import shared\nFIRST = 1\n")
    second.write_text("from luxar import shared\nSECOND = 2\n")
    shared_payload = b"SHARED_SENTINEL = 123\n"
    shared.write_bytes(shared_payload)
    real_parse = source_fingerprints.ast.parse
    shared_parse_count = 0

    def count_shared_parse(source: bytes) -> AST:
        nonlocal shared_parse_count
        if source == shared_payload:
            shared_parse_count += 1
        return real_parse(source)

    monkeypatch.setattr(source_fingerprints.ast, "parse", count_shared_parse)

    demo_source_fingerprint(first, package_root=package_root)
    demo_source_fingerprint(second, package_root=package_root)

    assert shared_parse_count == 1


@pytest.mark.parametrize(
    ("demo", "dependencies"),
    [
        ("demo_dmri_tractography.py", {"demos/_cinematic_camera.py"}),
        (
            "demo_ocean_currents_earth.py",
            {"demos/_cinematic_camera.py", "demos/_globe_common.py"},
        ),
        ("demo_global_rivers_earth.py", {"demos/_globe_common.py"}),
        (
            "demo_biodiversity_planetary_scale.py",
            {"demos/_cinematic_camera.py", "demos/_globe_common.py"},
        ),
        ("demo_gsplats_3d_cryoem_virus.py", {"demos/_lod_policy.py"}),
        (
            "demo_particle_collision_animated.py",
            {"demos/demo_particle_collision.py"},
        ),
    ],
)
def test_real_demo_source_graph_includes_shared_scene_writers(
    demo: str, dependencies: set[str]
) -> None:
    package_root = Path(__file__).resolve().parents[2]
    demos_root = package_root / "demos"

    sources = imported_source_files(demos_root / demo, (package_root.parent,))
    source_names = {path.relative_to(package_root).as_posix() for path in sources}

    assert dependencies <= source_names


def test_production_fingerprint_is_cached_across_demos(tmp_path: Path) -> None:
    package_root = tmp_path / "luxar"
    writer = package_root / "encoding/writer.py"
    writer.parent.mkdir(parents=True)
    writer.write_text("ENCODING_VERSION = 1\n")

    production_source_fingerprint.cache_clear()
    first = production_source_fingerprint(package_root)
    second = production_source_fingerprint(package_root)

    assert second == first
    assert production_source_fingerprint.cache_info().hits == 1
    assert production_source_fingerprint.cache_info().misses == 1


def test_production_fingerprint_excludes_test_only_sources(tmp_path: Path) -> None:
    package_root = tmp_path / "luxar"
    writer = package_root / "encoding/writer.py"
    test = package_root / "encoding/tests/test_writer.py"
    conftest = package_root / "conftest.py"
    writer.parent.mkdir(parents=True)
    test.parent.mkdir(parents=True)
    writer.write_text("ENCODING_VERSION = 1\n")
    test.write_text("def test_writer(): pass\n")
    conftest.write_text("PYTEST_ONLY = 1\n")

    production_source_fingerprint.cache_clear()
    before = production_source_fingerprint(package_root)
    test.write_text("def test_writer(): assert False\n")
    conftest.write_text("PYTEST_ONLY = 2\n")
    production_source_fingerprint.cache_clear()

    assert production_source_fingerprint(package_root) == before


def test_missing_production_tree_has_no_fingerprint(tmp_path: Path) -> None:
    production_source_fingerprint.cache_clear()
    assert production_source_fingerprint(tmp_path / "missing") == ""


def test_fingerprint_changes_with_zarr_format_environment(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    package_root = tmp_path / "luxar"
    source = package_root / "demos/demo_thing.py"
    source.parent.mkdir(parents=True)
    source.write_text("LINE_OPACITY = 0.95\n")

    monkeypatch.delenv("LUXAR_ZARR_FORMAT", raising=False)
    before = demo_source_fingerprint(source, package_root=package_root)
    monkeypatch.setenv("LUXAR_ZARR_FORMAT", "2")

    assert demo_source_fingerprint(source, package_root=package_root) != before


def test_fingerprint_of_an_unreadable_source_is_empty(tmp_path: Path) -> None:
    assert demo_source_fingerprint(tmp_path / "nope.py") == ""


# ------------------------------------------------------------------- staleness


def test_missing_scene_is_not_current(tmp_path: Path) -> None:
    assert scene_is_current(tmp_path / "absent.luxar.zarr", "abc123") is False


def test_matching_fingerprint_is_current(tmp_path: Path) -> None:
    scene = _write_scene(tmp_path / "s.luxar.zarr", "abc123")
    assert scene_is_current(scene, "abc123") is True


def test_matching_fingerprint_on_unfinished_scene_is_stale(tmp_path: Path) -> None:
    scene = _write_scene(tmp_path / "s.luxar.zarr", "abc123", finished=False)
    assert scene_is_current(scene, "abc123") is False


def test_differing_fingerprint_is_stale(tmp_path: Path) -> None:
    scene = _write_scene(tmp_path / "s.luxar.zarr", "abc123")
    assert scene_is_current(scene, "def456") is False


def test_unstamped_legacy_scene_is_stale(tmp_path: Path) -> None:
    """A scene from before fingerprinting rebuilds ONCE, then stamps itself.

    This is the #1957 case exactly: the store on disk carried no fingerprint
    because the builder that wrote it predated the mechanism.
    """
    scene = _write_scene(tmp_path / "s.luxar.zarr", None)
    assert scene_is_current(scene, "abc123") is False


def test_recompute_forces_a_rebuild_even_when_current(tmp_path: Path) -> None:
    scene = _write_scene(tmp_path / "s.luxar.zarr", "abc123")
    assert scene_is_current(scene, "abc123", recompute=True) is False


def test_keep_stale_reuses_a_scene_from_an_older_builder(tmp_path: Path) -> None:
    """The escape hatch: reuse an expensive scene the caller knows is fine."""
    scene = _write_scene(tmp_path / "s.luxar.zarr", "abc123")
    assert scene_is_current(scene, "def456", keep_stale=True) is True


def test_keep_stale_does_not_reuse_an_unfinished_scene(tmp_path: Path) -> None:
    """The escape hatch accepts old output, never a partial interrupted write."""
    scene = _write_scene(tmp_path / "s.luxar.zarr", "abc123", finished=False)
    assert scene_is_current(scene, "def456", keep_stale=True) is False


def test_recompute_beats_keep_stale(tmp_path: Path) -> None:
    """An explicit rebuild request wins over an explicit reuse request."""
    scene = _write_scene(tmp_path / "s.luxar.zarr", "abc123")
    assert scene_is_current(scene, "abc123", recompute=True, keep_stale=True) is False


def test_unreadable_source_reuses_rather_than_rebuilding(tmp_path: Path) -> None:
    """An empty fingerprint is no evidence of staleness.

    Rebuilding a large scene on a bad guess is worse than serving the one on
    disk, so the check degrades to a plain existence test.
    """
    scene = _write_scene(tmp_path / "s.luxar.zarr", "abc123")
    assert scene_is_current(scene, "") is True


# ------------------------------------------------------------------------ flag


def test_keep_stale_flag_is_parsed(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr("sys.argv", ["demo.py", "--keep-stale"])
    assert parse_demo_flags()["keep_stale"] is True
    monkeypatch.setattr("sys.argv", ["demo.py"])
    assert parse_demo_flags()["keep_stale"] is False
