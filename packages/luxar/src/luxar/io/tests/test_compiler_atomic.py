"""Atomic (staged + renamed) directory-store builds in ``LuxarZarrCompiler``.

A demo rebuild replaces a scene that a live server may be reading and that the
catalogue checks for its "built" marker, so it must never expose a partial
store and a failed build must leave the previous one alone.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

from luxar import Dimension, Dimensions, LuxarZarrCompiler
from luxar._zarr_compat import read_node_attrs


def _dims() -> Dimensions:
    return Dimensions([Dimension(n, unit="um", display=True) for n in "xyz"])


def _build(path: Path, n: int, **kwargs: object) -> LuxarZarrCompiler:
    with LuxarZarrCompiler(path, **kwargs) as compiler:  # type: ignore[arg-type]
        scene = compiler.create_scene(dimensions=_dims())
        scene.add_points("p", np.random.rand(n, 3).astype("float32"), radii=0.1)
        # While building, nothing has been written at the final path yet.
        if kwargs.get("atomic"):
            assert compiler.store_path != str(path)
            assert compiler.final_store_path == str(path)
    return compiler


def _siblings(path: Path) -> list[str]:
    return sorted(p.name for p in path.parent.iterdir())


def test_atomic_build_publishes_only_on_success(tmp_path: Path) -> None:
    target = tmp_path / "scene.luxar.zarr"
    compiler = _build(target, 50, atomic=True)
    assert compiler.store_path == str(target)
    assert read_node_attrs(target)["type"] == "scene"
    assert _siblings(target) == ["scene.luxar.zarr"]  # no staging left behind


def test_atomic_rebuild_replaces_the_previous_store(tmp_path: Path) -> None:
    target = tmp_path / "scene.luxar.zarr"
    _build(target, 50, atomic=True)
    first = read_node_attrs(target)["content_hash"]
    _build(target, 80, atomic=True)
    assert read_node_attrs(target)["content_hash"] != first
    assert _siblings(target) == ["scene.luxar.zarr"]


def test_a_failed_atomic_build_leaves_the_previous_store_intact(
    tmp_path: Path,
) -> None:
    target = tmp_path / "scene.luxar.zarr"
    _build(target, 50, atomic=True)
    before = read_node_attrs(target)

    with pytest.raises(RuntimeError, match="boom"):
        with LuxarZarrCompiler(target, atomic=True) as compiler:
            compiler.create_scene(dimensions=_dims())
            raise RuntimeError("boom")

    after = read_node_attrs(target)
    assert after == before
    assert "incomplete" not in after
    assert _siblings(target) == ["scene.luxar.zarr"]


def test_non_atomic_default_still_writes_in_place(tmp_path: Path) -> None:
    target = tmp_path / "scene.luxar.zarr"
    with LuxarZarrCompiler(target) as compiler:
        assert compiler.store_path == str(target)
        compiler.create_scene(dimensions=_dims()).add_points(
            "p", np.random.rand(10, 3).astype("float32"), radii=0.1
        )


def test_demo_outputs_are_atomic_by_default(tmp_path: Path, monkeypatch) -> None:
    monkeypatch.setenv("LUXAR_DATASETS_DIR", str(tmp_path / "datasets"))
    from luxar.utils.paths import get_demos_output_dir

    target = get_demos_output_dir() / "lorenz.luxar.zarr"
    with LuxarZarrCompiler(target) as compiler:
        assert compiler.store_path != str(target)  # staging sibling
        compiler.create_scene(dimensions=_dims()).add_points(
            "p", np.random.rand(10, 3).astype("float32"), radii=0.1
        )
    assert read_node_attrs(target)["type"] == "scene"


def test_atomic_refuses_a_symlinked_target(tmp_path: Path) -> None:
    real = tmp_path / "real.luxar.zarr"
    real.mkdir()
    link = tmp_path / "link.luxar.zarr"
    link.symlink_to(real)
    with pytest.raises(ValueError, match="symlink"):
        LuxarZarrCompiler(link, atomic=True)


# ───────────────────────── failure and crash recovery ────────────────────────
def _host() -> str:
    from luxar.io.compiler import _host_tag

    return _host_tag()


def _dead_pid() -> int:
    import subprocess
    import sys

    proc = subprocess.Popen([sys.executable, "-c", "pass"])
    proc.wait()
    return proc.pid


def test_a_failure_inside_finalize_keeps_the_previous_store(
    tmp_path: Path, monkeypatch
) -> None:
    target = tmp_path / "scene.luxar.zarr"
    _build(target, 50, atomic=True)
    before = read_node_attrs(target)

    def boom(self, store):  # noqa: ANN001
        raise RuntimeError("hashing failed")

    monkeypatch.setattr(LuxarZarrCompiler, "_compute_content_hashes", boom)
    with pytest.raises(ValueError, match="hashing failed"):
        _build(target, 80, atomic=True)
    assert read_node_attrs(target) == before
    assert _siblings(target) == ["scene.luxar.zarr"]


@pytest.mark.parametrize("error", [OSError("rename failed"), KeyboardInterrupt()])
def test_a_failure_inside_the_swap_restores_the_previous_store(
    tmp_path: Path, monkeypatch, error: BaseException
) -> None:
    import luxar.io.compiler as compiler_mod

    target = tmp_path / "scene.luxar.zarr"
    _build(target, 50, atomic=True)
    before = read_node_attrs(target)
    real_replace = compiler_mod.os.replace

    def flaky_replace(src, dst):  # noqa: ANN001
        # Moving the old store aside works; renaming the new one in fails.
        # (zarr's own writes also use os.replace, so match the publish exactly.)
        if Path(dst) == target and ".compile-" in Path(src).name:
            raise error
        return real_replace(src, dst)

    monkeypatch.setattr(compiler_mod.os, "replace", flaky_replace)
    with pytest.raises((ValueError, KeyboardInterrupt)):
        _build(target, 80, atomic=True)
    monkeypatch.setattr(compiler_mod.os, "replace", real_replace)
    assert read_node_attrs(target) == before
    assert _siblings(target) == ["scene.luxar.zarr"]


@pytest.mark.parametrize("zarr_format", [2, 3])
def test_published_store_is_consolidated_with_the_same_hash(
    tmp_path: Path, monkeypatch, zarr_format: int
) -> None:
    """The rename must not disturb consolidation or the reproducible digest."""
    import luxar._zarr_compat as zc
    from luxar._zarr_compat import is_consolidated, read_consolidated_attrs

    monkeypatch.setattr(zc, "ZARR_FORMAT", zarr_format)
    stores = {}
    for atomic in (True, False):
        np.random.seed(7)
        target = tmp_path / f"{atomic}.luxar.zarr"
        _build(target, 64, atomic=atomic)
        stores[atomic] = target
    meta_doc = "zarr.json" if zarr_format == 3 else ".zmetadata"
    assert (stores[True] / meta_doc).is_file()
    assert is_consolidated(stores[True])
    hashes = {a: read_node_attrs(p)["content_hash"] for a, p in stores.items()}
    assert hashes[True] == hashes[False]
    # The consolidated index (what the viewer reads) carries the same digest.
    assert read_consolidated_attrs(stores[True])["/"]["content_hash"] == hashes[True]


def test_dead_leftovers_are_swept_and_a_lost_store_restored(tmp_path: Path) -> None:
    from luxar.io.compiler import sweep_atomic_leftovers

    target = tmp_path / "scene.luxar.zarr"
    _build(target, 50, atomic=True)
    before = read_node_attrs(target)
    pid = _dead_pid()
    # A crash between "move old aside" and "rename new in": no final store.
    old = tmp_path / f".scene.luxar.zarr.old-{_host()}-{pid}-abcd1234"
    target.rename(old)
    stale = tmp_path / f".scene.luxar.zarr.compile-{_host()}-{pid}-deadbeef"
    stale.mkdir()
    # A LIVE build (this process) must be left alone.
    import os

    live = tmp_path / f".scene.luxar.zarr.compile-{_host()}-{os.getpid()}-0123abcd"
    live.mkdir()
    other = tmp_path / f".other.luxar.zarr.compile-{_host()}-{pid}-feedface"
    other.mkdir()

    handled = sweep_atomic_leftovers(target)
    assert sorted(p.name for p in handled) == sorted([old.name, stale.name])
    assert read_node_attrs(target) == before
    assert live.exists() and other.exists()
    assert not stale.exists() and not old.exists()


def test_a_new_atomic_build_sweeps_first(tmp_path: Path) -> None:
    target = tmp_path / "scene.luxar.zarr"
    stale = tmp_path / f".scene.luxar.zarr.compile-{_host()}-{_dead_pid()}-deadbeef"
    stale.mkdir()
    _build(target, 20, atomic=True)
    assert _siblings(target) == ["scene.luxar.zarr"]


def test_a_regular_file_at_the_target_is_refused(tmp_path: Path) -> None:
    target = tmp_path / "scene.luxar.zarr"
    target.write_text("not a store")
    with pytest.raises(ValueError, match="regular file"):
        LuxarZarrCompiler(target, atomic=True)
    assert target.read_text() == "not a store"


def _demo_link(tmp_path: Path, monkeypatch, target: Path) -> Path:
    monkeypatch.setenv("LUXAR_DATASETS_DIR", str(tmp_path / "datasets"))
    from luxar.utils.paths import get_demos_output_dir

    link = get_demos_output_dir() / "lorenz.luxar.zarr"
    link.symlink_to(target)
    return link


def test_a_symlink_to_non_store_data_is_refused_and_the_data_kept(
    tmp_path: Path, monkeypatch
) -> None:
    """Following it would REPLACE the target: a thesis is not a rebuildable scene."""
    precious = tmp_path / "my_precious_data"
    precious.mkdir()
    (precious / "thesis.txt").write_text("years of work")
    link = _demo_link(tmp_path, monkeypatch, precious)
    with pytest.raises(ValueError, match="not a Zarr store"):
        LuxarZarrCompiler(link)
    assert (precious / "thesis.txt").read_text() == "years of work"
    assert sorted(p.name for p in precious.iterdir()) == ["thesis.txt"]
    assert link.is_symlink()


@pytest.mark.parametrize("target", [".", "..", "../..", "HOME"])
def test_a_symlink_to_an_ancestor_or_home_is_refused(
    tmp_path: Path, monkeypatch, target: str
) -> None:
    monkeypatch.setenv("LUXAR_DATASETS_DIR", str(tmp_path / "datasets"))
    from luxar.utils.paths import get_demos_output_dir

    demos = get_demos_output_dir()
    (demos / "keep.txt").write_text("x")
    link = demos / "lorenz.luxar.zarr"
    link.symlink_to(Path.home() if target == "HOME" else Path(target))
    with pytest.raises(ValueError, match="refuses"):
        LuxarZarrCompiler(link)
    assert (demos / "keep.txt").exists()


def test_a_symlinked_demo_output_is_built_at_its_target(
    tmp_path: Path, monkeypatch, capsys
) -> None:
    """The link survives and points at the new scene; nothing is refused."""
    monkeypatch.setenv("LUXAR_DATASETS_DIR", str(tmp_path / "datasets"))
    from luxar.utils.paths import get_demos_output_dir

    real = tmp_path / "elsewhere.luxar.zarr"
    real.mkdir()
    link = get_demos_output_dir() / "lorenz.luxar.zarr"
    link.symlink_to(real)
    for _ in range(2):  # a rebuild through the link works too
        with LuxarZarrCompiler(link) as compiler:
            compiler.create_scene(dimensions=_dims())
        assert link.is_symlink() and link.resolve() == real.resolve()
        assert read_node_attrs(link)["type"] == "scene"
    assert "symlink" in capsys.readouterr().out
    assert sorted(p.name for p in tmp_path.iterdir()) == ["datasets", real.name]


def test_a_malformed_variant_env_does_not_break_unrelated_compiles(
    tmp_path: Path, monkeypatch
) -> None:
    monkeypatch.setenv("LUXAR_DEMO_VARIANT", "../escape")
    target = tmp_path / "scene.luxar.zarr"
    _build(target, 10)
    assert read_node_attrs(target)["type"] == "scene"


def test_scene_to_zarr_copies_from_an_atomic_writer(tmp_path: Path) -> None:
    target = tmp_path / "scene.luxar.zarr"
    copy = tmp_path / "copy.luxar.zarr"
    with LuxarZarrCompiler(target, atomic=True) as compiler:
        scene = compiler.create_scene(dimensions=_dims())
        scene.add_points("p", np.random.rand(10, 3).astype("float32"), radii=0.1)
        scene.to_zarr(copy)
    assert read_node_attrs(target)["type"] == "scene"
    assert (
        read_node_attrs(copy)["content_hash"] == read_node_attrs(target)["content_hash"]
    )
    assert _siblings(target) == ["copy.luxar.zarr", "scene.luxar.zarr"]


def test_leftovers_from_another_host_are_never_swept(tmp_path: Path) -> None:
    """On shared storage another machine's PID cannot be judged from here."""
    from luxar.io.compiler import (
        find_atomic_leftovers,
        foreign_atomic_leftovers,
        sweep_atomic_leftovers,
    )

    target = tmp_path / "scene.luxar.zarr"
    pid = _dead_pid()  # dead HERE; maybe very much alive on the other node
    foreign_build = tmp_path / f".scene.luxar.zarr.compile-0badc0de-{pid}-aaaa1111"
    foreign_old = tmp_path / f".scene.luxar.zarr.old-0badc0de-{pid}-bbbb2222"
    foreign_build.mkdir()
    foreign_old.mkdir()
    assert find_atomic_leftovers(target) == []
    assert sweep_atomic_leftovers(target) == []
    assert foreign_build.exists() and foreign_old.exists()
    assert not target.exists()  # the foreign .old- was NOT "restored" mid-swap
    assert sorted(foreign_atomic_leftovers(target)) == sorted(
        [foreign_build, foreign_old]
    )


def test_host_tag_is_parse_safe_for_any_hostname(monkeypatch) -> None:
    import socket

    from luxar.io import compiler as compiler_mod

    for name in ("gpu-sm02-02.czbiohub.org", "a.b-c", "x" * 200, ""):
        monkeypatch.setattr(socket, "gethostname", lambda name=name: name)
        tag = compiler_mod._host_tag()
        path = compiler_mod._atomic_sibling(Path("/d/s.luxar.zarr"), "compile")
        match = compiler_mod._ATOMIC_LEFTOVER_RE.match(path.name)
        assert match is not None and match["host"] == tag, name
        assert match["name"] == "s.luxar.zarr"


def test_a_racing_sweep_never_fails_the_build(tmp_path: Path, monkeypatch) -> None:
    """A leftover another build removed first is not this compile's problem."""
    import shutil as _shutil

    from luxar.io import compiler as compiler_mod

    target = tmp_path / "scene.luxar.zarr"
    stale = tmp_path / f".scene.luxar.zarr.compile-{_host()}-{_dead_pid()}-cafe0001"
    stale.mkdir()

    real_rmtree = _shutil.rmtree

    def raced(path, *a, **k):  # noqa: ANN001, ANN002, ANN003
        if Path(path) == stale:
            real_rmtree(path)  # the other build got there first...
            raise FileNotFoundError(path)
        return real_rmtree(path, *a, **k)

    monkeypatch.setattr(compiler_mod.shutil, "rmtree", raced)
    assert compiler_mod.sweep_atomic_leftovers(target) == []
    monkeypatch.undo()
    _build(target, 10, atomic=True)
    assert read_node_attrs(target)["type"] == "scene"
