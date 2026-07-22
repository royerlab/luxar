"""Enforcement + registry tests for DEMO_META.

Every demo script must carry a valid DEMO_META literal — these tests are the
single durable gate (the block generator that seeded them was a one-off).
"""

from __future__ import annotations

import json
import time
from pathlib import Path

import pytest

from luxar.demos import registry
from luxar.demos.registry import (
    DemoMetaError,
    extract_demo_meta,
    get_demo,
    inventory_caches,
    iter_demos,
)

DEMO_PATHS = sorted(registry._DEMOS_DIR.glob("demo_*.py"))
MANIFEST_PATH = registry._DEMOS_DIR.parents[4] / "scripts" / "gallery" / "manifest.json"


def test_demos_exist() -> None:
    assert len(DEMO_PATHS) >= 70, "demo files disappeared?"


@pytest.mark.parametrize("path", DEMO_PATHS, ids=lambda p: p.name)
def test_every_demo_has_valid_meta(path: Path) -> None:
    """Schema-validates DEMO_META in every demo file (AST, no import)."""
    meta = extract_demo_meta(path)  # raises DemoMetaError on any violation
    assert meta["key"]


@pytest.mark.parametrize("path", DEMO_PATHS, ids=lambda p: p.name)
def test_every_demo_compiles(path: Path) -> None:
    """The file must compile to bytecode — not merely ast.parse.

    Inserting DEMO_META ahead of a ``from __future__`` import parses fine but
    is a hard SyntaxError at compile/import time; compiling catches it.
    """
    compile(path.read_text(encoding="utf-8"), str(path), "exec")


def test_keys_unique_and_resolvable() -> None:
    demos = iter_demos(refresh=True)
    assert len(demos) == len(DEMO_PATHS)
    keys = [d.key for d in demos]
    assert len(set(keys)) == len(keys)
    for demo in demos:
        assert get_demo(demo.key) == demo
        assert get_demo(str(demo.index)) == demo


def test_get_demo_suggests_close_matches() -> None:
    with pytest.raises(KeyError, match="unknown demo"):
        get_demo("lorentz-attractor-oops")
    with pytest.raises(KeyError, match="out of range"):
        get_demo("99999")


def test_iter_demos_is_fast() -> None:
    """The table must render quickly — AST extraction, no module imports."""
    iter_demos(refresh=True)  # warm the memo? no — refresh drops it; time cold:
    start = time.monotonic()
    iter_demos(refresh=True)
    cold = time.monotonic() - start
    assert cold < 1.0, f"cold iter_demos took {cold:.2f}s (budget 1.0s)"


def test_cache_root_matches_utils_demos() -> None:
    """registry.DEMO_CACHE_ROOT duplicates utils.demos' constant (import-weight);
    they must never diverge."""
    from luxar.utils.demos import _DEFAULT_CACHE_ROOT

    assert registry.DEMO_CACHE_ROOT == _DEFAULT_CACHE_ROOT


@pytest.mark.skipif(not MANIFEST_PATH.exists(), reason="gallery manifest not in tree")
def test_meta_cross_validates_against_gallery_manifest() -> None:
    """For every gallery entry with a script: key == id, geometry/category
    match, and the manifest dataset stem is among the demo's outputs."""
    manifest = json.loads(MANIFEST_PATH.read_text())["demos"]
    by_path = {d.path.name: d for d in iter_demos()}

    problems: list[str] = []
    for entry in manifest:
        script = entry.get("script")
        if not script:
            continue
        demo = by_path.get(script)
        if demo is None:
            problems.append(f"{script}: in manifest but not on disk")
            continue
        if demo.key != entry["id"]:
            problems.append(
                f"{script}: key {demo.key!r} != manifest id {entry['id']!r}"
            )
        if demo.geometry != entry["geometry"]:
            problems.append(
                f"{script}: geometry {demo.geometry!r} != manifest {entry['geometry']!r}"
            )
        if demo.category != entry["category"]:
            problems.append(
                f"{script}: category {demo.category!r} != manifest {entry['category']!r}"
            )
        stem = Path(entry["dataset"]).name.removesuffix(".luxar.zarr")
        if stem not in demo.outputs:
            problems.append(
                f"{script}: manifest dataset stem {stem!r} not in outputs {demo.outputs}"
            )
    assert not problems, "\n".join(problems)


def test_validate_meta_rejects_bad_blocks(tmp_path: Path) -> None:
    good = {
        "key": "x-demo",
        "title": "X",
        "description": "A demo.",
        "category": "synthetic",
        "geometry": "points",
        "requirements": {
            "download_mb": 0,
            "compute": "light",
            "gpu": "none",
            "local_data": None,
        },
        "caches": [],
        "outputs": ["x"],
    }
    registry.validate_meta(good, tmp_path / "demo_x.py")

    def variant(**changes):
        m = {**good, **changes}
        if "requirements" in changes and isinstance(changes["requirements"], dict):
            m["requirements"] = {**good["requirements"], **changes["requirements"]}
        return m

    bad_cases = [
        variant(key="Bad Key"),
        variant(category="cooking"),
        variant(geometry="voxels"),
        variant(description="two\nlines"),
        variant(requirements={"compute": "instant"}),
        variant(requirements={"gpu": "maybe"}),
        variant(requirements={"download_mb": -1}),
        variant(requirements={"local_data": "usb-stick"}),
        variant(caches=[1]),
        {k: v for k, v in good.items() if k != "outputs"},
    ]
    for bad in bad_cases:
        with pytest.raises(DemoMetaError):
            registry.validate_meta(bad, tmp_path / "demo_x.py")


def test_extract_rejects_missing_and_non_literal(tmp_path: Path) -> None:
    no_meta = tmp_path / "demo_none.py"
    no_meta.write_text('"""Doc."""\nX = 1\n')
    with pytest.raises(DemoMetaError, match="no top-level DEMO_META"):
        extract_demo_meta(no_meta)

    non_literal = tmp_path / "demo_expr.py"
    non_literal.write_text('"""Doc."""\nDEMO_META = {"key": "a" + "b"}\n')
    with pytest.raises(DemoMetaError, match="pure literal"):
        extract_demo_meta(non_literal)


def test_inventory_caches_maps_and_flags_orphans(tmp_path: Path) -> None:
    demos = iter_demos()
    claimed_name = next((d.caches[0] for d in demos if d.caches), None)
    assert claimed_name is not None, "no demo declares a cache namespace?"

    (tmp_path / claimed_name).mkdir()
    (tmp_path / claimed_name / "blob.bin").write_bytes(b"x" * 2048)
    (tmp_path / "totally-orphaned-dir").mkdir()

    entries = {e.path.name: e for e in inventory_caches(cache_root=tmp_path)}
    assert entries[claimed_name].demo_keys, "claimed dir reported as orphan"
    assert entries[claimed_name].size_bytes == 2048
    assert entries["totally-orphaned-dir"].demo_keys == ()


def test_output_and_cache_paths_resolve(tmp_path: Path) -> None:
    demo = get_demo("lorenz")
    outs = registry.demo_output_paths(demo, demos_dir=tmp_path)
    assert outs and outs[0] == tmp_path / "lorenz.luxar.zarr"
    dirs = registry.demo_cache_dirs(demo, cache_root=tmp_path)
    assert all(d.parent == tmp_path for d in dirs)
