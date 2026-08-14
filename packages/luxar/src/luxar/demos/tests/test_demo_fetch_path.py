"""Every hosted gsplat demo reaches its data through the manifest.

Two ways exist to load a precomputed gsplat dataset, and only one of them
consults the manifest:

  ``load_dataset_gsplats`` / ``load_dataset_bundle``
      resolve through :func:`~luxar.utils.data_fetch.ensure_dataset`: the file is
      checksum-verified against the manifest and the resolution order is cache ->
      in-repo git-LFS -> Zenodo.

  ``load_precomputed_gsplats`` / ``load_precomputed_bundle``
      read ``demos/data/<dir>/`` and the cache only, with an unverified
      ``shutil.copy2`` and no manifest involvement at all.

A demo on the second path cannot reach a Zenodo record however well its dataset
is pinned, and breaks outright once the git-LFS payload leaves the repository.
This gate holds the boundary: a ``zenodo``-bucket dataset must be reached through
the manifest.

``local-compute`` datasets are the deliberate exception, not an oversight. For
those, ``load_dataset_gsplats`` returns None by design, which would send the demo
into a from-scratch GPU refit instead of loading the file sitting right there --
so they keep the in-repo loader until their bucket changes.
"""

from __future__ import annotations

import ast
import json
from pathlib import Path

import pytest

from luxar.demos import registry

DEMO_PATHS = sorted(registry._DEMOS_DIR.glob("demo_*.py"))
MANIFEST = registry._DEMOS_DIR / "data_manifest.json"

LFS_ONLY = {"load_precomputed_gsplats", "load_precomputed_bundle"}
MANIFEST_DRIVEN = {"load_dataset_gsplats", "load_dataset_bundle", "ensure_dataset"}


def _manifest() -> dict:
    return json.loads(MANIFEST.read_text())["datasets"]


def _string_consts(tree: ast.Module) -> dict[str, str]:
    out: dict[str, str] = {}
    for node in ast.walk(tree):
        if isinstance(node, ast.Assign) and isinstance(node.value, ast.Constant):
            if isinstance(node.value.value, str):
                for t in node.targets:
                    if isinstance(t, ast.Name):
                        out[t.id] = node.value.value
    return out


def _fetch_calls(path: Path) -> dict[str, set[str]]:
    """``{helper: {first-argument value, ...}}`` for every fetch helper called.

    Demos pass a module constant rather than a literal, so constants are resolved;
    a regex over the quoted form finds nothing at all.
    """
    tree = ast.parse(path.read_text())
    consts = _string_consts(tree)
    out: dict[str, set[str]] = {}
    for node in ast.walk(tree):
        if not isinstance(node, ast.Call) or not node.args:
            continue
        f = node.func
        name = f.attr if isinstance(f, ast.Attribute) else getattr(f, "id", "")
        if name not in LFS_ONLY | MANIFEST_DRIVEN:
            continue
        a = node.args[0]
        if isinstance(a, ast.Constant) and isinstance(a.value, str):
            val = a.value
        else:
            val = consts.get(getattr(a, "id", ""), "<unresolved>")
        out.setdefault(name, set()).add(val)
    return out


@pytest.mark.parametrize("path", DEMO_PATHS, ids=lambda p: p.stem)
def test_hosted_datasets_are_fetched_through_the_manifest(path: Path) -> None:
    ds = _manifest()
    for helper, names in _fetch_calls(path).items():
        if helper not in LFS_ONLY:
            continue
        for name in names:
            spec = ds.get(name)
            assert spec is not None, (
                f"{path.name}: {helper}({name!r}) names no manifest dataset"
            )
            assert spec["bucket"] != "zenodo", (
                f"{path.name}: {name!r} is a `zenodo` dataset but is loaded via "
                f"{helper}, which never consults the manifest — its checksum is "
                f"not verified and the record can never be reached. Use "
                f"load_dataset_gsplats / load_dataset_bundle instead."
            )


def test_the_exception_list_is_exactly_the_local_compute_datasets() -> None:
    """Spell out who is still on the in-repo loader, so the set cannot grow quietly.

    A new demo added on the old path would otherwise slip in unnoticed as long as
    its dataset happened not to be `zenodo`.
    """
    ds = _manifest()
    still_lfs: dict[str, str] = {}
    for path in DEMO_PATHS:
        for helper, names in _fetch_calls(path).items():
            if helper in LFS_ONLY:
                for name in names:
                    still_lfs[path.stem] = name
    assert still_lfs == {
        "demo_gsplats_3d_acto3d_heart": "gsplats_acto3d_heart",
        "demo_gsplats_3d_tng_cosmic_web": "gsplats_tng_cosmic_web",
        "demo_gsplats_3d_tribolium_embryo": "gsplats_tribolium",
        "demo_gsplats_lod_embryo_line": "gsplats_tribolium",
        "demo_gsplats_lod_tribolium": "gsplats_tribolium",
        "demo_gsplats_recipes_tribolium": "gsplats_tribolium",
    }, "the in-repo-loader set changed; every entry must be a local-compute dataset"
    for name in set(still_lfs.values()):
        assert ds[name]["bucket"] == "local-compute", (
            f"{name} is no longer local-compute — migrate its demos to "
            f"load_dataset_gsplats and drop them from this list"
        )
