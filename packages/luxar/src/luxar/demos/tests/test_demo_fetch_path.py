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

The second half of this module holds the other side of that boundary (#1618):
what a demo may WRITE into the cache. The manifest owns
``~/.cache/luxar/<dataset>/<file>`` and quarantines anything there that fails its
sha256, so a demo's own locally computed stand-in has to live in the separate
``local/`` namespace or it is destroyed and recomputed on every launch.
"""

from __future__ import annotations

import ast
import importlib
import json
import re
from pathlib import Path

import numpy as np
import pytest

from luxar.demos import registry
from luxar.gsplats.gsplat_data import GSplatData

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


@pytest.mark.parametrize("path", DEMO_PATHS, ids=lambda p: p.stem)
def test_manifest_driven_loaders_only_name_hosted_datasets(path: Path) -> None:
    """The other direction, which fails silently rather than loudly.

    A manifest-driven loader returns ``None`` for a ``local-compute`` dataset by
    design, and every caller reads that as "build it yourself" — so pointing one
    at a non-hosted dataset does not raise, it quietly sends the demo into a
    from-scratch GPU refit while the file sits unread in ``demos/data/``.
    """
    ds = _manifest()
    for helper, names in _fetch_calls(path).items():
        if helper not in MANIFEST_DRIVEN:
            continue
        for name in names:
            if name == "<unresolved>":
                continue
            spec = ds.get(name)
            assert spec is not None, (
                f"{path.name}: {helper}({name!r}) names no manifest dataset"
            )
            assert spec["bucket"] == "zenodo", (
                f"{path.name}: {name!r} is a {spec['bucket']!r} dataset but is "
                f"loaded via {helper}, which returns None for anything not "
                f"hosted — the demo would refit from scratch instead of loading "
                f"the in-repo file. Use load_precomputed_gsplats / "
                f"load_precomputed_bundle until its bucket changes."
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


# --------------------------------------------------------------------------- #
# The local-fit namespace (#1618): a demo's own refit must not squat on the path
# the manifest fetch owns.
# --------------------------------------------------------------------------- #
#
# ``ensure_dataset`` resolves a manifest entry to ``~/.cache/luxar/<name>/<file>``
# and treats whatever it finds there as a candidate copy of the HOSTED file: it
# checks the pinned sha256 and QUARANTINES a mismatch. A locally computed
# stand-in (a GPU refit, when the record is unpublished and the git-LFS object was
# never pulled) can never match that hash, so storing one under the hosted name
# guarantees the next launch destroys it and refits — for ever. It belongs under
# ``local_fit_path(name, file)`` = ``<name>/local/<file>``, which the fetch never
# looks at.
#
# Only ``zenodo`` datasets are checked, because only they reach the checksum
# gate: ``ensure_dataset`` raises ``LocalComputeDataset`` for a ``local-compute``
# or ``regenerate`` bucket before touching the cache, and the demos on those
# buckets go through ``load_precomputed_gsplats``, which has no checksum and no
# quarantine. A dataset that is later PROMOTED to ``zenodo`` starts being checked
# here on the same commit that promotes it.

#: Callees a manifest-owned path may legitimately be handed to. Everything else
#: fails, so a new writer is caught by DEFAULT rather than by being remembered.
#: Add to this list only for something that cannot write the path, and say why.
_MANIFEST_PATH_READERS = frozenset(
    {
        "load",  # GSplatData.load / np.load
        "_load_labels",  # ct_totalsegmentator's npz reader
        "_load_colors_f32",  # visible_human_head's npz reader
        "is_lfs_pointer",  # peeks at the first bytes
        # visible_human_head copies its SHIPPED git-LFS pair into the manifest's
        # own cache path. Those bytes ARE the hosted artifact — they match the
        # pinned sha256 — so this is the one write that belongs there; it is
        # exactly what step 2 of `_ensure_one` does itself.
        "atomic_copy_file",
        "aprint",  # logging
        "print",
        "str",
    }
)


def _local_consts(tree: ast.Module) -> dict[str, str | list[str]]:
    """``{name: str}`` and ``{name: [str, ...]}`` for module string constants.

    Both shapes occur in the demos: a single ``GSPLATS_FILE = "x.zip"`` and a
    per-channel ``GSPLATS_FILES = ["x_ch0.zip", "x_ch1.zip"]`` indexed in the fit
    loop. Scope is ignored — a function-local name shadows a module one here —
    which is deliberate: the per-channel demos bind ``cache_file`` in
    ``fit_all_channels`` and write it in ``fit_channel``, and a scope-aware
    analysis would lose that pair.
    """
    out: dict[str, str | list[str]] = {}
    for node in ast.walk(tree):
        if not isinstance(node, ast.Assign):
            continue
        value = node.value
        if isinstance(value, ast.Constant) and isinstance(value.value, str):
            resolved: str | list[str] = value.value
        elif isinstance(value, (ast.List, ast.Tuple)) and all(
            isinstance(e, ast.Constant) and isinstance(e.value, str) for e in value.elts
        ):
            resolved = [e.value for e in value.elts]  # type: ignore[attr-defined]
        else:
            continue
        for target in node.targets:
            if isinstance(target, ast.Name):
                out[target.id] = resolved
    return out


def _div_chain(node: ast.expr) -> list[ast.expr]:
    """Flatten ``a / b / c`` (left-associative) into ``[a, b, c]``."""
    parts: list[ast.expr] = []
    while isinstance(node, ast.BinOp) and isinstance(node.op, ast.Div):
        parts.append(node.right)
        node = node.left
    parts.append(node)
    return list(reversed(parts))


def _as_str(node: ast.expr, consts: dict[str, str | list[str]]) -> str | None:
    if isinstance(node, ast.Constant) and isinstance(node.value, str):
        return node.value
    if isinstance(node, ast.Name):
        value = consts.get(node.id)
        if isinstance(value, str):
            return value
    return None


def _cache_dirs(tree: ast.Module, consts: dict[str, str | list[str]]) -> dict[str, str]:
    """``{name: dataset}`` for every ``Path.home() / ".cache" / "luxar" / <ds>``."""
    out: dict[str, str] = {}
    for node in ast.walk(tree):
        if not isinstance(node, ast.Assign):
            continue
        chain = _div_chain(node.value)
        literals = [_as_str(part, consts) for part in chain[1:]]
        if ".cache" not in literals or "luxar" not in literals:
            continue
        dataset = literals[-1]
        if dataset is None:
            continue
        for target in node.targets:
            if isinstance(target, ast.Name):
                out[target.id] = dataset
    return out


def _basenames(
    node: ast.expr, consts: dict[str, str | list[str]]
) -> tuple[list[str], re.Pattern[str] | None]:
    """Basenames a path-component expression can denote: exact names + a pattern.

    Covers the three shapes the demos actually use — a literal, a module constant
    (single or indexed out of a list), and an f-string built in a per-channel
    loop. The f-string becomes a regex with ``.*`` for each interpolation, so
    ``f"kidney_ch{i}.gsplats.zarr.zip"`` still matches the three pinned names.
    """
    literal = _as_str(node, consts)
    if literal is not None:
        return [literal], None
    if isinstance(node, ast.Subscript) and isinstance(node.value, ast.Name):
        value = consts.get(node.value.id)
        if isinstance(value, list):
            return value, None
    if isinstance(node, ast.JoinedStr):
        pattern = "".join(
            re.escape(str(v.value)) if isinstance(v, ast.Constant) else ".*"
            for v in node.values
        )
        return [], re.compile(pattern + r"\Z")
    return [], None


def _manifest_owned(
    node: ast.expr,
    cache_dirs: dict[str, str],
    consts: dict[str, str | list[str]],
    datasets: dict,
) -> tuple[str, list[str]] | None:
    """``(dataset, pinned names matched)`` if *node* builds a manifest dest."""
    chain = _div_chain(node)
    if len(chain) != 2 or not isinstance(chain[0], ast.Name):
        return None
    dataset = cache_dirs.get(chain[0].id)
    spec = datasets.get(dataset or "")
    if spec is None or spec.get("bucket") != "zenodo":
        return None
    pinned = {f["name"] for f in (spec.get("files") or [])}
    names, pattern = _basenames(chain[1], consts)
    matched = {n for n in names if n in pinned}
    if pattern is not None:
        matched |= {p for p in pinned if pattern.match(p)}
    return (dataset, sorted(matched)) if matched else None  # type: ignore[return-value]


def _searchable(args: list[ast.expr]) -> list[ast.expr]:
    """Every subexpression of a call's arguments, minus f-string interiors.

    A wrapper must not launder the path (``save(str(P))``), so arguments are
    searched as subtrees. An f-string is the exception: a path interpolated into
    one is text — an error message or a log line — and can write nothing.
    """
    out: list[ast.expr] = []
    stack = list(args)
    while stack:
        node = stack.pop()
        if isinstance(node, ast.JoinedStr):
            continue
        out.append(node)
        stack.extend(ast.iter_child_nodes(node))  # type: ignore[arg-type]
    return out


def local_fit_violations(source: str, datasets: dict) -> list[str]:
    """Manifest-owned paths handed to something that is not a known reader.

    Two-step, because construction alone is legal: ``ct_totalsegmentator`` builds
    ``CACHE_LABELS`` to READ the sidecar the fetch brought down.

      1. Bind every name assigned a ``<manifest cache dir> / <pinned file>``
         expression (and note the inline ones).
      2. Report each use of such a name (or expression) as a call argument,
         unless the callee is in :data:`_MANIFEST_PATH_READERS`.

    An argument is searched as a subtree, so a wrapper (``save(str(P))``) does
    not launder the path — except inside an f-string, which is text and never a
    write target, and is skipped so an error message may quote the path.

    What it CANNOT see, honestly: a path laundered through another variable
    (``p = CACHE_FILE; save(p)``), one built from a runtime value that is not a
    module constant, one assembled with ``os.path.join`` or ``.with_name()``,
    and any write that does not go through a call argument. It is a gate against
    the shapes the demos use, not a proof.
    """
    tree = ast.parse(source)
    consts = _local_consts(tree)
    cache_dirs = _cache_dirs(tree, consts)
    if not cache_dirs:
        return []

    owned_names: dict[str, tuple[str, list[str]]] = {}
    for node in ast.walk(tree):
        if not isinstance(node, ast.Assign):
            continue
        owned = _manifest_owned(node.value, cache_dirs, consts, datasets)
        if owned is None:
            continue
        for target in node.targets:
            if isinstance(target, ast.Name):
                owned_names[target.id] = owned

    violations: list[str] = []
    for node in ast.walk(tree):
        if not isinstance(node, ast.Call):
            continue
        func = node.func
        callee = (
            func.attr if isinstance(func, ast.Attribute) else getattr(func, "id", "")
        )
        if callee in _MANIFEST_PATH_READERS:
            continue
        args = list(node.args) + [kw.value for kw in node.keywords]
        for arg in _searchable(args):
            if isinstance(arg, ast.Name) and arg.id in owned_names:
                dataset, matched = owned_names[arg.id]
                where = f"{arg.id} (line {node.lineno})"
            else:
                owned = _manifest_owned(arg, cache_dirs, consts, datasets)
                if owned is None:
                    continue
                dataset, matched = owned
                where = f"an inline path (line {node.lineno})"
            violations.append(
                f"{callee}() is handed {where}, which resolves to "
                f"~/.cache/luxar/{dataset}/{{{', '.join(matched)}}} — the path "
                f"the manifest fetch owns for {dataset!r}"
            )
    return violations


@pytest.mark.parametrize("path", DEMO_PATHS, ids=lambda p: p.stem)
def test_a_local_artifact_never_squats_a_manifest_pinned_path(path: Path) -> None:
    """A demo's own computed artifact must go to ``local_fit_path``, not the fetch's.

    Reverting any one of the eleven demos migrated in #1618 makes this fail:
    each used to pass a ``CACHE_DIR / <pinned name>`` path to ``save``,
    ``save_with_lod`` or an ``_save_*`` sidecar writer, and every one of those
    call shapes is caught — verified by running this analysis over all eleven
    pre-fix sources (see :func:`local_fit_violations` for what is and is not
    visible to a static pass).
    """
    violations = local_fit_violations(path.read_text(), _manifest())
    assert not violations, (
        f"{path.name}: "
        + "; ".join(violations)
        + ". A locally computed stand-in can never match the manifest sha256, so "
        "the next fetch quarantines it and the demo recomputes on EVERY launch "
        "(#1618). Write it to luxar.utils.data_fetch.local_fit_path(<dataset>, "
        "<file>) instead. If the call really only READS the fetched file, add "
        "its name to _MANIFEST_PATH_READERS with a reason."
    )


def test_the_guard_sees_every_path_shape_the_demos_use() -> None:
    """The gate above is only worth its docstring if it FIRES. Prove each shape.

    A synthetic manifest and synthetic sources, one per shape found in the
    migrated demos, so this stays true even after every demo is fixed (a gate
    that can no longer fail on real input proves nothing about itself).
    """
    datasets = {
        "toy_ds": {
            "bucket": "zenodo",
            "files": [{"name": "toy_ch0.zip"}, {"name": "toy_ch1.zip"}],
        },
        "toy_local": {"bucket": "local-compute", "files": [{"name": "toy_ch0.zip"}]},
    }
    header = 'from pathlib import Path\nDS = "toy_ds"\nCACHE_DIR = Path.home() / ".cache" / "luxar" / DS\n'

    # 1. Module-level constant join, written by save_with_lod.
    assert local_fit_violations(
        header + 'F = "toy_ch0.zip"\nOUT = CACHE_DIR / F\nsave_with_lod(fit, OUT)\n',
        datasets,
    )
    # 2. f-string built inside a per-channel loop, written through a parameter.
    assert local_fit_violations(
        header
        + "def fit_one(p):\n    result.save(p)\n"
        + 'def fit_all():\n    for i in range(2):\n        p = CACHE_DIR / f"toy_ch{i}.zip"\n        fit_one(p)\n',
        datasets,
    )
    # 3. A list constant indexed by the loop counter.
    assert local_fit_violations(
        header
        + 'FILES = ["toy_ch0.zip", "toy_ch1.zip"]\n'
        + "for i in range(2):\n    result.save(CACHE_DIR / FILES[i])\n",
        datasets,
    )
    # 4. A non-gsplat sidecar written by a demo-local helper.
    assert local_fit_violations(
        header + 'L = CACHE_DIR / "toy_ch1.zip"\n_save_labels_u8(labels, L)\n',
        datasets,
    )
    # 5. An inline expression, never bound to a name.
    assert local_fit_violations(
        header + 'save_with_lod(fit, CACHE_DIR / "toy_ch0.zip")\n', datasets
    )
    # 6. A wrapper around the path must not launder it.
    assert local_fit_violations(
        header + 'OUT = CACHE_DIR / "toy_ch0.zip"\nsave_with_lod(fit, str(OUT))\n',
        datasets,
    )

    # And the negatives: the fixed shape, a READ of the fetched file, a file the
    # manifest does not pin, and a dataset the checksum gate never touches.
    assert not local_fit_violations(
        header + 'save_with_lod(fit, local_fit_path(DS, "toy_ch0.zip"))\n', datasets
    )
    assert not local_fit_violations(
        header + 'L = CACHE_DIR / "toy_ch1.zip"\nlabels = _load_labels(L)\n', datasets
    )
    assert not local_fit_violations(
        header + 'raw = CACHE_DIR / "source.tif"\nrequests.download(raw)\n', datasets
    )
    # An f-string interior is text — quoting the path in a message writes nothing.
    assert not local_fit_violations(
        header + 'P = CACHE_DIR / "toy_ch0.zip"\nraise RuntimeError(f"{P} is gone")\n',
        datasets,
    )
    assert not local_fit_violations(
        'from pathlib import Path\nCACHE_DIR = Path.home() / ".cache" / "luxar" / "toy_local"\n'
        'save_with_lod(fit, CACHE_DIR / "toy_ch0.zip")\n',
        datasets,
    )


def test_no_shipped_variant_is_named_local() -> None:
    """``<name>/local/`` must stay disjoint from every ``<name>/<variant>/``.

    ``ensure_dataset`` caches a variant's files under ``<name>/<variant>/``, so a
    variant called ``local`` would put the fetch's destinations back inside the
    namespace that exists to be out of its reach. ``local_fit_path`` refuses that
    variant at the call site; this holds the shipped manifest to it too.
    """
    from luxar.utils.data_fetch import LOCAL_FIT_DIRNAME

    for name, spec in _manifest().items():
        assert LOCAL_FIT_DIRNAME not in (spec.get("variants") or {}), (
            f"{name} has a variant named {LOCAL_FIT_DIRNAME!r}, which collides "
            "with the local-fit namespace"
        )


def test_ct_atlas_reaches_the_manifest_on_a_cold_cache(tmp_path, monkeypatch) -> None:
    """A cold cache is exactly when the fetch is needed, so it must not gate it.

    ``ct_totalsegmentator`` is the one migrated demo whose loader sits next to a
    cache-existence check, and hanging the call off that check would make the
    cache -> in-repo -> Zenodo path reachable only for someone who had already
    obtained the data by other means. The dataset lists its labels sidecar as a
    manifest file, so resolving the fit brings the labels down with it.
    """
    demo = importlib.import_module("luxar.demos.demo_gsplats_3d_ct_totalsegmentator")
    labels = tmp_path / "ct_atlas_labels.npz"
    calls: list[tuple] = []

    # The resolved pair is checked for positional alignment before it is
    # returned (#1670), so the stub fit is a real (tiny) GSplatData and the stub
    # sidecar matches its length. Its four splats are COINCIDENT and carry the
    # SAME label, which makes acceptance independent of the guard's `min_pairs`:
    # too few pairs to judge → accept as unverifiable; enough → agreement 1.0.
    # A `min_pairs` change must never redden a test about manifest fetching.
    stub_fit = GSplatData(
        centers=np.zeros((4, 3), dtype=np.float32),
        amplitudes=np.ones(4, dtype=np.float32),
        cholesky_factors=np.tile([1, 0, 1, 0, 0, 1], (4, 1)).astype(np.float32),
    )
    stub_labels = np.full(4, 5, dtype=np.int32)

    def _fake_fetch(*args, **kwargs):
        calls.append(args)
        labels.write_bytes(b"the sidecar rides along")  # what ensure_dataset does
        return [stub_fit]

    monkeypatch.setattr(demo, "RECOMPUTE", False)
    monkeypatch.setattr(demo, "LOCAL_FIT", tmp_path / "absent.gsplats.zarr.zip")
    monkeypatch.setattr(demo, "LOCAL_LABELS", tmp_path / "absent-local.npz")
    monkeypatch.setattr(demo, "CACHE_LABELS", labels)
    monkeypatch.setattr(demo, "LFS_FIT", tmp_path / "absent-lfs.gsplats.zarr.zip")
    monkeypatch.setattr(demo, "LFS_LABELS", tmp_path / "absent-lfs.npz")
    monkeypatch.setattr(demo, "load_dataset_gsplats", _fake_fetch)
    monkeypatch.setattr(demo, "_load_labels", lambda p: stub_labels)

    def _refit_is_a_failure():
        raise AssertionError("fell through to the download-and-refit path")

    monkeypatch.setattr(demo, "load_ct_and_labels", _refit_is_a_failure)

    fit, got_labels = demo.load_or_build()

    assert calls, "the manifest fetch was never reached on a cold cache"
    assert fit is stub_fit
    assert got_labels is stub_labels
