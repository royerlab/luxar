"""Every hosted demo dataset reaches its data through the manifest.

Two ways exist to load a precomputed gsplat dataset, and only one of them
consults the manifest:

  ``load_dataset_gsplats`` / ``load_dataset_bundle``
      resolve through :func:`~luxar.demos._support.datasets.data_fetch.ensure_dataset`: the file is
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
HOSTED_DATASET_EXCEPTIONS = {
    # Record unpublished; the module documents its machine-local store until CC BY publishes.
    "gsplats_4d_neuromast_2ch": "documented machine-local store",
    # Resolves the shipped scene zip by hand and is already an analysis blind spot below.
    "desi_galaxies": "shipped scene zip resolved directly",
    # Packaged arrays are still read directly rather than through the manifest.
    "census_umap_1m": "packaged NPZ read directly",
    "3d_umap_coords_human": "packaged Parquet read directly",
    "3d_umap_coords_mouse": "packaged Parquet read directly",
}


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


def test_every_hosted_dataset_is_reached_through_the_manifest() -> None:
    """A hosted artifact must have a real path through the checksum gate.

    Per-demo classification alone misses the third state: a manifest dataset
    whose demo calls neither loader family.  ``verify_cold_fetch.py`` would still
    exercise and count that dataset even though no user takes the verified path.
    Union every variant because those payloads have no top-level ``files`` entry.
    """
    datasets = _manifest()
    reached = {
        name
        for path in DEMO_PATHS
        for helper, names in _fetch_calls(path).items()
        if helper in MANIFEST_DRIVEN
        for name in names
        if name != "<unresolved>"
    }
    hosted = {
        name
        for name, spec in datasets.items()
        if spec.get("bucket") == "zenodo"
        and any(
            files
            for files in (
                spec.get("files", []),
                *(
                    variant.get("files", [])
                    for variant in spec.get("variants", {}).values()
                ),
            )
        )
    }
    assert "h2afva" in hosted, "variant-only datasets were skipped"

    unreached = hosted - reached
    assert unreached == set(HOSTED_DATASET_EXCEPTIONS), (
        "every hosted dataset must be reached by a demo through "
        "load_dataset_gsplats / load_dataset_bundle / ensure_dataset; the "
        "remaining direct readers must stay explicitly justified in "
        f"HOSTED_DATASET_EXCEPTIONS (unreached: {sorted(unreached)})"
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
#: Add to this list only for something that cannot write the path, and say why —
#: and add a case to :func:`test_the_guard_sees_every_path_shape_the_demos_use`
#: that NEEDS the entry, or it is an untested widening of the gate. (Four
#: entries — ``is_lfs_pointer``, ``aprint``, ``print``, ``str`` — were exactly
#: that and have been dropped: nothing reaches them with a manifest-owned path,
#: because an f-string interior is skipped and a wrapped path is still found
#: inside the outer call's argument subtree.)
_MANIFEST_PATH_READERS = frozenset(
    {
        "load",  # GSplatData.load / np.load
        "_load_labels",  # ct_totalsegmentator's npz reader
        "_load_colors_f32",  # visible_human_head's npz reader
    }
)

#: ``Path`` methods that cannot write. A method call on a manifest-owned path
#: whose name is not here is a write (``P.write_bytes(...)``, ``P.unlink()``)
#: and is reported — a hole the first version of this gate had.
#: ``open`` is deliberately absent from both this set and
#: :data:`_MANIFEST_PATH_READERS`: ``P.open("wb")`` and ``open(P, "wb")`` write,
#: so the mode decides and a static pass should not guess. No demo trips it.
_MANIFEST_PATH_READ_METHODS = frozenset(
    {
        "exists",
        "is_file",
        "is_dir",
        "is_symlink",
        "stat",
        "resolve",
        "absolute",
        "as_posix",
        "samefile",
        "read_bytes",
        "read_text",
        # Derive-another-path and enumerate helpers: they return a value, they
        # do not touch the manifest-owned path.
        "joinpath",
        "with_suffix",
        "relative_to",
        "is_relative_to",
        "glob",
        "rglob",
        "iterdir",
        "match",
    }
)

#: Names that denote the shared cache ROOT, ``~/.cache/luxar``. A dataset's
#: cache dir is one of these joined with the dataset name — the spelling
#: ``demo_gsplats_4d_cell_tracking_challenge`` already uses
#: (``registry.DEMO_CACHE_ROOT / DS``), which the literal-chain-only first
#: version of this analysis could not see at all.
_CACHE_ROOT_NAMES = frozenset({"DEMO_CACHE_ROOT", "_DEFAULT_CACHE_ROOT", "CACHE_ROOT"})


def _local_consts(tree: ast.Module) -> dict[str, str | list[str]]:
    """``{name: str}`` and ``{name: [str, ...]}`` for module string constants.

    Both shapes occur in the demos: a single ``GSPLATS_FILE = "x.zip"`` and a
    per-channel ``GSPLATS_FILES = ["x_ch0.zip", "x_ch1.zip"]`` indexed in the fit
    loop. Scope is ignored — a function-local name shadows a module one here —
    which is deliberate: the per-channel demos bind ``cache_file`` in
    ``fit_all_channels`` and write it in ``fit_channel``, and a scope-aware
    analysis would lose that pair.

    It cuts both ways, and the unhelpful direction is a real (unexercised) hole:
    last assignment in SOURCE ORDER wins, so a module-level ``F = "pinned.zip"``
    rebound later by a function-local ``F = "scratch.zip"`` resolves to the
    scratch name everywhere, and a squat spelled with the module constant goes
    clean. No demo shadows a file-name constant; if one ever does, make this
    scope-aware (and keep the cross-function pair above by binding
    function-locals into their enclosing module scope) rather than deleting the
    shadowing.
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
    """Flatten ``a / b / c`` (left-associative) into ``[a, b, c]``.

    ``a.joinpath(b, c)`` is the same join spelled as a method and is flattened
    the same way, so the ``/``-only version's silence on it is gone. The other
    non-``/`` joins (``os.path.join``, ``.with_name()``) are still invisible —
    see :func:`analyse_local_fit`'s honest-limits paragraph.
    """
    parts: list[ast.expr] = []
    while True:
        if isinstance(node, ast.BinOp) and isinstance(node.op, ast.Div):
            parts.append(node.right)
            node = node.left
        elif (
            isinstance(node, ast.Call)
            and isinstance(node.func, ast.Attribute)
            and node.func.attr == "joinpath"
            and node.args
            and not node.keywords
        ):
            parts.extend(reversed(node.args))
            node = node.func.value
        else:
            break
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


def _base_atom(node: ast.expr) -> ast.expr:
    """Strip attribute access and method calls down to the anchoring expression.

    ``Path(__file__).resolve().parent`` → ``Path(__file__)``; ``Path.home()``
    stays itself, since the call IS the anchor there.
    """
    while True:
        if isinstance(node, ast.Attribute):
            node = node.value
        elif isinstance(node, ast.Call) and isinstance(node.func, ast.Attribute):
            inner = node.func.value
            if isinstance(inner, ast.Name):
                return node  # Path.home(), Path.cwd()
            node = inner  # a method in a chain: …resolve(), …absolute()
        else:
            return node


#: What a directory expression is. ``("cache", ds)`` is one the manifest owns —
#: ``<root>/<ds>`` or ``<root>/<ds>/<declared variant>``, both of which
#: ``ensure_dataset`` fetches into and quarantines from; ``("root", None)`` is
#: ``~/.cache/luxar`` itself; ``("other", None)`` is a directory we can PROVE is
#: not the cache (a ``Path(__file__)``-rooted chain, a non-variant subdirectory
#: of a cache dir); ``("unknown", None)`` is one we cannot classify — the state that
#: used to be indistinguishable from "clean", and that any other ``Path(...)``
#: head now correctly lands in.
DirKind = tuple[str, str | None]


def _path_names(tree: ast.Module) -> frozenset[str]:
    """Local names denoting ``pathlib.Path``, aliased imports included.

    Alias handling lets the classifier prove that a ``Path(__file__)``-rooted
    destination is outside the cache rather than leaving it ``unknown``. A name
    comparison against the literal ``"Path"`` is not a spelling this gate gets
    to assume.
    """
    names = {"Path"}
    for node in ast.walk(tree):
        if isinstance(node, ast.ImportFrom) and node.module == "pathlib":
            for alias in node.names:
                if alias.name == "Path":
                    names.add(alias.asname or alias.name)
    return frozenset(names)


class _Ctx:
    """Resolved module facts the directory classifier needs."""

    def __init__(self, tree: ast.Module, datasets: dict) -> None:
        self.datasets = datasets
        self.consts = _local_consts(tree)
        self.path_names = _path_names(tree)
        self.dirs: dict[str, DirKind] = {}
        # Assignments in source order, twice: a name may be used before the pass
        # that classifies it has run (a helper defined above its constants).
        assigns = [n for n in ast.walk(tree) if isinstance(n, ast.Assign)]
        for _ in range(2):
            for node in assigns:
                kind = self.kind_of(node.value)
                if kind[0] == "unknown":
                    continue
                for target in node.targets:
                    if isinstance(target, ast.Name):
                        self.dirs[target.id] = kind

    def kind_of(self, node: ast.expr) -> DirKind:
        return self._kind_of_parts(_div_chain(node))

    def _kind_of_parts(self, parts: list[ast.expr]) -> DirKind:
        # 1. An explicit `.cache`/`luxar` pair anywhere in the chain fixes the
        #    root, whatever spelled it (`Path.home()`, an expanded env var, …).
        lits = [_as_str(p, self.consts) for p in parts]
        for i in range(len(parts) - 1):
            if lits[i] == ".cache" and lits[i + 1] == "luxar":
                return self._after_root(parts[i + 2 :])

        head, rest = parts[0], parts[1:]

        # 2. A name or attribute that IS the cache root (registry.DEMO_CACHE_ROOT).
        if isinstance(head, ast.Attribute) and head.attr in _CACHE_ROOT_NAMES:
            return self._after_root(rest)
        if isinstance(head, ast.Name) and head.id in _CACHE_ROOT_NAMES:
            return self._after_root(rest)

        # 3. A name already bound to a classified directory.
        if isinstance(head, ast.Name) and head.id in self.dirs:
            return self._extend(self.dirs[head.id], rest)

        # 4. A leftmost expression we can prove is not the cache: a
        #    `Path(__file__)`-rooted chain, i.e. the in-repo source tree. It is
        #    provably not `~/.cache/luxar` because `__file__` is the installed
        #    module. The `__file__` root is
        #    REQUIRED, not decoration: `Path("/tmp/scratch") / "data" /
        #    "myfit.zip"` would otherwise make an arbitrary local artifact look
        #    provably outside the cache — and any OTHER `Path(...)` head is a
        #    directory whose contents this pass cannot read, so it falls through
        #    to "unknown" below rather than claiming a proof it does not have.
        #    `Path.home() / ".cache/luxar" / DS` (one literal, not two) and
        #    `Path("~/.cache/luxar").expanduser() / DS` are exactly that shape:
        #    they ARE the cache dir, and calling them "other" would make the
        #    #1618 bug class re-introducible by one plausible edit.
        base = _base_atom(head)
        if isinstance(base, ast.Call):
            func = base.func
            is_path_call = (
                isinstance(func, ast.Name) and func.id in self.path_names
            ) or (
                isinstance(func, ast.Attribute)
                and isinstance(func.value, ast.Name)
                and func.value.id in self.path_names
            )
            rooted_in_file = any(
                isinstance(a, ast.Name) and a.id == "__file__" for a in base.args
            )
            if is_path_call and rooted_in_file:
                return ("other", None)

        return ("unknown", None)

    def _variants(self, dataset: str | None) -> frozenset[str]:
        """Variant names the manifest declares for *dataset*."""
        spec = self.datasets.get(dataset or "") or {}
        return frozenset(spec.get("variants") or {})

    def _after_root(self, rest: list[ast.expr]) -> DirKind:
        """Classify what follows ``~/.cache/luxar``."""
        if not rest:
            return ("root", None)
        dataset = _as_str(rest[0], self.consts)
        if dataset is None:
            return ("unknown", None)
        return self._extend(("cache", dataset), rest[1:])

    def _extend(self, kind: DirKind, rest: list[ast.expr]) -> DirKind:
        if not rest:
            return kind
        if kind[0] == "root":
            return self._after_root(rest)
        if kind[0] == "cache":
            # A subdirectory of the cache dir — with ONE exception the fix
            # rests on being right about: `ensure_dataset` caches a VARIANT's
            # files under `<root>/<ds>/<variant>/` and checksum-verifies and
            # quarantines them there exactly as it does the flat case (see
            # `data_fetch.ensure_dataset`), which is precisely why
            # `local_fit_path` inserts `local/` AFTER the variant. So a
            # declared variant is still the manifest's own directory; anything
            # else (`local/`, and — held by
            # `test_no_shipped_variant_is_named_local` — nothing named `local`
            # is ever a variant) is not.
            variants = self._variants(kind[1])
            sub = _as_str(rest[0], self.consts)
            if sub is None:
                # Unresolvable subdir: only a dataset that HAS variants could
                # have one that is manifest-owned. Claim no proof there.
                return ("unknown", None) if variants else ("other", None)
            if len(rest) == 1 and sub in variants:
                return ("cache", kind[1])
            return ("other", None)
        return (kind[0], None)


def _basenames(
    node: ast.expr, consts: dict[str, str | list[str]]
) -> tuple[list[str], re.Pattern[str] | None, bool]:
    """``(exact names, pattern, pattern-is-all-wildcard)`` for a path component.

    Covers the three shapes the demos actually use — a literal, a module constant
    (single or indexed out of a list), and an f-string built in a per-channel
    loop. The f-string becomes a regex with ``.*`` for each interpolation, so
    ``f"kidney_ch{i}.gsplats.zarr.zip"`` still matches the three pinned names.

    The third element flags an f-string with NO literal text of its own
    (``f"{name}"``): its pattern is ``.*``, which matches every pinned name and
    is therefore evidence of nothing. Reported as a CONFIRMED squat, it accused
    a raw-source download of being handed a file name its source never mentions,
    and the only way to satisfy it was to rename a variable.
    """
    literal = _as_str(node, consts)
    if literal is not None:
        return [literal], None, False
    if isinstance(node, ast.Subscript) and isinstance(node.value, ast.Name):
        value = consts.get(node.value.id)
        if isinstance(value, list):
            return value, None, False
    if isinstance(node, ast.JoinedStr):
        pattern = "".join(
            re.escape(str(v.value)) if isinstance(v, ast.Constant) else ".*"
            for v in node.values
        )
        wildcard = not any(
            isinstance(v, ast.Constant) and str(v.value) for v in node.values
        )
        return [], re.compile(pattern + r"\Z"), wildcard
    return [], None, False


def _pinned(datasets: dict, dataset: str) -> set[str]:
    """File names the manifest pins for a ``zenodo`` dataset (empty otherwise).

    Every VARIANT's files count too: ``ensure_dataset`` resolves a variant to
    ``<root>/<ds>/<variant>/<file>`` and checksum-verifies it there, so a
    variant-pinned name is a name the manifest owns. Reading only the top-level
    ``files`` made ``h2afva``'s ``253tp`` payload invisible to the whole
    analysis.
    """
    spec = datasets.get(dataset)
    if spec is None or spec.get("bucket") != "zenodo":
        return set()
    names = {f["name"] for f in (spec.get("files") or [])}
    for meta in (spec.get("variants") or {}).values():
        names |= {f["name"] for f in (meta.get("files") or [])}
    return names


def _referenced_pinned(tree: ast.Module, datasets: dict) -> dict[str, str]:
    """``{pinned file name: dataset}`` over every zenodo dataset the module names.

    Keyed on the DATASET NAME rather than on a path spelling: whatever a demo
    calls the directory, a pinned file name of a dataset it talks about is a
    name the manifest owns somewhere. This is what lets an UNRESOLVABLE
    directory still be reported instead of passing silently.
    """
    out: dict[str, str] = {}
    for node in ast.walk(tree):
        if isinstance(node, ast.Constant) and isinstance(node.value, str):
            for name in _pinned(datasets, node.value):
                out[name] = node.value
    return out


def _manifest_owned(
    node: ast.expr,
    ctx: _Ctx,
    datasets: dict,
    referenced: dict[str, str],
) -> tuple[str, str, list[str]] | None:
    """``(kind, dataset, names)`` if *node* builds — or may build — a manifest dest.

    *kind* is ``"owned"`` when the directory resolves to the manifest's own
    ``<root>/<dataset>`` AND the basename resolves to a name that dataset pins.
    It is ``"blind"`` in the two states the analysis cannot settle: the
    directory could not be classified but the basename is one this demo's own
    dataset pins, and — the stronger evidence of the two — the directory IS
    provably the manifest's own but the basename could not be resolved at all.
    ``matched`` is empty in that second case.
    """
    chain = _div_chain(node)
    if len(chain) < 2:
        return None
    dir_kind, dataset = ctx._kind_of_parts(chain[:-1])
    if dir_kind not in ("cache", "unknown"):
        return None
    names, pattern, wildcard = _basenames(chain[-1], ctx.consts)

    if dir_kind == "cache":
        assert dataset is not None
        pinned = _pinned(datasets, dataset)
        matched = {n for n in names if n in pinned}
        if pattern is not None:
            matched |= {p for p in pinned if pattern.match(p)}
        if matched and wildcard:
            # `f"{name}"` matches every pinned name and identifies none of
            # them: an unresolved basename, not a proven squat.
            return ("blind", dataset, [])
        if matched:
            return ("owned", dataset, sorted(matched))
        # An UNRESOLVABLE basename joined onto a directory PROVED to be the
        # manifest's own is the strongest evidence this analysis can hold, and
        # used to be its quietest answer — the exact asymmetry the weaker
        # "unknown dir + pinned name" case already got right. Report it.
        # (`names` non-empty means the basename resolved and simply is not
        # pinned: a demo's own scratch file in its cache dir, nobody's business.
        # An empty `pinned` set means the dataset never reaches the checksum
        # gate at all.)
        if not names and pattern is None and pinned:
            return ("blind", dataset, [])
        return None

    hits = {n: referenced[n] for n in names if n in referenced}
    if pattern is not None:
        hits |= {p: ds for p, ds in referenced.items() if pattern.match(p)}
    if not hits:
        return None
    return ("blind", sorted(set(hits.values()))[0], sorted(hits))


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


def analyse_local_fit(source: str, datasets: dict) -> tuple[list[str], list[str]]:
    """``(violations, blind spots)`` — manifest-owned paths handed to a writer.

    Two-step, because construction alone is legal: ``ct_totalsegmentator`` builds
    ``CACHE_LABELS`` to READ the sidecar the fetch brought down.

      1. Bind every name assigned a ``<manifest cache dir> / <pinned file>``
         expression (and note the inline ones, and plain ``q = P`` aliases).
      2. Report each use of such a name (or expression) as a call argument — or
         as the receiver of a writing method — unless the callee is in
         :data:`_MANIFEST_PATH_READERS`.

    A directory is resolved through :class:`_Ctx`, which knows four spellings of
    the cache dir: the literal ``Path.home() / ".cache" / "luxar" / <ds>`` chain,
    a ``<CACHE ROOT NAME> / <ds>`` join (``registry.DEMO_CACHE_ROOT``,
    ``_DEFAULT_CACHE_ROOT``), a name bound to either, and a name bound to a name
    bound to either.

    Neither half of a join passes silently when it cannot be settled — that was
    the rest of the hole, and it was asymmetric. A BLIND SPOT is returned when
    EITHER the directory is unknown and the basename is one the demo's own
    dataset pins, OR the directory is provably the manifest's own and the
    basename cannot be resolved (a ``zip``-bound loop variable, an f-string
    behind one hop, a fully interpolated ``f"{name}"``, a function parameter).
    Neither is proof of a squat; both are
    proof that the gate cannot vouch for the demo, which is a reviewable state
    rather than a green tick. (A directory the analysis can prove is NOT the
    cache — a ``Path(__file__)``-rooted path, whether or not it is the packaged
    ``data`` tree; a subdirectory of the cache dir that is not a declared
    VARIANT of that dataset, ``local/`` above all — is neither, and neither is a
    basename that resolves and simply is not pinned. A directory spelling that
    is merely unfamiliar is NOT a proof: only the ``__file__`` root is, so
    ``Path.home() / ".cache/luxar" / DS`` and ``Path("~/.cache/luxar")
    .expanduser() / DS`` come back as blind spots, not as green.)

    A VARIANT subdirectory is the manifest's own directory: ``ensure_dataset``
    caches ``<root>/<ds>/<variant>/<file>`` and checksum-verifies it there, so
    ``_pinned`` unions every variant's ``files`` and ``local/`` is exempt only
    because no variant may be called that.

    What it still CANNOT see, honestly, and where each hole is BOUNDED:

    * a join spelled with ``os.path.join(d, name)`` or ``d.with_name(name)``.
      ``d.joinpath(name)`` IS flattened like ``/``; those two are not, so a
      pinned basename reaching the cache dir through one of them is missed
      entirely — the "keys on the dataset name, not the spelling" backstop
      covers ``/`` and ``.joinpath`` only.
    * a write whose receiver is an expression rather than a name or a join —
      ``P.resolve().write_bytes(b)`` is invisible, because the receiver is a
      ``Call``.
    * ``pathlib`` reached as ``pathlib.Path(...)`` rather than an imported
      ``Path`` (aliased ``from pathlib import Path as P`` IS handled): a
      packaged-data destination spelled that way may be REPORTED rather than
      recognised as outside the cache, but it is not missed.
    """
    tree = ast.parse(source)
    ctx = _Ctx(tree, datasets)
    referenced = _referenced_pinned(tree, datasets)
    owned_names = _owned_names(tree, ctx, datasets, referenced)

    found: dict[str, list[str]] = {"owned": [], "blind": []}
    for node in ast.walk(tree):
        if isinstance(node, ast.Call):
            _scan_call(node, ctx, datasets, referenced, owned_names, found)
    return found["owned"], found["blind"]


def _owned_names(
    tree: ast.Module, ctx: _Ctx, datasets: dict, referenced: dict[str, str]
) -> dict[str, tuple[str, str, list[str]]]:
    """Names bound to a manifest-owned (or unvouched-for) path, aliases included."""
    owned_names: dict[str, tuple[str, str, list[str]]] = {}
    aliases: list[tuple[str, str]] = []
    for node in ast.walk(tree):
        if not isinstance(node, ast.Assign):
            continue
        owned = _manifest_owned(node.value, ctx, datasets, referenced)
        for target in node.targets:
            if not isinstance(target, ast.Name):
                continue
            if owned is not None:
                owned_names[target.id] = owned
            elif isinstance(node.value, ast.Name):
                aliases.append((target.id, node.value.id))
    # `q = P; save(q)` — laundering through a second name was a documented hole.
    for _ in range(len(aliases)):
        for dst, src in aliases:
            if src in owned_names and dst not in owned_names:
                owned_names[dst] = owned_names[src]
    return owned_names


def _describe(kind: str, dataset: str, matched: list[str], what: str) -> str:
    if kind == "owned":
        where = (
            f"~/.cache/luxar/{dataset}/{{{', '.join(matched)}}} — the path the "
            f"manifest fetch owns for {dataset!r}"
        )
    elif not matched:
        where = (
            f"~/.cache/luxar/{dataset}/<a basename this analysis cannot "
            f"resolve> — the directory the manifest fetch owns for {dataset!r}, "
            f"so the basename decides whether this is a squat"
        )
    else:
        where = (
            f"a directory this analysis cannot identify, joined with "
            f"{{{', '.join(matched)}}} — name(s) the manifest pins for "
            f"{dataset!r}, so it may or may not be the fetch's own path"
        )
    return f"{what} is handed {where}"


def _scan_call(
    node: ast.Call,
    ctx: _Ctx,
    datasets: dict,
    referenced: dict[str, str],
    owned_names: dict[str, tuple[str, str, list[str]]],
    found: dict[str, list[str]],
) -> None:
    """Record every manifest-owned path this one call could write."""

    def resolve(arg: ast.expr) -> tuple[str, str, list[str], str] | None:
        if isinstance(arg, ast.Name) and arg.id in owned_names:
            kind, dataset, matched = owned_names[arg.id]
            return kind, dataset, matched, arg.id
        owned = _manifest_owned(arg, ctx, datasets, referenced)
        return None if owned is None else (*owned, "an inline path")

    func = node.func
    callee = func.attr if isinstance(func, ast.Attribute) else getattr(func, "id", "")

    # `P.write_bytes(blob)` / `P.unlink()`: the path is the RECEIVER, not an
    # argument, so the argument scan below cannot see it.
    if isinstance(func, ast.Attribute) and callee not in _MANIFEST_PATH_READ_METHODS:
        hit = resolve(func.value)
        if hit is not None:
            kind, dataset, matched, what = hit
            found[kind].append(
                _describe(
                    kind, dataset, matched, f"{what}.{callee}() (line {node.lineno})"
                )
            )

    if callee in _MANIFEST_PATH_READERS:
        return
    args = list(node.args) + [kw.value for kw in node.keywords]
    for arg in _searchable(args):
        hit = resolve(arg)
        if hit is None:
            continue
        kind, dataset, matched, what = hit
        found[kind].append(
            _describe(
                kind, dataset, matched, f"{what} ({callee}(), line {node.lineno})"
            )
        )


def local_fit_violations(source: str, datasets: dict) -> list[str]:
    """Just the confirmed squats — see :func:`analyse_local_fit`."""
    return analyse_local_fit(source, datasets)[0]


#: Demos whose local-artifact writes this analysis cannot vouch for, with the
#: reason. Enumerated, and held exact by
#: :func:`test_the_unanalysable_demo_list_is_exactly_right` — "the gate could not
#: see this file" is an enumerated state, not a silent pass.
_ANALYSIS_BLIND_SPOTS: dict[str, str] = {
    # Downloads the eight DESI DR1 LSS source catalogs into its cache dir under
    # names read out of the nested `TRACERS[name]["files"]` dict, which this
    # analysis does not resolve. None of them can collide: the dataset pins one
    # file, `desi_dr1_cosmic_web.luxar.zarr.zip`, and every downloaded name ends
    # `_clustering.dat.fits`.
    "demo_desi_galaxies": "source-catalog names come from a nested dict literal",
}


@pytest.mark.parametrize("path", DEMO_PATHS, ids=lambda p: p.stem)
def test_a_local_artifact_never_squats_a_manifest_pinned_path(path: Path) -> None:
    """A demo's own computed artifact must go to ``local_fit_path``, not the fetch's.

    Reverting any one of the eleven demos migrated in #1618 makes this fail:
    each used to pass a ``CACHE_DIR / <pinned name>`` path to ``save``,
    ``save_with_lod`` or an ``_save_*`` sidecar writer, and every one of those
    call shapes is caught — verified by running this analysis over all eleven
    pre-fix sources (see :func:`analyse_local_fit` for what is and is not
    visible to a static pass).

    Five of those demos no longer bind a cache dir at all, which used to make
    this test pass *unconditionally* for them: the analysis bailed out on an
    empty cache-dir map and returned no violations whether or not there was
    anything wrong. It no longer bails, and the blind-spot half below turns
    "nothing I could analyse" into a reportable state.
    """
    violations, blind = analyse_local_fit(path.read_text(), _manifest())
    assert not violations, (
        f"{path.name}: "
        + "; ".join(violations)
        + ". A locally computed stand-in can never match the manifest sha256, so "
        "the next fetch quarantines it and the demo recomputes on EVERY launch "
        "(#1618). Write it to luxar.demos._support.datasets.data_fetch.local_fit_path(<dataset>, "
        "<file>) instead. If the call really only READS the fetched file, add "
        "its name to _MANIFEST_PATH_READERS with a reason."
    )
    if path.stem in _ANALYSIS_BLIND_SPOTS:
        return
    assert not blind, (
        f"{path.name}: "
        + "; ".join(blind)
        + ". Spell the directory in a way this gate can resolve (a "
        "`~/.cache/luxar/<ds>` chain, or a join onto DEMO_CACHE_ROOT / "
        "_DEFAULT_CACHE_ROOT), or — if the path really is not the fetch's — add "
        "the demo to _ANALYSIS_BLIND_SPOTS with a reason a reviewer can check."
    )


def test_the_unanalysable_demo_list_is_exactly_right() -> None:
    """``_ANALYSIS_BLIND_SPOTS`` names every demo the gate cannot vouch for.

    The same shape as ``test_the_exception_list_is_exactly_the_local_compute_datasets``
    above, and for the same reason: an exemption that is not enumerated is
    indistinguishable from a demo that passed.
    """
    datasets = _manifest()
    unanalysable = {
        path.stem
        for path in DEMO_PATHS
        if analyse_local_fit(path.read_text(), datasets)[1]
    }
    assert unanalysable == set(_ANALYSIS_BLIND_SPOTS), (
        "the set of demos this gate cannot analyse changed; every entry in "
        "_ANALYSIS_BLIND_SPOTS must be a demo that really is unanalysable, and "
        "every unanalysable demo must be listed there with a reason"
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
    # 7. A second NAME must not launder it either (`q = P; save(q)`).
    assert local_fit_violations(
        header + 'P = CACHE_DIR / "toy_ch0.zip"\nq = P\nsave_with_lod(fit, q)\n',
        datasets,
    )
    # 8. A write through a method ON the path, where it is the receiver.
    assert local_fit_violations(
        header + 'P = CACHE_DIR / "toy_ch0.zip"\nP.write_bytes(blob)\n', datasets
    )
    # 9. The `registry.DEMO_CACHE_ROOT / DS` head — the spelling
    #    demo_gsplats_4d_cell_tracking_challenge already uses, and the one the
    #    literal-chain-only version of this analysis was blind to.
    assert local_fit_violations(
        'from luxar.demos import registry\nDS = "toy_ds"\n'
        "CACHE_DIR = registry.DEMO_CACHE_ROOT / DS\n"
        'save_with_lod(fit, CACHE_DIR / "toy_ch0.zip")\n',
        datasets,
    )
    # 10. …and the bare-name form of the same root.
    assert local_fit_violations(
        'from luxar.demos.registry import DEMO_CACHE_ROOT\nDS = "toy_ds"\n'
        'save_with_lod(fit, DEMO_CACHE_ROOT / DS / "toy_ch0.zip")\n',
        datasets,
    )
    # 11. A demo with NO cache-dir constant at all: the whole chain inline.
    assert local_fit_violations(
        'from pathlib import Path\nDS = "toy_ds"\n'
        'save_with_lod(fit, Path.home() / ".cache" / "luxar" / DS / "toy_ch0.zip")\n',
        datasets,
    )
    # 12. A cache dir bound indirectly, through another name.
    assert local_fit_violations(
        header + 'D = CACHE_DIR\nsave_with_lod(fit, D / "toy_ch0.zip")\n', datasets
    )
    # 13. An `atomic_copy_file` is a write like any other. Even packaged bytes
    #     must go through the manifest helper so their digest is actually checked.
    assert local_fit_violations(
        header
        + 'atomic_copy_file(local_fit_path(DS, "toy_ch0.zip"), CACHE_DIR / "toy_ch0.zip")\n',
        datasets,
    )

    # 14. A source that merely has "data" somewhere in it is no different.
    assert local_fit_violations(
        header
        + 'atomic_copy_file(Path("/tmp/scratch") / "data" / "myfit.zip", CACHE_DIR / "toy_ch0.zip")\n',
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
    # A non-writing method on the path (the existence probe every demo makes).
    assert not local_fit_violations(
        header + 'P = CACHE_DIR / "toy_ch0.zip"\nif P.exists():\n    pass\n', datasets
    )
    # A packaged Git LFS source is still a violation: copying it by hand is the
    # exact path that bypassed the manifest checksum in #2343.
    assert local_fit_violations(
        header + 'DATA_DIR = Path(__file__).parent / "data" / DS\n'
        'LFS = DATA_DIR / "toy_ch0.zip"\natomic_copy_file(LFS, CACHE_DIR / "toy_ch0.zip")\n',
        datasets,
    )
    # …and the same copy spelled through an aliased pathlib import.
    assert local_fit_violations(
        'from pathlib import Path as P\nDS = "toy_ds"\n'
        'CACHE_DIR = P.home() / ".cache" / "luxar" / DS\n'
        'DATA_DIR = P(__file__).resolve().parent / "data" / DS\n'
        'atomic_copy_file(DATA_DIR / "toy_ch0.zip", CACHE_DIR / "toy_ch0.zip")\n',
        datasets,
    )
    # The local-fit namespace itself is a SUBDIRECTORY of the cache dir, not the
    # manifest's own path — the distinction the whole fix rests on.
    assert not local_fit_violations(
        header + 'save_with_lod(fit, CACHE_DIR / "local" / "toy_ch0.zip")\n', datasets
    )


def test_the_guard_reports_a_directory_it_cannot_identify() -> None:
    """An unresolvable directory is a BLIND SPOT, never a silent pass.

    This is the half that makes the gate hold against a DIRECTORY spelling
    nobody has thought of: it keys on the DATASET NAME (a file name that dataset
    pins), not on how the directory was built, so a directory built any way at
    all still gets reported as unvouched-for rather than green.

    Its reach is the JOIN, though, not the directory: ``d / name`` and
    ``d.joinpath(name)`` are seen, ``os.path.join(d, name)`` and
    ``d.with_name(name)`` are not — the last case below pins that limit so the
    docstrings and the behaviour cannot drift apart again.

    "Any way at all" is load-bearing and was once untrue: a ``Path(...)``-headed
    chain used to be classified as PROVABLY not the cache, which took three
    ordinary spellings of ``~/.cache/luxar`` out of the gate's reach silently.
    Only a ``Path(__file__)``-rooted head is a proof (it is the installed module,
    not the cache); the rest are pinned below.
    """
    datasets = {
        "toy_ds": {"bucket": "zenodo", "files": [{"name": "toy_ch0.zip"}]},
    }
    violations, blind = analyse_local_fit(
        'DS = "toy_ds"\nD = some_helper(DS)\nsave_with_lod(fit, D / "toy_ch0.zip")\n',
        datasets,
    )
    assert not violations
    assert blind and "toy_ds" in blind[0]

    # The same join spelled as a method: also reported (it used to be clean).
    _, blind_joinpath = analyse_local_fit(
        'DS = "toy_ds"\nD = some_helper(DS)\n'
        'save_with_lod(fit, D.joinpath("toy_ch0.zip"))\n',
        datasets,
    )
    assert blind_joinpath and "toy_ds" in blind_joinpath[0]

    # …and the two spellings that are NOT covered, recorded as the known limit
    # rather than left to be discovered as a surprise.
    for expr in ('os.path.join(D, "toy_ch0.zip")', 'D.with_name("toy_ch0.zip")'):
        assert analyse_local_fit(
            f'DS = "toy_ds"\nD = some_helper(DS)\nsave_with_lod(fit, {expr})\n',
            datasets,
        ) == ([], []), f"{expr} is documented as invisible; update the docstring"

    # Three ordinary spellings of the cache dir itself that the literal
    # `.cache`/`luxar` PAIR does not match. None of them is provably anything,
    # so each must come back as a blind spot rather than green — the #1618 bug
    # class is one plausible edit away from every one of them.
    for head in (
        'Path.home() / ".cache/luxar"',  # one literal, not two
        'Path("~/.cache/luxar").expanduser()',  # already a demos idiom
        'Path(os.path.expanduser("~/.cache/luxar"))',
    ):
        violations, spots = analyse_local_fit(
            f'from pathlib import Path\nimport os\nDS = "toy_ds"\n'
            f"CACHE_DIR = {head} / DS\n"
            'save_with_lod(fit, CACHE_DIR / "toy_ch0.zip")\n',
            datasets,
        )
        assert not violations, (head, violations)
        assert spots and "toy_ds" in spots[0], f"{head} passed silently"

    # A directory the analysis can PROVE is not the cache is not a blind spot:
    # a `Path(__file__)`-rooted chain, which is the installed module's own tree.
    for tail in ('/ "data" / DS', "/ DS"):
        _, blind_packaged = analyse_local_fit(
            'from pathlib import Path\nDS = "toy_ds"\n'
            f"D = Path(__file__).parent {tail}\n"
            'save_with_lod(fit, D / "toy_ch0.zip")\n',
            datasets,
        )
        assert not blind_packaged, tail

    # Nor is an unresolvable directory joined with a name the manifest does not
    # pin — a demo's own scratch file is nobody's business.
    _, blind_unpinned = analyse_local_fit(
        'DS = "toy_ds"\nD = some_helper(DS)\nsave_with_lod(fit, D / "scratch.zip")\n',
        datasets,
    )
    assert not blind_unpinned


def test_the_guard_reports_an_unresolvable_basename_in_the_manifests_own_dir() -> None:
    """The mirror of the test above, and the STRONGER evidence of the two.

    A directory proved to be ``~/.cache/luxar/<ds>`` joined with a basename the
    analysis cannot resolve used to be its quietest answer, while the weaker
    "unidentifiable directory + pinned name" was correctly reported. The
    asymmetry was backwards: these shapes are one token from the squats the
    migrated demos actually had.
    """
    datasets = {
        "toy_ds": {
            "bucket": "zenodo",
            "files": [{"name": "toy_ch0.zip"}, {"name": "toy_ch1.zip"}],
        },
        "toy_local": {"bucket": "local-compute", "files": [{"name": "toy_ch0.zip"}]},
    }
    header = 'from pathlib import Path\nDS = "toy_ds"\nCACHE_DIR = Path.home() / ".cache" / "luxar" / DS\n'

    def blind(src: str) -> list[str]:
        violations, spots = analyse_local_fit(src, datasets)
        assert not violations, violations
        return spots

    # A name bound by `zip(...)` — the kidney demos' idiom, one token away from
    # the indexed form (case 3 above) that IS a confirmed violation.
    assert blind(
        header + 'FILES = ["toy_ch0.zip", "toy_ch1.zip"]\n'
        "for volume, name in zip(volumes, FILES):\n"
        "    fit_channel(volume, CACHE_DIR / name)\n"
    )
    # A basename bound to an f-string, then joined (the f-string is not visible
    # through the extra hop, unlike the inline `CACHE_DIR / f"..."` of case 2).
    assert blind(
        header + "for i in range(2):\n"
        '    n = f"toy_ch{i}.zip"\n'
        "    save_with_lod(fit, CACHE_DIR / n)\n"
    )
    # A basename arriving as a function parameter.
    assert blind(
        header + "def fit_channel(volume, name):\n"
        "    save_with_lod(volume, CACHE_DIR / name)\n"
    )
    # An f-string with no literal text of its own is the same unresolved state
    # spelled differently: its `.*` pattern matches every pinned name and
    # identifies none. Reported as a CONFIRMED squat it accused a raw-source
    # download of being handed `toy_ch0.zip`, which its source never mentions —
    # and no edit short of renaming the variable could satisfy it.
    assert blind(header + 'requests.download(CACHE_DIR / f"{raw_name}")\n')
    # …while an f-string that DOES carry literal text still resolves to the
    # pinned names it matches, and stays a confirmed violation.
    assert local_fit_violations(
        header + 'save_with_lod(fit, CACHE_DIR / f"toy_ch{i}.zip")\n', datasets
    )
    assert not local_fit_violations(
        header + 'save_with_lod(fit, CACHE_DIR / f"scratch_{i}.zip")\n', datasets
    )

    # And the negatives, so this does not become a blanket "any join is blind":
    # a basename that RESOLVES and simply is not pinned is a demo's own scratch
    # file in its own cache dir, which is nobody's business.
    assert not blind(header + 'save_with_lod(fit, CACHE_DIR / "scratch.zip")\n')
    assert not blind(
        header + 'RAW = "source.tif"\nrequests.download(CACHE_DIR / RAW)\n'
    )
    # A dataset that never reaches the checksum gate is not checked at all.
    assert not blind(
        'from pathlib import Path\nDS = "toy_local"\n'
        'CACHE_DIR = Path.home() / ".cache" / "luxar" / DS\n'
        "save_with_lod(fit, CACHE_DIR / name)\n"
    )
    # Nor is the local-fit namespace, which is a SUBDIRECTORY of the cache dir.
    assert not blind(header + 'save_with_lod(fit, CACHE_DIR / "local" / name)\n')


def test_a_variant_subdirectory_is_the_manifests_own_directory_too() -> None:
    """``<root>/<ds>/<variant>/`` is a fetch destination, not a free subdirectory.

    ``ensure_dataset`` sets ``cache_dir = root / name / variant_name`` and
    checksum-verifies and quarantines there exactly as it does the flat case —
    which is *why* ``local_fit_path`` inserts ``local/`` after the variant. A
    blanket "any subdirectory of the cache dir is not the manifest's" rule made
    the whole variant namespace invisible, both here and in ``_pinned``, which
    read only the top-level ``files``.

    Latent today (no demo requests a variant), so it is pinned synthetically
    against the shape the shipped ``h2afva`` entry already has.
    """
    datasets = {
        "toy_ds": {
            "bucket": "zenodo",
            "files": [{"name": "toy_base.zip"}],
            "variants": {
                "light": {"default": True, "files": [{"name": "toy_light.zip"}]},
                "full": {"files": [{"name": "toy_full.zip"}]},
            },
        },
    }
    header = 'from pathlib import Path\nDS = "toy_ds"\nCACHE_DIR = Path.home() / ".cache" / "luxar" / DS\n'

    # The variant's own file, in the variant's own cache dir: a squat.
    assert local_fit_violations(
        header + 'save_with_lod(fit, CACHE_DIR / "full" / "toy_full.zip")\n', datasets
    )
    # Spelled as one inline chain, and through the root name.
    assert local_fit_violations(
        'from pathlib import Path\nDS = "toy_ds"\n'
        'save_with_lod(fit, Path.home() / ".cache" / "luxar" / DS / "light" / "toy_light.zip")\n',
        datasets,
    )
    assert local_fit_violations(
        'from luxar.demos import registry\nDS = "toy_ds"\n'
        'save_with_lod(fit, registry.DEMO_CACHE_ROOT / DS / "light" / "toy_light.zip")\n',
        datasets,
    )
    # A variant-pinned name is pinned for the dataset wherever it appears.
    assert local_fit_violations(
        header + 'save_with_lod(fit, CACHE_DIR / "toy_full.zip")\n', datasets
    )
    # `local/` INSIDE the variant is the exempt namespace — the reason the fix
    # puts it after the variant rather than before it.
    assert not local_fit_violations(
        header + 'save_with_lod(fit, CACHE_DIR / "full" / "local" / "toy_full.zip")\n',
        datasets,
    )
    # A subdirectory that is not a declared variant is still not a destination.
    assert not local_fit_violations(
        header + 'save_with_lod(fit, CACHE_DIR / "scratch" / "toy_full.zip")\n',
        datasets,
    )
    # An UNRESOLVABLE subdirectory of a dataset that HAS variants might be one,
    # so it is a blind spot rather than a silent pass. (For a dataset with no
    # variants the classifier can prove the subdirectory is not a destination —
    # but that is not observable from out here, because `CACHE_DIR / sub` is
    # itself an unresolvable basename in the manifest's own dir, which the rule
    # above this one already reports. Both spellings end up reviewed; only the
    # variant one is reported for the right reason.)
    _, blind_variant = analyse_local_fit(
        header + 'save_with_lod(fit, CACHE_DIR / sub / "toy_full.zip")\n', datasets
    )
    assert blind_variant and "toy_ds" in blind_variant[0]


def test_no_shipped_variant_is_named_local() -> None:
    """``<name>/local/`` must stay disjoint from every ``<name>/<variant>/``.

    ``ensure_dataset`` caches a variant's files under ``<name>/<variant>/``, so a
    variant called ``local`` would put the fetch's destinations back inside the
    namespace that exists to be out of its reach. ``local_fit_path`` refuses that
    variant at the call site; this holds the shipped manifest to it too.
    """
    from luxar.demos._support.datasets.data_fetch import LOCAL_FIT_DIRNAME

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


def test_visible_human_reaches_the_manifest_on_a_cold_cache(
    tmp_path, monkeypatch
) -> None:
    """The cold-fetch verifier must exercise the path the demo really takes."""
    demo = importlib.import_module("luxar.demos.demo_gsplats_3d_visible_human_head")
    fit_path = tmp_path / "vh_head.gsplats.zarr.zip"
    colors_path = tmp_path / "vh_head_colors.npz"
    calls: list[tuple] = []
    stub_fit = GSplatData(
        centers=np.zeros((4, 3), dtype=np.float32),
        amplitudes=np.ones(4, dtype=np.float32),
        cholesky_factors=np.tile([1, 0, 1, 0, 0, 1], (4, 1)).astype(np.float32),
    )
    stub_colors = np.full((4, 3), 0.5, dtype=np.float32)

    def _fake_fetch(*args, **kwargs):
        calls.append(args)
        return [fit_path, colors_path]

    monkeypatch.setattr(demo, "RECOMPUTE", False)
    monkeypatch.setattr(demo, "LOCAL_FIT", tmp_path / "absent-local.gsplats.zarr.zip")
    monkeypatch.setattr(demo, "LOCAL_COLORS", tmp_path / "absent-local.npz")
    monkeypatch.setattr(demo, "ensure_dataset", _fake_fetch)
    monkeypatch.setattr(
        demo.GSplatData,
        "load",
        lambda path, *, include_stats: stub_fit if path == fit_path else None,
    )
    monkeypatch.setattr(
        demo,
        "_load_colors_f32",
        lambda path: stub_colors if path == colors_path else None,
    )
    monkeypatch.setattr(demo, "_colors_match_fit", lambda *args: True)

    def _refit_is_a_failure():
        raise AssertionError("fell through to the download-and-refit path")

    monkeypatch.setattr(demo, "download_head_slices", _refit_is_a_failure)

    fit, got_colors = demo.load_or_build()

    assert calls == [(demo.DEMO_NAME,)]
    assert fit is stub_fit
    assert got_colors is stub_colors
