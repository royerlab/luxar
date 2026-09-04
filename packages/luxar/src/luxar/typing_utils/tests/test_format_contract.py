"""Guardrails for the single-sourced cross-language format contract.

``format-contract/contract.yaml`` is projected into
``typing_utils/_format_contract.py`` (this side) and the viewer's
``types/format-contract.ts`` (consumer side) by
``scripts/gen_format_contract.py``. These tests assert:

1. the hand-written Python symbols that *consume* the contract stay in lockstep
   with it (the enum / named constants that can't just import a tuple), and
2. the committed projections have not drifted from the YAML (the same gate CI
   runs as ``hatch run check-contract``, exercised here so a stale checkout
   fails the local suite too).
"""

from __future__ import annotations

import runpy
import sys
from pathlib import Path

import pytest

from luxar.typing_utils import _format_contract as fc
from luxar.typing_utils import config, constants
from luxar.typing_utils.enums import NodeType

REPO_ROOT = Path(__file__).resolve().parents[6]
GEN_SCRIPT = REPO_ROOT / "scripts" / "gen_format_contract.py"


def test_node_type_enum_matches_contract() -> None:
    """The hand-written ``NodeType`` enum must mirror the contract, in order."""
    assert tuple(member.value for member in NodeType) == fc.NODE_TYPES


def test_named_node_type_constants_match_contract() -> None:
    """The individual ``NODE_TYPE_*`` constants must be exactly the contract set."""
    named = {
        constants.NODE_TYPE_SCENE,
        constants.NODE_TYPE_GROUP,
        constants.NODE_TYPE_POINTS,
        constants.NODE_TYPE_LINES,
        constants.NODE_TYPE_GSPLATS,
        constants.NODE_TYPE_MESH,
    }
    assert named == set(fc.NODE_TYPES)


def test_loader_types_are_a_subset_of_geometry_types() -> None:
    """``loader_types`` names the VIEWER-DRAWABLE subset of ``geometry_types``.

    The two lists answer different questions — "is this a geometry leaf?" (which
    the Python writer asks) versus "can the viewer load and draw it?" — and they
    legitimately differ, because a type becomes authorable before it becomes
    drawable. What is NOT legitimate is the reverse containment: the viewer cannot
    dispatch on a type the writer has no vocabulary for, so such an entry would be
    unrepresentable on disk.
    """
    assert set(fc.LOADER_TYPES) <= set(fc.GEOMETRY_TYPES)
    assert fc.LOADER_TYPES, "loader sequence must not be empty"
    assert len(fc.LOADER_TYPES) == len(set(fc.LOADER_TYPES))


@pytest.mark.skipif(
    not GEN_SCRIPT.exists(),
    reason="generator script not present (packaged install without repo scripts/)",
)
@pytest.mark.parametrize(
    ("loader_types", "expected"),
    [
        pytest.param([], "must not be empty", id="empty"),
        pytest.param(
            ["points", "not_a_geometry_type"], "missing from geometry_types", id="stray"
        ),
        pytest.param(["points", "points"], "duplicate", id="duplicates"),
    ],
)
def test_generator_rejects_malformed_loader_types(
    loader_types: list, expected: str
) -> None:
    """The ``loader_types`` invariants are ENFORCED by codegen, not just asserted.

    Mirrors the ``geometry_types`` cases below. Without these a malformed contract
    would generate happily and break far away — an empty list emits an empty
    TypeScript union, and a stray entry gives the viewer a dispatch kind no store
    can contain.
    """
    import importlib.util

    spec = importlib.util.spec_from_file_location("gen_format_contract", GEN_SCRIPT)
    assert spec is not None and spec.loader is not None
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)

    good = mod.load_contract()
    tampered = {**good, "loader_types": loader_types}

    with pytest.raises(SystemExit) as exc:
        mod._loader_types(tampered)
    assert expected in str(exc.value)


def test_scene_version_consumers_single_sourced() -> None:
    """Scene-format constants must be the ones projected from the contract."""
    assert constants.LUXAR_VERSION_CURRENT == fc.SCENE_FORMAT_VERSION
    assert constants.DEFAULT_ZARR_VERSION == fc.SCENE_FORMAT_VERSION
    assert config.DEFAULT_VERSION == fc.SCENE_FORMAT_VERSION
    assert config.SUPPORTED_VERSIONS == fc.SUPPORTED_SCENE_VERSIONS
    assert fc.SCENE_FORMAT_VERSION in fc.SUPPORTED_SCENE_VERSIONS


def test_gsplats_version_consumers_single_sourced() -> None:
    """``save_gsplats`` must re-export the contract's gsplats version symbols."""
    # NB: the ``luxar.gsplats.io`` package re-exports a ``save_gsplats``
    # *function* that shadows the submodule attribute, so reach the module
    # object via importlib (bypasses the attribute shadow).
    import importlib

    save_gsplats_mod = importlib.import_module("luxar.gsplats.io.save_gsplats")

    assert save_gsplats_mod.FORMAT_VERSION == fc.GSPLATS_FORMAT_VERSION
    assert save_gsplats_mod.SUPPORTED_FORMAT_VERSIONS == fc.SUPPORTED_GSPLATS_VERSIONS
    assert fc.GSPLATS_FORMAT_VERSION in fc.SUPPORTED_GSPLATS_VERSIONS


#: Documentation surfaces that state the CURRENT gsplats format version, each
#: with a regex capturing exactly that claim.
#:
#: The 3.3 -> 3.4 bump reached the constant, the spec and nothing else: the
#: published `FORMAT_AND_MIGRATION.md` — a page whose entire job is a table of
#: current on-disk versions — still said v3.3, as did the CLI reference, the
#: API reference and two package READMEs (audit A11-02). N-1-of-N, and the kind
#: of error that gets copied into someone else's reader.
#:
#: Anchored regexes rather than "does 3.3 appear anywhere", because the tree
#: legitimately holds v3.3 HISTORY ("v3.3 added the delta filter") and section
#: numbers ("### 3.3 Spacing tokens"). A gate that cannot tell those apart
#: either fires constantly or is switched off.
CURRENT_VERSION_CLAIMS: tuple[tuple[str, str], ...] = (
    (
        "docs/guides/user/FORMAT_AND_MIGRATION.md",
        r'`format_type="gsplats_zarr"`\) \| \*\*v(\d+\.\d+)\*\* \|',
    ),
    # Anchored on the Gsplats bullet specifically. A bare `current \*\*(...)\*\*`
    # also matches the SCENE version two lines above ("current **0.1**") — which
    # this gate caught on its first run, red against a correctly-swept page.
    (
        "docs/guides/user/FORMAT_AND_MIGRATION.md",
        r"\*\*Gsplats \(`\.gsplats\.zarr`\):\*\* current \*\*(\d+\.\d+)\*\*",
    ),
    ("docs/guides/user/CLI_REFERENCE.md", r"to the current v(\d+\.\d+) format"),
    (
        "docs/api/gsplats.rst",
        r"the v(\d+\.\d+) ``\.gsplats\.zarr`` on-disk structure",
    ),
    (
        "packages/luxar/src/luxar/gsplats/io/README.md",
        r"\(\*\*format v(\d+\.\d+)\*\*",
    ),
    (
        "packages/luxar/src/luxar/gsplats/io/README.md",
        r"v3\.0-v(\d+\.\d+) files remain readable",
    ),
    (
        "packages/luxar/src/luxar/gsplats/io/README.md",
        r"Convert to a new v(\d+\.\d+) file",
    ),
    (
        "packages/luxar/src/luxar/gsplats/io/README.md",
        r"→ v(\d+\.\d+) bare leaf",
    ),
    (
        "packages/luxar/src/luxar/gsplats/io/README.md",
        r"→ v(\d+\.\d+) additive ladder leaf",
    ),
    (
        "packages/luxar/src/luxar/gsplats/io/README.md",
        r"v(\d+\.\d+) `kind=lod` group",
    ),
    (
        "packages/luxar/src/luxar/gsplats/io/README.md",
        r"→ v(\d+\.\d+) node tree",
    ),
    (
        "packages/luxar/src/luxar/gsplats/io/README.md",
        r"stamped v(\d+\.\d+)",
    ),
    (
        "packages/luxar/src/luxar/gsplats/io/README.md",
        r"writes the v(\d+\.\d+) root header",
    ),
    (
        "packages/luxar/src/luxar/gsplats/io/README.md",
        r'`SUPPORTED_FORMAT_VERSIONS` \(`"3\.0"`, `"3\.1"`, `"3\.2"`, `"3\.3"`, `"(\d+\.\d+)"`\)',
    ),
    (
        "packages/luxar/src/luxar/gsplats/io/README.md",
        r"current writer emits v(\d+\.\d+)",
    ),
    (
        "packages/luxar/src/luxar/gsplats/io/README.md",
        r"selector attrs — to v(\d+\.\d+)",
    ),
    (
        "packages/luxar/src/luxar/gsplats/io/README.md",
        r"current node-tree format \(v3\.0-v(\d+\.\d+)\)",
    ),
    (
        "packages/luxar/src/luxar/gsplats/io/README.md",
        r"migration to v(\d+\.\d+) node-tree layout",
    ),
    (
        "packages/luxar/src/luxar/gsplats/io/README.md",
        r"Format spec \(v(\d+\.\d+)\)",
    ),
    (
        "CLAUDE.md",
        r"\.gsplats\.zarr is format v(\d+\.\d+)",
    ),
    (
        "CLAUDE.md",
        r"pre-v3\.2 pixel_size lod selector attrs\) → v(\d+\.\d+)",
    ),
    (
        "packages/luxar/src/luxar/gsplats/README.md",
        r"standalone v(\d+\.\d+) \.gsplats\.zarr",
    ),
    (
        "packages/luxar/src/luxar/gsplats/README.md",
        r"levels; v(\d+\.\d+) kind=lod group",
    ),
    (
        "packages/luxar/src/luxar/gsplats/README.md",
        r"In v(\d+\.\d+) a saved `\.gsplats\.zarr`",
    ),
    (
        "packages/luxar/src/luxar/gsplats/README.md",
        r"layouts → v(\d+\.\d+)",
    ),
    (
        "packages/luxar/src/luxar/cli/README.md",
        r"standalone `\.gsplats\.zarr` is a v(\d+\.\d+) node subtree",
    ),
    (
        "packages/luxar/src/luxar/cli/README.md",
        r"single current-format \(v(\d+\.\d+)\)",
    ),
    (
        "packages/luxar-viewer/src/data/codecs/README.md",
        r"current \*\*standalone gsplat format is v(\d+\.\d+)\*\*",
    ),
    (
        "packages/luxar-viewer/src/data/codecs/README.md",
        r"Supported versions: 3\.0, 3\.1, 3\.2, 3\.3, (\d+\.\d+)",
    ),
    (
        "packages/luxar-viewer/src/data/codecs/README.md",
        r"Standalone gsplat format spec \(v(\d+\.\d+)\)",
    ),
    (
        "packages/luxar/src/luxar/core/group/lod/README.md",
        r"v(\d+\.\d+) node-tree (?:grammar|format)",
    ),
    (
        "packages/luxar/src/luxar/gsplats/lod/README.md",
        r"# v(\d+\.\d+) (?:leaf|kind=lod group)",
    ),
    (
        "packages/luxar/src/luxar/gsplats/lod/README.md",
        r"Output is written as a v(\d+\.\d+) `\.gsplats\.zarr` node tree",
    ),
    (
        "packages/luxar/src/luxar/gsplats/lod/README.md",
        r"on-disk container is a v(\d+\.\d+)",
    ),
    (
        "packages/luxar/src/luxar/core/group/gsplats_pipeline/README.md",
        r"v(\d+\.\d+) node tree grafted into the scene",
    ),
    (
        "packages/luxar/src/luxar/core/group/gsplats_pipeline/README.md",
        r"v(\d+\.\d+) node-tree format",
    ),
    (
        "scripts/reencode_gsplat_demos.py",
        r"format v(\d+\.\d+) \+ modern quantization",
    ),
    (
        "scripts/reencode_gsplat_demos.py",
        r"AUTO/v(\d+\.\d+), rebuilding its ladder",
    ),
    (
        "packages/luxar/src/luxar/gsplats/io/tests/test_format.py",
        r'Tests for the v(\d+\.\d+) ``\.gsplats\.zarr``',
    ),
    (
        "packages/luxar/src/luxar/gsplats/io/tests/test_format.py",
        r'``format_version`` = ``"(\d+\.\d+)"``',
    ),
    (
        "packages/luxar-viewer/src/data/scene-loader/lifecycle/load-scene.ts",
        r"v3\.0–v(\d+\.\d+) are all\s+// readable",
    ),
    (
        "docs/specs/GSPLATS_ZARR_FORMAT.md",
        r"The current format is \*\*v(\d+\.\d+)\*\*",
    ),
)


@pytest.mark.skipif(
    not (REPO_ROOT / "docs").is_dir(),
    reason="docs/ not present (packaged install without the repo tree)",
)
@pytest.mark.parametrize(("relpath", "pattern"), CURRENT_VERSION_CLAIMS)
def test_docs_state_the_current_gsplats_version(relpath: str, pattern: str) -> None:
    """Every published "current version" claim must equal the contract."""
    import re

    path = REPO_ROOT / relpath
    assert path.exists(), f"{relpath} is gone — update CURRENT_VERSION_CLAIMS"

    matches = re.findall(pattern, path.read_text())

    # An anchor that stops matching is the failure mode that matters: the claim
    # is still on the page, the gate silently stops reading it, and the next
    # bump goes N-1-of-N again with a green tick.
    assert matches, (
        f"{relpath}: the anchor {pattern!r} matched nothing. The wording moved; "
        "re-anchor it rather than deleting the entry, or this surface stops "
        "being checked."
    )
    stale = [v for v in matches if v != fc.GSPLATS_FORMAT_VERSION]
    assert not stale, (
        f"{relpath} claims gsplats format {stale} but the contract says "
        f"{fc.GSPLATS_FORMAT_VERSION!r}. Bump the docs, not the constant."
    )


def test_contract_sets_are_nonempty_and_unique() -> None:
    """A malformed contract (dupes / empties) should not slip through codegen."""
    for values in (
        fc.NODE_TYPES,
        fc.NODE_KINDS,
        fc.ENCODING_NAMES,
        fc.ATTR_KEYS,
        fc.ARRAY_NAMES,
        fc.SUPPORTED_SCENE_VERSIONS,
        fc.SUPPORTED_GSPLATS_VERSIONS,
    ):
        assert values, "contract sequence must not be empty"
        assert len(values) == len(set(values)), "contract sequence has duplicates"


@pytest.mark.skipif(
    not GEN_SCRIPT.exists(),
    reason="generator script not present (packaged install without repo scripts/)",
)
def test_committed_projections_match_contract() -> None:
    """The same drift gate CI runs: committed files must match the YAML."""
    argv = sys.argv[:]
    sys.argv = [str(GEN_SCRIPT), "--check"]
    try:
        with pytest.raises(SystemExit) as exc:
            runpy.run_path(str(GEN_SCRIPT), run_name="__main__")
    finally:
        sys.argv = argv
    assert exc.value.code == 0, (
        "format-contract projections are stale — run `make gen-contract` "
        "and commit the regenerated files"
    )


@pytest.mark.skipif(
    not GEN_SCRIPT.exists(),
    reason="generator script not present (packaged install without repo scripts/)",
)
def test_drift_gate_detects_stale_projection(monkeypatch: pytest.MonkeyPatch) -> None:
    """A mutated contract must make ``--check`` fail (the gate actually gates).

    Loads the generator as a module and monkeypatches ``load_contract`` to return
    a tampered contract, so the regenerated output no longer matches the
    committed files. Writes nothing (``--check`` is read-only), so the committed
    projections are untouched.
    """
    import importlib.util

    spec = importlib.util.spec_from_file_location("gen_format_contract", GEN_SCRIPT)
    assert spec is not None and spec.loader is not None
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)

    good = mod.load_contract()
    tampered = {**good, "gsplats_format": {"current": "9.9", "supported": ["9.9"]}}
    monkeypatch.setattr(mod, "load_contract", lambda: tampered)
    monkeypatch.setattr(sys, "argv", [str(GEN_SCRIPT), "--check"])

    assert mod.main() == 1, "drift gate should return non-zero on a stale projection"


def test_geometry_types_are_a_subset_of_node_types() -> None:
    """``geometry_types`` names the LEAF subset of ``node_types``.

    A geometry type that is not also a node type would be unrepresentable on
    disk (the writer stamps ``type`` from this vocabulary), so the containment
    is an invariant of the contract itself, not a convention.
    """
    assert set(fc.GEOMETRY_TYPES) <= set(fc.NODE_TYPES)
    assert fc.GEOMETRY_TYPES, "geometry sequence must not be empty"
    assert len(fc.GEOMETRY_TYPES) == len(set(fc.GEOMETRY_TYPES))


def test_geometry_types_exclude_containers() -> None:
    """The container node types are deliberately NOT geometry types."""
    assert "scene" not in fc.GEOMETRY_TYPES
    assert "group" not in fc.GEOMETRY_TYPES


@pytest.mark.skipif(
    not GEN_SCRIPT.exists(),
    reason="generator script not present (packaged install without repo scripts/)",
)
def test_generator_rejects_geometry_type_missing_from_node_types(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """The containment invariant is ENFORCED by codegen, not just asserted here.

    Without the check in ``_geometry_types`` a contract naming a geometry type
    absent from ``node_types`` would generate happily and only break far away,
    at write time.
    """
    import importlib.util

    spec = importlib.util.spec_from_file_location("gen_format_contract", GEN_SCRIPT)
    assert spec is not None and spec.loader is not None
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)

    good = mod.load_contract()
    tampered = {**good, "geometry_types": [*good["geometry_types"], "not_a_node_type"]}

    with pytest.raises(SystemExit) as exc:
        mod._geometry_types(tampered)
    assert "not_a_node_type" in str(exc.value)


@pytest.mark.skipif(
    not GEN_SCRIPT.exists(),
    reason="generator script not present (packaged install without repo scripts/)",
)
@pytest.mark.parametrize(
    ("geometry_types", "expected"),
    [
        pytest.param([], "must not be empty", id="empty"),
        pytest.param(["points", "group"], "container node types", id="container-group"),
        pytest.param(["points", "scene"], "container node types", id="container-scene"),
        pytest.param(["points", "points"], "duplicate", id="duplicates"),
    ],
)
def test_generator_rejects_malformed_geometry_types(
    geometry_types: list, expected: str
) -> None:
    """Each rule the ``_geometry_types`` docstring claims is actually enforced.

    Without these the generator emits broken or misleading projections rather
    than failing: an empty list renders ``Literal[]`` / ``export type X = ;``
    (syntax errors in both languages), a container type reaches per-geometry
    dispatch with no loader behind it, and a duplicate widens the tuple while
    leaving the union unchanged.
    """
    import importlib.util

    spec = importlib.util.spec_from_file_location("gen_format_contract", GEN_SCRIPT)
    assert spec is not None and spec.loader is not None
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)

    tampered = {**mod.load_contract(), "geometry_types": geometry_types}
    with pytest.raises(SystemExit) as exc:
        mod._geometry_types(tampered)
    assert expected in str(exc.value)
