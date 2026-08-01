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
    }
    assert named == set(fc.NODE_TYPES)


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
