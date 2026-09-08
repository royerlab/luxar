"""The public fit signature and its configuration dataclasses must agree.

Audit A2-02. `fit_gaussian_splats` is the documented entry point. Its raw
parameters are threaded through the internal pipeline in `FitParameters`, then
normalized into `FitConfig`; the three declarative config bundles expose useful
subsets. Every dataclass default must agree with the entry point.

**It had already drifted, six times.** The audit found one
(`asymmetric_penalty`); an AST comparison of every shared name found five in
`prepare_fit_config` plus one in `FitConfig`:

    asymmetric_penalty   10.0  vs 1.0     patience             25    vs 15
    gradient_clip        1.0   vs None    max_eccentricity     None  vs 10.0
    lr_reduction_factor  0.98  vs 0.9     FitConfig.max_eccentricity  None vs 10.0

The three bundle dataclasses are documented as
`fit_gaussian_splats(volume, **asdict(cfg))`, so every one of those was a
divergent default an external caller actually got. This test is the gate for the
remaining public-signature-to-dataclass restatements.
"""

from __future__ import annotations

import ast
import inspect
from pathlib import Path
from typing import Any

import pytest

from luxar.gsplats import GaussianSplatFitter, fit_gaussian_splats
from luxar.gsplats.fit_gsplats import _collect_fit_parameters
from luxar.gsplats.fitting import prepare_fit_config
from luxar.gsplats.fitting.config import (
    ConstraintConfig,
    FitConfig,
    FitParameters,
    LossConfig,
    OptimConfig,
)

#: Sites that restate `fit_gaussian_splats`' parameters and must not disagree.
RESTATEMENTS: dict[str, Any] = {
    "FitParameters": FitParameters,
    "OptimConfig": OptimConfig,
    "LossConfig": LossConfig,
    "ConstraintConfig": ConstraintConfig,
    "FitConfig": FitConfig,
}

#: Names a restatement may legitimately spell differently, with the reason.
#: Empty, and meant to stay that way — an entry here is a documented exception,
#: not a place to put a new drift.
ALLOWED_DIVERGENCE: dict[tuple[str, str], str] = {}


def _defaults(obj: Any) -> dict[str, Any]:
    """Parameter or field name -> default VALUE (not its source spelling).

    Compared by value rather than by source text so that `1.0` and `1` do not
    read as a disagreement, and so a default moved into a module constant still
    compares equal to the literal it replaced.
    """
    if isinstance(obj, type):
        import dataclasses

        defaults = {}
        for field in dataclasses.fields(obj):
            if field.default is not dataclasses.MISSING:
                defaults[field.name] = field.default
            elif field.default_factory is not dataclasses.MISSING:
                defaults[field.name] = field.default_factory()
        return defaults
    return {
        name: param.default
        for name, param in inspect.signature(obj).parameters.items()
        if param.default is not inspect.Parameter.empty
    }


ENTRY_DEFAULTS = _defaults(fit_gaussian_splats)


def test_internal_fit_hops_accept_one_parameter_object() -> None:
    """The raw fit schema must not be restated by either internal hop."""
    assert list(inspect.signature(GaussianSplatFitter.fit).parameters) == [
        "self",
        "parameters",
    ]
    assert list(inspect.signature(prepare_fit_config).parameters) == [
        "fitter",
        "parameters",
    ]


def test_fit_parameters_are_exported_with_the_fitter() -> None:
    """The parameter type required by the class API must share its import path."""
    import luxar.gsplats as gsplats

    assert gsplats.FitParameters is FitParameters


def test_fit_parameters_use_identity_equality_and_hashing() -> None:
    """Array-valued parameters must not synthesize broken value comparison."""
    import numpy as np

    left = FitParameters(V=np.zeros((2, 2), dtype=np.float32))
    right = FitParameters(V=np.zeros((2, 2), dtype=np.float32))

    assert left == left
    assert left != right
    assert isinstance(hash(left), int)


def test_public_call_arguments_are_collected_by_name() -> None:
    """Every raw parameter must retain its value entering the internal bundle."""
    values = {
        name: object()
        for name in inspect.signature(fit_gaussian_splats).parameters
        if name in _field_names(FitParameters)
    }
    parameters = _collect_fit_parameters(values)
    for name, value in values.items():
        assert getattr(parameters, name) is value


def test_dataclass_default_factories_are_included() -> None:
    """A future mutable config default must remain part of the comparison."""
    import dataclasses

    @dataclasses.dataclass
    class FactoryConfig:
        values: list[int] = dataclasses.field(default_factory=list)

    assert _defaults(FactoryConfig) == {"values": []}


def _shared(site: Any) -> list[tuple[str, Any, Any]]:
    """`(name, theirs, ours)` for every name this site shares with the entry point."""
    return [
        (name, default, ENTRY_DEFAULTS[name])
        for name, default in _defaults(site).items()
        if name in ENTRY_DEFAULTS
    ]


def test_the_entry_point_has_a_schema_to_compare_against() -> None:
    """Fail closed: an empty signature makes every test below vacuously true."""
    assert len(ENTRY_DEFAULTS) > 40, (
        f"fit_gaussian_splats declares only {len(ENTRY_DEFAULTS)} defaulted "
        f"parameters — the schema comparison would be checking almost nothing."
    )


@pytest.mark.parametrize("site_name", sorted(RESTATEMENTS))
def test_restatement_shares_enough_to_be_worth_checking(site_name: str) -> None:
    """A site that shares nothing is a renamed or moved schema, not a passing one."""
    shared = _shared(RESTATEMENTS[site_name])
    assert shared, (
        f"{site_name} shares no parameter names with fit_gaussian_splats. Either it "
        f"stopped restating the schema (remove it from RESTATEMENTS) or the names "
        f"were renamed on one side only."
    )


@pytest.mark.parametrize("site_name", sorted(RESTATEMENTS))
def test_no_default_disagrees_with_the_entry_point(site_name: str) -> None:
    """Every shared name must default to the same value at every site."""
    disagreements = [
        (name, theirs, ours)
        for name, theirs, ours in _shared(RESTATEMENTS[site_name])
        if theirs != ours and (site_name, name) not in ALLOWED_DIVERGENCE
    ]
    assert not disagreements, "\n".join(
        [
            f"{site_name} disagrees with fit_gaussian_splats on "
            f"{len(disagreements)} default(s):"
        ]
        + [
            f"  {name}: {site_name} says {theirs!r}, fit_gaussian_splats says {ours!r}"
            for name, theirs, ours in disagreements
        ]
        + [
            "",
            "fit_gaussian_splats is the documented entry point, so it is the one to "
            "match. If a site genuinely needs a different default, add it to "
            "ALLOWED_DIVERGENCE with the reason.",
        ]
    )


def _field_names(cls: type) -> set[str]:
    """Every dataclass field name, including ones with no default."""
    import dataclasses

    return {field.name for field in dataclasses.fields(cls)}


def test_the_two_FitConfig_classes_are_not_confused() -> None:
    """There are two classes named `FitConfig`; this suite means the fit one.

    `cli.gsplat_ops.batch.planning.FitConfig` is the batch-run config and shares
    the name but not the schema. A test that imported the wrong one would pass
    while checking nothing relevant.
    """
    from luxar.cli.gsplat_ops.batch.planning import FitConfig as BatchFitConfig

    assert FitConfig is not BatchFitConfig
    assert "max_eccentricity" in _field_names(FitConfig)
    assert "max_eccentricity" not in _field_names(BatchFitConfig)


def test_fit_parameters_omit_only_parameters_handled_elsewhere() -> None:
    """Every fit input not handled by the wrapper belongs in FitParameters."""
    entry_only = set(inspect.signature(fit_gaussian_splats).parameters) - _field_names(
        FitParameters
    )
    expected = {
        "cull_retention",
        "device",
        "dynamic_config",
        "enable_dynamic_ops",
        "use_cuda",
        "use_metal",
    }
    assert entry_only == expected, (
        "fit_gaussian_splats parameters absent from FitParameters changed: "
        f"expected {sorted(expected)}, got {sorted(entry_only)}. A new omission may "
        "be handled by neither the fitter nor the wrapper."
    )


def test_fit_parameters_match_the_normalized_config_shape() -> None:
    """A new raw fit knob must also enter the normalized config schema."""
    assert _field_names(FitParameters) - _field_names(FitConfig) == set()


def test_prepare_fit_config_reads_every_fit_parameter() -> None:
    """A field added to both schemas must still be wired through validation."""
    source = Path(inspect.getfile(prepare_fit_config)).read_text(encoding="utf-8")
    tree = ast.parse(source)
    function = next(
        node
        for node in ast.walk(tree)
        if isinstance(node, ast.FunctionDef)
        and node.name == prepare_fit_config.__name__
    )
    parameter_reads = {
        node.attr
        for node in ast.walk(function)
        if isinstance(node, ast.Attribute)
        and isinstance(node.value, ast.Name)
        and node.value.id == "parameters"
    }

    assert parameter_reads == _field_names(FitParameters)


def _source_literal_defaults(obj: Any) -> dict[str, Any]:
    """Read literal positional and keyword-only defaults from a function's AST."""
    unwrapped = inspect.unwrap(obj)
    source = Path(inspect.getfile(unwrapped)).read_text(encoding="utf-8")
    tree = ast.parse(source)
    node = next(
        n
        for n in ast.walk(tree)
        if isinstance(n, ast.FunctionDef) and n.name == unwrapped.__name__
    )
    args = node.args
    positional = args.posonlyargs + args.args
    positional_pairs = (
        list(zip(positional[-len(args.defaults) :], args.defaults, strict=True))
        if args.defaults
        else []
    )
    keyword_only_pairs = [
        (arg, default)
        for arg, default in zip(args.kwonlyargs, args.kw_defaults, strict=True)
        if default is not None
    ]
    defaults = {}
    for arg, default in positional_pairs + keyword_only_pairs:
        try:
            defaults[arg.arg] = ast.literal_eval(default)
        except (TypeError, ValueError):
            pass
    return defaults


def _keyword_only_default_fixture(*, value: int = 7) -> None:
    pass


def test_source_defaults_include_keyword_only_parameters() -> None:
    """Keyword-only defaults must be visible to the independent AST route."""
    assert _source_literal_defaults(_keyword_only_default_fixture) == {"value": 7}


@pytest.mark.parametrize("obj", [fit_gaussian_splats], ids=lambda obj: obj.__name__)
def test_source_defaults_match_the_imported_ones(obj: Any) -> None:
    """Guard against the comparison being fooled by import-time rebinding.

    Everything above reads defaults through `inspect`, which sees the objects as
    imported. Reading them straight from the source AST is an independent route
    to the same answer; if the two disagree, something is rewriting defaults at
    import time and neither this test nor the ones above mean what they say.
    """
    from_source = _source_literal_defaults(obj)
    imported = _defaults(obj)
    for name, value in from_source.items():
        assert imported[name] == value, (
            f"{obj.__name__}.{name} is {value!r} in the source but "
            f"{imported[name]!r} once imported"
        )
