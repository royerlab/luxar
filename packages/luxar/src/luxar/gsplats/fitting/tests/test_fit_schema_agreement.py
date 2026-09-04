"""The fit parameter schema is restated in several places; they must agree.

Audit A2-02. `fit_gaussian_splats` is the documented entry point, and its
parameter set is restated by `prepare_fit_config` (43 shared names) and by the
four config dataclasses users are told to unpack into it. Every restatement is a
chance for a default to drift, and adding a knob means editing several places
with nothing checking that you edited them all.

**It had already drifted, six times.** The audit found one
(`asymmetric_penalty`); an AST comparison of every shared name found five in
`prepare_fit_config` plus one in `FitConfig`:

    asymmetric_penalty   10.0  vs 1.0     patience             25    vs 15
    gradient_clip        1.0   vs None    max_eccentricity     None  vs 10.0
    lr_reduction_factor  0.98  vs 0.9     FitConfig.max_eccentricity  None vs 10.0

`prepare_fit_config` is public API (`gsplats.fitting.__all__`), and the three
bundle dataclasses are documented as `fit_gaussian_splats(volume, **asdict(cfg))`
— so every one of those was a divergent default an external caller actually got.

This test is the gate. Collapsing the signatures into a single threaded
`FitConfig` would remove the restatement outright, but it is a breaking change
across ~50 call sites; making the restatement *unable to drift* is the part that
fixes the defect, and it costs nothing.
"""

from __future__ import annotations

import ast
import inspect
from pathlib import Path
from typing import Any

import pytest

from luxar.gsplats import fit_gaussian_splats
from luxar.gsplats.fitting import prepare_fit_config
from luxar.gsplats.fitting.config import (
    ConstraintConfig,
    FitConfig,
    LossConfig,
    OptimConfig,
)

#: Sites that restate `fit_gaussian_splats`' parameters and must not disagree.
RESTATEMENTS: dict[str, Any] = {
    "prepare_fit_config": prepare_fit_config,
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

        return {
            field.name: field.default
            for field in dataclasses.fields(obj)
            if field.default is not dataclasses.MISSING
        }
    return {
        name: param.default
        for name, param in inspect.signature(obj).parameters.items()
        if param.default is not inspect.Parameter.empty
    }


ENTRY_DEFAULTS = _defaults(fit_gaussian_splats)


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


def test_the_bundle_dataclasses_are_a_no_op_when_unpacked() -> None:
    """Their published usage is `fit_gaussian_splats(volume, **asdict(cfg))`.

    A default bundle must therefore change nothing. This is the same property the
    test above checks, asserted the way the docstrings tell users to invoke it —
    so the promise and the check cannot drift apart either.
    """
    import dataclasses

    for bundle in (OptimConfig, LossConfig, ConstraintConfig):
        for name, value in dataclasses.asdict(bundle()).items():
            if name in ENTRY_DEFAULTS:
                assert value == ENTRY_DEFAULTS[name], (
                    f"{bundle.__name__}().{name} == {value!r}, but unpacking it into "
                    f"fit_gaussian_splats would override that function's own default "
                    f"of {ENTRY_DEFAULTS[name]!r}. A DEFAULT bundle must be a no-op."
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


def test_prepare_fit_config_restates_the_config_schema_and_little_else() -> None:
    """A2-02's structural claim, pinned — and corrected.

    The audit says "44 of prepare_fit_config's 45 params are literally FitConfig
    field names". Measured, that holds against the UNION of the four config
    dataclasses, not against `FitConfig` alone: `FitConfig` carries 24 of the 43
    defaulted parameters, and the optimiser/loss/constraint knobs live in the
    three bundles. Stated against the union so the number means what it says.

    Pinned because a parameter appearing here that is in NO config class means
    the flat schema and the config objects have begun to diverge in SHAPE, not
    just in defaults — a different and worse problem than drift.
    """
    union: set[str] = set()
    for cls in (FitConfig, OptimConfig, LossConfig, ConstraintConfig):
        union |= _field_names(cls)
    # `fitter` and `V` are positional with no default, so `_defaults` omits them.
    strays = sorted(set(_defaults(prepare_fit_config)) - union)
    assert len(strays) <= 1, (
        f"prepare_fit_config declares {len(strays)} parameters that are in none of "
        f"the four config dataclasses: {strays}"
    )


def test_source_defaults_match_the_imported_ones() -> None:
    """Guard against the comparison being fooled by import-time rebinding.

    Everything above reads defaults through `inspect`, which sees the objects as
    imported. Reading them straight from the source AST is an independent route
    to the same answer; if the two disagree, something is rewriting defaults at
    import time and neither this test nor the ones above mean what they say.
    """
    source = Path(inspect.getfile(prepare_fit_config)).read_text(encoding="utf-8")
    tree = ast.parse(source)
    node = next(
        n
        for n in ast.walk(tree)
        if isinstance(n, ast.FunctionDef) and n.name == "prepare_fit_config"
    )
    args = node.args
    positional = args.posonlyargs + args.args
    from_source = {
        arg.arg: ast.literal_eval(default)
        # strict=True: the slice is sized to match, so a mismatch means the
        # arithmetic is wrong -- exactly what should raise rather than
        # silently compare a prefix.
        for arg, default in zip(
            positional[len(positional) - len(args.defaults) :],
            args.defaults,
            strict=True,
        )
        if isinstance(default, ast.Constant)
    }
    imported = _defaults(prepare_fit_config)
    for name, value in from_source.items():
        assert imported[name] == value, (
            f"prepare_fit_config.{name} is {value!r} in the source but "
            f"{imported[name]!r} once imported"
        )
