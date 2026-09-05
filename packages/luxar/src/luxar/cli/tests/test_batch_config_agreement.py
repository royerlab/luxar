"""`batch-fit run` and `batch-fit submit` must not drift apart.

Audit A2-03 / A4-07. The two commands plan the same job — the whole
discover/decompose/manifest half is literally shared through
:func:`plan_batch` — and differ only in who executes the tasks. But each
declared its own flat flag list, and each mapped that list onto
:mod:`planning`'s four config dataclasses in its own code. `run` forwarded 56
keywords to a callee with a byte-identical 56-parameter signature, then rebuilt
the four dataclasses field-for-field even though ``build_plan_configs`` already
existed and ``submit`` used it.

**That had already cost a flag.** ``--calibration-samples`` was declared on
``submit`` only, while the LOCAL path consumes it —
``run_orchestration._resolve_local_deferred_floor`` passes
``manifest.calibration_samples`` straight to ``calibrate_all_channels``. So a
local ``--denoise`` run with a deferred percentile floor always calibrated on 5
timepoints with no way to ask for more. Exactly the "works on submit, silently
ignored on run" shape.

The fix routes both commands through the one mapper. These tests keep it that
way: a second construction site, an unfed config field, or a default that
disagrees between the siblings all go red.
"""

from __future__ import annotations

import ast
import inspect
from pathlib import Path
from typing import Any

import pytest

from luxar.cli.gsplat_ops.batch import plan_configs, run_orchestration, submit
from luxar.cli.gsplat_ops.batch import run as run_mod
from luxar.cli.gsplat_ops.batch.planning import (
    ContentKnobs,
    DenoiseConfig,
    FitConfig,
    MergeConfig,
)

#: The four dataclasses `plan_batch` takes. Every field must be reachable from
#: the CLI, and every construction must go through the shared mapper.
PLAN_CONFIGS: tuple[type, ...] = (FitConfig, DenoiseConfig, ContentKnobs, MergeConfig)

#: `build_plan_configs` parameters a command may legitimately not expose, with
#: the reason. Anything not listed here must be a flag on BOTH commands.
ONE_SIDED: dict[str, str] = {
    "batch_preprocess": (
        "submit-only by design: writing denoised.zarr up front is a separate "
        "dependent Slurm job, and the local runner denoises per tile on the fly. "
        "planning.py's own refusal text points users at `batch-fit submit`."
    ),
}

#: Shared flag names whose defaults are allowed to differ, with the reason.
#: Empty, and meant to stay that way. `--gpus` was the one entry: it meant a
#: device SELECTOR on `run` (str, 'auto'|'all'|'cpu'|'0,1,3') and a COUNT of
#: GPUs per Slurm task on `submit` (int, 1), so `--gpus 2` meant two different
#: things and `--gpus auto` was a parse error on one of them. Submit's is now
#: `--gpus-per-task`, which is also the sbatch directive it generates.
ALLOWED_DEFAULT_DIVERGENCE: dict[str, str] = {}


def _module_tree(module: Any) -> ast.Module:
    return ast.parse(Path(inspect.getfile(module)).read_text(encoding="utf-8"))


def _func(module: Any, name: str) -> ast.FunctionDef:
    for node in ast.walk(_module_tree(module)):
        if isinstance(node, ast.FunctionDef) and node.name == name:
            return node
    raise AssertionError(f"{name} not found in {module.__name__}")


def _param_names(node: ast.FunctionDef) -> list[str]:
    a = node.args
    return [x.arg for x in a.posonlyargs + a.args + a.kwonlyargs]


def _calls(node: ast.AST, callee: str) -> list[dict[str, str]]:
    """Every `callee(...)` under `node`, as {keyword: unparsed value}."""
    return [
        {kw.arg: ast.unparse(kw.value) for kw in sub.keywords if kw.arg}
        for sub in ast.walk(node)
        if isinstance(sub, ast.Call)
        and isinstance(sub.func, ast.Name)
        and sub.func.id == callee
    ]


def _typer_options(node: ast.FunctionDef) -> dict[str, str]:
    """CLI parameter -> the unparsed default inside its `typer.Option(...)`."""
    a = node.args
    pos = a.posonlyargs + a.args
    pairs = list(zip(pos[len(pos) - len(a.defaults) :], a.defaults, strict=True))
    pairs += [(k, d) for k, d in zip(a.kwonlyargs, a.kw_defaults, strict=True) if d]
    out: dict[str, str] = {}
    for arg, default in pairs:
        if isinstance(default, ast.Call) and default.args:
            out[arg.arg] = ast.unparse(default.args[0])
    return out


RUN_CMD = _func(run_mod, "run_batch_run")
SUBMIT_CMD = _func(submit, "run_batch_submit")
MAPPER = _func(plan_configs, "build_plan_configs")
ORCH = _func(run_orchestration, "run_batch_local_orchestration")


def test_the_commands_were_found_and_are_not_empty() -> None:
    """Fail closed — an AST walk that found nothing makes every test vacuous."""
    assert len(_param_names(RUN_CMD)) > 40
    assert len(_param_names(SUBMIT_CMD)) > 40
    assert len(_param_names(MAPPER)) > 40
    assert all(len(_dataclass_fields(cls)) > 3 for cls in PLAN_CONFIGS)


def _dataclass_fields(cls: type) -> list[str]:
    import dataclasses

    return [f.name for f in dataclasses.fields(cls)]


@pytest.mark.parametrize("cls", PLAN_CONFIGS, ids=lambda c: c.__name__)
def test_the_shared_mapper_feeds_every_field_of_every_plan_config(cls: type) -> None:
    """A field the mapper never sets is unreachable from either command."""
    built = _calls(MAPPER, cls.__name__)
    assert len(built) == 1, f"{cls.__name__} built {len(built)} times in the mapper"
    unfed = sorted(set(_dataclass_fields(cls)) - set(built[0]))
    assert not unfed, (
        f"{cls.__name__} field(s) {unfed} are never set by build_plan_configs, so no "
        f"CLI flag can reach them. Either wire them up or drop them from the dataclass."
    )


@pytest.mark.parametrize("cls", PLAN_CONFIGS, ids=lambda c: c.__name__)
def test_each_plan_config_is_constructed_in_exactly_one_cli_place(cls: type) -> None:
    """The regression guard for A4-07: no second, hand-rolled construction site.

    `run_orchestration` used to inline all four of these field-for-field beside
    the mapper's copies. Both copies then had to be edited in step, and one of
    them was not.
    """
    batch_dir = Path(inspect.getfile(plan_configs)).parent
    sites = [
        path.name
        for path in sorted(batch_dir.glob("*.py"))
        if _calls(ast.parse(path.read_text(encoding="utf-8")), cls.__name__)
    ]
    assert sites == [Path(inspect.getfile(plan_configs)).name], (
        f"{cls.__name__} is constructed in {sites}. It must be built only by "
        f"build_plan_configs, which both `batch-fit run` and `batch-fit submit` "
        f"call — a second site is how --calibration-samples went missing on run."
    )


def test_both_commands_expose_every_flag_the_mapper_consumes() -> None:
    """The defect class, gated: a mapper input missing from one command.

    `--calibration-samples` reached `submit` and not `run` while the local
    denoise path consumed it. Anything genuinely one-sided belongs in
    `ONE_SIDED` with a reason, not in a silent gap.
    """
    run_params, submit_params = (
        set(_param_names(RUN_CMD)),
        set(_param_names(SUBMIT_CMD)),
    )
    missing: list[str] = []
    for name in _param_names(MAPPER):
        if name in ONE_SIDED:
            continue
        for cmd, params in (("run", run_params), ("submit", submit_params)):
            if name not in params:
                missing.append(f"{name} (absent from `batch-fit {cmd}`)")
    assert not missing, (
        "build_plan_configs consumes flags one command cannot supply, so they "
        "silently take the dataclass default there:\n  " + "\n  ".join(missing)
    )


def test_the_siblings_agree_on_the_defaults_of_their_shared_flags() -> None:
    """A shared flag name that means two things is worse than two names."""
    run_opts, submit_opts = _typer_options(RUN_CMD), _typer_options(SUBMIT_CMD)
    disagree = [
        (name, run_opts[name], submit_opts[name])
        for name in sorted(set(run_opts) & set(submit_opts))
        if run_opts[name] != submit_opts[name]
        and name not in ALLOWED_DEFAULT_DIVERGENCE
    ]
    assert not disagree, "\n".join(
        [f"{n}: run={r}, submit={s}" for n, r, s in disagree]
        + ["Add to ALLOWED_DEFAULT_DIVERGENCE with a reason if deliberate."]
    )


def test_every_recorded_exception_is_still_a_real_one() -> None:
    """A stale allowance is a silent hole — shrink the lists, never grow them.

    Both dicts above suppress a check. If the underlying divergence is fixed or
    renamed away, the entry keeps suppressing something that no longer exists,
    and the next real one to appear under that name passes unnoticed.
    """
    run_opts, submit_opts = _typer_options(RUN_CMD), _typer_options(SUBMIT_CMD)
    # Two ways an entry goes stale, and the second is the one that bit: the
    # allowance survived `--gpus` being RENAMED on submit, because a name that
    # is no longer shared compares `"'auto'" != None` and read as still
    # diverging. An entry for a name only one command declares suppresses
    # nothing, so it is stale too.
    shared = set(run_opts) & set(submit_opts)
    stale = [
        name
        for name in ALLOWED_DEFAULT_DIVERGENCE
        if name not in shared or run_opts[name] == submit_opts[name]
    ]
    assert not stale, (
        f"ALLOWED_DEFAULT_DIVERGENCE entries {stale} no longer suppress anything — "
        f"the flag is either agreed on now or no longer declared by both commands. "
        f"Delete them."
    )

    run_params, submit_params = (
        set(_param_names(RUN_CMD)),
        set(_param_names(SUBMIT_CMD)),
    )
    mapper_params = set(_param_names(MAPPER))
    stale_one_sided = [
        name
        for name in ONE_SIDED
        if name not in mapper_params or (name in run_params and name in submit_params)
    ]
    assert not stale_one_sided, (
        f"ONE_SIDED entries {stale_one_sided} are either no longer consumed by the "
        f"mapper or are now on both commands. Delete them."
    )


def test_calibration_samples_reaches_the_local_plan(monkeypatch, tmp_path) -> None:
    """The defect itself, end to end through the real CLI.

    `run_orchestration._resolve_local_deferred_floor` hands
    `manifest.calibration_samples` to `calibrate_all_channels`, so this value is
    live on the local path — it just had no flag. Pre-fix this asserts 5 no
    matter what is passed, because `--calibration-samples` did not parse.
    """
    import numpy as np
    import zarr
    from typer.testing import CliRunner

    from luxar.cli.gsplat_commands import app_gsplat

    src = tmp_path / "vol.zarr"
    array = zarr.open_array(
        str(src), mode="w", shape=(32, 32, 32), chunks=(16, 16, 16), dtype="f4"
    )
    array[:] = np.float32(0.5)

    seen: dict[str, Any] = {}

    def _capture(**kwargs: Any) -> None:
        seen.update(kwargs)

    monkeypatch.setattr(run_mod, "run_batch_local_orchestration", _capture)
    result = CliRunner().invoke(
        app_gsplat,
        [
            "batch-fit",
            "run",
            str(src),
            str(tmp_path / "out"),
            "--axes",
            "z,y,x",
            "--tile-size",
            "64",
            "--gpus",
            "cpu",
            "--denoise",
            "--calibration-samples",
            "9",
            "--dry-run",
        ],
    )

    assert result.exit_code == 0, result.output
    assert seen["cfgs"].denoise.calibration_samples == 9
    assert seen["cfgs"].denoise.denoise is True
    # The other half of the pair stays submit-only, deliberately.
    assert seen["cfgs"].denoise.preprocess is False


def test_gpus_per_task_reaches_the_sbatch_directive(monkeypatch, tmp_path) -> None:
    """The rename, end to end: `--gpus-per-task N` -> `#SBATCH --gpus-per-task=N`.

    `--gpus` used to mean a COUNT here and a device SELECTOR on `batch-fit run`,
    so `--gpus 2` meant two different things one command apart. The new spelling
    is the directive the manifest ends up producing, which is what this pins —
    a rename that stopped short of `slurm_gen` would leave the flag inert.
    """
    import numpy as np
    import zarr
    from typer.testing import CliRunner

    from luxar.cli.gsplat_commands import app_gsplat
    from luxar.gsplats.batch import slurm_gen

    src = tmp_path / "vol.zarr"
    array = zarr.open_array(
        str(src), mode="w", shape=(32, 32, 32), chunks=(16, 16, 16), dtype="f4"
    )
    array[:] = np.float32(0.5)

    # Link 1: the flag parses and arrives at the manifest stamper as a count.
    seen: dict[str, Any] = {}
    real_stamp = submit.stamp_slurm_fields

    def _capture(manifest_arg: Any, **kwargs: Any) -> Any:
        seen.update(kwargs)
        seen["manifest"] = manifest_arg
        return real_stamp(manifest_arg, **kwargs)

    monkeypatch.setattr(submit, "stamp_slurm_fields", _capture)
    result = CliRunner().invoke(
        app_gsplat,
        [
            "batch-fit",
            "submit",
            str(src),
            str(tmp_path / "out"),
            "--axes",
            "z,y,x",
            "--tile-size",
            "64",
            "-p",
            "gpu",
            "--gpus-per-task",
            "3",
            "--dry-run",
        ],
    )
    assert result.exit_code == 0, result.output
    assert seen["gpus_per_task"] == 3
    assert seen["manifest"].slurm_gpus == 3

    # Link 2: the manifest field is what the sbatch directive is rendered from.
    script = slurm_gen.generate_fit_sbatch(seen["manifest"], "")
    assert "#SBATCH --gpus-per-task=3" in script


def test_run_no_longer_restates_the_config_schema_to_its_orchestrator() -> None:
    """A4-07's structural claim, pinned.

    `run.py` used to forward 56 keywords into a byte-identical 56-parameter
    signature. It now hands over the four assembled config objects, so the
    orchestrator shares no parameter name with the mapper at all.
    """
    overlap = sorted(set(_param_names(ORCH)) & set(_param_names(MAPPER)))
    assert not overlap, (
        f"run_batch_local_orchestration re-declares mapper parameters {overlap}. "
        f"It should take the assembled `cfgs`, not the flat flags again."
    )
    forwarded = _calls(RUN_CMD, "run_batch_local_orchestration")
    assert len(forwarded) == 1
    assert "cfgs" in forwarded[0]
    assert len(forwarded[0]) < 20, (
        f"the forward is back up to {len(forwarded[0])} keywords; it should pass "
        f"`cfgs` plus the execution-only flags."
    )
