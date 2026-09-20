# fit_command.py
"""Build the per-task ``luxar gsplat fit`` command for a batch job.

Two consumers share this:

- :func:`build_task_fit_argv` returns a concrete ``list[str]`` argv for one
  ``(t, c, slot)`` task — used by the LOCAL runner to spawn worker subprocesses.
- :func:`fit_args_to_tokens` is the one non-trivial shared bit (the
  ``fit_args`` dict → CLI flag expansion); the Slurm bash generator
  (:func:`luxar.gsplats.batch.slurm_gen.generate_fit_sbatch`) imports it too so
  the local and Slurm fit commands cannot drift.

The Slurm generator emits a bash *template* (``$K``/``$C``/``$T`` placeholders);
this module emits a *concrete* argv from a :class:`BatchJob`'s real indices.
They share the flag set and ordering decisions but not the value substitution.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any, Dict, Iterator, Optional, Tuple

from luxar.gsplats.batch.manifest import (
    BatchJob,
    BatchManifest,
    tile_local_read_plan,
)


def _tile_local_read_args(manifest: BatchManifest, job: BatchJob) -> list[str]:
    """Worker metadata that replaces its full-frame tile scan and read."""
    plan = tile_local_read_plan(manifest)
    if plan is None:
        return []
    if not 0 <= job.tile_index < len(plan.regions):
        raise ValueError(
            f"tile index {job.tile_index} is outside 0..{len(plan.regions) - 1}"
        )
    args = [
        "--tile-region",
        plan.regions[job.tile_index],
        "--tile-volume-shape",
        plan.volume_shape,
    ]
    if plan.nonempty_counts is not None:
        row_index = job.task_id // manifest.n_tiles
        if not 0 <= row_index < len(plan.nonempty_counts):
            raise ValueError(
                f"task {job.task_id} maps to missing tile-local count row {row_index}"
            )
        args += ["--tile-nonempty-count", str(plan.nonempty_counts[row_index])]
    return args


def iter_fit_arg_flags(fit_args: Dict[str, Any]) -> Iterator[Tuple[str, Optional[str]]]:
    """Yield ``(flag, value_or_None)`` for each ``fit_args`` entry.

    The single source of truth for the ``fit_args`` dict -> CLI flag mapping,
    shared by the concrete argv (:func:`fit_args_to_tokens` /
    :func:`build_task_fit_argv`) and the Slurm bash template
    (:func:`generate_fit_sbatch`), so the two cannot drift.  An empty-string
    value is a boolean flag (``value`` is ``None``); a ``None`` value is skipped.
    Keys map to flags by ``_`` -> ``-``.
    """
    for key, value in fit_args.items():
        if value is None:
            continue
        flag = f"--{key.replace('_', '-')}"
        yield (flag, None if value == "" else str(value))


def fit_args_to_tokens(fit_args: Dict[str, Any]) -> list[str]:
    """Expand a ``fit_args`` dict to a flat argv token list.

    ``{"seeds": "8000", "progressive": ""}`` -> ``["--seeds", "8000",
    "--progressive"]``.
    """
    toks: list[str] = []
    for flag, value in iter_fit_arg_flags(fit_args):
        toks.append(flag)
        if value is not None:
            toks.append(value)
    return toks


def build_task_fit_argv(
    manifest: BatchManifest,
    job: BatchJob,
    out_path: str | Path,
    *,
    argv0: Optional[list[str]] = None,
    denoise_h: Optional[float] = None,
) -> list[str]:
    """Concrete ``luxar gsplat fit`` argv for one batch task.

    The pure-python twin of the Slurm fit-command template
    (:func:`generate_fit_sbatch`), built from a job's *real* dataset indices:

    - uniform mode -> ``--tile {k}/{M} --tile-size … --overlap …``
    - content mode -> ``--tiling content --plan {plan} --plan-box {k}``

    ``--channel`` / ``--timepoint`` are emitted with the SAME gating as the bash
    generator (when there are multiple values or slicing selected specific
    indices), using ``job.channel`` / ``job.timepoint`` (already real indices).

    Device is NOT passed here: the local runner pins each GPU worker via
    ``CUDA_VISIBLE_DEVICES`` so it sees its card as ``cuda:0`` (sidestepping
    un-validated ``cuda:N`` strings). The runner appends ``--device cpu`` for
    the CPU sentinel so device auto-selection cannot fall through to MPS.

    Parameters
    ----------
    out_path
        Where the worker writes (the caller typically passes a ``…tmp`` path and
        does the atomic rename itself).
    argv0
        CLI prefix; defaults to :func:`luxar_argv0` (the active ``luxar`` script
        or ``python -m luxar``).
    denoise_h
        Pre-calibrated NLM ``h`` for ``job.channel`` (resolved by the caller from
        ``manifest.denoise_h_values``); appended as ``--denoise-h`` only for
        on-the-fly auto-calibrated denoise.
    """
    if argv0 is None:
        from luxar.gsplats.fit_tiled_parallel import luxar_argv0

        argv0 = luxar_argv0()

    cmd: list[str] = [*argv0, "gsplat", "fit", manifest.input_path, str(out_path)]

    if manifest.mode == "content":
        cmd += [
            "--tiling",
            "content",
            "--plan",
            manifest.plan_path or "",
            "--plan-box",
            str(job.tile_index),
        ]
    else:
        cmd += [
            "--tile",
            f"{job.tile_index}/{manifest.n_tiles}",
            "--tile-size",
            str(manifest.tile_size),
            "--overlap",
            str(manifest.tile_overlap),
            # A tile wholly below the run's background floor legitimately
            # fits 0 splats; the writer rejects empty stores, so the worker
            # must write an `.empty` marker and exit 0 (the runner finalizes
            # it and the merge skips it) instead of failing the task forever.
            "--allow-empty-tile",
        ]
        cmd += _tile_local_read_args(manifest, job)

    if manifest.array_key is not None:
        cmd += ["--array-key", manifest.array_key]
    # Same gating as slurm_gen: emit when there are multiple values OR slicing
    # selected specific indices (job.channel/timepoint are real dataset indices).
    if manifest.n_channels > 1 or manifest.channel_indices is not None:
        cmd += ["--channel", str(job.channel)]
    if manifest.n_timepoints > 1 or manifest.timepoint_indices is not None:
        cmd += ["--timepoint", str(job.timepoint)]
    if manifest.preset:
        cmd += ["--preset", manifest.preset]

    cmd += fit_args_to_tokens(manifest.fit_args)

    if manifest.axes:
        # Workers re-load the full store per task; without --axes they fall back
        # to the positional heuristic and may load a differently-ordered volume
        # than the planner used (corrupt tile grid / merge).
        cmd += ["--axes", manifest.axes]

    if (
        manifest.denoise
        and manifest.denoise_mode == "on-the-fly"
        and manifest.denoise_h is None
        and denoise_h is not None
    ):
        cmd += ["--denoise-h", str(denoise_h)]

    return cmd
