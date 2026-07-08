# fit_tiled_parallel.py
"""Parallel tiled fitting: spawn one subprocess per tile, then merge.

Drives multiple ``luxar gsplat fit ... --tile i/M`` worker processes
concurrently on a single GPU — each worker owns its own CUDA context and they
share the GPU memory pool — then reloads the per-tile ``.gsplats.zarr`` outputs
and merges them with the shared :func:`merge_tile_results` helper.

This is the local, no-Slurm counterpart to the Slurm ``batch --parallel``
packing (``gsplats/batch/slurm_gen.py``): same "independent processes on one
GPU" model, driven by a local thread pool instead of a scheduler.  Useful when
a single tile under-saturates the GPU and there is spare compute/VRAM.
"""

from __future__ import annotations

import os
import shutil
import subprocess
import sys
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from typing import Any, Callable, Optional, Sequence

import numpy as np
from arbol import aprint, asection

from luxar.gsplats.fit_tiled_gsplats import merge_tile_results
from luxar.gsplats.gsplat_data import GSplatData

# Builds the argv for tile ``i`` of ``M`` writing to a given output path.
WorkerCmdBuilder = Callable[[int, int, Path], "list[str]"]


def _empty_tile(ndim: int) -> GSplatData:
    """A 0-splat placeholder for a skipped (empty) tile.

    Mirrors the 0-splat result ``fit_tile`` returns for a near-zero tile so the
    parallel path's per-tile bookkeeping matches the sequential path.
    """
    from luxar.gsplats.utils.trils import tril_size

    return GSplatData(
        centers=np.zeros((0, ndim), dtype=np.float32),
        amplitudes=np.zeros((0,), dtype=np.float32),
        cholesky_factors=np.zeros((0, tril_size(ndim)), dtype=np.float32),
        stats={"skipped": True},
    )


def luxar_argv0() -> list[str]:
    """Return the argv prefix used to re-invoke the ``luxar`` CLI.

    Prefers the installed ``luxar`` console script (``pyproject.toml`` declares
    ``luxar = "luxar.cli:app"``).  Falls back to ``python -m luxar`` via the
    current interpreter (``luxar/__main__.py`` exists) when the script is not on
    ``PATH`` — this keeps the worker on the same interpreter/environment as the
    parent.
    """
    exe = shutil.which("luxar")
    if exe:
        return [exe]
    return [sys.executable, "-m", "luxar"]


def build_worker_cmd(
    argv0: list[str],
    input_path: str | Path,
    out_path: str | Path,
    tile_idx: int,
    num_tiles: int,
    tile_size: int,
    overlap: int,
    *,
    seeds: Optional[str] = None,
    iters: Optional[int] = None,
    device: Optional[str] = None,
    preset: Optional[str] = None,
    config: Optional[str | Path] = None,
    loss: Optional[str] = None,
    lr: Optional[float] = None,
    floor: Optional[str] = None,
    seed_method: Optional[str] = None,
    downscale: Optional[str] = None,
    channel: Optional[int] = None,
    timepoint: Optional[int] = None,
    array_key: Optional[str] = None,
    axes: Optional[str] = None,
    progressive: bool = False,
    max_splats_per_pass: int = 5000,
    psnr_patience: float = 0.5,
    max_passes: Optional[int] = None,
    denoise: bool = False,
    denoise_h: Optional[float] = None,
    denoise_patch_size: int = 3,
    denoise_search_distance: int = 5,
    denoise_backend: str = "auto",
    denoise_2d: bool = False,
    allow_empty_tile: bool = False,
) -> list[str]:
    """Build the ``luxar gsplat fit --tile i/M`` argv for one worker.

    Workers run in single-tile mode (which translates/rescales coordinates and
    disables per-tile culling internally) and write to ``out_path``.  Quiet by
    default to keep concurrent logs readable; ``--cull-retention 0`` defers all
    culling to the parent's merge.  ``allow_empty_tile`` makes a 0-splat tile
    write an ``.empty`` marker (and exit 0) instead of erroring on save.  The
    tiling driver flags ``--tiling``, ``--jobs`` and the parent's ``--compress``
    are intentionally **never** forwarded.
    """
    cmd = [
        *argv0,
        "gsplat",
        "fit",
        str(input_path),
        str(out_path),
        "--tile",
        f"{tile_idx}/{num_tiles}",
        "--tile-size",
        str(tile_size),
        "--overlap",
        str(overlap),
        "--quiet",
        "--cull-retention",
        "0",
    ]
    if seeds is not None:
        cmd += ["--seeds", str(seeds)]
    if iters is not None:
        cmd += ["--iters", str(iters)]
    if device:
        cmd += ["--device", device]
    if preset:
        cmd += ["--preset", preset]
    if config:
        cmd += ["--config", str(config)]
    if loss:
        cmd += ["--loss", loss]
    if lr is not None:
        cmd += ["--lr", str(lr)]
    if floor is not None:
        cmd += ["--floor", str(floor)]
    if seed_method:
        cmd += ["--seed-method", seed_method]
    if downscale is not None:
        cmd += ["--downscale", downscale]
    if channel is not None:
        cmd += ["--channel", str(channel)]
    if timepoint is not None:
        cmd += ["--timepoint", str(timepoint)]
    if array_key:
        cmd += ["--array-key", array_key]
    if axes:
        # Workers re-load the full store in single-tile mode; without --axes they
        # would fall back to the positional heuristic and load a differently-
        # shaped/ordered volume than the parent's tile grid (corrupt merge).
        cmd += ["--axes", axes]
    if progressive:
        cmd += [
            "--progressive",
            "--splats-per-pass",
            str(max_splats_per_pass),
            "--psnr-patience",
            str(psnr_patience),
        ]
        if max_passes is not None:
            cmd += ["--max-passes", str(max_passes)]
    if denoise:
        cmd += ["--denoise"]
        # Inject the once-calibrated h so each worker skips its own Noise2Self
        # calibration (avoids N redundant runs and seam-causing h drift).
        if denoise_h is not None:
            cmd += ["--denoise-h", str(denoise_h)]
        cmd += [
            "--denoise-patch-size",
            str(denoise_patch_size),
            "--denoise-search-distance",
            str(denoise_search_distance),
            "--denoise-backend",
            denoise_backend,
        ]
        if denoise_2d:
            cmd += ["--denoise-2d"]
    if allow_empty_tile:
        cmd += ["--allow-empty-tile"]
    return cmd


def resolve_jobs(
    jobs: str | int,
    *,
    tile_voxels: int,
    num_tiles: int,
    device: Optional[str] = None,
    dtype_bytes: int = 4,
) -> int:
    """Resolve the ``--jobs`` option to a concrete worker count.

    Explicit integers pass through (clamped to ``>= 1``).  ``"auto"`` probes
    free GPU VRAM and divides by a conservative per-tile working-set estimate
    (~2x the tile's voxel bytes, matching the Slurm parallel-packing heuristic);
    it falls back to half the CPU count when VRAM can't be queried (CPU / MPS
    devices have no ``mem_get_info``).  The result is always clamped to
    ``num_tiles`` — never spawn more workers than there are tiles.
    """
    if isinstance(jobs, str) and jobs.strip().lower() == "auto":
        free: Optional[int] = None
        try:
            from luxar.gsplats.metrics import _gpu_free_memory
            from luxar.gsplats.utils.device import resolve_torch_device

            dev = resolve_torch_device(device) if device else resolve_torch_device()
            free = _gpu_free_memory(dev)
        except Exception:
            free = None

        if free is None:
            n = max(1, (os.cpu_count() or 2) // 2)
        else:
            per_tile = max(1, int(2.0 * tile_voxels * dtype_bytes))
            n = max(1, int(free // per_tile))
        return max(1, min(n, num_tiles))

    return max(1, min(int(jobs), num_tiles))


def fit_tiled_parallel(
    *,
    num_tiles: int,
    jobs: int,
    tmp_dir: Path,
    worker_cmd_builder: WorkerCmdBuilder,
    volume_shape: tuple[int, ...],
    tile_size: int | Sequence[int],
    overlap: int | Sequence[int],
    progressive: bool,
    cull_retention: float | None,
    verbose: bool = True,
    keep_tiles: bool = False,
    partition: bool = False,
    recipe: Optional[str] = None,
    recipe_params: "Optional[Any]" = None,
) -> "Any":  # GSplatData (flat) or a GSplatNode (partition)
    """Fit all tiles via concurrent worker subprocesses, then merge.

    Each tile ``i`` is fit by a separate process built by ``worker_cmd_builder``
    and written to ``tmp_dir/tile_{i}.gsplats.zarr``.  Up to ``jobs`` workers run
    concurrently.  After all succeed, the per-tile outputs are reloaded and
    merged with :func:`merge_tile_results` (LOD-aware when ``progressive``).

    The workers re-invoke single-tile mode (``fit --tile i/M``), which already
    translates centers to global coordinates and — when ``--downscale`` is in
    play — rescales them back to original coordinates.  The merge therefore does
    **not** rescale again; ``volume_shape`` here is the (post-downscale) shape
    used only for stats and the empty-result fallback.

    Parameters
    ----------
    num_tiles : int
        Total number of tiles (``M``); workers are spawned for indices ``0..M-1``.
    jobs : int
        Maximum number of concurrent worker processes.
    tmp_dir : Path
        Directory for per-tile outputs.  Created if absent.  Removed on success
        unless ``keep_tiles`` is set; **retained** on failure for inspection.
    worker_cmd_builder : callable
        ``(tile_idx, num_tiles, out_path) -> argv`` returning the command to fit
        one tile.  This is the injection seam for testing.
    volume_shape, tile_size, overlap, progressive, cull_retention, verbose
        Forwarded to :func:`merge_tile_results`.
    keep_tiles : bool, default False
        Keep the per-tile temp outputs after a successful merge.

    Returns
    -------
    GSplatData
        Merged result. Multi-LOD if ``progressive`` and tiles carry sublods.

    Raises
    ------
    RuntimeError
        If any worker exits non-zero.  The message names the failing tiles and
        includes a tail of their stderr; ``tmp_dir`` is left in place.
    """
    tmp_dir = Path(tmp_dir)
    # Start from a clean slate: a retained dir from a prior (failed or
    # keep_tiles) run could otherwise leave a stale tile_{i} that this run
    # reloads instead of its own output. There is no resume logic.
    shutil.rmtree(tmp_dir, ignore_errors=True)
    tmp_dir.mkdir(parents=True, exist_ok=True)
    t0 = time.perf_counter()

    tile_paths = [tmp_dir / f"tile_{i}.gsplats.zarr" for i in range(num_tiles)]

    def _run(i: int) -> tuple[int, int, str]:
        try:
            cmd = [str(c) for c in worker_cmd_builder(i, num_tiles, tile_paths[i])]
            proc = subprocess.run(cmd, capture_output=True, text=True)
        except Exception as exc:  # bad argv → FileNotFoundError/OSError, builder bug
            # Funnel build/launch failures into the normal per-tile failure path
            # so the caller gets the curated error + retained tmp_dir, not a raw
            # traceback from a worker thread.
            return i, -1, f"failed to build/launch worker: {exc!r}"
        stream = proc.stderr or proc.stdout or ""
        return i, proc.returncode, stream

    failures: list[tuple[int, str]] = []
    with asection(
        f"Parallel tiled fitting: {num_tiles} tiles, {jobs} concurrent worker(s)"
    ):
        with ThreadPoolExecutor(max_workers=max(1, jobs)) as ex:
            futures = [ex.submit(_run, i) for i in range(num_tiles)]
            done = 0
            for fut in as_completed(futures):
                i, rc, stream = fut.result()
                done += 1
                if rc != 0:
                    tail = "\n".join(stream.strip().splitlines()[-20:])
                    failures.append((i, tail))
                    if verbose:
                        aprint(f"Tile {i} FAILED (exit {rc}) [{done}/{num_tiles}]")
                elif verbose:
                    aprint(f"Tile {i} done [{done}/{num_tiles}]")

    if failures:
        idxs = ", ".join(str(i) for i, _ in failures)
        detail = "\n\n".join(
            f"--- tile {i} stderr (tail) ---\n{msg}" for i, msg in failures
        )
        raise RuntimeError(
            f"{len(failures)} of {num_tiles} tile fits failed (tiles: {idxs}). "
            f"Temp outputs kept at {tmp_dir} for inspection.\n{detail}"
        )

    # Reload each tile. A successful worker either wrote its .gsplats.zarr, or —
    # for a tile windowed to near-zero signal (0 splats) — wrote a sibling
    # ".empty" marker (the gsplats writer rejects empty stores). An empty tile
    # contributes nothing to the merge (concatenate filters empties), exactly as
    # in the sequential path. A path that is neither present nor marked empty
    # after a clean exit is a silent spatial hole; a present-but-unreadable
    # store (e.g. a worker OOM-killed mid-save) is corrupt — both are failures.
    ndim = len(volume_shape)
    results: list[GSplatData] = []
    missing: list[int] = []
    corrupt: list[tuple[int, str]] = []
    for i, p in enumerate(tile_paths):
        if p.exists():
            try:
                results.append(GSplatData.load(p))
            except Exception as exc:  # present but unreadable/partial store
                corrupt.append((i, repr(exc)))
        elif Path(str(p) + ".empty").exists():
            # Legitimately-empty tile: append a 0-splat placeholder so the
            # merged `splats_per_tile` stat stays positionally aligned with the
            # sequential path (which keeps every tile's 0-splat result).
            # concatenate filters empties, so the merged splats are unchanged.
            results.append(_empty_tile(ndim))
        else:
            missing.append(i)

    if missing or corrupt:
        parts = []
        if missing:
            parts.append(f"wrote no output (tiles: {', '.join(map(str, missing))})")
        if corrupt:
            ids = ", ".join(f"{i}: {e}" for i, e in corrupt)
            parts.append(f"wrote an unreadable store (tiles: {ids})")
        raise RuntimeError(
            f"{len(missing) + len(corrupt)} of {num_tiles} tiles exited cleanly "
            f"but {'; '.join(parts)}. Temp outputs kept at {tmp_dir} for inspection."
        )

    elapsed = time.perf_counter() - t0
    merged = merge_tile_results(
        results,
        volume_shape=volume_shape,
        tile_size=tile_size,
        overlap=overlap,
        num_tiles=num_tiles,
        progressive=progressive,
        cull_retention=cull_retention,
        elapsed=elapsed,
        verbose=verbose,
        partition=partition,
        recipe=recipe,
        recipe_params=recipe_params,
    )

    if not keep_tiles:
        shutil.rmtree(tmp_dir, ignore_errors=True)

    return merged
