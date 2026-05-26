#!/usr/bin/env python3
"""Run ``luxar gsplat cal`` on every gsplat demo's preprocessed volume(s).

Pipeline per demo:
  1. Import the demo module by file path (the ``luxar`` package overrides
     ``luxar.demos`` so we go around it) with a sanitized sys.argv.
  2. Call the demo's own data loader to get the same preprocessed volume(s)
     the demo would fit.
  3. Persist each preprocessed sample as a ``.npy`` so the CLI can load it.
  4. Subprocess ``luxar gsplat cal volume.npy out.json --preset n2s --n-grid N
     --k-min ... --k-max ...`` — the same CLI command end users run.
  5. Parse the returned JSON and emit K* per channel / per representative
     timepoint / per tile crop.

For 4D demos (celegans, zebrafish) we calibrate 3 distributed timepoints and
take the median K*. The mostly-stationary structure across the timelapse
makes the median a stable representative.

Calling the CLI (not the Python ``calibrate()`` function directly) is
deliberate: it exercises the published ``luxar gsplat cal`` exactly as
end users would, so any regression in the CLI surfaces here.

Usage::

    hatch run python scripts/calibrate_gsplat_demos.py                # all demos
    hatch run python scripts/calibrate_gsplat_demos.py --only dapi    # one demo
    hatch run python scripts/calibrate_gsplat_demos.py --list         # show targets
    hatch run python scripts/calibrate_gsplat_demos.py --skip-existing
"""

from __future__ import annotations

import argparse
import importlib
import importlib.util
import json
import os
import shutil
import subprocess
import sys
import tempfile
import time
from contextlib import contextmanager
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional

import numpy as np
from arbol import Arbol, aprint, asection

REPO_ROOT = Path(__file__).resolve().parent.parent
RESULTS_DIR = REPO_ROOT / "scripts" / "calibration_results"
RESULTS_DIR.mkdir(parents=True, exist_ok=True)
DEMOS_DIR = REPO_ROOT / "packages" / "luxar" / "src" / "luxar" / "demos"

# Volume scratch dir: large .npy files we feed to `luxar gsplat cal`.
# Kept under /tmp by default so the repo stays clean; override via $LUXAR_CAL_SCRATCH.
SCRATCH_DIR = Path(os.environ.get("LUXAR_CAL_SCRATCH", "/tmp/luxar-cal-scratch"))
SCRATCH_DIR.mkdir(parents=True, exist_ok=True)

# K-sweep configuration. n-grid=10 (default), exp progression.
N_GRID = 10
PROGRESSION = "exp"
# Preset that matches the manuscript's blind-spot protocol — see PRESETS in
# packages/luxar/src/luxar/cli/gsplat_config.py.
PRESET = "n2s"


# ---------------------------------------------------------------------------
# Demo-module helpers
# ---------------------------------------------------------------------------


def _clean_argv():
    """Return a context that resets sys.argv so demo module imports don't
    pick up driver flags."""

    @contextmanager
    def _ctx():
        saved = sys.argv
        sys.argv = [saved[0] if saved else "driver"]
        try:
            yield
        finally:
            sys.argv = saved

    return _ctx()


def _import_demo(file_stem: str):
    """Import a demo module by loading its file directly.

    ``luxar/__init__.py`` overrides ``sys.modules['luxar.demos']`` to point at
    ``luxar/utils/demos.py``, which shadows the actual demos package.  To get
    at the demo files we have to bypass the regular import machinery and load
    them by absolute path with a fresh, namespaced module name.
    """
    file_path = DEMOS_DIR / f"{file_stem}.py"
    if not file_path.exists():
        raise FileNotFoundError(f"Demo file not found: {file_path}")

    mod_name = f"_luxar_demo_{file_stem}"
    if mod_name in sys.modules:
        return sys.modules[mod_name]

    with _clean_argv():
        spec = importlib.util.spec_from_file_location(mod_name, file_path)
        if spec is None or spec.loader is None:
            raise ImportError(f"Could not load spec for {file_path}")
        module = importlib.util.module_from_spec(spec)
        sys.modules[mod_name] = module
        spec.loader.exec_module(module)
    return module


def _load_dapi():
    mod = _import_demo("demo_gsplats_3d_organoid_dapi_nuclei")
    V = mod.load_dapi_data()
    return [("dapi", V)]


def _load_kidney_layers():
    mod = _import_demo("demo_gsplats_3d_kidney_multichannel_layers")
    vols = mod.load_kidney()
    return [(f"ch{i}", v) for i, v in enumerate(vols)]


def _load_kidney_toggles():
    # Same data source as layers; both demos call skimage.data.kidney().
    return _load_kidney_layers()


def _load_acto3d():
    mod = _import_demo("demo_gsplats_3d_acto3d_heart")
    vols, _voxel = mod.load_acto3d_heart_data()
    return [(f"ch{i}", v) for i, v in enumerate(vols)]


def _load_organoid_multi():
    mod = _import_demo("demo_gsplats_3d_organoid_multichannel")
    vols = mod.load_multichannel_data()
    return [(f"ch{i}", v) for i, v in enumerate(vols)]


def _load_opencell():
    mod = _import_demo("demo_gsplats_3d_opencell_map4")
    cache = mod.CACHE_DIR / "opencell_map4_stack.tif"
    if not cache.exists():
        cache = mod.download_opencell_map4()
    vols = mod.load_opencell_data(cache)
    return [(f"ch{i}", v) for i, v in enumerate(vols)]


def _load_tribolium():
    mod = _import_demo("demo_gsplats_3d_tribolium_embryo")
    V = mod.load_tribolium_volume()
    return [("tribolium", V)]


def _load_celegans():
    """3 distributed timepoints from the 400-tp confocal dataset."""
    mod = _import_demo("demo_gsplats_4d_celegans_tracking")
    tps = [50, 200, 350]
    samples = []
    for tp in tps:
        preproc = mod.CACHE_DIR / f"celegans_s1_t{tp:04d}_preprocessed.npy"
        if preproc.exists():
            V = np.load(preproc)
            samples.append((f"t{tp:04d}", V))
        else:
            aprint(
                f"  celegans: no preprocessed cache for tp={tp} "
                f"({preproc.name}) — skipping this timepoint"
            )
    if not samples:
        raise RuntimeError(
            "celegans: no preprocessed timepoints available; run "
            "demo_gsplats_4d_celegans_tracking.py once to populate the cache."
        )
    return samples


def _load_zebrafish():
    """3 distributed timepoints from the LSM timelapse."""
    mod = _import_demo("demo_gsplats_4d_zebrafish_timelapse")
    result = mod.load_zebrafish_volumes()
    if isinstance(result, tuple) and len(result) >= 1:
        volumes = result[0]
    else:
        volumes = result
    n = len(volumes)
    if n == 0:
        raise RuntimeError("zebrafish: no timepoints returned")
    if n >= 3:
        picks = [0, n // 2, n - 1]
    else:
        picks = list(range(n))
    return [(f"t{i:04d}", volumes[i]) for i in picks]


def _load_cmu1_pathology_tile():
    """Calibrate on a single representative tile-sized crop of the RGB image."""
    mod = _import_demo("demo_gsplats_2d_cmu1_pathology")
    channels = mod.load_cmu1_image()
    tile = mod.TILE_SIZE
    samples = []
    for i, ch in enumerate(channels):
        h, w = ch.shape
        cy, cx = h // 2, w // 2
        y0 = max(0, cy - tile // 2)
        x0 = max(0, cx - tile // 2)
        crop = ch[y0 : y0 + tile, x0 : x0 + tile]
        samples.append((f"ch{i}_tile", crop))
    return samples


def _load_codex_pancreas_tile():
    """Same strategy as CMU-1: representative tile-sized crop per channel."""
    mod = _import_demo("demo_gsplats_2d_codex_pancreas")
    tiff_dir = mod.ensure_extracted()
    samples = []
    tile = mod.TILE_SIZE
    for i, ch_config in enumerate(mod.CHANNELS):
        ch = mod.load_channel(tiff_dir, ch_config)
        h, w = ch.shape
        cy, cx = h // 2, w // 2
        y0 = max(0, cy - tile // 2)
        x0 = max(0, cx - tile // 2)
        crop = ch[y0 : y0 + tile, x0 : x0 + tile]
        samples.append((f"ch{i}_tile", crop))
    return samples


# Demo registry — name → loader, K bounds, aggregate rule.
# k_max is generous (256K/512K) so the curve has space to hit a plateau or
# peak under the corrected ``n2s`` preset.
DEMOS: List[Dict[str, Any]] = [
    {
        "name": "organoid_dapi",
        "loader": _load_dapi,
        "k_min": 1_000,
        "k_max": 256_000,
        "comment": "128^3 single-channel DAPI from IDR",
    },
    {
        "name": "kidney_multichannel_layers",
        "loader": _load_kidney_layers,
        "k_min": 1_000,
        "k_max": 256_000,
        "comment": "skimage kidney 3-channel ~16x512x512",
    },
    {
        "name": "kidney_multichannel_toggles",
        "loader": _load_kidney_toggles,
        "k_min": 1_000,
        "k_max": 256_000,
        "comment": "Same data as kidney_layers; reuses K*",
        "alias_of": "kidney_multichannel_layers",
    },
    {
        "name": "acto3d_heart",
        "loader": _load_acto3d,
        "k_min": 1_000,
        "k_max": 512_000,
        "comment": "256^3 3-channel light-sheet mouse heart",
    },
    {
        "name": "organoid_multichannel",
        "loader": _load_organoid_multi,
        "k_min": 1_000,
        "k_max": 256_000,
        "comment": "256^3 2-channel organoid",
    },
    {
        "name": "opencell_map4",
        "loader": _load_opencell,
        "k_min": 2_000,
        "k_max": 512_000,
        "comment": "Confocal stack, 2 channels",
    },
    {
        "name": "tribolium_embryo",
        "loader": _load_tribolium,
        "k_min": 2_000,
        "k_max": 512_000,
        "comment": "Light-sheet Tribolium embryo (large)",
    },
    {
        "name": "celegans_tracking",
        "loader": _load_celegans,
        "k_min": 1_000,
        "k_max": 256_000,
        "comment": "41x512x512 per timepoint; 3 distributed tp; median K*",
        "aggregate": "median",
    },
    {
        "name": "zebrafish_timelapse",
        "loader": _load_zebrafish,
        "k_min": 1_000,
        "k_max": 256_000,
        "comment": "Timelapse, 3 distributed tp; median K*",
        "aggregate": "median",
    },
    {
        "name": "cmu1_pathology",
        "loader": _load_cmu1_pathology_tile,
        "k_min": 1_000,
        "k_max": 256_000,
        "comment": "RGB H&E, per-tile (TILE_SIZE^2) crop per channel",
    },
    {
        "name": "codex_pancreas",
        "loader": _load_codex_pancreas_tile,
        "k_min": 1_000,
        "k_max": 256_000,
        "comment": "CODEX, per-tile (TILE_SIZE^2) crop per channel",
    },
]


# ---------------------------------------------------------------------------
# CLI subprocess wrapper
# ---------------------------------------------------------------------------


def _run_cli_cal(
    npy_path: Path,
    json_path: Path,
    k_min: int,
    k_max: int,
    n_grid: int = N_GRID,
    preset: str = PRESET,
    progression: str = PROGRESSION,
) -> None:
    """Invoke ``luxar gsplat cal <npy> <json> --preset n2s ...`` as a subprocess.

    Inherits stdout/stderr so per-K progress is visible in the driver log.
    Raises ``subprocess.CalledProcessError`` if the CLI returns non-zero.
    """
    cmd = [
        "hatch",
        "run",
        "luxar",
        "gsplat",
        "cal",
        str(npy_path),
        str(json_path),
        "--preset",
        preset,
        "--n-grid",
        str(n_grid),
        "--k-min",
        str(k_min),
        "--k-max",
        str(k_max),
        "--progression",
        progression,
        "--quiet",
    ]
    aprint(f"  $ {' '.join(cmd)}")
    subprocess.run(cmd, check=True, cwd=str(REPO_ROOT))


def _read_k_star(json_path: Path) -> Dict[str, Any]:
    """Pull the recommended K* and a few diagnostic fields from cal JSON."""
    raw = json.loads(json_path.read_text())
    peak = raw["held_out_peak"]
    nf = raw.get("noise_floor", {})
    return {
        "k_star": int(peak["k_star"]),
        "type": peak["type"],
        "confidence_db": float(peak["confidence_db"]),
        "noise_floor_psnr_db": (
            float(nf["psnr_max_db"]) if nf.get("psnr_max_db") is not None else None
        ),
        "k_values_effective": raw.get("k_values_effective"),
        "held_out_psnr_db": raw.get("held_out_psnr_db"),
    }


# ---------------------------------------------------------------------------
# Per-demo runner
# ---------------------------------------------------------------------------


def _calibrate_one_sample(
    label: str,
    V: np.ndarray,
    out_dir: Path,
    k_min: int,
    k_max: int,
) -> Dict[str, Any]:
    """Save V to a temp .npy, then subprocess ``luxar gsplat cal`` on it."""
    out_dir.mkdir(parents=True, exist_ok=True)
    json_path = out_dir / f"{label}.json"

    # Persist preprocessed volume for the CLI. Use the scratch dir so the
    # repo stays clean.
    npy_path = SCRATCH_DIR / f"{label}.npy"
    aprint(
        f"  Saving {label}: shape={V.shape} dtype={V.dtype} → {npy_path.name}"
    )
    np.save(npy_path, V.astype(np.float32, copy=False))

    t0 = time.perf_counter()
    try:
        _run_cli_cal(npy_path, json_path, k_min=k_min, k_max=k_max)
    finally:
        # We don't keep the giant scratch volumes around once cal is done.
        try:
            npy_path.unlink()
        except OSError:
            pass
    elapsed = time.perf_counter() - t0

    diag = _read_k_star(json_path)
    aprint(
        f"  ✓ {label}: K*={diag['k_star']} "
        f"({diag['type']}, {diag['confidence_db']:.2f} dB) "
        f"in {elapsed:.1f}s"
    )
    return {
        "label": label,
        "k_star": diag["k_star"],
        "type": diag["type"],
        "confidence_db": diag["confidence_db"],
        "volume_shape": list(V.shape),
        "elapsed_seconds": elapsed,
        "json_path": str(json_path.relative_to(REPO_ROOT)),
        "noise_floor_psnr_db": diag["noise_floor_psnr_db"],
    }


def _calibrate_demo(demo: Dict[str, Any], skip_existing: bool) -> Dict[str, Any]:
    name = demo["name"]
    out_dir = RESULTS_DIR / name
    summary_path = RESULTS_DIR / f"{name}.summary.json"

    if skip_existing and summary_path.exists():
        aprint(f"⊘ {name}: summary exists, skipping")
        return json.loads(summary_path.read_text())

    if "alias_of" in demo:
        alias_summary = RESULTS_DIR / f"{demo['alias_of']}.summary.json"
        if alias_summary.exists():
            aprint(f"↳ {name}: aliased to {demo['alias_of']}")
            data = json.loads(alias_summary.read_text())
            data = dict(data, name=name, alias_of=demo["alias_of"])
            summary_path.write_text(json.dumps(data, indent=2))
            return data

    with asection(f"=== {name} ==="):
        aprint(f"  Comment: {demo.get('comment', '')}")
        aprint(f"  Preset:  {PRESET}")
        aprint(f"  K grid:  exp[{demo['k_min']}..{demo['k_max']}], n={N_GRID}")

        aprint("  Loading volumes...")
        t_load = time.perf_counter()
        try:
            samples = demo["loader"]()
        except Exception as e:
            aprint(f"  ✗ {name}: loader failed: {e!r}")
            summary = {"name": name, "status": "loader_failed", "error": repr(e)}
            summary_path.write_text(json.dumps(summary, indent=2))
            return summary
        aprint(
            f"  Loaded {len(samples)} volume(s) in {time.perf_counter() - t_load:.1f}s"
        )

        results: List[Dict[str, Any]] = []
        for label, V in samples:
            sample_label = f"{name}__{label}"
            try:
                res = _calibrate_one_sample(
                    sample_label,
                    V,
                    out_dir,
                    k_min=demo["k_min"],
                    k_max=demo["k_max"],
                )
                results.append(res)
            except Exception as e:
                aprint(f"  ✗ {sample_label}: cal failed: {e!r}")
                results.append({"label": label, "status": "failed", "error": repr(e)})

    k_stars = [r["k_star"] for r in results if "k_star" in r]
    aggregate = demo.get("aggregate", "max")
    recommended = None
    if k_stars:
        if aggregate == "median":
            recommended = int(np.median(k_stars))
        elif aggregate == "max":
            recommended = int(max(k_stars))
        elif aggregate == "mean":
            recommended = int(round(np.mean(k_stars)))

    summary = {
        "name": name,
        "status": "ok",
        "k_min": demo["k_min"],
        "k_max": demo["k_max"],
        "n_grid": N_GRID,
        "preset": PRESET,
        "aggregate": aggregate,
        "recommended_k_star": recommended,
        "per_sample": results,
        "comment": demo.get("comment", ""),
    }
    summary_path.write_text(json.dumps(summary, indent=2))
    aprint(f"✓ {name}: recommended K* = {recommended} (aggregate={aggregate})")
    return summary


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------


def _print_summary_table(summaries: List[Dict[str, Any]]) -> None:
    aprint("\n" + "═" * 80)
    aprint("  CALIBRATION SUMMARY")
    aprint("═" * 80)
    aprint(f"  {'demo':<32} {'samples':>8} {'K*':>8}  notes")
    aprint("  " + "-" * 78)
    for s in summaries:
        if s.get("status") != "ok":
            aprint(f"  {s['name']:<32} {'—':>8} {'—':>8}  {s.get('status', '?')}")
            continue
        n_samples = sum(1 for r in s["per_sample"] if "k_star" in r)
        k_star = s.get("recommended_k_star") or "—"
        agg = s.get("aggregate", "max")
        comment = s.get("comment", "")
        aprint(f"  {s['name']:<32} {n_samples:>8d} {str(k_star):>8}  {agg}: {comment}")
    aprint("═" * 80)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--only",
        type=str,
        default=None,
        help="Run only the demo whose name contains this substring",
    )
    parser.add_argument(
        "--list",
        action="store_true",
        help="List demo targets and exit",
    )
    parser.add_argument(
        "--skip-existing",
        action="store_true",
        help="Skip demos that already have a summary.json",
    )
    args = parser.parse_args()

    if args.list:
        for d in DEMOS:
            aprint(f"  {d['name']:<32} [{d['k_min']}..{d['k_max']}] {d.get('comment', '')}")
        return 0

    Arbol.max_depth = 5

    targets = DEMOS
    if args.only:
        needle = args.only.lower()
        targets = [d for d in DEMOS if needle in d["name"].lower()]
        if not targets:
            aprint(f"No demo matches --only {args.only}")
            return 1

    aprint(f"Running calibration on {len(targets)} demo(s) via `luxar gsplat cal --preset {PRESET}`")
    aprint(f"Scratch volumes: {SCRATCH_DIR}")
    summaries = []
    for d in targets:
        try:
            summaries.append(_calibrate_demo(d, skip_existing=args.skip_existing))
        except Exception as e:
            aprint(f"FATAL on {d['name']}: {e!r}")
            summaries.append({"name": d["name"], "status": "fatal", "error": repr(e)})

    _print_summary_table(summaries)

    agg_path = RESULTS_DIR / "_all_summaries.json"
    agg_path.write_text(json.dumps(summaries, indent=2))
    aprint(f"\nWrote aggregate summary → {agg_path.relative_to(REPO_ROOT)}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
