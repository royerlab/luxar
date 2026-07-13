#!/usr/bin/env python3
"""Apply calibrated K* values to each gsplat demo's MAX_SPLATS constant.

Reads ``scripts/calibration_results/_all_summaries.json`` and updates the
``MAX_SPLATS = N`` or ``SEEDS_PER_TILE = N`` assignment in each demo file
in-place.

Also bumps ``MAX_SPLATS_PER_PASS`` proportionally so the progressive fitter
runs in roughly 4–8 passes regardless of the new total — keeping per-pass
costs and convergence patience consistent across demos.
"""

from __future__ import annotations

import json
import re
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
DEMOS_DIR = REPO_ROOT / "packages" / "luxar" / "src" / "luxar" / "demos"
SUMMARIES = REPO_ROOT / "scripts" / "calibration_results" / "_all_summaries.json"


# Mapping: calibration summary "name" → demo filename + variable to update.
# Some demos share a calibration (alias_of); each still gets its file updated.
DEMO_FILE_MAP = {
    "organoid_dapi": ("demo_gsplats_3d_organoid_dapi_nuclei.py", "MAX_SPLATS"),
    "cells3d_multichannel": (
        "demo_gsplats_3d_cells3d_multichannel.py",
        "MAX_SPLATS",
    ),
    "kidney_multichannel_layers": (
        "demo_gsplats_3d_kidney_multichannel_layers.py",
        "MAX_SPLATS",
    ),
    "kidney_multichannel_toggles": (
        "demo_gsplats_3d_kidney_multichannel_toggles.py",
        "MAX_SPLATS",
    ),
    "acto3d_heart": ("demo_gsplats_3d_acto3d_heart.py", "MAX_SPLATS"),
    "organoid_multichannel": (
        "demo_gsplats_3d_organoid_multichannel.py",
        "MAX_SPLATS",
    ),
    "opencell_map4": ("demo_gsplats_3d_opencell_map4.py", "MAX_SPLATS"),
    "tribolium_embryo": ("demo_gsplats_3d_tribolium_embryo.py", "MAX_SPLATS"),
    "celegans_tracking": ("demo_gsplats_4d_celegans_tracking.py", "MAX_SPLATS"),
    "zebrafish_timelapse": ("demo_gsplats_4d_zebrafish_timelapse.py", "MAX_SPLATS"),
    "cmu1_pathology": ("demo_gsplats_2d_cmu1_pathology.py", "SEEDS_PER_TILE"),
    "codex_pancreas": ("demo_gsplats_2d_codex_pancreas.py", "SEEDS_PER_TILE"),
}


def _round_clean(k: int) -> int:
    """Round K* to a clean human-readable number.

    Rules:
      <1k: nearest 100
      1k-10k: nearest 500
      10k-100k: nearest 1k
      >=100k: nearest 5k
    """
    if k < 1_000:
        return int(round(k / 100) * 100)
    if k < 10_000:
        return int(round(k / 500) * 500)
    if k < 100_000:
        return int(round(k / 1_000) * 1_000)
    return int(round(k / 5_000) * 5_000)


def _choose_per_pass(total: int) -> int:
    """Pick MAX_SPLATS_PER_PASS so the run uses ~4-6 passes."""
    target_passes = 5
    raw = max(500, total // target_passes)
    # Round to nearest 500.
    return int(round(raw / 500) * 500) or 500


def _update_constant(text: str, name: str, new_value: int) -> tuple[str, int]:
    """Replace `name = <int>` (with optional underscores & inline comment).

    Returns (new_text, n_replacements).  Only replaces the first occurrence
    on a left-margin line — top-level demo constants.
    """
    pattern = re.compile(
        rf"(?m)^({re.escape(name)}\s*=\s*)([0-9_]+)(\s*(?:#.*)?)$"
    )

    def repl(m: re.Match[str]) -> str:
        return f"{m.group(1)}{new_value}{m.group(3)}"

    new_text, n = pattern.subn(repl, text, count=1)
    return new_text, n


def _update_demo_file(
    path: Path,
    splats_var: str,
    new_total: int,
    new_per_pass: int | None,
) -> dict:
    """Update MAX_SPLATS (or SEEDS_PER_TILE) and optionally MAX_SPLATS_PER_PASS."""
    text = path.read_text()
    changes: dict = {"file": str(path.relative_to(REPO_ROOT))}

    text, n1 = _update_constant(text, splats_var, new_total)
    if n1 == 0:
        changes["error"] = f"{splats_var} not found in {path.name}"
        return changes
    changes[splats_var] = new_total

    if new_per_pass is not None and "MAX_SPLATS_PER_PASS" in text:
        text, n2 = _update_constant(text, "MAX_SPLATS_PER_PASS", new_per_pass)
        if n2 > 0:
            # Clamp per-pass to <= total
            if new_per_pass > new_total:
                new_per_pass = new_total
                text, _ = _update_constant(text, "MAX_SPLATS_PER_PASS", new_per_pass)
            changes["MAX_SPLATS_PER_PASS"] = new_per_pass

    path.write_text(text)
    return changes


def main() -> int:
    if not SUMMARIES.exists():
        print(f"No summary at {SUMMARIES}; run calibrate_gsplat_demos.py first.")
        return 1

    summaries = json.loads(SUMMARIES.read_text())
    by_name = {s["name"]: s for s in summaries if "name" in s}

    print(f"Loaded {len(by_name)} demo summary entries from {SUMMARIES.name}")
    print()
    print(f"{'demo':<32} {'K* raw':>10} {'K* rounded':>12} {'per-pass':>10}  file")
    print("-" * 100)

    rows = []
    for demo_name, (filename, var_name) in DEMO_FILE_MAP.items():
        s = by_name.get(demo_name)
        if s is None or s.get("status") != "ok":
            print(f"{demo_name:<32} {'—':>10} {'—':>12} {'—':>10}  (no calibration result)")
            continue
        k_star_raw = int(s["recommended_k_star"])
        k_star = _round_clean(k_star_raw)
        per_pass = _choose_per_pass(k_star)

        path = DEMOS_DIR / filename
        if not path.exists():
            print(f"{demo_name:<32} {k_star_raw:>10} {k_star:>12} {per_pass:>10}  MISSING: {filename}")
            continue

        # SEEDS_PER_TILE demos don't have MAX_SPLATS_PER_PASS.
        per_pass_arg = per_pass if var_name == "MAX_SPLATS" else None
        change = _update_demo_file(path, var_name, k_star, per_pass_arg)
        rows.append({"demo": demo_name, "k_star_raw": k_star_raw, **change})
        marker = f"{var_name}={k_star}"
        if per_pass_arg is not None and "MAX_SPLATS_PER_PASS" in change:
            marker += f" per_pass={change['MAX_SPLATS_PER_PASS']}"
        print(f"{demo_name:<32} {k_star_raw:>10} {k_star:>12} {per_pass:>10}  {filename} → {marker}")

    print()
    print(f"Updated {len(rows)} demo files.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
