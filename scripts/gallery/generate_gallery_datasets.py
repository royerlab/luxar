#!/usr/bin/env python3
"""Generate the demo datasets needed by the Luxar gallery harness.

Reads the single-source-of-truth manifest (``scripts/gallery/manifest.json``)
and, for every entry that declares a generator ``script``, runs that demo with
``--no-serve`` to produce its ``.luxar.zarr`` under ``datasets/demos/`` — unless
the dataset already exists (idempotent / resumable).

Entries with ``script: null`` live only on a feature branch; their dataset must
already be present (typically generated once, then committed/kept locally). Such
entries are reported as *capture-only* and skipped by the generator — the
capture spec still picks them up if their dataset is on disk.

Usage::

    hatch run python scripts/gallery/generate_gallery_datasets.py            # generate all missing
    hatch run python scripts/gallery/generate_gallery_datasets.py --only lorenz,desi_galaxies
    hatch run python scripts/gallery/generate_gallery_datasets.py --force     # regenerate even if present
    hatch run python scripts/gallery/generate_gallery_datasets.py --list      # just print the plan
"""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
from pathlib import Path
from typing import Any, cast

from arbol import aprint, asection

REPO_ROOT = Path(__file__).resolve().parents[2]
MANIFEST = Path(__file__).resolve().parent / "manifest.json"
DEMOS_DIR = REPO_ROOT / "packages" / "luxar" / "src" / "luxar" / "demos"

# Per-demo generation timeout (seconds). Some demos download data or fit
# Gaussian splats; give them room but don't hang the whole run forever.
GEN_TIMEOUT_S = 3600


def load_manifest() -> list[dict[str, Any]]:
    with MANIFEST.open() as fh:
        data = json.load(fh)
    return cast("list[dict[str, Any]]", data["demos"])


def dataset_exists(entry: dict[str, Any]) -> bool:
    return bool((REPO_ROOT / entry["dataset"]).exists())


def generate_one(entry: dict[str, Any]) -> tuple[str, str]:
    """Run a demo's generator. Returns (status, detail)."""
    script = entry.get("script")
    if not script:
        return ("capture-only", "no generator script (feature-branch demo)")

    script_path = DEMOS_DIR / script
    if not script_path.exists():
        return ("missing-script", str(script_path.relative_to(REPO_ROOT)))

    cmd = [sys.executable, str(script_path), "--no-serve"]
    with asection(f"Generating {entry['id']} ({script})"):
        aprint(" ".join(cmd))
        try:
            proc = subprocess.run(
                cmd,
                cwd=REPO_ROOT,
                timeout=GEN_TIMEOUT_S,
                capture_output=True,
                text=True,
            )
        except subprocess.TimeoutExpired:
            return ("timeout", f"exceeded {GEN_TIMEOUT_S}s")

        if proc.returncode != 0:
            tail = (proc.stderr or proc.stdout or "").strip().splitlines()[-8:]
            for line in tail:
                aprint(line)
            return ("failed", f"exit {proc.returncode}")

    if not dataset_exists(entry):
        return ("no-output", f"script ran but {entry['dataset']} not found")
    return ("generated", entry["dataset"])


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--only",
        help="Comma-separated list of demo ids to restrict to.",
        default=None,
    )
    parser.add_argument(
        "--force",
        action="store_true",
        help="Regenerate even if the dataset already exists.",
    )
    parser.add_argument(
        "--list",
        action="store_true",
        help="Print the plan (per-demo status) and exit without generating.",
    )
    args = parser.parse_args()

    demos = load_manifest()
    if args.only:
        wanted = {s.strip() for s in args.only.split(",") if s.strip()}
        demos = [d for d in demos if d["id"] in wanted]
        unknown = wanted - {d["id"] for d in demos}
        if unknown:
            aprint(f"⚠️  Unknown demo ids ignored: {', '.join(sorted(unknown))}")

    if args.list:
        with asection(f"Gallery plan ({len(demos)} demos)"):
            for d in demos:
                present = "ready" if dataset_exists(d) else "MISSING"
                gen = d.get("script") or "(capture-only)"
                aprint(f"[{present:>7}] {d['id']:<34} {gen}")
        return 0

    results: dict[str, list[str]] = {
        "generated": [],
        "already-present": [],
        "capture-only": [],
        "failed": [],
        "timeout": [],
        "no-output": [],
        "missing-script": [],
    }

    with asection(f"Generating {len(demos)} gallery datasets"):
        for entry in demos:
            if dataset_exists(entry) and not args.force:
                aprint(f"✓ {entry['id']} — already present, skipping")
                results["already-present"].append(entry["id"])
                continue
            status, detail = generate_one(entry)
            results.setdefault(status, []).append(entry["id"])
            symbol = {"generated": "✅"}.get(status, "⚠️ ")
            aprint(f"{symbol} {entry['id']} — {status}: {detail}")

    with asection("Summary"):
        for status, ids in results.items():
            if ids:
                aprint(f"{status:>16}: {len(ids)}  ({', '.join(ids)})")

    ready = [d for d in demos if dataset_exists(d)]
    aprint(f"\n{len(ready)}/{len(demos)} datasets ready for capture.")
    aprint("Next: cd packages/luxar-viewer && pnpm gallery")

    # Non-zero only on hard generation failures (capture-only / missing input
    # data are expected and not errors).
    hard_failures = results["failed"] + results["timeout"] + results["no-output"]
    return 1 if hard_failures else 0


if __name__ == "__main__":
    raise SystemExit(main())
