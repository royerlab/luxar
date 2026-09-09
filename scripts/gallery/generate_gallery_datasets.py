#!/usr/bin/env python3
"""Generate the demo datasets needed by the Luxar gallery harness.

Reads the single-source-of-truth manifest (``scripts/gallery/manifest.json``)
and, for every entry that declares a generator ``script``, runs that demo with
``--no-serve`` to produce its ``.luxar.zarr`` under ``datasets/demos/`` — unless
the dataset already exists (idempotent / resumable). Every store produced by
this run is then checked for LOD-ladder quality and embedded scene credits before
the gallery capture can proceed. Scene credits gate the build; ladder findings
are report-only until the demo corpus has been laddered. A second report-only
pass covers the complete local inventory, so already-present neighbours remain
visible without deciding whether a newly generated store may proceed.

Entries with ``script: null`` live only on a feature branch; their dataset must
already be present (typically generated once, then committed/kept locally). Such
entries are reported as *capture-only* and skipped by the generator — the
capture spec still picks them up if their dataset is on disk. The manifest does
not currently contain any capture-only entries, but the shape remains supported.

A demo whose ``DEMO_META`` declares machine-local ``local_data`` exits non-zero
wherever that input is absent, which used to make the whole run report a hard
``failed`` and return 1. ``manual-file`` (the Gaia catalog is CC BY-NC, so it has
to be placed by hand) and ``kaggle-auth`` are always treated this way. A
``git-lfs`` demo is treated this way only while one of its manifest-declared
checkout payloads is missing or still an unpulled pointer; once every payload is
present, a non-zero exit is a real failure. Record-backed downloads declare no
``local_data`` requirement and therefore fail hard. Such an entry is RUN like
any other (so the machine that does have the input regenerates its tile on every
route, ``--force`` included), and only a positive exit with unavailable input is
reclassified into the soft *manual-data* bucket instead of ``failed``.

A demo listed in ``UNBUILDABLE_IDS`` is skipped WITHOUT being run at all — not
demoted after the fact — because there is no machine on which it currently
succeeds; running it would only burn ``GEN_TIMEOUT_S`` (and its download) before
reporting a hard failure. That list is temporary by construction: each entry
names the issue that put it there and is deleted when the cause is gone.

Two limits of the demote-on-failure rule, both deliberate. A genuine bug inside
a manual-file / Kaggle demo (or an LFS demo whose payload is absent) also lands
in the soft bucket rather than returning 1 — its error tail is still printed, so
it stays visible. And a ``timeout``, a
``no-output`` (exit 0 having written nothing) or a death by signal (a negative
return code: the OOM killer, a segfault) stays HARD even for them. One real case
remains unrescued: ``arxiv_papers_kaggle`` needs no credentials to start and
declares a ~30 GB download, so on a cold machine it can exhaust
``GEN_TIMEOUT_S`` and land in ``timeout``. Skipping large downloads would stop
regenerating the arXiv tile on a machine whose cache is warm.

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
DATA_MANIFEST = DEMOS_DIR / "data_manifest.json"

# Per-demo generation timeout (seconds). Some demos download data or fit
# Gaussian splats; give them room but don't hang the whole run forever.
GEN_TIMEOUT_S = 3600

# Built-scene audits are metadata-heavy but should never hang a gallery build.
AUDIT_TIMEOUT_S = 600

# Provisioning modes that may make a positive demo exit mean "this machine does
# not have the input". Git LFS is checked against the observed payload state;
# unlike `luxar demo run-all`, the gallery still attempts every runnable entry.
LOCAL_INPUT_MODES = ("manual-file", "kaggle-auth", "git-lfs")

# Manifest ids that CANNOT currently be built on any machine, mapped to why.
# Soft-skipped without spawning anything, the way a machine-local `local_data`
# entry is demoted: the alternative is a hard `timeout` that returns 1 and aborts
# `make generate-gallery` before it captures a single tile.
#
# This is a temporary list, not a policy. DELETE an entry the moment its cause is
# gone — the demo is then generated like any other.
UNBUILDABLE_IDS: dict[str, str] = {}

SCENE_AUDITOR_NAMES = (
    ("check_demo_ladders.py", False),
    ("check_scene_credits.py", True),
)


def load_manifest() -> list[dict[str, Any]]:
    with MANIFEST.open() as fh:
        data = json.load(fh)
    return cast("list[dict[str, Any]]", data["demos"])


def dataset_exists(entry: dict[str, Any]) -> bool:
    return bool((REPO_ROOT / entry["dataset"]).exists())


def _git_lfs_input_missing(meta: dict[str, Any]) -> bool:
    """Whether a validated demo cache names a missing or unpulled LFS file."""
    try:
        with DATA_MANIFEST.open() as fh:
            datasets = json.load(fh)["datasets"]
    except (OSError, KeyError, TypeError, json.JSONDecodeError):
        return False
    if not isinstance(datasets, dict):
        return False

    paths: list[Path] = []
    for cache_key in meta["caches"]:
        record = datasets.get(cache_key)
        if not isinstance(record, dict):
            continue
        directory = record.get("dir", cache_key)
        files = record.get("files")
        if not isinstance(directory, str) or not isinstance(files, list) or not files:
            continue
        record_paths: list[Path] = []
        for file_info in files:
            if not isinstance(file_info, dict) or not isinstance(
                file_info.get("name"), str
            ):
                break
            record_paths.append(DEMOS_DIR / "data" / directory / file_info["name"])
        else:
            paths.extend(record_paths)

    if not paths:
        return False

    try:
        from luxar.demos import is_lfs_pointer
    except ImportError:
        return False
    return any(not path.exists() or is_lfs_pointer(path) for path in paths)


def needs_local_input(entry: dict[str, Any]) -> str | None:
    """The entry's demo's ``local_data`` mode, if it is machine-local.

    Returns ``"manual-file"`` / ``"kaggle-auth"`` / ``"git-lfs"``, or ``None``
    for everything else. Read straight from the demo's ``DEMO_META`` by AST
    (never importing it): the gallery manifest carries no such field, and the
    demo file is the source of truth the ``luxar demo`` commands already use. A
    malformed or missing block is *not* this predicate's business — report
    ``None`` and let the normal generation path fail loudly in its own vocabulary.
    """
    script = entry.get("script")
    if not script:
        return None
    script_path = DEMOS_DIR / script
    if not script_path.exists():
        return None
    # Deferred import: this script otherwise needs only stdlib + arbol, and a
    # module-level `luxar` import would make even --help traceback on a partial
    # environment. An uninstalled luxar means nothing can be demoted, which errs
    # toward reporting failures loudly (and `--list` still prints its plan).
    try:
        from luxar.demos import registry
    except ImportError:
        return None

    try:
        meta = registry.extract_demo_meta(script_path)
    except registry.DemoMetaError:
        return None
    mode = meta["requirements"]["local_data"]
    if mode not in LOCAL_INPUT_MODES:
        return None
    if mode == "git-lfs" and not _git_lfs_input_missing(meta):
        return None
    return str(mode)


def generate_one(entry: dict[str, Any]) -> tuple[str, str]:
    """Run a demo's generator. Returns (status, detail).

    A non-zero exit is ``failed`` — except for a demo whose input this machine
    may simply not have (:func:`needs_local_input`), which is demoted to the soft
    ``manual-data`` bucket. The demotion is decided only AFTER the run, so the
    tile is still regenerated wherever the input is present; a ``timeout``, a
    ``no-output`` or a death by SIGNAL stays hard even then.
    """
    unbuildable = UNBUILDABLE_IDS.get(entry["id"])
    if unbuildable is not None:
        # Skipped BEFORE spawning: unlike the local-input demos there is no
        # machine on which this one succeeds, so running it only burns the
        # timeout (and a gigabyte of bandwidth) before failing hard.
        return ("unbuildable", unbuildable)

    script = entry.get("script")
    if not script:
        return ("capture-only", "no generator script (capture-only entry)")

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
            # The tail is printed either way: a demoted failure is soft, not
            # silent, since the same exit code covers "no input here" and a real
            # bug in the demo.
            #
            # `> 0`, not `!= 0`: on POSIX a NEGATIVE returncode means the child
            # was killed by a signal (-9 = the OOM killer, -11 = a segfault in a
            # native dependency), which is never how a demo reports a missing
            # input — it is a real failure of a run that got far enough to
            # allocate, so it stays hard rather than being filed under "this
            # machine doesn't have the file".
            mode = needs_local_input(entry) if proc.returncode > 0 else None
            if mode is not None:
                return (
                    "manual-data",
                    f"exit {proc.returncode}; this machine appears not to have "
                    f"the {mode} input",
                )
            return ("failed", f"exit {proc.returncode}")

    if not dataset_exists(entry):
        return ("no-output", f"script ran but {entry['dataset']} not found")
    return ("generated", entry["dataset"])


def print_plan(demos: list[dict[str, Any]]) -> None:
    """Print the per-demo plan for ``--list`` (generates nothing)."""
    with asection(f"Gallery plan ({len(demos)} demos)"):
        for d in demos:
            present = "ready" if dataset_exists(d) else "MISSING"
            gen = d.get("script") or "(capture-only)"
            if d["id"] in UNBUILDABLE_IDS:
                gen += "  (unbuildable; skipped without running)"
            mode = needs_local_input(d)
            if mode is not None:
                # Still run — the mark says a failure here will be reported as
                # manual-data rather than as a hard failure.
                gen += f"  (needs {mode}; a failure is reported as manual-data)"
            aprint(f"[{present:>7}] {d['id']:<34} {gen}")


def _run_scene_auditors(
    paths: list[Path], *, title: str, enforce_configured_gates: bool
) -> list[str]:
    """Run every built-scene auditor and return only enforced failures."""
    failures = []
    with asection(title):
        for auditor_name, gates in SCENE_AUDITOR_NAMES:
            auditor = REPO_ROOT / "scripts" / auditor_name
            # Generated-path calls are non-empty by construction, so
            # --require-scenes is defense-in-depth there. On the report-only
            # inventory pass it makes an empty build box explicit in the output.
            cmd = [
                sys.executable,
                str(auditor),
                "--require-scenes",
                *(str(path) for path in paths),
            ]
            aprint(" ".join(cmd))
            sys.stdout.flush()
            failure: str | None
            try:
                result = subprocess.run(cmd, cwd=REPO_ROOT, timeout=AUDIT_TIMEOUT_S)
            except subprocess.TimeoutExpired:
                failure = f"timed out after {AUDIT_TIMEOUT_S}s"
            else:
                failure = (
                    f"failed with exit {result.returncode}"
                    if result.returncode != 0
                    else None
                )
            if failure is not None:
                enforced = enforce_configured_gates and gates
                if enforced:
                    failures.append(auditor.stem)
                suffix = "" if enforced else " (report-only)"
                aprint(f"❌ {auditor.stem} {failure}{suffix}")
    return failures


def audit_generated_stores(paths: list[Path]) -> list[str]:
    """Audit exactly the stores made here, enforcing configured gates."""
    if not paths:
        return []
    return _run_scene_auditors(
        paths,
        title=f"Auditing {len(paths)} newly generated scene(s)",
        enforce_configured_gates=True,
    )


def report_local_scene_inventory() -> None:
    """Report every built scene without making neighbouring stores a gate."""
    _run_scene_auditors(
        [],
        title="Reporting the complete local scene inventory",
        enforce_configured_gates=False,
    )


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
        print_plan(demos)
        return 0

    results: dict[str, list[str]] = {
        "generated": [],
        "already-present": [],
        "capture-only": [],
        "manual-data": [],
        "unbuildable": [],
        "failed": [],
        "timeout": [],
        "no-output": [],
        "missing-script": [],
    }
    generated_paths: list[Path] = []

    with asection(f"Generating {len(demos)} gallery datasets"):
        for entry in demos:
            if dataset_exists(entry) and not args.force:
                aprint(f"✓ {entry['id']} — already present, skipping")
                results["already-present"].append(entry["id"])
                continue
            status, detail = generate_one(entry)
            results.setdefault(status, []).append(entry["id"])
            if status == "generated":
                generated_paths.append(REPO_ROOT / entry["dataset"])
            symbol = {"generated": "✅"}.get(status, "⚠️ ")
            aprint(f"{symbol} {entry['id']} — {status}: {detail}")

    with asection("Summary"):
        for status, ids in results.items():
            if ids:
                aprint(f"{status:>16}: {len(ids)}  ({', '.join(ids)})")

    ready = [d for d in demos if dataset_exists(d)]
    aprint(f"\n{len(ready)}/{len(demos)} datasets ready for capture.")
    audit_failures = audit_generated_stores(generated_paths)
    report_local_scene_inventory()
    if not audit_failures:
        aprint("Next: cd packages/luxar-viewer && pnpm gallery")

    # Non-zero only on hard generation failures. capture-only, manual-data and
    # unbuildable are expected states, not errors: manual-data is a demo whose
    # machine-local input this machine does not have, unbuildable one that cannot
    # be built anywhere yet (see `generate_one`).
    hard_failures = results["failed"] + results["timeout"] + results["no-output"]
    return 1 if hard_failures or audit_failures else 0


if __name__ == "__main__":
    raise SystemExit(main())
