#!/usr/bin/env python3
"""Generate example datasets only when their producer inputs changed."""

from __future__ import annotations

import argparse
import hashlib
import json
import shutil
import subprocess
import sys
from pathlib import Path
from typing import Any, Sequence

import luxar
from luxar.utils.source_fingerprints import (
    fingerprint_production_sources,
    fingerprint_source_files,
    store_writer_environment,
)

REPO_ROOT = Path(__file__).resolve().parent.parent
OUTPUT_DIR = REPO_ROOT / "datasets/examples"
MARKER_NAME = ".fixture-build.json"
MARKER_VERSION = 2
# Keep synchronized with packages/luxar-viewer/tools/example-fixture-freshness.ts.
STALE_EXIT_CODE = 3
_WARNING_ERROR_BOOTSTRAP = """
import runpy
import sys
import warnings
from pathlib import Path

from luxar.io import ElementCapacityWarning

warnings.simplefilter("error", ElementCapacityWarning)
script = sys.argv[1]
sys.argv = sys.argv[1:]
sys.path[0] = str(Path(script).resolve().parent)
runpy.run_path(script, run_name="__main__")
"""


def source_fingerprint(repo_root: Path = REPO_ROOT) -> str:
    """Hash example builders and the production Python code that writes them."""
    package_root = repo_root / "packages/luxar/src/luxar"
    production = fingerprint_production_sources(package_root)
    examples = (repo_root / "packages/luxar/examples").glob("*.py")
    fixed = [repo_root / "pyproject.toml", repo_root / "scripts/run_examples.py"]
    supplemental = fingerprint_source_files(
        repo_root,
        (path for path in [*examples, *fixed] if path.is_file()),
    )
    digest = hashlib.sha256()
    if production:
        digest.update(bytes.fromhex(production))
    digest.update(bytes.fromhex(supplemental))
    return digest.hexdigest()


build_environment = store_writer_environment


def luxar_is_from_repo(repo_root: Path = REPO_ROOT) -> bool:
    """Return whether the imported Luxar package belongs to this checkout."""
    package_root = (repo_root / "packages/luxar/src/luxar").resolve()
    return Path(luxar.__file__).resolve().is_relative_to(package_root)


def _marker_path(output_dir: Path) -> Path:
    return output_dir / MARKER_NAME


def _read_marker(output_dir: Path) -> dict[str, Any] | None:
    try:
        marker = json.loads(_marker_path(output_dir).read_text())
    except (FileNotFoundError, json.JSONDecodeError, OSError):
        return None
    return marker if isinstance(marker, dict) else None


def _marker_outputs(marker: dict[str, Any] | None) -> list[str]:
    if marker is None:
        return []
    outputs = marker.get("outputs")
    if not isinstance(outputs, list) or not all(
        isinstance(name, str) for name in outputs
    ):
        return []
    return outputs


def fixtures_are_current(
    repo_root: Path = REPO_ROOT, output_dir: Path = OUTPUT_DIR
) -> bool:
    """Return whether the stamp matches and every stamped dataset still exists."""
    marker = _read_marker(output_dir)
    if marker is None:
        return False
    outputs = _marker_outputs(marker)
    return (
        marker.get("version") == MARKER_VERSION
        and marker.get("fingerprint") == source_fingerprint(repo_root)
        and marker.get("environment") == build_environment()
        and bool(outputs)
        and all((output_dir / name).is_dir() for name in outputs)
    )


def write_marker(
    repo_root: Path = REPO_ROOT,
    output_dir: Path = OUTPUT_DIR,
    *,
    fingerprint: str | None = None,
    environment: dict[str, str | None] | None = None,
    outputs: Sequence[str] | None = None,
) -> None:
    """Atomically stamp the source fingerprint and generated dataset inventory."""
    output_dir.mkdir(parents=True, exist_ok=True)
    generated_outputs = (
        sorted(path.name for path in output_dir.glob("*.zarr") if path.is_dir())
        if outputs is None
        else sorted(outputs)
    )
    if not generated_outputs:
        raise RuntimeError("example generation produced no .zarr datasets")
    marker = _marker_path(output_dir)
    temporary = marker.with_suffix(".tmp")
    temporary.write_text(
        json.dumps(
            {
                "version": MARKER_VERSION,
                "fingerprint": fingerprint or source_fingerprint(repo_root),
                "environment": environment or build_environment(),
                "outputs": generated_outputs,
            },
            indent=2,
        )
        + "\n"
    )
    temporary.replace(marker)


def _output_signatures(output_dir: Path) -> dict[str, tuple[tuple[str, int, int], ...]]:
    signatures: dict[str, tuple[tuple[str, int, int], ...]] = {}
    for output in output_dir.glob("*.zarr"):
        if not output.is_dir():
            continue
        entries: list[tuple[str, int, int]] = []
        for path in sorted(output.rglob("*")):
            stat = path.stat()
            entries.append(
                (path.relative_to(output).as_posix(), stat.st_size, stat.st_mtime_ns)
            )
        signatures[output.name] = tuple(entries)
    return signatures


def generate_examples(
    repo_root: Path = REPO_ROOT,
    output_dir: Path = OUTPUT_DIR,
    *,
    python: str = sys.executable,
    force: bool = False,
) -> int:
    """Build every example, continuing after failures, and stamp only success."""
    if not force and fixtures_are_current(repo_root, output_dir):
        print("✅ Example datasets are current; nothing to rebuild.", flush=True)
        return 0

    scripts = sorted((repo_root / "packages/luxar/examples").glob("*_example.py"))
    if not scripts:
        print("❌ No example scripts found.", file=sys.stderr, flush=True)
        return 1

    output_dir.mkdir(parents=True, exist_ok=True)
    previous_outputs = _marker_outputs(_read_marker(output_dir))
    before_signatures = _output_signatures(output_dir)
    fingerprint = source_fingerprint(repo_root)
    environment = build_environment()
    _marker_path(output_dir).unlink(missing_ok=True)
    failures: list[str] = []
    print("🚀 Rebuilding example datasets from current producer sources...", flush=True)
    print(f"📂 Output directory: {output_dir.relative_to(repo_root)}", flush=True)
    print("━" * 48, flush=True)
    for index, script in enumerate(scripts, start=1):
        print(f"\n[{index}/{len(scripts)}] 📊 Running {script.name}...", flush=True)
        print("─" * 48, flush=True)
        result = subprocess.run(
            [
                python,
                "-c",
                _WARNING_ERROR_BOOTSTRAP,
                str(script),
            ],
            cwd=repo_root,
            check=False,
        )
        if result.returncode == 0:
            print(f"✅ Success: {script.name}", flush=True)
        else:
            failures.append(script.name)
            print(f"❌ Failed: {script.name}", flush=True)

    print("\n" + "━" * 48, flush=True)
    if failures:
        print(f"❌ Examples FAILED: {' '.join(failures)}", flush=True)
        return 1
    after_signatures = _output_signatures(output_dir)
    generated_outputs = sorted(
        name
        for name, signature in after_signatures.items()
        if before_signatures.get(name) != signature
    )
    try:
        write_marker(
            repo_root,
            output_dir,
            fingerprint=fingerprint,
            environment=environment,
            outputs=generated_outputs,
        )
    except RuntimeError as error:
        print(f"❌ {error}", file=sys.stderr, flush=True)
        return 1
    for name in set(previous_outputs) - set(generated_outputs):
        path = output_dir / name
        if path.is_dir():
            shutil.rmtree(path)
    print("✅ All examples completed and stamped current!", flush=True)
    return 0


def main(argv: Sequence[str] | None = None) -> int:
    if not luxar_is_from_repo(REPO_ROOT):
        print(
            "❌ Imported Luxar does not belong to this checkout; run through this "
            "worktree's Hatch environment.",
            file=sys.stderr,
        )
        return 1

    parser = argparse.ArgumentParser(description=__doc__)
    mode = parser.add_mutually_exclusive_group()
    mode.add_argument("--check", action="store_true", help="check fixture freshness")
    mode.add_argument("--force", action="store_true", help="rebuild even when current")
    args = parser.parse_args(argv)

    if args.check:
        if fixtures_are_current(REPO_ROOT, OUTPUT_DIR):
            print("✅ Example datasets are current.")
            return 0
        print(
            '❌ Example datasets are stale; run "make run-examples" to rebuild them.',
            file=sys.stderr,
        )
        return STALE_EXIT_CODE
    return generate_examples(REPO_ROOT, OUTPUT_DIR, force=args.force)


if __name__ == "__main__":
    raise SystemExit(main())
