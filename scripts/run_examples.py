#!/usr/bin/env python3
"""Generate example datasets only when their producer inputs changed."""

from __future__ import annotations

import hashlib
import json
import shutil
import subprocess
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
OUTPUT_DIR = REPO_ROOT / "datasets/examples"
MARKER_NAME = ".fixture-build.json"
MARKER_VERSION = 1


def _source_files(repo_root: Path) -> list[Path]:
    examples = (repo_root / "packages/luxar/examples").glob("*.py")
    production = (
        path
        for path in (repo_root / "packages/luxar/src/luxar").rglob("*.py")
        if "tests" not in path.parts and "__pycache__" not in path.parts
    )
    fixed = [repo_root / "pyproject.toml", repo_root / "scripts/run_examples.py"]
    return sorted(
        (path for path in [*examples, *production, *fixed] if path.is_file()),
        key=lambda path: path.relative_to(repo_root).as_posix(),
    )


def source_fingerprint(repo_root: Path = REPO_ROOT) -> str:
    """Hash example builders and the production Python code that writes them."""
    digest = hashlib.sha256()
    for path in _source_files(repo_root):
        relative = path.relative_to(repo_root).as_posix().encode()
        digest.update(len(relative).to_bytes(4, "big"))
        digest.update(relative)
        payload = path.read_bytes()
        digest.update(len(payload).to_bytes(8, "big"))
        digest.update(payload)
    return digest.hexdigest()


def _marker_path(output_dir: Path) -> Path:
    return output_dir / MARKER_NAME


def fixtures_are_current(
    repo_root: Path = REPO_ROOT, output_dir: Path = OUTPUT_DIR
) -> bool:
    """Return whether the stamp matches and every stamped dataset still exists."""
    try:
        marker = json.loads(_marker_path(output_dir).read_text())
    except (FileNotFoundError, json.JSONDecodeError, OSError):
        return False
    outputs = marker.get("outputs")
    actual_outputs = sorted(
        path.name for path in output_dir.glob("*.zarr") if path.is_dir()
    )
    return (
        marker.get("version") == MARKER_VERSION
        and marker.get("fingerprint") == source_fingerprint(repo_root)
        and isinstance(outputs, list)
        and bool(outputs)
        and all(isinstance(name, str) for name in outputs)
        and outputs == actual_outputs
    )


def write_marker(repo_root: Path = REPO_ROOT, output_dir: Path = OUTPUT_DIR) -> None:
    """Atomically stamp the source fingerprint and generated dataset inventory."""
    output_dir.mkdir(parents=True, exist_ok=True)
    outputs = sorted(path.name for path in output_dir.glob("*.zarr") if path.is_dir())
    if not outputs:
        raise RuntimeError("example generation produced no .zarr datasets")
    marker = _marker_path(output_dir)
    temporary = marker.with_suffix(".tmp")
    temporary.write_text(
        json.dumps(
            {
                "version": MARKER_VERSION,
                "fingerprint": source_fingerprint(repo_root),
                "outputs": outputs,
            },
            indent=2,
        )
        + "\n"
    )
    temporary.replace(marker)


def _clean_generated_outputs(output_dir: Path) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    _marker_path(output_dir).unlink(missing_ok=True)
    for path in output_dir.glob("*.zarr"):
        if path.is_dir():
            shutil.rmtree(path)


def generate_examples(
    repo_root: Path = REPO_ROOT,
    output_dir: Path = OUTPUT_DIR,
    *,
    python: str = sys.executable,
) -> int:
    """Build every example, continuing after failures, and stamp only success."""
    if fixtures_are_current(repo_root, output_dir):
        print("✅ Example datasets are current; nothing to rebuild.")
        return 0

    scripts = sorted((repo_root / "packages/luxar/examples").glob("*_example.py"))
    if not scripts:
        print("❌ No example scripts found.", file=sys.stderr)
        return 1

    _clean_generated_outputs(output_dir)
    failures: list[str] = []
    print("🚀 Rebuilding example datasets from current producer sources...")
    print(f"📂 Output directory: {output_dir.relative_to(repo_root)}")
    print("━" * 48)
    for index, script in enumerate(scripts, start=1):
        print(f"\n[{index}/{len(scripts)}] 📊 Running {script.name}...")
        print("─" * 48)
        result = subprocess.run([python, str(script)], cwd=repo_root, check=False)
        if result.returncode == 0:
            print(f"✅ Success: {script.name}")
        else:
            failures.append(script.name)
            print(f"❌ Failed: {script.name}")

    print("\n" + "━" * 48)
    if failures:
        print(f"❌ Examples FAILED: {' '.join(failures)}")
        return 1
    try:
        write_marker(repo_root, output_dir)
    except RuntimeError as error:
        print(f"❌ {error}", file=sys.stderr)
        return 1
    print("✅ All examples completed and stamped current!")
    return 0


def main() -> int:
    return generate_examples()


if __name__ == "__main__":
    raise SystemExit(main())
