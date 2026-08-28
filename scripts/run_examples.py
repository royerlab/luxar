#!/usr/bin/env python3
"""Generate example datasets only when their producer inputs changed."""

from __future__ import annotations

import argparse
import ast
import hashlib
import json
import shutil
import subprocess
import sys
from pathlib import Path
from typing import Any, Mapping, Sequence

import luxar
from luxar.utils.source_fingerprints import store_writer_environment

REPO_ROOT = Path(__file__).resolve().parent.parent
OUTPUT_DIR = REPO_ROOT / "datasets/examples"
MARKER_NAME = ".fixture-build.json"
MARKER_VERSION = 3
# Keep synchronized with packages/luxar-viewer/tools/example-fixture-freshness.ts.
STALE_EXIT_CODE = 3


def _module_path(
    repo_root: Path, module: str, cache: dict[str, Path | None]
) -> Path | None:
    if module in cache:
        return cache[module]
    if module == "luxar" or module.startswith("luxar."):
        root = repo_root / "packages/luxar/src"
    elif "." not in module:
        root = repo_root / "packages/luxar/examples"
    else:
        cache[module] = None
        return None
    relative = Path(*module.split(".")) if module.startswith("luxar") else Path(module)
    module_file = root / relative.with_suffix(".py")
    if module_file.is_file():
        cache[module] = module_file
        return module_file
    package_file = root / relative / "__init__.py"
    resolved = package_file if package_file.is_file() else None
    cache[module] = resolved
    return resolved


def _module_name(repo_root: Path, path: Path) -> str:
    package_root = repo_root / "packages/luxar/src"
    examples_root = repo_root / "packages/luxar/examples"
    if path.is_relative_to(package_root):
        relative = path.relative_to(package_root)
        parts = list(relative.with_suffix("").parts)
        if parts[-1] == "__init__":
            parts.pop()
        return ".".join(parts)
    return path.relative_to(examples_root).stem


def _imported_modules(path: Path, module: str, cache: dict[Path, set[str]]) -> set[str]:
    if path in cache:
        return cache[path]
    try:
        tree = ast.parse(path.read_text())
    except (OSError, SyntaxError, UnicodeError):
        return set()
    imported: set[str] = set()
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            imported.update(alias.name for alias in node.names)
            continue
        if not isinstance(node, ast.ImportFrom):
            continue
        if node.level:
            package = module.split(".")
            if path.name != "__init__.py":
                package.pop()
            ascend = node.level - 1
            if ascend > len(package):
                continue
            prefix = package[: len(package) - ascend]
            base = ".".join([*prefix, *(node.module or "").split(".")]).rstrip(".")
        else:
            base = node.module or ""
        if base:
            imported.add(base)
        imported.update(
            f"{base}.{alias.name}" if base else alias.name
            for alias in node.names
            if alias.name != "*"
        )
    cache[path] = imported
    return imported


def example_source_files(
    repo_root: Path,
    script: Path,
    *,
    import_cache: dict[Path, set[str]] | None = None,
    module_cache: dict[str, Path | None] | None = None,
) -> list[Path]:
    """Return local sources imported by one example, including shared helpers."""
    imports = {} if import_cache is None else import_cache
    modules = {} if module_cache is None else module_cache
    sources = {script}
    pending = list(_imported_modules(script, script.stem, imports))
    visited: set[str] = set()
    while pending:
        module = pending.pop()
        if module in visited:
            continue
        visited.add(module)
        if module.startswith("luxar."):
            parts = module.split(".")
            pending.extend(".".join(parts[:index]) for index in range(1, len(parts)))
        path = _module_path(repo_root, module, modules)
        if path is None or path in sources:
            continue
        sources.add(path)
        pending.extend(_imported_modules(path, _module_name(repo_root, path), imports))
    return sorted(sources, key=lambda path: path.relative_to(repo_root).as_posix())


def example_fingerprint(
    repo_root: Path,
    script: Path,
    *,
    import_cache: dict[Path, set[str]] | None = None,
    module_cache: dict[str, Path | None] | None = None,
    source_hash_cache: dict[Path, bytes] | None = None,
) -> str:
    """Hash one example and the local sources reachable from its imports."""
    sources = example_source_files(
        repo_root,
        script,
        import_cache=import_cache,
        module_cache=module_cache,
    )
    return _fingerprint_sources(repo_root, sources, source_hash_cache)


def _fingerprint_sources(
    repo_root: Path,
    sources: Sequence[Path],
    source_hash_cache: dict[Path, bytes] | None = None,
) -> str:
    cache = {} if source_hash_cache is None else source_hash_cache
    digest = hashlib.sha256()
    for path in sources:
        relative = path.relative_to(repo_root).as_posix().encode()
        digest.update(len(relative).to_bytes(4, "big"))
        digest.update(relative)
        source_hash = cache.get(path)
        if source_hash is None:
            source_hash = hashlib.sha256(path.read_bytes()).digest()
            cache[path] = source_hash
        digest.update(source_hash)
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


def _marker_examples(marker: dict[str, Any] | None) -> dict[str, dict[str, Any]]:
    if marker is None:
        return {}
    examples = marker.get("examples")
    if not isinstance(examples, dict):
        return {}
    stamps: dict[str, dict[str, Any]] = {}
    recorded_outputs: set[str] = set()
    for producer, stamp in examples.items():
        if not isinstance(producer, str) or not isinstance(stamp, dict):
            return {}
        fingerprint = stamp.get("fingerprint")
        sources = stamp.get("sources")
        outputs = stamp.get("outputs")
        if (
            not isinstance(fingerprint, str)
            or not isinstance(sources, list)
            or not isinstance(outputs, list)
        ):
            return {}
        if (
            not sources
            or not all(isinstance(name, str) for name in sources)
            or not outputs
            or not all(isinstance(name, str) for name in outputs)
        ):
            return {}
        source_paths = [Path(name) for name in sources]
        output_paths = [Path(name) for name in outputs]
        if (
            Path(producer).name != producer
            or not producer.endswith("_example.py")
            or any(path.is_absolute() or ".." in path.parts for path in source_paths)
            or any(path.name != name for path, name in zip(output_paths, outputs))
            or recorded_outputs.intersection(outputs)
        ):
            return {}
        recorded_outputs.update(outputs)
        stamps[producer] = {
            "fingerprint": fingerprint,
            "sources": sources,
            "outputs": outputs,
        }
    return stamps


def _marker_outputs(marker: dict[str, Any] | None) -> list[str]:
    if marker is not None and isinstance(marker.get("outputs"), list):
        outputs = marker["outputs"]
        if all(isinstance(name, str) for name in outputs):
            return sorted(outputs)
    return sorted(
        name for stamp in _marker_examples(marker).values() for name in stamp["outputs"]
    )


def _example_scripts(repo_root: Path) -> list[Path]:
    return sorted((repo_root / "packages/luxar/examples").glob("*_example.py"))


def _stale_producers(
    scripts: Sequence[Path],
    marker: dict[str, Any] | None,
    repo_root: Path,
    output_dir: Path,
) -> list[str]:
    if (
        marker is None
        or marker.get("version") != MARKER_VERSION
        or marker.get("environment") != build_environment()
    ):
        return [script.name for script in scripts]
    stamps = _marker_examples(marker)
    script_names = {script.name for script in scripts}
    stale = sorted(set(stamps) - script_names)
    source_hash_cache: dict[Path, bytes] = {}
    for script in scripts:
        stamp = stamps.get(script.name)
        if stamp is None:
            stale.append(script.name)
            continue
        sources = [repo_root / name for name in stamp["sources"]]
        try:
            fingerprint = _fingerprint_sources(repo_root, sources, source_hash_cache)
        except OSError:
            fingerprint = None
        if fingerprint != stamp["fingerprint"] or not all(
            (output_dir / name).is_dir() for name in stamp["outputs"]
        ):
            stale.append(script.name)
    return stale


def stale_examples(
    repo_root: Path = REPO_ROOT, output_dir: Path = OUTPUT_DIR
) -> list[str]:
    """Return producers whose fingerprint, environment, or outputs are stale."""
    scripts = _example_scripts(repo_root)
    marker = _read_marker(output_dir)
    return _stale_producers(scripts, marker, repo_root, output_dir)


def fixtures_are_current(
    repo_root: Path = REPO_ROOT, output_dir: Path = OUTPUT_DIR
) -> bool:
    """Return whether the stamp matches and every stamped dataset still exists."""
    marker = _read_marker(output_dir)
    return (
        bool(_example_scripts(repo_root))
        and marker is not None
        and marker.get("version") == MARKER_VERSION
        and marker.get("environment") == build_environment()
        and not stale_examples(repo_root, output_dir)
    )


def write_marker(
    repo_root: Path = REPO_ROOT,
    output_dir: Path = OUTPUT_DIR,
    *,
    environment: dict[str, str | None] | None = None,
    examples: Mapping[str, Mapping[str, Any]],
) -> None:
    """Atomically stamp per-producer fingerprints and output inventories."""
    output_dir.mkdir(parents=True, exist_ok=True)
    if not examples:
        raise RuntimeError("example generation produced no .zarr datasets")
    serialized = {
        producer: {
            "fingerprint": stamp["fingerprint"],
            "sources": sorted(stamp["sources"]),
            "outputs": sorted(stamp["outputs"]),
        }
        for producer, stamp in sorted(examples.items())
    }
    marker = _marker_path(output_dir)
    temporary = marker.with_suffix(".tmp")
    temporary.write_text(
        json.dumps(
            {
                "version": MARKER_VERSION,
                "environment": environment or build_environment(),
                "examples": serialized,
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
    """Build stale examples, continuing after failures, and stamp each success."""
    if not force and fixtures_are_current(repo_root, output_dir):
        print("✅ Example datasets are current; nothing to rebuild.", flush=True)
        return 0

    scripts = _example_scripts(repo_root)
    if not scripts:
        print("❌ No example scripts found.", file=sys.stderr, flush=True)
        return 1

    output_dir.mkdir(parents=True, exist_ok=True)
    marker = _read_marker(output_dir)
    previous_stamps = _marker_examples(marker)
    stamps = dict(previous_stamps)
    legacy_outputs = (
        set(_marker_outputs(marker))
        if marker is not None and marker.get("version") != MARKER_VERSION
        else set()
    )
    environment = build_environment()
    if (
        marker is None
        or marker.get("version") != MARKER_VERSION
        or marker.get("environment") != environment
    ):
        stamps = {}
    current_names = {script.name for script in scripts}
    removed_outputs = {
        output
        for producer, stamp in previous_stamps.items()
        if producer not in current_names
        for output in stamp["outputs"]
    }
    for producer in set(stamps) - current_names:
        del stamps[producer]
    stale_names = (
        current_names
        if force
        else set(_stale_producers(scripts, marker, repo_root, output_dir))
        & current_names
    )
    for producer in stale_names:
        stamps.pop(producer, None)
    if stamps:
        write_marker(repo_root, output_dir, environment=environment, examples=stamps)
    else:
        _marker_path(output_dir).unlink(missing_ok=True)
    scripts_to_run = [script for script in scripts if script.name in stale_names]
    import_cache: dict[Path, set[str]] = {}
    module_cache: dict[str, Path | None] = {}
    source_hash_cache: dict[Path, bytes] = {}
    sources = {
        script.name: example_source_files(
            repo_root,
            script,
            import_cache=import_cache,
            module_cache=module_cache,
        )
        for script in scripts_to_run
    }
    fingerprints = {
        script.name: _fingerprint_sources(
            repo_root, sources[script.name], source_hash_cache
        )
        for script in scripts_to_run
    }
    failures: list[str] = []
    print(
        f"🚀 Rebuilding {len(scripts_to_run)} stale example dataset producer(s)...",
        flush=True,
    )
    print(f"📂 Output directory: {output_dir.relative_to(repo_root)}", flush=True)
    print("━" * 48, flush=True)
    for index, script in enumerate(scripts_to_run, start=1):
        print(
            f"\n[{index}/{len(scripts_to_run)}] 📊 Running {script.name}...",
            flush=True,
        )
        print("─" * 48, flush=True)
        before_signatures = _output_signatures(output_dir)
        result = subprocess.run([python, str(script)], cwd=repo_root, check=False)
        after_signatures = _output_signatures(output_dir)
        generated_outputs = sorted(
            name
            for name, signature in after_signatures.items()
            if before_signatures.get(name) != signature
        )
        if result.returncode != 0 or not generated_outputs:
            failures.append(script.name)
            stamps.pop(script.name, None)
            print(f"❌ Failed: {script.name}", flush=True)
            continue
        previous_outputs = set(previous_stamps.get(script.name, {}).get("outputs", []))
        removed_outputs.update(previous_outputs - set(generated_outputs))
        stamps[script.name] = {
            "fingerprint": fingerprints[script.name],
            "sources": [
                path.relative_to(repo_root).as_posix() for path in sources[script.name]
            ],
            "outputs": generated_outputs,
        }
        write_marker(repo_root, output_dir, environment=environment, examples=stamps)
        print(f"✅ Success: {script.name}", flush=True)

    print("\n" + "━" * 48, flush=True)
    try:
        write_marker(repo_root, output_dir, environment=environment, examples=stamps)
    except RuntimeError as error:
        _marker_path(output_dir).unlink(missing_ok=True)
        print(f"❌ {error}", file=sys.stderr, flush=True)
        return 1
    if not failures:
        stamped_outputs = {
            output for stamp in stamps.values() for output in stamp["outputs"]
        }
        removed_outputs.update(legacy_outputs - stamped_outputs)
    for name in removed_outputs:
        path = output_dir / name
        if path.is_dir():
            shutil.rmtree(path)
    if failures:
        print(f"❌ Examples FAILED: {' '.join(failures)}", flush=True)
        return 1
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
        stale = stale_examples(REPO_ROOT, OUTPUT_DIR)
        detail = ", ".join(stale) if stale else "fixture marker or producer inventory"
        print(
            "❌ Stale example producers: "
            + detail
            + '; run "make run-examples" to rebuild them.',
            file=sys.stderr,
        )
        return STALE_EXIT_CODE
    return generate_examples(REPO_ROOT, OUTPUT_DIR, force=args.force)


if __name__ == "__main__":
    raise SystemExit(main())
