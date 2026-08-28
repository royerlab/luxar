"""Stable fingerprints for Python sources that produce Luxar stores."""

from __future__ import annotations

import ast
import hashlib
import os
from functools import lru_cache
from pathlib import Path
from typing import Iterable, Sequence

import numcodecs
import zarr

ZARR_FORMAT_ENV_VAR = "LUXAR_ZARR_FORMAT"


def fingerprint_source_files(
    root: Path,
    paths: Iterable[Path],
    source_hash_cache: dict[Path, bytes] | None = None,
) -> str:
    """Hash source paths and contents in a stable, boundary-safe order."""
    cache = {} if source_hash_cache is None else source_hash_cache
    digest = hashlib.sha256()
    for path in sorted(paths, key=lambda item: item.relative_to(root).as_posix()):
        relative = path.relative_to(root).as_posix().encode()
        digest.update(len(relative).to_bytes(4, "big"))
        digest.update(relative)
        source_hash = cache.get(path)
        if source_hash is None:
            source_hash = hashlib.sha256(path.read_bytes()).digest()
            cache[path] = source_hash
        digest.update(source_hash)
    return digest.hexdigest()


def _module_path(
    module: str,
    import_roots: Sequence[Path],
    cache: dict[str, Path | None],
) -> Path | None:
    if module in cache:
        return cache[module]
    relative = Path(*module.split("."))
    for root in import_roots:
        module_file = root / relative.with_suffix(".py")
        if module_file.is_file():
            cache[module] = module_file
            return module_file
        package_file = root / relative / "__init__.py"
        if package_file.is_file():
            cache[module] = package_file
            return package_file
    cache[module] = None
    return None


def _module_name(path: Path, import_roots: Sequence[Path]) -> str:
    for root in import_roots:
        if not path.is_relative_to(root):
            continue
        parts = list(path.relative_to(root).with_suffix("").parts)
        if parts[-1] == "__init__":
            parts.pop()
        return ".".join(parts)
    return path.stem


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


def imported_source_files(
    module_file: Path,
    import_roots: Iterable[Path],
    *,
    import_cache: dict[Path, set[str]] | None = None,
    module_cache: dict[str, Path | None] | None = None,
) -> tuple[Path, ...]:
    """Return a module and local sources reachable through static imports.

    Imports assembled dynamically from strings and non-Python inputs are outside
    this source graph and must be invalidated by their caller's own provenance.
    """
    source = module_file.resolve()
    roots = tuple(Path(root).resolve() for root in import_roots)
    imports = {} if import_cache is None else import_cache
    modules = {} if module_cache is None else module_cache
    sources = {source}
    pending = list(_imported_modules(source, _module_name(source, roots), imports))
    visited: set[str] = set()
    while pending:
        module = pending.pop()
        if module in visited:
            continue
        visited.add(module)
        parts = module.split(".")
        pending.extend(".".join(parts[:index]) for index in range(1, len(parts)))
        path = _module_path(module, roots, modules)
        if path is None or path in sources:
            continue
        sources.add(path)
        pending.extend(_imported_modules(path, _module_name(path, roots), imports))
    return tuple(sorted(sources, key=lambda path: path.as_posix()))


def fingerprint_imported_sources(
    root: Path,
    module_file: Path,
    import_roots: Iterable[Path],
    *,
    import_cache: dict[Path, set[str]] | None = None,
    module_cache: dict[str, Path | None] | None = None,
    source_hash_cache: dict[Path, bytes] | None = None,
) -> str:
    """Hash a module and the local Python sources reachable from its imports."""
    sources = imported_source_files(
        module_file,
        import_roots,
        import_cache=import_cache,
        module_cache=module_cache,
    )
    return fingerprint_source_files(root.resolve(), sources, source_hash_cache)


def fingerprint_production_sources(package_root: Path | None = None) -> str:
    """Hash production Luxar Python sources under one package root."""
    root = (
        Path(__file__).resolve().parents[1]
        if package_root is None
        else Path(package_root).resolve()
    )
    paths = tuple(
        path
        for path in root.rglob("*.py")
        if "tests" not in path.relative_to(root).parts
        and "__pycache__" not in path.relative_to(root).parts
        and path.name != "conftest.py"
    )
    if not paths:
        return ""
    try:
        return fingerprint_source_files(root, paths)
    except OSError:
        return ""


@lru_cache(maxsize=None)
def production_source_fingerprint(package_root: Path | None = None) -> str:
    """Hash production sources once per package root and process."""
    return fingerprint_production_sources(package_root)


def store_writer_environment() -> dict[str, str | None]:
    """Return installed and configured inputs that affect Zarr output."""
    return {
        ZARR_FORMAT_ENV_VAR: os.environ.get(ZARR_FORMAT_ENV_VAR),
        "numcodecs": numcodecs.__version__,
        "zarr": zarr.__version__,
    }
