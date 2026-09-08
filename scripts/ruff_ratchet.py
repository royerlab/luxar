"""Shared fail-closed guards for Ruff-backed baseline ratchets."""

from __future__ import annotations

import hashlib
import json
import re
import subprocess
import sys
import tomllib
from collections.abc import Mapping, Sequence
from pathlib import Path

_UNSCANNED_RE = re.compile(r"Failed to lint ")
_RUFF_CONFIG_NAMES = ("ruff.toml", ".ruff.toml", "pyproject.toml")


def baseline_entries_for_update(path: Path, field: str) -> dict[str, object]:
    """Recover existing baseline keys without validating stale metadata."""
    if not path.exists():
        return {}
    try:
        data = json.loads(path.read_text())
    except (json.JSONDecodeError, OSError):
        return {}
    if not isinstance(data, dict):
        return {}
    entries = data.get(field)
    return dict(entries) if isinstance(entries, dict) else {}


def format_retired_keys(
    previous: Mapping[str, object], current: Mapping[str, object]
) -> str | None:
    """Describe baseline keys a deliberate settings refresh will retire."""
    retired = sorted(previous.keys() - current.keys())
    if not retired:
        return None
    shown = "\n".join(f"  {key}" for key in retired[:20])
    suffix = f"\n  ... and {len(retired) - 20} more" if len(retired) > 20 else ""
    noun = "key" if len(retired) == 1 else "keys"
    return f"Retiring {len(retired)} baseline {noun} after the settings refresh:\n{shown}{suffix}"


def settings_fingerprint(
    target: str,
    project_root: Path,
    selectors: Sequence[str],
) -> str:
    """Hash Ruff's checkout-independent resolved settings for ``selectors``."""
    command = [
        sys.executable,
        "-m",
        "ruff",
        "check",
        "--show-settings",
        "--select",
        ",".join(selectors),
        target,
    ]
    try:
        proc = subprocess.run(command, cwd=project_root, capture_output=True, text=True)
    except OSError as exc:  # pragma: no cover - environment failure
        raise RuntimeError(f"Could not run ruff ({' '.join(command)}): {exc}") from exc

    if proc.returncode != 0:
        if "No files found under the given path" in proc.stderr:
            raise RuntimeError(
                "cannot resolve Ruff settings because the target has no Python "
                f"files: {target}"
            )
        raise RuntimeError(
            f"ruff --show-settings failed (exit {proc.returncode}): "
            f"{' '.join(command)}\n{proc.stderr.strip()}"
        )

    resolved_lines = [
        line
        for line in proc.stdout.splitlines()
        if not line.startswith("Resolved settings for:")
    ]
    normalized = "\n".join(resolved_lines)
    for root in {str(project_root), project_root.as_posix()}:
        normalized = normalized.replace(root, "")
    if not normalized.strip():
        raise RuntimeError(
            "ruff --show-settings returned no resolved settings, so the lint "
            "configuration fingerprint cannot be trusted"
        )
    return hashlib.sha256(normalized.encode()).hexdigest()


def _ruff_config_candidates(target: str, project_root: Path) -> set[Path]:
    """Return possible Ruff config files affecting one lint target."""
    candidates: set[Path] = set()
    target_path = Path(target)
    if not target_path.is_absolute():
        target_path = project_root / target_path
    target_path = target_path.resolve()
    target_dir = target_path if target_path.is_dir() else target_path.parent

    if target_path.is_dir():
        for name in _RUFF_CONFIG_NAMES:
            candidates.update(target_dir.rglob(name))

    for directory in (target_dir, *target_dir.parents):
        if directory == project_root or project_root not in directory.parents:
            break
        candidates.update(
            config
            for name in _RUFF_CONFIG_NAMES
            if (config := directory / name).is_file()
        )
    return candidates


def _is_nested_ruff_config(config: Path, project_root: Path) -> bool:
    """Return whether ``config`` is a non-root config Ruff will consume."""
    if config.parent == project_root:
        return False
    if config.name != "pyproject.toml":
        return True
    try:
        pyproject = tomllib.loads(config.read_text())
    except (OSError, UnicodeDecodeError, tomllib.TOMLDecodeError) as exc:
        raise RuntimeError(f"Could not inspect nested {config}: {exc}") from exc
    tool = pyproject.get("tool")
    return isinstance(tool, dict) and "ruff" in tool


def find_nested_configs(targets: Sequence[str], project_root: Path) -> list[Path]:
    """Return non-root Ruff config files that can affect ``targets``."""
    project_root = project_root.resolve()
    candidates: set[Path] = set()
    for target in targets:
        candidates.update(_ruff_config_candidates(target, project_root))
    return [
        config
        for config in sorted(candidates)
        if _is_nested_ruff_config(config, project_root)
    ]


def ensure_no_nested_configs(targets: Sequence[str], project_root: Path) -> None:
    """Reject hierarchical Ruff configs outside the root fingerprint."""
    nested_configs = find_nested_configs(targets, project_root)
    if not nested_configs:
        return
    displayed = []
    for config in nested_configs:
        try:
            displayed.append(config.relative_to(project_root.resolve()).as_posix())
        except ValueError:
            displayed.append(str(config))
    raise RuntimeError(
        "Nested Ruff configuration cannot be checked against the root settings "
        "fingerprint. Move these settings into the repository-root config:\n"
        + "\n".join(f"  {config}" for config in displayed)
    )


def list_files(
    targets: Sequence[str],
    project_root: Path,
    selectors: Sequence[str],
) -> set[str]:
    """Return the repo-relative files Ruff says it will scan."""
    command = [
        sys.executable,
        "-m",
        "ruff",
        "check",
        "--show-files",
        "--select",
        ",".join(selectors),
        *targets,
    ]
    try:
        proc = subprocess.run(command, cwd=project_root, capture_output=True, text=True)
    except OSError as exc:  # pragma: no cover - environment failure
        raise RuntimeError(f"Could not run ruff ({' '.join(command)}): {exc}") from exc

    if proc.returncode != 0:
        raise RuntimeError(
            f"ruff --show-files failed (exit {proc.returncode}): "
            f"{' '.join(command)}\n{proc.stderr.strip()}"
        )

    unscanned = [
        line for line in proc.stderr.splitlines() if _UNSCANNED_RE.search(line)
    ]
    if unscanned:
        raise RuntimeError(
            "ruff could not enumerate every target, so the scan is PARTIAL:\n"
            + "\n".join(f"  {line}" for line in unscanned)
        )

    files: set[str] = set()
    for line in proc.stdout.splitlines():
        path = Path(line)
        try:
            files.add(path.relative_to(project_root).as_posix())
        except ValueError:
            files.add(path.as_posix())
    return files


def ensure_baselined_files_scanned(
    baseline: Mapping[str, object], scanned_files: set[str], project_root: Path
) -> None:
    """Fail when an existing baselined file was silently excluded by Ruff."""
    omitted_paths = sorted(
        path
        for path in {key.rpartition("::")[0] for key in baseline}
        if (project_root / path).exists() and path not in scanned_files
    )
    if omitted_paths:
        omitted = [
            key for key in sorted(baseline) if key.rpartition("::")[0] in omitted_paths
        ]
        shown = "\n".join(f"  {key}" for key in omitted[:20])
        suffix = f"\n  ... and {len(omitted) - 20} more" if len(omitted) > 20 else ""
        raise RuntimeError(
            "ruff omitted existing baselined files, so the scan is PARTIAL. "
            "Check Ruff excludes and the checkout. If the exclusion is "
            "deliberate, remove those files' keys from the baseline by hand; "
            "--update-baseline is blocked by this same guard. These keys would "
            "be dropped:\n"
            f"{shown}{suffix}"
        )
