#!/usr/bin/env python3
"""
Built-wheel inspector

Asserts that a built ``luxar`` wheel contains what it must and nothing it must
not, then leaves the caller to install it and run the CLI.

WHY THIS EXISTS
---------------
Nothing built, inspected or installed the Python wheel until a release tag was
pushed: ``hatch build`` appeared exactly once in the repository, in
``publish.yml`` on ``v*`` (audit A12-01). The npm side already had exactly the
right gate — ``release-readiness`` — and the Python side never mirrored it.

The failure this closes is not hypothetical. A ``.gitignore`` pattern once
matched a package directory under ``packages/luxar/src/luxar``, hatchling
dropped it from the wheel, every test stayed green (they import from the source
tree), and the break surfaced only as ``luxar --version`` failing on a clean
install. Nothing in CI could have caught it, because nothing in CI had ever
looked inside a wheel.

WHAT IT CHECKS, AND WHY EACH IS DERIVED RATHER THAN LISTED
-----------------------------------------------------------
Every check reads its expectation from the repository, so the gate cannot drift
out of agreement with the thing it is guarding:

1. **Package completeness.** Every importable subpackage under
   ``packages/luxar/src/luxar`` that the wheel configuration does not exclude
   must be in the wheel. Derived by walking ``__init__.py`` on disk — a
   hardcoded list would have to be updated by the same person who just added
   the package, i.e. exactly when it would not be.
2. **Excluded content is absent.** The ``exclude`` patterns are read from
   ``[tool.hatch.build.targets.wheel]`` in ``pyproject.toml``, so this asserts
   the build honoured its own configuration rather than restating it.
3. **No Git-LFS pointer files.** ``publish.yml``'s checkout has no ``lfs: true``,
   so a regressed exclude would ship ~130-byte text stubs where data should be.
   Those are the *plausible wrong answer* case: present, named correctly, and
   useless.
4. **No member over PyPI's 100 MB per-file limit**, which rejects the upload
   after the release has otherwise succeeded.
5. **The viewer dist is bundled** (``luxar/_viewer_dist/``), mirroring the check
   ``publish.yml`` already performs — kept so this gate is a superset of the
   release-time one rather than a divergent second opinion.

Usage::

    python scripts/check_wheel.py dist/luxar-*.whl
    python scripts/check_wheel.py dist/luxar-*.whl --project-root .
"""

from __future__ import annotations

import argparse
import re
import sys
import tomllib
import zipfile
from dataclasses import dataclass, field
from pathlib import Path

# Deliberately NOT `arbol`, which the rest of scripts/ uses and CLAUDE.md asks
# for. This one inspects a BUILT WHEEL, so it has to run where the wheel is
# built — the `wheel-viewer` CI job, which installs `hatch` and nothing else,
# and any minimal release environment. A wheel inspector that needs the project
# environment installed cannot check a wheel before that environment exists.
#
# Not theoretical: the first CI run of this gate died with
# `ModuleNotFoundError: No module named 'arbol'`. Keep this file stdlib-only.
aprint = print

PROJECT_ROOT = Path(__file__).resolve().parent.parent

#: PyPI rejects any single file in a distribution above this size.
PYPI_MAX_FILE_BYTES = 100 * 1024 * 1024

#: The first bytes of a Git-LFS pointer stub.
LFS_POINTER_PREFIX = b"version https://git-lfs.github.com/spec/v1"

#: A pointer file is a few hundred bytes at most; reading every member would be
#: wasteful, so only plausibly-sized members are sniffed.
LFS_POINTER_MAX_BYTES = 400

#: Where the package sources live, relative to the project root.
PACKAGE_SRC_RELPATH = "packages/luxar/src/luxar"

#: The wheel's top-level import package.
WHEEL_ROOT = "luxar"


@dataclass
class WheelReport:
    """Problems found in a wheel. Empty in every list means it is sound."""

    missing_packages: list[str] = field(default_factory=list)
    excluded_present: list[str] = field(default_factory=list)
    lfs_pointers: list[str] = field(default_factory=list)
    oversized: list[str] = field(default_factory=list)
    missing_viewer_dist: bool = False

    def problems(self) -> int:
        """Total number of distinct problems."""
        return (
            len(self.missing_packages)
            + len(self.excluded_present)
            + len(self.lfs_pointers)
            + len(self.oversized)
            + int(self.missing_viewer_dist)
        )


def glob_to_regex(pattern: str) -> re.Pattern[str]:
    """Translate a hatchling exclude glob into a regex over POSIX wheel paths.

    Deliberately hand-rolled rather than ``fnmatch.translate``, which maps ``*``
    onto ``.*`` and so cannot distinguish ``*`` from ``**`` — the distinction the
    exclude patterns rely on (``**/demos/data/**`` must cross directories;
    ``*.pyc`` must not). ``pathlib``'s ``full_match`` would do it, but only from
    Python 3.13, and this must run wherever the wheel is built.
    """
    out = ["^(?:.*/)?" if "/" not in pattern.rstrip("/") else "^"]
    i = 0
    while i < len(pattern):
        if pattern.startswith("**/", i):
            # Zero or more leading directories.
            out.append("(?:.*/)?")
            i += 3
        elif pattern.startswith("/**", i):
            # Everything below this directory.
            out.append("/.*")
            i += 3
        elif pattern.startswith("**", i):
            out.append(".*")
            i += 2
        elif pattern[i] == "*":
            out.append("[^/]*")
            i += 1
        elif pattern[i] == "?":
            out.append("[^/]")
            i += 1
        else:
            out.append(re.escape(pattern[i]))
            i += 1
    out.append("$")
    return re.compile("".join(out))


def read_wheel_excludes(project_root: Path) -> list[str]:
    """The ``exclude`` patterns hatchling applies to the wheel target.

    Read from ``pyproject.toml`` rather than restated here: this gate's job is
    to prove the BUILD honoured its configuration, and a second copy of the list
    could only ever agree with itself.
    """
    pyproject = project_root / "pyproject.toml"
    data = tomllib.loads(pyproject.read_text())
    try:
        wheel = data["tool"]["hatch"]["build"]["targets"]["wheel"]
    except KeyError as exc:  # pragma: no cover - malformed pyproject
        raise ValueError(
            f"{pyproject} has no [tool.hatch.build.targets.wheel] table, so the "
            "wheel's exclude policy cannot be read."
        ) from exc
    excludes = wheel.get("exclude", [])
    if not isinstance(excludes, list) or not all(isinstance(p, str) for p in excludes):
        raise ValueError(
            f"{pyproject}: [tool.hatch.build.targets.wheel] exclude must be a "
            "list of strings."
        )
    return excludes


def expected_packages(project_root: Path, excludes: list[str]) -> set[str]:
    """Importable subpackage paths that the wheel is expected to contain.

    Walks ``__init__.py`` under the package source and returns wheel-relative
    directory paths (``luxar/gsplats/lod``). Anything an exclude pattern covers
    is dropped, so the expectation and the build read the same policy.
    """
    src = project_root / PACKAGE_SRC_RELPATH
    if not src.is_dir():
        raise ValueError(f"Package source not found at {src}")

    matchers = [glob_to_regex(p) for p in excludes]
    found: set[str] = set()
    for init in src.rglob("__init__.py"):
        relative = init.relative_to(src.parent).as_posix()  # luxar/...
        if any(m.match(relative) for m in matchers):
            continue
        found.add(relative.rsplit("/", 1)[0])
    return found


def inspect_wheel(wheel_path: Path, project_root: Path) -> WheelReport:
    """Compare a built wheel against what the repository says it should be."""
    if not wheel_path.is_file():
        raise ValueError(f"No such wheel: {wheel_path}")

    with zipfile.ZipFile(wheel_path) as zf:
        infos = zf.infolist()
        names = [i.filename for i in infos]

        # A wheel with no members is not a passing wheel. Without this the
        # comparisons below all trivially succeed on a truncated or empty file.
        if not names:
            raise ValueError(f"{wheel_path} contains no members at all.")

        report = WheelReport()

        packaged = {
            name.rsplit("/", 1)[0]
            for name in names
            if name.endswith("/__init__.py") and name.startswith(f"{WHEEL_ROOT}/")
        }

        excludes = read_wheel_excludes(project_root)
        expected = expected_packages(project_root, excludes)
        report.missing_packages = sorted(expected - packaged)

        matchers = [glob_to_regex(p) for p in excludes]
        report.excluded_present = sorted(
            name for name in names if any(m.match(name) for m in matchers)
        )

        report.oversized = sorted(
            f"{i.filename} ({i.file_size / 1e6:.1f} MB)"
            for i in infos
            if i.file_size > PYPI_MAX_FILE_BYTES
        )

        for info in infos:
            if info.file_size <= LFS_POINTER_MAX_BYTES:
                if zf.read(info.filename).startswith(LFS_POINTER_PREFIX):
                    report.lfs_pointers.append(info.filename)
        report.lfs_pointers.sort()

        report.missing_viewer_dist = not any(
            name.startswith(f"{WHEEL_ROOT}/_viewer_dist/") for name in names
        )

    return report


#: How many offending members to itemise before summarising the rest. A
#: regressed exclude can match hundreds; the first twenty identify the pattern.
_MAX_ITEMISED = 20


def _problem_sections(
    report: WheelReport,
) -> list[tuple[str, list[str], str]]:
    """(heading, offending members, actionable hint) for each populated check."""
    sections = [
        (
            f"{len(report.missing_packages)} subpackage(s) exist in the source "
            "tree but are NOT in the wheel:",
            report.missing_packages,
            "A wheel can lose a package silently — every test imports from the "
            "source tree, so only a clean install notices. Check for a "
            ".gitignore pattern matching the directory.",
        ),
        (
            f"{len(report.excluded_present)} member(s) match an exclude pattern "
            "from [tool.hatch.build.targets.wheel] but shipped anyway:",
            report.excluded_present,
            "The build did not honour its own configuration.",
        ),
        (
            f"{len(report.lfs_pointers)} Git-LFS POINTER file(s) shipped:",
            report.lfs_pointers,
            "These are ~130-byte text stubs, not the data they name. The build "
            "ran without `git lfs pull`.",
        ),
        (
            f"{len(report.oversized)} member(s) exceed PyPI's 100 MB per-file "
            "limit, which rejects the upload:",
            report.oversized,
            "",
        ),
    ]
    return [s for s in sections if s[1]]


def _print_section(heading: str, items: list[str], hint: str) -> None:
    """Print one problem section, itemised up to ``_MAX_ITEMISED``."""
    aprint(f"\n❌ {heading}")
    for name in items[:_MAX_ITEMISED]:
        aprint(f"  ❌ {name}")
    if len(items) > _MAX_ITEMISED:
        aprint(f"  … and {len(items) - _MAX_ITEMISED} more")
    if hint:
        aprint(f"   {hint}")


def _print_report(report: WheelReport, wheel_path: Path, entries: int) -> int:
    """Print the human summary; returns the exit code.

    The per-problem sections are table-driven rather than five near-identical
    ``if`` blocks — which is also what keeps this under the C901 limit the
    repository ratchets.
    """
    aprint("=" * 70)
    aprint("📦 WHEEL CONTENTS")
    aprint("=" * 70)
    aprint(f"🔢 {wheel_path.name}: {entries} members")

    for heading, items, hint in _problem_sections(report):
        _print_section(heading, items, hint)

    if report.missing_viewer_dist:
        aprint(
            "\n❌ luxar/_viewer_dist/ is absent — the wheel would ship without "
            "the bundled web viewer. Run `make build-viewer` before building."
        )

    if report.problems():
        return 1

    aprint("\n✅ Wheel contents are sound.")
    return 0


def main(argv: list[str] | None = None) -> int:
    """Inspect a built wheel; returns the process exit code."""
    parser = argparse.ArgumentParser(description="Inspect a built luxar wheel")
    parser.add_argument("wheel", type=Path, help="path to the .whl to inspect")
    parser.add_argument(
        "--project-root",
        type=Path,
        default=None,
        help="repository root (default: the parent of this script's directory)",
    )
    args = parser.parse_args(argv)

    project_root = args.project_root.resolve() if args.project_root else PROJECT_ROOT

    try:
        report = inspect_wheel(args.wheel, project_root)
        with zipfile.ZipFile(args.wheel) as zf:
            entries = len(zf.namelist())
    except (ValueError, zipfile.BadZipFile, OSError) as exc:
        aprint(f"❌ {exc}")
        return 2

    return _print_report(report, args.wheel, entries)


if __name__ == "__main__":
    sys.exit(main())
