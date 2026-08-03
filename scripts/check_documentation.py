#!/usr/bin/env python3
"""
Documentation Quality Checker

Validates documentation completeness and quality across the Luxar project.
Run this script before commits or in CI/CD to ensure documentation standards.

This checker is a baseline-driven *ratchet*: pre-existing documentation debt is
tolerated via a checked-in baseline (``scripts/docs_baseline.json``), but any NEW
missing README / docstring / JSDoc finding fails the check. Regenerate the
baseline with ``--update-baseline``; tighten it (drop fixed entries) the same way
once debt is paid down.

Usage:
    python scripts/check_documentation.py [--verbose]
    python scripts/check_documentation.py --json
    python scripts/check_documentation.py --update-baseline
    python scripts/check_documentation.py --no-baseline
"""

import argparse
import json
import re
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import List, Optional, Set, Tuple

from arbol import aprint

# Baseline path relative to the project root.
DEFAULT_BASELINE_RELPATH = "scripts/docs_baseline.json"

BASELINE_COMMENT = (
    "Documentation-debt baseline for scripts/check_documentation.py. "
    "Regenerate with: hatch run python scripts/check_documentation.py "
    "--update-baseline. Each entry is '<check_name>::<repo-relative-path>'. "
    "New findings not listed here fail the check."
)


@dataclass
class CheckResult:
    """Result of a documentation check."""

    passed: bool
    file_path: str
    check_name: str
    message: str
    line_number: Optional[int] = None


# ---------------------------------------------------------------------------
# Pure, unit-testable ratchet helpers
# ---------------------------------------------------------------------------


def relative_path(file_path: str, project_root: Path) -> str:
    """Return ``file_path`` relative to ``project_root`` in POSIX form.

    Falls back to the POSIX form of the path itself if it is not located under
    ``project_root`` (keys stay stable and portable across machines/OSes).
    """
    path = Path(file_path)
    try:
        return path.relative_to(project_root).as_posix()
    except ValueError:
        return path.as_posix()


def result_key(result: CheckResult, project_root: Path) -> str:
    """Build a stable, portable key ``"<check_name>::<relative posix path>"``.

    The key deliberately omits the human ``message`` (which embeds volatile
    coverage percentages) so it is stable across runs. There is at most one
    failing result per (check_name, file), so keys are unique.
    """
    return f"{result.check_name}::{relative_path(result.file_path, project_root)}"


def failure_keys(results: List[CheckResult], project_root: Path) -> Set[str]:
    """Return the set of stable keys for all FAILED results."""
    return {result_key(r, project_root) for r in results if not r.passed}


def load_baseline(path: Path) -> Set[str]:
    """Load the baseline failure keys from ``path``.

    Returns an empty set if the file does not exist. If the file exists but is
    malformed (invalid JSON or missing/ill-typed ``failures`` list), raises a
    clear ``ValueError``.
    """
    if not path.exists():
        return set()

    try:
        data = json.loads(path.read_text())
    except json.JSONDecodeError as exc:
        raise ValueError(f"Baseline file {path} is not valid JSON: {exc}") from exc

    if not isinstance(data, dict) or "failures" not in data:
        raise ValueError(
            f"Baseline file {path} is malformed: expected a JSON object with a "
            "'failures' list."
        )

    failures = data["failures"]
    if not isinstance(failures, list) or not all(isinstance(k, str) for k in failures):
        raise ValueError(
            f"Baseline file {path} is malformed: 'failures' must be a list of strings."
        )

    return set(failures)


def save_baseline(path: Path, keys: Set[str]) -> None:
    """Write ``keys`` to ``path`` as deterministic, human-diffable JSON.

    The ``failures`` list is sorted; the file is written with ``indent=2`` and a
    trailing newline.
    """
    payload = {
        "_comment": BASELINE_COMMENT,
        "failures": sorted(keys),
    }
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2) + "\n")


def evaluate_ratchet(
    current: Set[str], baseline: Set[str]
) -> Tuple[Set[str], Set[str], Set[str]]:
    """Split findings into (new, fixed, still_present) relative to the baseline.

    - new = current - baseline (regressions that must fail the check)
    - fixed = baseline - current (debt paid down; run --update-baseline)
    - still_present = current & baseline (tolerated pre-existing debt)
    """
    new_findings = current - baseline
    fixed_findings = baseline - current
    still_present = current & baseline
    return new_findings, fixed_findings, still_present


class DocumentationChecker:
    """Checks documentation quality across Python and TypeScript files."""

    def __init__(self, project_root: Path, verbose: bool = False):
        self.project_root = project_root
        self.verbose = verbose
        self.results: List[CheckResult] = []

    def check_all(self, quiet: bool = False) -> bool:
        """Run all documentation checks. Returns True if all pass.

        When ``quiet`` is set, no human output is emitted (keeps stdout clean
        for ``--json`` so the report is machine-parseable).
        """
        if not quiet:
            aprint("🔍 Checking documentation quality...\n")

        # Check Python packages
        self._check_python_packages()

        # Check TypeScript packages
        self._check_typescript_packages()

        return all(r.passed for r in self.results)

    def _check_python_packages(self):
        """Check all Python packages for documentation completeness."""
        packages_dir = self.project_root / "packages" / "luxar" / "src" / "luxar"

        if not packages_dir.exists():
            return

        for package_dir in packages_dir.iterdir():
            if package_dir.is_dir() and not package_dir.name.startswith("_"):
                self._check_python_package(package_dir)

    def _check_python_package(self, package_dir: Path):
        """Check a single Python package."""
        package_name = package_dir.relative_to(
            self.project_root / "packages" / "luxar" / "src" / "luxar"
        )

        # Check for README.md
        readme = package_dir / "README.md"
        if not readme.exists():
            self.results.append(
                CheckResult(
                    passed=False,
                    file_path=str(readme),
                    check_name="README existence",
                    message=f"Missing README.md for package: {package_name}",
                )
            )
        else:
            self._check_readme_quality(readme, package_name)

        # Check Python files for docstrings
        for py_file in package_dir.glob("*.py"):
            if not py_file.name.startswith("_") or py_file.name == "__init__.py":
                self._check_python_file_docstrings(py_file, package_name)

    def _check_readme_quality(self, readme: Path, package_name: str):
        """Check README.md quality."""
        content = readme.read_text()

        # Check for a Quick Start or Getting Started section.
        if "## Quick Start" not in content and "## Getting Started" not in content:
            self.results.append(
                CheckResult(
                    passed=False,
                    file_path=str(readme),
                    check_name="Quick Start section",
                    message=f"README missing 'Quick Start' or 'Getting Started' section: {package_name}",
                )
            )
        else:
            self.results.append(
                CheckResult(
                    passed=True,
                    file_path=str(readme),
                    check_name="Quick Start section",
                    message=f"✓ README has Quick Start section: {package_name}",
                )
            )

        # Check minimum length (should have substance)
        if len(content) < 500:
            self.results.append(
                CheckResult(
                    passed=False,
                    file_path=str(readme),
                    check_name="README length",
                    message=f"README too short (<500 chars): {package_name}",
                )
            )

        # Check for code examples
        if "```python" not in content and "```typescript" not in content:
            self.results.append(
                CheckResult(
                    passed=False,
                    file_path=str(readme),
                    check_name="Code examples",
                    message=f"README missing code examples: {package_name}",
                )
            )

    def _check_python_file_docstrings(self, py_file: Path, package_name: str):
        """Check Python file for docstring coverage."""
        content = py_file.read_text()

        # Check for module docstring
        if not content.strip().startswith('"""') and not content.strip().startswith(
            "'''"
        ):
            self.results.append(
                CheckResult(
                    passed=False,
                    file_path=str(py_file),
                    check_name="Module docstring",
                    message=f"Missing module docstring: {package_name}/{py_file.name}",
                )
            )

        # Count functions/classes and docstrings
        func_class_pattern = r"^(def |class )"
        docstring_pattern = r'^\s+["\']'

        lines = content.split("\n")
        func_class_count = 0
        docstring_count = 0

        for i, line in enumerate(lines):
            if re.match(func_class_pattern, line):
                func_class_count += 1
                # Check if next non-empty line is a docstring
                for j in range(i + 1, min(i + 3, len(lines))):
                    if lines[j].strip():
                        if re.match(docstring_pattern, lines[j]):
                            docstring_count += 1
                        break

        if func_class_count > 0:
            coverage = (docstring_count / func_class_count) * 100
            if coverage < 70:  # Minimum 70% coverage
                self.results.append(
                    CheckResult(
                        passed=False,
                        file_path=str(py_file),
                        check_name="Docstring coverage",
                        message=f"Low docstring coverage ({coverage:.0f}%): {package_name}/{py_file.name}",
                    )
                )

    def _check_typescript_packages(self):
        """Check TypeScript packages for JSDoc coverage."""
        viewer_dir = self.project_root / "packages" / "luxar-viewer" / "src"

        if not viewer_dir.exists():
            return

        for package_dir in viewer_dir.iterdir():
            if package_dir.is_dir():
                self._check_typescript_package(package_dir)

    def _check_typescript_package(self, package_dir: Path):
        """Check a single TypeScript package."""
        package_name = package_dir.name

        # Check for README.md
        readme = package_dir / "README.md"
        if not readme.exists():
            self.results.append(
                CheckResult(
                    passed=False,
                    file_path=str(readme),
                    check_name="README existence",
                    message=f"Missing README.md for TypeScript package: {package_name}",
                )
            )

        # Check TypeScript files for JSDoc
        for ts_file in package_dir.glob("*.ts"):
            if not ts_file.name.endswith(".test.ts"):
                self._check_typescript_jsdoc(ts_file, package_name)

    def _check_typescript_jsdoc(self, ts_file: Path, package_name: str):
        """Check TypeScript file for JSDoc coverage."""
        content = ts_file.read_text()

        # Count exported functions/classes
        export_pattern = r"^export (function|class|interface|type|const)"
        jsdoc_pattern = r"/\*\*"

        lines = content.split("\n")
        export_count = 0
        jsdoc_count = 0

        for i, line in enumerate(lines):
            if re.match(export_pattern, line):
                export_count += 1
                # Check if there's a JSDoc comment before (within 10 lines)
                for j in range(max(0, i - 10), i):
                    if re.search(jsdoc_pattern, lines[j]):
                        jsdoc_count += 1
                        break

        if export_count > 0:
            coverage = (jsdoc_count / export_count) * 100
            if coverage < 70:  # Minimum 70% JSDoc coverage
                self.results.append(
                    CheckResult(
                        passed=False,
                        file_path=str(ts_file),
                        check_name="JSDoc coverage",
                        message=f"Low JSDoc coverage ({coverage:.0f}%): {package_name}/{ts_file.name}",
                    )
                )

    def print_summary(self):
        """Print the per-check human summary of all checks."""
        aprint("\n" + "=" * 70)
        aprint("📊 DOCUMENTATION QUALITY REPORT")
        aprint("=" * 70 + "\n")

        passed = [r for r in self.results if r.passed]
        failed = [r for r in self.results if not r.passed]

        aprint(f"✅ Passed: {len(passed)}")
        aprint(f"❌ Failed: {len(failed)}")
        total = len(self.results)
        pct = (len(passed) / total * 100) if total else 100.0
        aprint(f"📈 Overall: {len(passed)}/{total} ({pct:.0f}%)\n")

        if failed:
            aprint("Failed checks:\n")
            for result in failed:
                aprint(f"  ❌ {result.check_name}")
                aprint(f"     {result.message}")
                aprint(f"     File: {result.file_path}\n")

        if self.verbose and passed:
            aprint("\nPassed checks:\n")
            for result in passed:
                aprint(f"  ✅ {result.message}")


# ---------------------------------------------------------------------------
# JSON output
# ---------------------------------------------------------------------------


def build_json_report(
    results: List[CheckResult],
    project_root: Path,
    baseline: Optional[Set[str]] = None,
) -> dict:
    """Build the deterministic machine-readable report object.

    Findings (both passed and failed) are pre-sorted by key. When ``baseline``
    is provided, a top-level ``"ratchet"`` object classifies the current
    failures against it (``new`` / ``fixed`` / ``still_present``, each a sorted
    list); when it is ``None`` the ``"ratchet"`` key is ``null``.
    """
    passed = sum(1 for r in results if r.passed)
    failed = sum(1 for r in results if not r.passed)

    findings = [
        {
            "key": result_key(r, project_root),
            "check_name": r.check_name,
            "file": relative_path(r.file_path, project_root),
            "passed": r.passed,
            "message": r.message,
            "line_number": r.line_number,
        }
        for r in results
    ]
    findings.sort(key=lambda f: f["key"])

    ratchet: Optional[dict] = None
    if baseline is not None:
        current = failure_keys(results, project_root)
        new_findings, fixed_findings, still_present = evaluate_ratchet(
            current, baseline
        )
        ratchet = {
            "new": sorted(new_findings),
            "fixed": sorted(fixed_findings),
            "still_present": sorted(still_present),
        }

    return {
        "summary": {
            "passed": passed,
            "failed": failed,
            "total": len(results),
        },
        "findings": findings,
        "ratchet": ratchet,
    }


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------


def main() -> None:
    parser = argparse.ArgumentParser(description="Check documentation quality")
    parser.add_argument(
        "--verbose", action="store_true", help="Show all checks including passed"
    )
    parser.add_argument(
        "--json",
        action="store_true",
        help="Print the full results as a JSON object (machine-readable)",
    )
    parser.add_argument(
        "--baseline",
        type=str,
        default=None,
        help="Path to the baseline file (default: scripts/docs_baseline.json)",
    )
    parser.add_argument(
        "--no-baseline",
        action="store_true",
        help="Ignore the baseline entirely (legacy strict mode)",
    )
    parser.add_argument(
        "--update-baseline",
        action="store_true",
        help="(Re)write the baseline from current failures and exit 0",
    )
    args = parser.parse_args()

    # Find project root
    script_dir = Path(__file__).parent
    project_root = script_dir.parent

    baseline_path = (
        Path(args.baseline)
        if args.baseline
        else project_root / DEFAULT_BASELINE_RELPATH
    )

    checker = DocumentationChecker(project_root, verbose=args.verbose)
    checker.check_all(quiet=args.json)
    current = failure_keys(checker.results, project_root)

    def _load_baseline_or_exit() -> Set[str]:
        """Load the baseline, reporting a malformed file cleanly (exit code 2)."""
        try:
            return load_baseline(baseline_path)
        except ValueError as exc:
            print(f"❌ {exc}", file=sys.stderr)
            sys.exit(2)

    # --update-baseline: write current failures and exit 0.
    if args.update_baseline:
        save_baseline(baseline_path, current)
        # To stderr so `--json --update-baseline` keeps a clean stdout stream.
        print(
            f"📝 Wrote baseline with {len(current)} entries to {baseline_path}",
            file=sys.stderr,
        )
        sys.exit(0)

    # --json: print the machine-readable report; exit code still honors ratchet.
    if args.json:
        baseline = None if args.no_baseline else _load_baseline_or_exit()
        report = build_json_report(checker.results, project_root, baseline=baseline)
        print(json.dumps(report, indent=2, sort_keys=False))
        if baseline is None:
            # Strict mode: any failure fails; the baseline file is never read.
            sys.exit(1 if current else 0)
        new_findings, _, _ = evaluate_ratchet(current, baseline)
        sys.exit(1 if new_findings else 0)

    # Human mode: always print the per-check summary.
    checker.print_summary()

    # Legacy strict mode: any failure fails.
    if args.no_baseline:
        if current:
            aprint(
                "\n⚠️  Some documentation checks failed. Please address the "
                "issues above."
            )
            sys.exit(1)
        aprint("\n✅ All documentation checks passed!")
        sys.exit(0)

    # Baseline (ratchet) mode.
    if not baseline_path.exists():
        aprint(
            f"\nℹ️  No baseline found at {baseline_path}. Run with "
            "--update-baseline to create one (existing debt will be tolerated)."
        )
    baseline = _load_baseline_or_exit()
    new_findings, fixed_findings, still_present = evaluate_ratchet(current, baseline)

    aprint("\n" + "=" * 70)
    aprint("📐 DOCUMENTATION RATCHET")
    aprint("=" * 70 + "\n")
    aprint(f"🆕 New findings (regressions): {len(new_findings)}")
    aprint(f"🧱 Still baselined (tolerated debt): {len(still_present)}")
    aprint(f"✨ Fixed since baseline: {len(fixed_findings)}")

    if fixed_findings:
        aprint(
            "\n   Nice — some debt was paid down. Run --update-baseline to "
            "tighten the baseline so it can't come back."
        )

    if new_findings:
        aprint("\n❌ New documentation findings (must be fixed):\n")
        for key in sorted(new_findings):
            aprint(f"  ❌ {key}")
        aprint(
            "\n⚠️  New documentation regressions detected. Add docs for the "
            "items above (do NOT baseline them away)."
        )
        sys.exit(1)

    aprint("\n✅ No new documentation regressions.")
    sys.exit(0)


if __name__ == "__main__":
    main()
