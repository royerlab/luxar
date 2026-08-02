#!/usr/bin/env python3
"""
Documentation Quality Checker

Validates documentation completeness and quality across the Luxar project.
Run this script before commits or in CI/CD to ensure documentation standards.

Usage:
    python scripts/check_documentation.py [--verbose]
"""

import argparse
import re
import subprocess
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import List

from arbol import aprint


@dataclass
class CheckResult:
    """Result of a documentation check."""

    passed: bool
    file_path: str
    check_name: str
    message: str
    line_number: int = None


class DocumentationChecker:
    """Checks documentation quality across Python and TypeScript files."""

    def __init__(self, project_root: Path, verbose: bool = False):
        self.project_root = project_root
        self.verbose = verbose
        self.results: List[CheckResult] = []

    def check_all(self) -> bool:
        """Run all documentation checks. Returns True if all pass."""
        aprint("🔍 Checking documentation quality...\n")

        # Check Python packages
        self._check_python_packages()

        # Check TypeScript packages
        self._check_typescript_packages()

        # Check backticked path references in package READMEs resolve
        self._check_markdown_path_references()

        # Print summary
        self._print_summary()

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

    def _check_markdown_path_references(self):
        """Flag backticked repo-path references in package READMEs that don't resolve."""
        # Resolve the root once so symlinked checkouts (CI, macOS /tmp, symlinked
        # home) don't break `.relative_to(root)` against `.resolve()`d targets.
        root = self.project_root.resolve()
        # Enumerate markdown files via git (never fall back to a filesystem walk,
        # which would re-include generated/gitignored dirs and cause false positives).
        try:
            proc = subprocess.run(
                ["git", "ls-files", "-z"],
                cwd=root,
                capture_output=True,
                text=True,
                errors="replace",
                check=True,
            )
        except (subprocess.CalledProcessError, FileNotFoundError, OSError):
            self.results.append(
                CheckResult(
                    passed=True,
                    file_path=str(root),
                    check_name="Markdown path reference",
                    message="Skipped path-reference check (git unavailable)",
                )
            )
            return

        # `-z` emits NUL-separated, UNQUOTED paths (no octal-escaping of
        # non-ASCII), so basenames stay intact.
        tracked = [p for p in proc.stdout.split("\0") if p]
        by_name: dict[str, list[str]] = {}
        for path in tracked:
            by_name.setdefault(Path(path).name, []).append(path)

        readme_targets = [
            p
            for p in tracked
            if p.startswith("packages/") and Path(p).name == "README.md"
        ]

        backtick_re = re.compile(r"`([^`\n]+)`")
        clean_re = re.compile(r"[\w./\-]+")
        source_exts = {
            ".py",
            ".ts",
            ".tsx",
            ".js",
            ".jsx",
            ".mjs",
            ".cjs",
            ".rs",
            ".md",
            ".rst",
            ".json",
            ".jsonc",
            ".yml",
            ".yaml",
            ".toml",
            ".cfg",
            ".ini",
            ".txt",
            ".sh",
            ".go",
            ".css",
            ".scss",
            ".html",
            ".glsl",
            ".wgsl",
            ".vert",
            ".frag",
        }
        skip_segments = {
            "node_modules",
            "target",
            "dist",
            "build",
            "__pycache__",
            ".venv",
        }

        def is_candidate(token: str) -> bool:
            if "/" not in token or token.startswith("/"):
                return False
            if not clean_re.fullmatch(token):
                return False
            last_segment = token.rsplit("/", 1)[-1]
            if "." not in last_segment:
                return False
            ext = "." + last_segment.rsplit(".", 1)[-1]
            if ext not in source_exts:
                return False
            if any(seg in skip_segments for seg in token.split("/")):
                return False
            return True

        def resolves(token: str, readme_path: str) -> bool:
            if token.startswith("./") or token.startswith("../"):
                resolved = (root / Path(readme_path).parent / token).resolve()
                try:
                    rel = resolved.relative_to(root).as_posix()
                except ValueError:
                    # Falls outside the project root; don't flag.
                    return True
                return rel in tracked or (root / rel).is_dir()

            if token in tracked or (root / token).is_dir():
                return True
            readme_dir = Path(readme_path).parent.as_posix()
            cand = token if readme_dir == "." else readme_dir + "/" + token
            if cand in tracked or (root / cand).is_dir():
                return True
            basename = Path(token).name
            for p in by_name.get(basename, []):
                if p == token or p.endswith("/" + token):
                    return True
            return False

        for readme_path in readme_targets:
            content = (root / readme_path).read_text(encoding="utf-8", errors="replace")
            for line_number, line in enumerate(content.splitlines(), start=1):
                for match in backtick_re.finditer(line):
                    token = match.group(1)
                    if not is_candidate(token):
                        continue
                    if resolves(token, readme_path):
                        continue
                    # False-positive filter: only report NON-relative tokens
                    # whose basename exists elsewhere in the repo (else it's an
                    # npm import, served URL, placeholder, or build-output
                    # description). A `./` or `../` token is an unambiguous path
                    # claim, so a broken relative ref is flagged even when its
                    # basename no longer exists anywhere (fully deleted file).
                    is_relative = token.startswith("./") or token.startswith("../")
                    if not is_relative and Path(token).name not in by_name:
                        continue
                    self.results.append(
                        CheckResult(
                            passed=False,
                            file_path=str(root / readme_path),
                            check_name="Markdown path reference",
                            message=(
                                f"README cites path that does not resolve: `{token}`"
                            ),
                            line_number=line_number,
                        )
                    )

    def _print_summary(self):
        """Print summary of all checks."""
        aprint("\n" + "=" * 70)
        aprint("📊 DOCUMENTATION QUALITY REPORT")
        aprint("=" * 70 + "\n")

        passed = [r for r in self.results if r.passed]
        failed = [r for r in self.results if not r.passed]

        aprint(f"✅ Passed: {len(passed)}")
        aprint(f"❌ Failed: {len(failed)}")
        aprint(
            f"📈 Overall: {len(passed)}/{len(self.results)} ({len(passed) / len(self.results) * 100:.0f}%)\n"
        )

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


def main():
    parser = argparse.ArgumentParser(description="Check documentation quality")
    parser.add_argument(
        "--verbose", action="store_true", help="Show all checks including passed"
    )
    args = parser.parse_args()

    # Find project root
    script_dir = Path(__file__).parent
    project_root = script_dir.parent

    checker = DocumentationChecker(project_root, verbose=args.verbose)
    success = checker.check_all()

    if not success:
        aprint(
            "\n⚠️  Some documentation checks failed. Please address the issues above."
        )
        sys.exit(1)
    else:
        aprint("\n✅ All documentation checks passed!")
        sys.exit(0)


if __name__ == "__main__":
    main()
