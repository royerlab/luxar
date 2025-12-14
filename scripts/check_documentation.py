#!/usr/bin/env python3
"""
Documentation Quality Checker

Validates documentation completeness and quality across the Luxar project.
Run this script before commits or in CI/CD to ensure documentation standards.

Usage:
    python scripts/check_documentation.py [--fix] [--verbose]
"""

import argparse
import re
import sys
from pathlib import Path
from typing import List, Tuple, Dict
from dataclasses import dataclass


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
        print("🔍 Checking documentation quality...\n")

        # Check Python packages
        self._check_python_packages()

        # Check TypeScript packages
        self._check_typescript_packages()

        # Print summary
        self._print_summary()

        return all(r.passed for r in self.results)

    def _check_python_packages(self):
        """Check all Python packages for documentation completeness."""
        packages_dir = self.project_root / "packages" / "luxar" / "src" / "luxar"

        if not packages_dir.exists():
            return

        for package_dir in packages_dir.iterdir():
            if package_dir.is_dir() and not package_dir.name.startswith('_'):
                self._check_python_package(package_dir)

    def _check_python_package(self, package_dir: Path):
        """Check a single Python package."""
        package_name = package_dir.relative_to(self.project_root / "packages" / "luxar" / "src" / "luxar")

        # Check for README.md
        readme = package_dir / "README.md"
        if not readme.exists():
            self.results.append(CheckResult(
                passed=False,
                file_path=str(readme),
                check_name="README existence",
                message=f"Missing README.md for package: {package_name}"
            ))
        else:
            self._check_readme_quality(readme, package_name)

        # Check for SPECIFICATIONS.md
        specs = package_dir / "SPECIFICATIONS.md"
        if not specs.exists():
            self.results.append(CheckResult(
                passed=False,
                file_path=str(specs),
                check_name="SPECIFICATIONS existence",
                message=f"Missing SPECIFICATIONS.md for package: {package_name}"
            ))
        else:
            self._check_specifications_quality(specs, package_name)

        # Check Python files for docstrings
        for py_file in package_dir.glob("*.py"):
            if not py_file.name.startswith("_") or py_file.name == "__init__.py":
                self._check_python_file_docstrings(py_file, package_name)

    def _check_readme_quality(self, readme: Path, package_name: str):
        """Check README.md quality."""
        content = readme.read_text()

        # Check for Quick Start section (Phase 3 enhancement)
        if "## Quick Start" not in content and "## Getting Started" not in content:
            self.results.append(CheckResult(
                passed=False,
                file_path=str(readme),
                check_name="Quick Start section",
                message=f"README missing 'Quick Start' or 'Getting Started' section: {package_name}"
            ))
        else:
            self.results.append(CheckResult(
                passed=True,
                file_path=str(readme),
                check_name="Quick Start section",
                message=f"✓ README has Quick Start section: {package_name}"
            ))

        # Check minimum length (should have substance)
        if len(content) < 500:
            self.results.append(CheckResult(
                passed=False,
                file_path=str(readme),
                check_name="README length",
                message=f"README too short (<500 chars): {package_name}"
            ))

        # Check for code examples
        if "```python" not in content and "```typescript" not in content:
            self.results.append(CheckResult(
                passed=False,
                file_path=str(readme),
                check_name="Code examples",
                message=f"README missing code examples: {package_name}"
            ))

    def _check_specifications_quality(self, specs: Path, package_name: str):
        """Check SPECIFICATIONS.md quality."""
        content = specs.read_text()

        # Check for required sections
        required_sections = [
            "## Purpose",
            "## Core Concepts",
            "## Data Structures",
            "## Algorithms"
        ]

        for section in required_sections:
            if section not in content:
                self.results.append(CheckResult(
                    passed=False,
                    file_path=str(specs),
                    check_name=f"Required section: {section}",
                    message=f"SPECIFICATIONS missing {section}: {package_name}"
                ))

        # Check for changelog
        if "## Changelog" not in content and "# Changelog" not in content:
            self.results.append(CheckResult(
                passed=False,
                file_path=str(specs),
                check_name="Changelog",
                message=f"SPECIFICATIONS missing Changelog: {package_name}"
            ))

        # Check for complexity analysis in algorithms (Phase 3 enhancement)
        if "## Algorithms" in content:
            # Look for algorithm sections
            algorithm_sections = re.findall(r'###\s+(.+)', content)
            has_complexity = bool(re.search(r'\*\*Complexity\*\*:', content))

            if algorithm_sections and not has_complexity:
                self.results.append(CheckResult(
                    passed=False,
                    file_path=str(specs),
                    check_name="Algorithm complexity",
                    message=f"SPECIFICATIONS has algorithms but missing complexity analysis: {package_name}"
                ))

    def _check_python_file_docstrings(self, py_file: Path, package_name: str):
        """Check Python file for docstring coverage."""
        content = py_file.read_text()

        # Check for module docstring
        if not content.strip().startswith('"""') and not content.strip().startswith("'''"):
            self.results.append(CheckResult(
                passed=False,
                file_path=str(py_file),
                check_name="Module docstring",
                message=f"Missing module docstring: {package_name}/{py_file.name}"
            ))

        # Count functions/classes and docstrings
        func_class_pattern = r'^(def |class )'
        docstring_pattern = r'^\s+["\']'

        lines = content.split('\n')
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
                self.results.append(CheckResult(
                    passed=False,
                    file_path=str(py_file),
                    check_name="Docstring coverage",
                    message=f"Low docstring coverage ({coverage:.0f}%): {package_name}/{py_file.name}"
                ))

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
            self.results.append(CheckResult(
                passed=False,
                file_path=str(readme),
                check_name="README existence",
                message=f"Missing README.md for TypeScript package: {package_name}"
            ))

        # Check TypeScript files for JSDoc
        for ts_file in package_dir.glob("*.ts"):
            if not ts_file.name.endswith(".test.ts"):
                self._check_typescript_jsdoc(ts_file, package_name)

    def _check_typescript_jsdoc(self, ts_file: Path, package_name: str):
        """Check TypeScript file for JSDoc coverage."""
        content = ts_file.read_text()

        # Count exported functions/classes
        export_pattern = r'^export (function|class|interface|type|const)'
        jsdoc_pattern = r'/\*\*'

        lines = content.split('\n')
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
                self.results.append(CheckResult(
                    passed=False,
                    file_path=str(ts_file),
                    check_name="JSDoc coverage",
                    message=f"Low JSDoc coverage ({coverage:.0f}%): {package_name}/{ts_file.name}"
                ))

    def _print_summary(self):
        """Print summary of all checks."""
        print("\n" + "="*70)
        print("📊 DOCUMENTATION QUALITY REPORT")
        print("="*70 + "\n")

        passed = [r for r in self.results if r.passed]
        failed = [r for r in self.results if not r.passed]

        print(f"✅ Passed: {len(passed)}")
        print(f"❌ Failed: {len(failed)}")
        print(f"📈 Overall: {len(passed)}/{len(self.results)} ({len(passed)/len(self.results)*100:.0f}%)\n")

        if failed:
            print("Failed checks:\n")
            for result in failed:
                print(f"  ❌ {result.check_name}")
                print(f"     {result.message}")
                print(f"     File: {result.file_path}\n")

        if self.verbose and passed:
            print("\nPassed checks:\n")
            for result in passed:
                print(f"  ✅ {result.message}")


def main():
    parser = argparse.ArgumentParser(description="Check documentation quality")
    parser.add_argument("--fix", action="store_true", help="Attempt to fix issues automatically")
    parser.add_argument("--verbose", action="store_true", help="Show all checks including passed")
    args = parser.parse_args()

    # Find project root
    script_dir = Path(__file__).parent
    project_root = script_dir.parent

    checker = DocumentationChecker(project_root, verbose=args.verbose)
    success = checker.check_all()

    if not success:
        print("\n⚠️  Some documentation checks failed. Please address the issues above.")
        sys.exit(1)
    else:
        print("\n✅ All documentation checks passed!")
        sys.exit(0)


if __name__ == "__main__":
    main()
