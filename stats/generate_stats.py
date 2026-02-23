#!/usr/bin/env python3
"""
Generate project statistics report for Luxar.

Analyzes all languages in the codebase (Python, TypeScript, Rust, CUDA, CSS,
Shell, Configuration files) to produce comprehensive statistics including
lines of code, file counts, function/class counts, test coverage, and more.
"""

import html
import json
import re
import subprocess
from collections import defaultdict
from datetime import datetime
from pathlib import Path

from arbol import aprint, asection

# Language configurations
LANGUAGE_CONFIG = {
    "python": {
        "extensions": [".py"],
        "comment_single": "#",
        "comment_multi_start": '"""',
        "comment_multi_end": '"""',
        "color": "#3572A5",
    },
    "typescript": {
        "extensions": [".ts", ".tsx"],
        "comment_single": "//",
        "comment_multi_start": "/*",
        "comment_multi_end": "*/",
        "color": "#3178C6",
    },
    "rust": {
        "extensions": [".rs"],
        "comment_single": "//",
        "comment_multi_start": "/*",
        "comment_multi_end": "*/",
        "color": "#DEA584",
    },
    "cuda": {
        "extensions": [".cu", ".cuh"],
        "comment_single": "//",
        "comment_multi_start": "/*",
        "comment_multi_end": "*/",
        "color": "#76B900",
    },
    "css": {
        "extensions": [".css", ".scss"],
        "comment_single": None,
        "comment_multi_start": "/*",
        "comment_multi_end": "*/",
        "color": "#563D7C",
    },
    "javascript": {
        "extensions": [".js", ".jsx", ".mjs"],
        "comment_single": "//",
        "comment_multi_start": "/*",
        "comment_multi_end": "*/",
        "color": "#F7DF1E",
    },
    "shell": {
        "extensions": [".sh", ".bash"],
        "comment_single": "#",
        "comment_multi_start": None,
        "comment_multi_end": None,
        "color": "#89E051",
    },
    "json": {
        "extensions": [".json"],
        "comment_single": None,
        "comment_multi_start": None,
        "comment_multi_end": None,
        "color": "#292929",
    },
    "toml": {
        "extensions": [".toml"],
        "comment_single": "#",
        "comment_multi_start": None,
        "comment_multi_end": None,
        "color": "#9C4121",
    },
    "yaml": {
        "extensions": [".yaml", ".yml"],
        "comment_single": "#",
        "comment_multi_start": None,
        "comment_multi_end": None,
        "color": "#CB171E",
    },
    "markdown": {
        "extensions": [".md"],
        "comment_single": None,
        "comment_multi_start": None,
        "comment_multi_end": None,
        "color": "#083FA1",
    },
    "html": {
        "extensions": [".html", ".htm"],
        "comment_single": None,
        "comment_multi_start": "<!--",
        "comment_multi_end": "-->",
        "color": "#E34C26",
    },
    "makefile": {
        "extensions": ["Makefile", ".mk"],
        "comment_single": "#",
        "comment_multi_start": None,
        "comment_multi_end": None,
        "color": "#427819",
    },
}

# Directories to skip
SKIP_DIRS = {
    ".git",
    "node_modules",
    "__pycache__",
    ".pytest_cache",
    "coverage",
    "dist",
    "build",
    ".eggs",
    ".mypy_cache",
    ".ruff_cache",
    "target",
    ".hatch",
    ".venv",
    "venv",
    "htmlcov",
    ".tox",
}


def count_lines_in_file(filepath, lang_config):
    """Count total, code, comment, and blank lines in a file."""
    try:
        with open(filepath, "r", encoding="utf-8", errors="ignore") as f:
            lines = f.readlines()
    except Exception:
        return {"total": 0, "code": 0, "comment": 0, "blank": 0}

    total = len(lines)
    blank = sum(1 for line in lines if line.strip() == "")

    comment = 0
    in_block_comment = False
    comment_single = lang_config.get("comment_single")
    comment_multi_start = lang_config.get("comment_multi_start")
    comment_multi_end = lang_config.get("comment_multi_end")

    for line in lines:
        stripped = line.strip()

        # Handle block comments
        if comment_multi_start and comment_multi_end:
            if comment_multi_start in stripped and not in_block_comment:
                in_block_comment = True
            if in_block_comment:
                comment += 1
                if comment_multi_end in stripped:
                    in_block_comment = False
                continue

        # Handle single-line comments
        if comment_single and stripped.startswith(comment_single):
            comment += 1

    code = max(0, total - blank - comment)

    return {"total": total, "code": code, "comment": comment, "blank": blank}


def count_python_definitions(filepath):
    """Count classes and functions in Python file."""
    try:
        with open(filepath, "r", encoding="utf-8", errors="ignore") as f:
            content = f.read()
    except Exception:
        return {"classes": 0, "functions": 0, "methods": 0}

    classes = len(re.findall(r"^\s*class\s+\w+", content, re.MULTILINE))
    all_functions = len(re.findall(r"^\s*def\s+\w+", content, re.MULTILINE))
    methods = len(re.findall(r"^\s{4,}def\s+\w+", content, re.MULTILINE))
    functions = all_functions - methods

    return {"classes": classes, "functions": functions, "methods": methods}


def count_typescript_definitions(filepath):
    """Count classes, functions, and interfaces in TypeScript file."""
    try:
        with open(filepath, "r", encoding="utf-8", errors="ignore") as f:
            content = f.read()
    except Exception:
        return {"classes": 0, "functions": 0, "interfaces": 0, "types": 0}

    classes = len(re.findall(r"^\s*(?:export\s+)?class\s+\w+", content, re.MULTILINE))
    functions = len(
        re.findall(r"^\s*(?:export\s+)?function\s+\w+", content, re.MULTILINE)
    )
    functions += len(
        re.findall(
            r"^\s*(?:export\s+)?const\s+\w+\s*=\s*(?:async\s+)?\([^)]*\)\s*=>",
            content,
            re.MULTILINE,
        )
    )
    interfaces = len(
        re.findall(r"^\s*(?:export\s+)?interface\s+\w+", content, re.MULTILINE)
    )
    types = len(re.findall(r"^\s*(?:export\s+)?type\s+\w+\s*=", content, re.MULTILINE))

    return {
        "classes": classes,
        "functions": functions,
        "interfaces": interfaces,
        "types": types,
    }


def count_rust_definitions(filepath):
    """Count structs, functions, traits, and impls in Rust file."""
    try:
        with open(filepath, "r", encoding="utf-8", errors="ignore") as f:
            content = f.read()
    except Exception:
        return {"structs": 0, "functions": 0, "traits": 0, "impls": 0, "enums": 0}

    structs = len(re.findall(r"^\s*(?:pub\s+)?struct\s+\w+", content, re.MULTILINE))
    functions = len(re.findall(r"^\s*(?:pub\s+)?fn\s+\w+", content, re.MULTILINE))
    traits = len(re.findall(r"^\s*(?:pub\s+)?trait\s+\w+", content, re.MULTILINE))
    impls = len(re.findall(r"^\s*impl(?:\s*<[^>]*>)?\s+\w+", content, re.MULTILINE))
    enums = len(re.findall(r"^\s*(?:pub\s+)?enum\s+\w+", content, re.MULTILINE))

    return {
        "structs": structs,
        "functions": functions,
        "traits": traits,
        "impls": impls,
        "enums": enums,
    }


def count_cuda_definitions(filepath):
    """Count kernels and device functions in CUDA file."""
    try:
        with open(filepath, "r", encoding="utf-8", errors="ignore") as f:
            content = f.read()
    except Exception:
        return {"kernels": 0, "device_functions": 0, "host_functions": 0}

    kernels = len(re.findall(r"__global__\s+\w+\s+\w+", content, re.MULTILINE))
    device_functions = len(
        re.findall(r"__device__\s+(?!__host__)[\w\s*&]+\s+\w+\s*\(", content)
    )
    host_functions = len(re.findall(r"__host__\s+[\w\s*&]+\s+\w+\s*\(", content))

    return {
        "kernels": kernels,
        "device_functions": device_functions,
        "host_functions": host_functions,
    }


def count_css_definitions(filepath):
    """Count CSS rules, selectors, and variables."""
    try:
        with open(filepath, "r", encoding="utf-8", errors="ignore") as f:
            content = f.read()
    except Exception:
        return {"rules": 0, "variables": 0, "media_queries": 0}

    # Count rule blocks (simplified)
    rules = len(re.findall(r"\{[^}]*\}", content))
    variables = len(re.findall(r"--[\w-]+\s*:", content))
    media_queries = len(re.findall(r"@media", content))

    return {"rules": rules, "variables": variables, "media_queries": media_queries}


def get_test_statistics(project_root):
    """Get test count and coverage statistics for Python, TypeScript, and Rust."""
    test_stats = {
        "python": {
            "test_files": 0,
            "test_count": 0,
            "test_passed": 0,
            "test_failed": 0,
            "coverage_percent": 0,
        },
        "typescript": {
            "test_files": 0,
            "test_count": 0,
            "test_passed": 0,
            "test_failed": 0,
            "coverage_percent": 0,
        },
        "rust": {"test_files": 0, "test_count": 0, "test_passed": 0, "test_failed": 0},
        "e2e": {"test_files": 0, "test_count": 0},
        "error": None,
    }

    root_path = Path(project_root)

    # === Python Tests ===
    with asection("Python tests"):
        test_files = list(root_path.rglob("test_*.py")) + list(
            root_path.rglob("*_test.py")
        )
        test_files = [
            f
            for f in test_files
            if "__pycache__" not in str(f) and ".hatch" not in str(f)
        ]
        test_stats["python"]["test_files"] = len(test_files)
        aprint(f"Found {len(test_files)} test files")

        # Collect test count
        try:
            result = subprocess.run(
                [
                    "hatch",
                    "run",
                    "pytest",
                    "--collect-only",
                    "-q",
                    "packages/luxar/src/luxar",
                ],
                capture_output=True,
                text=True,
                timeout=60,
                cwd=project_root,
            )

            for line in result.stdout.split("\n"):
                if "selected" in line or "collected" in line:
                    numbers = re.findall(r"\d+", line)
                    if numbers:
                        test_stats["python"]["test_count"] = int(numbers[0])
                        aprint(f"Collected {numbers[0]} tests")
                        break
        except Exception as e:
            aprint(f"Could not collect tests: {e}")

        # Run tests with coverage
        aprint("Running tests with coverage...")
        try:
            result = subprocess.run(
                [
                    "hatch",
                    "run",
                    "pytest",
                    "--cov=packages/luxar/src/luxar",
                    "--cov-report=term",
                    "packages/luxar/src/luxar",
                    "-q",
                    "--tb=no",
                ],
                capture_output=True,
                text=True,
                timeout=300,
                cwd=project_root,
            )

            # Parse coverage
            for line in result.stdout.split("\n") + result.stderr.split("\n"):
                if line.strip().startswith("TOTAL"):
                    parts = line.split()
                    for part in reversed(parts):
                        if "%" in part:
                            pct_match = re.search(r"(\d+(?:\.\d+)?)%", part)
                            if pct_match:
                                test_stats["python"]["coverage_percent"] = float(
                                    pct_match.group(1)
                                )
                                aprint(f"Coverage: {pct_match.group(1)}%")
                                break
                    break

            # Count passing tests
            for line in result.stdout.split("\n"):
                if "passed" in line:
                    match = re.search(r"(\d+)\s+passed", line)
                    if match:
                        test_stats["python"]["test_passed"] = int(match.group(1))
                if "failed" in line:
                    match = re.search(r"(\d+)\s+failed", line)
                    if match:
                        test_stats["python"]["test_failed"] = int(match.group(1))

        except subprocess.TimeoutExpired:
            aprint("Test run timed out")
        except Exception as e:
            aprint(f"Could not run tests: {e}")

    # === TypeScript Tests ===
    viewer_path = root_path / "packages" / "luxar-viewer"
    if viewer_path.exists():
        with asection("TypeScript tests"):
            ts_test_files = list(viewer_path.rglob("*.test.ts")) + list(
                viewer_path.rglob("*.test.tsx")
            )
            ts_test_files = [f for f in ts_test_files if "node_modules" not in str(f)]
            test_stats["typescript"]["test_files"] = len(ts_test_files)
            aprint(f"Found {len(ts_test_files)} unit test files")

            # E2E test files
            e2e_files = list(viewer_path.rglob("*.spec.ts"))
            e2e_files = [f for f in e2e_files if "node_modules" not in str(f)]
            test_stats["e2e"]["test_files"] = len(e2e_files)
            aprint(f"Found {len(e2e_files)} E2E test files")

            # Run TypeScript tests with coverage
            # Exclude performance benchmark tests that may timeout and block coverage
            aprint("Running unit tests with coverage...")
            try:
                result = subprocess.run(
                    [
                        "npx",
                        "vitest",
                        "--run",
                        "--coverage",
                        "--exclude",
                        "**/wasm-performance.test.ts",
                    ],
                    capture_output=True,
                    text=True,
                    timeout=180,
                    cwd=viewer_path,
                )

                # Parse test counts from output
                for line in result.stdout.split("\n") + result.stderr.split("\n"):
                    if "Tests" in line and "passed" in line:
                        total_match = re.search(r"\((\d+)\)", line)
                        if total_match:
                            test_stats["typescript"]["test_count"] = int(
                                total_match.group(1)
                            )
                        passed_match = re.search(r"(\d+)\s+passed", line)
                        if passed_match:
                            test_stats["typescript"]["test_passed"] = int(
                                passed_match.group(1)
                            )
                        failed_match = re.search(r"(\d+)\s+failed", line)
                        if failed_match:
                            test_stats["typescript"]["test_failed"] = int(
                                failed_match.group(1)
                            )
                        break

                # Try to read coverage from JSON file (more reliable than stdout)
                coverage_json = viewer_path / "coverage" / "coverage-summary.json"
                if coverage_json.exists():
                    try:
                        with open(coverage_json) as f:
                            cov_data = json.load(f)
                        total_cov = cov_data.get("total", {})
                        statements = total_cov.get("statements", {})
                        if statements:
                            pct = statements.get("pct", 0)
                            test_stats["typescript"]["coverage_percent"] = float(pct)
                            aprint(f"Coverage (from JSON): {pct}%")
                    except Exception as e:
                        aprint(f"Could not read coverage JSON: {e}")

                # Fallback: parse coverage from stdout
                if test_stats["typescript"]["coverage_percent"] == 0:
                    for line in result.stdout.split("\n") + result.stderr.split("\n"):
                        # Look for lines like "All files  |   85.23 |..."
                        if "All files" in line and "|" in line:
                            parts = line.split("|")
                            if len(parts) >= 2:
                                try:
                                    pct = float(parts[1].strip())
                                    test_stats["typescript"]["coverage_percent"] = pct
                                    aprint(f"Coverage (from stdout): {pct}%")
                                    break
                                except ValueError:
                                    pass

                aprint(
                    f"Passed: {test_stats['typescript']['test_passed']}, "
                    f"Failed: {test_stats['typescript']['test_failed']}"
                )

            except subprocess.TimeoutExpired:
                aprint("Test run timed out")
            except Exception as e:
                aprint(f"Could not run tests: {e}")

    # === Rust Tests ===
    rust_path = root_path / "packages" / "luxar-viewer" / "src" / "wasm" / "rust"
    if rust_path.exists():
        with asection("Rust tests"):
            rust_test_files = list(rust_path.rglob("*.rs"))
            test_stats["rust"]["test_files"] = len(
                [
                    f
                    for f in rust_test_files
                    if "test" in f.read_text(errors="ignore").lower()
                ]
            )

            # Run cargo test
            aprint("Running cargo tests...")
            try:
                result = subprocess.run(
                    ["cargo", "test", "--", "--test-threads=1"],
                    capture_output=True,
                    text=True,
                    timeout=120,
                    cwd=rust_path,
                )

                # Parse test results
                for line in result.stdout.split("\n"):
                    if "test result:" in line:
                        passed_match = re.search(r"(\d+)\s+passed", line)
                        if passed_match:
                            test_stats["rust"]["test_passed"] = int(
                                passed_match.group(1)
                            )
                            test_stats["rust"]["test_count"] = int(
                                passed_match.group(1)
                            )
                        failed_match = re.search(r"(\d+)\s+failed", line)
                        if failed_match:
                            test_stats["rust"]["test_failed"] = int(
                                failed_match.group(1)
                            )
                            test_stats["rust"]["test_count"] += int(
                                failed_match.group(1)
                            )
                        break

                aprint(
                    f"Passed: {test_stats['rust']['test_passed']}, "
                    f"Failed: {test_stats['rust']['test_failed']}"
                )

            except Exception as e:
                aprint(f"Could not run Rust tests: {e}")

    return test_stats


def get_git_statistics(project_root):
    """Get Git repository statistics."""
    git_stats = {
        "total_commits": 0,
        "contributors": 0,
        "first_commit_date": None,
        "last_commit_date": None,
        "branches": 0,
        "tags": 0,
        "commits_last_30_days": 0,
        "top_contributors": [],
    }

    try:
        # Total commits
        result = subprocess.run(
            ["git", "rev-list", "--count", "HEAD"],
            capture_output=True,
            text=True,
            timeout=30,
            cwd=project_root,
        )
        if result.returncode == 0:
            git_stats["total_commits"] = int(result.stdout.strip())

        # Contributors
        result = subprocess.run(
            ["git", "shortlog", "-sn", "--all"],
            capture_output=True,
            text=True,
            timeout=30,
            cwd=project_root,
        )
        if result.returncode == 0:
            contributors = result.stdout.strip().split("\n")
            git_stats["contributors"] = len([c for c in contributors if c.strip()])
            # Get top 5 contributors
            for line in contributors[:5]:
                match = re.match(r"\s*(\d+)\s+(.+)", line)
                if match:
                    git_stats["top_contributors"].append(
                        {"commits": int(match.group(1)), "name": match.group(2).strip()}
                    )

        # First and last commit dates
        result = subprocess.run(
            ["git", "log", "--format=%ai", "--reverse"],
            capture_output=True,
            text=True,
            timeout=30,
            cwd=project_root,
        )
        if result.returncode == 0 and result.stdout.strip():
            dates = result.stdout.strip().split("\n")
            if dates:
                git_stats["first_commit_date"] = dates[0].split()[0]
                git_stats["last_commit_date"] = dates[-1].split()[0]

        # Branch count
        result = subprocess.run(
            ["git", "branch", "-a"],
            capture_output=True,
            text=True,
            timeout=30,
            cwd=project_root,
        )
        if result.returncode == 0:
            branches = [b for b in result.stdout.strip().split("\n") if b.strip()]
            git_stats["branches"] = len(branches)

        # Tag count
        result = subprocess.run(
            ["git", "tag"],
            capture_output=True,
            text=True,
            timeout=30,
            cwd=project_root,
        )
        if result.returncode == 0:
            tags = [t for t in result.stdout.strip().split("\n") if t.strip()]
            git_stats["tags"] = len(tags)

        # Commits in last 30 days
        result = subprocess.run(
            ["git", "rev-list", "--count", "--since=30 days ago", "HEAD"],
            capture_output=True,
            text=True,
            timeout=30,
            cwd=project_root,
        )
        if result.returncode == 0:
            git_stats["commits_last_30_days"] = int(result.stdout.strip() or 0)

    except Exception as e:
        git_stats["error"] = str(e)

    return git_stats


def get_dependency_statistics(project_root):
    """Get dependency information from package files."""
    deps = {
        "python": {"production": 0, "dev": 0, "packages": []},
        "node": {"production": 0, "dev": 0, "packages": []},
        "rust": {"production": 0, "dev": 0, "packages": []},
    }

    root_path = Path(project_root)

    # Python dependencies from pyproject.toml
    pyproject = root_path / "pyproject.toml"
    if pyproject.exists():
        try:
            content = pyproject.read_text()
            # Count dependencies (simplified parsing)
            in_deps = False
            in_dev_deps = False
            for line in content.split("\n"):
                if "[project.dependencies]" in line or "dependencies = [" in line:
                    in_deps = True
                    in_dev_deps = False
                elif (
                    "[project.optional-dependencies]" in line
                    or "dev-dependencies" in line
                ):
                    in_dev_deps = True
                    in_deps = False
                elif line.strip().startswith("[") and not line.strip().startswith("[["):
                    in_deps = False
                    in_dev_deps = False
                elif in_deps and "=" in line or (in_deps and '"' in line):
                    deps["python"]["production"] += 1
                elif in_dev_deps and "=" in line or (in_dev_deps and '"' in line):
                    deps["python"]["dev"] += 1
        except Exception:
            pass

    # Node.js dependencies from package.json
    viewer_package = root_path / "packages" / "luxar-viewer" / "package.json"
    if viewer_package.exists():
        try:
            with open(viewer_package) as f:
                pkg = json.load(f)
            deps["node"]["production"] = len(pkg.get("dependencies", {}))
            deps["node"]["dev"] = len(pkg.get("devDependencies", {}))
            deps["node"]["packages"] = list(pkg.get("dependencies", {}).keys())[:10]
        except Exception:
            pass

    # Rust dependencies from Cargo.toml
    cargo_toml = (
        root_path / "packages" / "luxar-viewer" / "src" / "wasm" / "rust" / "Cargo.toml"
    )
    if cargo_toml.exists():
        try:
            content = cargo_toml.read_text()
            in_deps = False
            in_dev_deps = False
            for line in content.split("\n"):
                if "[dependencies]" in line:
                    in_deps = True
                    in_dev_deps = False
                elif "[dev-dependencies]" in line:
                    in_dev_deps = True
                    in_deps = False
                elif line.strip().startswith("["):
                    in_deps = False
                    in_dev_deps = False
                elif in_deps and "=" in line and not line.strip().startswith("#"):
                    deps["rust"]["production"] += 1
                elif in_dev_deps and "=" in line and not line.strip().startswith("#"):
                    deps["rust"]["dev"] += 1
        except Exception:
            pass

    return deps


def analyze_directory(root_dir, language_name):
    """Analyze all files for a specific language."""
    lang_config = LANGUAGE_CONFIG[language_name]
    extensions = lang_config["extensions"]

    stats = {
        "files": 0,
        "total_lines": 0,
        "code_lines": 0,
        "comment_lines": 0,
        "blank_lines": 0,
        "definitions": defaultdict(int),
        "by_subdir": defaultdict(lambda: defaultdict(int)),
        "largest_files": [],
    }

    root_path = Path(root_dir)
    file_sizes = []

    for ext in extensions:
        # Handle special case for Makefile
        if ext == "Makefile":
            for filepath in root_path.rglob(ext):
                if any(part in filepath.parts for part in SKIP_DIRS):
                    continue
                process_file(
                    filepath, root_path, stats, lang_config, language_name, file_sizes
                )
        else:
            for filepath in root_path.rglob(f"*{ext}"):
                if any(part in filepath.parts for part in SKIP_DIRS):
                    continue
                process_file(
                    filepath, root_path, stats, lang_config, language_name, file_sizes
                )

    # Get top 5 largest files
    file_sizes.sort(key=lambda x: x[1], reverse=True)
    stats["largest_files"] = file_sizes[:5]

    return stats


def process_file(filepath, root_path, stats, lang_config, language_name, file_sizes):
    """Process a single file and update stats."""
    try:
        rel_path = filepath.relative_to(root_path)
        if len(rel_path.parts) > 1:
            subdir = rel_path.parts[0]
        else:
            subdir = "root"
    except ValueError:
        subdir = "root"

    stats["files"] += 1

    # Count lines
    line_stats = count_lines_in_file(filepath, lang_config)
    stats["total_lines"] += line_stats["total"]
    stats["code_lines"] += line_stats["code"]
    stats["comment_lines"] += line_stats["comment"]
    stats["blank_lines"] += line_stats["blank"]

    # Track file sizes
    file_sizes.append((str(filepath.relative_to(root_path)), line_stats["code"]))

    # Count definitions based on language
    if language_name == "python":
        defs = count_python_definitions(filepath)
        stats["definitions"]["classes"] += defs["classes"]
        stats["definitions"]["functions"] += defs["functions"]
        stats["definitions"]["methods"] += defs["methods"]
    elif language_name == "typescript":
        defs = count_typescript_definitions(filepath)
        stats["definitions"]["classes"] += defs["classes"]
        stats["definitions"]["functions"] += defs["functions"]
        stats["definitions"]["interfaces"] += defs["interfaces"]
        stats["definitions"]["types"] += defs["types"]
    elif language_name == "rust":
        defs = count_rust_definitions(filepath)
        stats["definitions"]["structs"] += defs["structs"]
        stats["definitions"]["functions"] += defs["functions"]
        stats["definitions"]["traits"] += defs["traits"]
        stats["definitions"]["impls"] += defs["impls"]
        stats["definitions"]["enums"] += defs["enums"]
    elif language_name == "cuda":
        defs = count_cuda_definitions(filepath)
        stats["definitions"]["kernels"] += defs["kernels"]
        stats["definitions"]["device_functions"] += defs["device_functions"]
        stats["definitions"]["host_functions"] += defs["host_functions"]
    elif language_name == "css":
        defs = count_css_definitions(filepath)
        stats["definitions"]["rules"] += defs["rules"]
        stats["definitions"]["variables"] += defs["variables"]
        stats["definitions"]["media_queries"] += defs["media_queries"]

    # Update subdir stats
    stats["by_subdir"][subdir]["files"] += 1
    stats["by_subdir"][subdir]["code_lines"] += line_stats["code"]


def generate_html_report(stats, output_file):
    """Generate comprehensive HTML report from statistics."""
    now = datetime.now()

    # Calculate totals
    total_files = sum(stats["languages"][lang]["files"] for lang in stats["languages"])
    total_code_lines = sum(
        stats["languages"][lang]["code_lines"] for lang in stats["languages"]
    )
    total_lines = sum(
        stats["languages"][lang]["total_lines"] for lang in stats["languages"]
    )

    # Primary languages (code files)
    primary_langs = [
        "python",
        "typescript",
        "rust",
        "cuda",
        "css",
        "javascript",
        "shell",
    ]
    config_langs = ["json", "toml", "yaml"]
    doc_langs = ["markdown", "html"]

    # Test totals
    test_stats = stats["tests"]
    total_test_files = (
        test_stats["python"]["test_files"]
        + test_stats["typescript"]["test_files"]
        + test_stats["e2e"]["test_files"]
    )
    total_tests = (
        test_stats["python"]["test_count"]
        + test_stats["typescript"]["test_count"]
        + test_stats["rust"]["test_count"]
    )
    total_passed = (
        test_stats["python"]["test_passed"]
        + test_stats["typescript"]["test_passed"]
        + test_stats["rust"]["test_passed"]
    )

    # Calculate weighted average coverage (only for languages with coverage data)
    py_coverage = test_stats["python"]["coverage_percent"]
    ts_coverage = test_stats["typescript"]["coverage_percent"]
    py_loc = stats["languages"]["python"]["code_lines"]
    ts_loc = stats["languages"]["typescript"]["code_lines"]

    if py_loc + ts_loc > 0:
        weighted_coverage = (py_coverage * py_loc + ts_coverage * ts_loc) / (
            py_loc + ts_loc
        )
    else:
        weighted_coverage = 0

    report_html = f"""<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>Luxar Project Statistics Report</title>
    <style>
        * {{ margin: 0; padding: 0; box-sizing: border-box; }}
        body {{
            font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, 'Helvetica Neue', Arial, sans-serif;
            line-height: 1.6;
            color: #333;
            background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
            min-height: 100vh;
            padding: 2rem;
        }}
        .container {{
            max-width: 1400px;
            margin: 0 auto;
            background: white;
            border-radius: 12px;
            box-shadow: 0 20px 60px rgba(0,0,0,0.3);
            overflow: hidden;
        }}
        header {{
            background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
            color: white;
            padding: 3rem 2rem;
            text-align: center;
        }}
        header h1 {{ font-size: 2.5rem; margin-bottom: 0.5rem; font-weight: 700; }}
        header p {{ font-size: 1.1rem; opacity: 0.9; }}
        .content {{ padding: 2rem; }}

        .summary {{
            display: grid;
            grid-template-columns: repeat(auto-fit, minmax(180px, 1fr));
            gap: 1.25rem;
            margin-bottom: 2rem;
        }}
        .stat-card {{
            background: linear-gradient(135deg, #f5f7fa 0%, #c3cfe2 100%);
            padding: 1.25rem;
            border-radius: 8px;
            box-shadow: 0 4px 6px rgba(0,0,0,0.1);
            transition: transform 0.2s;
        }}
        .stat-card:hover {{ transform: translateY(-4px); box-shadow: 0 8px 12px rgba(0,0,0,0.15); }}
        .stat-card h3 {{
            font-size: 0.8rem;
            text-transform: uppercase;
            letter-spacing: 0.05em;
            color: #667eea;
            margin-bottom: 0.5rem;
            font-weight: 600;
        }}
        .stat-card .value {{ font-size: 2rem; font-weight: 700; color: #333; }}
        .stat-card .label {{ font-size: 0.8rem; color: #666; margin-top: 0.25rem; }}

        section {{ margin-bottom: 2.5rem; }}
        section h2 {{
            font-size: 1.6rem;
            color: #667eea;
            margin-bottom: 1rem;
            padding-bottom: 0.5rem;
            border-bottom: 3px solid #667eea;
        }}
        section h3 {{
            font-size: 1.2rem;
            color: #444;
            margin: 1.5rem 0 1rem 0;
        }}

        table {{ width: 100%; border-collapse: collapse; margin-top: 1rem; font-size: 0.9rem; }}
        th {{
            background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
            color: white;
            padding: 0.75rem;
            text-align: left;
            font-weight: 600;
            text-transform: uppercase;
            font-size: 0.75rem;
            letter-spacing: 0.05em;
        }}
        td {{ padding: 0.6rem 0.75rem; border-bottom: 1px solid #e0e0e0; }}
        tr:hover {{ background: #f8f9fa; }}
        .number {{ text-align: right; font-family: 'Monaco', 'Courier New', monospace; font-weight: 500; }}

        .lang-badge {{
            display: inline-block;
            padding: 0.2rem 0.5rem;
            border-radius: 4px;
            font-size: 0.75rem;
            font-weight: 600;
            color: white;
        }}

        .progress-bar {{
            height: 16px;
            background: #e0e0e0;
            border-radius: 8px;
            overflow: hidden;
            margin-top: 0.5rem;
        }}
        .progress-fill {{
            height: 100%;
            transition: width 0.3s ease;
        }}

        .lang-breakdown {{
            display: grid;
            grid-template-columns: repeat(auto-fit, minmax(280px, 1fr));
            gap: 1.5rem;
        }}
        .lang-card {{
            background: #f8f9fa;
            border-radius: 8px;
            padding: 1.25rem;
            border-left: 4px solid;
        }}
        .lang-card h4 {{
            font-size: 1rem;
            margin-bottom: 0.75rem;
            display: flex;
            align-items: center;
            gap: 0.5rem;
        }}
        .lang-stats {{ font-size: 0.85rem; color: #555; }}
        .lang-stats div {{ margin-bottom: 0.25rem; }}

        .two-column {{
            display: grid;
            grid-template-columns: repeat(auto-fit, minmax(400px, 1fr));
            gap: 2rem;
        }}

        .insight {{
            margin-bottom: 1rem;
            padding: 1rem;
            background: #f8f9fa;
            border-left: 4px solid #667eea;
            border-radius: 4px;
        }}
        .insight strong {{ color: #667eea; }}

        footer {{
            text-align: center;
            padding: 2rem;
            color: #666;
            font-size: 0.85rem;
            border-top: 1px solid #e0e0e0;
        }}

        .health-indicator {{
            display: inline-block;
            padding: 0.15rem 0.4rem;
            border-radius: 3px;
            font-size: 0.75rem;
            font-weight: 500;
        }}
        .health-good {{ background: #d4edda; color: #155724; }}
        .health-warning {{ background: #fff3cd; color: #856404; }}
        .health-danger {{ background: #f8d7da; color: #721c24; }}
    </style>
</head>
<body>
    <div class="container">
        <header>
            <h1>Luxar Project Statistics</h1>
            <p>Comprehensive codebase analysis | Generated {now.strftime("%B %d, %Y at %H:%M")}</p>
        </header>

        <div class="content">
            <!-- Executive Summary -->
            <div class="summary">
                <div class="stat-card">
                    <h3>Total Files</h3>
                    <div class="value">{total_files:,}</div>
                    <div class="label">All languages</div>
                </div>
                <div class="stat-card">
                    <h3>Lines of Code</h3>
                    <div class="value">{total_code_lines:,}</div>
                    <div class="label">Executable code</div>
                </div>
                <div class="stat-card">
                    <h3>Total Lines</h3>
                    <div class="value">{total_lines:,}</div>
                    <div class="label">Including comments</div>
                </div>
                <div class="stat-card">
                    <h3>Languages</h3>
                    <div class="value">{len([l for l in stats["languages"] if stats["languages"][l]["files"] > 0])}</div>
                    <div class="label">Active in codebase</div>
                </div>
                <div class="stat-card">
                    <h3>Test Files</h3>
                    <div class="value">{total_test_files:,}</div>
                    <div class="label">Py + TS + E2E</div>
                </div>
                <div class="stat-card">
                    <h3>Tests</h3>
                    <div class="value">{total_tests:,}</div>
                    <div class="label">Total test cases</div>
                </div>
                <div class="stat-card">
                    <h3>Coverage</h3>
                    <div class="value">{weighted_coverage:.1f}%</div>
                    <div class="label">Weighted average</div>
                </div>
                <div class="stat-card">
                    <h3>Commits</h3>
                    <div class="value">{stats["git"]["total_commits"]:,}</div>
                    <div class="label">{stats["git"]["commits_last_30_days"]} in last 30d</div>
                </div>
            </div>

            <!-- Language Breakdown -->
            <section>
                <h2>Language Breakdown</h2>
                <table>
                    <tr>
                        <th>Language</th>
                        <th class="number">Files</th>
                        <th class="number">Code Lines</th>
                        <th class="number">Total Lines</th>
                        <th class="number">Comments</th>
                        <th class="number">% of Codebase</th>
                    </tr>
"""

    # Add rows for each language with code
    for lang in primary_langs + config_langs + doc_langs:
        lang_stats = stats["languages"].get(lang, {})
        if lang_stats.get("files", 0) == 0:
            continue

        pct = (
            (lang_stats["code_lines"] / total_code_lines * 100)
            if total_code_lines > 0
            else 0
        )
        color = LANGUAGE_CONFIG[lang]["color"]

        report_html += f"""                    <tr>
                        <td><span class="lang-badge" style="background: {color}">{lang.upper()}</span></td>
                        <td class="number">{lang_stats["files"]:,}</td>
                        <td class="number">{lang_stats["code_lines"]:,}</td>
                        <td class="number">{lang_stats["total_lines"]:,}</td>
                        <td class="number">{lang_stats["comment_lines"]:,}</td>
                        <td class="number">{pct:.1f}%</td>
                    </tr>
"""

    report_html += """                </table>

                <h3>Code Distribution</h3>
                <div class="progress-bar" style="height: 24px; display: flex;">
"""

    # Visual bar showing language distribution
    for lang in primary_langs:
        lang_stats = stats["languages"].get(lang, {})
        if lang_stats.get("code_lines", 0) == 0:
            continue
        pct = (
            (lang_stats["code_lines"] / total_code_lines * 100)
            if total_code_lines > 0
            else 0
        )
        if pct > 0.5:  # Only show if significant
            color = LANGUAGE_CONFIG[lang]["color"]
            report_html += f'                    <div class="progress-fill" style="width: {pct}%; background: {color};" title="{lang.title()}: {pct:.1f}%"></div>\n'

    report_html += """                </div>
                <div style="display: flex; gap: 1rem; flex-wrap: wrap; margin-top: 0.75rem; font-size: 0.85rem;">
"""

    for lang in primary_langs:
        lang_stats = stats["languages"].get(lang, {})
        if lang_stats.get("code_lines", 0) == 0:
            continue
        color = LANGUAGE_CONFIG[lang]["color"]
        report_html += f'                    <span><span style="display:inline-block;width:12px;height:12px;background:{color};border-radius:2px;margin-right:4px;"></span>{lang.title()}</span>\n'

    report_html += """                </div>
            </section>

            <!-- Primary Languages Detail -->
            <section>
                <h2>Primary Languages</h2>
                <div class="lang-breakdown">
"""

    # Python details
    py = stats["languages"]["python"]
    if py["files"] > 0:
        report_html += f"""                    <div class="lang-card" style="border-color: {LANGUAGE_CONFIG["python"]["color"]}">
                        <h4><span class="lang-badge" style="background: {LANGUAGE_CONFIG["python"]["color"]}">PYTHON</span> Core Backend</h4>
                        <div class="lang-stats">
                            <div><strong>{py["files"]:,}</strong> files | <strong>{py["code_lines"]:,}</strong> lines of code</div>
                            <div>Classes: {py["definitions"].get("classes", 0):,} | Functions: {py["definitions"].get("functions", 0):,} | Methods: {py["definitions"].get("methods", 0):,}</div>
                            <div>Coverage: <strong>{test_stats["python"]["coverage_percent"]:.1f}%</strong> | Tests: {test_stats["python"]["test_count"]:,}</div>
                        </div>
                    </div>
"""

    # TypeScript details
    ts = stats["languages"]["typescript"]
    if ts["files"] > 0:
        report_html += f"""                    <div class="lang-card" style="border-color: {LANGUAGE_CONFIG["typescript"]["color"]}">
                        <h4><span class="lang-badge" style="background: {LANGUAGE_CONFIG["typescript"]["color"]}">TYPESCRIPT</span> Viewer Frontend</h4>
                        <div class="lang-stats">
                            <div><strong>{ts["files"]:,}</strong> files | <strong>{ts["code_lines"]:,}</strong> lines of code</div>
                            <div>Classes: {ts["definitions"].get("classes", 0):,} | Functions: {ts["definitions"].get("functions", 0):,} | Interfaces: {ts["definitions"].get("interfaces", 0):,} | Types: {ts["definitions"].get("types", 0):,}</div>
                            <div>Coverage: <strong>{test_stats["typescript"]["coverage_percent"]:.1f}%</strong> | Tests: {test_stats["typescript"]["test_count"]:,}</div>
                        </div>
                    </div>
"""

    # Rust details
    rust = stats["languages"]["rust"]
    if rust["files"] > 0:
        report_html += f"""                    <div class="lang-card" style="border-color: {LANGUAGE_CONFIG["rust"]["color"]}">
                        <h4><span class="lang-badge" style="background: {LANGUAGE_CONFIG["rust"]["color"]}">RUST</span> WASM Module</h4>
                        <div class="lang-stats">
                            <div><strong>{rust["files"]:,}</strong> files | <strong>{rust["code_lines"]:,}</strong> lines of code</div>
                            <div>Structs: {rust["definitions"].get("structs", 0):,} | Functions: {rust["definitions"].get("functions", 0):,} | Traits: {rust["definitions"].get("traits", 0):,} | Impls: {rust["definitions"].get("impls", 0):,}</div>
                            <div>Tests: {test_stats["rust"]["test_count"]:,} passed</div>
                        </div>
                    </div>
"""

    # CUDA details
    cuda = stats["languages"]["cuda"]
    if cuda["files"] > 0:
        report_html += f"""                    <div class="lang-card" style="border-color: {LANGUAGE_CONFIG["cuda"]["color"]}">
                        <h4><span class="lang-badge" style="background: {LANGUAGE_CONFIG["cuda"]["color"]}">CUDA</span> GPU Acceleration</h4>
                        <div class="lang-stats">
                            <div><strong>{cuda["files"]:,}</strong> files | <strong>{cuda["code_lines"]:,}</strong> lines of code</div>
                            <div>Kernels: {cuda["definitions"].get("kernels", 0):,} | Device Functions: {cuda["definitions"].get("device_functions", 0):,} | Host Functions: {cuda["definitions"].get("host_functions", 0):,}</div>
                        </div>
                    </div>
"""

    # CSS details
    css = stats["languages"]["css"]
    if css["files"] > 0:
        report_html += f"""                    <div class="lang-card" style="border-color: {LANGUAGE_CONFIG["css"]["color"]}">
                        <h4><span class="lang-badge" style="background: {LANGUAGE_CONFIG["css"]["color"]}">CSS</span> Styling</h4>
                        <div class="lang-stats">
                            <div><strong>{css["files"]:,}</strong> files | <strong>{css["code_lines"]:,}</strong> lines of code</div>
                            <div>Rules: {css["definitions"].get("rules", 0):,} | Variables: {css["definitions"].get("variables", 0):,} | Media Queries: {css["definitions"].get("media_queries", 0):,}</div>
                        </div>
                    </div>
"""

    report_html += """                </div>
            </section>

            <!-- Test & Quality Metrics -->
            <section>
                <h2>Test & Quality Metrics</h2>
                <table>
                    <tr>
                        <th>Metric</th>
                        <th class="number">Python</th>
                        <th class="number">TypeScript</th>
                        <th class="number">Rust</th>
                        <th class="number">E2E</th>
                        <th class="number">Total</th>
                        <th>Status</th>
                    </tr>
                    <tr>
                        <td>Test Files</td>
                        <td class="number">{py_test_files}</td>
                        <td class="number">{ts_test_files}</td>
                        <td class="number">-</td>
                        <td class="number">{e2e_files}</td>
                        <td class="number">{total_test_files}</td>
                        <td><span class="health-indicator health-good">Good</span></td>
                    </tr>
                    <tr>
                        <td>Test Count</td>
                        <td class="number">{py_tests:,}</td>
                        <td class="number">{ts_tests:,}</td>
                        <td class="number">{rust_tests:,}</td>
                        <td class="number">-</td>
                        <td class="number">{total_tests:,}</td>
                        <td><span class="health-indicator health-good">Comprehensive</span></td>
                    </tr>
                    <tr>
                        <td>Tests Passed</td>
                        <td class="number">{py_passed:,}</td>
                        <td class="number">{ts_passed:,}</td>
                        <td class="number">{rust_passed:,}</td>
                        <td class="number">-</td>
                        <td class="number">{total_passed:,}</td>
                        <td>{pass_status}</td>
                    </tr>
                    <tr>
                        <td>Code Coverage</td>
                        <td class="number">{py_coverage:.1f}%</td>
                        <td class="number">{ts_coverage:.1f}%</td>
                        <td class="number">-</td>
                        <td class="number">-</td>
                        <td class="number">{weighted_coverage:.1f}%</td>
                        <td>{coverage_status}</td>
                    </tr>
                    <tr>
                        <td>Tests per 1K LOC</td>
                        <td class="number">{py_tests_per_kloc:.1f}</td>
                        <td class="number">{ts_tests_per_kloc:.1f}</td>
                        <td class="number">-</td>
                        <td class="number">-</td>
                        <td class="number">{total_tests_per_kloc:.1f}</td>
                        <td>{tests_per_kloc_status}</td>
                    </tr>
                </table>
            </section>
""".format(
        py_test_files=test_stats["python"]["test_files"],
        ts_test_files=test_stats["typescript"]["test_files"],
        e2e_files=test_stats["e2e"]["test_files"],
        total_test_files=total_test_files,
        py_tests=test_stats["python"]["test_count"],
        ts_tests=test_stats["typescript"]["test_count"],
        rust_tests=test_stats["rust"]["test_count"],
        total_tests=total_tests,
        py_passed=test_stats["python"]["test_passed"],
        ts_passed=test_stats["typescript"]["test_passed"],
        rust_passed=test_stats["rust"]["test_passed"],
        total_passed=total_passed,
        pass_status='<span class="health-indicator health-good">All Passing</span>'
        if total_passed == total_tests
        else '<span class="health-indicator health-warning">Some Failures</span>',
        py_coverage=test_stats["python"]["coverage_percent"],
        ts_coverage=test_stats["typescript"]["coverage_percent"],
        weighted_coverage=weighted_coverage,
        coverage_status='<span class="health-indicator health-good">Excellent</span>'
        if weighted_coverage >= 80
        else '<span class="health-indicator health-warning">Good</span>'
        if weighted_coverage >= 60
        else '<span class="health-indicator health-danger">Needs Work</span>',
        py_tests_per_kloc=(
            test_stats["python"]["test_count"] / (py["code_lines"] / 1000)
        )
        if py["code_lines"] > 0
        else 0,
        ts_tests_per_kloc=(
            test_stats["typescript"]["test_count"] / (ts["code_lines"] / 1000)
        )
        if ts["code_lines"] > 0
        else 0,
        total_tests_per_kloc=(total_tests / (total_code_lines / 1000))
        if total_code_lines > 0
        else 0,
        tests_per_kloc_status='<span class="health-indicator health-good">Well Tested</span>'
        if total_tests / max(1, total_code_lines / 1000) > 15
        else '<span class="health-indicator health-warning">Adequate</span>',
    )

    # Git Statistics
    git = stats["git"]
    report_html += f"""
            <section>
                <h2>Git Statistics</h2>
                <div class="two-column">
                    <div>
                        <table>
                            <tr>
                                <th>Metric</th>
                                <th class="number">Value</th>
                            </tr>
                            <tr>
                                <td>Total Commits</td>
                                <td class="number">{git["total_commits"]:,}</td>
                            </tr>
                            <tr>
                                <td>Contributors</td>
                                <td class="number">{git["contributors"]}</td>
                            </tr>
                            <tr>
                                <td>Branches</td>
                                <td class="number">{git["branches"]}</td>
                            </tr>
                            <tr>
                                <td>Tags</td>
                                <td class="number">{git["tags"]}</td>
                            </tr>
                            <tr>
                                <td>Commits (Last 30 Days)</td>
                                <td class="number">{git["commits_last_30_days"]}</td>
                            </tr>
                            <tr>
                                <td>First Commit</td>
                                <td class="number">{git["first_commit_date"] or "N/A"}</td>
                            </tr>
                            <tr>
                                <td>Last Commit</td>
                                <td class="number">{git["last_commit_date"] or "N/A"}</td>
                            </tr>
                        </table>
                    </div>
                    <div>
                        <h3>Top Contributors</h3>
                        <table>
                            <tr>
                                <th>Name</th>
                                <th class="number">Commits</th>
                            </tr>
"""

    for contributor in git.get("top_contributors", [])[:5]:
        # Escape contributor name to prevent XSS
        safe_name = html.escape(contributor["name"])
        report_html += f"""                            <tr>
                                <td>{safe_name}</td>
                                <td class="number">{contributor["commits"]:,}</td>
                            </tr>
"""

    report_html += """                        </table>
                    </div>
                </div>
            </section>
"""

    # Dependencies
    deps = stats["dependencies"]
    report_html += f"""
            <section>
                <h2>Dependencies</h2>
                <table>
                    <tr>
                        <th>Ecosystem</th>
                        <th class="number">Production</th>
                        <th class="number">Development</th>
                        <th class="number">Total</th>
                    </tr>
                    <tr>
                        <td><span class="lang-badge" style="background: {LANGUAGE_CONFIG["python"]["color"]}">Python</span></td>
                        <td class="number">{deps["python"]["production"]}</td>
                        <td class="number">{deps["python"]["dev"]}</td>
                        <td class="number">{deps["python"]["production"] + deps["python"]["dev"]}</td>
                    </tr>
                    <tr>
                        <td><span class="lang-badge" style="background: {LANGUAGE_CONFIG["typescript"]["color"]}">Node.js</span></td>
                        <td class="number">{deps["node"]["production"]}</td>
                        <td class="number">{deps["node"]["dev"]}</td>
                        <td class="number">{deps["node"]["production"] + deps["node"]["dev"]}</td>
                    </tr>
                    <tr>
                        <td><span class="lang-badge" style="background: {LANGUAGE_CONFIG["rust"]["color"]}">Rust</span></td>
                        <td class="number">{deps["rust"]["production"]}</td>
                        <td class="number">{deps["rust"]["dev"]}</td>
                        <td class="number">{deps["rust"]["production"] + deps["rust"]["dev"]}</td>
                    </tr>
                </table>
            </section>
"""

    # Documentation & Configuration
    md = stats["languages"]["markdown"]
    json_stats = stats["languages"]["json"]
    yaml_stats = stats["languages"]["yaml"]
    toml_stats = stats["languages"]["toml"]

    report_html += f"""
            <section>
                <h2>Documentation & Configuration</h2>
                <div class="two-column">
                    <div>
                        <h3>Documentation</h3>
                        <table>
                            <tr>
                                <th>Type</th>
                                <th class="number">Files</th>
                                <th class="number">Lines</th>
                            </tr>
                            <tr>
                                <td>Markdown (.md)</td>
                                <td class="number">{md["files"]}</td>
                                <td class="number">{md["total_lines"]:,}</td>
                            </tr>
                            <tr>
                                <td>HTML</td>
                                <td class="number">{stats["languages"]["html"]["files"]}</td>
                                <td class="number">{stats["languages"]["html"]["total_lines"]:,}</td>
                            </tr>
                        </table>
                    </div>
                    <div>
                        <h3>Configuration</h3>
                        <table>
                            <tr>
                                <th>Type</th>
                                <th class="number">Files</th>
                                <th class="number">Lines</th>
                            </tr>
                            <tr>
                                <td>JSON</td>
                                <td class="number">{json_stats["files"]}</td>
                                <td class="number">{json_stats["total_lines"]:,}</td>
                            </tr>
                            <tr>
                                <td>YAML</td>
                                <td class="number">{yaml_stats["files"]}</td>
                                <td class="number">{yaml_stats["total_lines"]:,}</td>
                            </tr>
                            <tr>
                                <td>TOML</td>
                                <td class="number">{toml_stats["files"]}</td>
                                <td class="number">{toml_stats["total_lines"]:,}</td>
                            </tr>
                        </table>
                    </div>
                </div>
            </section>
"""

    # Key Insights
    code_density = (total_code_lines / total_lines * 100) if total_lines > 0 else 0
    primary_code_lines = sum(
        stats["languages"][l]["code_lines"]
        for l in primary_langs
        if stats["languages"].get(l, {}).get("code_lines", 0) > 0
    )
    py_pct = (
        (py["code_lines"] / primary_code_lines * 100) if primary_code_lines > 0 else 0
    )
    ts_pct = (
        (ts["code_lines"] / primary_code_lines * 100) if primary_code_lines > 0 else 0
    )
    rust_pct = (
        (rust["code_lines"] / primary_code_lines * 100) if primary_code_lines > 0 else 0
    )
    cuda_pct = (
        (cuda["code_lines"] / primary_code_lines * 100) if primary_code_lines > 0 else 0
    )

    report_html += f"""
            <section>
                <h2>Key Insights</h2>
                <div class="insight">
                    <strong>Code Density:</strong> {code_density:.1f}% of all lines are executable code (excluding blanks and comments)
                </div>
                <div class="insight">
                    <strong>Language Mix:</strong> Python {py_pct:.1f}% | TypeScript {ts_pct:.1f}% | Rust {rust_pct:.1f}% | CUDA {cuda_pct:.1f}%
                </div>
                <div class="insight">
                    <strong>Architecture:</strong> {py["definitions"].get("classes", 0)} Python classes, {ts["definitions"].get("interfaces", 0)} TypeScript interfaces, {rust["definitions"].get("structs", 0)} Rust structs
                </div>
                <div class="insight">
                    <strong>Test Health:</strong> {total_tests:,} tests across {total_test_files} files with {weighted_coverage:.1f}% weighted coverage
                </div>
                <div class="insight">
                    <strong>Project Activity:</strong> {git["commits_last_30_days"]} commits in the last 30 days by {git["contributors"]} contributor(s)
                </div>
                <div class="insight">
                    <strong>Documentation:</strong> {md["files"]} markdown files with {md["total_lines"]:,} lines of documentation
                </div>
            </section>
        </div>

        <footer>
            Generated by Luxar Project Analyzer | {now.strftime("%Y")} |
            Excluding: node_modules, __pycache__, coverage, dist, build, target
        </footer>
    </div>
</body>
</html>
"""

    with open(output_file, "w", encoding="utf-8") as f:
        f.write(report_html)


def main():
    """Main analysis function."""
    with asection("Analyzing Luxar project"):
        # Get project root
        script_path = Path(__file__).resolve()
        if script_path.parent.name == "stats":
            project_root = script_path.parent.parent
        else:
            project_root = script_path.parent

        # Analyze all languages
        all_stats = {"languages": {}}

        with asection("Analyzing code by language"):
            for lang in LANGUAGE_CONFIG:
                aprint(f"Scanning {lang}...")
                all_stats["languages"][lang] = analyze_directory(project_root, lang)
                files = all_stats["languages"][lang]["files"]
                lines = all_stats["languages"][lang]["code_lines"]
                if files > 0:
                    aprint(f"  {files} files, {lines:,} lines of code")

        # Get test statistics (runs actual tests with coverage)
        with asection("Running tests and gathering coverage"):
            all_stats["tests"] = get_test_statistics(project_root)

        # Get Git statistics
        with asection("Gathering Git statistics"):
            all_stats["git"] = get_git_statistics(project_root)
            aprint(f"Total commits: {all_stats['git']['total_commits']}")
            aprint(f"Contributors: {all_stats['git']['contributors']}")

        # Get dependency statistics
        with asection("Analyzing dependencies"):
            all_stats["dependencies"] = get_dependency_statistics(project_root)

        # Generate HTML report
        stats_dir = project_root / "stats"
        stats_dir.mkdir(exist_ok=True)
        output_file = stats_dir / "project_stats.html"

        with asection("Generating HTML report"):
            generate_html_report(all_stats, output_file)
            aprint(f"Report saved to: {output_file}")

    # Print summary
    aprint("\n" + "=" * 60)
    aprint("SUMMARY")
    aprint("=" * 60)

    total_files = sum(
        all_stats["languages"][lang]["files"] for lang in all_stats["languages"]
    )
    total_code = sum(
        all_stats["languages"][lang]["code_lines"] for lang in all_stats["languages"]
    )

    aprint(f"Total files: {total_files:,}")
    aprint(f"Total lines of code: {total_code:,}")
    aprint("")

    # Primary languages summary
    primary_langs = ["python", "typescript", "rust", "cuda", "css"]
    for lang in primary_langs:
        stats = all_stats["languages"][lang]
        if stats["files"] > 0:
            aprint(
                f"{lang.title():12} {stats['files']:>4} files, {stats['code_lines']:>6,} LOC"
            )

    aprint("")

    # Test summary
    tests = all_stats["tests"]
    total_tests = (
        tests["python"]["test_count"]
        + tests["typescript"]["test_count"]
        + tests["rust"]["test_count"]
    )
    aprint(f"Total tests: {total_tests:,}")
    aprint(
        f"Python coverage: {tests['python']['coverage_percent']:.1f}% | "
        f"TypeScript coverage: {tests['typescript']['coverage_percent']:.1f}%"
    )

    aprint("")
    aprint(f"Report: {output_file}")


if __name__ == "__main__":
    main()
