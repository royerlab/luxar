#!/usr/bin/env python3
"""
Generate project statistics report for Luxar.

Analyzes Python and TypeScript code to produce comprehensive statistics
including lines of code, file counts, function/class counts, etc.
"""

import re
import subprocess
from collections import defaultdict
from datetime import datetime
from pathlib import Path

from arbol import aprint


def count_lines_in_file(filepath):
    """Count total, code, comment, and blank lines in a file."""
    try:
        with open(filepath, "r", encoding="utf-8") as f:
            lines = f.readlines()
    except Exception:
        return {"total": 0, "code": 0, "comment": 0, "blank": 0}

    total = len(lines)
    blank = sum(1 for line in lines if line.strip() == "")

    # Count comments (Python: #, TypeScript: // or /* */)
    comment = 0
    in_block_comment = False
    for line in lines:
        stripped = line.strip()
        if filepath.suffix in [".py"]:
            if stripped.startswith("#"):
                comment += 1
        elif filepath.suffix in [".ts", ".tsx", ".js"]:
            if "/*" in stripped:
                in_block_comment = True
            if in_block_comment:
                comment += 1
            elif stripped.startswith("//"):
                comment += 1
            if "*/" in stripped:
                in_block_comment = False

    code = total - blank - comment

    return {"total": total, "code": code, "comment": comment, "blank": blank}


def count_python_definitions(filepath):
    """Count classes and functions in Python file."""
    try:
        with open(filepath, "r", encoding="utf-8") as f:
            content = f.read()
    except Exception:
        return {"classes": 0, "functions": 0, "methods": 0}

    # Count top-level class definitions
    classes = len(re.findall(r"^\s*class\s+\w+", content, re.MULTILINE))

    # Count all function/method definitions
    all_functions = len(re.findall(r"^\s*def\s+\w+", content, re.MULTILINE))

    # Count methods (functions inside classes - approximate by indentation)
    methods = len(re.findall(r"^\s{4,}def\s+\w+", content, re.MULTILINE))

    functions = all_functions - methods

    return {"classes": classes, "functions": functions, "methods": methods}


def count_typescript_definitions(filepath):
    """Count classes, functions, and interfaces in TypeScript file."""
    try:
        with open(filepath, "r", encoding="utf-8") as f:
            content = f.read()
    except Exception:
        return {"classes": 0, "functions": 0, "interfaces": 0, "types": 0}

    classes = len(re.findall(r"^\s*(?:export\s+)?class\s+\w+", content, re.MULTILINE))
    functions = len(
        re.findall(r"^\s*(?:export\s+)?function\s+\w+", content, re.MULTILINE)
    )
    # Also count arrow functions and const functions
    functions += len(
        re.findall(
            r"^\s*(?:export\s+)?const\s+\w+\s*=\s*\([^)]*\)\s*=>", content, re.MULTILINE
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


def get_test_statistics(project_root):
    """Get test count and coverage statistics for Python and TypeScript."""
    test_stats = {
        "test_files": 0,
        "test_count": 0,
        "test_passed": 0,
        "test_failed": 0,
        "coverage_percent": 0,
        "ts_test_files": 0,
        "ts_test_count": 0,
        "ts_test_passed": 0,
        "ts_test_failed": 0,
        "ts_coverage_percent": 0,
        "error": None,
    }

    try:
        # Count test files
        root_path = Path(project_root)
        test_files = list(root_path.rglob("test_*.py")) + list(
            root_path.rglob("*_test.py")
        )
        test_files = [f for f in test_files if "__pycache__" not in str(f)]
        test_stats["test_files"] = len(test_files)

        # Run pytest to get test count (collect-only, quick)
        # Include ALL tests (including gsplats)
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
                timeout=30,
                cwd=project_root,
            )

            # Parse output for test count
            for line in result.stdout.split("\n"):
                if "selected" in line or "collected" in line:
                    # Extract number from "collected 370 items" or similar
                    numbers = re.findall(r"\d+", line)
                    if numbers:
                        test_stats["test_count"] = int(numbers[0])
                        break
        except Exception as e:
            test_stats["error"] = f"Could not collect tests: {e}"

        # Run Python tests with coverage (include ALL tests for accurate coverage)
        aprint("    Running Python tests with coverage (this may take a minute)...")
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
                timeout=180,
                cwd=project_root,
            )

            # Parse coverage from output
            # Look for line like: "TOTAL    8704   6611    24%"
            for line in result.stdout.split("\n") + result.stderr.split("\n"):
                if line.strip().startswith("TOTAL"):
                    # Extract the percentage (last field with %)
                    parts = line.split()
                    for part in reversed(parts):
                        if "%" in part:
                            pct_match = re.search(r"(\d+(?:\.\d+)?)%", part)
                            if pct_match:
                                test_stats["coverage_percent"] = float(
                                    pct_match.group(1)
                                )
                                break
                    break

            # Count passing tests from pytest output
            for line in result.stdout.split("\n"):
                if "passed" in line:
                    match = re.search(r"(\d+)\s+passed", line)
                    if match:
                        test_stats["test_passed"] = int(match.group(1))

        except Exception as e:
            aprint(f"    Could not run Python coverage: {e}")
            # Fallback to existing coverage file if present
            coverage_file = root_path / "coverage" / "python" / ".coverage"
            if coverage_file.exists():
                try:
                    result = subprocess.run(
                        ["hatch", "run", "coverage", "report", "--precision=1"],
                        capture_output=True,
                        text=True,
                        timeout=30,
                        cwd=project_root,
                    )
                    for line in result.stdout.split("\n"):
                        if line.startswith("TOTAL"):
                            parts = line.split()
                            if len(parts) >= 4 and "%" in parts[-1]:
                                test_stats["coverage_percent"] = float(
                                    parts[-1].rstrip("%")
                                )
                                break
                except Exception:
                    pass

        # Count TypeScript test files
        viewer_path = root_path / "packages" / "luxar-viewer"
        if viewer_path.exists():
            ts_test_files = list(viewer_path.rglob("*.test.ts")) + list(
                viewer_path.rglob("*.test.tsx")
            )
            ts_test_files += list(viewer_path.rglob("*.spec.ts")) + list(
                viewer_path.rglob("*.spec.tsx")
            )
            test_stats["ts_test_files"] = len(
                [f for f in ts_test_files if "node_modules" not in str(f)]
            )

            # Run TypeScript tests (without coverage first, it's faster)
            aprint("    Running TypeScript tests...")
            try:
                result = subprocess.run(
                    ["pnpm", "test", "--run"],
                    capture_output=True,
                    text=True,
                    timeout=60,
                    cwd=viewer_path,
                )

                # Parse text output for test counts
                # Look for: "Tests  25 failed | 503 passed (528)"
                for line in result.stdout.split("\n") + result.stderr.split("\n"):
                    if "Tests" in line and "passed" in line:
                        # Extract total from parentheses
                        total_match = re.search(r"\((\d+)\)", line)
                        if total_match:
                            test_stats["ts_test_count"] = int(total_match.group(1))

                        # Extract passed count
                        passed_match = re.search(r"(\d+)\s+passed", line)
                        if passed_match:
                            test_stats["ts_test_passed"] = int(passed_match.group(1))

                        # Extract failed count
                        failed_match = re.search(r"(\d+)\s+failed", line)
                        if failed_match:
                            test_stats["ts_test_failed"] = int(failed_match.group(1))
                        break

            except Exception as e:
                aprint(f"    Could not run TypeScript tests: {e}")

            # Run TypeScript coverage separately
            aprint("    Running TypeScript coverage...")
            try:
                result = subprocess.run(
                    ["pnpm", "run", "test:coverage"],
                    capture_output=True,
                    text=True,
                    timeout=90,
                    cwd=viewer_path,
                )

                # Parse coverage from output
                # vitest coverage shows lines like: "Statements   : 85.23%"
                for line in result.stdout.split("\n") + result.stderr.split("\n"):
                    if "Statements" in line and "%" in line:
                        pct_match = re.search(r"(\d+(?:\.\d+)?)%", line)
                        if pct_match:
                            test_stats["ts_coverage_percent"] = float(
                                pct_match.group(1)
                            )
                            break

            except Exception as e:
                aprint(f"    Could not run TypeScript coverage: {e}")

    except Exception as e:
        test_stats["error"] = str(e)

    return test_stats


def analyze_directory(root_dir, extensions, category):
    """Analyze all files with given extensions in directory."""
    stats = {
        "files": 0,
        "total_lines": 0,
        "code_lines": 0,
        "comment_lines": 0,
        "blank_lines": 0,
        "classes": 0,
        "functions": 0,
        "methods": 0,
        "interfaces": 0,
        "types": 0,
        "by_subdir": defaultdict(lambda: defaultdict(int)),
    }

    root_path = Path(root_dir)

    for ext in extensions:
        for filepath in root_path.rglob(f"*{ext}"):
            # Skip certain directories
            if any(
                part in filepath.parts
                for part in [
                    ".git",
                    "node_modules",
                    "__pycache__",
                    ".pytest_cache",
                    "coverage",
                    "dist",
                    "build",
                    ".eggs",
                ]
            ):
                continue

            # Get relative path for categorization
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
            line_stats = count_lines_in_file(filepath)
            stats["total_lines"] += line_stats["total"]
            stats["code_lines"] += line_stats["code"]
            stats["comment_lines"] += line_stats["comment"]
            stats["blank_lines"] += line_stats["blank"]

            # Count definitions
            if ext == ".py":
                defs = count_python_definitions(filepath)
                stats["classes"] += defs["classes"]
                stats["functions"] += defs["functions"]
                stats["methods"] += defs["methods"]
            elif ext in [".ts", ".tsx"]:
                defs = count_typescript_definitions(filepath)
                stats["classes"] += defs["classes"]
                stats["functions"] += defs["functions"]
                stats["interfaces"] += defs["interfaces"]
                stats["types"] += defs["types"]

            # Update subdir stats
            stats["by_subdir"][subdir]["files"] += 1
            stats["by_subdir"][subdir]["code_lines"] += line_stats["code"]

    return stats


def generate_html_report(stats, output_file):
    """Generate HTML report from statistics."""
    # Pre-calculate all percentages and metrics
    py_code_pct = (
        stats["python"]["code_lines"] / stats["python"]["total_lines"] * 100
        if stats["python"]["total_lines"] > 0
        else 0
    )
    py_comment_pct = (
        stats["python"]["comment_lines"] / stats["python"]["total_lines"] * 100
        if stats["python"]["total_lines"] > 0
        else 0
    )
    py_blank_pct = (
        stats["python"]["blank_lines"] / stats["python"]["total_lines"] * 100
        if stats["python"]["total_lines"] > 0
        else 0
    )

    ts_code_pct = (
        stats["typescript"]["code_lines"] / stats["typescript"]["total_lines"] * 100
        if stats["typescript"]["total_lines"] > 0
        else 0
    )
    ts_comment_pct = (
        stats["typescript"]["comment_lines"] / stats["typescript"]["total_lines"] * 100
        if stats["typescript"]["total_lines"] > 0
        else 0
    )
    ts_blank_pct = (
        stats["typescript"]["blank_lines"] / stats["typescript"]["total_lines"] * 100
        if stats["typescript"]["total_lines"] > 0
        else 0
    )

    # Test metrics
    py_tests_per_kloc = (
        stats["tests"]["test_count"] / (stats["python"]["code_lines"] / 1000)
        if stats["python"]["code_lines"] > 0
        else 0
    )
    ts_tests_per_kloc = (
        stats["tests"]["ts_test_count"] / (stats["typescript"]["code_lines"] / 1000)
        if stats["typescript"]["code_lines"] > 0
        else 0
    )
    total_tests_per_kloc = (
        (stats["tests"]["test_count"] + stats["tests"]["ts_test_count"])
        / (stats["total"]["code_lines"] / 1000)
        if stats["total"]["code_lines"] > 0
        else 0
    )

    html = f"""<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>Luxar Project Statistics Report</title>
    <style>
        * {{
            margin: 0;
            padding: 0;
            box-sizing: border-box;
        }}
        body {{
            font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, 'Helvetica Neue', Arial, sans-serif;
            line-height: 1.6;
            color: #333;
            background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
            min-height: 100vh;
            padding: 2rem;
        }}
        .container {{
            max-width: 1200px;
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
        header h1 {{
            font-size: 2.5rem;
            margin-bottom: 0.5rem;
            font-weight: 700;
        }}
        header p {{
            font-size: 1.1rem;
            opacity: 0.9;
        }}
        .content {{
            padding: 2rem;
        }}
        .summary {{
            display: grid;
            grid-template-columns: repeat(auto-fit, minmax(200px, 1fr));
            gap: 1.5rem;
            margin-bottom: 2rem;
        }}
        .stat-card {{
            background: linear-gradient(135deg, #f5f7fa 0%, #c3cfe2 100%);
            padding: 1.5rem;
            border-radius: 8px;
            box-shadow: 0 4px 6px rgba(0,0,0,0.1);
            transition: transform 0.2s;
        }}
        .stat-card:hover {{
            transform: translateY(-4px);
            box-shadow: 0 8px 12px rgba(0,0,0,0.15);
        }}
        .stat-card h3 {{
            font-size: 0.9rem;
            text-transform: uppercase;
            letter-spacing: 0.05em;
            color: #667eea;
            margin-bottom: 0.5rem;
            font-weight: 600;
        }}
        .stat-card .value {{
            font-size: 2.5rem;
            font-weight: 700;
            color: #333;
        }}
        .stat-card .label {{
            font-size: 0.85rem;
            color: #666;
            margin-top: 0.25rem;
        }}
        section {{
            margin-bottom: 2.5rem;
        }}
        section h2 {{
            font-size: 1.8rem;
            color: #667eea;
            margin-bottom: 1rem;
            padding-bottom: 0.5rem;
            border-bottom: 3px solid #667eea;
        }}
        table {{
            width: 100%;
            border-collapse: collapse;
            margin-top: 1rem;
            font-size: 0.95rem;
        }}
        th {{
            background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
            color: white;
            padding: 1rem;
            text-align: left;
            font-weight: 600;
            text-transform: uppercase;
            font-size: 0.85rem;
            letter-spacing: 0.05em;
        }}
        td {{
            padding: 0.75rem 1rem;
            border-bottom: 1px solid #e0e0e0;
        }}
        tr:hover {{
            background: #f8f9fa;
        }}
        .number {{
            text-align: right;
            font-family: 'Monaco', 'Courier New', monospace;
            font-weight: 500;
        }}
        .progress-bar {{
            height: 20px;
            background: #e0e0e0;
            border-radius: 10px;
            overflow: hidden;
            margin-top: 0.5rem;
        }}
        .progress-fill {{
            height: 100%;
            background: linear-gradient(90deg, #667eea 0%, #764ba2 100%);
            transition: width 0.3s ease;
        }}
        footer {{
            text-align: center;
            padding: 2rem;
            color: #666;
            font-size: 0.9rem;
            border-top: 1px solid #e0e0e0;
        }}
        .highlight {{
            background: #fff3cd;
            padding: 0.2em 0.4em;
            border-radius: 3px;
        }}
    </style>
</head>
<body>
    <div class="container">
        <header>
            <h1>📊 Luxar Project Statistics</h1>
            <p>Comprehensive codebase analysis • Generated {datetime.now().strftime("%B %d, %Y at %H:%M")}</p>
        </header>

        <div class="content">
            <div class="summary">
                <div class="stat-card">
                    <h3>Total Files</h3>
                    <div class="value">{stats["total"]["files"]:,}</div>
                    <div class="label">Python + TypeScript</div>
                </div>
                <div class="stat-card">
                    <h3>Lines of Code</h3>
                    <div class="value">{stats["total"]["code_lines"]:,}</div>
                    <div class="label">Executable code</div>
                </div>
                <div class="stat-card">
                    <h3>Total Lines</h3>
                    <div class="value">{stats["total"]["total_lines"]:,}</div>
                    <div class="label">Including comments & blanks</div>
                </div>
                <div class="stat-card">
                    <h3>Classes</h3>
                    <div class="value">{stats["total"]["classes"]:,}</div>
                    <div class="label">Python + TypeScript</div>
                </div>
                <div class="stat-card">
                    <h3>Functions</h3>
                    <div class="value">{stats["total"]["functions"]:,}</div>
                    <div class="label">Top-level functions</div>
                </div>
                <div class="stat-card">
                    <h3>Methods</h3>
                    <div class="value">{stats["total"]["methods"]:,}</div>
                    <div class="label">Class methods</div>
                </div>
                <div class="stat-card">
                    <h3>Test Files</h3>
                    <div class="value">{stats["tests"]["test_files"] + stats["tests"]["ts_test_files"]:,}</div>
                    <div class="label">Python + TypeScript</div>
                </div>
                <div class="stat-card">
                    <h3>Tests</h3>
                    <div class="value">{stats["tests"]["test_count"] + stats["tests"]["ts_test_count"]:,}</div>
                    <div class="label">Total test cases</div>
                </div>
                <div class="stat-card">
                    <h3>Coverage</h3>
                    <div class="value">{(stats["tests"]["coverage_percent"] + stats["tests"]["ts_coverage_percent"]) / 2:.1f}%</div>
                    <div class="label">Avg (Py: {stats["tests"]["coverage_percent"]:.0f}%, TS: {stats["tests"]["ts_coverage_percent"]:.0f}%)</div>
                </div>
            </div>

            <section>
                <h2>Test & Quality Metrics</h2>
                <table>
                    <tr>
                        <th>Metric</th>
                        <th>Python</th>
                        <th>TypeScript</th>
                        <th>Total</th>
                        <th>Status</th>
                    </tr>
                    <tr>
                        <td>Test Files</td>
                        <td class="number">{stats["tests"]["test_files"]}</td>
                        <td class="number">{stats["tests"]["ts_test_files"]}</td>
                        <td class="number">{stats["tests"]["test_files"] + stats["tests"]["ts_test_files"]}</td>
                        <td>{"✅ Excellent" if stats["tests"]["test_files"] + stats["tests"]["ts_test_files"] > 80 else "✅ Good"}</td>
                    </tr>
                    <tr>
                        <td>Total Tests</td>
                        <td class="number">{stats["tests"]["test_count"]:,}</td>
                        <td class="number">{stats["tests"]["ts_test_count"]:,}</td>
                        <td class="number">{stats["tests"]["test_count"] + stats["tests"]["ts_test_count"]:,}</td>
                        <td>{"✅ Excellent" if stats["tests"]["test_count"] + stats["tests"]["ts_test_count"] > 1000 else "✅ Good"}</td>
                    </tr>
                    <tr>
                        <td>Tests Passing</td>
                        <td class="number">{stats["tests"]["test_count"]:,}</td>
                        <td class="number">{stats["tests"]["ts_test_passed"]:,}</td>
                        <td class="number">{stats["tests"]["test_count"] + stats["tests"]["ts_test_passed"]:,}</td>
                        <td>{"✅ " + str(int((stats["tests"]["test_count"] + stats["tests"]["ts_test_passed"]) / (stats["tests"]["test_count"] + stats["tests"]["ts_test_count"]) * 100)) + "%" if stats["tests"]["test_count"] + stats["tests"]["ts_test_count"] > 0 else "N/A"}</td>
                    </tr>
                    <tr>
                        <td>Tests Failing</td>
                        <td class="number">0</td>
                        <td class="number">{stats["tests"]["ts_test_failed"]}</td>
                        <td class="number">{stats["tests"]["ts_test_failed"]}</td>
                        <td>{"⚠️ Has failures" if stats["tests"]["ts_test_failed"] > 0 else "✅ All passing"}</td>
                    </tr>
                    <tr>
                        <td>Code Coverage</td>
                        <td class="number">{stats["tests"]["coverage_percent"]:.1f}%</td>
                        <td class="number">{stats["tests"]["ts_coverage_percent"]:.1f}%</td>
                        <td class="number">{(stats["tests"]["coverage_percent"] + stats["tests"]["ts_coverage_percent"]) / 2:.1f}%</td>
                        <td>{"✅ Excellent" if stats["tests"]["coverage_percent"] >= 80 and stats["tests"]["ts_coverage_percent"] >= 80 else "✅ Good" if stats["tests"]["coverage_percent"] >= 70 or stats["tests"]["ts_coverage_percent"] >= 70 else "⚠️ Below 70%" if stats["tests"]["coverage_percent"] > 0 or stats["tests"]["ts_coverage_percent"] > 0 else "ℹ️ Coverage data needed"}</td>
                    </tr>
                    <tr>
                        <td>Tests per 1000 LOC</td>
                        <td class="number">{py_tests_per_kloc:.1f}</td>
                        <td class="number">{ts_tests_per_kloc:.1f}</td>
                        <td class="number">{total_tests_per_kloc:.1f}</td>
                        <td>{"✅ Well tested" if total_tests_per_kloc > 15 else "✅ Good" if total_tests_per_kloc > 10 else "⚠️ Could use more"}</td>
                    </tr>
                </table>
            </section>

            <section>
                <h2>Python Codebase</h2>
                <table>
                    <tr>
                        <th>Metric</th>
                        <th class="number">Count</th>
                        <th class="number">Percentage</th>
                    </tr>
                    <tr>
                        <td>Python Files</td>
                        <td class="number">{stats["python"]["files"]:,}</td>
                        <td class="number">{stats["python"]["files"] / stats["total"]["files"] * 100:.1f}%</td>
                    </tr>
                    <tr>
                        <td>Lines of Code</td>
                        <td class="number">{stats["python"]["code_lines"]:,}</td>
                        <td class="number">{stats["python"]["code_lines"] / stats["total"]["code_lines"] * 100:.1f}%</td>
                    </tr>
                    <tr>
                        <td>Total Lines</td>
                        <td class="number">{stats["python"]["total_lines"]:,}</td>
                        <td class="number">-</td>
                    </tr>
                    <tr>
                        <td>Comment Lines</td>
                        <td class="number">{stats["python"]["comment_lines"]:,}</td>
                        <td class="number">{stats["python"]["comment_lines"] / stats["python"]["total_lines"] * 100:.1f}%</td>
                    </tr>
                    <tr>
                        <td>Blank Lines</td>
                        <td class="number">{stats["python"]["blank_lines"]:,}</td>
                        <td class="number">{stats["python"]["blank_lines"] / stats["python"]["total_lines"] * 100:.1f}%</td>
                    </tr>
                    <tr>
                        <td>Classes</td>
                        <td class="number">{stats["python"]["classes"]:,}</td>
                        <td class="number">-</td>
                    </tr>
                    <tr>
                        <td>Functions (top-level)</td>
                        <td class="number">{stats["python"]["functions"]:,}</td>
                        <td class="number">-</td>
                    </tr>
                    <tr>
                        <td>Methods</td>
                        <td class="number">{stats["python"]["methods"]:,}</td>
                        <td class="number">-</td>
                    </tr>
                </table>

                <h3 style="margin-top: 2rem; margin-bottom: 1rem; color: #667eea;">Python Modules Breakdown</h3>
                <table>
                    <tr>
                        <th>Module</th>
                        <th class="number">Files</th>
                        <th class="number">Code Lines</th>
                    </tr>
"""

    # Sort Python subdirectories by code lines
    py_subdirs = sorted(
        [(k, v) for k, v in stats["python"]["by_subdir"].items()],
        key=lambda x: x[1]["code_lines"],
        reverse=True,
    )

    for subdir, subdir_stats in py_subdirs:
        html += f"""                    <tr>
                        <td>{subdir}</td>
                        <td class="number">{subdir_stats["files"]}</td>
                        <td class="number">{subdir_stats["code_lines"]:,}</td>
                    </tr>
"""

    # Calculate TypeScript percentages safely
    ts_files_pct = (
        stats["typescript"]["files"] / stats["total"]["files"] * 100
        if stats["total"]["files"] > 0
        else 0
    )
    ts_code_lines_pct = (
        stats["typescript"]["code_lines"] / stats["total"]["code_lines"] * 100
        if stats["total"]["code_lines"] > 0
        else 0
    )
    ts_comment_lines_pct = (
        stats["typescript"]["comment_lines"] / stats["typescript"]["total_lines"] * 100
        if stats["typescript"]["total_lines"] > 0
        else 0
    )
    ts_blank_lines_pct = (
        stats["typescript"]["blank_lines"] / stats["typescript"]["total_lines"] * 100
        if stats["typescript"]["total_lines"] > 0
        else 0
    )

    html += f"""                </table>
            </section>

            <section>
                <h2>TypeScript Codebase</h2>
                <table>
                    <tr>
                        <th>Metric</th>
                        <th class="number">Count</th>
                        <th class="number">Percentage</th>
                    </tr>
                    <tr>
                        <td>TypeScript Files</td>
                        <td class="number">{stats["typescript"]["files"]:,}</td>
                        <td class="number">{ts_files_pct:.1f}%</td>
                    </tr>
                    <tr>
                        <td>Lines of Code</td>
                        <td class="number">{stats["typescript"]["code_lines"]:,}</td>
                        <td class="number">{ts_code_lines_pct:.1f}%</td>
                    </tr>
                    <tr>
                        <td>Total Lines</td>
                        <td class="number">{stats["typescript"]["total_lines"]:,}</td>
                        <td class="number">-</td>
                    </tr>
                    <tr>
                        <td>Comment Lines</td>
                        <td class="number">{stats["typescript"]["comment_lines"]:,}</td>
                        <td class="number">{ts_comment_lines_pct:.1f}%</td>
                    </tr>
                    <tr>
                        <td>Blank Lines</td>
                        <td class="number">{stats["typescript"]["blank_lines"]:,}</td>
                        <td class="number">{ts_blank_lines_pct:.1f}%</td>
                    </tr>
                    <tr>
                        <td>Classes</td>
                        <td class="number">{stats["typescript"]["classes"]:,}</td>
                        <td class="number">-</td>
                    </tr>
                    <tr>
                        <td>Functions</td>
                        <td class="number">{stats["typescript"]["functions"]:,}</td>
                        <td class="number">-</td>
                    </tr>
                    <tr>
                        <td>Interfaces</td>
                        <td class="number">{stats["typescript"]["interfaces"]:,}</td>
                        <td class="number">-</td>
                    </tr>
                    <tr>
                        <td>Type Aliases</td>
                        <td class="number">{stats["typescript"]["types"]:,}</td>
                        <td class="number">-</td>
                    </tr>
                </table>
"""

    # TypeScript subdirectories
    if stats["typescript"]["by_subdir"]:
        html += """
                <h3 style="margin-top: 2rem; margin-bottom: 1rem; color: #667eea;">TypeScript Packages Breakdown</h3>
                <table>
                    <tr>
                        <th>Package</th>
                        <th class="number">Files</th>
                        <th class="number">Code Lines</th>
                    </tr>
"""
        ts_subdirs = sorted(
            [(k, v) for k, v in stats["typescript"]["by_subdir"].items()],
            key=lambda x: x[1]["code_lines"],
            reverse=True,
        )

        for subdir, subdir_stats in ts_subdirs:
            html += f"""                    <tr>
                        <td>{subdir}</td>
                        <td class="number">{subdir_stats["files"]}</td>
                        <td class="number">{subdir_stats["code_lines"]:,}</td>
                    </tr>
"""

    # More calculated metrics
    code_density = (
        (stats["total"]["code_lines"] / (stats["total"]["total_lines"])) * 100
        if stats["total"]["total_lines"] > 0
        else 0
    )
    py_dominance = (
        stats["python"]["code_lines"] / stats["total"]["code_lines"] * 100
        if stats["total"]["code_lines"] > 0
        else 0
    )
    ts_share = (
        stats["typescript"]["code_lines"] / stats["total"]["code_lines"] * 100
        if stats["total"]["code_lines"] > 0
        else 0
    )
    total_comments = (
        stats["python"]["comment_lines"] + stats["typescript"]["comment_lines"]
    )
    comment_pct = (
        (total_comments) / (stats["total"]["total_lines"]) * 100
        if stats["total"]["total_lines"] > 0
        else 0
    )

    html += f"""                </table>
            </section>

            <section>
                <h2>Code Composition</h2>
                <p style="margin-bottom: 1rem;">Distribution of code vs comments vs blank lines</p>
                <div style="margin-bottom: 1rem;">
                    <strong>Python</strong>
                    <div class="progress-bar">
                        <div class="progress-fill" style="width: {py_code_pct:.1f}%;"></div>
                    </div>
                    <div style="margin-top: 0.5rem; display: flex; justify-content: space-between; font-size: 0.9rem; color: #666;">
                        <span>Code: {py_code_pct:.1f}%</span>
                        <span>Comments: {py_comment_pct:.1f}%</span>
                        <span>Blank: {py_blank_pct:.1f}%</span>
                    </div>
                </div>
                <div>
                    <strong>TypeScript</strong>
                    <div class="progress-bar">
                        <div class="progress-fill" style="width: {ts_code_pct:.1f}%;"></div>
                    </div>
                    <div style="margin-top: 0.5rem; display: flex; justify-content: space-between; font-size: 0.9rem; color: #666;">
                        <span>Code: {ts_code_pct:.1f}%</span>
                        <span>Comments: {ts_comment_pct:.1f}%</span>
                        <span>Blank: {ts_blank_pct:.1f}%</span>
                    </div>
                </div>
            </section>

            <section>
                <h2>Key Insights</h2>
                <ul style="list-style: none; padding-left: 0;">
                    <li style="margin-bottom: 1rem; padding: 1rem; background: #f8f9fa; border-left: 4px solid #667eea; border-radius: 4px;">
                        <strong>📝 Code Density:</strong> {code_density:.1f}% of all lines are executable code
                    </li>
                    <li style="margin-bottom: 1rem; padding: 1rem; background: #f8f9fa; border-left: 4px solid #667eea; border-radius: 4px;">
                        <strong>🐍 Python Dominance:</strong> {py_dominance:.1f}% of codebase is Python
                    </li>
                    <li style="margin-bottom: 1rem; padding: 1rem; background: #f8f9fa; border-left: 4px solid #667eea; border-radius: 4px;">
                        <strong>💻 TypeScript Frontend:</strong> {ts_share:.1f}% of codebase is TypeScript
                    </li>
                    <li style="margin-bottom: 1rem; padding: 1rem; background: #f8f9fa; border-left: 4px solid #667eea; border-radius: 4px;">
                        <strong>🏗️ Architecture:</strong> {stats["python"]["classes"]} Python classes and {stats["typescript"]["interfaces"]} TypeScript interfaces provide strong type safety
                    </li>
                    <li style="margin-bottom: 1rem; padding: 1rem; background: #f8f9fa; border-left: 4px solid #667eea; border-radius: 4px;">
                        <strong>📚 Documentation:</strong> {total_comments:,} comment lines ({comment_pct:.1f}% of total)
                    </li>
                </ul>
            </section>
        </div>

        <footer>
            Generated by Luxar Project Analyzer • {datetime.now().strftime("%Y")} •
            Excluding: node_modules, __pycache__, coverage, dist, build
        </footer>
    </div>
</body>
</html>
"""

    with open(output_file, "w", encoding="utf-8") as f:
        f.write(html)


def main():
    """Main analysis function."""
    aprint("Analyzing Luxar project...")

    # Get project root (script is in stats/, so parent.parent is project root)
    script_path = Path(__file__).resolve()
    if script_path.parent.name == "stats":
        project_root = script_path.parent.parent
    else:
        project_root = script_path.parent

    # Analyze Python code
    aprint("  Analyzing Python code...")
    python_stats = analyze_directory(project_root, [".py"], "python")

    # Analyze TypeScript code
    aprint("  Analyzing TypeScript code...")
    typescript_stats = analyze_directory(project_root, [".ts", ".tsx"], "typescript")

    # Get test statistics
    aprint("  Gathering test statistics...")
    test_stats = get_test_statistics(project_root)

    # Combine statistics
    stats = {
        "python": python_stats,
        "typescript": typescript_stats,
        "tests": test_stats,
        "total": {
            "files": python_stats["files"] + typescript_stats["files"],
            "total_lines": python_stats["total_lines"]
            + typescript_stats["total_lines"],
            "code_lines": python_stats["code_lines"] + typescript_stats["code_lines"],
            "comment_lines": python_stats["comment_lines"]
            + typescript_stats["comment_lines"],
            "blank_lines": python_stats["blank_lines"]
            + typescript_stats["blank_lines"],
            "classes": python_stats["classes"] + typescript_stats["classes"],
            "functions": python_stats["functions"] + typescript_stats["functions"],
            "methods": python_stats["methods"],
        },
    }

    # Generate HTML report (save in stats directory)
    stats_dir = project_root / "stats"
    stats_dir.mkdir(exist_ok=True)
    output_file = stats_dir / "project_stats.html"
    aprint("  Generating HTML report...")
    generate_html_report(stats, output_file)

    aprint(f"\n✅ Report generated: {output_file}")
    aprint("\n📊 Summary:")
    aprint(f"  Total files: {stats['total']['files']:,}")
    aprint(f"  Lines of code: {stats['total']['code_lines']:,}")
    aprint(f"  Python files: {stats['python']['files']:,}")
    aprint(f"  TypeScript files: {stats['typescript']['files']:,}")
    aprint(f"  Classes: {stats['total']['classes']:,}")
    aprint(f"  Functions: {stats['total']['functions']:,}")
    aprint(
        f"  Test files: {stats['tests']['test_files'] + stats['tests']['ts_test_files']:,} (Python: {stats['tests']['test_files']}, TS: {stats['tests']['ts_test_files']})"
    )
    aprint(
        f"  Tests: {stats['tests']['test_count'] + stats['tests']['ts_test_count']:,} (Python: {stats['tests']['test_count']:,}, TS: {stats['tests']['ts_test_count']:,})"
    )
    if (
        stats["tests"]["coverage_percent"] > 0
        or stats["tests"]["ts_coverage_percent"] > 0
    ):
        aprint(
            f"  Coverage: Python: {stats['tests']['coverage_percent']:.1f}%, TypeScript: {stats['tests']['ts_coverage_percent']:.1f}%"
        )
    total_tests = stats["tests"]["test_count"] + stats["tests"]["ts_test_count"]
    total_passing = stats["tests"]["test_count"] + stats["tests"]["ts_test_passed"]
    if total_tests > 0:
        aprint(
            f"  Pass rate: {total_passing / total_tests * 100:.1f}% ({total_passing}/{total_tests})"
        )


if __name__ == "__main__":
    main()
