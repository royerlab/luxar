#!/usr/bin/env python3
"""Generate the Luxar project statistics report.

Scans the codebase, parses package manifests, optionally runs the test
suites with coverage, and renders an HTML report (and optional JSON).

Usage:
    python stats/generate_stats.py                 # full report with tests
    python stats/generate_stats.py --no-tests      # skip test runs (fast)
    python stats/generate_stats.py --no-coverage   # count tests, skip coverage
    python stats/generate_stats.py --json          # also write project_stats.json
"""

from __future__ import annotations

import argparse
import ast
import html
import json
import re
import subprocess
import tokenize
import tomllib
from collections import defaultdict
from collections.abc import Iterator
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Any

from arbol import aprint, asection

# ---------------------------------------------------------------------------
# Language configuration
# ---------------------------------------------------------------------------

LANGUAGE_CONFIG: dict[str, dict[str, Any]] = {
    "python": {
        "extensions": [".py"],
        "comment_single": "#",
        "comment_multi_start": None,  # handled by tokenize, not regex
        "comment_multi_end": None,
        "color": "#3572A5",
        "label": "Python",
    },
    "typescript": {
        "extensions": [".ts", ".tsx"],
        "comment_single": "//",
        "comment_multi_start": "/*",
        "comment_multi_end": "*/",
        "color": "#3178C6",
        "label": "TypeScript",
    },
    "rust": {
        "extensions": [".rs"],
        "comment_single": "//",
        "comment_multi_start": "/*",
        "comment_multi_end": "*/",
        "color": "#DEA584",
        "label": "Rust",
    },
    "cuda": {
        "extensions": [".cu", ".cuh"],
        "comment_single": "//",
        "comment_multi_start": "/*",
        "comment_multi_end": "*/",
        "color": "#76B900",
        "label": "CUDA",
    },
    "go": {
        "extensions": [".go"],
        "comment_single": "//",
        "comment_multi_start": "/*",
        "comment_multi_end": "*/",
        "color": "#00ADD8",
        "label": "Go",
    },
    "css": {
        "extensions": [".css", ".scss"],
        "comment_single": None,
        "comment_multi_start": "/*",
        "comment_multi_end": "*/",
        "color": "#563D7C",
        "label": "CSS",
    },
    "javascript": {
        "extensions": [".js", ".jsx", ".mjs"],
        "comment_single": "//",
        "comment_multi_start": "/*",
        "comment_multi_end": "*/",
        "color": "#F7DF1E",
        "label": "JavaScript",
    },
    "shell": {
        "extensions": [".sh", ".bash"],
        "comment_single": "#",
        "comment_multi_start": None,
        "comment_multi_end": None,
        "color": "#89E051",
        "label": "Shell",
    },
    "json": {
        "extensions": [".json"],
        "comment_single": None,
        "comment_multi_start": None,
        "comment_multi_end": None,
        "color": "#292929",
        "label": "JSON",
    },
    "toml": {
        "extensions": [".toml"],
        "comment_single": "#",
        "comment_multi_start": None,
        "comment_multi_end": None,
        "color": "#9C4121",
        "label": "TOML",
    },
    "yaml": {
        "extensions": [".yaml", ".yml"],
        "comment_single": "#",
        "comment_multi_start": None,
        "comment_multi_end": None,
        "color": "#CB171E",
        "label": "YAML",
    },
    "markdown": {
        "extensions": [".md"],
        "comment_single": None,
        "comment_multi_start": None,
        "comment_multi_end": None,
        "color": "#083FA1",
        "label": "Markdown",
    },
    "html": {
        "extensions": [".html", ".htm"],
        "comment_single": None,
        "comment_multi_start": "<!--",
        "comment_multi_end": "-->",
        "color": "#E34C26",
        "label": "HTML",
    },
    "makefile": {
        "extensions": ["Makefile"],
        "comment_single": "#",
        "comment_multi_start": None,
        "comment_multi_end": None,
        "color": "#427819",
        "label": "Makefile",
    },
}

# Directory names skipped anywhere in the path. Generated/build artifacts and
# vendored dependencies are excluded so the report reflects the human-written
# codebase rather than Sphinx output or Playwright traces.
SKIP_DIR_NAMES = {
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
    "_build",
    "playwright-report",
    ".playwright-mcp",
    "test-results",
    "delme",
    "datasets",
    "build-cuda-logs",
    "cuda-build-logs",
    ".idea",
    ".claude",
    ".vscode",
}

# Subpackages we want to call out in the HTML breakdown.
PYTHON_PACKAGE_ROOT = Path("packages/luxar/src/luxar")
TS_PACKAGE_ROOT = Path("packages/luxar-viewer/src")


# ---------------------------------------------------------------------------
# Data classes
# ---------------------------------------------------------------------------


@dataclass
class LineStats:
    total: int = 0
    code: int = 0
    comment: int = 0
    blank: int = 0


@dataclass
class LanguageStats:
    files: int = 0
    total_lines: int = 0
    code_lines: int = 0
    comment_lines: int = 0
    blank_lines: int = 0
    definitions: dict[str, int] = field(default_factory=lambda: defaultdict(int))
    by_subdir: dict[str, dict[str, int]] = field(
        default_factory=lambda: defaultdict(lambda: defaultdict(int))
    )
    largest_files: list[tuple[str, int]] = field(default_factory=list)

    def as_dict(self) -> dict[str, Any]:
        return {
            "files": self.files,
            "total_lines": self.total_lines,
            "code_lines": self.code_lines,
            "comment_lines": self.comment_lines,
            "blank_lines": self.blank_lines,
            "definitions": dict(self.definitions),
            "by_subdir": {k: dict(v) for k, v in self.by_subdir.items()},
            "largest_files": self.largest_files,
        }


# ---------------------------------------------------------------------------
# Line and definition counting
# ---------------------------------------------------------------------------


def _count_lines_generic(filepath: Path, lang_config: dict[str, Any]) -> LineStats:
    """Count lines for languages other than Python.

    Block-comment detection is line-based: any line that contains the start
    or end token (or sits between them) is treated as a comment line. This
    is approximate for languages where string literals can embed the same
    delimiter, but acceptable as a project-level metric.
    """
    try:
        with open(filepath, "r", encoding="utf-8", errors="ignore") as f:
            lines = f.readlines()
    except OSError:
        return LineStats()

    total = len(lines)
    blank = sum(1 for line in lines if line.strip() == "")

    comment_single = lang_config.get("comment_single")
    multi_start = lang_config.get("comment_multi_start")
    multi_end = lang_config.get("comment_multi_end")

    comment = 0
    in_block = False
    for raw in lines:
        stripped = raw.strip()
        if not stripped:
            continue
        if in_block:
            comment += 1
            if multi_end and multi_end in stripped:
                in_block = False
            continue
        if multi_start and multi_start in stripped:
            comment += 1
            # Single-line block comment closes on same line
            after_start = stripped.split(multi_start, 1)[1]
            if not (multi_end and multi_end in after_start):
                in_block = True
            continue
        if comment_single and stripped.startswith(comment_single):
            comment += 1

    code = max(0, total - blank - comment)
    return LineStats(total=total, code=code, comment=comment, blank=blank)


def _count_lines_python(filepath: Path) -> LineStats:
    """Accurate Python line accounting using the tokenize module.

    - blank: lines containing only whitespace
    - comment: lines that are either pure `#` comments or pure module/class/
      function docstrings (a string expression statement). Inline comments and
      string literals used as values are NOT counted as comments.
    - code: everything else (total - blank - comment).
    """
    try:
        with open(filepath, "rb") as f:
            data = f.read()
    except OSError:
        return LineStats()

    text = data.decode("utf-8", errors="ignore")
    lines = text.splitlines()
    total = len(lines)
    blank = sum(1 for line in lines if line.strip() == "")

    comment_lines: set[int] = set()

    # Pure `#` comment lines via tokenize.
    try:
        import io

        tokens = list(tokenize.tokenize(io.BytesIO(data).readline))
    except (tokenize.TokenError, SyntaxError, IndentationError):
        tokens = []

    for tok in tokens:
        if tok.type == tokenize.COMMENT:
            start_line = tok.start[0]
            line_text = lines[start_line - 1] if 0 < start_line <= total else ""
            if line_text.lstrip().startswith("#"):
                comment_lines.add(start_line)

    # Docstrings (module / class / function / async function) via AST.
    try:
        tree = ast.parse(data)
    except SyntaxError:
        tree = None

    if tree is not None:
        docstring_parents: list[ast.AST] = [tree]
        for node in ast.walk(tree):
            if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef)):
                docstring_parents.append(node)
        for parent in docstring_parents:
            body = getattr(parent, "body", None)
            if not body:
                continue
            first = body[0]
            if (
                isinstance(first, ast.Expr)
                and isinstance(first.value, ast.Constant)
                and isinstance(first.value.value, str)
            ):
                start = first.lineno
                end = getattr(first, "end_lineno", start)
                for ln in range(start, end + 1):
                    comment_lines.add(ln)

    comment = len(comment_lines)
    code = max(0, total - blank - comment)
    return LineStats(total=total, code=code, comment=comment, blank=blank)


def count_lines_in_file(filepath: Path, lang_name: str) -> LineStats:
    if lang_name == "python":
        return _count_lines_python(filepath)
    return _count_lines_generic(filepath, LANGUAGE_CONFIG[lang_name])


def count_python_definitions(filepath: Path) -> dict[str, int]:
    """AST-based Python definition counting.

    - classes: every ClassDef
    - methods: any function/async-function whose parent in the body chain is
      a ClassDef
    - functions: every other function/async-function (module-level and
      nested-in-function)
    """
    try:
        with open(filepath, "rb") as f:
            tree = ast.parse(f.read())
    except (OSError, SyntaxError):
        return {"classes": 0, "functions": 0, "methods": 0}

    classes = 0
    functions = 0
    methods = 0

    def walk(node: ast.AST, *, in_class: bool) -> None:
        nonlocal classes, functions, methods
        for child in ast.iter_child_nodes(node):
            if isinstance(child, ast.ClassDef):
                classes += 1
                walk(child, in_class=True)
            elif isinstance(child, (ast.FunctionDef, ast.AsyncFunctionDef)):
                if in_class:
                    methods += 1
                else:
                    functions += 1
                walk(child, in_class=False)
            else:
                walk(child, in_class=in_class)

    walk(tree, in_class=False)
    return {"classes": classes, "functions": functions, "methods": methods}


def count_typescript_definitions(filepath: Path) -> dict[str, int]:
    try:
        content = filepath.read_text(encoding="utf-8", errors="ignore")
    except OSError:
        return {"classes": 0, "functions": 0, "interfaces": 0, "types": 0}

    classes = len(re.findall(r"^\s*(?:export\s+)?class\s+\w+", content, re.MULTILINE))
    functions = len(
        re.findall(r"^\s*(?:export\s+)?function\s+\w+", content, re.MULTILINE)
    )
    functions += len(
        re.findall(
            r"^\s*(?:export\s+)?const\s+\w+(?:\s*:\s*[^=]+)?\s*=\s*"
            r"(?:async\s+)?(?:function\b|\([^)]*\)\s*=>|\w+\s*=>)",
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


def count_rust_definitions(filepath: Path) -> dict[str, int]:
    try:
        content = filepath.read_text(encoding="utf-8", errors="ignore")
    except OSError:
        return {"structs": 0, "functions": 0, "traits": 0, "impls": 0, "enums": 0}

    structs = len(
        re.findall(r"^\s*(?:pub(?:\([^)]+\))?\s+)?struct\s+\w+", content, re.MULTILINE)
    )
    functions = len(
        re.findall(
            r"^\s*(?:pub(?:\([^)]+\))?\s+)?(?:async\s+)?fn\s+\w+", content, re.MULTILINE
        )
    )
    traits = len(
        re.findall(r"^\s*(?:pub(?:\([^)]+\))?\s+)?trait\s+\w+", content, re.MULTILINE)
    )
    impls = len(re.findall(r"^\s*impl\b", content, re.MULTILINE))
    enums = len(
        re.findall(r"^\s*(?:pub(?:\([^)]+\))?\s+)?enum\s+\w+", content, re.MULTILINE)
    )
    return {
        "structs": structs,
        "functions": functions,
        "traits": traits,
        "impls": impls,
        "enums": enums,
    }


def count_cuda_definitions(filepath: Path) -> dict[str, int]:
    try:
        content = filepath.read_text(encoding="utf-8", errors="ignore")
    except OSError:
        return {"kernels": 0, "device_functions": 0, "host_functions": 0}

    kernels = len(re.findall(r"__global__\s+\w+\s+\w+\s*\(", content))
    device_functions = len(
        re.findall(r"__device__\s+(?!__host__)[\w\s*&:<>,]+?\s+\w+\s*\(", content)
    )
    host_functions = len(re.findall(r"__host__\s+[\w\s*&:<>,]+?\s+\w+\s*\(", content))
    return {
        "kernels": kernels,
        "device_functions": device_functions,
        "host_functions": host_functions,
    }


def count_go_definitions(filepath: Path) -> dict[str, int]:
    try:
        content = filepath.read_text(encoding="utf-8", errors="ignore")
    except OSError:
        return {"functions": 0, "structs": 0, "interfaces": 0}

    # `func Name(...)` or `func (recv T) Name(...)`
    functions = len(
        re.findall(r"^func\s+(?:\([^)]+\)\s+)?\w+\s*\(", content, re.MULTILINE)
    )
    structs = len(re.findall(r"^type\s+\w+\s+struct\b", content, re.MULTILINE))
    interfaces = len(re.findall(r"^type\s+\w+\s+interface\b", content, re.MULTILINE))
    return {"functions": functions, "structs": structs, "interfaces": interfaces}


def count_css_definitions(filepath: Path) -> dict[str, int]:
    try:
        content = filepath.read_text(encoding="utf-8", errors="ignore")
    except OSError:
        return {"rules": 0, "variables": 0, "media_queries": 0}

    # Approximate: count `{...}` blocks. Nested blocks (SCSS) under-count but
    # are reasonable as a structural signal.
    rules = len(re.findall(r"\{[^}]*\}", content))
    variables = len(re.findall(r"--[\w-]+\s*:", content))
    media_queries = len(re.findall(r"@media\b", content))
    return {"rules": rules, "variables": variables, "media_queries": media_queries}


# ---------------------------------------------------------------------------
# File scanning
# ---------------------------------------------------------------------------


def _is_skipped(path: Path) -> bool:
    return any(part in SKIP_DIR_NAMES or part.endswith(".zarr") for part in path.parts)


def _iter_files_for_extension(root: Path, ext: str) -> Iterator[Path]:
    """Yield files matching ``ext`` under ``root``, skipping ignored dirs."""
    if ext.startswith("."):
        pattern = f"*{ext}"
    else:
        pattern = ext  # exact filename (e.g. "Makefile")
    for filepath in root.rglob(pattern):
        if _is_skipped(filepath):
            continue
        yield filepath


def analyze_language(root: Path, language_name: str) -> LanguageStats:
    lang_config = LANGUAGE_CONFIG[language_name]
    stats = LanguageStats()
    file_sizes: list[tuple[str, int]] = []

    for ext in lang_config["extensions"]:
        for filepath in _iter_files_for_extension(root, ext):
            stats.files += 1

            line_stats = count_lines_in_file(filepath, language_name)
            stats.total_lines += line_stats.total
            stats.code_lines += line_stats.code
            stats.comment_lines += line_stats.comment
            stats.blank_lines += line_stats.blank

            try:
                rel = filepath.relative_to(root)
            except ValueError:
                rel = filepath
            file_sizes.append((str(rel), line_stats.code))
            subdir = rel.parts[0] if len(rel.parts) > 1 else "root"
            stats.by_subdir[subdir]["files"] += 1
            stats.by_subdir[subdir]["code_lines"] += line_stats.code

            if language_name == "python":
                defs = count_python_definitions(filepath)
            elif language_name == "typescript":
                defs = count_typescript_definitions(filepath)
            elif language_name == "rust":
                defs = count_rust_definitions(filepath)
            elif language_name == "cuda":
                defs = count_cuda_definitions(filepath)
            elif language_name == "go":
                defs = count_go_definitions(filepath)
            elif language_name == "css":
                defs = count_css_definitions(filepath)
            else:
                defs = {}
            for k, v in defs.items():
                stats.definitions[k] += v

    file_sizes.sort(key=lambda x: x[1], reverse=True)
    stats.largest_files = file_sizes[:10]
    return stats


def analyze_package_breakdown(root: Path, pkg_root: Path) -> list[dict[str, Any]]:
    """Return per-subdirectory stats for files directly under ``pkg_root``.

    Aggregates by immediate child directory and counts Python (for the luxar
    package) or TypeScript files. Used to render a per-subpackage breakdown
    in the HTML.
    """
    abs_root = root / pkg_root
    if not abs_root.exists():
        return []

    # Determine which extensions to track based on package root.
    is_python_root = "luxar/src" in str(pkg_root).replace("\\", "/")
    if is_python_root:
        exts = LANGUAGE_CONFIG["python"]["extensions"]
        lang_name = "python"
    else:
        exts = LANGUAGE_CONFIG["typescript"]["extensions"]
        lang_name = "typescript"

    entries: dict[str, dict[str, int]] = defaultdict(
        lambda: {"files": 0, "code_lines": 0, "comment_lines": 0}
    )

    for ext in exts:
        for filepath in abs_root.rglob(f"*{ext}"):
            if _is_skipped(filepath):
                continue
            try:
                rel = filepath.relative_to(abs_root)
            except ValueError:
                continue
            subpkg = rel.parts[0] if len(rel.parts) > 1 else "<root>"
            line_stats = count_lines_in_file(filepath, lang_name)
            entries[subpkg]["files"] += 1
            entries[subpkg]["code_lines"] += line_stats.code
            entries[subpkg]["comment_lines"] += line_stats.comment

    out = [
        {"name": name, **vals}
        for name, vals in sorted(entries.items(), key=lambda x: -x[1]["code_lines"])
    ]
    return out


# ---------------------------------------------------------------------------
# Tests / coverage
# ---------------------------------------------------------------------------


def _empty_test_section() -> dict[str, Any]:
    return {
        "test_files": 0,
        "test_count": 0,
        "test_passed": 0,
        "test_failed": 0,
        "coverage_percent": 0.0,
        # Structured failure signal for a REQUESTED-but-incomplete measurement.
        # `incomplete` is a human-readable reason (or None when complete);
        # `incomplete_kind` is one of "timeout" / "tool_missing" / "coverage" /
        # "run_failed" so `validate_measurements` can gate coverage-only
        # failures on `run_coverage`. See `_mark_incomplete`.
        "incomplete": None,
        "incomplete_kind": None,
    }


def _mark_incomplete(section: dict[str, Any], kind: str, reason: str) -> None:
    """Record a failure signal, preserving the root cause.

    The first signal wins: a later coverage check must never overwrite an
    earlier timeout / tool-missing reason (one reason per section).
    """
    if section.get("incomplete") is None:
        section["incomplete"] = reason
        section["incomplete_kind"] = kind


def collect_test_file_counts(root: Path) -> dict[str, Any]:
    """File-level inventory (no test execution)."""
    py_tests = [
        p
        for p in list(root.rglob("test_*.py")) + list(root.rglob("*_test.py"))
        if not _is_skipped(p)
    ]
    viewer = root / "packages" / "luxar-viewer"
    ts_tests = [
        p
        for p in list(viewer.rglob("*.test.ts")) + list(viewer.rglob("*.test.tsx"))
        if not _is_skipped(p)
    ]
    e2e_tests = (
        [p for p in viewer.rglob("*.spec.ts") if not _is_skipped(p)]
        if viewer.exists()
        else []
    )
    return {
        "python_count": len(py_tests),
        "ts_count": len(ts_tests),
        "e2e_count": len(e2e_tests),
    }


def get_test_statistics(
    root: Path, *, run_tests: bool, run_coverage: bool
) -> dict[str, Any]:
    """Gather test inventory and optionally execute tests with coverage."""
    test_stats: dict[str, Any] = {
        "python": _empty_test_section(),
        "typescript": _empty_test_section(),
        "rust": {
            "test_files": 0,
            "test_count": 0,
            "test_passed": 0,
            "test_failed": 0,
            "incomplete": None,
            "incomplete_kind": None,
        },
        "e2e": {"test_files": 0, "test_count": 0},
        "error": None,
    }

    inventory = collect_test_file_counts(root)
    test_stats["python"]["test_files"] = inventory["python_count"]
    test_stats["typescript"]["test_files"] = inventory["ts_count"]
    test_stats["e2e"]["test_files"] = inventory["e2e_count"]

    # Rust test-file heuristic: any .rs file containing a #[test] or #[cfg(test)].
    rust_path = root / "packages" / "luxar-viewer" / "src" / "wasm" / "rust"
    if rust_path.exists():
        rust_test_files = 0
        for rs_file in rust_path.rglob("*.rs"):
            if _is_skipped(rs_file):
                continue
            try:
                txt = rs_file.read_text(errors="ignore")
            except OSError:
                continue
            if "#[test]" in txt or "#[cfg(test)]" in txt:
                rust_test_files += 1
        test_stats["rust"]["test_files"] = rust_test_files

    if not run_tests:
        aprint("Skipping test execution (--no-tests)")
        return test_stats

    _run_python_tests(root, test_stats, run_coverage=run_coverage)
    _run_typescript_tests(root, test_stats, run_coverage=run_coverage)
    _run_rust_tests(root, test_stats)
    return test_stats


def _run_python_tests(
    root: Path, test_stats: dict[str, Any], *, run_coverage: bool
) -> None:
    with asection("Python tests"):
        # Collect count
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
                timeout=120,
                cwd=root,
            )
            for line in result.stdout.split("\n"):
                if "selected" in line or "collected" in line:
                    nums = re.findall(r"\d+", line)
                    if nums:
                        test_stats["python"]["test_count"] = int(nums[0])
                        aprint(f"Collected {nums[0]} tests")
                        break
        except FileNotFoundError:
            aprint("hatch not found; skipping Python test collection")
            _mark_incomplete(test_stats["python"], "tool_missing", "hatch not found")
            return
        except subprocess.TimeoutExpired:
            aprint("Python test collection timed out")
            _mark_incomplete(
                test_stats["python"], "timeout", "test collection timed out"
            )
            return
        except Exception as e:
            aprint(f"Python test collection failed: {e}")

        cmd = ["hatch", "run", "pytest", "packages/luxar/src/luxar", "-q", "--tb=no"]
        if run_coverage:
            cmd[3:3] = ["--cov=packages/luxar/src/luxar", "--cov-report=term"]
        aprint(f"Running: {' '.join(cmd)}")
        # 90 min budget: the full Python suite (5.8K+ tests with coverage) is
        # the long pole of the report. The previous 30 min budget was exceeded
        # once the suite passed ~5K tests, and a timeout here is silent —
        # coverage stays at its 0.0% default and the weighted total silently
        # halves — so the budget is generous on purpose. Bump it again rather
        # than let the report publish a zero.
        try:
            result = subprocess.run(
                cmd,
                capture_output=True,
                text=True,
                timeout=5400,
                cwd=root,
            )
        except subprocess.TimeoutExpired:
            aprint("Python test run timed out after 90 minutes")
            test_stats["python"]["timed_out"] = True
            _mark_incomplete(
                test_stats["python"], "timeout", "test run timed out after 90 minutes"
            )
            return
        except FileNotFoundError:
            aprint("hatch not found; skipping Python test run")
            _mark_incomplete(test_stats["python"], "tool_missing", "hatch not found")
            return

        coverage_measured = False
        for line in (result.stdout + "\n" + result.stderr).split("\n"):
            if line.strip().startswith("TOTAL") and "%" in line:
                m = re.search(r"(\d+(?:\.\d+)?)%", line)
                if m:
                    test_stats["python"]["coverage_percent"] = float(m.group(1))
                    coverage_measured = True
                    aprint(f"Coverage: {m.group(1)}%")
                    break

        for line in result.stdout.split("\n"):
            m = re.search(r"(\d+)\s+passed", line)
            if m:
                test_stats["python"]["test_passed"] = int(m.group(1))
            m = re.search(r"(\d+)\s+failed", line)
            if m:
                test_stats["python"]["test_failed"] = int(m.group(1))

        # pytest: 0 = all passed, 1 = tests ran but some failed — both are COMPLETE
        # measurements. Any other code (2 interrupted, 3 internal error, 4 usage,
        # 5 no tests collected) means the run produced no trustworthy numbers.
        if result.returncode not in (0, 1):
            _mark_incomplete(
                test_stats["python"],
                "run_failed",
                f"test run exited with code {result.returncode} (no measurement produced)",
            )
        elif (
            test_stats["python"]["test_passed"] == 0
            and test_stats["python"]["test_failed"] == 0
        ):
            # Exit 0/1 with no parseable pass/fail counts. `hatch run` reports
            # its OWN failures (missing env, dependency resolution) as exit 1,
            # which is indistinguishable from "tests ran and some failed" by
            # exit code alone — so require that the run actually produced a
            # summary before trusting its numbers.
            _mark_incomplete(
                test_stats["python"],
                "run_failed",
                f"test run exited with code {result.returncode} but produced no "
                "parseable test results",
            )

        # Coverage was requested but the run reported none. Key on whether a
        # TOTAL line was actually parsed, not on the 0.0 default: a genuine 0%
        # is a valid measurement and must not be rejected forever.
        if run_coverage and not coverage_measured:
            _mark_incomplete(
                test_stats["python"],
                "coverage",
                "coverage requested but none was produced",
            )


def _run_typescript_tests(
    root: Path, test_stats: dict[str, Any], *, run_coverage: bool
) -> None:
    viewer = root / "packages" / "luxar-viewer"
    if not viewer.exists():
        return
    with asection("TypeScript tests"):
        cmd = [
            "npx",
            "vitest",
            "--run",
        ]
        # mtime of a pre-existing coverage summary we could NOT delete; used
        # below to tell "vitest rewrote it" from "the old file is still there".
        stale_cov_mtime: float | None = None
        if run_coverage:
            cmd.insert(3, "--coverage")
            # Delete any prior coverage summary first: a crashed run leaves the
            # OLD file in place, which would then launder as fresh coverage and
            # pass the freshness check below. Removing it means "no fresh
            # coverage" is detectable as an absent file.
            stale_summary = viewer / "coverage" / "coverage-summary.json"
            try:
                stale_summary.unlink(missing_ok=True)
            except OSError as e:
                aprint(f"Could not remove stale coverage summary: {e}")
                try:
                    stale_cov_mtime = stale_summary.stat().st_mtime
                except OSError:
                    # Can't delete it and can't stat it — the read below will
                    # fail too, which the no-fresh-coverage guard catches.
                    stale_cov_mtime = None
        aprint(f"Running: {' '.join(cmd)}")
        try:
            result = subprocess.run(
                cmd,
                capture_output=True,
                text=True,
                timeout=300,
                cwd=viewer,
            )
        except FileNotFoundError:
            aprint("npx not found; skipping TypeScript test run")
            _mark_incomplete(test_stats["typescript"], "tool_missing", "npx not found")
            return
        except subprocess.TimeoutExpired:
            aprint("TypeScript test run timed out")
            _mark_incomplete(test_stats["typescript"], "timeout", "test run timed out")
            return

        # vitest's summary line is `Tests  2 failed | 148 passed (150)`; an
        # all-failing run omits the "passed" half, so match on either side.
        summary_seen = False
        for line in (result.stdout + "\n" + result.stderr).split("\n"):
            if line.strip().startswith("Tests") and (
                "passed" in line or "failed" in line
            ):
                summary_seen = True
                total_m = re.search(r"\((\d+)\)", line)
                if total_m:
                    test_stats["typescript"]["test_count"] = int(total_m.group(1))
                pm = re.search(r"(\d+)\s+passed", line)
                if pm:
                    test_stats["typescript"]["test_passed"] = int(pm.group(1))
                fm = re.search(r"(\d+)\s+failed", line)
                if fm:
                    test_stats["typescript"]["test_failed"] = int(fm.group(1))
                break

        # vitest: 0 = pass, 1 = test failures (both complete). Other nonzero = a
        # crash/config error that produced no fresh numbers — guard it here, before
        # trusting coverage-summary.json.
        if result.returncode not in (0, 1):
            _mark_incomplete(
                test_stats["typescript"],
                "run_failed",
                f"test run exited with code {result.returncode} (no measurement produced)",
            )
        elif not summary_seen:
            # vitest also exits 1 for startup, config and collection errors, so
            # exit 1 alone does not prove tests ran. Require the summary line:
            # without it there is no measurement, only zeros.
            _mark_incomplete(
                test_stats["typescript"],
                "run_failed",
                f"test run exited with code {result.returncode} but produced no "
                "parseable test results",
            )

        if run_coverage:
            cov_json = viewer / "coverage" / "coverage-summary.json"
            coverage_measured = False
            if cov_json.exists():
                try:
                    if (
                        stale_cov_mtime is not None
                        and cov_json.stat().st_mtime <= stale_cov_mtime
                    ):
                        # The pre-run delete failed and vitest never rewrote the
                        # file: this is last run's number, not this run's.
                        aprint("Coverage summary is stale (not rewritten by this run)")
                    else:
                        cov = json.loads(cov_json.read_text())
                        pct = cov.get("total", {}).get("statements", {}).get("pct")
                        if pct is not None:
                            test_stats["typescript"]["coverage_percent"] = float(pct)
                            coverage_measured = True
                            aprint(f"Coverage (JSON): {pct}%")
                except (OSError, TypeError, ValueError) as e:
                    aprint(f"Could not read coverage JSON: {e}")

            # No FRESH coverage was read — flag it so the weighted headline
            # can't silently drop TypeScript. Keyed on whether a value was
            # actually parsed, not on the 0.0 default: a genuine 0% is a valid
            # measurement.
            if not coverage_measured:
                _mark_incomplete(
                    test_stats["typescript"],
                    "coverage",
                    "coverage requested but none was produced",
                )


def _run_rust_tests(root: Path, test_stats: dict[str, Any]) -> None:
    rust_path = root / "packages" / "luxar-viewer" / "src" / "wasm" / "rust"
    if not rust_path.exists():
        return
    with asection("Rust tests"):
        try:
            result = subprocess.run(
                ["cargo", "test", "--", "--test-threads=1"],
                capture_output=True,
                text=True,
                timeout=180,
                cwd=rust_path,
            )
        except FileNotFoundError:
            aprint("cargo not found; skipping Rust tests")
            _mark_incomplete(test_stats["rust"], "tool_missing", "cargo not found")
            return
        except subprocess.TimeoutExpired:
            aprint("Rust test run timed out")
            _mark_incomplete(test_stats["rust"], "timeout", "test run timed out")
            return

        passed = failed = 0
        for line in result.stdout.split("\n"):
            if "test result:" not in line:
                continue
            pm = re.search(r"(\d+)\s+passed", line)
            fm = re.search(r"(\d+)\s+failed", line)
            if pm:
                passed += int(pm.group(1))
            if fm:
                failed += int(fm.group(1))
        test_stats["rust"]["test_passed"] = passed
        test_stats["rust"]["test_failed"] = failed
        test_stats["rust"]["test_count"] = passed + failed
        aprint(f"Passed: {passed}, Failed: {failed}")

        # cargo test returns 101 for BOTH test failures and compile errors, so the
        # exit code alone can't tell them apart. A GENUINE failing run always
        # parses `failed > 0`, so a nonzero exit with ZERO parsed failures
        # unambiguously means the run crashed/failed to measure (compile error,
        # segfault, or a later test binary aborting before printing its summary)
        # rather than a real run whose tests merely failed. `passed + failed == 0`
        # would miss the partial-crash case where an earlier binary already
        # reported passes; keying on `failed == 0` catches it.
        if result.returncode != 0 and failed == 0:
            _mark_incomplete(
                test_stats["rust"],
                "run_failed",
                f"test run exited with code {result.returncode} and produced no results",
            )


def validate_measurements(
    test_stats: dict[str, Any], *, run_tests: bool, run_coverage: bool
) -> list[str]:
    """List every REQUESTED-but-incomplete test/coverage measurement.

    A partial number published as if it were complete is worse than no update:
    a timed-out Python coverage run turns a genuine 84% into a headline 47%
    once `weighted_coverage` drops the language that came back 0.0. This gate
    lets `main` refuse to overwrite the reports when a requested measurement
    did not actually complete.

    - `run_tests=False` (`--no-tests`): nothing was requested, so nothing can
      be incomplete — returns [].
    - `run_tests=True`: reports any section whose test run timed out or whose
      toolchain was missing.
    - Coverage failures (a coverage-bearing language produced no fresh
      coverage) count only when BOTH `run_tests` and `run_coverage` are True;
      under `--no-coverage` a 0.0 coverage is legitimate.
    """
    if not run_tests:
        return []

    problems: list[str] = []
    for lang in ("python", "typescript", "rust"):
        section = test_stats.get(lang, {})
        reason = section.get("incomplete")
        if reason is None:
            continue
        # A coverage-only failure is legitimate under --no-coverage.
        if section.get("incomplete_kind") == "coverage" and not run_coverage:
            continue
        problems.append(f"{lang}: {reason}")
    return problems


# ---------------------------------------------------------------------------
# Git
# ---------------------------------------------------------------------------


def _git(root: Path, *args: str, timeout: int = 30) -> str | None:
    try:
        result = subprocess.run(
            ["git", *args],
            capture_output=True,
            text=True,
            timeout=timeout,
            cwd=root,
        )
    except (FileNotFoundError, subprocess.TimeoutExpired):
        return None
    if result.returncode != 0:
        return None
    return result.stdout


def get_git_statistics(root: Path) -> dict[str, Any]:
    git_stats: dict[str, Any] = {
        "total_commits": 0,
        "contributors": 0,
        "first_commit_date": None,
        "last_commit_date": None,
        "tags": 0,
        "commits_last_30_days": 0,
        "files_changed_last_30_days": 0,
        "top_contributors": [],
    }

    out = _git(root, "rev-list", "--count", "HEAD")
    if out:
        git_stats["total_commits"] = int(out.strip())

    # Scope shortlog to HEAD (not --all) so contributor counts share the exact
    # same commit set as total_commits above. Using --all here counted commits
    # on every local ref (including unmerged branches), which let a top
    # contributor's count exceed total_commits (see issue #764).
    out = _git(root, "shortlog", "-sn", "HEAD")
    if out:
        contributors = [line for line in out.strip().split("\n") if line.strip()]
        git_stats["contributors"] = len(contributors)
        for line in contributors[:5]:
            m = re.match(r"\s*(\d+)\s+(.+)", line)
            if m:
                git_stats["top_contributors"].append(
                    {"commits": int(m.group(1)), "name": m.group(2).strip()}
                )

    out = _git(root, "log", "--format=%ai", "--reverse")
    if out and out.strip():
        dates = out.strip().split("\n")
        git_stats["first_commit_date"] = dates[0].split()[0]
        git_stats["last_commit_date"] = dates[-1].split()[0]

    out = _git(root, "tag")
    if out:
        git_stats["tags"] = len([t for t in out.strip().split("\n") if t.strip()])

    out = _git(root, "rev-list", "--count", "--since=30 days ago", "HEAD")
    if out:
        try:
            git_stats["commits_last_30_days"] = int(out.strip())
        except ValueError:
            pass

    out = _git(root, "log", "--since=30 days ago", "--name-only", "--pretty=format:")
    if out:
        files = {line.strip() for line in out.split("\n") if line.strip()}
        git_stats["files_changed_last_30_days"] = len(files)

    return git_stats


# ---------------------------------------------------------------------------
# Dependencies
# ---------------------------------------------------------------------------


def get_dependency_statistics(root: Path) -> dict[str, Any]:
    deps: dict[str, Any] = {
        "python": {"production": 0, "dev": 0, "groups": {}, "packages": []},
        "node": {"production": 0, "dev": 0, "packages": []},
        "rust": {"production": 0, "dev": 0, "packages": []},
    }

    # Python via pyproject.toml
    pyproject = root / "pyproject.toml"
    if pyproject.exists():
        try:
            data = tomllib.loads(pyproject.read_text())
            project = data.get("project", {})
            prod = project.get("dependencies", []) or []
            deps["python"]["production"] = len(prod)
            deps["python"]["packages"] = [
                re.split(r"[<>=!~\[ ]", spec, maxsplit=1)[0] for spec in prod
            ]
            optional = project.get("optional-dependencies", {}) or {}
            for grp, items in optional.items():
                deps["python"]["groups"][grp] = len(items)
                deps["python"]["dev"] += len(items)
        except (OSError, tomllib.TOMLDecodeError) as e:
            aprint(f"Could not parse pyproject.toml: {e}")

    # Node via packages/luxar-viewer/package.json
    pkg_json = root / "packages" / "luxar-viewer" / "package.json"
    if pkg_json.exists():
        try:
            pkg = json.loads(pkg_json.read_text())
            deps["node"]["production"] = len(pkg.get("dependencies", {}))
            deps["node"]["dev"] = len(pkg.get("devDependencies", {}))
            deps["node"]["packages"] = list(pkg.get("dependencies", {}).keys())
        except (OSError, json.JSONDecodeError) as e:
            aprint(f"Could not parse package.json: {e}")

    # Rust via Cargo.toml
    cargo = root / "packages" / "luxar-viewer" / "src" / "wasm" / "rust" / "Cargo.toml"
    if cargo.exists():
        try:
            data = tomllib.loads(cargo.read_text())
            deps["rust"]["production"] = len(data.get("dependencies", {}) or {})
            deps["rust"]["dev"] = len(data.get("dev-dependencies", {}) or {})
            deps["rust"]["packages"] = list(data.get("dependencies", {}).keys())
        except (OSError, tomllib.TOMLDecodeError) as e:
            aprint(f"Could not parse Cargo.toml: {e}")

    return deps


# ---------------------------------------------------------------------------
# Extras: CI workflows, CHANGELOG, total project size
# ---------------------------------------------------------------------------


def get_extras(root: Path) -> dict[str, Any]:
    extras: dict[str, Any] = {
        "ci_workflows": 0,
        "changelog_versions": 0,
        "project_size_bytes": 0,
    }

    wf_dir = root / ".github" / "workflows"
    if wf_dir.exists():
        extras["ci_workflows"] = sum(
            1 for p in wf_dir.iterdir() if p.suffix in {".yml", ".yaml"}
        )

    changelog = root / "CHANGELOG.md"
    if changelog.exists():
        try:
            text = changelog.read_text(encoding="utf-8", errors="ignore")
            extras["changelog_versions"] = len(
                re.findall(r"^##\s+", text, re.MULTILINE)
            )
        except OSError:
            pass

    # Project size (text + binaries under tracked, non-skipped dirs).
    total = 0
    for p in root.rglob("*"):
        if not p.is_file():
            continue
        if _is_skipped(p):
            continue
        try:
            total += p.stat().st_size
        except OSError:
            pass
    extras["project_size_bytes"] = total

    return extras


# ---------------------------------------------------------------------------
# HTML rendering
# ---------------------------------------------------------------------------


def _fmt_size(bytes_count: int) -> str:
    if bytes_count >= 1024**3:
        return f"{bytes_count / 1024**3:.2f} GB"
    if bytes_count >= 1024**2:
        return f"{bytes_count / 1024**2:.1f} MB"
    if bytes_count >= 1024:
        return f"{bytes_count / 1024:.1f} KB"
    return f"{bytes_count} B"


def weighted_coverage(py_cov: float, py_loc: int, ts_cov: float, ts_loc: int) -> float:
    """LOC-weighted coverage across Python and TypeScript.

    A language whose coverage is 0.0 is treated as UNMEASURED and left out of
    the average, rather than averaged in as a real zero. Neither suite can
    plausibly sit at a true 0%, so a zero here means its run failed or timed
    out — and folding that in silently halves the headline number (a timed-out
    Python run once turned a genuine 84% into a published 47%).
    """
    parts = [(cov, loc) for cov, loc in ((py_cov, py_loc), (ts_cov, ts_loc)) if cov > 0]
    if not parts:
        return 0.0
    return sum(cov * loc for cov, loc in parts) / max(1, sum(loc for _, loc in parts))


def generate_html_report(stats: dict[str, Any], output_file: Path) -> None:
    now = datetime.now()
    langs = stats["languages"]
    primary_langs = [
        "python",
        "typescript",
        "rust",
        "cuda",
        "go",
        "css",
        "javascript",
        "shell",
    ]
    config_langs = ["json", "toml", "yaml", "makefile"]
    doc_langs = ["markdown", "html"]

    total_files = sum(langs[lang]["files"] for lang in langs)
    total_code_lines = sum(langs[lang]["code_lines"] for lang in langs)
    total_lines = sum(langs[lang]["total_lines"] for lang in langs)
    total_comment_lines = sum(langs[lang]["comment_lines"] for lang in langs)
    active_langs = sum(1 for lang in langs if langs[lang]["files"] > 0)

    tests = stats["tests"]
    total_test_files = (
        tests["python"]["test_files"]
        + tests["typescript"]["test_files"]
        + tests["e2e"]["test_files"]
        + tests["rust"]["test_files"]
    )
    total_tests = (
        tests["python"]["test_count"]
        + tests["typescript"]["test_count"]
        + tests["rust"]["test_count"]
    )
    total_passed = (
        tests["python"]["test_passed"]
        + tests["typescript"]["test_passed"]
        + tests["rust"]["test_passed"]
    )
    total_failed = (
        tests["python"]["test_failed"]
        + tests["typescript"]["test_failed"]
        + tests["rust"]["test_failed"]
    )

    py = langs["python"]
    ts = langs["typescript"]
    rust = langs["rust"]
    cuda = langs["cuda"]
    go = langs["go"]
    css = langs["css"]
    md = langs["markdown"]
    html_lang = langs["html"]
    json_stats = langs["json"]
    yaml_stats = langs["yaml"]
    toml_stats = langs["toml"]

    py_cov = tests["python"]["coverage_percent"]
    ts_cov = tests["typescript"]["coverage_percent"]
    weighted_cov = weighted_coverage(py_cov, py["code_lines"], ts_cov, ts["code_lines"])

    git = stats["git"]
    extras = stats["extras"]
    deps = stats["dependencies"]

    code_density = (total_code_lines / total_lines * 100) if total_lines else 0
    comment_density = (total_comment_lines / total_lines * 100) if total_lines else 0
    primary_code_lines = sum(
        langs[lang]["code_lines"]
        for lang in primary_langs
        if langs[lang]["code_lines"] > 0
    )

    def pct(n: int) -> float:
        return (n / primary_code_lines * 100) if primary_code_lines else 0

    # Build the HTML in pieces to keep f-strings tractable.
    out: list[str] = []
    out.append(f"""<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="UTF-8">
<meta name="viewport" content="width=device-width, initial-scale=1.0">
<title>Luxar Project Statistics</title>
<style>
* {{ margin: 0; padding: 0; box-sizing: border-box; }}
body {{
    font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, sans-serif;
    line-height: 1.6;
    color: #222;
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
header h1 {{ font-size: 2.4rem; margin-bottom: 0.4rem; font-weight: 700; }}
header p {{ font-size: 1rem; opacity: 0.92; }}
.content {{ padding: 2rem; }}
.summary {{
    display: grid;
    grid-template-columns: repeat(auto-fit, minmax(180px, 1fr));
    gap: 1.25rem;
    margin-bottom: 2rem;
}}
.stat-card {{
    background: linear-gradient(135deg, #f5f7fa 0%, #c3cfe2 100%);
    padding: 1.1rem;
    border-radius: 8px;
    box-shadow: 0 4px 6px rgba(0,0,0,0.08);
    transition: transform 0.15s;
}}
.stat-card:hover {{ transform: translateY(-3px); box-shadow: 0 8px 12px rgba(0,0,0,0.13); }}
.stat-card h3 {{
    font-size: 0.78rem;
    text-transform: uppercase;
    letter-spacing: 0.05em;
    color: #5a5fb0;
    margin-bottom: 0.4rem;
    font-weight: 600;
}}
.stat-card .value {{ font-size: 1.9rem; font-weight: 700; color: #222; }}
.stat-card .label {{ font-size: 0.78rem; color: #666; margin-top: 0.2rem; }}
section {{ margin-bottom: 2.5rem; }}
section h2 {{
    font-size: 1.5rem;
    color: #5a5fb0;
    margin-bottom: 0.85rem;
    padding-bottom: 0.45rem;
    border-bottom: 3px solid #5a5fb0;
}}
section h3 {{
    font-size: 1.1rem;
    color: #444;
    margin: 1.35rem 0 0.85rem 0;
}}
table {{ width: 100%; border-collapse: collapse; margin-top: 0.85rem; font-size: 0.9rem; }}
th {{
    background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
    color: white;
    padding: 0.6rem 0.75rem;
    text-align: left;
    font-weight: 600;
    text-transform: uppercase;
    font-size: 0.72rem;
    letter-spacing: 0.05em;
}}
td {{ padding: 0.55rem 0.75rem; border-bottom: 1px solid #e7e7e7; }}
tr:hover {{ background: #fafbff; }}
.number {{ text-align: right; font-family: 'SF Mono', Menlo, 'Monaco', monospace; font-weight: 500; }}
.lang-badge {{
    display: inline-block;
    padding: 0.18rem 0.5rem;
    border-radius: 4px;
    font-size: 0.72rem;
    font-weight: 600;
    color: white;
}}
.progress-bar {{ height: 16px; background: #e0e0e0; border-radius: 8px; overflow: hidden; margin-top: 0.5rem; }}
.progress-fill {{ height: 100%; transition: width 0.3s ease; }}
.lang-breakdown {{ display: grid; grid-template-columns: repeat(auto-fit, minmax(280px, 1fr)); gap: 1.25rem; }}
.lang-card {{ background: #f8f9fa; border-radius: 8px; padding: 1.1rem; border-left: 4px solid; }}
.lang-card h4 {{ font-size: 1rem; margin-bottom: 0.65rem; display: flex; align-items: center; gap: 0.5rem; }}
.lang-stats {{ font-size: 0.85rem; color: #555; }}
.lang-stats div {{ margin-bottom: 0.2rem; }}
.two-column {{ display: grid; grid-template-columns: repeat(auto-fit, minmax(420px, 1fr)); gap: 1.75rem; }}
.insight {{ margin-bottom: 0.75rem; padding: 0.85rem 1rem; background: #f5f6fb; border-left: 4px solid #5a5fb0; border-radius: 4px; font-size: 0.92rem; }}
.insight strong {{ color: #5a5fb0; }}
.health-indicator {{ display: inline-block; padding: 0.15rem 0.4rem; border-radius: 3px; font-size: 0.72rem; font-weight: 500; }}
.health-good {{ background: #d4edda; color: #155724; }}
.health-warning {{ background: #fff3cd; color: #856404; }}
.health-danger {{ background: #f8d7da; color: #721c24; }}
footer {{ text-align: center; padding: 1.5rem; color: #666; font-size: 0.82rem; border-top: 1px solid #e0e0e0; }}
.mono {{ font-family: 'SF Mono', Menlo, 'Monaco', monospace; font-size: 0.85em; }}
</style>
</head>
<body>
<div class="container">
<header>
    <h1>Luxar Project Statistics</h1>
    <p>Comprehensive codebase analysis &middot; Generated {now.strftime("%B %d, %Y at %H:%M")}</p>
</header>
<div class="content">

<div class="summary">
    <div class="stat-card"><h3>Total Files</h3><div class="value">{total_files:,}</div><div class="label">All languages</div></div>
    <div class="stat-card"><h3>Lines of Code</h3><div class="value">{total_code_lines:,}</div><div class="label">Executable only</div></div>
    <div class="stat-card"><h3>Total Lines</h3><div class="value">{total_lines:,}</div><div class="label">Incl. comments &amp; blanks</div></div>
    <div class="stat-card"><h3>Languages</h3><div class="value">{active_langs}</div><div class="label">Active in tree</div></div>
    <div class="stat-card"><h3>Test Files</h3><div class="value">{total_test_files:,}</div><div class="label">Py + TS + Rust + E2E</div></div>
    <div class="stat-card"><h3>Tests</h3><div class="value">{total_tests:,}</div><div class="label">Collected cases</div></div>
    <div class="stat-card"><h3>Coverage</h3><div class="value">{weighted_cov:.1f}%</div><div class="label">Py/TS weighted</div></div>
    <div class="stat-card"><h3>Commits</h3><div class="value">{git["total_commits"]:,}</div><div class="label">{git["commits_last_30_days"]} in last 30d</div></div>
    <div class="stat-card"><h3>Project Size</h3><div class="value">{_fmt_size(extras["project_size_bytes"])}</div><div class="label">Source &amp; assets</div></div>
    <div class="stat-card"><h3>CI Workflows</h3><div class="value">{extras["ci_workflows"]}</div><div class="label">.github/workflows</div></div>
</div>
""")

    # Language breakdown table
    out.append("""<section><h2>Language Breakdown</h2>
<table>
<tr><th>Language</th><th class="number">Files</th><th class="number">Code Lines</th><th class="number">Total Lines</th><th class="number">Comments</th><th class="number">% of Codebase</th></tr>
""")
    for lang in primary_langs + config_langs + doc_langs:
        s = langs.get(lang, {})
        if not s.get("files"):
            continue
        cfg = LANGUAGE_CONFIG[lang]
        share = (s["code_lines"] / total_code_lines * 100) if total_code_lines else 0
        out.append(f"""<tr>
<td><span class="lang-badge" style="background:{cfg["color"]}">{cfg["label"].upper()}</span></td>
<td class="number">{s["files"]:,}</td>
<td class="number">{s["code_lines"]:,}</td>
<td class="number">{s["total_lines"]:,}</td>
<td class="number">{s["comment_lines"]:,}</td>
<td class="number">{share:.1f}%</td>
</tr>
""")
    out.append("</table>")

    out.append(
        '<h3>Primary-language code distribution</h3><div class="progress-bar" style="height:24px;display:flex;">'
    )
    for lang in primary_langs:
        s = langs.get(lang, {})
        if not s.get("code_lines"):
            continue
        share = (
            (s["code_lines"] / primary_code_lines * 100) if primary_code_lines else 0
        )
        if share < 0.5:
            continue
        cfg = LANGUAGE_CONFIG[lang]
        out.append(
            f'<div class="progress-fill" style="width:{share}%;background:{cfg["color"]};" '
            f'title="{cfg["label"]}: {share:.1f}%"></div>'
        )
    out.append(
        '</div><div style="display:flex;gap:1rem;flex-wrap:wrap;margin-top:0.55rem;font-size:0.85rem;">'
    )
    for lang in primary_langs:
        s = langs.get(lang, {})
        if not s.get("code_lines"):
            continue
        cfg = LANGUAGE_CONFIG[lang]
        out.append(
            f'<span><span style="display:inline-block;width:12px;height:12px;background:{cfg["color"]};border-radius:2px;margin-right:4px;vertical-align:middle;"></span>{cfg["label"]}</span>'
        )
    out.append("</div></section>")

    # Primary languages detail cards
    out.append('<section><h2>Primary Languages</h2><div class="lang-breakdown">')

    def card(lang_key: str, headline: str, body_inner: str) -> str:
        cfg = LANGUAGE_CONFIG[lang_key]
        return (
            f'<div class="lang-card" style="border-color:{cfg["color"]}">'
            f'<h4><span class="lang-badge" style="background:{cfg["color"]}">{cfg["label"].upper()}</span> {headline}</h4>'
            f'<div class="lang-stats">{body_inner}</div></div>'
        )

    if py["files"]:
        out.append(
            card(
                "python",
                "Core Backend",
                (
                    f"<div><strong>{py['files']:,}</strong> files | <strong>{py['code_lines']:,}</strong> LOC</div>"
                    f"<div>Classes: {py['definitions'].get('classes', 0):,} | "
                    f"Functions: {py['definitions'].get('functions', 0):,} | "
                    f"Methods: {py['definitions'].get('methods', 0):,}</div>"
                    f"<div>Coverage: <strong>{py_cov:.1f}%</strong> | "
                    f"Tests: {tests['python']['test_count']:,}</div>"
                ),
            )
        )
    if ts["files"]:
        out.append(
            card(
                "typescript",
                "Viewer Frontend",
                (
                    f"<div><strong>{ts['files']:,}</strong> files | <strong>{ts['code_lines']:,}</strong> LOC</div>"
                    f"<div>Classes: {ts['definitions'].get('classes', 0):,} | "
                    f"Functions: {ts['definitions'].get('functions', 0):,} | "
                    f"Interfaces: {ts['definitions'].get('interfaces', 0):,} | "
                    f"Types: {ts['definitions'].get('types', 0):,}</div>"
                    f"<div>Coverage: <strong>{ts_cov:.1f}%</strong> | "
                    f"Tests: {tests['typescript']['test_count']:,}</div>"
                ),
            )
        )
    if rust["files"]:
        out.append(
            card(
                "rust",
                "WASM Module",
                (
                    f"<div><strong>{rust['files']:,}</strong> files | <strong>{rust['code_lines']:,}</strong> LOC</div>"
                    f"<div>Structs: {rust['definitions'].get('structs', 0):,} | "
                    f"Functions: {rust['definitions'].get('functions', 0):,} | "
                    f"Traits: {rust['definitions'].get('traits', 0):,} | "
                    f"Impls: {rust['definitions'].get('impls', 0):,} | "
                    f"Enums: {rust['definitions'].get('enums', 0):,}</div>"
                    f"<div>Tests: {tests['rust']['test_count']:,} ({tests['rust']['test_passed']} passed)</div>"
                ),
            )
        )
    if cuda["files"]:
        out.append(
            card(
                "cuda",
                "GPU Kernels",
                (
                    f"<div><strong>{cuda['files']:,}</strong> files | <strong>{cuda['code_lines']:,}</strong> LOC</div>"
                    f"<div>Kernels: {cuda['definitions'].get('kernels', 0):,} | "
                    f"Device fns: {cuda['definitions'].get('device_functions', 0):,} | "
                    f"Host fns: {cuda['definitions'].get('host_functions', 0):,}</div>"
                ),
            )
        )
    if go["files"]:
        out.append(
            card(
                "go",
                "Native Launchers",
                (
                    f"<div><strong>{go['files']:,}</strong> files | <strong>{go['code_lines']:,}</strong> LOC</div>"
                    f"<div>Functions: {go['definitions'].get('functions', 0):,} | "
                    f"Structs: {go['definitions'].get('structs', 0):,} | "
                    f"Interfaces: {go['definitions'].get('interfaces', 0):,}</div>"
                ),
            )
        )
    if css["files"]:
        out.append(
            card(
                "css",
                "Styling",
                (
                    f"<div><strong>{css['files']:,}</strong> files | <strong>{css['code_lines']:,}</strong> LOC</div>"
                    f"<div>Rules: {css['definitions'].get('rules', 0):,} | "
                    f"Vars: {css['definitions'].get('variables', 0):,} | "
                    f"@media: {css['definitions'].get('media_queries', 0):,}</div>"
                ),
            )
        )
    out.append("</div></section>")

    # Per-package breakdown
    py_pkg = stats["package_breakdown"]["python"]
    ts_pkg = stats["package_breakdown"]["typescript"]

    if py_pkg or ts_pkg:
        out.append('<section><h2>Per-Package Breakdown</h2><div class="two-column">')
        if py_pkg:
            out.append(
                "<div><h3>Python (packages/luxar/src/luxar/)</h3><table>"
                '<tr><th>Subpackage</th><th class="number">Files</th>'
                '<th class="number">Code Lines</th><th class="number">Comments</th></tr>'
            )
            for entry in py_pkg:
                out.append(
                    f'<tr><td class="mono">{html.escape(entry["name"])}</td>'
                    f'<td class="number">{entry["files"]:,}</td>'
                    f'<td class="number">{entry["code_lines"]:,}</td>'
                    f'<td class="number">{entry["comment_lines"]:,}</td></tr>'
                )
            out.append("</table></div>")
        if ts_pkg:
            out.append(
                "<div><h3>TypeScript (packages/luxar-viewer/src/)</h3><table>"
                '<tr><th>Subpackage</th><th class="number">Files</th>'
                '<th class="number">Code Lines</th><th class="number">Comments</th></tr>'
            )
            for entry in ts_pkg:
                out.append(
                    f'<tr><td class="mono">{html.escape(entry["name"])}</td>'
                    f'<td class="number">{entry["files"]:,}</td>'
                    f'<td class="number">{entry["code_lines"]:,}</td>'
                    f'<td class="number">{entry["comment_lines"]:,}</td></tr>'
                )
            out.append("</table></div>")
        out.append("</div></section>")

    # Test metrics
    py_loc = py["code_lines"]
    ts_loc = ts["code_lines"]
    py_tpk = tests["python"]["test_count"] / (py_loc / 1000) if py_loc else 0
    ts_tpk = tests["typescript"]["test_count"] / (ts_loc / 1000) if ts_loc else 0
    total_tpk = total_tests / (total_code_lines / 1000) if total_code_lines else 0
    pass_status = (
        '<span class="health-indicator health-good">All Passing</span>'
        if total_tests and total_failed == 0
        else '<span class="health-indicator health-warning">Some Failures</span>'
        if total_tests
        else '<span class="health-indicator health-warning">Not Run</span>'
    )
    cov_status = (
        '<span class="health-indicator health-good">Excellent</span>'
        if weighted_cov >= 80
        else '<span class="health-indicator health-warning">Good</span>'
        if weighted_cov >= 60
        else '<span class="health-indicator health-warning">Not Measured</span>'
        if weighted_cov == 0
        else '<span class="health-indicator health-danger">Needs Work</span>'
    )
    tpk_status = (
        '<span class="health-indicator health-good">Well Tested</span>'
        if total_tpk > 15
        else '<span class="health-indicator health-warning">Adequate</span>'
    )

    out.append(f"""<section><h2>Test &amp; Quality Metrics</h2>
<table>
<tr><th>Metric</th><th class="number">Python</th><th class="number">TypeScript</th><th class="number">Rust</th><th class="number">E2E</th><th class="number">Total</th><th>Status</th></tr>
<tr><td>Test Files</td>
    <td class="number">{tests["python"]["test_files"]:,}</td>
    <td class="number">{tests["typescript"]["test_files"]:,}</td>
    <td class="number">{tests["rust"]["test_files"]:,}</td>
    <td class="number">{tests["e2e"]["test_files"]:,}</td>
    <td class="number">{total_test_files:,}</td>
    <td><span class="health-indicator health-good">Good</span></td></tr>
<tr><td>Test Count</td>
    <td class="number">{tests["python"]["test_count"]:,}</td>
    <td class="number">{tests["typescript"]["test_count"]:,}</td>
    <td class="number">{tests["rust"]["test_count"]:,}</td>
    <td class="number">&mdash;</td>
    <td class="number">{total_tests:,}</td>
    <td><span class="health-indicator health-good">Comprehensive</span></td></tr>
<tr><td>Tests Passed</td>
    <td class="number">{tests["python"]["test_passed"]:,}</td>
    <td class="number">{tests["typescript"]["test_passed"]:,}</td>
    <td class="number">{tests["rust"]["test_passed"]:,}</td>
    <td class="number">&mdash;</td>
    <td class="number">{total_passed:,}</td>
    <td>{pass_status}</td></tr>
<tr><td>Code Coverage</td>
    <td class="number">{py_cov:.1f}%</td>
    <td class="number">{ts_cov:.1f}%</td>
    <td class="number">&mdash;</td>
    <td class="number">&mdash;</td>
    <td class="number">{weighted_cov:.1f}%</td>
    <td>{cov_status}</td></tr>
<tr><td>Tests per 1K LOC</td>
    <td class="number">{py_tpk:.1f}</td>
    <td class="number">{ts_tpk:.1f}</td>
    <td class="number">&mdash;</td>
    <td class="number">&mdash;</td>
    <td class="number">{total_tpk:.1f}</td>
    <td>{tpk_status}</td></tr>
</table>
</section>
""")

    # Git statistics
    out.append(f"""<section><h2>Git Activity</h2><div class="two-column">
<div>
<table>
<tr><th>Metric</th><th class="number">Value</th></tr>
<tr><td>Total Commits</td><td class="number">{git["total_commits"]:,}</td></tr>
<tr><td>Contributors (all-time)</td><td class="number">{git["contributors"]}</td></tr>
<tr><td>Tags</td><td class="number">{git["tags"]}</td></tr>
<tr><td>Commits (last 30 days)</td><td class="number">{git["commits_last_30_days"]:,}</td></tr>
<tr><td>Files Changed (last 30 days)</td><td class="number">{git["files_changed_last_30_days"]:,}</td></tr>
<tr><td>First Commit</td><td class="number">{git["first_commit_date"] or "N/A"}</td></tr>
<tr><td>Last Commit</td><td class="number">{git["last_commit_date"] or "N/A"}</td></tr>
</table>
</div>
<div><h3>Top Contributors</h3><table>
<tr><th>Name</th><th class="number">Commits</th></tr>
""")
    for c in git.get("top_contributors", [])[:5]:
        out.append(
            f"<tr><td>{html.escape(c['name'])}</td>"
            f'<td class="number">{c["commits"]:,}</td></tr>'
        )
    out.append("</table></div></div></section>")

    # Dependencies
    py_group_str = ", ".join(
        f"{g}={n}" for g, n in sorted(deps["python"]["groups"].items())
    )
    out.append(f"""<section><h2>Dependencies</h2>
<table>
<tr><th>Ecosystem</th><th class="number">Production</th><th class="number">Development</th><th class="number">Total</th><th>Notes</th></tr>
<tr><td><span class="lang-badge" style="background:{LANGUAGE_CONFIG["python"]["color"]}">Python</span></td>
    <td class="number">{deps["python"]["production"]}</td>
    <td class="number">{deps["python"]["dev"]}</td>
    <td class="number">{deps["python"]["production"] + deps["python"]["dev"]}</td>
    <td class="mono">{html.escape(py_group_str) or "&mdash;"}</td></tr>
<tr><td><span class="lang-badge" style="background:{LANGUAGE_CONFIG["javascript"]["color"]};color:#222">Node.js</span></td>
    <td class="number">{deps["node"]["production"]}</td>
    <td class="number">{deps["node"]["dev"]}</td>
    <td class="number">{deps["node"]["production"] + deps["node"]["dev"]}</td>
    <td>packages/luxar-viewer/package.json</td></tr>
<tr><td><span class="lang-badge" style="background:{LANGUAGE_CONFIG["rust"]["color"]};color:#222">Rust</span></td>
    <td class="number">{deps["rust"]["production"]}</td>
    <td class="number">{deps["rust"]["dev"]}</td>
    <td class="number">{deps["rust"]["production"] + deps["rust"]["dev"]}</td>
    <td>luxar-viewer/src/wasm/rust/Cargo.toml</td></tr>
</table>
</section>
""")

    # Largest files
    out.append('<section><h2>Largest Source Files</h2><div class="two-column">')
    for lang_key, heading in [("python", "Python"), ("typescript", "TypeScript")]:
        files = langs[lang_key].get("largest_files") or []
        if not files:
            continue
        out.append(
            f"<div><h3>{heading}</h3><table>"
            '<tr><th>Path</th><th class="number">Code Lines</th></tr>'
        )
        for path, lines in files[:10]:
            out.append(
                f'<tr><td class="mono">{html.escape(path)}</td>'
                f'<td class="number">{lines:,}</td></tr>'
            )
        out.append("</table></div>")
    out.append("</div></section>")

    # Documentation & Configuration
    out.append(f"""<section><h2>Documentation &amp; Configuration</h2><div class="two-column">
<div><h3>Documentation</h3><table>
<tr><th>Type</th><th class="number">Files</th><th class="number">Lines</th></tr>
<tr><td>Markdown (.md)</td><td class="number">{md["files"]:,}</td><td class="number">{md["total_lines"]:,}</td></tr>
<tr><td>HTML</td><td class="number">{html_lang["files"]}</td><td class="number">{html_lang["total_lines"]:,}</td></tr>
<tr><td>CHANGELOG entries (## headings)</td><td class="number">{extras["changelog_versions"]}</td><td class="number">&mdash;</td></tr>
</table></div>
<div><h3>Configuration</h3><table>
<tr><th>Type</th><th class="number">Files</th><th class="number">Lines</th></tr>
<tr><td>JSON</td><td class="number">{json_stats["files"]}</td><td class="number">{json_stats["total_lines"]:,}</td></tr>
<tr><td>YAML</td><td class="number">{yaml_stats["files"]}</td><td class="number">{yaml_stats["total_lines"]:,}</td></tr>
<tr><td>TOML</td><td class="number">{toml_stats["files"]}</td><td class="number">{toml_stats["total_lines"]:,}</td></tr>
<tr><td>CI workflows (.yml in .github/workflows/)</td><td class="number">{extras["ci_workflows"]}</td><td class="number">&mdash;</td></tr>
</table></div>
</div></section>
""")

    # Key insights
    arch_bits = []
    if py["files"]:
        arch_bits.append(f"{py['definitions'].get('classes', 0)} Python classes")
    if ts["files"]:
        arch_bits.append(
            f"{ts['definitions'].get('interfaces', 0)} TypeScript interfaces"
        )
    if rust["files"]:
        arch_bits.append(f"{rust['definitions'].get('functions', 0)} Rust functions")
    arch_text = ", ".join(arch_bits) if arch_bits else "n/a"

    mix_bits = []
    for lang in ("python", "typescript", "rust", "cuda", "go"):
        s = langs.get(lang)
        if s and s["code_lines"]:
            mix_bits.append(
                f"{LANGUAGE_CONFIG[lang]['label']} {pct(s['code_lines']):.1f}%"
            )
    mix_text = " &middot; ".join(mix_bits) if mix_bits else "n/a"

    out.append(f"""<section><h2>Key Insights</h2>
<div class="insight"><strong>Code density:</strong> {code_density:.1f}% executable code, {comment_density:.1f}% comments, blanks account for the rest.</div>
<div class="insight"><strong>Language mix (primary):</strong> {mix_text}</div>
<div class="insight"><strong>Architecture:</strong> {arch_text}</div>
<div class="insight"><strong>Test health:</strong> {total_tests:,} tests across {total_test_files:,} files with {weighted_cov:.1f}% weighted coverage</div>
<div class="insight"><strong>Project activity:</strong> {git["commits_last_30_days"]:,} commits and {git["files_changed_last_30_days"]:,} files touched in the last 30 days by {git["contributors"]} contributor(s).</div>
<div class="insight"><strong>Documentation:</strong> {md["files"]:,} markdown files with {md["total_lines"]:,} lines.</div>
</section>
</div>
<footer>
Generated by Luxar Project Analyzer &middot; {now.strftime("%Y-%m-%d %H:%M")} &middot;
Excludes: node_modules, __pycache__, coverage, dist, build, target, _build, playwright-report, datasets, delme, .venv
</footer>
</div>
</body>
</html>
""")

    output_file.write_text("".join(out), encoding="utf-8")


def generate_markdown_report(stats: dict[str, Any], output_file: Path) -> None:
    """Write a GitHub-friendly markdown summary of the stats.

    Mirrors the high-impact sections of the HTML report: summary cards,
    language breakdown, per-package tables, tests, git activity,
    dependencies, and the largest files. Includes a link to the rich
    HTML report at ``stats/project_stats.html``.
    """
    now = datetime.now()
    langs = stats["languages"]
    tests = stats["tests"]
    git = stats["git"]
    deps = stats["dependencies"]
    extras = stats["extras"]

    total_files = sum(langs[lang]["files"] for lang in langs)
    total_code = sum(langs[lang]["code_lines"] for lang in langs)
    total_lines = sum(langs[lang]["total_lines"] for lang in langs)
    active_langs = sum(1 for lang in langs if langs[lang]["files"] > 0)
    py = langs["python"]
    ts = langs["typescript"]
    py_loc = py["code_lines"]
    ts_loc = ts["code_lines"]
    py_cov = tests["python"]["coverage_percent"]
    ts_cov = tests["typescript"]["coverage_percent"]
    weighted_cov = weighted_coverage(py_cov, py_loc, ts_cov, ts_loc)

    total_test_files = (
        tests["python"]["test_files"]
        + tests["typescript"]["test_files"]
        + tests["e2e"]["test_files"]
        + tests["rust"]["test_files"]
    )
    total_tests = (
        tests["python"]["test_count"]
        + tests["typescript"]["test_count"]
        + tests["rust"]["test_count"]
    )

    lines: list[str] = []
    lines.append("# Luxar Project Statistics")
    lines.append("")
    lines.append(
        f"_Generated {now.strftime('%Y-%m-%d %H:%M')} &middot; "
        f"For the styled report with progress bars and per-language detail, "
        f"open [`project_stats.html`](./project_stats.html) locally._"
    )
    lines.append("")
    lines.append(
        "_To refresh both this file and the HTML report, run `make stats` "
        "from the project root._"
    )
    lines.append("")

    # Headline figures
    lines.append("## Summary")
    lines.append("")
    lines.append("| Metric | Value |")
    lines.append("| --- | ---: |")
    lines.append(f"| Total files | {total_files:,} |")
    lines.append(f"| Lines of code (executable) | {total_code:,} |")
    lines.append(f"| Total lines | {total_lines:,} |")
    lines.append(f"| Active languages | {active_langs} |")
    lines.append(f"| Test files | {total_test_files:,} |")
    lines.append(f"| Tests collected | {total_tests:,} |")
    lines.append(f"| Coverage (Py/TS weighted) | {weighted_cov:.1f}% |")
    lines.append(f"| Total commits | {git['total_commits']:,} |")
    lines.append(f"| Commits in last 30 days | {git['commits_last_30_days']:,} |")
    lines.append(f"| Project size | {_fmt_size(extras['project_size_bytes'])} |")
    lines.append(f"| CI workflows | {extras['ci_workflows']} |")
    lines.append("")

    # Language breakdown
    lines.append("## Language Breakdown")
    lines.append("")
    lines.append("| Language | Files | Code | Total | Comments | Share |")
    lines.append("| --- | ---: | ---: | ---: | ---: | ---: |")
    for lang in (
        "python",
        "typescript",
        "rust",
        "cuda",
        "go",
        "css",
        "javascript",
        "shell",
        "json",
        "toml",
        "yaml",
        "markdown",
        "html",
        "makefile",
    ):
        s = langs.get(lang, {})
        if not s.get("files"):
            continue
        share = (s["code_lines"] / total_code * 100) if total_code else 0
        label = LANGUAGE_CONFIG[lang]["label"]
        lines.append(
            f"| {label} | {s['files']:,} | {s['code_lines']:,} | "
            f"{s['total_lines']:,} | {s['comment_lines']:,} | {share:.1f}% |"
        )
    lines.append("")

    # Primary languages summary
    lines.append("## Primary Languages")
    lines.append("")
    if py["files"]:
        lines.append(
            f"- **Python** &mdash; {py['files']:,} files, {py_loc:,} LOC, "
            f"{py['definitions'].get('classes', 0):,} classes, "
            f"{py['definitions'].get('functions', 0):,} functions, "
            f"{py['definitions'].get('methods', 0):,} methods "
            f"(coverage {py_cov:.1f}%, tests {tests['python']['test_count']:,})."
        )
    if ts["files"]:
        lines.append(
            f"- **TypeScript** &mdash; {ts['files']:,} files, {ts_loc:,} LOC, "
            f"{ts['definitions'].get('classes', 0):,} classes, "
            f"{ts['definitions'].get('functions', 0):,} functions, "
            f"{ts['definitions'].get('interfaces', 0):,} interfaces, "
            f"{ts['definitions'].get('types', 0):,} types "
            f"(coverage {ts_cov:.1f}%, tests {tests['typescript']['test_count']:,})."
        )
    rust = langs["rust"]
    if rust["files"]:
        lines.append(
            f"- **Rust (WASM)** &mdash; {rust['files']:,} files, "
            f"{rust['code_lines']:,} LOC, "
            f"{rust['definitions'].get('functions', 0):,} functions "
            f"(tests {tests['rust']['test_count']:,})."
        )
    cuda = langs["cuda"]
    if cuda["files"]:
        lines.append(
            f"- **CUDA** &mdash; {cuda['files']:,} files, {cuda['code_lines']:,} LOC, "
            f"{cuda['definitions'].get('kernels', 0):,} kernels."
        )
    go = langs["go"]
    if go["files"]:
        lines.append(
            f"- **Go (launchers)** &mdash; {go['files']:,} files, "
            f"{go['code_lines']:,} LOC."
        )
    lines.append("")

    # Per-package breakdown
    py_pkg = stats["package_breakdown"]["python"]
    ts_pkg = stats["package_breakdown"]["typescript"]
    if py_pkg:
        lines.append("## Python subpackages (`packages/luxar/src/luxar/`)")
        lines.append("")
        lines.append("| Subpackage | Files | Code Lines | Comments |")
        lines.append("| --- | ---: | ---: | ---: |")
        for entry in py_pkg:
            lines.append(
                f"| `{entry['name']}` | {entry['files']:,} | "
                f"{entry['code_lines']:,} | {entry['comment_lines']:,} |"
            )
        lines.append("")
    if ts_pkg:
        lines.append("## TypeScript subpackages (`packages/luxar-viewer/src/`)")
        lines.append("")
        lines.append("| Subpackage | Files | Code Lines | Comments |")
        lines.append("| --- | ---: | ---: | ---: |")
        for entry in ts_pkg:
            lines.append(
                f"| `{entry['name']}` | {entry['files']:,} | "
                f"{entry['code_lines']:,} | {entry['comment_lines']:,} |"
            )
        lines.append("")

    # Tests
    lines.append("## Tests & Coverage")
    lines.append("")
    lines.append("| Metric | Python | TypeScript | Rust | E2E | Total |")
    lines.append("| --- | ---: | ---: | ---: | ---: | ---: |")
    lines.append(
        f"| Test files | {tests['python']['test_files']:,} | "
        f"{tests['typescript']['test_files']:,} | "
        f"{tests['rust']['test_files']:,} | "
        f"{tests['e2e']['test_files']:,} | {total_test_files:,} |"
    )
    lines.append(
        f"| Tests collected | {tests['python']['test_count']:,} | "
        f"{tests['typescript']['test_count']:,} | "
        f"{tests['rust']['test_count']:,} | &mdash; | {total_tests:,} |"
    )
    total_passed = (
        tests["python"]["test_passed"]
        + tests["typescript"]["test_passed"]
        + tests["rust"]["test_passed"]
    )
    lines.append(
        f"| Tests passed | {tests['python']['test_passed']:,} | "
        f"{tests['typescript']['test_passed']:,} | "
        f"{tests['rust']['test_passed']:,} | &mdash; | {total_passed:,} |"
    )
    lines.append(
        f"| Coverage | {py_cov:.1f}% | {ts_cov:.1f}% | &mdash; | &mdash; | "
        f"{weighted_cov:.1f}% |"
    )
    lines.append("")

    # Git
    lines.append("## Git Activity")
    lines.append("")
    lines.append("| Metric | Value |")
    lines.append("| --- | ---: |")
    lines.append(f"| Total commits | {git['total_commits']:,} |")
    lines.append(f"| Contributors (all-time) | {git['contributors']} |")
    lines.append(f"| Tags | {git['tags']} |")
    lines.append(f"| Commits (last 30 days) | {git['commits_last_30_days']:,} |")
    lines.append(
        f"| Files changed (last 30 days) | {git['files_changed_last_30_days']:,} |"
    )
    lines.append(f"| First commit | {git['first_commit_date'] or 'N/A'} |")
    lines.append(f"| Last commit | {git['last_commit_date'] or 'N/A'} |")
    lines.append("")

    if git.get("top_contributors"):
        lines.append("### Top contributors")
        lines.append("")
        lines.append("| Name | Commits |")
        lines.append("| --- | ---: |")
        for c in git["top_contributors"][:5]:
            lines.append(f"| {c['name']} | {c['commits']:,} |")
        lines.append("")

    # Dependencies
    py_groups = (
        ", ".join(f"{g}={n}" for g, n in sorted(deps["python"]["groups"].items()))
        or "&mdash;"
    )
    lines.append("## Dependencies")
    lines.append("")
    lines.append("| Ecosystem | Production | Development | Total | Notes |")
    lines.append("| --- | ---: | ---: | ---: | --- |")
    lines.append(
        f"| Python | {deps['python']['production']} | "
        f"{deps['python']['dev']} | "
        f"{deps['python']['production'] + deps['python']['dev']} | "
        f"groups: {py_groups} |"
    )
    lines.append(
        f"| Node.js | {deps['node']['production']} | {deps['node']['dev']} | "
        f"{deps['node']['production'] + deps['node']['dev']} | "
        f"`packages/luxar-viewer/package.json` |"
    )
    lines.append(
        f"| Rust | {deps['rust']['production']} | {deps['rust']['dev']} | "
        f"{deps['rust']['production'] + deps['rust']['dev']} | "
        f"`packages/luxar-viewer/src/wasm/rust/Cargo.toml` |"
    )
    lines.append("")

    # Largest files
    py_largest = py.get("largest_files") or []
    ts_largest = ts.get("largest_files") or []
    if py_largest or ts_largest:
        lines.append("## Largest Source Files")
        lines.append("")
        if py_largest:
            lines.append("### Python")
            lines.append("")
            lines.append("| Path | Code Lines |")
            lines.append("| --- | ---: |")
            for path, n in py_largest[:10]:
                lines.append(f"| `{path}` | {n:,} |")
            lines.append("")
        if ts_largest:
            lines.append("### TypeScript")
            lines.append("")
            lines.append("| Path | Code Lines |")
            lines.append("| --- | ---: |")
            for path, n in ts_largest[:10]:
                lines.append(f"| `{path}` | {n:,} |")
            lines.append("")

    lines.append("---")
    lines.append("")
    lines.append(
        "_Excludes: `node_modules`, `__pycache__`, `coverage`, `dist`, "
        "`build`, `target`, `_build`, `playwright-report`, `datasets`, "
        "`delme`, `.venv`. Source: `stats/generate_stats.py`._"
    )
    lines.append("")

    output_file.write_text("\n".join(lines), encoding="utf-8")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--no-tests", action="store_true", help="Skip running tests (file counts only)"
    )
    parser.add_argument(
        "--no-coverage",
        action="store_true",
        help="Run tests but skip coverage collection",
    )
    parser.add_argument(
        "--json", action="store_true", help="Also emit stats/project_stats.json"
    )
    args = parser.parse_args()

    script_path = Path(__file__).resolve()
    project_root = (
        script_path.parent.parent
        if script_path.parent.name == "stats"
        else script_path.parent
    )

    # A path component that matches a skipped directory name makes the walker
    # prune EVERY file, and the report then publishes zeros as if measured.
    # `.claude` is the one that bites: agent worktrees live under
    # `<repo>/.claude/worktrees/<name>/`, so a run from there scans nothing.
    # Fail loudly instead — a stale report beats a zeroed one.
    shadowed = SKIP_DIR_NAMES.intersection(project_root.parts)
    if shadowed:
        raise SystemExit(
            f"Refusing to run: the project root {project_root} contains path "
            f"component(s) {sorted(shadowed)} that the scanner skips, so every "
            "file would be pruned and the report would read 0 files / 0% "
            "coverage. Run from a checkout whose path has no such component "
            "(e.g. a worktree under ~/workspace/... rather than .claude/worktrees/...)."
        )

    with asection("Analyzing Luxar project"):
        all_stats: dict[str, Any] = {"languages": {}}

        with asection("Analyzing code by language"):
            for name in LANGUAGE_CONFIG:
                ls = analyze_language(project_root, name)
                all_stats["languages"][name] = ls.as_dict()
                if ls.files:
                    aprint(f"{name:12s} {ls.files:4d} files, {ls.code_lines:6,} LOC")

        if not sum(all_stats["languages"][n]["files"] for n in all_stats["languages"]):
            raise SystemExit(
                f"Refusing to run: scanned {project_root} and found 0 source "
                "files. Something is wrong with the checkout or the skip rules; "
                "writing this report would replace real numbers with zeros."
            )

        with asection("Per-package breakdown"):
            all_stats["package_breakdown"] = {
                "python": analyze_package_breakdown(project_root, PYTHON_PACKAGE_ROOT),
                "typescript": analyze_package_breakdown(project_root, TS_PACKAGE_ROOT),
            }
            aprint(
                f"Python subpackages: {len(all_stats['package_breakdown']['python'])}"
            )
            aprint(
                f"TS subpackages: {len(all_stats['package_breakdown']['typescript'])}"
            )

        with asection("Tests and coverage"):
            all_stats["tests"] = get_test_statistics(
                project_root,
                run_tests=not args.no_tests,
                run_coverage=not args.no_coverage,
            )

        # A requested test/coverage measurement that timed out, could not start,
        # or produced no fresh coverage would publish a misleading partial
        # number as if it were complete (weighted_coverage silently drops a
        # language that came back 0.0). Fail loudly and leave the existing
        # reports untouched — a stale report beats a zeroed one.
        problems = validate_measurements(
            all_stats["tests"],
            run_tests=not args.no_tests,
            run_coverage=not args.no_coverage,
        )
        if problems:
            problem_lines = "\n".join(f"  - {p}" for p in problems)
            raise SystemExit(
                "Refusing to write reports: the following requested "
                "measurements did not complete, so publishing would replace "
                "real numbers with a misleading partial result:\n"
                f"{problem_lines}\n"
                "The existing reports were left untouched (a stale report beats "
                "a zeroed one). Re-run once the toolchain is available and the "
                "suites finish within their time budget."
            )

        with asection("Git statistics"):
            all_stats["git"] = get_git_statistics(project_root)
            aprint(f"Total commits: {all_stats['git']['total_commits']}")
            aprint(f"Contributors:  {all_stats['git']['contributors']}")

        with asection("Dependencies"):
            all_stats["dependencies"] = get_dependency_statistics(project_root)
            d = all_stats["dependencies"]
            aprint(
                f"Python: {d['python']['production']} prod + "
                f"{d['python']['dev']} dev (groups: "
                f"{', '.join(sorted(d['python']['groups'])) or 'none'})"
            )
            aprint(f"Node:   {d['node']['production']} prod + {d['node']['dev']} dev")
            aprint(f"Rust:   {d['rust']['production']} prod + {d['rust']['dev']} dev")

        with asection("Extras"):
            all_stats["extras"] = get_extras(project_root)
            aprint(f"CI workflows:  {all_stats['extras']['ci_workflows']}")
            aprint(
                f"Project size:  {_fmt_size(all_stats['extras']['project_size_bytes'])}"
            )

        stats_dir = project_root / "stats"
        stats_dir.mkdir(exist_ok=True)
        html_file = stats_dir / "project_stats.html"
        md_file = stats_dir / "PROJECT_STATS.md"
        with asection("Generating reports"):
            generate_html_report(all_stats, html_file)
            aprint(f"HTML:  {html_file}")
            generate_markdown_report(all_stats, md_file)
            aprint(f"MD:    {md_file}")

        if args.json:
            json_file = stats_dir / "project_stats.json"
            json_file.write_text(json.dumps(all_stats, indent=2, default=str))
            aprint(f"JSON:  {json_file}")

    # Summary line
    total_files = sum(
        all_stats["languages"][lang]["files"] for lang in all_stats["languages"]
    )
    total_code = sum(
        all_stats["languages"][lang]["code_lines"] for lang in all_stats["languages"]
    )
    aprint("")
    aprint("=" * 60)
    aprint(f"SUMMARY: {total_files:,} files, {total_code:,} LOC")
    aprint("=" * 60)


if __name__ == "__main__":
    main()
